import json
import re
from flask import Flask, request, jsonify, Response, stream_with_context

from config import Config
from cleanup import sanitize_llm_answer
from ingest.embed import embed_query
from rag.retriever import Retriever
from rag.llm import chat_completion, chat_completion_stream
from rag import settings as RAG

import logging

logging.basicConfig(level=logging.INFO)


app = Flask(__name__)

# Load retriever once at startup (connects to Qdrant)
retriever = Retriever()

# Max conversation turns to include in chat mode
MAX_CHAT_TURNS = 20


def detect_coverage(hits) -> str:
    """
    Very simple heuristic coverage guess from retrieval.
    We also ask the model to self-report coverage; if it does, we use that.
    """
    if not hits:
        return "none"
    best = (hits[0].text or "").strip()
    if len(best) < RAG.MIN_BEST_CHUNK_CHARS_FOR_FULL:
        return "partial"
    return "full"


def render_sources(hits):
    """
    Build:
      - sources: structured list for UI
      - blocks: text blocks inserted into the LLM prompt
     Truncates long chunks to keep prompts fast.
    """
    sources = []
    blocks = []

    for i, h in enumerate(hits, start=1):
        tag = f"[S{i}]"
        title = (h.heading or "").strip()
        src = h.source or ""

        txt = (h.text or "").strip()
        if len(txt) > RAG.MAX_CHARS_PER_CHUNK:
            txt = txt[: RAG.MAX_CHARS_PER_CHUNK].rstrip() + "\n…(truncated)…"

        header = f"{tag}"
        if title:
            header += f" {title}"

        blocks.append(f"{header}\n{txt}\n")

        sources.append(
            {
                "tag": tag,
                "source": src,
                "heading": title,
                "chunk_id": h.idx,
                "score": h.score,
                "chars": len(txt),
                "truncated": (len((h.text or "")) > RAG.MAX_CHARS_PER_CHUNK),
            }
        )

    return sources, blocks


def _sse(event: str, data) -> str:
    """Format a Server-Sent Event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_answer_from_problem(problem_text: str) -> str:
    """
    Backward-compat parser for old problem payloads that embedded:
      ANSWER: ...
    """
    if not problem_text:
        return ""

    m = re.search(r"(?is)\bANSWER\s*:\s*(.+)$", problem_text)
    if not m:
        return ""
    return m.group(1).strip()


def _build_chat_history_block(history: list) -> str:
    """
    Format previous turns into a text block for the system prompt.
    Each turn has a question and answer.
    """
    if not history:
        return ""

    lines = ["PREVIOUS CONVERSATION:"]
    for i, turn in enumerate(history[-MAX_CHAT_TURNS:], start=1):
        lines.append(f"Student Q{i}: {turn.get('question', '')}")
        lines.append(f"Your A{i}: {turn.get('answer', '')}")
        lines.append("")

    return "\n".join(lines)


@app.get("/health")
def health():
    return jsonify(status="ok")


# Original non-streaming endpoint (kept for backwards compatibility)
@app.post("/ask")
def ask():
    data = request.get_json(force=True) or {}

    q = (data.get("question") or "").strip()
    include_extra = bool(data.get("include_extra", RAG.INCLUDE_EXTRA_DEFAULT))
    extra_mode = (data.get("extra_mode") or RAG.EXTRA_MODE_DEFAULT).lower()

    if not q:
        return jsonify(error="Missing 'question'"), 400
    if len(q) > Config.MAX_QUESTION_LENGTH:
        return jsonify(error="Question too long"), 400
    if extra_mode not in ("auto", "always", "never"):
        return jsonify(error="extra_mode must be one of: auto, always, never"), 400

    extra_answer = None
    
    # 1) Retrieve from notes
    q_emb = embed_query(q)
    top_k = int(data.get("top_k", RAG.TOP_K_DEFAULT))
    hits = retriever.search(q_emb, top_k=top_k, log_hits=True)
    sources, source_blocks = render_sources(hits)
    retrieval_coverage = detect_coverage(hits)

    app.logger.info(
        "RETR q=%r top_k=%d hits=%d cov=%s best_chars=%d",
        q[:120],
        top_k,
        len(hits),
        retrieval_coverage,
        len((hits[0].text or "")) if hits else 0,
    )

    # 2) Pass 1: notes-only answer (must cite)
    system_1 = (
        "You are a course assistant. Use ONLY the provided SOURCES from the lecture notes. "
        "Do not use outside knowledge. If the answer is not in the sources, say you don't know. "
        "Write a clear explanation suitable for a student.\n"
        " Include at least one concrete example and one intuitive interpretation. Use LaTeX for formulas.\n"
        " RULES:\n"
        " - you MUST provide one example per answer. This is mandatory, don't skip it.\n"
        " - do not provide just the summary or just the example, you need to provide both.\n"
        "Be concise: 3–6 sentences max. No preamble.\n\n"
        "Cite sources like [S1], [S2] for every factual claim.\n\n"
        "At the end of your response, include a single line exactly in this format:\n"
        "COVERAGE: full|partial|none"
    )

    user_1 = (
        "SOURCES:\n"
        + "\n".join(source_blocks)
        + "\nQUESTION:\n"
        + q
        + "\n\nAnswer based only on SOURCES. Include citations. End with COVERAGE line."
    )

    notes_answer_raw = chat_completion(
        [{"role": "system", "content": system_1}, {"role": "user", "content": user_1}],
        temperature=RAG.NOTES_TEMPERATURE, max_tokens=RAG.NOTES_MAX_TOKENS,
    )

    notes_answer, model_coverage = sanitize_llm_answer(notes_answer_raw)
    coverage = model_coverage or retrieval_coverage

    # Decide if we do pass 2
    do_extra = False
    if include_extra and extra_mode != "never":
        if extra_mode == "always":
            do_extra = True
        else:
            # auto: only add extra when notes coverage isn't full
            do_extra = model_coverage != "full"

    if do_extra:
        system_2 = (
            "You are a helpful tutor. Add extra context NOT necessarily from the notes. "
            "Do NOT contradict the notes-based answer. If you add facts not present in the notes, "
            "label them clearly as general context.\n\n"
            "Output format (follow exactly):\n"
            "Extra context (not from notes):\n"
            "- 3–6 bullet points of intuition/examples\n"
            "- If relevant, include a short worked example\n"
        )
        user_2 = (
            "Question:\n" + q
            + "\n\nNotes-based answer (authoritative for course-specific claims):\n" + notes_answer
            + "\n\n(For consistency only) Retrieved sources:\n" + "\n".join(source_blocks)
        )
        extra_answer_raw = chat_completion(
            [{"role": "system", "content": system_2}, {"role": "user", "content": user_2}],
            temperature=RAG.EXTRA_TEMPERATURE, max_tokens=RAG.EXTRA_MAX_TOKENS,
        )
        extra_answer, _ = sanitize_llm_answer(extra_answer_raw)
        if extra_answer and not extra_answer.lstrip().lower().startswith("extra context (not from notes):"):
            extra_answer = "Extra context (not from notes):\n" + extra_answer.strip()

    return jsonify({
        "answer_notes": notes_answer,
        "answer_extra": extra_answer,
        "coverage": coverage,
        "sources": sources,
    })


# Streaming endpoint (supports both single and chat mode)
@app.post("/ask-stream")
def ask_stream():
    data = request.get_json(force=True) or {}

    q = (data.get("question") or "").strip()
    include_extra = bool(data.get("include_extra", RAG.INCLUDE_EXTRA_DEFAULT))
    extra_mode = (data.get("extra_mode") or RAG.EXTRA_MODE_DEFAULT).lower()
    chat_mode = bool(data.get("chat_mode", False))
    history = data.get("history", [])

    if not q:
        return jsonify(error="Missing 'question'"), 400
    if len(q) > Config.MAX_QUESTION_LENGTH:
        return jsonify(error="Question too long"), 400
    if extra_mode not in ("auto", "always", "never"):
        return jsonify(error="extra_mode must be one of: auto, always, never"), 400

    # Truncate history to max turns
    if history and len(history) > MAX_CHAT_TURNS:
        history = history[-MAX_CHAT_TURNS:]

    def generate():
        # Phase: Thinking (retrieval)
        yield _sse("status", {"phase": "thinking"})

        q_emb = embed_query(q)
        top_k = int(data.get("top_k", RAG.TOP_K_DEFAULT))
        hits = retriever.search(q_emb, top_k=top_k, log_hits=True)
        sources, source_blocks = render_sources(hits)
        retrieval_coverage = detect_coverage(hits)

        # Send sources immediately
        yield _sse("sources", sources)

        app.logger.info(
            "STREAM RETR q=%r top_k=%d hits=%d cov=%s chat=%s turns=%d",
            q[:120], top_k, len(hits), retrieval_coverage,
            chat_mode, len(history),
        )

        # Phase: Notes answer (streaming)
        yield _sse("status", {"phase": "notes"})

        # Build system prompt
        if chat_mode and history:
            history_block = _build_chat_history_block(history)
            system_1 = (
                "You are a course assistant having a conversation with a student. "
                "Use ONLY the provided SOURCES from the lecture notes. "
                "Do not use outside knowledge. If the answer is not in the sources, say you don't know.\n"
                "Write a clear explanation suitable for a student. "
                "Use LaTeX for formulas.\n\n"
                "You have access to the previous conversation for context. "
                "The student may refer to previous questions and answers. "
                "Answer the NEW question, using conversation history for context.\n\n"
                "RULES:\n"
                "- Cite sources like [S1], [S2] for every factual claim.\n"
                "- If the student asks a follow-up, use the conversation context.\n"
                "- Be concise but thorough.\n\n"
                + history_block + "\n"
                "At the end of your response, include a single line exactly in this format:\n"
                "COVERAGE: full|partial|none"
            )
        else:
            system_1 = (
                "You are a course assistant. Use ONLY the provided SOURCES from the lecture notes. "
                "Do not use outside knowledge. If the answer is not in the sources, say you don't know. "
                "Write a clear explanation suitable for a student.\n"
                " Include at least one concrete example and one intuitive interpretation. Use LaTeX for formulas.\n"
                " RULES:\n"
                " - you MUST provide one example per answer. This is mandatory, don't skip it.\n"
                " - do not provide just the summary or just the example, you need to provide both.\n"
                "Be concise: 3–6 sentences max. No preamble.\n\n"
                "Cite sources like [S1], [S2] for every factual claim.\n\n"
                "At the end of your response, include a single line exactly in this format:\n"
                "COVERAGE: full|partial|none"
            )

        user_1 = (
            "SOURCES:\n" + "\n".join(source_blocks)
            + "\nQUESTION:\n" + q
            + "\n\nAnswer based only on SOURCES. Include citations. End with COVERAGE line."
        )

        notes_raw = ""
        for token in chat_completion_stream(
            [{"role": "system", "content": system_1}, {"role": "user", "content": user_1}],
            temperature=RAG.NOTES_TEMPERATURE, max_tokens=RAG.NOTES_MAX_TOKENS,
        ):
            notes_raw += token
            yield _sse("token", {"target": "notes", "content": token})

        # Clean up and extract coverage
        notes_answer, model_coverage = sanitize_llm_answer(notes_raw)
        coverage = model_coverage or retrieval_coverage

        yield _sse("notes_done", {"coverage": coverage})

        app.logger.info(
            "STREAM notes len=%d cov=%s chat=%s",
            len(notes_answer or ""), coverage, chat_mode,
        )

        # Phase: Extra (only in single question mode)
        if not chat_mode:
            do_extra = False
            if include_extra and extra_mode != "never":
                if extra_mode == "always":
                    do_extra = True
                else:
                    do_extra = model_coverage != "full"

            if do_extra:
                yield _sse("status", {"phase": "extra"})

                system_2 = (
                    "You are a helpful tutor. Add extra context NOT necessarily from the notes. "
                    "Do NOT contradict the notes-based answer. If you add facts not present in the notes, "
                    "label them clearly as general context.\n\n"
                    "Output format (follow exactly):\n"
                    "Extra context (not from notes):\n"
                    "- 3–6 bullet points of intuition/examples\n"
                    "- If relevant, include a short worked example\n"
                )
                user_2 = (
                    "Question:\n" + q
                    + "\n\nNotes-based answer (authoritative for course-specific claims):\n" + notes_answer
                    + "\n\n(For consistency only) Retrieved sources:\n" + "\n".join(source_blocks)
                )

                for token in chat_completion_stream(
                    [{"role": "system", "content": system_2}, {"role": "user", "content": user_2}],
                    temperature=RAG.EXTRA_TEMPERATURE, max_tokens=RAG.EXTRA_MAX_TOKENS,
                ):
                    yield _sse("token", {"target": "extra", "content": token})

                yield _sse("extra_done", {})

        # Done
        yield _sse("done", {"coverage": coverage})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

# Problem generation endpoint
@app.post("/problem")
def generate_problem():
    data = request.get_json(force=True) or {}

    q = (data.get("question") or "").strip()
    notes_answer = (data.get("notes_answer") or "").strip()

    if not q:
        return jsonify(error="Missing 'question'"), 400

    # Retrieve relevant chunks for the topic
    q_emb = embed_query(q)
    hits = retriever.search(q_emb, top_k=RAG.TOP_K_DEFAULT, log_hits=True)
    sources, source_blocks = render_sources(hits)

    app.logger.info("PROBLEM q=%r hits=%d", q[:120], len(hits))

    def generate():
        yield _sse("status", {"phase": "thinking"})

        system_p = (
            "You are a course assistant that generates practice problems for students.\n"
            "Based on the SOURCES and the student's question, create ONE mathematical problem.\n\n"
            "RULES:\n"
            "- The problem must be solvable using the material in the sources\n"
            "- Include a clear, specific task (e.g. 'Compute...', 'Find...', 'Show that...')\n"
            "- The problem should have a concrete numerical or symbolic answer\n"
            "- Use LaTeX for all math formulas\n"
            "- Do NOT include the final answer or solution steps\n\n"
            "Output format:\n"
            "**Problem:**\n"
            "(problem statement only)\n"
        )

        user_p = (
            "SOURCES:\n" + "\n".join(source_blocks)
            + "\n\nStudent's question: " + q
        )

        if notes_answer:
            user_p += "\n\nThe answer given to the student (for context):\n" + notes_answer

        user_p += "\n\nGenerate one practice problem related to this topic."

        yield _sse("status", {"phase": "generating"})

        raw = ""
        for token in chat_completion_stream(
            [{"role": "system", "content": system_p}, {"role": "user", "content": user_p}],
            temperature=0.5, max_tokens=800,
        ):
            raw += token
            yield _sse("token", {"content": token})

        yield _sse("done", {})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/calculate-answer")
def calculate_answer():
    data = request.get_json(force=True) or {}

    problem = (data.get("problem") or "").strip()
    question = (data.get("question") or "").strip()
    notes_answer = (data.get("notes_answer") or "").strip()

    if not problem:
        return jsonify(error="Missing 'problem'"), 400

    retrieval_query = f"{question}\n\n{problem}".strip() if question else problem
    q_emb = embed_query(retrieval_query)
    hits = retriever.search(q_emb, top_k=RAG.TOP_K_DEFAULT, log_hits=True)
    _, source_blocks = render_sources(hits)

    app.logger.info(
        "CALC_ANSWER problem_len=%d question_len=%d hits=%d",
        len(problem), len(question), len(hits),
    )

    def generate():
        yield _sse("status", {"phase": "thinking"})

        system_c = (
            "You are a course assistant that solves math practice problems.\n"
            "Use the SOURCES when relevant, and provide the final answer to the problem.\n\n"
            "RULES:\n"
            "- Give a correct final answer\n"
            "- Keep the output concise\n"
            "- Use LaTeX for formulas\n"
            "- Start the final line with: ANSWER:\n"
            "- Do not add extra preamble text\n"
        )

        user_c = (
            "SOURCES:\n" + "\n".join(source_blocks)
            + "\n\nOriginal student question (context):\n" + (question or "(not provided)")
            + "\n\nProblem:\n" + problem
        )
        if notes_answer:
            user_c += "\n\nNotes answer used for context:\n" + notes_answer

        user_c += "\n\nCompute the correct final answer."

        yield _sse("status", {"phase": "generating"})

        for token in chat_completion_stream(
            [{"role": "system", "content": system_c}, {"role": "user", "content": user_c}],
            temperature=0.2, max_tokens=300,
        ):
            yield _sse("token", {"content": token})

        yield _sse("done", {})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Answer assessment endpoint
@app.post("/assess")
def assess_answer():
    data = request.get_json(force=True) or {}

    problem = (data.get("problem") or "").strip()
    correct_answer = (data.get("correct_answer") or "").strip()
    student_answer = (data.get("student_answer") or "").strip()

    if not problem:
        return jsonify(error="Missing 'problem'"), 400
    if not correct_answer:
        # Backward compatibility with older clients that sent "ANSWER:" inside problem.
        correct_answer = _extract_answer_from_problem(problem)
    if not correct_answer:
        return jsonify(error="Missing 'correct_answer'"), 400
    if not student_answer:
        return jsonify(error="Missing 'student_answer'"), 400

    app.logger.info(
        "ASSESS problem_len=%d student_len=%d correct_len=%d",
        len(problem), len(student_answer), len(correct_answer),
    )

    def generate():
        yield _sse("status", {"phase": "assessing"})

        system_a = (
            "You are a course assistant that assesses student answers to math problems.\n\n"
            "RULES:\n"
            "- Compare the student's answer to the provided correct answer\n"
            "- Be encouraging but honest\n"
            "- If wrong, explain where the mistake is and give a hint\n"
            "- If partially correct, acknowledge what's right and point out what's missing\n"
            "- If correct, confirm and optionally add a brief insight\n"
            "- Use LaTeX for math formulas\n\n"
            "Output format:\n"
            "**Result:** ✅ Correct / ⚠️ Partially correct / ❌ Incorrect\n\n"
            "(explanation)\n"
        )

        user_a = (
            "PROBLEM:\n" + problem
            + "\n\nCORRECT ANSWER:\n" + correct_answer
            + "\n\nSTUDENT'S ANSWER:\n" + student_answer
            + "\n\nAssess the student's answer."
        )

        raw = ""
        for token in chat_completion_stream(
            [{"role": "system", "content": system_a}, {"role": "user", "content": user_a}],
            temperature=0.3, max_tokens=600,
        ):
            raw += token
            yield _sse("token", {"content": token})

        yield _sse("done", {})

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
