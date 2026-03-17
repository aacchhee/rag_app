import json
import re
import time
import uuid
import logging

from flask import Flask, request, jsonify, Response, stream_with_context, g
from werkzeug.exceptions import HTTPException

from config import Config
from cleanup import sanitize_llm_answer
from ingest.embed import embed_query
from log_utils import current_request_id, preview_text
from rag.retriever import Retriever
from rag.llm import chat_completion, chat_completion_stream
from rag import settings as RAG

app = Flask(__name__)


def _configure_logging(flask_app: Flask) -> None:
    gunicorn_logger = logging.getLogger("gunicorn.error")
    if gunicorn_logger.handlers:
        flask_app.logger.handlers = gunicorn_logger.handlers
        flask_app.logger.setLevel(gunicorn_logger.level or logging.INFO)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
        flask_app.logger.setLevel(logging.INFO)
    flask_app.logger.propagate = False


def _missing_config_keys() -> list[str]:
    missing = []
    if not Config.LLM_API_KEY:
        missing.append("LLM_API_KEY")
    if not Config.EMBEDDINGS_URL:
        missing.append("EMBEDDINGS_URL")
    if not Config.EMBEDDINGS_MODEL:
        missing.append("EMBEDDINGS_MODEL")
    if not Config.CHAT_URL:
        missing.append("CHAT_URL")
    if not Config.CHAT_MODEL:
        missing.append("CHAT_MODEL")
    if not Config.QDRANT_URL:
        missing.append("QDRANT_URL")
    return missing


def _log_startup_config() -> None:
    missing = _missing_config_keys()
    if missing:
        app.logger.warning("startup config missing=%s", ",".join(missing))
    app.logger.info(
        "startup config debug=%s chat_model=%s chat_base=%s embed_model=%s embed_base=%s qdrant_url=%s collection=%s timeout=%s",
        Config.DEBUG,
        Config.CHAT_MODEL,
        Config.chat_base_url() if Config.CHAT_URL else "",
        Config.EMBEDDINGS_MODEL,
        Config.embeddings_base_url() if Config.EMBEDDINGS_URL else "",
        Config.QDRANT_URL,
        Config.QDRANT_COLLECTION,
        Config.LLM_TIMEOUT,
    )


def _request_started_at() -> float:
    return getattr(g, "request_started_at", time.perf_counter())


def _payload_summary(
    q: str,
    *,
    extra_mode: str | None = None,
    include_extra: bool | None = None,
    chat_mode: bool | None = None,
    history: list | None = None,
    top_k = None,
) -> str:
    parts = [f"q_len={len(q)}", f"q={preview_text(q, 120)!r}"]
    if include_extra is not None:
        parts.append(f"include_extra={include_extra}")
    if extra_mode is not None:
        parts.append(f"extra_mode={extra_mode}")
    if chat_mode is not None:
        parts.append(f"chat_mode={chat_mode}")
    if history is not None:
        parts.append(f"history_turns={len(history)}")
    if top_k is not None:
        parts.append(f"top_k={top_k}")
    return " ".join(parts)


_configure_logging(app)
_log_startup_config()

# Load retriever once at startup (connects to Qdrant)
retriever = Retriever()

# Max conversation turns to include in chat mode
MAX_CHAT_TURNS = 20


@app.before_request
def _before_request():
    g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
    g.request_started_at = time.perf_counter()
    forwarded_for = request.headers.get("X-Forwarded-For")
    remote_addr = forwarded_for or request.remote_addr or "-"
    app.logger.info(
        "[req:%s] -> %s %s remote=%s content_length=%s ua=%r",
        current_request_id(),
        request.method,
        request.path,
        remote_addr,
        request.content_length,
        (request.user_agent.string or "")[:200],
    )


@app.after_request
def _after_request(response):
    request_id = current_request_id()
    response.headers["X-Request-ID"] = request_id
    app.logger.info(
        "[req:%s] <- %s %s status=%s dur_ms=%.1f content_type=%s",
        request_id,
        request.method,
        request.path,
        response.status_code,
        (time.perf_counter() - _request_started_at()) * 1000,
        response.headers.get("Content-Type"),
    )
    return response


@app.errorhandler(Exception)
def _handle_unexpected_error(exc):
    if isinstance(exc, HTTPException):
        return exc
    app.logger.exception(
        "[req:%s] unhandled exception on %s %s",
        current_request_id(),
        request.method,
        request.path,
    )
    return jsonify(error="Internal server error", request_id=current_request_id()), 500


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


def _stream_response(generator):
    return Response(
        stream_with_context(generator),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Request-ID": current_request_id(),
        },
    )


@app.get("/health")
def health():
    return jsonify(status="ok", request_id=current_request_id())


# Original non-streaming endpoint (kept for backwards compatibility)
@app.post("/ask")
def ask():
    data = request.get_json(force=True) or {}

    q = (data.get("question") or "").strip()
    include_extra = bool(data.get("include_extra", RAG.INCLUDE_EXTRA_DEFAULT))
    extra_mode = (data.get("extra_mode") or RAG.EXTRA_MODE_DEFAULT).lower()
    request_id = current_request_id()
    request_started = time.perf_counter()

    try:
        top_k = int(data.get("top_k", RAG.TOP_K_DEFAULT))
    except (TypeError, ValueError):
        return jsonify(error="top_k must be an integer", request_id=request_id), 400

    if not q:
        return jsonify(error="Missing 'question'", request_id=request_id), 400
    if len(q) > Config.MAX_QUESTION_LENGTH:
        return jsonify(error="Question too long", request_id=request_id), 400
    if extra_mode not in ("auto", "always", "never"):
        return jsonify(error="extra_mode must be one of: auto, always, never", request_id=request_id), 400
    if top_k <= 0:
        return jsonify(error="top_k must be positive", request_id=request_id), 400

    app.logger.info(
        "[req:%s] /ask payload %s",
        request_id,
        _payload_summary(
            q,
            include_extra=include_extra,
            extra_mode=extra_mode,
            top_k=top_k,
        ),
    )

    extra_answer = None

    # 1) Retrieve from notes
    retrieval_started = time.perf_counter()
    q_emb = embed_query(q)
    hits = retriever.search(q_emb, top_k=top_k, log_hits=True)
    sources, source_blocks = render_sources(hits)
    retrieval_coverage = detect_coverage(hits)

    app.logger.info(
        "[req:%s] /ask retrieval top_k=%d hits=%d cov=%s best_chars=%d source_tags=%s dur_ms=%.1f",
        request_id,
        top_k,
        len(hits),
        retrieval_coverage,
        len((hits[0].text or "")) if hits else 0,
        ",".join(s["tag"] for s in sources),
        (time.perf_counter() - retrieval_started) * 1000,
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
    app.logger.info(
        "[req:%s] /ask notes_answer raw_len=%d clean_len=%d model_cov=%s final_cov=%s",
        request_id,
        len(notes_answer_raw or ""),
        len(notes_answer or ""),
        model_coverage,
        coverage,
    )

    # Decide if we do pass 2
    do_extra = False
    if include_extra and extra_mode != "never":
        if extra_mode == "always":
            do_extra = True
        else:
            # auto: only add extra when notes coverage isn't full
            do_extra = model_coverage != "full"

    app.logger.info(
        "[req:%s] /ask extra_decision include_extra=%s extra_mode=%s do_extra=%s",
        request_id,
        include_extra,
        extra_mode,
        do_extra,
    )

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
        app.logger.info(
            "[req:%s] /ask extra_answer raw_len=%d clean_len=%d",
            request_id,
            len(extra_answer_raw or ""),
            len(extra_answer or ""),
        )

    app.logger.info(
        "[req:%s] /ask complete notes_len=%d extra_len=%d dur_ms=%.1f",
        request_id,
        len(notes_answer or ""),
        len(extra_answer or ""),
        (time.perf_counter() - request_started) * 1000,
    )

    return jsonify({
        "answer_notes": notes_answer,
        "answer_extra": extra_answer,
        "coverage": coverage,
        "sources": sources,
        "request_id": request_id,
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
    request_id = current_request_id()
    request_started = time.perf_counter()

    try:
        top_k = int(data.get("top_k", RAG.TOP_K_DEFAULT))
    except (TypeError, ValueError):
        return jsonify(error="top_k must be an integer", request_id=request_id), 400

    if not q:
        return jsonify(error="Missing 'question'", request_id=request_id), 400
    if len(q) > Config.MAX_QUESTION_LENGTH:
        return jsonify(error="Question too long", request_id=request_id), 400
    if extra_mode not in ("auto", "always", "never"):
        return jsonify(error="extra_mode must be one of: auto, always, never", request_id=request_id), 400
    if top_k <= 0:
        return jsonify(error="top_k must be positive", request_id=request_id), 400
    if history is None:
        history = []
    if not isinstance(history, list):
        return jsonify(error="history must be a list", request_id=request_id), 400

    # Truncate history to max turns
    if history and len(history) > MAX_CHAT_TURNS:
        history = history[-MAX_CHAT_TURNS:]

    app.logger.info(
        "[req:%s] /ask-stream payload %s",
        request_id,
        _payload_summary(
            q,
            include_extra=include_extra,
            extra_mode=extra_mode,
            chat_mode=chat_mode,
            history=history,
            top_k=top_k,
        ),
    )

    def generate():
        coverage = None
        notes_raw = ""
        extra_raw = ""
        extra_answer = ""

        yield _sse(
            "meta",
            {
                "request_id": request_id,
                "chat_mode": chat_mode,
                "history_turns": len(history),
            },
        )
        # Phase: Thinking (retrieval)
        yield _sse("status", {"phase": "thinking"})

        retrieval_started = time.perf_counter()
        q_emb = embed_query(q)
        hits = retriever.search(q_emb, top_k=top_k, log_hits=True)
        sources, source_blocks = render_sources(hits)
        retrieval_coverage = detect_coverage(hits)

        # Send sources immediately
        yield _sse("sources", sources)

        app.logger.info(
            "[req:%s] /ask-stream retrieval hits=%d cov=%s best_chars=%d dur_ms=%.1f chat=%s turns=%d",
            request_id,
            len(hits),
            retrieval_coverage,
            len((hits[0].text or "")) if hits else 0,
            (time.perf_counter() - retrieval_started) * 1000,
            chat_mode,
            len(history),
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

        notes_tokens = 0
        notes_raw = ""
        for token in chat_completion_stream(
            [{"role": "system", "content": system_1}, {"role": "user", "content": user_1}],
            temperature=RAG.NOTES_TEMPERATURE, max_tokens=RAG.NOTES_MAX_TOKENS,
        ):
            notes_tokens += 1
            notes_raw += token
            yield _sse("token", {"target": "notes", "content": token})

        # Clean up and extract coverage
        notes_answer, model_coverage = sanitize_llm_answer(notes_raw)
        coverage = model_coverage or retrieval_coverage

        yield _sse("notes_done", {"coverage": coverage, "answer": notes_answer})

        app.logger.info(
            "[req:%s] /ask-stream notes tokens=%d raw_len=%d clean_len=%d cov=%s chat=%s",
            request_id,
            notes_tokens,
            len(notes_raw or ""),
            len(notes_answer or ""),
            coverage,
            chat_mode,
        )

        # Phase: Extra (only in single question mode)
        if not chat_mode:
            do_extra = False
            if include_extra and extra_mode != "never":
                if extra_mode == "always":
                    do_extra = True
                else:
                    do_extra = model_coverage != "full"

            app.logger.info(
                "[req:%s] /ask-stream extra_decision include_extra=%s extra_mode=%s do_extra=%s",
                request_id,
                include_extra,
                extra_mode,
                do_extra,
            )

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

                extra_tokens = 0
                extra_raw = ""
                for token in chat_completion_stream(
                    [{"role": "system", "content": system_2}, {"role": "user", "content": user_2}],
                    temperature=RAG.EXTRA_TEMPERATURE, max_tokens=RAG.EXTRA_MAX_TOKENS,
                ):
                    extra_tokens += 1
                    extra_raw += token
                    yield _sse("token", {"target": "extra", "content": token})

                extra_answer, _ = sanitize_llm_answer(extra_raw)
                if extra_answer and not extra_answer.lstrip().lower().startswith("extra context (not from notes):"):
                    extra_answer = "Extra context (not from notes):\n" + extra_answer.strip()

                app.logger.info(
                    "[req:%s] /ask-stream extra tokens=%d raw_len=%d clean_len=%d",
                    request_id,
                    extra_tokens,
                    len(extra_raw or ""),
                    len(extra_answer or ""),
                )
                yield _sse("extra_done", {"answer": extra_answer})

        # Done
        app.logger.info(
            "[req:%s] /ask-stream complete notes_len=%d extra_len=%d dur_ms=%.1f",
            request_id,
            len(notes_raw or ""),
            len(extra_raw or ""),
            (time.perf_counter() - request_started) * 1000,
        )
        yield _sse(
            "done",
            {
                "coverage": coverage,
                "request_id": request_id,
                "notes_len": len(notes_raw or ""),
                "extra_len": len(extra_raw or ""),
                "ok": True,
            },
        )

    return _stream_response(generate())

# Problem generation endpoint
@app.post("/problem")
def generate_problem():
    data = request.get_json(force=True) or {}
    request_id = current_request_id()

    q = (data.get("question") or "").strip()
    notes_answer = (data.get("notes_answer") or "").strip()

    if not q:
        return jsonify(error="Missing 'question'", request_id=request_id), 400

    # Retrieve relevant chunks for the topic
    q_emb = embed_query(q)
    hits = retriever.search(q_emb, top_k=RAG.TOP_K_DEFAULT, log_hits=True)
    sources, source_blocks = render_sources(hits)

    app.logger.info("[req:%s] /problem q=%r hits=%d", request_id, q[:120], len(hits))

    def generate():
        yield _sse("meta", {"request_id": request_id})
        yield _sse("status", {"phase": "thinking"})

        system_p = (
            "You are a course assistant that generates practice problems for students.\n"
            "Based on the SOURCES and the student's question, create ONE mathematical problem.\n\n"
            "RULES:\n"
            "- The problem must be solvable using the material in the sources\n"
            "- Include a clear, specific task (e.g. 'Compute...', 'Find...', 'Show that...')\n"
            "- The problem should have a concrete numerical or symbolic answer\n"
            "- Use LaTeX for all math formulas\n"
            "- After the problem, on a separate line write ANSWER: followed by the correct answer\n"
            "- Keep the answer concise (a number, expression, or short derivation)\n\n"
            "Output format:\n"
            "**Problem:**\n"
            "(problem statement)\n\n"
            "ANSWER: (correct answer)\n"
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

        app.logger.info("[req:%s] /problem complete raw_len=%d", request_id, len(raw))
        yield _sse("done", {"request_id": request_id, "ok": True})

    return _stream_response(generate())


# Answer assessment endpoint
@app.post("/assess")
def assess_answer():
    data = request.get_json(force=True) or {}
    request_id = current_request_id()

    problem = (data.get("problem") or "").strip()
    student_answer = (data.get("student_answer") or "").strip()

    if not problem:
        return jsonify(error="Missing 'problem'", request_id=request_id), 400
    if not student_answer:
        return jsonify(error="Missing 'student_answer'", request_id=request_id), 400

    app.logger.info(
        "[req:%s] /assess problem_len=%d answer_len=%d",
        request_id,
        len(problem),
        len(student_answer),
    )

    def generate():
        yield _sse("meta", {"request_id": request_id})
        yield _sse("status", {"phase": "assessing"})

        system_a = (
            "You are a course assistant that assesses student answers to math problems.\n\n"
            "RULES:\n"
            "- Compare the student's answer to the correct answer in the problem\n"
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
            "PROBLEM AND CORRECT ANSWER:\n" + problem
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

        app.logger.info("[req:%s] /assess complete raw_len=%d", request_id, len(raw))
        yield _sse("done", {"request_id": request_id, "ok": True})

    return _stream_response(generate())
