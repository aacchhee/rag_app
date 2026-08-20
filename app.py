import json
import re
import time
import uuid
import logging

from flask import Flask, request, jsonify, Response, stream_with_context, g
from werkzeug.exceptions import HTTPException

from config import Config
from cleanup import sanitize_llm_answer, sanitize_llm_text
from ingest.embed import embed_query
from ingest.ingest import list_courses
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
    if not Config.allowed_chat_models():
        missing.append("CHAT_MODEL (or the LLM API must be reachable)")
    if not Config.QDRANT_URL:
        missing.append("QDRANT_URL")
    return missing


def _log_startup_config() -> None:
    missing = _missing_config_keys()
    if missing:
        app.logger.warning("startup config missing=%s", ",".join(missing))
    extra_body = Config.chat_extra_body()
    app.logger.info(
        "startup config debug=%s chat_default_model=%s allowed_chat_models=%s chat_base=%s embed_model=%s embed_base=%s qdrant_url=%s collection=%s timeout=%s thinking=false extra_body_keys=%s",
        Config.DEBUG,
        Config.default_chat_model() or "-",
        ",".join(Config.allowed_chat_models().keys()) or "-",
        Config.chat_base_url() if Config.CHAT_URL else "",
        Config.EMBEDDINGS_MODEL,
        Config.embeddings_base_url() if Config.EMBEDDINGS_URL else "",
        Config.QDRANT_URL,
        Config.QDRANT_COLLECTION,
        Config.LLM_TIMEOUT,
        ",".join(sorted(extra_body.keys())) if extra_body else "-",
    )


def _request_started_at() -> float:
    return getattr(g, "request_started_at", time.perf_counter())


def _payload_summary(
    q: str,
    *,
    extra_mode: str | None = None,
    include_extra: bool | None = None,
    chat_model: str | None = None,
    chat_mode: bool | None = None,
    history: list | None = None,
    top_k = None,
    course: str | None = None,
) -> str:
    parts = [f"q_len={len(q)}", f"q={preview_text(q, 120)!r}"]
    if include_extra is not None:
        parts.append(f"include_extra={include_extra}")
    if extra_mode is not None:
        parts.append(f"extra_mode={extra_mode}")
    if chat_model is not None:
        parts.append(f"chat_model={chat_model}")
    if chat_mode is not None:
        parts.append(f"chat_mode={chat_mode}")
    if history is not None:
        parts.append(f"history_turns={len(history)}")
    if top_k is not None:
        parts.append(f"top_k={top_k}")
    if course is not None:
        parts.append(f"course={course}")
    return " ".join(parts)


def _resolve_request_chat_model(data: dict, *, request_id: str):
    requested_chat_model = data.get("chat_model")

    if requested_chat_model is not None and not isinstance(requested_chat_model, str):
        return None, (
            jsonify(error="chat_model must be a string", request_id=request_id),
            400,
        )

    try:
        return Config.resolve_chat_model(requested_chat_model), None
    except ValueError as exc:
        return None, (jsonify(error=str(exc), request_id=request_id), 400)
    except RuntimeError as exc:
        return None, (jsonify(error=str(exc), request_id=request_id), 500)


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


_PROBLEM_PREFIX_RE = re.compile(r"^\s*\*\*Problem:?\*\*\s*", re.IGNORECASE)
_PLAIN_PROBLEM_PREFIX_RE = re.compile(r"^\s*Problem:?\s*", re.IGNORECASE)
_PROBLEM_TRAILING_SECTION_RE = re.compile(
    r"(?im)^\s*(?:answer|solution|final answer)\s*:"
)


def _normalize_problem_text(raw: str) -> str:
    text = sanitize_llm_text(raw)
    if not text:
        return ""

    text = _PROBLEM_TRAILING_SECTION_RE.split(text, maxsplit=1)[0].strip()
    text = _PROBLEM_PREFIX_RE.sub("", text, count=1)
    text = _PLAIN_PROBLEM_PREFIX_RE.sub("", text, count=1)

    if not text:
        return ""

    return "**Problem:**\n" + text.strip()


def _problem_body(problem: str) -> str:
    text = _normalize_problem_text(problem)
    text = _PROBLEM_PREFIX_RE.sub("", text, count=1)
    return text.strip()


def _extract_embedded_answer(problem: str) -> str:
    text = sanitize_llm_text(problem)
    parts = _PROBLEM_TRAILING_SECTION_RE.split(text, maxsplit=1)
    if len(parts) < 2:
        return ""
    return sanitize_llm_text(parts[1]).strip()


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


@app.get("/chat-models")
def chat_models():
    return jsonify(
        models=Config.public_chat_models(),
        default_chat_model=Config.default_chat_model(),
        request_id=current_request_id(),
    )


@app.get("/diagnose-llm")
def diagnose_llm():
    """Hit the LLM with a trivial prompt using our own wrappers.

    This exercises the exact _extract_delta / _extract_message logic
    so we can see whether the model returns content, reasoning, or nothing.
    """
    request_id = current_request_id()
    chat_model, error_response = _resolve_request_chat_model({}, request_id=request_id)
    if error_response:
        return error_response

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say exactly the word 'pong' and nothing else."},
    ]

    # --- Non-streaming through our wrapper ---
    ns_content = "(error)"
    ns_reasoning = "(error)"
    ns_error = None
    try:
        ns_content, ns_diagnostics = chat_completion(
            messages,
            temperature=0.0,
            max_tokens=50,
            chat_model=chat_model,
        )
        ns_reasoning_len = ns_diagnostics.get("reasoning_len", -1)
    except Exception as e:
        ns_error = str(e)
        ns_reasoning_len = -1

    # --- Streaming through our wrapper ---
    stream_tokens: list[str] = []
    stream_error = None
    try:
        for token in chat_completion_stream(
            messages,
            temperature=0.0,
            max_tokens=50,
            chat_model=chat_model,
        ):
            stream_tokens.append(token)
    except Exception as e:
        stream_error = str(e)

    return jsonify(
        request_id=request_id,
        chat_model=chat_model,
        base_url=Config.chat_base_url(),
        notes_max_tokens=RAG.NOTES_MAX_TOKENS,
        env_chat_enable_thinking=Config.CHAT_ENABLE_THINKING,
        non_streaming={
            "content": ns_content,
            "content_len": len(ns_content) if ns_content != "(error)" else -1,
            "reasoning_len": ns_reasoning_len,
            "error": ns_error,
        },
        streaming={
            "token_count": len(stream_tokens),
            "total_chars": sum(len(t) for t in stream_tokens),
            "tokens": stream_tokens,
            "error": stream_error,
        },
    )


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

    chat_model, error_response = _resolve_request_chat_model(data, request_id=request_id)
    if error_response:
        return error_response

    if not q:
        return jsonify(error="Missing 'question'", request_id=request_id), 400
    if len(q) > Config.MAX_QUESTION_LENGTH:
        return jsonify(error="Question too long", request_id=request_id), 400
    if extra_mode not in ("auto", "always", "never"):
        return jsonify(error="extra_mode must be one of: auto, always, never", request_id=request_id), 400
    if top_k <= 0:
        return jsonify(error="top_k must be positive", request_id=request_id), 400

    course = (data.get("course") or "").strip() or None

    app.logger.info(
        "[req:%s] /ask payload %s",
        request_id,
        _payload_summary(
            q,
            include_extra=include_extra,
            extra_mode=extra_mode,
            chat_model=chat_model,
            top_k=top_k,
            course=course,
        ),
    )

    extra_answer = None

    # 1) Retrieve from notes
    retrieval_started = time.perf_counter()
    q_emb = embed_query(q)
    hits = retriever.search(q_emb, top_k=top_k, log_hits=True, course=course)
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
        chat_model=chat_model,
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
            chat_model=chat_model,
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
        "chat_model": chat_model,
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

    chat_model, error_response = _resolve_request_chat_model(data, request_id=request_id)
    if error_response:
        return error_response

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

    course = (data.get("course") or "").strip() or None

    app.logger.info(
        "[req:%s] /ask-stream payload %s",
        request_id,
        _payload_summary(
            q,
            include_extra=include_extra,
            extra_mode=extra_mode,
            chat_model=chat_model,
            chat_mode=chat_mode,
            history=history,
            top_k=top_k,
            course=course,
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
                "chat_model": chat_model,
            },
        )
        # Phase: Thinking (retrieval)
        yield _sse("status", {"phase": "thinking"})

        retrieval_started = time.perf_counter()
        q_emb = embed_query(q)
        hits = retriever.search(q_emb, top_k=top_k, log_hits=True, course=course)
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
            chat_model=chat_model,
        ):
            notes_tokens += 1
            notes_raw += token
            app.logger.info(
                "[req:%s] /ask-stream token #%d len=%d token_preview=%r",
                request_id,
                notes_tokens,
                len(token),
                token[:80],
            )
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
                    chat_model=chat_model,
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
                "chat_model": chat_model,
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

    # Support both single-question mode and chat-mode (full conversation context)
    conversation_context = (data.get("conversation_context") or "").strip()
    q = (data.get("question") or "").strip()
    notes_answer = (data.get("notes_answer") or "").strip()
    difficulty = (data.get("difficulty") or "medium").strip().lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    chat_model, error_response = _resolve_request_chat_model(data, request_id=request_id)
    if error_response:
        return error_response

    if not conversation_context and not q:
        return jsonify(error="Missing 'question' or 'conversation_context'", request_id=request_id), 400

    # Use conversation context for retrieval when available; otherwise fall back to the single question
    retrieval_query = conversation_context if conversation_context else q
    q_emb = embed_query(retrieval_query)
    hits = retriever.search(q_emb, top_k=RAG.TOP_K_DEFAULT, log_hits=True)
    sources, source_blocks = render_sources(hits)

    app.logger.info(
        "[req:%s] /problem q_len=%d ctx_len=%d hits=%d chat_model=%s",
        request_id,
        len(q),
        len(conversation_context),
        len(hits),
        chat_model,
    )

    def generate():
        yield _sse("meta", {"request_id": request_id, "chat_model": chat_model})
        yield _sse("status", {"phase": "thinking"})

        system_p = (
            "You are a course assistant that generates practice problems for students.\n"
            "Based on the SOURCES and the student's context, create ONE mathematical problem.\n\n"
            "RULES:\n"
            "- The problem must be solvable using the material in the sources\n"
            "- Include a clear, specific task (e.g. 'Compute...', 'Find...', 'Show that...')\n"
            "- The problem should have a concrete numerical or symbolic answer\n"
            "- Do not include any solution, hints, answer key, or final answer\n"
            "- Use LaTeX for all math formulas\n"
            "- Keep the problem self-contained\n\n"
            "Output format:\n"
            "**Problem:**\n"
            "(problem statement)\n"
        )

        user_p = (
            "SOURCES:\n" + "\n".join(source_blocks)
        )

        if conversation_context:
            user_p += "\n\nConversation context:\n" + conversation_context
        else:
            user_p += "\n\nStudent's question: " + q

        if notes_answer:
            user_p += "\n\nThe answer given to the student (for context):\n" + notes_answer

        user_p += f"\n\nDifficulty level: {difficulty}."
        user_p += "\n\nGenerate one practice problem related to this topic."

        yield _sse("status", {"phase": "generating"})

        raw = ""
        for token in chat_completion_stream(
            [{"role": "system", "content": system_p}, {"role": "user", "content": user_p}],
            temperature=0.5, max_tokens=2000,
            chat_model=chat_model,
        ):
            raw += token
            yield _sse("token", {"content": token})

        problem_text = _normalize_problem_text(raw)
        app.logger.info(
            "[req:%s] /problem complete raw_len=%d clean_len=%d",
            request_id,
            len(raw),
            len(problem_text),
        )
        yield _sse("problem_done", {"problem": problem_text})
        yield _sse("done", {"request_id": request_id, "chat_model": chat_model, "ok": True})

    return _stream_response(generate())


# Generated answer endpoint
@app.post("/calculate-answer")
def calculate_answer():
    data = request.get_json(force=True) or {}
    request_id = current_request_id()

    problem = (data.get("problem") or "").strip()
    topic_question = (data.get("topic_question") or data.get("question") or "").strip()
    notes_answer = (data.get("notes_answer") or "").strip()
    conversation_context = (data.get("conversation_context") or "").strip()
    chat_model, error_response = _resolve_request_chat_model(data, request_id=request_id)
    if error_response:
        return error_response

    if not problem:
        return jsonify(error="Missing 'problem'", request_id=request_id), 400

    problem_text = _problem_body(problem)
    if not problem_text:
        return jsonify(error="Problem is empty after normalization", request_id=request_id), 400
    retrieval_query = problem_text
    if topic_question:
        retrieval_query = topic_question + "\n\n" + problem_text
    elif conversation_context:
        retrieval_query = conversation_context + "\n\n" + problem_text

    q_emb = embed_query(retrieval_query)
    hits = retriever.search(q_emb, top_k=RAG.TOP_K_DEFAULT, log_hits=True)
    _, source_blocks = render_sources(hits)

    app.logger.info(
        "[req:%s] /calculate-answer problem_len=%d topic_len=%d ctx_len=%d hits=%d chat_model=%s",
        request_id,
        len(problem_text),
        len(topic_question),
        len(conversation_context),
        len(hits),
        chat_model,
    )

    def generate():
        yield _sse("meta", {"request_id": request_id, "chat_model": chat_model})
        yield _sse("status", {"phase": "thinking"})

        system_s = (
            "You are a course assistant that solves math practice problems.\n"
            "Use the SOURCES when they are relevant and rely on standard mathematical reasoning.\n"
            "Solve the exact problem you are given.\n\n"
            "RULES:\n"
            "- Do not change the problem or introduce a different task\n"
            "- Show the key steps clearly and concisely\n"
            "- Use LaTeX for all math formulas\n"
            "- End with a line exactly in this format:\n"
            "FINAL ANSWER: (final answer)\n"
        )

        user_s = (
            "SOURCES:\n" + "\n".join(source_blocks)
            + "\n\nGENERATED PRACTICE PROBLEM:\n" + problem_text
        )

        if topic_question:
            user_s += "\n\nOriginal student topic request:\n" + topic_question
        if conversation_context:
            user_s += "\n\nConversation context:\n" + conversation_context
        if notes_answer:
            user_s += "\n\nContext from the earlier notes-based answer:\n" + notes_answer

        user_s += "\n\nSolve the generated practice problem."

        yield _sse("status", {"phase": "solving"})

        raw = ""
        for token in chat_completion_stream(
            [{"role": "system", "content": system_s}, {"role": "user", "content": user_s}],
            temperature=0.3, max_tokens=2000,
            chat_model=chat_model,
        ):
            raw += token
            yield _sse("token", {"content": token})

        solution_text = sanitize_llm_text(raw)
        app.logger.info(
            "[req:%s] /calculate-answer complete raw_len=%d clean_len=%d",
            request_id,
            len(raw),
            len(solution_text),
        )
        yield _sse("solution_done", {"answer": solution_text})
        yield _sse(
            "done",
            {
                "request_id": request_id,
                "chat_model": chat_model,
                "answer_len": len(solution_text),
                "ok": True,
            },
        )

    return _stream_response(generate())


# Hint endpoint
@app.post("/hint")
def generate_hint():
    data = request.get_json(force=True) or {}
    request_id = current_request_id()

    problem = (data.get("problem") or "").strip()
    chat_model, error_response = _resolve_request_chat_model(data, request_id=request_id)
    if error_response:
        return error_response

    if not problem:
        return jsonify(error="Missing 'problem'", request_id=request_id), 400

    app.logger.info(
        "[req:%s] /hint problem_len=%d chat_model=%s",
        request_id,
        len(problem),
        chat_model,
    )

    def generate():
        yield _sse("meta", {"request_id": request_id, "chat_model": chat_model})
        yield _sse("status", {"phase": "thinking"})

        system_h = (
            "You are a helpful tutor. A student is stuck on a math problem.\n"
            "Give them ONE small hint. Do NOT solve the problem.\n"
            "Do NOT give the answer or reveal the full method.\n"
            "Just nudge the student toward the correct approach.\n\n"
            "Output format:\n"
            "**Hint:**\n"
            "(one sentence hint)\n"
        )

        user_h = (
            "I am stuck on this problem:\n\n"
            + problem
            + "\n\nGive me one hint."
        )

        yield _sse("status", {"phase": "generating"})

        raw = ""
        for token in chat_completion_stream(
            [{"role": "system", "content": system_h}, {"role": "user", "content": user_h}],
            temperature=0.4, max_tokens=300,
            chat_model=chat_model,
        ):
            raw += token
            yield _sse("token", {"content": token})

        app.logger.info(
            "[req:%s] /hint complete raw_len=%d",
            request_id,
            len(raw),
        )
        yield _sse("done", {"request_id": request_id, "chat_model": chat_model, "ok": True})

    return _stream_response(generate())


# Answer assessment endpoint
@app.post("/assess")
def assess_answer():
    data = request.get_json(force=True) or {}
    request_id = current_request_id()

    problem = (data.get("problem") or "").strip()
    generated_answer = (data.get("generated_answer") or "").strip()
    student_answer = (data.get("student_answer") or "").strip()
    chat_model, error_response = _resolve_request_chat_model(data, request_id=request_id)
    if error_response:
        return error_response

    if not generated_answer:
        generated_answer = _extract_embedded_answer(problem)
        problem = _normalize_problem_text(problem)

    if not problem:
        return jsonify(error="Missing 'problem'", request_id=request_id), 400
    if not generated_answer:
        return jsonify(error="Missing 'generated_answer'", request_id=request_id), 400
    if not student_answer:
        return jsonify(error="Missing 'student_answer'", request_id=request_id), 400

    app.logger.info(
        "[req:%s] /assess problem_len=%d generated_len=%d answer_len=%d chat_model=%s",
        request_id,
        len(problem),
        len(generated_answer),
        len(student_answer),
        chat_model,
    )

    def generate():
        yield _sse("meta", {"request_id": request_id, "chat_model": chat_model})
        yield _sse("status", {"phase": "assessing"})

        system_a = (
            "You are a course assistant that assesses student answers to math problems.\n\n"
            "RULES:\n"
            "- Compare the student's answer to the official generated solution\n"
            "- Accept mathematically equivalent answers even if the wording or steps differ\n"
            "- If the student gives only the final answer, judge whether that final answer is correct\n"
            "- Be encouraging but honest\n"
            "- If wrong, explain where the mistake is and give a hint\n"
            "- If partially correct, acknowledge what's right and point out what's missing\n"
            "- If correct, confirm and optionally add a brief insight\n"
            "- Use LaTeX for math formulas\n\n"
            "Output format:\n"
            "**Result:** Correct / Partially correct / Incorrect\n\n"
            "(explanation)\n"
        )

        user_a = (
            "PROBLEM:\n" + problem
            + "\n\nOFFICIAL GENERATED SOLUTION:\n" + generated_answer
            + "\n\nSTUDENT'S ANSWER:\n" + student_answer
            + "\n\nAssess the student's answer."
        )

        raw = ""
        for token in chat_completion_stream(
            [{"role": "system", "content": system_a}, {"role": "user", "content": user_a}],
            temperature=0.3, max_tokens=2000,
            chat_model=chat_model,
        ):
            raw += token
            yield _sse("token", {"content": token})

        app.logger.info("[req:%s] /assess complete raw_len=%d", request_id, len(raw))
        yield _sse("done", {"request_id": request_id, "chat_model": chat_model, "ok": True})

    return _stream_response(generate())


@app.get("/courses")
def get_courses():
    return jsonify(list_courses())


@app.get("/status-page")
def status_page():
    return Response(_STATUS_PAGE, mimetype="text/html")


_STATUS_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ingest Service Status</title>
<style>
  * { box-sizing: border-box; }
  body {
    font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
    background: #f6f7f9;
    color: #1a1a1a;
    display: flex;
    flex-direction: column;
    align-items: center;
    min-height: 100vh;
    margin: 0;
    padding: 3rem 1rem;
    gap: 1.5rem;
  }
  .card {
    background: #fff;
    border-radius: 12px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    max-width: 720px;
    width: 100%;
    padding: 2rem 2.2rem;
  }
  h1 { margin: 0 0 0.2rem; font-size: 1.35rem; }
  .subtitle { color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }
  .row { display: flex; align-items: center; justify-content: space-between; padding: 0.7rem 0; border-bottom: 1px solid #eee; }
  .row:last-child { border-bottom: none; }
  .label { font-weight: 500; color: #444; }
  .badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-size: 0.85rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em;
    padding: 0.35rem 0.75rem; border-radius: 999px;
  }
  .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .status-idle { background: #e8f5e9; color: #2a7a2a; }
  .status-idle .dot { background: #2a7a2a; }
  .status-running { background: #fff3e0; color: #b8860b; }
  .status-running .dot { background: #b8860b; }
  .status-done { background: #e3f2fd; color: #1565c0; }
  .status-done .dot { background: #1565c0; }
  .status-error { background: #ffebee; color: #c0392b; }
  .status-error .dot { background: #c0392b; }
  .status-down { background: #f5f5f5; color: #666; }
  .status-down .dot { background: #999; }
  .timestamp { font-variant-numeric: tabular-nums; color: #555; }
  .log-box {
    background: #111; color: #ddd;
    border-radius: 8px; padding: 1rem;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.82rem; line-height: 1.45;
    white-space: pre-wrap; word-break: break-word;
    max-height: 300px; overflow-y: auto;
    margin-top: 0.8rem;
  }
  .error-msg { color: #c0392b; font-weight: 500; }
  .footer { text-align: center; margin-top: 1.2rem; font-size: 0.8rem; color: #888; }

  /* --- History bars --- */
  .bar-chart { display: flex; align-items: flex-end; gap: 2px; height: 60px; margin-top: 0.8rem; }
  .bar {
    flex: 1;
    min-width: 2px;
    border-radius: 2px 2px 0 0;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  .bar:hover { opacity: 0.7; }
  .bar-idle { background: #2a7a2a; }
  .bar-running { background: #b8860b; }
  .bar-done { background: #1565c0; }
  .bar-error { background: #c0392b; }
  .bar-down { background: #d0d0d0; }
  .bar-legend { display: flex; gap: 1rem; font-size: 0.75rem; color: #555; margin-top: 0.5rem; flex-wrap: wrap; }
  .bar-legend span { display: inline-flex; align-items: center; gap: 0.3rem; }
  .bar-legend i { width: 10px; height: 10px; border-radius: 2px; display: inline-block; }
  .range-btns { display: flex; gap: 0.4rem; margin-bottom: 0.8rem; }
  .range-btns button { font-size: 0.8rem; padding: 0.25rem 0.6rem; border: 1px solid #ccc; background: #fff; border-radius: 6px; cursor: pointer; }
  .range-btns button.active { background: #1a1a1a; color: #fff; border-color: #1a1a1a; }
  #history-info { font-size: 0.8rem; color: #666; margin-top: 0.3rem; min-height: 1.2rem; }
</style>
</head>
<body>
  <div class="card">
    <h1>Ingest Service Status</h1>
    <div class="subtitle">Live view into the document ingestion pipeline</div>

    <div class="row">
      <span class="label">State</span>
      <span id="state-badge" class="badge status-down"><span class="dot"></span>Unknown</span>
    </div>
    <div class="row">
      <span class="label">Started</span>
      <span id="started" class="timestamp">&mdash;</span>
    </div>
    <div class="row">
      <span class="label">Finished</span>
      <span id="finished" class="timestamp">&mdash;</span>
    </div>
    <div class="row">
      <span class="label">Last checked</span>
      <span id="last-check" class="timestamp">&mdash;</span>
    </div>

    <div id="error-row" class="row" style="display:none;">
      <span class="label">Error</span>
      <span id="error-text" class="error-msg"></span>
    </div>

    <div id="log-container" style="display:none;">
      <div class="row" style="border-bottom:none;padding-bottom:0.2rem;">
        <span class="label">Log output</span>
      </div>
      <pre id="log" class="log-box"></pre>
    </div>

    <div class="footer">Refreshes automatically every 5 seconds</div>
  </div>

  <div class="card">
    <h1>Ingest History</h1>
    <div class="subtitle">Per-bucket availability over the selected window</div>

    <div class="range-btns">
      <button id="btn-24h" class="active" onclick="setRange(24)">Last 24h</button>
      <button id="btn-7d" onclick="setRange(7 * 24)">Last 7 days</button>
      <button id="btn-30d" onclick="setRange(30 * 24)">Last 30 days</button>
    </div>

    <div id="bar-chart" class="bar-chart"></div>
    <div id="history-info"></div>

    <div class="bar-legend">
      <span><i class="bar-idle"></i> Idle</span>
      <span><i class="bar-running"></i> Running</span>
      <span><i class="bar-done"></i> Done</span>
      <span><i class="bar-error"></i> Error</span>
      <span><i class="bar-down"></i> Down / No data</span>
    </div>
  </div>

<script>
const badge = document.getElementById('state-badge');
const startedEl = document.getElementById('started');
const finishedEl = document.getElementById('finished');
const lastCheckEl = document.getElementById('last-check');
const errorRow = document.getElementById('error-row');
const errorText = document.getElementById('error-text');
const logContainer = document.getElementById('log-container');
const logEl = document.getElementById('log');

function fmt(ts) {
  if (!ts) return '&mdash;';
  try { return new Date(ts).toLocaleString(); } catch { return ts; }
}

async function poll() {
  let state = { status: 'down', started_at: null, finished_at: null, error: null, log: '' };
  try {
    const res = await fetch('/ingest/status', { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    state = await res.json();
  } catch (e) {
    state.status = 'down';
    state.error = e.message;
  }

  const status = state.status || 'down';
  badge.className = 'badge status-' + status;
  badge.innerHTML = '<span class="dot"></span>' + status;

  startedEl.textContent = fmt(state.started_at);
  finishedEl.textContent = fmt(state.finished_at);
  lastCheckEl.textContent = new Date().toLocaleTimeString();

  if (state.error) {
    errorRow.style.display = 'flex';
    errorText.textContent = state.error;
  } else {
    errorRow.style.display = 'none';
  }

  if (state.log) {
    logContainer.style.display = 'block';
    logEl.textContent = state.log;
    logEl.scrollTop = logEl.scrollHeight;
  } else {
    logContainer.style.display = 'none';
  }
}

poll();
setInterval(poll, 5000);

/* --- History bars --- */
let currentRangeHours = 24;

function bucketCount(hours) {
  if (hours <= 24) return 48;      // 30 min buckets
  if (hours <= 7 * 24) return 84;  // 2 h buckets
  return 60;                       // 12 h buckets for 30d
}

function setRange(hours) {
  currentRangeHours = hours;
  for (const btn of document.querySelectorAll('.range-btns button')) btn.classList.remove('active');
  if (hours <= 24) document.getElementById('btn-24h').classList.add('active');
  else if (hours <= 7 * 24) document.getElementById('btn-7d').classList.add('active');
  else document.getElementById('btn-30d').classList.add('active');
  renderHistory();
}

function parseTs(ts) {
  try { return new Date(ts).getTime(); } catch { return 0; }
}

function statePriority(s) {
  return { error: 4, running: 3, done: 2, idle: 1, down: 0 }[s] || 0;
}

async function renderHistory() {
  const chart = document.getElementById('bar-chart');
  const info = document.getElementById('history-info');
  chart.innerHTML = '';
  info.textContent = 'Loading...';

  const now = Date.now();
  const rangeMs = currentRangeHours * 3600_000;
  const startMs = now - rangeMs;
  const nBuckets = bucketCount(currentRangeHours);
  const bucketMs = rangeMs / nBuckets;

  let events = [];
  try {
    const res = await fetch('/ingest/history');
    if (res.ok) events = (await res.json()).events || [];
  } catch (e) {
    info.textContent = 'Failed to load history';
    return;
  }

  // Oldest first so carry-forward works
  events = events.slice().reverse();

  const buckets = Array(nBuckets).fill(null).map((_, i) => ({
    t0: startMs + i * bucketMs,
    t1: startMs + (i + 1) * bucketMs,
    states: [],
  }));

  let carriedState = 'down';
  let evIdx = 0;

  for (let i = 0; i < nBuckets; i++) {
    const b = buckets[i];
    // If there were no events at all before this bucket, it's down until first event
    if (evIdx === 0 && events.length > 0 && parseTs(events[0].t) > b.t1) {
      carriedState = 'down';
    }

    while (evIdx < events.length && parseTs(events[evIdx].t) <= b.t1) {
      const ev = events[evIdx];
      if (ev.type === 'run_start') carriedState = 'running';
      else if (ev.type === 'run_done') carriedState = 'done';
      else if (ev.type === 'run_error') carriedState = 'error';
      b.states.push(carriedState);
      evIdx++;
    }

    // If no events fell exactly in this bucket, use carried state
    if (!b.states.length) {
      b.finalState = carriedState;
    } else {
      // Pick highest priority state in bucket
      b.finalState = b.states.reduce((a, s) => statePriority(s) > statePriority(a) ? s : a, b.states[0]);
    }
  }

  // Render bars
  for (const b of buckets) {
    const bar = document.createElement('div');
    bar.className = 'bar bar-' + (b.finalState || 'down');
    const d0 = new Date(b.t0);
    const d1 = new Date(b.t1);
    bar.title = d0.toLocaleString() + ' — ' + d1.toLocaleString() + ': ' + (b.finalState || 'down');
    chart.appendChild(bar);
  }

  const total = buckets.length;
  const counts = {};
  for (const b of buckets) counts[b.finalState] = (counts[b.finalState] || 0) + 1;
  const pct = s => ((counts[s] || 0) / total * 100).toFixed(1);
  info.textContent = 'Idle ' + pct('idle') + '% | Running ' + pct('running') + '% | Done ' + pct('done') + '% | Error ' + pct('error') + '% | Down ' + pct('down') + '%';
}

renderHistory();
setInterval(renderHistory, 30000);
</script>
</body>
</html>
"""
