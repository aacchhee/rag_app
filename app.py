import re
from flask import Flask, request, jsonify

from config import Config
from ingest.embed import embed_texts
from rag.retriever import Retriever
from rag.llm import chat_completion
from rag import settings as RAG

import logging
logging.basicConfig(level=logging.INFO)



app = Flask(__name__)

# Load retriever once at startup
retriever = Retriever(Config.VECTOR_DB_PATH)


def strip_think(text: str) -> str:
    """
    Some models emit hidden reasoning blocks like:
      <think> ... </think>
    Remove them before returning to users.
    """
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


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
            txt = txt[:RAG.MAX_CHARS_PER_CHUNK].rstrip() + "\n…(truncated)…"

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


import re

# ---- Coverage parsing ----

def extract_coverage(text: str) -> str | None:
    """
    Parse a line like:
      COVERAGE: full|partial|none
    from the model output.
    """
    m = re.search(
        r"^\s*COVERAGE:\s*(full|partial|none)\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    return m.group(1).lower() if m else None


def remove_coverage_line(text: str) -> str:
    return re.sub(
        r"^\s*COVERAGE:\s*(full|partial|none)\s*$",
        "",
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    ).strip()


# ---- Think stripping + general cleanup ----

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

# common dangling "connectors" we want to drop if they appear at the very end
_DANGLING_END_RE = re.compile(r"(\n|\s)+(og|and|but|som|which|that)\s*$", re.IGNORECASE)

# used to trim a non-terminated final fragment back to the last strong punctuation
_LAST_PUNCT_RE = re.compile(r"[.!?:»”\)](?!.*[.!?:»”\)])")


def sanitize_llm_text(s: str) -> str:
    """
    Removes chain-of-thought tags and cleans up common truncated endings.
    Keeps LaTeX intact.
    """
    if not s:
        return ""

    # Remove <think>...</think> blocks
    s = _THINK_BLOCK_RE.sub("", s)
    # Remove stray <think> or </think>
    s = _THINK_TAG_RE.sub("", s)

    # Strip surrounding whitespace
    s = s.strip()

    # Drop dangling connector at end ("og", "and", etc.)
    s = _DANGLING_END_RE.sub("", s).strip()

    # Only do punctuation-trim if the answer is reasonably long.
    # Otherwise Markdown headings like "### Sammenfatning:" can get nuked.
#    if len(s) >= 300 and s[-1] not in ".?!:»”)]":
#        m = _LAST_PUNCT_RE.search(s)
#        if m:
#            s = s[: m.end()].strip()

    return s


def sanitize_llm_answer(raw: str) -> tuple[str, str | None]:
    """
    Returns:
      (clean_text_without_coverage_line, coverage_or_none)

    Order matters:
      - remove think
      - extract coverage
      - remove coverage line
      - final cleanup
    """
    if not raw:
        return "", None

    raw = sanitize_llm_text(raw)

    cov = extract_coverage(raw)

    txt = remove_coverage_line(raw)
    txt = sanitize_llm_text(txt)  # run once more after removing coverage line

    return txt, cov


@app.get("/health")
def health():
    return jsonify(status="ok")


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

    extra_answer_raw = ""
    extra_answer = None


    # 1) Retrieve from notes
    q_emb = embed_texts([q])[0]
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
        [
            {"role": "system", "content": system_1},
            {"role": "user", "content": user_1},
        ],
        temperature=RAG.NOTES_TEMPERATURE,
        max_tokens=RAG.NOTES_MAX_TOKENS
    )
    app.logger.info("RAW notes len=%d tail=%r", len(notes_answer_raw or ""), (notes_answer_raw or "")[-120:])

    notes_answer, model_coverage = sanitize_llm_answer(notes_answer_raw)
    coverage = (model_coverage or retrieval_coverage)

    app.logger.info("SAN notes len=%d tail=%r", len(notes_answer or ""), (notes_answer or "")[-120:])

    # Decide if we do pass 2
    do_extra = False
    if include_extra and extra_mode != "never":
        if extra_mode == "always":
            do_extra = True
        else:
            # auto: only add extra when notes coverage isn't full
            do_extra = (model_coverage != "full")

    extra_answer = None
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
            "Question:\n"
            + q
            + "\n\nNotes-based answer (authoritative for course-specific claims):\n"
            + notes_answer
            + "\n\n(For consistency only) Retrieved sources:\n"
            + "\n".join(source_blocks)
        )

        extra_answer_raw = chat_completion(
            [
                {"role": "system", "content": system_2},
                {"role": "user", "content": user_2},
            ],
            temperature=RAG.EXTRA_TEMPERATURE,
            max_tokens=RAG.EXTRA_MAX_TOKENS,
        )
        extra_answer, _ = sanitize_llm_answer(extra_answer_raw)

        # If model fails to include the header, add it defensively
        if extra_answer and not extra_answer.lstrip().lower().startswith("extra context (not from notes):"):
            extra_answer = "Extra context (not from notes):\n" + extra_answer.strip()

    return jsonify({
        "answer_notes": notes_answer,
        "answer_extra": extra_answer,
        "coverage": coverage,
        "sources": sources,
    })

