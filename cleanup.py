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
_DANGLING_END_RE = re.compile(
    r"(\n|\s)+(og|and|but|som|which|that)\s*$", re.IGNORECASE
)

# used to trim a non-terminated final fragment back to the last strong punctuation
_LAST_PUNCT_RE = re.compile(r'[.!?:»"\)](?!.*[.!?:»"\)])')


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