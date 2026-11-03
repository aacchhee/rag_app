import re
from typing import List, Dict


def _strip_yaml_front_matter(text: str) -> str:
    # Removes Quarto/YAML front matter if present
    # ---\n ... \n---\n
    return re.sub(r"(?s)\A---\n.*?\n---\n", "", text, count=1)


# Bump when the chunk text for unchanged files changes, so the ingest re-chunks them
CHUNKER_VERSION = "2"

# Images embedded inline as data URIs (e.g. pasted into an uploaded .md file): one
# such image can be hundreds of KB of base64, which would fill chunks with noise
_DATA_IMAGE_MD_RE = re.compile(r"!\[([^\]]*)\]\(\s*<?data:[^)]*\)")
_DATA_IMAGE_HTML_RE = re.compile(r"<img\b[^>]*\bsrc\s*=\s*[\"']data:[^>]*>", re.IGNORECASE)
_ALT_RE = re.compile(r"\balt\s*=\s*[\"']([^\"']*)[\"']", re.IGNORECASE)


def _image_placeholder(alt: str) -> str:
    alt = " ".join(alt.split())
    return f"[Figure: {alt}]" if alt else ""


def strip_embedded_images(text: str) -> str:
    """Replace data-URI images with their alt text."""
    text = _DATA_IMAGE_MD_RE.sub(lambda m: _image_placeholder(m.group(1)), text)
    def html(m):
        alt = _ALT_RE.search(m.group(0))
        return _image_placeholder(alt.group(1) if alt else "")
    return _DATA_IMAGE_HTML_RE.sub(html, text)


def prepare_markdown(text: str) -> str:
    """The text the chunker splits: no front matter, no embedded images."""
    return strip_embedded_images(_strip_yaml_front_matter(text)).strip()


def chunk_markdown(text: str, source_path: str, max_chars: int = 4000, overlap_chars: int = 300) -> List[Dict]:
    """
    Simple, robust chunker for .md/.qmd:
    - strips YAML front matter and data-URI images
    - splits by headings (#, ##, ###...)
    - further splits long sections by character count with overlap
    Returns list of dicts: {title, text, source}
    """
    text = prepare_markdown(text)
    if not text:
        return []

    # Split on headings; keep headings
    # This yields segments starting with a heading line (or the initial content).
    parts = re.split(r"(?m)^(#{1,6})\s+(.+?)\s*$", text)
    # re.split produces: [preamble, hlevel, htitle, body, hlevel, htitle, body, ...]
    chunks = []

    preamble = parts[0].strip()
    if preamble:
        chunks.extend(_split_long(
            title="Preamble",
            body=preamble,
            source_path=source_path,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        ))

    i = 1
    while i + 2 < len(parts):
        _hlevel = parts[i]
        htitle = parts[i + 1].strip()
        body = parts[i + 2].strip()
        i += 3

        if not body:
            continue

        chunks.extend(_split_long(
            title=htitle,
            body=body,
            source_path=source_path,
            max_chars=max_chars,
            overlap_chars=overlap_chars,
        ))

    return chunks


def _split_long(title: str, body: str, source_path: str, max_chars: int, overlap_chars: int) -> List[Dict]:
    # Split body into windows of max_chars with overlap_chars overlap
    out = []
    body = body.strip()
    if len(body) <= max_chars:
        out.append({"title": title, "text": body, "source": source_path})
        return out

    start = 0
    while start < len(body):
        end = min(len(body), start + max_chars)
        window = body[start:end].strip()
        if window:
            out.append({"title": title, "text": window, "source": source_path})
        if end == len(body):
            break
        start = max(0, end - overlap_chars)

    return out
