"""
Rebuild readable text for a source from its chunks in Qdrant.

Uploaded files (PDFs converted at ingest time, Markdown) have no public page,
but every chunk keeps its text, heading and position. Put back in order, they
give the document again: the chunker drops heading lines and overlaps the
windows of long sections, so headings are put back and the overlap removed.
The source reader shows it a page (about PAGE_CHARS) at a time.
"""

import re

PAGE_CHARS = 6000
CITED_MARKER = "<span id=\"source-cited\"></span>"

# Quarto syntax that means nothing outside a Quarto build: ::: div fences
# (::: {.callout-note}, ::: {#id}, :::) and shortcodes ({{< include … >}})
_QUARTO_NOISE = re.compile(r"(?m)^[ \t]*(?::{3,}.*|\{\{<[^>]*>\}\})[ \t]*(?:\n|$)")


def _paginate(chunks: list) -> list[list]:
    pages: list[list] = []
    size = 0
    for chunk in chunks:
        n = len(chunk.text or "")
        if pages and size + n <= PAGE_CHARS:
            pages[-1].append(chunk)
            size += n
        else:
            pages.append([chunk])
            size = n
    return pages


def _overlap_start(prev: str, nxt: str) -> int | None:
    """
    Where nxt's start reappears near the end of prev: the chunker overlaps the
    windows of a long section by up to 300 chars (then strips both).
    """
    probe = nxt[:60]
    if len(probe) >= 20:
        pos = prev.rfind(probe)
        if pos != -1 and len(prev) - pos <= 400:
            return pos
    return None


def _join_windows(prev: str, nxt: str) -> str:
    """Join two windows of one long section, dropping the overlap."""
    pos = _overlap_start(prev, nxt)
    return prev[:pos] + nxt if pos is not None else prev + "\n\n" + nxt


def _drop_overlap(prev: str, nxt: str) -> str:
    """
    nxt without the part it repeats from the end of prev (a page break between
    windows). Not stripped: it continues prev exactly, whitespace included.
    """
    pos = _overlap_start(prev, nxt)
    return nxt[len(prev) - pos:] if pos is not None else nxt


def _safe_break(text: str) -> int:
    """Offset of the last paragraph break in `text` outside $…$ math (0 if none)."""
    pos = len(text)
    while True:
        pos = text.rfind("\n\n", 0, pos)
        if pos <= 0:
            return 0
        if text.count("$", 0, pos) % 2 == 0:
            return pos


def _page_texts(pages: list[list], chunk_index: int | None) -> tuple[list[str], int | None, int | None]:
    """
    Markdown for every page, plus the page and offset where `chunk_index` starts.
    Built in one pass so that a section running over a page break is split at
    a paragraph: the unfinished last paragraph moves to the next page, where
    the section continues (windows end wherever 4000 chars ran out).
    """
    texts: list[str] = []
    cited_page = cited_at = None
    prev_chunk = None
    for p, page in enumerate(pages):
        text = ""
        for chunk in page:
            body = (chunk.text or "").strip()
            heading = (chunk.heading or "").strip()
            same_section = prev_chunk is not None and heading == (prev_chunk.heading or "").strip()
            if chunk.chunk_index == chunk_index:
                cited_page, cited_at = p, len(text)
            if same_section and text:
                text = _join_windows(text, body)  # the next window of the same long section
            elif same_section:
                text = _drop_overlap((prev_chunk.text or "").strip(), body)  # continues the previous page
            else:
                if heading and heading != "Preamble":
                    body = "### " + heading + "\n\n" + body
                text = text + "\n\n" + body if text else body
            prev_chunk = chunk
        texts.append(text)

    for p in range(len(pages) - 1):
        continues = (pages[p + 1][0].heading or "").strip() == (pages[p][-1].heading or "").strip()
        if not continues:
            continue
        cut = _safe_break(texts[p])
        if not cut:
            continue
        # The next page continues this one exactly, so the tail just moves over
        tail = texts[p][cut:].lstrip("\n")
        texts[p] = texts[p][:cut].rstrip()
        texts[p + 1] = tail + texts[p + 1]
        if cited_page == p + 1 and cited_at is not None:
            cited_at += len(tail)
        elif cited_page == p and cited_at is not None and cited_at >= cut:
            cited_page, cited_at = p + 1, 0
    return [t.strip() for t in texts], cited_page, cited_at


def build_page(chunks: list, chunk_index: int | None) -> dict | None:
    """
    The page of `chunks` (Hits of one source, in order) that holds `chunk_index`,
    as Markdown. A marker before the paragraph where the cited chunk starts lets
    the reader scroll to it.
    """
    if not chunks:
        return None
    pages = _paginate(chunks)
    texts, cited_page, cited_at = _page_texts(pages, chunk_index)
    page_no = cited_page if cited_page is not None else 0
    text = texts[page_no]

    if cited_at:
        # Insert at a paragraph break, never inside math or mid-sentence
        cut = _safe_break(text[: cited_at + 2])  # +2: the break right at the cited chunk counts
        if cut:
            text = text[:cut] + "\n\n" + CITED_MARKER + "\n\n" + text[cut:].lstrip("\n")

    cited = next((c for c in pages[page_no] if c.chunk_index == chunk_index), pages[page_no][0])
    heading = (cited.heading or "").strip()
    return {
        "heading": "" if heading == "Preamble" else heading,
        "text": _QUARTO_NOISE.sub("", text),
        "page": page_no + 1,
        "pages": len(pages),
        "prev_chunk": pages[page_no - 1][0].chunk_index if page_no > 0 else None,
        "next_chunk": pages[page_no + 1][0].chunk_index if page_no + 1 < len(pages) else None,
    }
