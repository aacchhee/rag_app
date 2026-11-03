"""
Links from retrieved chunks back to the published lecture notes.

A chunk's metadata holds its source file (relative to the notes repo), its
heading and its index within the file. The notes are Quarto websites: content
lives in _includes/**/*.md, pulled into pages/*.qmd with {{< include >}}. So
the map is built from the repos, without touching the embeddings:

  1. Walk every page with its includes expanded, in document order, and give
     each heading the id Pandoc gives it (repeated headings get -1, -2, ...).
  2. For each source file, re-chunk it exactly as the ingest does and find the
     last real heading before each chunk (chunker splits also on '#' comments
     inside code blocks, which are not headings).

Result: (course, source, chunk_index) -> https://site/pages/page2.html#anchor.
Built when the backend starts; rag_update_and_ingest restarts rag2 after the
notes change, which rebuilds it.
"""

import logging
import re
from pathlib import Path

from ingest.chunker import chunk_markdown, prepare_markdown

logger = logging.getLogger("gunicorn.error")

_INCLUDE_RE = re.compile(r"\{\{<\s*include\s+([^\s>]+)\s*>\}\}")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_ATTRS_RE = re.compile(r"\s*\{([^{}]*)\}\s*$")
_FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
_DIV_RE = re.compile(r"^(:{3,})\s*(.*)$")
_MIN_CHUNK_CHARS = 50  # same filter as ingest/ingest.py


def _stringify_inline(text: str) -> str:
    """Plain text of a heading's inline Markdown, as Pandoc's stringify sees it."""
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)  # images -> alt text
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links -> text
    text = re.sub(r"\[\^[^\]]*\]", "", text)  # footnote refs
    text = re.sub(r"<[^>]+>", "", text)  # inline HTML tags
    # Math and code keep their raw text (u_{xx} -> u_xx); only the prose around
    # them loses its emphasis markers
    parts = re.split(r"(\$+[^$]*\$+|`[^`]*`)", text)
    out = []
    for i, part in enumerate(parts):
        if i % 2:
            out.append(part.strip("$`"))
        else:
            out.append(re.sub(r"(\*\*|__|\*|_(?=\S)|(?<=\S)_)", "", part))
    return "".join(out)


def heading_label(heading_text: str) -> str:
    """A heading as plain text for display: '**0.1 [EVALUATING](#page-5-0)**' -> '0.1 EVALUATING'."""
    text, _ = _split_attrs(heading_text or "")
    return " ".join(_stringify_inline(text).split())


def pandoc_identifier(heading_text: str) -> str:
    """Pandoc's auto_identifiers (non-GFM): what Quarto uses for section ids."""
    text = _stringify_inline(heading_text).lower()
    text = "".join(c for c in text if c.isspace() or c.isalnum() or c in "-_.")
    text = "-".join(text.split())
    i = 0  # drop everything before the first letter
    while i < len(text) and not text[i].isalpha():
        i += 1
    text = text[i:]
    return text or "section"


def _split_attrs(heading_text: str) -> tuple[str, str | None]:
    """'Title {#id .class}' -> ('Title', 'id')."""
    m = _ATTRS_RE.search(heading_text)
    if not m:
        return heading_text, None
    explicit = None
    for part in m.group(1).split():
        if part.startswith("#"):
            explicit = part[1:]
    return heading_text[: m.start()], explicit


def _scan(text: str):
    """
    Yield (offset, kind, match) for each line outside code blocks: kind is
    "heading", "div", "include" or "text". Like Pandoc, a '#' line that
    continues a paragraph (no blank line before it) is text, not a heading.
    """
    fence = None
    offset = 0
    in_paragraph = False
    for line in text.splitlines(keepends=True):
        line_offset, offset = offset, offset + len(line)
        stripped = line.strip()
        m = _FENCE_RE.match(line)
        if m:
            run = m.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and not stripped[len(run):].strip():
                fence = None
            in_paragraph = False
            continue
        if fence is not None:
            continue
        if not stripped:
            in_paragraph = False
            continue
        div = _DIV_RE.match(stripped)
        if div:
            in_paragraph = False
            yield line_offset, "div", div
            continue
        include = _INCLUDE_RE.search(stripped)
        if include and stripped.startswith("{{<"):
            in_paragraph = False
            yield line_offset, "include", include
            continue
        h = _HEADING_RE.match(line.rstrip("\r\n"))
        if h and not in_paragraph:
            yield line_offset, "heading", h
            continue
        in_paragraph = True
        yield line_offset, "text", None


def _headings(text: str) -> list[tuple[int, str, str | None]]:
    """(offset, title, explicit_id) of the real headings."""
    out = []
    for offset, kind, m in _scan(text):
        if kind == "heading":
            title, explicit = _split_attrs(m.group(2))
            out.append((offset, title.strip(), explicit))
    return out


class SourceLinks:
    def __init__(self):
        self._chunk_urls: dict[tuple[str, str], list[str | None]] = {}
        self._heading_urls: dict[tuple[str, str], dict[str, str]] = {}
        self._page_urls: dict[tuple[str, str], str] = {}

    # --- building -------------------------------------------------------

    @classmethod
    def build(cls, repos: list[tuple[Path, str]], site_urls: dict[str, str]) -> "SourceLinks":
        links = cls()
        for repo_dir, course in repos:
            site = site_urls.get(course)
            if not site or not repo_dir.exists():
                continue
            try:
                links._build_repo(Path(repo_dir).resolve(), course, site.rstrip("/") + "/")
            except Exception:
                logger.exception("[source-links] failed to map %s", repo_dir)
        logger.info("[source-links] mapped %d source files", len(links._page_urls))
        return links

    def _build_repo(self, repo: Path, course: str, site: str) -> None:
        pages = [repo / "index.qmd"] if (repo / "index.qmd").exists() else []
        pages += sorted((repo / "pages").glob("*.qmd")) if (repo / "pages").exists() else []
        # Quarto drops the ids of tab headings; the CTM package's tab-ids.lua filter keeps them
        quarto_yml = repo / "_quarto.yml"
        keeps_tab_ids = quarto_yml.exists() and "tab-ids.lua" in quarto_yml.read_text(encoding="utf-8", errors="replace")
        # source file -> [(offset in its stripped text, url or None)] for its headings;
        # None: the heading has no id on the page (a callout title or a dropped tab heading)
        anchors: dict[str, list[tuple[int, str | None]]] = {}

        for page in pages:
            rel_page = page.relative_to(repo).as_posix()
            page_url = site + ("" if rel_page == "index.qmd" else rel_page[: -len(".qmd")] + ".html")
            state = {"used": set(), "divs": [], "keeps_tab_ids": keeps_tab_ids}
            self._walk(repo, course, page, page_url, state, anchors, depth=0)

        for source, entries in anchors.items():
            self._map_chunks(repo, course, source, entries)

    def _walk(self, repo: Path, course: str, path: Path, page_url: str, state: dict,
              anchors: dict[str, list[tuple[int, str | None]]], depth: int) -> None:
        """
        Assign ids to the headings of `path` (includes expanded) in document order,
        as Pandoc does: every heading takes an id, also those Quarto later turns
        into a callout title or a tab title (which then has none on the page).
        `state` carries the used ids and the open ::: divs across includes.
        """
        if depth > 10 or not path.exists():
            return
        source = path.relative_to(repo).as_posix()
        text = prepare_markdown(path.read_text(encoding="utf-8", errors="replace"))
        first_page = source not in anchors  # a file included on several pages links to the first
        if first_page:
            anchors[source] = []
            self._page_urls[(course, source)] = page_url
        divs = state["divs"]

        for line_offset, kind, m in _scan(text):
            if kind == "div":
                attrs = m.group(2).strip()
                if attrs:
                    is_callout = bool(re.search(r"(^|[\s{.])callout-", attrs))
                    divs.append({
                        "tabset": "panel-tabset" in attrs,
                        "tab_level": None,
                        # Quarto makes the callout's first heading its title, unless title= is given
                        "title_pending": is_callout and "title=" not in attrs,
                    })
                elif divs:
                    divs.pop()
                continue
            if kind == "include":
                self._walk(repo, course, (path.parent / m.group(1)).resolve(), page_url, state, anchors, depth + 1)
                continue

            top = divs[-1] if divs else None
            if kind == "text":
                if top:
                    top["title_pending"] = False
                continue

            h = m
            title, explicit = _split_attrs(h.group(2))
            level = len(h.group(1))
            has_id = True
            if top and top["title_pending"]:
                has_id = False  # becomes the callout's title
            elif top and top["tabset"]:
                if top["tab_level"] is None:
                    top["tab_level"] = level
                if level == top["tab_level"] and not state["keeps_tab_ids"]:
                    has_id = False  # becomes a tab title
            if top:
                top["title_pending"] = False

            if explicit:
                ident = explicit
            else:
                base = pandoc_identifier(title)
                ident, n = base, 0
                while ident in state["used"]:
                    n += 1
                    ident = f"{base}-{n}"
            state["used"].add(ident)
            if first_page:
                anchors[source].append((line_offset, page_url + "#" + ident if has_id else None))

    def _map_chunks(self, repo: Path, course: str, source: str, entries: list[tuple[int, str | None]]) -> None:
        page_url = self._page_urls.get((course, source))
        path = repo / source
        text = prepare_markdown(path.read_text(encoding="utf-8", errors="replace"))

        # A heading without an id on the page links to the nearest heading above it that has one
        linked: list[tuple[int, str | None]] = []
        last = None
        for offset, url in entries:
            last = url or last
            linked.append((offset, last))
        entries = linked

        heading_urls: dict[str, str] = {}
        titles = {offset: title for offset, title, _ in _headings(text)}
        for offset, url in entries:
            if url:
                heading_urls.setdefault(titles.get(offset, ""), url)
        self._heading_urls[(course, source)] = heading_urls

        # Same chunking and filtering as the ingest, so chunk_index lines up
        chunks = [c for c in chunk_markdown(text, source_path=source) if len(c["text"].strip()) >= _MIN_CHUNK_CHARS]
        urls: list[str | None] = []
        pos = 0
        for c in chunks:
            probe = c["text"].strip()[:120]
            found = text.find(probe, pos)
            if found == -1:
                found = text.find(probe)
            if found == -1:
                urls.append(None)
                continue
            pos = found
            before = [url for offset, url in entries if offset < found and url]
            urls.append(before[-1] if before else page_url)
        self._chunk_urls[(course, source)] = urls

    # --- lookup ---------------------------------------------------------

    def url_for(self, course: str | None, source: str, chunk_index: int | None = None,
                heading: str | None = None) -> str | None:
        """Best link for a chunk: its heading, else its heading text, else its page."""
        if not course:  # points ingested without a course: any course with this file
            course = next((c for c, s in self._page_urls if s == source), "")
        key = (course, source)
        if chunk_index is not None:
            urls = self._chunk_urls.get(key) or []
            if 0 <= chunk_index < len(urls) and urls[chunk_index]:
                return urls[chunk_index]
        if heading:
            url = self._heading_urls.get(key, {}).get(heading.strip())
            if url:
                return url
        return self._page_urls.get(key)
