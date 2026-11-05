"""
Ingest pipeline: collect markdown files → chunk → embed → store in Qdrant.
"""

from pathlib import Path
from typing import NamedTuple
import hashlib
import json
import os
import uuid
import warnings

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointIdsList, VectorParams

from ingest.chunker import CHUNKER_VERSION, chunk_markdown
from ingest.embed import get_embeddings_model
from ingest.pdf import convert_pdf_to_markdown
from ingest import repos
from ingest.runs import DATA_DIR, acquire_lock, is_running, tracked_run
from config import Config

# Qdrant runs on the same machine (localhost), so an API key over plain http is fine
warnings.filterwarnings("ignore", message="Api key is used with an insecure connection")

# The notes repositories are not listed here: they are managed on the ingest page
# and stored in /app/data/repos.json (ingest/repos.py).

# Course-scoped upload directory for PDFs uploaded via the ingest web UI.
# Each subdirectory is a course name, e.g. /app/data/courses/ma1101/week1.pdf
COURSES_DIR = DATA_DIR / "courses"

# Uploaded files marked as copyrighted in the ingest UI, by source path as
# stored in Qdrant (e.g. "IMAx3011/book.md"). Answers may still use them, but
# the chat UI never shows their text: each chunk carries metadata.copyrighted.
RIGHTS_FILE = COURSES_DIR / "rights.json"


def load_copyrighted() -> set[str]:
    try:
        data = json.loads(RIGHTS_FILE.read_text(encoding="utf-8"))
        return set(data.get("copyrighted", []))
    except (OSError, ValueError):
        return set()


def save_copyrighted(sources: set[str]) -> None:
    RIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = RIGHTS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps({"copyrighted": sorted(sources)}, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(RIGHTS_FILE)


def set_copyrighted_in_qdrant(source: str, copyrighted: bool) -> None:
    """Tag an already indexed source's chunks, without re-embedding."""
    client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
    if Config.QDRANT_COLLECTION not in [c.name for c in client.get_collections().collections]:
        return
    client.set_payload(
        collection_name=Config.QDRANT_COLLECTION,
        payload={"copyrighted": copyrighted},
        key="metadata",
        points=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]),
    )


def _rel_source(f: Path) -> str:
    """An uploaded file's source name: its path in the uploads folder, e.g. IMAx3011/book.md."""
    try:
        return f.relative_to(COURSES_DIR).as_posix()
    except ValueError:
        return f.name


class SourceFile(NamedTuple):
    path: Path
    key: str  # identity in the manifest and Qdrant point ids: unique across courses
    source: str  # metadata.source: what answers, links and the reader show and look up
    course: str | None


def collect_files() -> list[SourceFile]:
    files: list[SourceFile] = []

    for repo in repos.list_repos():
        root = repo.path
        if not root.exists():
            print(f"[ingest] warning: {repo.course}: clone {root} is missing; its files are skipped")
            continue
        found: list[Path] = []
        # 1) root index.qmd
        if (root / "index.qmd").exists():
            found.append(root / "index.qmd")
        # 2) wrappers in /pages
        if (root / "pages").exists():
            found.extend(sorted((root / "pages").glob("*.qmd")))
        # 3) actual content in /_includes/**/*.md (recursive)
        if (root / "_includes").exists():
            found.extend(sorted((root / "_includes").rglob("*.md")))
        # 4) PDFs anywhere in the repo
        found.extend(sorted(p for p in root.rglob("*.pdf") if ".git" not in p.relative_to(root).parts))

        for p in found:
            rel = p.relative_to(root).as_posix()
            if repo.excluded(rel):
                continue
            # Two repos may both have pages/uke10.qmd: the course keeps them apart
            files.append(SourceFile(p, f"{repo.course}:{rel}", rel, repo.course))

    # 5) Uploaded files via the ingest web UI (course-scoped): PDF, Markdown, QMD
    if COURSES_DIR.exists():
        for pattern in ("*.pdf", "*.md", "*.qmd"):
            for p in sorted(COURSES_DIR.rglob(pattern)):
                rel = _rel_source(p)
                parts = rel.split("/")
                files.append(SourceFile(p, rel, rel, parts[0] if len(parts) > 1 else None))

    return files


def list_courses() -> list[str]:
    """Courses from the notes repositories, the upload folders and the ingest manifest."""
    courses: set[str] = set(repos.repo_courses())
    if COURSES_DIR.exists():
        courses.update(p.name for p in COURSES_DIR.iterdir() if p.is_dir())
    manifest = _load_manifest()
    if manifest:
        for meta in manifest.get("sources", {}).values():
            course = meta.get("course")
            if course:
                courses.add(course)
    return sorted(courses)


def _manifest_path() -> Path:
    return Path(Config.VECTOR_DB_PATH) / "manifest.json"


def _load_manifest() -> dict | None:
    path = _manifest_path()
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_manifest(manifest: dict) -> None:
    path = _manifest_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _chunk_id(key: str, index: int) -> str:
    # Stable per source key + position, independent of what else changes in
    # the same ingest run, so unrelated files' chunks keep the same Qdrant
    # point IDs across runs.
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{key}::{index}"))


def main(force_full: bool = False, trigger: str = "cli", lock=None):
    """
    One ingest run, recorded in ingest/runs.py (lock, status page, history)
    whichever way it was started. Raises RunBusy if another run is in progress.
    """
    with tracked_run(trigger, force_full=force_full, lock=lock):
        _ingest(force_full)


def _ingest(force_full: bool) -> None:
    Config.validate()

    files = collect_files()
    if not files:
        raise RuntimeError(
            "No files found: no notes repository has files to ingest and nothing is uploaded. "
            "Check the repositories on the ingest page."
        )

    embeddings = get_embeddings_model()
    embedding_dim = len(embeddings.embed_query("dimension probe"))

    client = QdrantClient(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
    )

    manifest = _load_manifest()
    collection_exists = Config.QDRANT_COLLECTION in [
        c.name for c in client.get_collections().collections
    ]
    force_full = force_full or os.getenv("FORCE_FULL_REINGEST", "false").lower() == "true"

    # A full rebuild is required if: the collection doesn't exist yet, we
    # have no manifest to trust (e.g. first run after upgrading from the old
    # always-rebuild pipeline, or the manifest was wiped), the embedding
    # dimension changed (Qdrant can't resize an existing collection), or the
    # user explicitly asked for one via FORCE_FULL_REINGEST=true.
    full_rebuild = (
        not collection_exists
        or manifest is None
        or manifest.get("embedding_dim") != embedding_dim
        or force_full
    )

    if full_rebuild:
        if collection_exists:
            client.delete_collection(Config.QDRANT_COLLECTION)
            print(f"[ingest] full rebuild: deleted existing collection {Config.QDRANT_COLLECTION}")
        client.create_collection(
            collection_name=Config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )
        print(f"[ingest] created collection: {Config.QDRANT_COLLECTION} (dim={embedding_dim})")
        sources: dict = {}
    else:
        sources = dict(manifest.get("sources", {}))

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=Config.QDRANT_COLLECTION,
        embedding=embeddings,
    )

    copyrighted = load_copyrighted()
    to_delete_ids: list[str] = []
    to_upsert_docs: list[Document] = []
    to_upsert_ids: list[str] = []
    seen_sources: set[str] = set()
    skipped = 0
    changed = 0

    for sf in files:
        f, key, rel, course = sf
        seen_sources.add(key)
        raw_bytes = f.read_bytes()
        # The chunker version is part of the hash, so a chunker change re-chunks every file
        content_hash = hashlib.sha256(raw_bytes + f"\0chunker-v{CHUNKER_VERSION}".encode()).hexdigest()

        prior = sources.get(key)
        if prior and prior.get("hash") == content_hash:
            skipped += 1
            continue

        changed += 1
        if f.suffix.lower() == ".pdf":
            print(f"[ingest] converting PDF via marker: {key}")
            text = convert_pdf_to_markdown(f)
        else:
            try:
                text = raw_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = raw_bytes.decode("utf-8", errors="replace")

        chunks = chunk_markdown(text, source_path=rel)
        chunks = [c for c in chunks if len(c["text"].strip()) >= 50]

        if prior:
            to_delete_ids.extend(prior.get("chunk_ids", []))

        new_ids = [_chunk_id(key, i) for i in range(len(chunks))]
        for i, c in enumerate(chunks):
            meta: dict = {
                "source": rel,
                "heading": c.get("title", ""),
                "chunk_index": i,
            }
            if course:
                meta["course"] = course
            if rel in copyrighted:
                meta["copyrighted"] = True
            to_upsert_docs.append(
                Document(
                    page_content=c["text"],
                    metadata=meta,
                )
            )
            to_upsert_ids.append(new_ids[i])

        entry: dict = {"hash": content_hash, "chunk_ids": new_ids, "source": rel}
        if course:
            entry["course"] = course
        sources[key] = entry

    # Sources that used to exist but were removed (deleted file, file removed via the UI,
    # repo removed or now skipped, or an entry from before keys included the course)
    removed_sources = set(sources.keys()) - seen_sources
    for key in removed_sources:
        to_delete_ids.extend(sources[key].get("chunk_ids", []))
        del sources[key]

    total_chunks = sum(len(s["chunk_ids"]) for s in sources.values())
    if total_chunks == 0:
        raise RuntimeError("No chunks produced across any source file (after filtering).")

    if to_delete_ids:
        client.delete(
            collection_name=Config.QDRANT_COLLECTION,
            points_selector=PointIdsList(points=to_delete_ids),
        )

    if to_upsert_docs:
        vector_store.add_documents(to_upsert_docs, ids=to_upsert_ids)

    _save_manifest(
        {
            "qdrant_collection": Config.QDRANT_COLLECTION,
            "embedding_dim": embedding_dim,
            "sources": sources,
        }
    )

    print(
        f"[ingest] files: {len(files)} | changed: {changed} | skipped (unchanged): {skipped} "
        f"| removed: {len(removed_sources)} | total chunks: {total_chunks} | dim: {embedding_dim}"
    )
    print(f"[ingest] stored in Qdrant collection: {Config.QDRANT_COLLECTION}")


if __name__ == "__main__":
    # From the command line (the notes-repo trigger): wait for a run that is in
    # progress instead of skipping, so the notes changes still get ingested
    if is_running():
        print("[ingest] another ingest run is in progress; waiting for it to finish", flush=True)
    main(trigger="cli", lock=acquire_lock(wait=True))