"""
Ingest pipeline: collect markdown files → chunk → embed → store in Qdrant.
"""

from pathlib import Path
import hashlib
import json
import os
import uuid

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointIdsList, VectorParams

from ingest.chunker import chunk_markdown
from ingest.embed import get_embeddings_model
from ingest.pdf import convert_pdf_to_markdown
from config import Config


REPO_DIR = Path("/home/rag/notes_repo")

# Course-scoped upload directory for PDFs uploaded via the ingest web UI.
# Each subdirectory is a course name, e.g. /app/data/courses/ma1101/week1.pdf
COURSES_DIR = Path("/app/data/courses")


def _course_for(f: Path) -> str | None:
    """Return course name inferred from file location, or None."""
    try:
        rel = f.relative_to(COURSES_DIR)
        parts = rel.parts
        return parts[0] if len(parts) > 1 else None
    except ValueError:
        pass
    try:
        rel = f.relative_to(REPO_DIR)
        return "IMAx2024"
    except ValueError:
        pass
    return None


def _rel_source(f: Path) -> str:
    for base, prefix in ((REPO_DIR, ""), (COURSES_DIR, "")):
        try:
            return prefix + str(f.relative_to(base))
        except ValueError:
            continue
    return f.name


def collect_files() -> list[Path]:
    files: list[Path] = []

    # 1) root index.qmd
    idx = REPO_DIR / "index.qmd"
    if idx.exists():
        files.append(idx)

    # 2) wrappers in /pages
    pages_dir = REPO_DIR / "pages"
    if pages_dir.exists():
        files.extend(sorted(pages_dir.glob("*.qmd")))

    # 3) actual content in /_includes/**/*.md (recursive)
    inc = REPO_DIR / "_includes"
    if inc.exists():
        files.extend(sorted(inc.rglob("*.md")))

    # 4) PDFs anywhere in the repo
    files.extend(sorted(REPO_DIR.rglob("*.pdf")))

    # 5) PDFs uploaded via the ingest web UI (course-scoped)
    if COURSES_DIR.exists():
        files.extend(sorted(COURSES_DIR.rglob("*.pdf")))

    return files


def list_courses() -> list[str]:
    """Return sorted list of existing course folder names."""
    if not COURSES_DIR.exists():
        return []
    return sorted([p.name for p in COURSES_DIR.iterdir() if p.is_dir()])


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


def _chunk_id(source: str, index: int) -> str:
    # Stable per source path + position, independent of what else changes in
    # the same ingest run, so unrelated files' chunks keep the same Qdrant
    # point IDs across runs.
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{source}::{index}"))


def main(force_full: bool = False):
    Config.validate()

    files = collect_files()
    if not files:
        raise RuntimeError(
            f"No files found under {REPO_DIR}. Check repo layout / paths."
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

    to_delete_ids: list[str] = []
    to_upsert_docs: list[Document] = []
    to_upsert_ids: list[str] = []
    seen_sources: set[str] = set()
    skipped = 0
    changed = 0

    for f in files:
        rel = _rel_source(f)
        seen_sources.add(rel)
        raw_bytes = f.read_bytes()
        content_hash = hashlib.sha256(raw_bytes).hexdigest()

        prior = sources.get(rel)
        if prior and prior.get("hash") == content_hash:
            skipped += 1
            continue

        changed += 1
        if f.suffix.lower() == ".pdf":
            print(f"[ingest] converting PDF via marker: {rel}")
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

        new_ids = [_chunk_id(rel, i) for i in range(len(chunks))]
        course = _course_for(f)
        for i, c in enumerate(chunks):
            meta: dict = {
                "source": c.get("source", rel),
                "heading": c.get("title", ""),
                "chunk_index": i,
            }
            if course:
                meta["course"] = course
            to_upsert_docs.append(
                Document(
                    page_content=c["text"],
                    metadata=meta,
                )
            )
            to_upsert_ids.append(new_ids[i])

        entry: dict = {"hash": content_hash, "chunk_ids": new_ids}
        if course:
            entry["course"] = course
        sources[rel] = entry

    # Sources that used to exist but were removed (deleted file / PDF removed via UI)
    removed_sources = set(sources.keys()) - seen_sources
    for rel in removed_sources:
        to_delete_ids.extend(sources[rel].get("chunk_ids", []))
        del sources[rel]

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
    main()