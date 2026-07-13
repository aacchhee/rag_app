"""
Ingest pipeline: collect markdown files → chunk → embed → store in Qdrant.
"""

from pathlib import Path
import json
import uuid

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from ingest.chunker import chunk_markdown
from ingest.embed import get_embeddings_model
from ingest.pdf import convert_pdf_to_markdown
from config import Config


REPO_DIR = Path("/home/rag/notes_repo")

# Separate, writable mount for PDFs uploaded via the ingest web UI (ingest/web.py).
# Kept outside of REPO_DIR, which is mounted read-only.
PDF_UPLOAD_DIR = Path("/app/data/pdf_uploads")


def _rel_source(f: Path) -> str:
    for base, prefix in ((REPO_DIR, ""), (PDF_UPLOAD_DIR, "uploaded_pdfs/")):
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

    # 4) PDFs anywhere in the repo (converted to markdown via `marker`)
    files.extend(sorted(REPO_DIR.rglob("*.pdf")))

    # 5) PDFs uploaded via the ingest web UI
    if PDF_UPLOAD_DIR.exists():
        files.extend(sorted(PDF_UPLOAD_DIR.rglob("*.pdf")))

    return files


def main():
    Config.validate()

    files = collect_files()
    if not files:
        raise RuntimeError(
            f"No files found under {REPO_DIR}. Check repo layout / paths."
        )

    # 1) Chunk all files
    all_chunks = []
    for f in files:
        if f.suffix.lower() == ".pdf":
            print(f"[ingest] converting PDF via marker: {f}")
            text = convert_pdf_to_markdown(f)
        else:
            try:
                text = f.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = f.read_text(encoding="utf-8", errors="replace")

        rel = _rel_source(f)
        chunks = chunk_markdown(text, source_path=rel)
        # Filter out tiny chunks
        chunks = [c for c in chunks if len(c["text"].strip()) >= 50]
        all_chunks.extend(chunks)

    if not all_chunks:
        raise RuntimeError("No chunks produced (after filtering).")

    # 2) Convert to LangChain Documents
    documents = []
    for i, c in enumerate(all_chunks):
        doc = Document(
            page_content=c["text"],
            metadata={
                "source": c.get("source", ""),
                "heading": c.get("title", ""),
                "chunk_index": i,
            },
        )
        documents.append(doc)

    # 3) Determine embedding dimension from the actual configured model
    embeddings = get_embeddings_model()
    embedding_dim = len(embeddings.embed_query("dimension probe"))

    # 4) Connect to Qdrant and recreate collection
    client = QdrantClient(
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
    )

    # Delete existing collection if it exists (clean re-index)
    collections = [c.name for c in client.get_collections().collections]
    if Config.QDRANT_COLLECTION in collections:
        client.delete_collection(Config.QDRANT_COLLECTION)
        print(f"[ingest] deleted existing collection: {Config.QDRANT_COLLECTION}")

    # Create fresh collection
    client.create_collection(
        collection_name=Config.QDRANT_COLLECTION,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    print(f"[ingest] created collection: {Config.QDRANT_COLLECTION} (dim={embedding_dim})")

    # 5) Add documents via LangChain (handles embedding + upload)
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=Config.QDRANT_COLLECTION,
        embedding=embeddings,
    )

    # Generate stable UUIDs based on index
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk-{i}")) for i in range(len(documents))]

    vector_store.add_documents(documents, ids=ids)

    # 6) Save info for reference (optional, not used at runtime)
    out_dir = Path(Config.VECTOR_DB_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "repo_dir": str(REPO_DIR),
                "files_indexed": [_rel_source(p) for p in files],
                "chunks": len(all_chunks),
                "embedding_dim": embedding_dim,
                "qdrant_collection": Config.QDRANT_COLLECTION,
                "storage": "qdrant",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[ingest] files: {len(files)} | chunks: {len(all_chunks)} | dim: {embedding_dim}")
    print(f"[ingest] stored in Qdrant collection: {Config.QDRANT_COLLECTION}")


if __name__ == "__main__":
    main()