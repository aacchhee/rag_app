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
from config import Config


REPO_DIR = Path("/home/rag/notes_repo")

# Must match your embedding model output
EMBEDDING_DIM = 4096


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
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = f.read_text(encoding="utf-8", errors="replace")

        rel = str(f.relative_to(REPO_DIR))
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

    # 3) Connect to Qdrant and recreate collection
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
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    print(f"[ingest] created collection: {Config.QDRANT_COLLECTION} (dim={EMBEDDING_DIM})")

    # 4) Add documents via LangChain (handles embedding + upload)
    embeddings = get_embeddings_model()

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=Config.QDRANT_COLLECTION,
        embedding=embeddings,
    )

    # Generate stable UUIDs based on index
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk-{i}")) for i in range(len(documents))]

    vector_store.add_documents(documents, ids=ids)

    # 5) Save info for reference (optional, not used at runtime)
    out_dir = Path(Config.VECTOR_DB_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "repo_dir": str(REPO_DIR),
                "files_indexed": [str(p.relative_to(REPO_DIR)) for p in files],
                "chunks": len(all_chunks),
                "embedding_dim": EMBEDDING_DIM,
                "qdrant_collection": Config.QDRANT_COLLECTION,
                "storage": "qdrant",
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[ingest] files: {len(files)} | chunks: {len(all_chunks)} | dim: {EMBEDDING_DIM}")
    print(f"[ingest] stored in Qdrant collection: {Config.QDRANT_COLLECTION}")


if __name__ == "__main__":
    main()