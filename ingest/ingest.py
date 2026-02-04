from pathlib import Path
import json
import faiss
import numpy as np

from ingest.chunker import chunk_markdown
from ingest.embed import embed_texts
from config import Config


REPO_DIR = Path("/home/rag/notes_repo")


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

    out_dir = Path(Config.VECTOR_DB_PATH)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = collect_files()
    if not files:
        raise RuntimeError(f"No files found under {REPO_DIR}. Check repo layout / paths.")

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

    texts = [c["text"] for c in all_chunks]
    embeddings = embed_texts(texts)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "index.faiss"))
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    with open(out_dir / "info.json", "w", encoding="utf-8") as f:
        json.dump({
            "repo_dir": str(REPO_DIR),
            "files_indexed": [str(p.relative_to(REPO_DIR)) for p in files],
            "chunks": len(all_chunks),
            "embedding_dim": dim,
        }, f, ensure_ascii=False, indent=2)

    print(f"[ingest] files: {len(files)} | chunks: {len(all_chunks)} | dim: {dim}")
    print(f"[ingest] wrote: {out_dir / 'index.faiss'}")
    print(f"[ingest] wrote: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
