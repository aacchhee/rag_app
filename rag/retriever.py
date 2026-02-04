import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np


@dataclass
class Hit:
    idx: int
    score: float
    text: str
    source: str
    heading: str | None = None


class Retriever:
    """
    Loads:
      - index.faiss
      - meta.json (expects either a list of chunks OR {"chunks":[...]} )

    Each chunk dict should contain:
      - text
      - source (or path)
      - heading (optional)
    """

    def __init__(self, vector_dir: str):
        vdir = Path(vector_dir)
        self.index = faiss.read_index(str(vdir / "index.faiss"))
        meta = json.loads((vdir / "meta.json").read_text(encoding="utf-8"))

        if isinstance(meta, dict) and "chunks" in meta:
            self.chunks: List[Dict[str, Any]] = meta["chunks"]
        elif isinstance(meta, list):
            self.chunks = meta
        else:
            raise RuntimeError("meta.json format not recognized (expected list or {chunks:[...]})")

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Hit]:
        q = query_vec.astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        D, I = self.index.search(q, top_k)  # D: distances/scores, I: indices
        hits: List[Hit] = []
        for score, idx in zip(D[0].tolist(), I[0].tolist()):
            if idx < 0:
                continue
            c = self.chunks[idx]
            hits.append(
                Hit(
                    idx=int(idx),
                    score=float(score),
                    text=c.get("text", ""),
                    source=c.get("source", c.get("path", "")),
                    heading=c.get("heading"),
                )
            )
        return hits
