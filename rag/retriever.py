import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np

logger = logging.getLogger("rag.retriever")

def _preview(s: str, n: int = 180) -> str:
    s = (s or "").replace("\n", "\\n")
    return s[:n] + ("…" if len(s) > n else "")


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

    def search(self, query_vec: np.ndarray, top_k: int, *, log_hits: bool = True) -> List[Hit]:
        q = query_vec.astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        D, I = self.index.search(q, top_k)  # D: distances/scores, I: indices
        hits: List[Hit] = []

        if log_hits:
            logger.info("[retrieve] top_k=%d q_shape=%s", top_k, tuple(q.shape))

        for rank, (score, idx) in enumerate(zip(D[0].tolist(), I[0].tolist()), start=1):
            if idx < 0:
                continue
            c = self.chunks[idx]
            hit = Hit(
                idx=int(idx),
                score=float(score),
                text=c.get("text", ""),
                source=c.get("source", c.get("path", "")),
                heading=c.get("heading"),
            )
            hits.append(hit)

            if log_hits:
                logger.info(
                    "[retrieve] #%d idx=%d score=%.6f src=%s heading=%s chars=%d preview=%s",
                    rank,
                    hit.idx,
                    hit.score,
                    hit.source,
                    (hit.heading or ""),
                    len(hit.text or ""),
                    _preview(hit.text or ""),
                )

        return hits
