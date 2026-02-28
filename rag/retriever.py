"""
Qdrant retriever — replaces FAISS file-based retrieval.
Keeps the same Hit dataclass and search() API.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams

from config import Config

logger = logging.getLogger("gunicorn.error")


def _preview(s: str, n: int = 1800) -> str:
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
    Searches Qdrant for similar vectors.
    Interface:
        retriever.search(query_vec, top_k) -> List[Hit]
    """

    def __init__(self):
        self.client = QdrantClient(
            url=Config.QDRANT_URL,
            api_key=Config.QDRANT_API_KEY,
        )
        self.collection = Config.QDRANT_COLLECTION

        # Verify collection exists
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            raise RuntimeError(
                f"Qdrant collection '{self.collection}' not found. "
                f"Available: {collections}. Run ingest first."
            )

        # Get point count for logging
        info = self.client.get_collection(self.collection)
        logger.info(
            "[retriever] connected to Qdrant collection '%s' (%d points)",
            self.collection,
            info.points_count,
        )

    def search(
        self, query_vec: np.ndarray, top_k: int, *, log_hits: bool = True
    ) -> List[Hit]:
        q = query_vec.astype("float32")
        if q.ndim != 1:
            q = q.flatten()

        if log_hits:
            logger.info("[retrieve] top_k=%d vec_dim=%d", top_k, len(q))

        results = self.client.search(
            collection_name=self.collection,
            query_vector=q.tolist(),
            limit=top_k,
            with_payload=True,
        )

        hits: List[Hit] = []
        for rank, point in enumerate(results, start=1):
            payload = point.payload or {}

            hit = Hit(
                idx=payload.get("chunk_index", rank),
                score=float(point.score),
                text=payload.get("page_content", payload.get("text", "")),
                source=payload.get("source", ""),
                heading=payload.get("heading"),
            )
            hits.append(hit)

            if log_hits:
                logger.info(
                    "[retrieve] #%d id=%s score=%.6f src=%s heading=%s chars=%d preview=%s",
                    rank,
                    point.id,
                    hit.score,
                    hit.source,
                    (hit.heading or ""),
                    len(hit.text or ""),
                    _preview(hit.text or ""),
                )

        return hits