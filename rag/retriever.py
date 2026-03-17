"""
Qdrant retriever — replaces FAISS file-based retrieval.
Keeps the same Hit dataclass and search() API.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from qdrant_client import QdrantClient

from config import Config
from log_utils import current_request_id

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
            "[req:%s] [retriever] connected to Qdrant collection '%s' (%d points)",
            current_request_id(),
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
            logger.info(
                "[req:%s] [retrieve] top_k=%d vec_dim=%d",
                current_request_id(),
                top_k,
                len(q),
            )

        try:
            results = self.client.query_points(
                collection_name=self.collection,
                query=q.tolist(),
                limit=top_k,
                with_payload=True,
            ).points
        except Exception:
            logger.exception(
                "[req:%s] [retrieve] query_points failed collection=%s top_k=%d vec_dim=%d",
                current_request_id(),
                self.collection,
                top_k,
                len(q),
            )
            raise

        hits: List[Hit] = []
        for rank, point in enumerate(results, start=1):
            payload = point.payload or {}
            meta = payload.get("metadata", {})

            hit = Hit(
                idx=meta.get("chunk_index", rank),
                score=float(point.score),
                text=payload.get("page_content", ""),
                source=meta.get("source", ""),
                heading=meta.get("heading"),
            )
            hits.append(hit)

            if log_hits:
                logger.info(
                    "[req:%s] [retrieve] #%d id=%s score=%.6f src=%s heading=%s chars=%d preview=%s",
                    current_request_id(),
                    rank,
                    point.id,
                    hit.score,
                    hit.source,
                    (hit.heading or ""),
                    len(hit.text or ""),
                    _preview(hit.text or ""),
                )

        return hits
