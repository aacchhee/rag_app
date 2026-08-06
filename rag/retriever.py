"""
Qdrant retriever — replaces FAISS file-based retrieval.
Keeps the same Hit dataclass and search() API.
"""

import logging
from dataclasses import dataclass
from typing import List

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

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
            logger.warning(
                "[req:%s] [retriever] Qdrant collection '%s' not found. "
                "Available: %s. Ingest notes to populate it.",
                current_request_id(),
                self.collection,
                collections,
            )
            self._ready = False
        else:
            self._ready = True
            # Get point count for logging
            info = self.client.get_collection(self.collection)
            logger.info(
                "[req:%s] [retriever] connected to Qdrant collection '%s' (%d points)",
                current_request_id(),
                self.collection,
                info.points_count,
            )

    def _ensure_ready(self) -> bool:
        if self._ready:
            return True
        # Re-check: collection may have been created after startup
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection in collections:
            self._ready = True
            info = self.client.get_collection(self.collection)
            logger.info(
                "[req:%s] [retriever] collection '%s' now available (%d points)",
                current_request_id(),
                self.collection,
                info.points_count,
            )
        return self._ready

    def search(
        self, query_vec: np.ndarray, top_k: int, *, log_hits: bool = True, course: str | None = None
    ) -> List[Hit]:
        if not self._ensure_ready():
            if log_hits:
                logger.info(
                    "[req:%s] [retrieve] collection not ready, returning empty hits",
                    current_request_id(),
                )
            return []

        q = query_vec.astype("float32")
        if q.ndim != 1:
            q = q.flatten()

        if log_hits:
            logger.info(
                "[req:%s] [retrieve] top_k=%d vec_dim=%d course=%s",
                current_request_id(),
                top_k,
                len(q),
                course or "-",
            )

        q_filter = None
        if course:
            q_filter = Filter(
                must=[
                    FieldCondition(
                        key="metadata.course",
                        match=MatchValue(value=course),
                    )
                ]
            )

        try:
            results = self.client.query_points(
                collection_name=self.collection,
                query=q.tolist(),
                limit=top_k,
                query_filter=q_filter,
                with_payload=True,
            ).points
        except Exception:
            logger.exception(
                "[req:%s] [retrieve] query_points failed collection=%s top_k=%d vec_dim=%d course=%s",
                current_request_id(),
                self.collection,
                top_k,
                len(q),
                course or "-",
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
