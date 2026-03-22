"""
LangChain-based embedding wrapper.
"""

import logging
import time

import numpy as np
from langchain_openai import OpenAIEmbeddings
from config import Config
from log_utils import current_request_id, preview_text

# Module-level singleton
_embeddings_model: OpenAIEmbeddings | None = None
logger = logging.getLogger("gunicorn.error")


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        Config.validate()
        logger.info(
            "[req:%s] [embed] creating embeddings client model=%s base_url=%s timeout=%s",
            current_request_id(),
            Config.EMBEDDINGS_MODEL,
            Config.embeddings_base_url(),
            Config.LLM_TIMEOUT,
        )
        _embeddings_model = OpenAIEmbeddings(
            model=Config.EMBEDDINGS_MODEL,
            openai_api_key=Config.LLM_API_KEY,
            openai_api_base=Config.embeddings_base_url(),
            timeout=Config.LLM_TIMEOUT,
        )
    return _embeddings_model


def get_embeddings_model() -> OpenAIEmbeddings:
    """
    Return the shared OpenAIEmbeddings instance.
    Used by ingest to pass to Qdrant.
    """
    return _get_embeddings()


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    model = _get_embeddings()
    started = time.perf_counter()
    logger.info(
        "[req:%s] [embed] embed_texts count=%d batch_size=%d sample=%r",
        current_request_id(),
        len(texts),
        batch_size,
        preview_text(texts[0], 120) if texts else "",
    )
    try:
        vectors = model.embed_documents(texts)
        arr = np.array(vectors, dtype="float32")
        if arr.ndim != 2:
            raise RuntimeError(f"Embeddings array has unexpected shape: {arr.shape}")
        logger.info(
            "[req:%s] [embed] embed_texts complete shape=%s dur_ms=%.1f",
            current_request_id(),
            arr.shape,
            (time.perf_counter() - started) * 1000,
        )
        return arr
    except Exception:
        logger.exception(
            "[req:%s] [embed] embed_texts failed count=%d",
            current_request_id(),
            len(texts),
        )
        raise


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string. Returns np.ndarray of shape (dim,).
    Uses embed_query (some providers optimize query vs document embedding differently).
    """
    model = _get_embeddings()
    started = time.perf_counter()
    logger.info(
        "[req:%s] [embed] embed_query len=%d preview=%r",
        current_request_id(),
        len(text or ""),
        preview_text(text, 120),
    )
    try:
        vector = model.embed_query(text)
        arr = np.array(vector, dtype="float32")
        logger.info(
            "[req:%s] [embed] embed_query complete dim=%d dur_ms=%.1f",
            current_request_id(),
            len(arr),
            (time.perf_counter() - started) * 1000,
        )
        return arr
    except Exception:
        logger.exception(
            "[req:%s] [embed] embed_query failed preview=%r",
            current_request_id(),
            preview_text(text, 120),
        )
        raise
