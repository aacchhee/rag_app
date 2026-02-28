"""
LangChain-based embedding wrapper.
"""

import numpy as np
from langchain_openai import OpenAIEmbeddings
from config import Config

# Module-level singleton
_embeddings_model: OpenAIEmbeddings | None = None


def _get_embeddings() -> OpenAIEmbeddings:
    global _embeddings_model
    if _embeddings_model is None:
        Config.validate()
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
    vectors = model.embed_documents(texts)
    arr = np.array(vectors, dtype="float32")
    if arr.ndim != 2:
        raise RuntimeError(f"Embeddings array has unexpected shape: {arr.shape}")
    return arr


def embed_query(text: str) -> np.ndarray:
    """
    Embed a single query string. Returns np.ndarray of shape (dim,).
    Uses embed_query (some providers optimize query vs document embedding differently).
    """
    model = _get_embeddings()
    vector = model.embed_query(text)
    return np.array(vector, dtype="float32")