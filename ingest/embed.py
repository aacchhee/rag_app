import requests
import numpy as np
from config import Config


def embed_texts(texts, batch_size: int = 64) -> np.ndarray:
    """
    Embed texts using the university API.
    Expected endpoint: {LLM_API_URL}/embeddings
    Expected payload: {"input": [...]}
    Expected response: {"data": [{"embedding": [...]}, ...]}
    """
    Config.validate()

    headers = {
        "Authorization": f"Bearer {Config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"input": batch}

        r = requests.post(
            Config.LLM_API_URL.rstrip("/") + "/embeddings",
            headers=headers,
            json=payload,
            timeout=Config.LLM_TIMEOUT,
        )
        r.raise_for_status()

        j = r.json()
        data = j.get("data")
        if not data or len(data) != len(batch):
            raise RuntimeError(f"Unexpected embeddings response shape: got {len(data) if data else 0}, expected {len(batch)}")

        vecs = [item["embedding"] for item in data]
        all_vecs.extend(vecs)

    arr = np.array(all_vecs, dtype="float32")
    if arr.ndim != 2:
        raise RuntimeError(f"Embeddings array has unexpected shape: {arr.shape}")
    return arr
