import requests
import numpy as np
from config import Config


def embed_texts(texts, batch_size: int = 64) -> np.ndarray:
    Config.validate()

    headers = {
        "Authorization": f"Bearer {Config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        payload = {"model": Config.EMBEDDINGS_MODEL, "input": batch}

        r = requests.post(
            Config.EMBEDDINGS_URL,
            headers=headers,
            json=payload,
            timeout=Config.LLM_TIMEOUT,
        )

        if r.status_code >= 400:
            raise RuntimeError(
                f"Embeddings request failed: {r.status_code}\n"
                f"URL: {Config.EMBEDDINGS_URL}\n"
                f"Payload keys: {list(payload.keys())}\n"
                f"Response: {r.text[:2000]}"
            )

        j = r.json()
        data = j.get("data", [])
        if len(data) != len(batch):
            raise RuntimeError(f"Unexpected embeddings response shape: got {len(data)}, expected {len(batch)}")

        all_vecs.extend([item["embedding"] for item in data])

    arr = np.array(all_vecs, dtype="float32")
    if arr.ndim != 2:
        raise RuntimeError(f"Embeddings array has unexpected shape: {arr.shape}")
    return arr
