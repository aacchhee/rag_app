"""
Client for the Idun NTNU LLM API /v1/models endpoint.
Provides dynamic model discovery with TTL caching.
"""

import logging
import time
from typing import Any

import requests

logger = logging.getLogger("gunicorn.error")

_CACHE_TTL_SECONDS = 300
_cache: dict[str, Any] = {"models": {}, "expires_at": 0}

# Optional pretty labels for known models.
MODEL_LABEL_OVERRIDES: dict[str, str] = {
    "moonshotai/Kimi-K2.6": "Kimi K2.6",
    "NorwAI/NorwAI-Magistral-24B-reasoning": "NorwAI Magistral 24B",
    "norallm/normistral-11b-thinking": "Normistral 11B",
    "NbAiLab/borealis-27b": "Borealis 27B",
    "openai/gpt-oss-120b": "GPT-OSS 120B",
    "Qwen/Qwen3-Embedding-8B": "Qwen3 Embedding 8B",
    "Qwen/Qwen3.6-27B-FP8": "Qwen 3.6 27B",
    "nvidia/GLM-5.2-NVFP4": "GLM 5.2",
    "mistralai/Mistral-Medium-3.5-128B": "Mistral Medium 3.5",
    "MiniMaxAI/MiniMax-M3-MXFP8": "MiniMax M3",
    "mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4": "Mistral Large 3",
    "zai-org/GLM-4.7-FP8": "GLM 4.7",
    "Qwen/Qwen3.5-122B-A10B-FP8": "Qwen 3.5 122B",
}


def _derive_label(model_id: str) -> str:
    """Derive a human-readable label from a model id."""
    if model_id in MODEL_LABEL_OVERRIDES:
        return MODEL_LABEL_OVERRIDES[model_id]

    # Take the last segment after '/' and clean it up
    name = model_id.split("/")[-1]
    name = name.replace("-", " ").replace("_", " ")
    # Title-case each word
    return " ".join(word.capitalize() for word in name.split())


def fetch_available_models(
    base_url: str,
    api_key: str | None,
    timeout: float = 30.0,
) -> dict[str, dict[str, str]]:
    """
    Fetch the live model list from the Idun API.
    Returns a dict mapping model_id -> {"label": ...}.
    Uses a module-level TTL cache to avoid hammering the API.
    On failure logs a warning and returns the stale cache (if any).
    """
    now = time.time()
    if now < _cache["expires_at"] and _cache["models"]:
        return _cache["models"]

    if not base_url or base_url == "/v1":
        logger.warning("[models_client] No valid base_url provided, skipping fetch")
        return _cache["models"]

    url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        logger.warning("[models_client] Failed to fetch models from %s: %s", url, exc)
        return _cache["models"]
    except ValueError as exc:
        logger.warning("[models_client] Invalid JSON from %s: %s", url, exc)
        return _cache["models"]

    data = payload.get("data", [])
    if not isinstance(data, list):
        logger.warning("[models_client] Unexpected 'data' type from %s: %s", url, type(data).__name__)
        return _cache["models"]

    models: dict[str, dict[str, str]] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        model_id = str(entry.get("id", "")).strip()
        if not model_id:
            continue
        label = _derive_label(model_id)
        models[model_id] = {"label": label}

    if not models:
        logger.warning("[models_client] No models returned from %s", url)
        return _cache["models"]

    logger.info("[models_client] Fetched %d models from %s", len(models), url)
    _cache["models"] = models
    _cache["expires_at"] = now + _CACHE_TTL_SECONDS
    return models
