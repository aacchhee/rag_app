import os

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v is not None and v.strip() != "" else default

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v is not None and v.strip() != "" else default

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v is not None and v.strip() != "" else default


# Retrieval
TOP_K_DEFAULT = _env_int("RAG_TOP_K_DEFAULT", 3)
MAX_CHARS_PER_CHUNK = _env_int("RAG_MAX_CHARS_PER_CHUNK", 1200)

# Coverage heuristic
MIN_BEST_CHUNK_CHARS_FOR_FULL = _env_int("RAG_MIN_BEST_CHUNK_CHARS_FOR_FULL", 120)

# Pass 1 (notes) generation controls
NOTES_TEMPERATURE = _env_float("RAG_NOTES_TEMPERATURE", 0.2)
NOTES_MAX_TOKENS = _env_int("RAG_NOTES_MAX_TOKENS", 450)

# Pass 2 (extra) generation controls
EXTRA_TEMPERATURE = _env_float("RAG_EXTRA_TEMPERATURE", 0.5)
EXTRA_MAX_TOKENS = _env_int("RAG_EXTRA_MAX_TOKENS", 900)

# UI behavior defaults (backend default when client doesn't specify)
EXTRA_MODE_DEFAULT = _env_str("RAG_EXTRA_MODE_DEFAULT", "auto")  # auto|always|never
INCLUDE_EXTRA_DEFAULT = _env_int("RAG_INCLUDE_EXTRA_DEFAULT", 1) == 1
