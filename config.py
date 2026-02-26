import os

class Config:
    # --- Chat ---
    CHAT_URL = os.getenv("CHAT_URL")
    CHAT_MODEL = os.getenv("CHAT_MODEL")
    CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
    CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "900"))

    # --- Flask ---
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # --- Auth / API ---
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

    # --- Embeddings ---
    EMBEDDINGS_URL = os.getenv("EMBEDDINGS_URL")
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

    # --- Retrieval ---
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")
    TOP_K = int(os.getenv("TOP_K", "5"))

    # --- Safety ---
    MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))

    # --- LangChain-compatible base URL ---
    # OpenAI-compatible endpoints: strip /v1/chat/completions or /v1/embeddings
    # to get the base URL that LangChain expects
    @classmethod
    def chat_base_url(cls) -> str:
        """Return base URL for ChatOpenAI (strip trailing path after /v1)."""
        url = cls.CHAT_URL or ""
        # If URL ends with /v1/chat/completions, strip that
        for suffix in ["/chat/completions", "/v1/chat/completions"]:
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        # Ensure it ends with /v1
        if not url.endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        return url

    @classmethod
    def embeddings_base_url(cls) -> str:
        """Return base URL for OpenAIEmbeddings (strip trailing path after /v1)."""
        url = cls.EMBEDDINGS_URL or ""
        for suffix in ["/embeddings", "/v1/embeddings"]:
            if url.endswith(suffix):
                url = url[: -len(suffix)]
                break
        if not url.endswith("/v1"):
            url = url.rstrip("/") + "/v1"
        return url

    @classmethod
    def validate(cls) -> None:
        missing = []
        if not cls.LLM_API_KEY:
            missing.append("LLM_API_KEY")
        if not cls.EMBEDDINGS_URL:
            missing.append("EMBEDDINGS_URL")
        if not cls.EMBEDDINGS_MODEL:
            missing.append("EMBEDDINGS_MODEL")
        if not cls.CHAT_URL:
            missing.append("CHAT_URL")
        if not cls.CHAT_MODEL:
            missing.append("CHAT_MODEL")

        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )