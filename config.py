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
            raise RuntimeError("Missing required environment variables: " + ", ".join(missing))
