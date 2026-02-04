import os

class Config:
    # --- Flask ---
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # --- LLM ---
    LLM_API_URL = os.getenv("LLM_API_URL")
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

    # --- Retrieval ---
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")
    TOP_K = int(os.getenv("TOP_K", "5"))

    # --- Safety ---
    MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))

    @classmethod
    def validate(cls):
        missing = []
        if not cls.LLM_API_URL:
            missing.append("LLM_API_URL")
        if not cls.LLM_API_KEY:
            missing.append("LLM_API_KEY")

        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
            )
