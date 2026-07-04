import copy
import json
import os


import models_client


class Config:
    # --- Chat ---
    CHAT_URL = os.getenv("CHAT_URL")
    CHAT_MODEL = os.getenv("CHAT_MODEL")
    CHAT_TEMPERATURE = float(os.getenv("CHAT_TEMPERATURE", "0.2"))
    CHAT_MAX_TOKENS = int(os.getenv("CHAT_MAX_TOKENS", "900"))
    CHAT_ENABLE_THINKING = os.getenv("CHAT_ENABLE_THINKING")
    CHAT_RETRY_WITH_THINKING_DISABLED = os.getenv("CHAT_RETRY_WITH_THINKING_DISABLED", "true")
    CHAT_EXTRA_BODY_JSON = os.getenv("CHAT_EXTRA_BODY_JSON")

    # --- Legacy override for static labels ---
    # Kept purely for optional env-based manual overrides.  Most users should rely
    # on the dynamic fetch from the Idun API; this dict is merged *under* the live
    # list so it only affects label/description overrides.
    STATIC_MODEL_OVERRIDES: dict[str, dict[str, str]] = {}

    # --- Flask ---
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    # --- Auth / API ---
    LLM_API_KEY = os.getenv("LLM_API_KEY")
    LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

    # --- Embeddings ---
    EMBEDDINGS_URL = os.getenv("EMBEDDINGS_URL")
    EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL")

    # --- Qdrant ---
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "course_notes")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # None = no auth

    # --- Retrieval ---
    VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_store")
    TOP_K = int(os.getenv("TOP_K", "5"))

    # --- Safety ---
    MAX_QUESTION_LENGTH = int(os.getenv("MAX_QUESTION_LENGTH", "1000"))

    # --- LangChain-compatible base URL ---
    # OpenAI-compatible endpoints: strip /v1/chat/completions or /v1/embeddings
    # to get the base URL that LangChain expects
    @staticmethod
    def _parse_bool(raw: str | None, *, default: bool | None = None) -> bool | None:
        if raw is None or raw.strip() == "":
            return default
        value = raw.strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise RuntimeError(f"Invalid boolean value: {raw!r}")

    @classmethod
    def chat_enable_thinking(cls) -> bool | None:
        return cls._parse_bool(cls.CHAT_ENABLE_THINKING, default=None)

    @classmethod
    def chat_retry_with_thinking_disabled(cls) -> bool:
        return bool(cls._parse_bool(cls.CHAT_RETRY_WITH_THINKING_DISABLED, default=True))

    @classmethod
    def allowed_chat_models(cls) -> dict[str, dict[str, str]]:
        # 1) Fetch live model list from the Idun API (cached)
        models = models_client.fetch_available_models(
            base_url=cls.chat_base_url(),
            api_key=cls.LLM_API_KEY,
            timeout=cls.LLM_TIMEOUT,
        )

        # 2) Merge in any static overrides so env-based labels/descriptions win
        for model_id, metadata in cls.STATIC_MODEL_OVERRIDES.items():
            normalized_model_id = str(model_id or "").strip()
            if not normalized_model_id:
                raise RuntimeError("STATIC_MODEL_OVERRIDES contains an empty model id")
            if not isinstance(metadata, dict):
                raise RuntimeError(
                    f"STATIC_MODEL_OVERRIDES[{normalized_model_id!r}] must be a dict"
                )
            entry = models.get(normalized_model_id, {}).copy()
            if metadata.get("label"):
                entry["label"] = str(metadata["label"])
            if metadata.get("description"):
                entry["description"] = str(metadata["description"]).strip()
            models[normalized_model_id] = entry

        # 3) Ensure the fallback CHAT_MODEL is present even if the API is down
        fallback_model = (cls.CHAT_MODEL or "").strip()
        if fallback_model and fallback_model not in models:
            models[fallback_model] = {
                "label": fallback_model,
                "description": "Default env-configured chat model.",
            }

        return models

    @classmethod
    def default_chat_model(cls) -> str:
        configured_default = (cls.CHAT_MODEL or "").strip()
        if configured_default:
            return configured_default

        models = cls.allowed_chat_models()
        return next(iter(models), "")

    @classmethod
    def resolve_chat_model(cls, requested_model: str | None = None) -> str:
        requested = (requested_model or "").strip()
        models = cls.allowed_chat_models()

        if requested:
            if requested not in models:
                allowed = ", ".join(models.keys()) if models else "-"
                raise ValueError(
                    f"Unsupported chat_model {requested!r}. Allowed models: {allowed}"
                )
            return requested

        default_model = cls.default_chat_model()
        if default_model:
            return default_model

        raise RuntimeError(
            "No chat model configured. Set CHAT_MODEL or ensure the LLM API is reachable."
        )

    @classmethod
    def public_chat_models(cls) -> list[dict[str, str | bool]]:
        default_model = cls.default_chat_model()
        public_models: list[dict[str, str | bool]] = []

        for model_id, metadata in cls.allowed_chat_models().items():
            item: dict[str, str | bool] = {
                "id": model_id,
                "label": metadata.get("label", model_id),
                "default": model_id == default_model,
            }
            description = metadata.get("description", "")
            if description:
                item["description"] = description
            public_models.append(item)

        return public_models

    @classmethod
    def chat_extra_body(cls, *, enable_thinking: bool | None = None) -> dict | None:
        extra_body: dict = {}
        raw = cls.CHAT_EXTRA_BODY_JSON
        if raw and raw.strip():
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise RuntimeError("CHAT_EXTRA_BODY_JSON must decode to a JSON object")
            extra_body = copy.deepcopy(parsed)

        effective_enable_thinking = (
            enable_thinking
            if enable_thinking is not None
            else cls.chat_enable_thinking()
        )
        if effective_enable_thinking is not None:
            chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
            if not isinstance(chat_template_kwargs, dict):
                raise RuntimeError(
                    "CHAT_EXTRA_BODY_JSON.chat_template_kwargs must be a JSON object"
                )
            chat_template_kwargs["enable_thinking"] = effective_enable_thinking

        return extra_body or None

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
        if not cls.allowed_chat_models():
            missing.append("CHAT_MODEL (or LLM API must be reachable)")
        if not cls.QDRANT_URL:
            missing.append("QDRANT_URL")

        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )
