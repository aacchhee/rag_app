"""
LangChain-based LLM wrapper.
Supports both blocking (chat_completion) and streaming (chat_completion_stream).
"""

import json
import logging
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import Config
from typing import Any, Generator
from log_utils import current_request_id

# Module-level singleton (initialized on first call)
_llm_cache: dict[str, ChatOpenAI] = {}
logger = logging.getLogger("gunicorn.error")


def _message_summary(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        content = msg.get("content") or ""
        parts.append(f"{msg.get('role', '?')}:{len(content)}")
    return ",".join(parts)


def _to_lc_messages(messages: list[dict]) -> list:
    """Convert dict messages to LangChain message objects."""
    lc_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        else:
            lc_messages.append(AIMessage(content=content))
    return lc_messages


def _coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def _metadata_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _reasoning_text_from_message(message: Any) -> str:
    candidates: list[Any] = []

    additional_kwargs = _metadata_dict(getattr(message, "additional_kwargs", None))
    response_metadata = _metadata_dict(getattr(message, "response_metadata", None))
    provider_specific_fields = _metadata_dict(additional_kwargs.get("provider_specific_fields"))

    candidates.extend(
        [
            additional_kwargs.get("reasoning_content"),
            additional_kwargs.get("reasoning"),
            provider_specific_fields.get("reasoning_content"),
            provider_specific_fields.get("reasoning"),
            response_metadata.get("reasoning_content"),
            response_metadata.get("reasoning"),
        ]
    )

    for candidate in candidates:
        text = _coerce_text(candidate)
        if text:
            return text
    return ""


def _message_diagnostics(message: Any) -> dict[str, Any]:
    additional_kwargs = _metadata_dict(getattr(message, "additional_kwargs", None))
    response_metadata = _metadata_dict(getattr(message, "response_metadata", None))
    return {
        "reasoning_len": len(_reasoning_text_from_message(message)),
        "additional_keys": sorted(additional_kwargs.keys()),
        "response_metadata_keys": sorted(response_metadata.keys()),
    }


def _extra_body_key(extra_body: dict[str, Any] | None) -> str:
    if not extra_body:
        return "-"
    return json.dumps(extra_body, sort_keys=True, ensure_ascii=True)


def _get_llm(
    temperature: float,
    max_tokens: int,
    streaming: bool = False,
    *,
    chat_model: str | None = None,
    enable_thinking: bool | None = None,
) -> ChatOpenAI:
    """
    Return a cached ChatOpenAI instance.
    Streaming instances are cached separately.
    """
    resolved_chat_model = Config.resolve_chat_model(chat_model)
    extra_body = Config.chat_extra_body(enable_thinking=enable_thinking)
    key = (
        f"{resolved_chat_model}:"
        f"{temperature}:{max_tokens}:{'s' if streaming else 'b'}:{_extra_body_key(extra_body)}"
    )
    if key not in _llm_cache:
        Config.validate()
        logger.info(
            "[req:%s] [llm] creating client model=%s streaming=%s temp=%s max_tokens=%s base_url=%s timeout=%s extra_body=%s",
            current_request_id(),
            resolved_chat_model,
            streaming,
            temperature,
            max_tokens,
            Config.chat_base_url(),
            Config.LLM_TIMEOUT,
            extra_body,
        )
        _llm_cache[key] = ChatOpenAI(
            model=resolved_chat_model,
            openai_api_key=Config.LLM_API_KEY,
            openai_api_base=Config.chat_base_url(),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=Config.LLM_TIMEOUT,
            streaming=streaming,
            extra_body=extra_body,
        )
    return _llm_cache[key]


def _invoke_once(
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    chat_model: str | None = None,
    enable_thinking: bool | None = None,
    attempt: str = "primary",
) -> tuple[str, dict[str, Any]]:
    resolved_chat_model = Config.resolve_chat_model(chat_model)
    llm = _get_llm(
        temperature,
        max_tokens,
        streaming=False,
        chat_model=resolved_chat_model,
        enable_thinking=enable_thinking,
    )
    started = time.perf_counter()
    logger.info(
        "[req:%s] [llm] invoke start attempt=%s model=%s streaming=false temp=%s max_tokens=%s enable_thinking=%s messages=%d summary=%s",
        current_request_id(),
        attempt,
        resolved_chat_model,
        temperature,
        max_tokens,
        enable_thinking,
        len(messages),
        _message_summary(messages),
    )
    response = llm.invoke(_to_lc_messages(messages))
    content = _coerce_text(response.content)
    diagnostics = _message_diagnostics(response)
    logger.info(
        "[req:%s] [llm] invoke complete attempt=%s len=%d reasoning_len=%d additional_keys=%s response_metadata_keys=%s dur_ms=%.1f",
        current_request_id(),
        attempt,
        len(content),
        diagnostics["reasoning_len"],
        ",".join(diagnostics["additional_keys"]) or "-",
        ",".join(diagnostics["response_metadata_keys"]) or "-",
        (time.perf_counter() - started) * 1000,
    )
    return content, diagnostics


def chat_completion(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    chat_model: str | None = None,
) -> str:
    """
    Blocking call. Returns the full response as a string.
    """
    temp = Config.CHAT_TEMPERATURE if temperature is None else temperature
    tok = Config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens

    try:
        content, diagnostics = _invoke_once(
            messages,
            temperature=temp,
            max_tokens=tok,
            chat_model=chat_model,
            enable_thinking=None,
            attempt="primary",
        )
        if content:
            return content

        if Config.chat_retry_with_thinking_disabled() and Config.chat_enable_thinking() is not False:
            logger.warning(
                "[req:%s] [llm] empty content on primary invoke; retrying with enable_thinking=false reasoning_len=%d",
                current_request_id(),
                diagnostics["reasoning_len"],
            )
            fallback_content, fallback_diagnostics = _invoke_once(
                messages,
                temperature=temp,
                max_tokens=tok,
                chat_model=chat_model,
                enable_thinking=False,
                attempt="fallback_nonthinking",
            )
            if fallback_content:
                logger.warning(
                    "[req:%s] [llm] nonthinking fallback restored content len=%d reasoning_len=%d",
                    current_request_id(),
                    len(fallback_content),
                    fallback_diagnostics["reasoning_len"],
                )
                return fallback_content
            logger.warning(
                "[req:%s] [llm] nonthinking fallback still empty reasoning_len=%d",
                current_request_id(),
                fallback_diagnostics["reasoning_len"],
            )

        return content
    except Exception:
        logger.exception(
            "[req:%s] [llm] invoke failed temp=%s max_tokens=%s messages=%d",
            current_request_id(),
            temp,
            tok,
            len(messages),
        )
        raise


def chat_completion_stream(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    chat_model: str | None = None,
) -> Generator[str, None, None]:
    """
    Streaming call. Yields tokens one at a time as strings.
    """
    temp = Config.CHAT_TEMPERATURE if temperature is None else temperature
    tok = Config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens

    try:
        resolved_chat_model = Config.resolve_chat_model(chat_model)
        llm = _get_llm(temp, tok, streaming=True, chat_model=resolved_chat_model)
        started = time.perf_counter()
        chunk_count = 0
        total_chars = 0
        reasoning_chars = 0
        first_token_ms = None
        logger.info(
            "[req:%s] [llm] stream start model=%s temp=%s max_tokens=%s enable_thinking=%s messages=%d summary=%s",
            current_request_id(),
            resolved_chat_model,
            temp,
            tok,
            Config.chat_enable_thinking(),
            len(messages),
            _message_summary(messages),
        )
        for chunk in llm.stream(_to_lc_messages(messages)):
            content = _coerce_text(getattr(chunk, "content", None))
            if content:
                chunk_count += 1
                total_chars += len(content)
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                    logger.info(
                        "[req:%s] [llm] stream first_token_ms=%.1f",
                        current_request_id(),
                        first_token_ms,
                    )
                yield content
                continue

            reasoning = _reasoning_text_from_message(chunk)
            if reasoning:
                reasoning_chars += len(reasoning)

        logger.info(
            "[req:%s] [llm] stream complete chunks=%d chars=%d reasoning_chars=%d dur_ms=%.1f",
            current_request_id(),
            chunk_count,
            total_chars,
            reasoning_chars,
            (time.perf_counter() - started) * 1000,
        )

        if (
            total_chars == 0
            and Config.chat_retry_with_thinking_disabled()
            and Config.chat_enable_thinking() is not False
        ):
            logger.warning(
                "[req:%s] [llm] empty stream content; retrying once with non-streaming enable_thinking=false",
                current_request_id(),
            )
            fallback_content, fallback_diagnostics = _invoke_once(
                messages,
                temperature=temp,
                max_tokens=tok,
                chat_model=resolved_chat_model,
                enable_thinking=False,
                attempt="stream_fallback_nonthinking",
            )
            if fallback_content:
                logger.warning(
                    "[req:%s] [llm] stream fallback restored content len=%d reasoning_len=%d",
                    current_request_id(),
                    len(fallback_content),
                    fallback_diagnostics["reasoning_len"],
                )
                yield fallback_content
            else:
                logger.warning(
                    "[req:%s] [llm] stream fallback still empty reasoning_len=%d",
                    current_request_id(),
                    fallback_diagnostics["reasoning_len"],
                )
    except Exception:
        logger.exception(
            "[req:%s] [llm] stream failed after chunks=%d chars=%d",
            current_request_id(),
            locals().get("chunk_count", 0),
            locals().get("total_chars", 0),
        )
        raise
