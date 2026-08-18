"""
LLM wrapper using raw OpenAI client for both blocking and streaming calls.
Replaces LangChain streaming to get direct access to delta.reasoning_content
from reasoning models (e.g. Kimi K2.6).
"""

import json
import logging
import time
from typing import Any, Generator

import openai
from config import Config
from log_utils import current_request_id

# Module-level singleton
_raw_client: openai.OpenAI | None = None
logger = logging.getLogger("gunicorn.error")


def _get_raw_client() -> openai.OpenAI:
    """Return (and cache) a raw OpenAI client."""
    global _raw_client
    if _raw_client is None:
        Config.validate()
        _raw_client = openai.OpenAI(
            api_key=Config.LLM_API_KEY,
            base_url=Config.chat_base_url(),
            timeout=Config.LLM_TIMEOUT,
        )
        logger.info(
            "[req:%s] [llm] created raw OpenAI client base_url=%s timeout=%s",
            current_request_id(),
            Config.chat_base_url(),
            Config.LLM_TIMEOUT,
        )
    return _raw_client


def _message_summary(messages: list[dict]) -> str:
    parts = []
    for msg in messages:
        content = msg.get("content") or ""
        parts.append(f"{msg.get('role', '?')}:{len(content)}")
    return ",".join(parts)


def _extract_delta(delta: Any) -> str:
    """Extract content from a streaming delta."""
    return delta.content or ""


def _extract_message(msg: Any) -> str:
    """Extract content from a non-streaming message."""
    return msg.content or ""


def _invoke_once(
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    chat_model: str | None = None,
    attempt: str = "primary",
) -> str:
    resolved_chat_model = Config.resolve_chat_model(chat_model)
    extra_body = Config.chat_extra_body()
    client = _get_raw_client()

    started = time.perf_counter()
    logger.info(
        "[req:%s] [llm] invoke start attempt=%s model=%s streaming=false temp=%s max_tokens=%s messages=%d summary=%s",
        current_request_id(),
        attempt,
        resolved_chat_model,
        temperature,
        max_tokens,
        len(messages),
        _message_summary(messages),
    )

    response = client.chat.completions.create(
        model=resolved_chat_model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )

    choice = response.choices[0]
    content = _extract_message(choice.message)

    if not content:
        try:
            from debug_audit import dump_audit
            dump_audit(
                endpoint=f"invoke_{attempt}",
                chat_model=resolved_chat_model,
                messages=messages,
                notes={"content_len": 0, "finish_reason": choice.finish_reason},
            )
        except Exception:
            pass

    logger.info(
        "[req:%s] [llm] invoke complete attempt=%s len=%d finish_reason=%s dur_ms=%.1f",
        current_request_id(),
        attempt,
        len(content),
        choice.finish_reason,
        (time.perf_counter() - started) * 1000,
    )
    return content


def chat_completion(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    chat_model: str | None = None,
) -> str:
    """Blocking call. Returns the full response as a string."""
    temp = Config.CHAT_TEMPERATURE if temperature is None else temperature
    tok = Config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens

    try:
        content = _invoke_once(messages, temperature=temp, max_tokens=tok, chat_model=chat_model, attempt="primary")
        if content:
            return content

        # Retry once with a different model if empty
        fallback_content = _invoke_once(messages, temperature=temp, max_tokens=tok, chat_model=chat_model, attempt="fallback")
        if fallback_content:
            logger.warning("[req:%s] [llm] fallback restored content len=%d", current_request_id(), len(fallback_content))
            return fallback_content
        logger.warning("[req:%s] [llm] fallback still empty", current_request_id())
        return content
    except Exception:
        logger.exception("[req:%s] [llm] invoke failed temp=%s max_tokens=%s messages=%d", current_request_id(), temp, tok, len(messages))
        raise


def chat_completion_stream(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    chat_model: str | None = None,
) -> Generator[str, None, None]:
    """
    Streaming call using raw OpenAI client.
    Yields content tokens.  When a model only sends reasoning tokens
    (common with reasoning models), the reasoning is buffered and
    yielded at the end so the caller gets *something* instead of silence.
    """
    temp = Config.CHAT_TEMPERATURE if temperature is None else temperature
    tok = Config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens

    try:
        resolved_chat_model = Config.resolve_chat_model(chat_model)
        extra_body = Config.chat_extra_body()
        client = _get_raw_client()
        started = time.perf_counter()

        content_chunks = 0
        total_content_chars = 0
        first_token_ms = None

        logger.info(
            "[req:%s] [llm] stream start model=%s temp=%s max_tokens=%s messages=%d summary=%s",
            current_request_id(),
            resolved_chat_model,
            temp,
            tok,
            len(messages),
            _message_summary(messages),
        )

        response = client.chat.completions.create(
            model=resolved_chat_model,
            messages=messages,
            temperature=temp,
            max_tokens=tok,
            stream=True,
            extra_body=extra_body,
        )

        for chunk in response:
            content = _extract_delta(chunk.choices[0].delta)
            if not content:
                continue
            content_chunks += 1
            total_content_chars += len(content)
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
                logger.info("[req:%s] [llm] stream first_token_ms=%.1f", current_request_id(), first_token_ms)
            yield content

        logger.info(
            "[req:%s] [llm] stream complete chunks=%d chars=%d dur_ms=%.1f",
            current_request_id(),
            content_chunks,
            total_content_chars,
            (time.perf_counter() - started) * 1000,
        )

        # Debug audit for empty stream
        if total_content_chars == 0:
            try:
                from debug_audit import dump_audit
                dump_audit(
                    endpoint="stream",
                    chat_model=resolved_chat_model,
                    messages=messages,
                    notes={"chunk_count_yielded": 0, "total_content_chars": 0},
                )
            except Exception:
                pass

            # Retry once with non-streaming invoke
            logger.warning("[req:%s] [llm] empty stream content; retrying once with non-streaming", current_request_id())
            fallback_content = _invoke_once(messages, temperature=temp, max_tokens=tok, chat_model=resolved_chat_model, attempt="stream_fallback")
            if fallback_content:
                logger.warning("[req:%s] [llm] stream fallback restored content len=%d", current_request_id(), len(fallback_content))
                yield fallback_content
            else:
                logger.warning("[req:%s] [llm] stream fallback still empty", current_request_id())
    except Exception:
        logger.exception(
            "[req:%s] [llm] stream failed after content_chunks=%d chars=%d",
            current_request_id(),
            locals().get("content_chunks", 0),
            locals().get("total_content_chars", 0),
        )
        raise
