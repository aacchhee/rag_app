"""
LLM wrapper using raw OpenAI client for both blocking and streaming calls.
Replaces LangChain streaming to get direct access to delta.reasoning_content
from reasoning models (e.g. Kimi K2.6).
"""

import json
import logging
import re
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


# ---------------------------------------------------------------------------
# Tag stripping
# ---------------------------------------------------------------------------
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_THINK_TAG_RE2 = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> and <thinking>...</thinking> blocks."""
    text = _THINK_TAG_RE.sub("", text)
    text = _THINK_TAG_RE2.sub("", text)
    return text


def _extract_delta(delta: Any) -> tuple[str, str]:
    """
    Extract content and reasoning_content from a streaming delta.
    Falls back through several strategies for provider-specific fields.
    """
    content = delta.content or ""
    reasoning = getattr(delta, "reasoning_content", None) or ""

    if not reasoning and hasattr(delta, "model_extra"):
        extras = delta.model_extra or {}
        reasoning = extras.get("reasoning_content", "") or extras.get("reasoning", "")

    if not reasoning:
        try:
            d = delta.model_dump()
            reasoning = d.get("reasoning_content", "") or d.get("reasoning", "")
        except Exception:
            pass

    logger.info(
        "[req:%s] [llm] _extract_delta content=%r reasoning=%r raw_delta=%r",
        current_request_id(),
        content,
        reasoning,
        delta,
    )
    return content or "", reasoning or ""


def _extract_message(msg: Any) -> tuple[str, str]:
    """
    Extract content and reasoning_content from a non-streaming message.
    """
    content = msg.content or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""

    if not reasoning and hasattr(msg, "model_extra"):
        extras = msg.model_extra or {}
        reasoning = extras.get("reasoning_content", "") or extras.get("reasoning", "")

    if not reasoning and hasattr(msg, "provider_specific_fields"):
        psf = msg.provider_specific_fields or {}
        reasoning = psf.get("reasoning_content") or psf.get("reasoning") or ""

    if not reasoning:
        try:
            d = msg.model_dump()
            reasoning = d.get("reasoning_content", "") or d.get("reasoning", "")
        except Exception:
            pass

    logger.info(
        "[req:%s] [llm] _extract_message content=%r reasoning=%r raw_msg=%r",
        current_request_id(),
        content,
        reasoning,
        msg,
    )
    return content or "", reasoning or ""


# ---------------------------------------------------------------------------
# Invoke (blocking)
# ---------------------------------------------------------------------------
def _invoke_once(
    messages: list[dict],
    *,
    temperature: float,
    max_tokens: int,
    chat_model: str | None = None,
    enable_thinking: bool = False,
    attempt: str = "primary",
) -> tuple[str, dict[str, Any]]:
    resolved_chat_model = Config.resolve_chat_model(chat_model)
    extra_body = Config.chat_extra_body(enable_thinking=enable_thinking)
    client = _get_raw_client()

    started = time.perf_counter()
    logger.info(
        "[req:%s] [llm] invoke start attempt=%s model=%s streaming=false temp=%s max_tokens=%s enable_thinking=true messages=%d summary=%s",
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
    msg = choice.message
    content, reasoning = _extract_message(msg)

    diagnostics = {
        "reasoning_len": len(reasoning),
        "finish_reason": choice.finish_reason,
    }

    # Debug audit whenever content looks empty or reasoning looks present
    if not content or reasoning:
        try:
            from debug_audit import dump_audit

            dump_audit(
                endpoint=f"invoke_{attempt}",
                chat_model=resolved_chat_model,
                messages=messages,
                notes={
                    "content_len": len(content),
                    "reasoning_len": len(reasoning),
                    "finish_reason": choice.finish_reason,
                },
            )
        except Exception:
            pass

    # Strip think tags from content before returning
    if content:
        content = _strip_think_tags(content)

    logger.info(
        "[req:%s] [llm] invoke complete attempt=%s len=%d reasoning_len=%d finish_reason=%s dur_ms=%.1f",
        current_request_id(),
        attempt,
        len(content),
        len(reasoning),
        choice.finish_reason,
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
            attempt="primary",
        )
        if content:
            return content

        # Only retry if we truly got nothing back
        if not content:
            logger.warning(
                "[req:%s] [llm] empty content on primary invoke; retrying reasoning_len=%d",
                current_request_id(),
                diagnostics["reasoning_len"],
            )
            fallback_content, fallback_diagnostics = _invoke_once(
                messages,
                temperature=temp,
                max_tokens=tok,
                chat_model=chat_model,
                attempt="fallback",
            )
            if fallback_content:
                logger.warning(
                    "[req:%s] [llm] fallback restored content len=%d reasoning_len=%d",
                    current_request_id(),
                    len(fallback_content),
                    fallback_diagnostics["reasoning_len"],
                )
                return fallback_content
            logger.warning(
                "[req:%s] [llm] fallback still empty reasoning_len=%d",
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


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------
def chat_completion_stream(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
    chat_model: str | None = None,
) -> Generator[str, None, None]:
    """
    Streaming call using raw OpenAI client.
    Yields content tokens only; reasoning tokens are extracted internally
    but never yielded to the caller.  <think> tags are stripped from content.
    """
    temp = Config.CHAT_TEMPERATURE if temperature is None else temperature
    tok = Config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens

    try:
        resolved_chat_model = Config.resolve_chat_model(chat_model)
        extra_body = Config.chat_extra_body(enable_thinking=Config.chat_enable_thinking())
        client = _get_raw_client()
        started = time.perf_counter()

        content_chunks = 0
        total_content_chars = 0
        reasoning_buffer: list[str] = []
        total_reasoning_chars = 0
        first_token_ms = None

        logger.info(
            "[req:%s] [llm] stream start model=%s temp=%s max_tokens=%s enable_thinking=true messages=%d summary=%s",
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
            delta = chunk.choices[0].delta
            content, reasoning = _extract_delta(delta)

            if content:
                content = _strip_think_tags(content)
                content_chunks += 1
                total_content_chars += len(content)
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                    logger.info(
                        "[req:%s] [llm] stream first_token_ms=%.1f",
                        current_request_id(),
                        first_token_ms,
                    )
                yield content
                continue

            if reasoning:
                reasoning_buffer.append(reasoning)
                total_reasoning_chars += len(reasoning)
                continue

        logger.info(
            "[req:%s] [llm] stream complete chunks=%d chars=%d reasoning_chars=%d dur_ms=%.1f",
            current_request_id(),
            content_chunks,
            total_content_chars,
            total_reasoning_chars,
            (time.perf_counter() - started) * 1000,
        )

        # Debug audit for empty or reasoning-heavy streams
        if total_content_chars == 0 or total_reasoning_chars > 0:
            try:
                from debug_audit import dump_audit

                dump_audit(
                    endpoint="stream",
                    chat_model=resolved_chat_model,
                    messages=messages,
                    notes={
                        "chunk_count_yielded": content_chunks,
                        "total_content_chars": total_content_chars,
                        "total_reasoning_chars": total_reasoning_chars,
                        "enable_thinking": True,
                    },
                )
            except Exception:
                pass

        # Fallbacks when stream produced no visible content
        if total_content_chars == 0:
            logger.warning(
                "[req:%s] [llm] stream empty content; retrying once with non-streaming",
                current_request_id(),
            )
            fallback_content, fallback_diagnostics = _invoke_once(
                messages,
                temperature=temp,
                max_tokens=tok * 2,
                chat_model=resolved_chat_model,
                enable_thinking=False,
                attempt="stream_fallback",
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
            "[req:%s] [llm] stream failed after content_chunks=%d chars=%d",
            current_request_id(),
            locals().get("content_chunks", 0),
            locals().get("total_content_chars", 0),
        )
        raise
