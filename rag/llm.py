"""
LangChain-based LLM wrapper.
Supports both blocking (chat_completion) and streaming (chat_completion_stream).
"""

import logging
import time

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import Config
from typing import Generator
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


def _get_llm(temperature: float, max_tokens: int, streaming: bool = False) -> ChatOpenAI:
    """
    Return a cached ChatOpenAI instance.
    Streaming instances are cached separately.
    """
    key = f"{temperature}:{max_tokens}:{'s' if streaming else 'b'}"
    if key not in _llm_cache:
        Config.validate()
        logger.info(
            "[req:%s] [llm] creating client model=%s streaming=%s temp=%s max_tokens=%s base_url=%s timeout=%s",
            current_request_id(),
            Config.CHAT_MODEL,
            streaming,
            temperature,
            max_tokens,
            Config.chat_base_url(),
            Config.LLM_TIMEOUT,
        )
        _llm_cache[key] = ChatOpenAI(
            model=Config.CHAT_MODEL,
            openai_api_key=Config.LLM_API_KEY,
            openai_api_base=Config.chat_base_url(),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=Config.LLM_TIMEOUT,
            streaming=streaming,
        )
    return _llm_cache[key]


def chat_completion(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Blocking call. Returns the full response as a string.
    """
    temp = Config.CHAT_TEMPERATURE if temperature is None else temperature
    tok = Config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens

    llm = _get_llm(temp, tok, streaming=False)
    started = time.perf_counter()
    logger.info(
        "[req:%s] [llm] invoke start streaming=false temp=%s max_tokens=%s messages=%d summary=%s",
        current_request_id(),
        temp,
        tok,
        len(messages),
        _message_summary(messages),
    )
    try:
        response = llm.invoke(_to_lc_messages(messages))
        content = response.content
        logger.info(
            "[req:%s] [llm] invoke complete len=%d dur_ms=%.1f",
            current_request_id(),
            len(content or ""),
            (time.perf_counter() - started) * 1000,
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
) -> Generator[str, None, None]:
    """
    Streaming call. Yields tokens one at a time as strings.
    """
    temp = Config.CHAT_TEMPERATURE if temperature is None else temperature
    tok = Config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens

    llm = _get_llm(temp, tok, streaming=True)
    started = time.perf_counter()
    chunk_count = 0
    total_chars = 0
    first_token_ms = None
    logger.info(
        "[req:%s] [llm] stream start temp=%s max_tokens=%s messages=%d summary=%s",
        current_request_id(),
        temp,
        tok,
        len(messages),
        _message_summary(messages),
    )
    try:
        for chunk in llm.stream(_to_lc_messages(messages)):
            if chunk.content:
                chunk_count += 1
                total_chars += len(chunk.content)
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                    logger.info(
                        "[req:%s] [llm] stream first_token_ms=%.1f",
                        current_request_id(),
                        first_token_ms,
                    )
                yield chunk.content
        logger.info(
            "[req:%s] [llm] stream complete chunks=%d chars=%d dur_ms=%.1f",
            current_request_id(),
            chunk_count,
            total_chars,
            (time.perf_counter() - started) * 1000,
        )
    except Exception:
        logger.exception(
            "[req:%s] [llm] stream failed after chunks=%d chars=%d",
            current_request_id(),
            chunk_count,
            total_chars,
        )
        raise
