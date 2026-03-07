"""
LangChain-based LLM wrapper.
Supports both blocking (chat_completion) and streaming (chat_completion_stream).
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from config import Config
from typing import Generator

# Module-level singleton (initialized on first call)
_llm_cache: dict[str, ChatOpenAI] = {}


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
    response = llm.invoke(_to_lc_messages(messages))
    return response.content


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

    for chunk in llm.stream(_to_lc_messages(messages)):
        if chunk.content:
            yield chunk.content