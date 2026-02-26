from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from config import Config

# Module-level singleton (initialized on first call)
_llm_cache: dict[str, ChatOpenAI] = {}


def _get_llm(temperature: float, max_tokens: int) -> ChatOpenAI:
    """
    Return a cached ChatOpenAI instance for the given (temperature, max_tokens).
    Re-uses instances to avoid re-creating HTTP sessions.
    """
    key = f"{temperature}:{max_tokens}"
    if key not in _llm_cache:
        Config.validate()
        _llm_cache[key] = ChatOpenAI(
            model=Config.CHAT_MODEL,
            openai_api_key=Config.LLM_API_KEY,
            openai_api_base=Config.chat_base_url(),
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=Config.LLM_TIMEOUT,
        )
    return _llm_cache[key]


def chat_completion(
    messages: list[dict],
    *,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """
    Same signature as the old function so app.py doesn't need to change its calls.

    Accepts:
      messages: [{"role": "system"|"user"|"assistant", "content": "..."}]

    Returns:
      str: the assistant's reply content
    """
    temp = Config.CHAT_TEMPERATURE if temperature is None else temperature
    tok = Config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens

    llm = _get_llm(temp, tok)

    # Convert dict messages to LangChain message objects
    lc_messages = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        else:
            # assistant messages (rare in your flow, but handle it)
            from langchain_core.messages import AIMessage
            lc_messages.append(AIMessage(content=content))

    response = llm.invoke(lc_messages)
    return response.content