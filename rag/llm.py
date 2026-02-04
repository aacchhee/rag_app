import requests
from config import Config


def chat_completion(messages, *, temperature=None, max_tokens=None) -> str:
    payload = {
        "model": Config.CHAT_MODEL,
        "messages": messages,
        "temperature": Config.CHAT_TEMPERATURE if temperature is None else temperature,
        "max_tokens": Config.CHAT_MAX_TOKENS if max_tokens is None else max_tokens,
    }

    r = requests.post(
        Config.CHAT_URL,
        headers={"Authorization": f"Bearer {Config.LLM_API_KEY}", "Content-Type": "application/json"},
        json=payload,
        timeout=Config.LLM_TIMEOUT,
    )
    if r.status_code >= 400:
        raise RuntimeError(f"Chat request failed {r.status_code}: {r.text[:2000]}")
    j = r.json()
    return j["choices"][0]["message"]["content"]
