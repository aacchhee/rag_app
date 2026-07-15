import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from log_utils import current_request_id

DUMP_DIR = os.getenv("DEBUG_AUDIT_DIR", "debug_dumps")
_enabled = os.getenv("DEBUG_AUDIT_ENABLED", "true").lower() in ("1", "true", "yes", "on")


def _ensure_dir() -> str:
    if not os.path.isdir(DUMP_DIR):
        os.makedirs(DUMP_DIR, exist_ok=True)
    return DUMP_DIR


def _reasoning_text(msg: Any) -> str:
    """Mirror of _reasoning_text_from_message without causing a circular import."""
    ak = getattr(msg, "additional_kwargs", None)
    rm = getattr(msg, "response_metadata", None)
    candidates = []
    if isinstance(ak, dict):
        candidates.extend([ak.get("reasoning_content"), ak.get("reasoning")])
        psf = ak.get("provider_specific_fields")
        if isinstance(psf, dict):
            candidates.extend([psf.get("reasoning_content"), psf.get("reasoning")])
    if isinstance(rm, dict):
        candidates.extend([rm.get("reasoning_content"), rm.get("reasoning")])
    for c in candidates:
        if c is None:
            continue
        if isinstance(c, str) and c.strip():
            return c
        if isinstance(c, list):
            parts = []
            for item in c:
                if isinstance(item, str) and item:
                    parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("text")
                    if isinstance(t, str) and t:
                        parts.append(t)
            joined = "".join(parts)
            if joined:
                return joined
    return ""


def _serialize_chunk(chunk: Any) -> dict[str, Any]:
    """Serialize a LangChain message chunk to a JSONable dict."""
    result: dict[str, Any] = {"_type": type(chunk).__name__}

    # content
    content = getattr(chunk, "content", None)
    result["content"] = content

    # additional_kwargs
    ak = getattr(chunk, "additional_kwargs", {})
    if isinstance(ak, dict):
        result["additional_kwargs"] = dict(ak)
    else:
        result["additional_kwargs"] = {"_repr": str(ak)}

    # response_metadata on chunk
    rm = getattr(chunk, "response_metadata", {})
    if isinstance(rm, dict):
        result["response_metadata"] = dict(rm)
    else:
        result["response_metadata"] = {"_repr": str(rm)}

    reasoning = _reasoning_text(chunk)
    if reasoning:
        result["_reasoning_text"] = reasoning

    # shape/type hints
    for attr in ("role", "name", "id", "type"):
        val = getattr(chunk, attr, None)
        if val is not None:
            result[attr] = str(val)

    return result


def _serialize_message(msg: Any) -> dict[str, Any]:
    """Serialize a full LangChain message (non-streaming response) to a dict."""
    result: dict[str, Any] = {"_type": type(msg).__name__}

    content = getattr(msg, "content", None)
    result["content"] = content

    ak = getattr(msg, "additional_kwargs", {})
    if isinstance(ak, dict):
        result["additional_kwargs"] = dict(ak)
    else:
        result["additional_kwargs"] = {"_repr": str(ak)}

    rm = getattr(msg, "response_metadata", {})
    if isinstance(rm, dict):
        result["response_metadata"] = dict(rm)
    else:
        result["response_metadata"] = {"_repr": str(rm)}

    reasoning = _reasoning_text(msg)
    if reasoning:
        result["_reasoning_text"] = reasoning

    for attr in ("role", "name", "id", "type"):
        val = getattr(msg, attr, None)
        if val is not None:
            result[attr] = str(val)

    return result


def dump_audit(
    *,
    endpoint: str,
    chat_model: str,
    messages: list[dict],
    chunks: list[Any] | None = None,
    response: Any | None = None,
    notes: dict[str, Any] | None = None,
) -> str | None:
    """
    Write an audit JSON file with the full raw response structure.
    Returns the path written, or None if disabled.
    """
    if not _enabled:
        return None

    _ensure_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    safe_model = chat_model.replace("/", "_").replace("\\", "_")
    filename = f"{ts}_{safe_model}_{endpoint}.json"
    filepath = os.path.join(DUMP_DIR, filename)

    audit_data: dict[str, Any] = {
        "timestamp_utc": ts,
        "request_id": current_request_id(),
        "endpoint": endpoint,
        "chat_model": chat_model,
        "messages_summary": [
            {"role": m.get("role"), "len": len(m.get("content", ""))} for m in messages
        ],
        "messages": messages,
        "notes": notes or {},
    }

    if chunks is not None:
        audit_data["chunks"] = [_serialize_chunk(c) for c in chunks]
        audit_data["chunk_count"] = len(chunks)
        total_content = sum(
            len(c.get("content") or "") for c in audit_data["chunks"]
        )
        total_reasoning = sum(
            len(c.get("_reasoning_text") or "") for c in audit_data["chunks"]
        )
        audit_data["total_content_chars"] = total_content
        audit_data["total_reasoning_chars"] = total_reasoning

    if response is not None:
        audit_data["response"] = _serialize_message(response)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(audit_data, f, indent=2, ensure_ascii=False, default=str)

    return filepath
