import re

from flask import g, has_request_context


def current_request_id() -> str:
    if has_request_context():
        return getattr(g, "request_id", "-")
    return "-"


def preview_text(value: str | None, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "..."
