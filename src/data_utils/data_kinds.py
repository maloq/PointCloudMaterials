from __future__ import annotations

from typing import Any


def normalize_data_kind(kind: Any, *, default: str | None = None) -> str:
    if kind is None:
        if default is None:
            return ""
        kind = default
    normalized = str(kind).strip().lower()
    if normalized == "real":
        return "static"
    return normalized


def is_static_data_kind(kind: Any) -> bool:
    return normalize_data_kind(kind) == "static"
