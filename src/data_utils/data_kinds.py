from __future__ import annotations

def normalize_data_kind(kind: str) -> str:
    return kind.strip().lower()


def is_static_data_kind(kind: str) -> bool:
    return normalize_data_kind(kind) in {"static", "line_static"}
