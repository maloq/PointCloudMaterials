"""Helpers for retrying failed runs after NaN loss failures."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Sequence, Tuple

from omegaconf import OmegaConf

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Keep patterns focused on loss NaNs to avoid false positives from unrelated
# NaNs (e.g., downstream clustering diagnostics).
NAN_LOSS_PATTERNS = (
    re.compile(r"non-finite losses", re.IGNORECASE),
    re.compile(r"\bloss_nonfinite\b", re.IGNORECASE),
    re.compile(r"\bloss\b[^\n]{0,120}\bnan\b", re.IGNORECASE),
    re.compile(r"\bnan\b[^\n]{0,120}\bloss\b", re.IGNORECASE),
)


def validate_nan_restart_settings(max_retries: int, lr_factor: float) -> None:
    """Validate NaN-restart configuration."""
    if max_retries < 0:
        raise ValueError(
            f"nan_restart_max_retries must be >= 0, got {max_retries}."
        )
    if not math.isfinite(lr_factor):
        raise ValueError(
            f"nan_restart_lr_factor must be finite, got {lr_factor!r}."
        )
    if not (0.0 < lr_factor < 1.0):
        raise ValueError(
            f"nan_restart_lr_factor must be in (0, 1), got {lr_factor}."
        )


def detect_nan_loss_in_logs(log_paths: Sequence[Path]) -> Optional[Tuple[Path, str]]:
    """Return (log_path, pattern) if logs indicate NaN/non-finite loss."""
    for log_path in log_paths:
        if not log_path.exists():
            continue
        text = ANSI_RE.sub("", log_path.read_text(errors="replace"))
        for pattern in NAN_LOSS_PATTERNS:
            if pattern.search(text):
                return log_path, pattern.pattern
    return None


def resolve_learning_rate(
    *,
    repo_root: Path,
    config_name: str | None,
    overrides: Sequence[str],
    context: str,
) -> float:
    """Resolve effective learning rate from overrides or config."""
    from_overrides = _parse_lr_from_overrides(overrides, context=context)
    if from_overrides is not None:
        return from_overrides

    if config_name is None:
        raise ValueError(
            f"{context}: cannot infer learning rate because config_name is missing."
        )

    config_path = _resolve_config_path(repo_root, config_name)
    cfg = OmegaConf.load(config_path)
    if not isinstance(cfg, dict) and not OmegaConf.is_config(cfg):
        raise ValueError(
            f"{context}: config {config_path} is not a mapping config; "
            f"got {type(cfg).__name__}."
        )

    for key in ("learning_rate", "lr"):
        raw_value = cfg.get(key)
        if raw_value is None:
            continue
        try:
            parsed = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context}: config {config_path} has non-numeric {key}={raw_value!r}."
            ) from exc
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(
                f"{context}: config {config_path} has invalid {key}={parsed}. "
                "Expected a positive finite value."
            )
        return parsed

    raise ValueError(
        f"{context}: neither overrides nor config {config_path} define "
        "'learning_rate' (or 'lr')."
    )


def build_scaled_lr_override(current_lr: float, lr_factor: float) -> Tuple[str, float]:
    """Build a Hydra override that scales LR down by ``lr_factor``."""
    if not math.isfinite(current_lr) or current_lr <= 0.0:
        raise ValueError(
            f"Cannot scale invalid current_lr={current_lr!r}; expected positive finite."
        )
    if not math.isfinite(lr_factor) or not (0.0 < lr_factor < 1.0):
        raise ValueError(
            f"Cannot scale learning rate with lr_factor={lr_factor!r}; "
            "expected finite value in (0, 1)."
        )
    new_lr = current_lr * lr_factor
    if not math.isfinite(new_lr) or new_lr <= 0.0:
        raise ValueError(
            f"Scaling produced invalid learning rate {new_lr!r} "
            f"(current_lr={current_lr}, lr_factor={lr_factor})."
        )
    return f"learning_rate={new_lr:.12g}", new_lr


def _parse_lr_from_overrides(
    overrides: Sequence[str],
    *,
    context: str,
) -> Optional[float]:
    lr_value: Optional[float] = None
    for override in overrides:
        if "=" not in override:
            continue
        raw_key, raw_value = override.split("=", 1)
        key = raw_key.strip().lstrip("+")
        if key not in {"learning_rate", "lr"}:
            continue

        value_text = raw_value.strip().strip("'").strip('"')
        try:
            parsed = float(value_text)
        except ValueError as exc:
            raise ValueError(
                f"{context}: override {override!r} sets {key} to non-numeric value "
                f"{value_text!r}."
            ) from exc
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(
                f"{context}: override {override!r} sets {key} to invalid value "
                f"{parsed}. Expected positive finite."
            )
        lr_value = parsed
    return lr_value


def _resolve_config_path(repo_root: Path, config_name: str) -> Path:
    raw = config_name.strip()
    if raw == "":
        raise ValueError("config_name cannot be empty when resolving learning rate.")

    name_path = Path(raw)
    candidate_paths = []
    search_roots = [repo_root / "configs", repo_root]

    if name_path.suffix:
        for root in search_roots:
            candidate_paths.append((root / name_path).resolve())
    else:
        for root in search_roots:
            base = root / name_path
            candidate_paths.append(base.with_suffix(".yaml").resolve())
            candidate_paths.append(base.with_suffix(".yml").resolve())

    seen = set()
    unique_candidates = []
    for candidate in candidate_paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    formatted = "\n".join(f"  - {p}" for p in unique_candidates)
    raise FileNotFoundError(
        f"Could not locate config {config_name!r}. Checked:\n{formatted}"
    )
