from typing import Any

import torch
from torch.utils.data import random_split
from omegaconf import MISSING, OmegaConf

from src.utils.logging_config import setup_logging
logger = setup_logging()


# Sentinel used by `_cfg_get` to distinguish "no default provided" (strict required) from
# any real default value the caller might want to pass (including `None`).
_REQUIRED = object()


def _cfg_has(cfg: Any, key: str) -> bool:
    """Return True if ``cfg`` defines ``key`` and the value is not the OmegaConf MISSING sentinel."""
    return key in cfg and OmegaConf.select(cfg, key, default=MISSING) is not MISSING


def _cfg_get(cfg: Any, key: str, *, default: Any = _REQUIRED, context: str = "data config") -> Any:
    """Strictly fetch ``key`` from ``cfg``.

    - If ``default`` is not provided, a missing key raises ``KeyError`` with ``context``
      in the message — this is the "strict" path used for semantically important keys.
    - If ``default`` is provided, the caller is opting into a safe fallback; the default
      is returned when the key is absent.
    """
    if _cfg_has(cfg, key):
        return cfg[key]
    if default is _REQUIRED:
        raise KeyError(
            f"{context}: required key {key!r} is missing. "
            "Set it explicitly in the config; unclear defaults are not allowed."
        )
    return default


def _resolve_split_seed(cfg, *, default: int = 42) -> int:
    raw_seed = OmegaConf.select(cfg, "data.split_seed", default=None)
    if raw_seed is None:
        raw_seed = OmegaConf.select(cfg, "split_seed", default=default)
    try:
        seed = int(raw_seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"split_seed must be an integer >= 0, got {raw_seed!r}"
        ) from exc
    if seed < 0:
        raise ValueError(f"split_seed must be >= 0, got {seed}")
    return seed


def _seeded_random_split(dataset, lengths: list[int], *, seed: int):
    """Thin wrapper over ``torch.utils.data.random_split`` that seeds a fresh generator.

    ``random_split`` itself validates that ``sum(lengths) == len(dataset)`` and raises
    on length/type mismatch, so we don't re-validate here.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return random_split(dataset, [int(v) for v in lengths], generator=generator)


def _resolve_temporal_window_start_frames(
    *,
    frame_count: int,
    sequence_length: int,
    frame_stride: int,
    frame_start: int,
    frame_stop: int | None,
    window_stride: int,
) -> list[int]:
    """Compute anchor (start) frames for each temporal sampling window.

    Scalar invariants (sequence_length/frame_stride/window_stride > 0, frame_start >= 0)
    are validated at the dataset boundary in ``TemporalLAMMPSDumpDataset.__init__``; this
    helper only validates the slice-window invariants unique to it (``frame_count`` and
    the ``frame_stop`` vs. ``frame_start``/``frame_count`` relationships).
    """
    frame_count = int(frame_count)
    sequence_length = int(sequence_length)
    frame_stride = int(frame_stride)
    frame_start = int(frame_start)
    window_stride = int(window_stride)
    stop = frame_count if frame_stop is None else int(frame_stop)

    if frame_count <= 0:
        raise ValueError(f"frame_count must be > 0, got {frame_count}")
    if frame_start >= frame_count:
        raise ValueError(
            f"frame_start must be < frame_count, got frame_start={frame_start}, "
            f"frame_count={frame_count}."
        )
    if stop <= frame_start:
        raise ValueError(
            f"frame_stop must be > frame_start, got frame_start={frame_start}, frame_stop={stop}."
        )
    if stop > frame_count:
        raise ValueError(
            f"frame_stop must be <= frame_count, got frame_stop={stop}, frame_count={frame_count}."
        )

    last_required_frame = frame_start + (sequence_length - 1) * frame_stride
    if last_required_frame >= stop:
        return []
    max_start = stop - (sequence_length - 1) * frame_stride
    return list(range(frame_start, max_start, window_stride))


def _split_temporal_window_start_frames(
    anchor_frames: list[int],
    *,
    train_ratio: float,
    seed: int,
    context: str,
) -> tuple[list[int], list[int]]:
    if not anchor_frames:
        raise ValueError(f"{context}: anchor_frames must be non-empty")

    train_ratio = float(train_ratio)
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"{context}: train_ratio must be in (0, 1), got {train_ratio}")

    num_frames = len(anchor_frames)
    train_size = int(train_ratio * num_frames)
    val_size = num_frames - train_size
    if train_size <= 0 or val_size <= 0:
        raise ValueError(
            f"{context}: temporal split produced an empty subset. "
            f"num_windows={num_frames}, train_ratio={train_ratio}, "
            f"train_size={train_size}, val_size={val_size}."
        )

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    perm = torch.randperm(num_frames, generator=generator).tolist()
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]
    train_frames = sorted(int(anchor_frames[idx]) for idx in train_idx)
    val_frames = sorted(int(anchor_frames[idx]) for idx in val_idx)
    return train_frames, val_frames


# Module-level helper
def _to_container(cfg):
    if cfg is None:
        return None
    return OmegaConf.to_container(cfg, resolve=True)
