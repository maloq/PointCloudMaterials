from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data_utils.temporal_lammps_dataset import TemporalLAMMPSDumpDataset


@dataclass(frozen=True)
class TrajectorySpec:
    path: Path
    run_id: str
    cache_dir: Path | None


@dataclass(frozen=True)
class TemporalAnchorSplit:
    train: np.ndarray
    validation: np.ndarray
    boundary_frame: int


def resolve_lag_frame_offset(
    timesteps: np.ndarray,
    *,
    lag_frames: int | None = None,
    lag_timesteps: int | None = None,
) -> int:
    """Resolve a frame offset, requiring exact physical-time alignment."""
    if (lag_frames is None) == (lag_timesteps is None):
        raise ValueError("Set exactly one of lag_frames or lag_timesteps.")
    times = np.asarray(timesteps, dtype=np.int64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError(
            f"Temporal lag resolution requires at least two 1D timesteps, got {times.shape}."
        )
    if lag_frames is not None:
        offset = int(lag_frames)
        if offset <= 0:
            raise ValueError(f"lag_frames must be > 0, got {offset}.")
        if offset >= times.size:
            raise ValueError(
                f"lag_frames={offset} leaves no temporal pairs for {times.size} frames."
            )
        return offset

    requested = int(lag_timesteps)
    if requested <= 0:
        raise ValueError(f"lag_timesteps must be > 0, got {requested}.")
    deltas = np.diff(times)
    unique_deltas = np.unique(deltas)
    if unique_deltas.size != 1 or int(unique_deltas[0]) <= 0:
        raise ValueError(
            "lag_timesteps currently requires a uniformly sampled trajectory. "
            f"Observed timestep deltas={unique_deltas.tolist()}. Use lag_frames instead."
        )
    frame_delta = int(unique_deltas[0])
    if requested % frame_delta != 0:
        raise ValueError(
            f"lag_timesteps={requested} is not an exact multiple of the frame timestep "
            f"delta={frame_delta}."
        )
    offset = requested // frame_delta
    if offset >= times.size:
        raise ValueError(
            f"lag_timesteps={requested} resolves to {offset} frames and leaves no pairs."
        )
    return int(offset)


def contiguous_temporal_split(
    *,
    frame_count: int,
    lag_frames: int,
    train_ratio: float,
    frame_start: int = 0,
    frame_stop: int | None = None,
    window_stride: int = 1,
    boundary_gap_frames: int = 0,
) -> TemporalAnchorSplit:
    """Split pair anchors into non-overlapping contiguous time blocks.

    Training pairs end strictly before the boundary (and optional gap), while
    validation pairs start at or after it. Therefore no pair straddles the split.
    """
    stop = int(frame_count) if frame_stop is None else int(frame_stop)
    start = int(frame_start)
    lag = int(lag_frames)
    stride = int(window_stride)
    gap = int(boundary_gap_frames)
    ratio = float(train_ratio)
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0, 1), got {ratio}.")
    if start < 0 or stop > int(frame_count) or start >= stop:
        raise ValueError(
            f"Invalid frame interval [{start}, {stop}) for frame_count={frame_count}."
        )
    if lag <= 0 or stride <= 0 or gap < 0:
        raise ValueError(
            f"Expected lag_frames>0, window_stride>0, boundary_gap_frames>=0; "
            f"got lag={lag}, stride={stride}, gap={gap}."
        )

    boundary = start + int(np.floor((stop - start) * ratio))
    train_stop = boundary - gap - lag
    validation_start = boundary + gap
    last_anchor_exclusive = stop - lag
    train = np.arange(start, train_stop, stride, dtype=np.int64)
    validation = np.arange(validation_start, last_anchor_exclusive, stride, dtype=np.int64)
    if train.size == 0 or validation.size == 0:
        raise ValueError(
            "Contiguous temporal split produced an empty partition. "
            f"frames=[{start}, {stop}), lag={lag}, boundary={boundary}, gap={gap}, "
            f"train_pairs={train.size}, validation_pairs={validation.size}."
        )
    return TemporalAnchorSplit(train=train, validation=validation, boundary_frame=boundary)


def all_pair_anchors(
    *,
    frame_count: int,
    lag_frames: int,
    frame_start: int = 0,
    frame_stop: int | None = None,
    window_stride: int = 1,
) -> np.ndarray:
    stop = int(frame_count) if frame_stop is None else int(frame_stop)
    return np.arange(
        int(frame_start),
        stop - int(lag_frames),
        int(window_stride),
        dtype=np.int64,
    )


class TemporalPairDataset(Dataset):
    """Two-frame view of the repository's PBC-aware tracked-atom dataset."""

    def __init__(self, dataset: TemporalLAMMPSDumpDataset, *, run_id: str) -> None:
        if int(dataset.sequence_length) != 2:
            raise ValueError(
                "TemporalPairDataset requires a two-frame base dataset, "
                f"got sequence_length={dataset.sequence_length}."
            )
        self.dataset = dataset
        self.run_id = str(run_id)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        points = sample["points"]
        frames = sample["frame_indices"]
        timesteps = sample["timesteps"]
        centers = sample["center_positions"]
        return {
            "points0": points[0],
            "points1": points[1],
            "atom_id": sample["center_atom_id"],
            "frame0": frames[0],
            "frame1": frames[1],
            "timestep0": timesteps[0],
            "timestep1": timesteps[1],
            "coords0": centers[0],
            "coords1": centers[1],
            "run_id": self.run_id,
        }

    def __getitems__(self, indices: Sequence[int]) -> dict[str, Any]:
        batch = self.dataset.__getitems__(indices)
        points = batch["points"]
        frames = batch["frame_indices"]
        timesteps = batch["timesteps"]
        centers = batch["center_positions"]
        return {
            "points0": points[:, 0],
            "points1": points[:, 1],
            "atom_id": batch["center_atom_id"],
            "frame0": frames[:, 0],
            "frame1": frames[:, 1],
            "timestep0": timesteps[:, 0],
            "timestep1": timesteps[:, 1],
            "coords0": centers[:, 0],
            "coords1": centers[:, 1],
            "run_id": [self.run_id] * int(points.shape[0]),
        }


def identity_batch_collate(batch: Any) -> Any:
    return batch


def build_temporal_pair_dataset(
    *,
    trajectory: TrajectorySpec,
    anchor_frames: Sequence[int],
    lag_frames: int,
    num_points: int,
    radius: float,
    center_selection_mode: str,
    center_atom_ids: Sequence[int] | None,
    center_atom_stride: int | None,
    max_center_atoms: int | None,
    center_selection_seed: int,
    center_grid_overlap: float | None,
    center_grid_reference_frame_index: int | None,
    normalize: bool,
    center_neighborhoods: bool,
    selection_method: str,
    rebuild_cache: bool,
    tree_cache_size: int,
    precompute_neighbor_indices: bool,
) -> TemporalPairDataset:
    base = TemporalLAMMPSDumpDataset(
        dump_file=trajectory.path,
        cache_dir=trajectory.cache_dir,
        sequence_length=2,
        num_points=int(num_points),
        radius=float(radius),
        frame_stride=int(lag_frames),
        window_stride=1,
        anchor_frame_indices=[int(value) for value in anchor_frames],
        center_selection_mode=str(center_selection_mode),
        center_atom_ids=center_atom_ids,
        center_atom_stride=center_atom_stride,
        max_center_atoms=max_center_atoms,
        center_selection_seed=int(center_selection_seed),
        center_grid_overlap=center_grid_overlap,
        center_grid_reference_frame_index=center_grid_reference_frame_index,
        normalize=bool(normalize),
        center_neighborhoods=bool(center_neighborhoods),
        selection_method=str(selection_method),
        rebuild_cache=bool(rebuild_cache),
        tree_cache_size=int(tree_cache_size),
        precompute_neighbor_indices=bool(precompute_neighbor_indices),
    )
    return TemporalPairDataset(base, run_id=trajectory.run_id)


__all__ = [
    "TemporalAnchorSplit",
    "TemporalPairDataset",
    "TrajectorySpec",
    "all_pair_anchors",
    "build_temporal_pair_dataset",
    "contiguous_temporal_split",
    "identity_batch_collate",
    "resolve_lag_frame_offset",
]
