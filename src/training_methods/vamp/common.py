from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from src.analysis.config import load_checkpoint_training_config
from src.data_utils.data_load import PointCloudDataset
from src.data_utils.temporal_lammps_dataset import (
    TemporalLAMMPSDumpDataset,
    estimate_lammps_dump_cutoff_radius,
)
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.utils.model_utils import load_model_from_checkpoint


def _identity_or_default_collate(batch: Any) -> Any:
    if isinstance(batch, dict):
        return batch
    return default_collate(batch)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return resolved


def log_progress(component: str, message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{timestamp} - [{component}] {message}", file=sys.stderr, flush=True)


def resolve_device(cuda_device: int | None = 0) -> str:
    if torch.cuda.is_available():
        return f"cuda:{0 if cuda_device is None else int(cuda_device)}"
    return "cpu"


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",") if token.strip()]
        if not tokens:
            raise ValueError("Expected a non-empty comma-separated integer list.")
        return [int(token) for token in tokens]
    values = [int(v) for v in value]
    if not values:
        raise ValueError("Expected a non-empty integer list.")
    return values


def load_contrastive_checkpoint(
    checkpoint_path: str | Path,
    *,
    cuda_device: int | None = 0,
) -> tuple[VICRegModule, DictConfig, str]:
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    cfg = load_checkpoint_training_config(str(checkpoint))
    device = resolve_device(cuda_device)
    model: VICRegModule = load_model_from_checkpoint(
        str(checkpoint),
        cfg,
        device=device,
        module=VICRegModule,
    )
    model.to(device).eval()
    return model, cfg, device


def resolve_num_points(model_cfg: DictConfig, override: int | None) -> int:
    if override is not None:
        resolved = int(override)
        if resolved <= 0:
            raise ValueError(f"num_points must be > 0, got {resolved}.")
        return resolved
    data_cfg = getattr(model_cfg, "data", None)
    num_points = getattr(data_cfg, "num_points", None)
    if num_points is None:
        raise ValueError(
            "Could not resolve num_points from the checkpoint config. "
            "Pass --num-points explicitly."
        )
    resolved = int(num_points)
    if resolved <= 0:
        raise ValueError(f"Resolved num_points must be > 0, got {resolved}.")
    return resolved


def resolve_radius(
    dump_file: str | Path,
    model_cfg: DictConfig,
    *,
    num_points: int,
    radius_override: float | None,
) -> tuple[float, str, dict[str, Any] | None]:
    if radius_override is not None:
        radius = float(radius_override)
        if radius <= 0.0:
            raise ValueError(f"radius must be > 0, got {radius}.")
        return radius, "cli_override", None

    data_cfg = getattr(model_cfg, "data", None)
    model_radius_raw = getattr(data_cfg, "radius", None)
    auto_cutoff_cfg_raw = getattr(data_cfg, "auto_cutoff", None)
    auto_cutoff_cfg = None
    if auto_cutoff_cfg_raw is not None:
        auto_cutoff_cfg = PointCloudDataset._resolve_auto_cutoff_config(
            OmegaConf.to_container(auto_cutoff_cfg_raw, resolve=True),
            default_target_points=int(num_points),
            default_radius=(
                0.0 if model_radius_raw is None else float(model_radius_raw)
            ),
        )

    if auto_cutoff_cfg is not None:
        reference_frame_index = int(auto_cutoff_cfg.get("reference_frame_index", 0))
        estimation = estimate_lammps_dump_cutoff_radius(
            dump_file,
            reference_frame_index=reference_frame_index,
            target_points=max(
                int(num_points),
                int(auto_cutoff_cfg.get("target_points", num_points)),
            ),
            quantile=float(auto_cutoff_cfg.get("quantile", 1.0)),
            estimation_samples=int(
                auto_cutoff_cfg.get("estimation_samples_per_file", 4096)
            ),
            seed=int(auto_cutoff_cfg.get("seed", 0)),
            safety_factor=float(auto_cutoff_cfg.get("safety_factor", 1.0)),
            boundary_margin=auto_cutoff_cfg.get("boundary_margin", None),
            periodic=True,
        )
        return float(estimation["estimated_radius"]), "auto_cutoff", estimation

    if model_radius_raw is None:
        raise ValueError(
            "No cutoff radius could be resolved from the checkpoint config. "
            "Pass --radius explicitly or add data.radius / data.auto_cutoff to the checkpoint config."
        )
    radius = float(model_radius_raw)
    if radius <= 0.0:
        raise ValueError(f"Resolved radius must be > 0, got {radius}.")
    return radius, "checkpoint_data_radius", None


def build_full_trajectory_dataset(
    dump_file: str | Path,
    *,
    radius: float,
    num_points: int,
    cache_dir: str | Path | None,
    normalize: bool,
    center_neighborhoods: bool,
    selection_method: str,
    precompute_neighbor_indices: bool,
    tree_cache_size: int,
    frame_start: int | None = None,
    frame_stop: int | None = None,
    anchor_frame_indices: Sequence[int] | None = None,
    anchor_source_names: Sequence[str] | None = None,
    center_selection_mode: str | None = None,
    center_atom_stride: int | None = None,
    max_center_atoms: int | None = None,
    center_selection_seed: int = 0,
    center_grid_overlap: float | None = None,
    center_grid_reference_frame_index: int | None = None,
) -> TemporalLAMMPSDumpDataset:
    resolved_frame_start = 0 if frame_start is None else int(frame_start)
    resolved_frame_stop = None if frame_stop is None else int(frame_stop)
    resolved_anchor_frame_indices = (
        None
        if anchor_frame_indices is None
        else [int(v) for v in anchor_frame_indices]
    )
    resolved_anchor_source_names = (
        None
        if anchor_source_names is None
        else [str(v) for v in anchor_source_names]
    )

    resolved_center_selection_mode = (
        None if center_selection_mode is None else str(center_selection_mode).strip().lower()
    )
    resolved_center_atom_stride = (
        None if center_atom_stride is None else int(center_atom_stride)
    )
    resolved_max_center_atoms = (
        None if max_center_atoms is None else int(max_center_atoms)
    )

    if resolved_center_selection_mode is None:
        if resolved_max_center_atoms is not None:
            resolved_center_selection_mode = "random_subset"
        else:
            resolved_center_selection_mode = "atom_stride"
            if resolved_center_atom_stride is None:
                resolved_center_atom_stride = 1

    return TemporalLAMMPSDumpDataset(
        dump_file=dump_file,
        cache_dir=cache_dir,
        sequence_length=1,
        num_points=int(num_points),
        radius=float(radius),
        frame_stride=1,
        frame_start=resolved_frame_start,
        frame_stop=resolved_frame_stop,
        anchor_frame_indices=resolved_anchor_frame_indices,
        anchor_source_names=resolved_anchor_source_names,
        center_selection_mode=resolved_center_selection_mode,
        center_atom_stride=resolved_center_atom_stride,
        max_center_atoms=resolved_max_center_atoms,
        center_selection_seed=int(center_selection_seed),
        center_grid_overlap=(
            None if center_grid_overlap is None else float(center_grid_overlap)
        ),
        center_grid_reference_frame_index=(
            None
            if center_grid_reference_frame_index is None
            else int(center_grid_reference_frame_index)
        ),
        normalize=bool(normalize),
        center_neighborhoods=bool(center_neighborhoods),
        selection_method=str(selection_method),
        precompute_neighbor_indices=bool(precompute_neighbor_indices),
        tree_cache_size=int(tree_cache_size),
    )


def build_temporal_dataloader(
    dataset: TemporalLAMMPSDumpDataset,
    *,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    resolved_batch = int(batch_size)
    if resolved_batch <= 0:
        raise ValueError(f"batch_size must be > 0, got {resolved_batch}.")
    resolved_workers = int(num_workers)
    if resolved_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {resolved_workers}.")
    return DataLoader(
        dataset,
        batch_size=resolved_batch,
        shuffle=False,
        drop_last=False,
        num_workers=resolved_workers,
        pin_memory=True,
        persistent_workers=bool(resolved_workers > 0),
        collate_fn=_identity_or_default_collate,
    )


@dataclass(frozen=True)
class FrameWindow:
    name: str
    start: int
    stop: int

    @property
    def length(self) -> int:
        return int(self.stop - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "start": int(self.start),
            "stop": int(self.stop),
            "length": int(self.length),
        }


@dataclass(frozen=True)
class FrameSplit:
    name: str
    start: int
    stop: int

    @property
    def length(self) -> int:
        return int(self.stop - self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": str(self.name),
            "start": int(self.start),
            "stop": int(self.stop),
            "length": int(self.length),
        }


@dataclass(frozen=True)
class LaggedPairs:
    x0: np.ndarray
    x1: np.ndarray
    frame_indices_t: np.ndarray
    frame_indices_tlag: np.ndarray
    timesteps_t: np.ndarray
    timesteps_tlag: np.ndarray
    atom_ids: np.ndarray
    centers_t: np.ndarray
    centers_tlag: np.ndarray

    @property
    def pair_count(self) -> int:
        return int(self.x0.shape[0])


@dataclass(frozen=True)
class TrajectoryEmbeddings:
    invariant_embeddings: np.ndarray
    center_positions: np.ndarray
    atom_ids: np.ndarray
    frame_indices: np.ndarray
    timesteps: np.ndarray
    metadata: dict[str, Any]

    @property
    def frame_count(self) -> int:
        return int(self.invariant_embeddings.shape[0])

    @property
    def num_atoms(self) -> int:
        return int(self.invariant_embeddings.shape[1])

    @property
    def latent_dim(self) -> int:
        return int(self.invariant_embeddings.shape[2])

    def save(self, path: str | Path) -> Path:
        resolved = Path(path).expanduser().resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            resolved,
            invariant_embeddings=self.invariant_embeddings.astype(np.float32, copy=False),
            center_positions=self.center_positions.astype(np.float32, copy=False),
            atom_ids=self.atom_ids.astype(np.int64, copy=False),
            frame_indices=self.frame_indices.astype(np.int64, copy=False),
            timesteps=self.timesteps.astype(np.int64, copy=False),
        )
        meta_payload = dict(self.metadata)
        meta_payload.update(
            {
                "artifact_version": 1,
                "npz_path": str(resolved),
                "frame_count": int(self.frame_count),
                "num_atoms": int(self.num_atoms),
                "latent_dim": int(self.latent_dim),
            }
        )
        save_json(meta_payload, resolved.with_suffix(resolved.suffix + ".meta.json"))
        return resolved

    @classmethod
    def load(cls, path: str | Path) -> "TrajectoryEmbeddings":
        resolved = Path(path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Embedding artifact does not exist: {resolved}")
        meta_path = resolved.with_suffix(resolved.suffix + ".meta.json")
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Embedding artifact metadata does not exist: {meta_path}"
            )
        with meta_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        with np.load(resolved) as payload:
            invariant_embeddings = np.asarray(payload["invariant_embeddings"], dtype=np.float32)
            center_positions = np.asarray(payload["center_positions"], dtype=np.float32)
            atom_ids = np.asarray(payload["atom_ids"], dtype=np.int64)
            frame_indices = np.asarray(payload["frame_indices"], dtype=np.int64)
            timesteps = np.asarray(payload["timesteps"], dtype=np.int64)
        if invariant_embeddings.ndim != 3:
            raise ValueError(
                "Expected invariant_embeddings with shape (frames, atoms, dim), "
                f"got {tuple(invariant_embeddings.shape)}."
            )
        expected_shape = invariant_embeddings.shape[:2] + (3,)
        if center_positions.shape != expected_shape:
            raise ValueError(
                "center_positions shape mismatch in embedding artifact. "
                f"expected={expected_shape}, got={tuple(center_positions.shape)}."
            )
        if atom_ids.shape != (invariant_embeddings.shape[1],):
            raise ValueError(
                "atom_ids shape mismatch in embedding artifact. "
                f"expected={(invariant_embeddings.shape[1],)}, got={tuple(atom_ids.shape)}."
            )
        if frame_indices.shape != (invariant_embeddings.shape[0],):
            raise ValueError(
                "frame_indices shape mismatch in embedding artifact. "
                f"expected={(invariant_embeddings.shape[0],)}, got={tuple(frame_indices.shape)}."
            )
        if timesteps.shape != (invariant_embeddings.shape[0],):
            raise ValueError(
                "timesteps shape mismatch in embedding artifact. "
                f"expected={(invariant_embeddings.shape[0],)}, got={tuple(timesteps.shape)}."
            )
        return cls(
            invariant_embeddings=invariant_embeddings,
            center_positions=center_positions,
            atom_ids=atom_ids,
            frame_indices=frame_indices,
            timesteps=timesteps,
            metadata=metadata,
        )


def resolve_frame_window(
    frame_count: int,
    *,
    window: str,
    frame_start: int | None = None,
    frame_stop: int | None = None,
) -> FrameWindow:
    total = int(frame_count)
    if total <= 0:
        raise ValueError(f"frame_count must be > 0, got {total}.")
    name = str(window).strip().lower()
    if name == "custom":
        if frame_start is None or frame_stop is None:
            raise ValueError(
                "window='custom' requires both frame_start and frame_stop to be set."
            )
        start = int(frame_start)
        stop = int(frame_stop)
    elif name == "full":
        start, stop = 0, total
    elif name in {"early", "middle", "late"}:
        edges = np.linspace(0, total, num=4, dtype=np.int64)
        mapping = {
            "early": (int(edges[0]), int(edges[1])),
            "middle": (int(edges[1]), int(edges[2])),
            "late": (int(edges[2]), int(edges[3])),
        }
        start, stop = mapping[name]
    else:
        raise ValueError(
            "window must be one of ['full', 'early', 'middle', 'late', 'custom'], "
            f"got {window!r}."
        )
    if start < 0 or stop > total or start >= stop:
        raise ValueError(
            "Resolved frame window is invalid. "
            f"start={start}, stop={stop}, frame_count={total}, window={window!r}."
        )
    return FrameWindow(name=name, start=start, stop=stop)


def build_frame_splits(
    window: FrameWindow,
    *,
    train_fraction: float,
    val_fraction: float,
) -> dict[str, FrameSplit]:
    train_fraction = float(train_fraction)
    val_fraction = float(val_fraction)
    if not (0.0 < train_fraction < 1.0):
        raise ValueError(
            f"train_fraction must be in (0, 1), got {train_fraction}."
        )
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}.")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError(
            "train_fraction + val_fraction must be < 1 so the test block is non-empty. "
            f"Got train_fraction={train_fraction}, val_fraction={val_fraction}."
        )

    length = int(window.length)
    if length < 3:
        raise ValueError(
            f"Frame window must contain at least 3 frames to form train/val/test blocks, got {length}."
        )

    train_frames = max(1, int(math.floor(length * train_fraction)))
    val_frames = max(1, int(math.floor(length * val_fraction)))
    test_frames = length - train_frames - val_frames

    if test_frames <= 0:
        raise ValueError(
            "Resolved test split is empty. "
            f"window_length={length}, train_frames={train_frames}, val_frames={val_frames}."
        )

    train_start = int(window.start)
    train_stop = train_start + train_frames
    val_start = train_stop
    val_stop = val_start + val_frames
    test_start = val_stop
    test_stop = int(window.stop)

    return {
        "train": FrameSplit("train", train_start, train_stop),
        "val": FrameSplit("val", val_start, val_stop),
        "test": FrameSplit("test", test_start, test_stop),
    }


def _build_pairs_for_range(
    embeddings: TrajectoryEmbeddings,
    *,
    lag: int,
    frame_start: int,
    frame_stop: int,
) -> LaggedPairs:
    lag_int = int(lag)
    if lag_int <= 0:
        raise ValueError(f"lag must be > 0, got {lag_int}.")
    if frame_start < 0 or frame_stop > embeddings.frame_count or frame_start >= frame_stop:
        raise ValueError(
            "Invalid frame range for lagged pairs. "
            f"frame_start={frame_start}, frame_stop={frame_stop}, frame_count={embeddings.frame_count}."
        )
    usable_frames = frame_stop - frame_start - lag_int
    if usable_frames <= 0:
        raise ValueError(
            "Lagged pair range contains no valid anchor frames. "
            f"frame_start={frame_start}, frame_stop={frame_stop}, lag={lag_int}."
        )

    anchor_frames = np.arange(frame_start, frame_stop - lag_int, dtype=np.int64)
    x0 = embeddings.invariant_embeddings[anchor_frames]
    x1 = embeddings.invariant_embeddings[anchor_frames + lag_int]
    centers_t = embeddings.center_positions[anchor_frames]
    centers_tlag = embeddings.center_positions[anchor_frames + lag_int]

    pair_count = int(anchor_frames.size) * int(embeddings.num_atoms)
    latent_dim = int(embeddings.latent_dim)
    x0_flat = np.asarray(x0, dtype=np.float64).reshape(pair_count, latent_dim)
    x1_flat = np.asarray(x1, dtype=np.float64).reshape(pair_count, latent_dim)
    centers_t_flat = np.asarray(centers_t, dtype=np.float32).reshape(pair_count, 3)
    centers_tlag_flat = np.asarray(centers_tlag, dtype=np.float32).reshape(pair_count, 3)

    frame_indices_t = np.repeat(
        embeddings.frame_indices[anchor_frames].astype(np.int64, copy=False),
        embeddings.num_atoms,
    )
    frame_indices_tlag = np.repeat(
        embeddings.frame_indices[anchor_frames + lag_int].astype(np.int64, copy=False),
        embeddings.num_atoms,
    )
    timesteps_t = np.repeat(
        embeddings.timesteps[anchor_frames].astype(np.int64, copy=False),
        embeddings.num_atoms,
    )
    timesteps_tlag = np.repeat(
        embeddings.timesteps[anchor_frames + lag_int].astype(np.int64, copy=False),
        embeddings.num_atoms,
    )
    atom_ids = np.tile(embeddings.atom_ids.astype(np.int64, copy=False), anchor_frames.size)

    return LaggedPairs(
        x0=x0_flat,
        x1=x1_flat,
        frame_indices_t=frame_indices_t,
        frame_indices_tlag=frame_indices_tlag,
        timesteps_t=timesteps_t,
        timesteps_tlag=timesteps_tlag,
        atom_ids=atom_ids,
        centers_t=centers_t_flat,
        centers_tlag=centers_tlag_flat,
    )


def build_lagged_pairs(
    embeddings: TrajectoryEmbeddings,
    *,
    lag: int,
    splits: dict[str, FrameSplit] | None = None,
    window: FrameWindow | None = None,
) -> dict[str, LaggedPairs]:
    if splits is not None and window is not None:
        raise ValueError("Pass either splits or window, not both.")
    if splits is None:
        if window is None:
            window = FrameWindow(name="full", start=0, stop=embeddings.frame_count)
        return {
            "all": _build_pairs_for_range(
                embeddings,
                lag=lag,
                frame_start=int(window.start),
                frame_stop=int(window.stop),
            )
        }

    output: dict[str, LaggedPairs] = {}
    for split_name, split in splits.items():
        output[str(split_name)] = _build_pairs_for_range(
            embeddings,
            lag=lag,
            frame_start=int(split.start),
            frame_stop=int(split.stop),
        )
    return output


def load_local_neighborhoods(
    embeddings: TrajectoryEmbeddings,
    *,
    frame_indices: Sequence[int],
    atom_ids: Sequence[int],
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    frame_idx_arr = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
    atom_id_arr = np.asarray(atom_ids, dtype=np.int64).reshape(-1)
    if frame_idx_arr.shape != atom_id_arr.shape:
        raise ValueError(
            "frame_indices and atom_ids must have the same shape when loading local neighborhoods, "
            f"got frame_indices.shape={frame_idx_arr.shape}, atom_ids.shape={atom_id_arr.shape}."
        )
    if frame_idx_arr.size == 0:
        raise ValueError("No samples were requested while loading local neighborhoods.")

    dump_file = embeddings.metadata.get("dump_file", None)
    if dump_file is None or str(dump_file).strip() == "":
        raise ValueError(
            "Embedding metadata is missing dump_file, so representative neighborhoods cannot be reconstructed."
        )
    radius = embeddings.metadata.get("radius", None)
    if radius is None:
        raise ValueError(
            "Embedding metadata is missing radius, so representative neighborhoods cannot be reconstructed."
        )
    num_points = embeddings.metadata.get("num_points", None)
    if num_points is None:
        raise ValueError(
            "Embedding metadata is missing num_points, so representative neighborhoods cannot be reconstructed."
        )

    unique_frames = np.unique(frame_idx_arr.astype(np.int64, copy=False))
    dataset = TemporalLAMMPSDumpDataset(
        dump_file=dump_file,
        cache_dir=cache_dir,
        sequence_length=1,
        num_points=int(num_points),
        radius=float(radius),
        frame_stride=1,
        anchor_frame_indices=[int(v) for v in unique_frames.tolist()],
        center_selection_mode="atom_stride",
        center_atom_stride=1,
        normalize=bool(embeddings.metadata.get("normalize", True)),
        center_neighborhoods=bool(
            embeddings.metadata.get("center_neighborhoods", True)
        ),
        selection_method=str(embeddings.metadata.get("selection_method", "closest")),
        tree_cache_size=int(embeddings.metadata.get("tree_cache_size", 4)),
    )
    frame_to_slot = {int(frame): idx for idx, frame in enumerate(unique_frames.tolist())}
    atom_positions = np.searchsorted(dataset.atom_ids, atom_id_arr)
    valid = (atom_positions >= 0) & (atom_positions < int(dataset.num_atoms))
    if not np.all(valid):
        missing = atom_id_arr[~valid]
        raise ValueError(
            "One or more requested atom ids were outside the cached atom-id range while "
            "reconstructing representative neighborhoods. "
            f"missing_atom_ids={missing.tolist()}."
        )
    matched = dataset.atom_ids[atom_positions]
    if not np.array_equal(matched, atom_id_arr):
        raise ValueError(
            "One or more requested atom ids were not found while reconstructing neighborhoods. "
            f"requested_atom_ids={atom_id_arr.tolist()}."
        )
    sample_indices = np.asarray(
        [
            int(frame_to_slot[int(frame_idx)]) * int(dataset.center_count) + int(atom_slot)
            for frame_idx, atom_slot in zip(frame_idx_arr.tolist(), atom_positions.tolist(), strict=True)
        ],
        dtype=np.int64,
    )
    batch = dataset.__getitems__(sample_indices.tolist())
    points = batch["points"]
    if not torch.is_tensor(points):
        points = torch.as_tensor(points)
    if points.ndim != 4 or points.shape[1] != 1 or points.shape[-1] != 3:
        raise ValueError(
            "Unexpected representative neighborhood tensor shape. "
            f"Expected (B, 1, N, 3), got {tuple(points.shape)}."
        )
    return points[:, 0].detach().cpu().numpy().astype(np.float32, copy=False)
