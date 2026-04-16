from __future__ import annotations

import hashlib
import json
import os
import time
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import MethodType
from typing import Any, Sequence

import numpy as np
import torch
from numpy.lib.format import open_memmap
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("temporal_lammps_dataset")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    def _print(self, msg: str) -> None:
        self.info(msg)

    logger.print = MethodType(_print, logger)
    return logger


logger = _setup_logger()


_SUPPORTED_POSITION_COLUMNS: tuple[tuple[str, str, str], ...] = (
    ("x", "y", "z"),
    ("xu", "yu", "zu"),
)
_PRECOMPUTED_NEIGHBOR_INDEX_PROCESS_CACHE: dict[str, np.ndarray] = {}


def _normalize_point_cloud(points: np.ndarray, radius: float | None) -> np.ndarray:
    if radius is None:
        raise ValueError(
            "Temporal local-structure normalization requires an explicit cutoff radius. "
            "Resolve data.radius or enable data.auto_cutoff so the normalization scale is well-defined."
        )
    return points / float(radius)


def _sanitize_periodic_points(points: np.ndarray, box_lengths: np.ndarray) -> np.ndarray:
    """Force coordinates into the half-open periodic domain [0, box_length)."""
    box_lengths = np.asarray(box_lengths, dtype=np.float32)
    upper = np.nextafter(box_lengths, np.zeros_like(box_lengths, dtype=np.float32))
    sanitized = np.asarray(points, dtype=np.float32)
    if np.any(sanitized < 0.0) or np.any(sanitized >= box_lengths[None, :]):
        sanitized = np.array(sanitized, dtype=np.float32, copy=True)
        np.clip(sanitized, 0.0, upper[None, :], out=sanitized)
    return sanitized


def _stable_unique_int(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    _, first_positions = np.unique(arr, return_index=True)
    return arr[np.sort(first_positions)].astype(np.int64, copy=False)


@dataclass(frozen=True)
class _DumpScanResult:
    frame_count: int
    num_atoms: int
    atom_columns: tuple[str, ...]
    timesteps: np.ndarray
    box_low: np.ndarray
    box_high: np.ndarray


_SCAN_RESULT_PROCESS_CACHE: dict[tuple[str, int, int], _DumpScanResult] = {}


def inspect_lammps_dump_file(dump_file: str | Path) -> dict[str, Any]:
    dump_path = Path(dump_file).expanduser().resolve()
    if not dump_path.exists():
        raise FileNotFoundError(f"LAMMPS dump file does not exist: {dump_path}")
    if not dump_path.is_file():
        raise ValueError(f"dump_file must be a file, got: {dump_path}")

    scan = TemporalLAMMPSDumpDataset.scan_dump_file(dump_path)
    timestep_deltas = np.diff(scan.timesteps).astype(np.int64, copy=False)
    unique_deltas = np.unique(timestep_deltas).astype(np.int64, copy=False)
    position_cache_bytes = int(scan.frame_count) * int(scan.num_atoms) * 3 * np.dtype(np.float32).itemsize

    return {
        "source_path": str(dump_path),
        "frame_count": int(scan.frame_count),
        "num_atoms": int(scan.num_atoms),
        "atom_columns": list(scan.atom_columns),
        "first_timestep": int(scan.timesteps[0]),
        "last_timestep": int(scan.timesteps[-1]),
        "unique_timestep_deltas": unique_deltas.tolist(),
        "box_low_first_frame": scan.box_low[0].astype(np.float64).tolist(),
        "box_high_first_frame": scan.box_high[0].astype(np.float64).tolist(),
        "estimated_positions_cache_bytes": position_cache_bytes,
        "estimated_positions_cache_gib": position_cache_bytes / float(1024 ** 3),
    }


def estimate_lammps_dump_cutoff_radius(
    dump_file: str | Path,
    *,
    reference_frame_index: int = 0,
    target_points: int,
    quantile: float,
    estimation_samples: int,
    seed: int,
    safety_factor: float,
    boundary_margin: float | None = None,
    periodic: bool = True,
) -> dict[str, Any]:
    dump_path = Path(dump_file).expanduser().resolve()
    if not dump_path.exists():
        raise FileNotFoundError(f"LAMMPS dump file does not exist: {dump_path}")
    if not dump_path.is_file():
        raise ValueError(f"dump_file must be a file, got: {dump_path}")
    if target_points <= 0:
        raise ValueError(f"target_points must be > 0, got {target_points}.")
    if not (0.0 < float(quantile) <= 1.0):
        raise ValueError(f"quantile must be in (0, 1], got {quantile}.")
    if estimation_samples <= 0:
        raise ValueError(f"estimation_samples must be > 0, got {estimation_samples}.")
    if safety_factor <= 0.0:
        raise ValueError(f"safety_factor must be > 0, got {safety_factor}.")
    if boundary_margin is not None and float(boundary_margin) < 0.0:
        raise ValueError(f"boundary_margin must be >= 0 when provided, got {boundary_margin}.")

    points, box_lengths, timestep = TemporalLAMMPSDumpDataset.load_dump_frame_positions(
        dump_path,
        frame_index=int(reference_frame_index),
    )
    num_atoms = int(points.shape[0])
    k = min(int(target_points), num_atoms)
    rng = np.random.default_rng(int(seed))
    candidate_indices = np.arange(num_atoms, dtype=np.int64)
    if boundary_margin is not None and float(boundary_margin) > 0.0:
        boundary_margin_value = float(boundary_margin)
        if periodic:
            lower = np.full((3,), boundary_margin_value, dtype=np.float32)
            upper = np.asarray(box_lengths, dtype=np.float32) - boundary_margin_value
            interior_mask = np.all(
                (points >= lower[None, :]) & (points <= upper[None, :]),
                axis=1,
            )
        else:
            min_coords = points.min(axis=0)
            max_coords = points.max(axis=0)
            interior_mask = np.all(
                (points >= (min_coords + boundary_margin_value))
                & (points <= (max_coords - boundary_margin_value)),
                axis=1,
            )
        interior_indices = np.flatnonzero(interior_mask)
        if interior_indices.size > 0:
            candidate_indices = interior_indices.astype(np.int64, copy=False)

    sampled = min(int(estimation_samples), int(candidate_indices.size))
    center_indices = rng.choice(candidate_indices, size=sampled, replace=False)
    tree = cKDTree(
        points,
        boxsize=box_lengths if bool(periodic) else None,
        balanced_tree=False,
    )
    dists, _ = tree.query(points[center_indices], k=k)
    dists = np.asarray(dists, dtype=np.float64)
    kth = dists.reshape(-1) if k == 1 else dists[:, k - 1]
    estimated_radius = float(np.quantile(kth, float(quantile))) * float(safety_factor)
    coverage = float(np.mean(kth <= estimated_radius))
    return {
        "reference_frame_index": int(reference_frame_index),
        "reference_timestep": int(timestep),
        "target_points": int(target_points),
        "quantile": float(quantile),
        "estimation_samples": int(sampled),
        "seed": int(seed),
        "safety_factor": float(safety_factor),
        "boundary_margin": None if boundary_margin is None else float(boundary_margin),
        "periodic": bool(periodic),
        "estimated_radius": float(estimated_radius),
        "coverage": float(coverage),
        "kth_distance_mean": float(np.mean(kth)),
        "kth_distance_max": float(np.max(kth)),
    }


class TemporalLAMMPSDumpDataset(Dataset):
    """Temporal dataset for tracked local neighborhoods from LAMMPS dump files.

    The dataset converts the text dump into a persistent binary cache on first use,
    ordered by atom id for stable cross-frame tracking. Each dataset item is a
    temporal window of local structures centered on the same tracked atom across
    consecutive frames.

    Returned sample keys:
        - "points": float32 tensor of shape (T, N, 3)
        - "center_positions": float32 tensor of shape (T, 3)
        - "timesteps": int64 tensor of shape (T,)
        - "frame_indices": int64 tensor of shape (T,)
        - "center_atom_id": int64 tensor scalar
        - "source_path": absolute source dump path as string
    """

    cache_version: int = 1
    neighbor_index_cache_version: int = 1

    def __init__(
        self,
        dump_file: str | Path,
        *,
        sequence_length: int,
        num_points: int,
        radius: float | None = None,
        frame_stride: int = 1,
        window_stride: int = 1,
        frame_start: int = 0,
        frame_stop: int | None = None,
        anchor_frame_indices: Sequence[int] | None = None,
        anchor_source_names: Sequence[str] | None = None,
        center_selection_mode: str | None = None,
        center_atom_ids: Sequence[int] | None = None,
        center_atom_stride: int | None = None,
        max_center_atoms: int | None = None,
        center_selection_seed: int = 0,
        center_grid_overlap: float | None = None,
        center_grid_reference_frame_index: int | None = None,
        normalize: bool = True,
        center_neighborhoods: bool = True,
        selection_method: str = "closest",
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
        tree_cache_size: int = 4,
        precompute_neighbor_indices: bool = False,
        build_lock_timeout_sec: float = 7200.0,
        build_lock_stale_sec: float = 86400.0,
    ) -> None:
        super().__init__()
        self.dump_file = Path(dump_file).expanduser().resolve()
        if not self.dump_file.exists():
            raise FileNotFoundError(f"LAMMPS dump file does not exist: {self.dump_file}")
        if not self.dump_file.is_file():
            raise ValueError(f"dump_file must be a file, got: {self.dump_file}")

        self.sequence_length = int(sequence_length)
        self.num_points = int(num_points)
        self.radius = None if radius is None else float(radius)
        self.frame_stride = int(frame_stride)
        self.window_stride = int(window_stride)
        self.frame_start = int(frame_start)
        self.frame_stop = None if frame_stop is None else int(frame_stop)
        self.anchor_frame_indices = (
            None if anchor_frame_indices is None else np.asarray(anchor_frame_indices, dtype=np.int64)
        )
        self.anchor_source_names = None if anchor_source_names is None else [str(v) for v in anchor_source_names]
        self.center_selection_mode = (
            None if center_selection_mode is None else str(center_selection_mode).strip().lower()
        )
        self.normalize = bool(normalize)
        self.center_neighborhoods = bool(center_neighborhoods)
        self.center_selection_seed = int(center_selection_seed)
        self.center_grid_overlap = (
            None
            if center_grid_overlap is None
            else float(center_grid_overlap)
        )
        self.center_grid_reference_frame_index = (
            None
            if center_grid_reference_frame_index is None
            else int(center_grid_reference_frame_index)
        )
        self.tree_cache_size = int(tree_cache_size)
        self.precompute_neighbor_indices = bool(precompute_neighbor_indices)
        self.build_lock_timeout_sec = float(build_lock_timeout_sec)
        self.build_lock_stale_sec = float(build_lock_stale_sec)

        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be > 0, got {self.sequence_length}")
        if self.num_points <= 0:
            raise ValueError(f"num_points must be > 0, got {self.num_points}")
        if self.radius is not None and self.radius <= 0.0:
            raise ValueError(f"radius must be > 0 when provided, got {self.radius}")
        if self.normalize and self.radius is None:
            raise ValueError(
                "TemporalLAMMPSDumpDataset normalization requires an explicit cutoff radius. "
                "Pass radius=<resolved_cutoff> from data.radius or data.auto_cutoff before constructing the dataset."
            )
        if self.frame_stride <= 0:
            raise ValueError(f"frame_stride must be > 0, got {self.frame_stride}")
        if self.window_stride <= 0:
            raise ValueError(f"window_stride must be > 0, got {self.window_stride}")
        if self.frame_start < 0:
            raise ValueError(f"frame_start must be >= 0, got {self.frame_start}")
        if self.tree_cache_size <= 0:
            raise ValueError(f"tree_cache_size must be > 0, got {self.tree_cache_size}")
        if (
            self.center_grid_overlap is not None
            and self.center_grid_overlap >= 2.0
        ):
            raise ValueError(
                "center_grid_overlap must be < 2.0 so the derived center spacing stays positive. "
                f"Got {self.center_grid_overlap}."
            )
        if (
            self.center_grid_reference_frame_index is not None
            and self.center_grid_reference_frame_index < 0
        ):
            raise ValueError(
                "center_grid_reference_frame_index must be >= 0, "
                f"got {self.center_grid_reference_frame_index}."
            )

        selection_method = str(selection_method).strip().lower()
        if selection_method not in {"closest", "radius_then_closest"}:
            raise ValueError(
                "selection_method must be 'closest' or 'radius_then_closest', "
                f"got {selection_method!r}."
            )
        if selection_method == "radius_then_closest" and self.radius is None:
            raise ValueError(
                "selection_method='radius_then_closest' requires radius to be set, "
                f"got radius={self.radius}."
            )
        self.selection_method = selection_method

        self.cache_dir = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else self.dump_file.with_suffix(".temporal_cache")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._prepare_cache(rebuild_cache=bool(rebuild_cache))
        self._load_cache()
        if self.num_points > self.num_atoms:
            raise ValueError(
                f"num_points ({self.num_points}) cannot exceed num_atoms ({self.num_atoms}) "
                f"for dump_file={self.dump_file}."
            )

        self._tree_cache: OrderedDict[int, cKDTree] = OrderedDict()
        self._precomputed_neighbor_indices: np.ndarray | None = None
        self._precomputed_neighbor_cache_path: Path | None = None
        self._center_atom_indices = self._resolve_center_atom_indices(
            center_selection_mode=self.center_selection_mode,
            center_atom_ids=center_atom_ids,
            center_atom_stride=center_atom_stride,
            max_center_atoms=max_center_atoms,
        )
        if self._center_atom_indices.size == 0:
            raise ValueError(
                "Center atom selection produced zero atoms. "
                f"dump_file={self.dump_file}, num_atoms={self.num_atoms}."
            )

        self._window_start_frames = self._resolve_window_start_frames()
        if self._window_start_frames.size == 0:
            raise ValueError(
                "Temporal window configuration produced zero valid samples. "
                f"frame_count={self.frame_count}, sequence_length={self.sequence_length}, "
                f"frame_stride={self.frame_stride}, frame_start={self.frame_start}, "
                f"frame_stop={self.frame_stop}, window_stride={self.window_stride}."
            )

        self.window_source_names = self._resolve_window_source_names()
        self.sample_source_names = [
            str(self.window_source_names[window_slot])
            for window_slot in range(int(self._window_start_frames.size))
            for _ in range(int(self._center_atom_indices.size))
        ]
        self.sample_anchor_frame_indices = np.repeat(
            self._window_start_frames.astype(np.int64, copy=False),
            int(self._center_atom_indices.size),
        )
        if self.precompute_neighbor_indices:
            self._prepare_precomputed_neighbor_indices()

        logger.print(
            "[temporal-lammps] "
            f"Loaded dataset from {self.dump_file} with "
            f"{self.frame_count} frames, {self.num_atoms} atoms, "
            f"{self._center_atom_indices.size} tracked centers "
            f"(selection={self.center_selection_mode or 'legacy'}), "
            f"{self._window_start_frames.size} windows, total_samples={len(self)}."
        )

    def __len__(self) -> int:
        return self.window_count * self.center_count

    @property
    def center_count(self) -> int:
        return int(self._center_atom_indices.size)

    @property
    def center_atom_indices(self) -> np.ndarray:
        return np.asarray(self._center_atom_indices, dtype=np.int64).copy()

    @property
    def center_atom_ids(self) -> np.ndarray:
        return np.asarray(self.atom_ids[self._center_atom_indices], dtype=np.int64).copy()

    @property
    def window_count(self) -> int:
        return int(self._window_start_frames.size)

    @property
    def window_start_frames(self) -> np.ndarray:
        return np.asarray(self._window_start_frames, dtype=np.int64).copy()

    def __getitem__(self, index: int) -> dict[str, Any]:
        batch = self._build_batch_from_indices(np.asarray([index], dtype=np.int64))
        return {
            "points": batch["points"][0],
            "local_atom_ids": batch["local_atom_ids"][0],
            "coords": batch["coords"][0],
            "center_positions": batch["center_positions"][0],
            "timesteps": batch["timesteps"][0],
            "frame_indices": batch["frame_indices"][0],
            "center_atom_id": batch["center_atom_id"][0],
            "instance_id": batch["instance_id"][0],
            "anchor_frame_index": batch["anchor_frame_index"][0],
            "anchor_timestep": batch["anchor_timestep"][0],
            "source_path": batch["source_path"][0],
        }

    def __getitems__(self, indices: Sequence[int]) -> dict[str, Any]:
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        return self._build_batch_from_indices(index_array)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_tree_cache"] = OrderedDict()
        state["_precomputed_neighbor_indices"] = None
        return state

    def _neighbor_index_cache_spec(self) -> dict[str, Any]:
        source_stat = self.dump_file.stat()
        center_atom_indices_bytes = np.asarray(
            self._center_atom_indices,
            dtype=np.int64,
        ).tobytes()
        return {
            "neighbor_index_cache_version": self.neighbor_index_cache_version,
            "source_path": str(self.dump_file),
            "source_size_bytes": int(source_stat.st_size),
            "source_mtime_ns": int(source_stat.st_mtime_ns),
            "frame_count": int(self.frame_count),
            "num_atoms": int(self.num_atoms),
            "center_count": int(self.center_count),
            "num_points": int(self.num_points),
            "selection_method": str(self.selection_method),
            "radius": None if self.radius is None else float(self.radius),
            "center_atom_indices_sha1": hashlib.sha1(center_atom_indices_bytes).hexdigest(),
        }

    def _resolve_neighbor_index_cache_paths(self) -> tuple[dict[str, Any], Path, Path]:
        spec = self._neighbor_index_cache_spec()
        cache_key = hashlib.sha1(
            json.dumps(spec, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        cache_path = self.cache_dir / f"neighbor_indices_{cache_key}.npy"
        manifest_path = self.cache_dir / f"neighbor_indices_{cache_key}.json"
        return spec, cache_path, manifest_path

    def _neighbor_index_cache_is_valid(
        self,
        *,
        spec: dict[str, Any],
        cache_path: Path,
        manifest_path: Path,
    ) -> bool:
        if not cache_path.exists() or not manifest_path.exists():
            return False
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest != spec:
            return False

        cache_array = np.load(cache_path, mmap_mode="r")
        expected_shape = (self.frame_count, self.center_count, self.num_points)
        if tuple(cache_array.shape) != expected_shape:
            return False
        if cache_array.dtype != np.int32:
            return False
        return True

    def _load_precomputed_neighbor_indices(self) -> np.ndarray | None:
        if self._precomputed_neighbor_indices is not None:
            return self._precomputed_neighbor_indices
        if self._precomputed_neighbor_cache_path is None:
            return None

        cache_key = str(self._precomputed_neighbor_cache_path)
        cached = _PRECOMPUTED_NEIGHBOR_INDEX_PROCESS_CACHE.get(cache_key)
        if cached is not None:
            self._precomputed_neighbor_indices = cached
            return cached

        if not self._precomputed_neighbor_cache_path.exists():
            raise FileNotFoundError(
                "Precomputed temporal neighbor-index cache is missing. "
                f"cache_path={self._precomputed_neighbor_cache_path}."
            )

        cache_array = np.load(self._precomputed_neighbor_cache_path, mmap_mode="r")
        expected_shape = (self.frame_count, self.center_count, self.num_points)
        if tuple(cache_array.shape) != expected_shape:
            raise ValueError(
                "Precomputed temporal neighbor-index cache has an unexpected shape. "
                f"expected={expected_shape}, got={tuple(cache_array.shape)}, "
                f"cache_path={self._precomputed_neighbor_cache_path}."
            )
        if cache_array.dtype != np.int32:
            raise ValueError(
                "Precomputed temporal neighbor-index cache must use int32 indices. "
                f"got dtype={cache_array.dtype}, cache_path={self._precomputed_neighbor_cache_path}."
            )

        _PRECOMPUTED_NEIGHBOR_INDEX_PROCESS_CACHE[cache_key] = cache_array
        self._precomputed_neighbor_indices = cache_array
        return cache_array

    def _prepare_precomputed_neighbor_indices(self) -> None:
        if self.num_atoms > np.iinfo(np.int32).max:
            raise ValueError(
                "Precomputed temporal neighbor-index cache requires num_atoms <= int32 max. "
                f"Got num_atoms={self.num_atoms}."
            )

        spec, cache_path, manifest_path = self._resolve_neighbor_index_cache_paths()
        self._precomputed_neighbor_cache_path = cache_path

        if self._neighbor_index_cache_is_valid(
            spec=spec,
            cache_path=cache_path,
            manifest_path=manifest_path,
        ):
            self._load_precomputed_neighbor_indices()
            logger.print(
                "[temporal-lammps] "
                f"Loaded precomputed neighbor-index cache from {cache_path}."
            )
            return

        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        self._acquire_build_lock(lock_path)
        try:
            if self._neighbor_index_cache_is_valid(
                spec=spec,
                cache_path=cache_path,
                manifest_path=manifest_path,
            ):
                self._load_precomputed_neighbor_indices()
                logger.print(
                    "[temporal-lammps] "
                    f"Loaded precomputed neighbor-index cache from {cache_path}."
                )
                return

            if cache_path.exists():
                cache_path.unlink()
            if manifest_path.exists():
                manifest_path.unlink()

            logger.print(
                "[temporal-lammps] "
                f"Building precomputed neighbor-index cache in {cache_path}."
            )
            cache_array = open_memmap(
                cache_path,
                mode="w+",
                dtype=np.int32,
                shape=(self.frame_count, self.center_count, self.num_points),
            )
            for frame_idx in range(self.frame_count):
                frame_points = np.asarray(self.positions[frame_idx], dtype=np.float32)
                centers = np.asarray(frame_points[self._center_atom_indices], dtype=np.float32)
                selected = self._query_local_structures(frame_idx=frame_idx, centers=centers)
                expected_shape = (self.center_count, self.num_points)
                if tuple(selected.shape) != expected_shape:
                    raise RuntimeError(
                        "Precomputed temporal neighbor-index cache produced an unexpected shape. "
                        f"frame_idx={frame_idx}, expected_shape={expected_shape}, "
                        f"got_shape={tuple(selected.shape)}, cache_path={cache_path}."
                    )
                cache_array[frame_idx] = selected.astype(np.int32, copy=False)
                if frame_idx == 0 or (frame_idx + 1) % 10 == 0 or (frame_idx + 1) == self.frame_count:
                    logger.print(
                        "[temporal-lammps] "
                        f"Cached temporal neighbors for frame {frame_idx + 1}/{self.frame_count}."
                    )
            cache_array.flush()
            with manifest_path.open("w", encoding="utf-8") as handle:
                json.dump(spec, handle, indent=2)
        finally:
            self._release_build_lock(lock_path)

        self._load_precomputed_neighbor_indices()
        logger.print(
            "[temporal-lammps] "
            f"Finished building precomputed neighbor-index cache: {cache_path}."
        )

    @property
    def frame_count(self) -> int:
        return int(self._manifest["frame_count"])

    @property
    def num_atoms(self) -> int:
        return int(self._manifest["num_atoms"])

    def _prepare_cache(self, *, rebuild_cache: bool) -> None:
        lock_path = self.cache_dir / ".build.lock"
        self._acquire_build_lock(lock_path)
        try:
            if not rebuild_cache and self._cache_is_valid():
                return
            self._clear_cache_payload()
            scan = self.scan_dump_file(self.dump_file)
            self._build_cache(scan)
        finally:
            self._release_build_lock(lock_path)

    def _acquire_build_lock(self, lock_path: Path) -> None:
        start_time = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "pid": os.getpid(),
                                "created_unix": time.time(),
                                "source_path": str(self.dump_file),
                            }
                        )
                    )
                return
            except FileExistsError:
                if lock_path.exists():
                    age_sec = time.time() - lock_path.stat().st_mtime
                    if age_sec > self.build_lock_stale_sec:
                        logger.print(
                            "[temporal-lammps] "
                            f"Removing stale cache build lock: {lock_path} (age={age_sec:.1f}s)."
                        )
                        lock_path.unlink()
                        continue
                waited_sec = time.time() - start_time
                if waited_sec > self.build_lock_timeout_sec:
                    raise TimeoutError(
                        "Timed out while waiting for temporal cache build lock. "
                        f"lock_path={lock_path}, waited_sec={waited_sec:.1f}, "
                        f"source_path={self.dump_file}."
                    )
                time.sleep(1.0)

    @staticmethod
    def _release_build_lock(lock_path: Path) -> None:
        if lock_path.exists():
            lock_path.unlink()

    def _cache_is_valid(self) -> bool:
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            return False
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        required_files = [
            self.cache_dir / "positions.npy",
            self.cache_dir / "atom_ids.npy",
            self.cache_dir / "timesteps.npy",
            self.cache_dir / "box_low.npy",
            self.cache_dir / "box_high.npy",
        ]
        for path in required_files:
            if not path.exists():
                return False

        if int(manifest.get("cache_version", -1)) != self.cache_version:
            return False
        if manifest.get("source_path") != str(self.dump_file):
            return False

        stat = self.dump_file.stat()
        if int(manifest.get("source_size_bytes", -1)) != int(stat.st_size):
            return False
        if int(manifest.get("source_mtime_ns", -1)) != int(stat.st_mtime_ns):
            return False
        return True

    def _clear_cache_payload(self) -> None:
        for filename in [
            "manifest.json",
            "positions.npy",
            "atom_ids.npy",
            "atom_types.npy",
            "timesteps.npy",
            "box_low.npy",
            "box_high.npy",
        ]:
            path = self.cache_dir / filename
            if path.exists():
                path.unlink()

    @classmethod
    def _scan_cache_key(cls, dump_path: Path) -> tuple[str, int, int]:
        stat = dump_path.stat()
        return str(dump_path), int(stat.st_size), int(stat.st_mtime_ns)

    @classmethod
    def _resolve_cache_dir_for_dump(
        cls,
        dump_file: str | Path,
        *,
        cache_dir: str | Path | None = None,
    ) -> Path:
        dump_path = Path(dump_file).expanduser().resolve()
        if cache_dir is not None:
            return Path(cache_dir).expanduser().resolve()
        return dump_path.with_suffix(".temporal_cache")

    @classmethod
    def _load_scan_result_from_cache(
        cls,
        dump_path: Path,
        *,
        cache_dir: str | Path | None = None,
    ) -> _DumpScanResult | None:
        resolved_cache_dir = cls._resolve_cache_dir_for_dump(dump_path, cache_dir=cache_dir)
        manifest_path = resolved_cache_dir / "manifest.json"
        if not manifest_path.exists():
            return None

        try:
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception as exc:
            logger.print(
                "[temporal-lammps] "
                f"Ignoring unreadable cache manifest at {manifest_path}: {exc}."
            )
            return None

        required_files = [
            resolved_cache_dir / "positions.npy",
            resolved_cache_dir / "atom_ids.npy",
            resolved_cache_dir / "timesteps.npy",
            resolved_cache_dir / "box_low.npy",
            resolved_cache_dir / "box_high.npy",
        ]
        for path in required_files:
            if not path.exists():
                return None

        source_stat = dump_path.stat()
        if int(manifest.get("cache_version", -1)) != cls.cache_version:
            return None
        if manifest.get("source_path") != str(dump_path):
            return None
        if int(manifest.get("source_size_bytes", -1)) != int(source_stat.st_size):
            return None
        if int(manifest.get("source_mtime_ns", -1)) != int(source_stat.st_mtime_ns):
            return None

        atom_columns_raw = manifest.get("atom_columns")
        if not isinstance(atom_columns_raw, list) or not atom_columns_raw:
            return None

        timesteps = np.load(resolved_cache_dir / "timesteps.npy", mmap_mode="r")
        box_low = np.load(resolved_cache_dir / "box_low.npy", mmap_mode="r")
        box_high = np.load(resolved_cache_dir / "box_high.npy", mmap_mode="r")
        return _DumpScanResult(
            frame_count=int(manifest["frame_count"]),
            num_atoms=int(manifest["num_atoms"]),
            atom_columns=tuple(str(name) for name in atom_columns_raw),
            timesteps=timesteps.astype(np.int64, copy=False),
            box_low=box_low.astype(np.float32, copy=False),
            box_high=box_high.astype(np.float32, copy=False),
        )

    @classmethod
    def scan_dump_file(
        cls,
        dump_file: str | Path,
        *,
        cache_dir: str | Path | None = None,
    ) -> _DumpScanResult:
        dump_path = Path(dump_file).expanduser().resolve()
        if not dump_path.exists():
            raise FileNotFoundError(f"LAMMPS dump file does not exist: {dump_path}")
        if not dump_path.is_file():
            raise ValueError(f"dump_file must be a file, got: {dump_path}")

        cache_key = cls._scan_cache_key(dump_path)
        cached_scan = _SCAN_RESULT_PROCESS_CACHE.get(cache_key)
        if cached_scan is not None:
            return cached_scan

        scan_from_cache = cls._load_scan_result_from_cache(dump_path, cache_dir=cache_dir)
        if scan_from_cache is not None:
            logger.print(
                "[temporal-lammps] "
                f"Loaded cached scan metadata from {cls._resolve_cache_dir_for_dump(dump_path, cache_dir=cache_dir)}."
            )
            _SCAN_RESULT_PROCESS_CACHE[cache_key] = scan_from_cache
            return scan_from_cache

        frame_count = 0
        num_atoms_expected: int | None = None
        atom_columns_expected: tuple[str, ...] | None = None
        timesteps: list[int] = []
        box_low: list[np.ndarray] = []
        box_high: list[np.ndarray] = []

        logger.print(f"[temporal-lammps] Scanning dump file: {dump_path}")
        with dump_path.open("r", encoding="utf-8") as handle:
            while True:
                header = cls._read_frame_header(handle, source_path=dump_path)
                if header is None:
                    break

                if num_atoms_expected is None:
                    num_atoms_expected = int(header["num_atoms"])
                elif int(header["num_atoms"]) != num_atoms_expected:
                    raise ValueError(
                        "All frames must contain the same number of atoms for tracked temporal neighborhoods. "
                        f"Expected {num_atoms_expected}, got {header['num_atoms']} at frame {frame_count} "
                        f"in {dump_path}."
                    )

                atom_columns = tuple(str(name) for name in header["atom_columns"])
                if atom_columns_expected is None:
                    atom_columns_expected = atom_columns
                elif atom_columns != atom_columns_expected:
                    raise ValueError(
                        "LAMMPS atom columns must remain identical across frames. "
                        f"Expected {atom_columns_expected}, got {atom_columns} at frame {frame_count} "
                        f"in {dump_path}."
                    )

                flat = np.fromfile(
                    handle,
                    dtype=np.float64,
                    count=int(header["num_atoms"]) * len(atom_columns),
                    sep=" ",
                )
                expected_values = int(header["num_atoms"]) * len(atom_columns)
                if flat.size != expected_values:
                    raise ValueError(
                        "Failed to read the full atom block while scanning the dump file. "
                        f"frame={frame_count}, expected_values={expected_values}, got={flat.size}, "
                        f"source_path={dump_path}."
                    )

                timesteps.append(int(header["timestep"]))
                box_low.append(np.asarray(header["box_low"], dtype=np.float32))
                box_high.append(np.asarray(header["box_high"], dtype=np.float32))
                frame_count += 1

        if frame_count == 0:
            raise ValueError(f"No frames were found in LAMMPS dump file: {dump_path}")
        if num_atoms_expected is None or atom_columns_expected is None:
            raise RuntimeError(
                "Dump scan finished without resolving atom metadata. "
                f"source_path={dump_path}, frame_count={frame_count}."
            )

        logger.print(
            "[temporal-lammps] "
            f"Scan complete: frames={frame_count}, num_atoms={num_atoms_expected}, "
            f"atom_columns={list(atom_columns_expected)}."
        )
        result = _DumpScanResult(
            frame_count=frame_count,
            num_atoms=int(num_atoms_expected),
            atom_columns=atom_columns_expected,
            timesteps=np.asarray(timesteps, dtype=np.int64),
            box_low=np.asarray(box_low, dtype=np.float32),
            box_high=np.asarray(box_high, dtype=np.float32),
        )
        _SCAN_RESULT_PROCESS_CACHE[cache_key] = result
        return result

    @classmethod
    def load_dump_frame_positions(
        cls,
        dump_file: str | Path,
        *,
        frame_index: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        dump_path = Path(dump_file).expanduser().resolve()
        if not dump_path.exists():
            raise FileNotFoundError(f"LAMMPS dump file does not exist: {dump_path}")
        if not dump_path.is_file():
            raise ValueError(f"dump_file must be a file, got: {dump_path}")
        if frame_index < 0:
            raise ValueError(f"frame_index must be >= 0, got {frame_index}.")

        with dump_path.open("r", encoding="utf-8") as handle:
            for current_frame_index in range(int(frame_index) + 1):
                header = cls._read_frame_header(handle, source_path=dump_path)
                if header is None:
                    raise IndexError(
                        "Requested frame_index exceeds the number of frames in the LAMMPS dump. "
                        f"frame_index={frame_index}, source_path={dump_path}."
                    )
                atom_columns = tuple(str(name) for name in header["atom_columns"])
                values = np.fromfile(
                    handle,
                    dtype=np.float64,
                    count=int(header["num_atoms"]) * len(atom_columns),
                    sep=" ",
                )
                expected_values = int(header["num_atoms"]) * len(atom_columns)
                if values.size != expected_values:
                    raise ValueError(
                        "Failed to read the full atom block while loading a dump frame. "
                        f"frame_index={current_frame_index}, expected_values={expected_values}, "
                        f"got={values.size}, source_path={dump_path}."
                    )
                if current_frame_index != int(frame_index):
                    continue

                position_columns = cls._resolve_position_columns(atom_columns)
                frame_table = values.reshape(int(header["num_atoms"]), len(atom_columns))
                coords = frame_table[:, position_columns].astype(np.float32, copy=False)
                box_low = np.asarray(header["box_low"], dtype=np.float32)
                box_high = np.asarray(header["box_high"], dtype=np.float32)
                box_lengths = box_high - box_low
                if np.any(box_lengths <= 0.0):
                    raise ValueError(
                        "Encountered invalid box lengths while loading dump frame positions. "
                        f"frame_index={current_frame_index}, box_low={box_low.tolist()}, "
                        f"box_high={box_high.tolist()}, source_path={dump_path}."
                    )
                wrapped = np.mod(coords - box_low[None, :], box_lengths[None, :]).astype(np.float32, copy=False)
                return (
                    _sanitize_periodic_points(wrapped, box_lengths),
                    box_lengths.astype(np.float32, copy=False),
                    int(header["timestep"]),
                )

        raise RuntimeError(
            "Failed to load the requested dump frame even though the file traversal completed. "
            f"frame_index={frame_index}, source_path={dump_path}."
        )

    def _build_cache(self, scan: _DumpScanResult) -> None:
        positions_path = self.cache_dir / "positions.npy"
        positions_memmap = open_memmap(
            positions_path,
            mode="w+",
            dtype=np.float32,
            shape=(scan.frame_count, scan.num_atoms, 3),
        )

        atom_ids: np.ndarray | None = None
        atom_types: np.ndarray | None = None
        position_columns = self._resolve_position_columns(scan.atom_columns)
        id_column = self._resolve_required_column(scan.atom_columns, "id")
        type_column = self._resolve_optional_column(scan.atom_columns, "type")

        logger.print(f"[temporal-lammps] Building binary cache in {self.cache_dir}")
        with self.dump_file.open("r", encoding="utf-8") as handle:
            for frame_idx in range(scan.frame_count):
                header = self._read_frame_header(handle, source_path=self.dump_file)
                if header is None:
                    raise RuntimeError(
                        "Unexpected end of file while building temporal cache. "
                        f"frame_idx={frame_idx}, expected_frame_count={scan.frame_count}, "
                        f"source_path={self.dump_file}."
                    )
                atom_columns = tuple(str(name) for name in header["atom_columns"])
                values = np.fromfile(
                    handle,
                    dtype=np.float64,
                    count=int(header["num_atoms"]) * len(atom_columns),
                    sep=" ",
                )
                expected_values = int(header["num_atoms"]) * len(atom_columns)
                if values.size != expected_values:
                    raise ValueError(
                        "Failed to read the full atom block while building the cache. "
                        f"frame_idx={frame_idx}, expected_values={expected_values}, got={values.size}, "
                        f"source_path={self.dump_file}."
                    )

                frame_table = values.reshape(scan.num_atoms, len(scan.atom_columns))
                frame_ids = frame_table[:, id_column].astype(np.int64, copy=False)
                order = np.argsort(frame_ids, kind="mergesort")
                sorted_ids = frame_ids[order]
                if atom_ids is None:
                    atom_ids = np.array(sorted_ids, dtype=np.int64, copy=True)
                elif not np.array_equal(sorted_ids, atom_ids):
                    raise ValueError(
                        "Atom ids changed across frames, so tracked-atom temporal neighborhoods are not well-defined. "
                        f"frame_idx={frame_idx}, source_path={self.dump_file}."
                    )

                if type_column is not None:
                    sorted_types = frame_table[:, type_column].astype(np.int32, copy=False)[order]
                    if atom_types is None:
                        atom_types = np.array(sorted_types, dtype=np.int32, copy=True)
                    elif not np.array_equal(sorted_types, atom_types):
                        raise ValueError(
                            "Atom types changed across frames for the same atom ids. "
                            f"frame_idx={frame_idx}, source_path={self.dump_file}."
                        )

                coords = frame_table[:, position_columns].astype(np.float32, copy=False)[order]
                box_low = scan.box_low[frame_idx]
                box_high = scan.box_high[frame_idx]
                box_lengths = box_high - box_low
                if np.any(box_lengths <= 0.0):
                    raise ValueError(
                        "Encountered invalid box lengths while building the temporal cache. "
                        f"frame_idx={frame_idx}, box_low={box_low.tolist()}, box_high={box_high.tolist()}, "
                        f"source_path={self.dump_file}."
                    )
                wrapped = np.mod(coords - box_low[None, :], box_lengths[None, :]).astype(np.float32, copy=False)
                positions_memmap[frame_idx] = _sanitize_periodic_points(wrapped, box_lengths)

                if frame_idx == 0 or (frame_idx + 1) % 10 == 0 or (frame_idx + 1) == scan.frame_count:
                    logger.print(
                        "[temporal-lammps] "
                        f"Cached frame {frame_idx + 1}/{scan.frame_count} "
                        f"(timestep={scan.timesteps[frame_idx]})."
                    )

        positions_memmap.flush()
        if atom_ids is None:
            raise RuntimeError(
                "Temporal cache build finished without atom ids. "
                f"source_path={self.dump_file}, cache_dir={self.cache_dir}."
            )

        np.save(self.cache_dir / "atom_ids.npy", atom_ids)
        if atom_types is not None:
            np.save(self.cache_dir / "atom_types.npy", atom_types)
        np.save(self.cache_dir / "timesteps.npy", scan.timesteps)
        np.save(self.cache_dir / "box_low.npy", scan.box_low)
        np.save(self.cache_dir / "box_high.npy", scan.box_high)

        stat = self.dump_file.stat()
        manifest = {
            "cache_version": self.cache_version,
            "source_path": str(self.dump_file),
            "source_size_bytes": int(stat.st_size),
            "source_mtime_ns": int(stat.st_mtime_ns),
            "frame_count": int(scan.frame_count),
            "num_atoms": int(scan.num_atoms),
            "atom_columns": list(scan.atom_columns),
            "position_columns": list(position_columns),
            "has_type_column": bool(atom_types is not None),
        }
        with (self.cache_dir / "manifest.json").open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

    def _load_cache(self) -> None:
        with (self.cache_dir / "manifest.json").open("r", encoding="utf-8") as handle:
            self._manifest = json.load(handle)

        self.positions = np.load(self.cache_dir / "positions.npy", mmap_mode="r")
        self.atom_ids = np.load(self.cache_dir / "atom_ids.npy", mmap_mode="r")
        atom_types_path = self.cache_dir / "atom_types.npy"
        self.atom_types = np.load(atom_types_path, mmap_mode="r") if atom_types_path.exists() else None
        self.timesteps = np.load(self.cache_dir / "timesteps.npy", mmap_mode="r")
        self.box_low = np.load(self.cache_dir / "box_low.npy", mmap_mode="r")
        self.box_high = np.load(self.cache_dir / "box_high.npy", mmap_mode="r")
        self.box_lengths = self.box_high - self.box_low

        expected_positions_shape = (self.frame_count, self.num_atoms, 3)
        if tuple(self.positions.shape) != expected_positions_shape:
            raise ValueError(
                "Cached positions shape does not match manifest metadata. "
                f"expected={expected_positions_shape}, got={tuple(self.positions.shape)}, "
                f"cache_dir={self.cache_dir}."
            )
        if tuple(self.atom_ids.shape) != (self.num_atoms,):
            raise ValueError(
                "Cached atom_ids shape does not match manifest metadata. "
                f"expected={(self.num_atoms,)}, got={tuple(self.atom_ids.shape)}, "
                f"cache_dir={self.cache_dir}."
            )
        if tuple(self.timesteps.shape) != (self.frame_count,):
            raise ValueError(
                "Cached timesteps shape does not match manifest metadata. "
                f"expected={(self.frame_count,)}, got={tuple(self.timesteps.shape)}, "
                f"cache_dir={self.cache_dir}."
            )
        if tuple(self.box_low.shape) != (self.frame_count, 3) or tuple(self.box_high.shape) != (self.frame_count, 3):
            raise ValueError(
                "Cached box bounds have invalid shapes. "
                f"box_low.shape={tuple(self.box_low.shape)}, box_high.shape={tuple(self.box_high.shape)}, "
                f"cache_dir={self.cache_dir}."
            )

    def _resolve_center_atom_indices(
        self,
        *,
        center_selection_mode: str | None,
        center_atom_ids: Sequence[int] | None,
        center_atom_stride: int | None,
        max_center_atoms: int | None,
    ) -> np.ndarray:
        mode = self._resolve_center_selection_mode(
            center_selection_mode=center_selection_mode,
            center_atom_ids=center_atom_ids,
            center_atom_stride=center_atom_stride,
            max_center_atoms=max_center_atoms,
        )

        if mode == "atom_ids":
            if center_atom_ids is None:
                raise RuntimeError("Internal error: atom_ids mode resolved without center_atom_ids.")
            requested = np.asarray(center_atom_ids, dtype=np.int64)
            if requested.ndim != 1:
                raise ValueError(
                    f"center_atom_ids must be a 1D sequence of atom ids, got shape {requested.shape}."
                )
            positions = np.searchsorted(self.atom_ids, requested)
            valid = (positions >= 0) & (positions < self.num_atoms)
            if not np.all(valid):
                missing = requested[~valid]
                raise ValueError(
                    "Some requested center atom ids were outside the cached atom id range. "
                    f"missing={missing.tolist()}, source_path={self.dump_file}."
                )
            matched = self.atom_ids[positions]
            exact_match = matched == requested
            if not np.all(exact_match):
                missing = requested[~exact_match]
                raise ValueError(
                    "Some requested center atom ids were not found in the cached atom list. "
                    f"missing={missing.tolist()}, source_path={self.dump_file}."
                )
            return positions.astype(np.int64, copy=False)

        if mode == "atom_stride":
            if center_atom_stride is None:
                raise RuntimeError("Internal error: atom_stride mode resolved without center_atom_stride.")
            stride = int(center_atom_stride)
            if stride <= 0:
                raise ValueError(f"center_atom_stride must be > 0, got {stride}")
            return np.arange(0, self.num_atoms, stride, dtype=np.int64)

        if mode == "regular_grid":
            return self._resolve_regular_grid_center_atom_indices()

        if mode != "random_subset":
            raise RuntimeError(f"Unsupported center selection mode resolved: {mode!r}.")

        if max_center_atoms is None:
            raise RuntimeError("Internal error: random_subset mode resolved without max_center_atoms.")
        count = int(max_center_atoms)
        if count <= 0:
            raise ValueError(f"max_center_atoms must be > 0, got {count}")
        count = min(count, self.num_atoms)
        rng = np.random.default_rng(self.center_selection_seed)
        picked = rng.choice(self.num_atoms, size=count, replace=False)
        return np.sort(picked.astype(np.int64, copy=False))

    def _resolve_center_selection_mode(
        self,
        *,
        center_selection_mode: str | None,
        center_atom_ids: Sequence[int] | None,
        center_atom_stride: int | None,
        max_center_atoms: int | None,
    ) -> str:
        if center_selection_mode is None:
            specified_modes = sum(
                option is not None
                for option in (center_atom_ids, center_atom_stride, max_center_atoms)
            )
            if specified_modes == 0:
                raise ValueError(
                    "Explicit center atom selection is required for TemporalLAMMPSDumpDataset. "
                    "Set center_selection_mode='regular_grid' for periodic grid sampling, or set one of "
                    "center_atom_ids, center_atom_stride, or max_center_atoms."
                )
            if specified_modes > 1:
                raise ValueError(
                    "Use exactly one legacy center atom selection mode when center_selection_mode is omitted. "
                    f"Got center_atom_ids={center_atom_ids is not None}, "
                    f"center_atom_stride={center_atom_stride}, max_center_atoms={max_center_atoms}."
                )
            if center_atom_ids is not None:
                return "atom_ids"
            if center_atom_stride is not None:
                return "atom_stride"
            return "random_subset"

        mode = str(center_selection_mode).strip().lower()
        if mode not in {"atom_ids", "atom_stride", "random_subset", "regular_grid"}:
            raise ValueError(
                "center_selection_mode must be one of "
                "['atom_ids', 'atom_stride', 'random_subset', 'regular_grid'], "
                f"got {center_selection_mode!r}."
            )
        if mode == "atom_ids":
            if center_atom_ids is None:
                raise ValueError("center_selection_mode='atom_ids' requires center_atom_ids to be set.")
            if center_atom_stride is not None or max_center_atoms is not None:
                raise ValueError(
                    "center_selection_mode='atom_ids' is incompatible with center_atom_stride, "
                    "and max_center_atoms."
                )
        elif mode == "atom_stride":
            if center_atom_stride is None:
                raise ValueError("center_selection_mode='atom_stride' requires center_atom_stride to be set.")
            if center_atom_ids is not None or max_center_atoms is not None:
                raise ValueError(
                    "center_selection_mode='atom_stride' is incompatible with center_atom_ids, "
                    "and max_center_atoms."
                )
        elif mode == "random_subset":
            if max_center_atoms is None:
                raise ValueError("center_selection_mode='random_subset' requires max_center_atoms to be set.")
            if center_atom_ids is not None or center_atom_stride is not None:
                raise ValueError(
                    "center_selection_mode='random_subset' is incompatible with center_atom_ids, "
                    "and center_atom_stride."
                )
        else:
            if center_atom_ids is not None or center_atom_stride is not None:
                raise ValueError(
                    "center_selection_mode='regular_grid' is incompatible with center_atom_ids "
                    "and center_atom_stride."
                )
            if max_center_atoms is not None:
                raise ValueError(
                    "center_selection_mode='regular_grid' is incompatible with max_center_atoms. "
                    "Use center_grid_overlap to control temporal regular-grid density."
                )
            if self.radius is None:
                raise ValueError(
                    "center_selection_mode='regular_grid' requires radius to be set so the grid spacing "
                    "can be derived from local-structure size."
                )
            if self.center_grid_overlap is None:
                raise ValueError(
                    "center_selection_mode='regular_grid' requires center_grid_overlap to be set."
                )
        return mode

    def _resolve_regular_grid_center_atom_indices(
        self,
    ) -> np.ndarray:
        if self.radius is None:
            raise RuntimeError("Regular-grid center selection requires radius, but radius is None.")
        if self.center_grid_overlap is None:
            raise RuntimeError(
                "Regular-grid center selection requires center_grid_overlap, but it is None."
            )
        reference_frame_idx = self._resolve_center_grid_reference_frame_index()
        box_lengths = np.asarray(self.box_lengths[reference_frame_idx], dtype=np.float64)
        if np.any(box_lengths <= 0.0):
            raise ValueError(
                "Regular-grid center selection encountered non-positive box lengths. "
                f"reference_frame_idx={reference_frame_idx}, box_lengths={box_lengths.tolist()}."
            )

        desired_stride = (2.0 - float(self.center_grid_overlap)) * float(self.radius)
        if desired_stride <= 0.0:
            raise ValueError(
                "Regular-grid center selection requires positive grid stride. "
                f"radius={self.radius}, center_grid_overlap={self.center_grid_overlap}, "
                f"derived_stride={desired_stride}."
            )

        counts = np.maximum(1, np.floor(box_lengths / desired_stride).astype(np.int64))
        if np.prod(counts.astype(np.int64), dtype=np.int64) <= 0:
            raise RuntimeError(
                "Regular-grid center selection produced a non-positive grid size. "
                f"counts={counts.tolist()}, reference_frame_idx={reference_frame_idx}."
            )

        actual_spacing = box_lengths / counts.astype(np.float64)
        grid_axes = [np.arange(int(count), dtype=np.float64) for count in counts.tolist()]
        mesh = np.meshgrid(*grid_axes, indexing="ij")
        grid_ijk = np.column_stack([axis.ravel() for axis in mesh]).astype(np.float64, copy=False)
        grid_centers = (grid_ijk + 0.5) * actual_spacing[None, :]

        tree = self._get_tree(reference_frame_idx)
        _, nearest_indices = tree.query(grid_centers.astype(np.float32, copy=False), k=1)
        return _stable_unique_int(nearest_indices)

    def _resolve_center_grid_reference_frame_index(self) -> int:
        if self.center_grid_reference_frame_index is not None:
            reference_frame_idx = int(self.center_grid_reference_frame_index)
        elif self.anchor_frame_indices is not None and self.anchor_frame_indices.size > 0:
            reference_frame_idx = int(np.asarray(self.anchor_frame_indices, dtype=np.int64).reshape(-1)[0])
        else:
            reference_frame_idx = int(self.frame_start)
        if reference_frame_idx < 0 or reference_frame_idx >= self.frame_count:
            raise ValueError(
                "center_grid_reference_frame_index is out of range. "
                f"reference_frame_idx={reference_frame_idx}, frame_count={self.frame_count}."
            )
        return reference_frame_idx

    def _resolve_window_start_frames(self) -> np.ndarray:
        if self.anchor_frame_indices is not None:
            anchors = np.asarray(self.anchor_frame_indices, dtype=np.int64).reshape(-1)
            if anchors.size == 0:
                return np.zeros((0,), dtype=np.int64)
            if np.any(anchors < 0):
                raise ValueError(
                    f"anchor_frame_indices must be >= 0, got {anchors.tolist()}."
                )
            max_start = self.frame_count - (self.sequence_length - 1) * self.frame_stride - 1
            if max_start < 0:
                return np.zeros((0,), dtype=np.int64)
            if np.any(anchors > max_start):
                raise ValueError(
                    "Some anchor_frame_indices do not leave enough trailing frames for the requested sequence. "
                    f"max_valid_start={max_start}, anchor_frame_indices={anchors.tolist()}, "
                    f"sequence_length={self.sequence_length}, frame_stride={self.frame_stride}."
                )
            unique_anchors = np.unique(anchors.astype(np.int64, copy=False))
            return np.sort(unique_anchors)

        stop = self.frame_count if self.frame_stop is None else self.frame_stop
        if stop <= self.frame_start:
            raise ValueError(
                f"frame_stop must be > frame_start, got frame_start={self.frame_start}, frame_stop={stop}."
            )
        last_required_frame = self.frame_start + (self.sequence_length - 1) * self.frame_stride
        if last_required_frame >= stop:
            return np.zeros((0,), dtype=np.int64)
        max_start = stop - (self.sequence_length - 1) * self.frame_stride
        return np.arange(self.frame_start, max_start, self.window_stride, dtype=np.int64)

    def _resolve_window_source_names(self) -> list[str]:
        if self.anchor_source_names is not None:
            if len(self.anchor_source_names) != int(self._window_start_frames.size):
                raise ValueError(
                    "anchor_source_names must match the number of resolved window start frames. "
                    f"len(anchor_source_names)={len(self.anchor_source_names)}, "
                    f"num_windows={int(self._window_start_frames.size)}."
                )
            return [str(v) for v in self.anchor_source_names]
        return [
            self._default_window_source_name(int(frame_idx))
            for frame_idx in self._window_start_frames.tolist()
        ]

    def _default_window_source_name(self, frame_idx: int) -> str:
        timestep = int(self.timesteps[int(frame_idx)])
        return f"frame_{int(frame_idx):03d}_t{timestep}"

    def _get_tree(self, frame_idx: int) -> cKDTree:
        tree = self._tree_cache.get(frame_idx)
        if tree is not None:
            self._tree_cache.move_to_end(frame_idx)
            return tree

        points = _sanitize_periodic_points(self.positions[frame_idx], self.box_lengths[frame_idx])
        box_lengths = np.asarray(self.box_lengths[frame_idx], dtype=np.float32)
        tree = cKDTree(points, boxsize=box_lengths, balanced_tree=False)
        self._tree_cache[frame_idx] = tree
        while len(self._tree_cache) > self.tree_cache_size:
            self._tree_cache.popitem(last=False)
        return tree

    def _query_local_structure(self, *, frame_idx: int, center: np.ndarray) -> np.ndarray:
        tree = self._get_tree(frame_idx)
        distances, indices = tree.query(center, k=self.num_points)
        distances = np.atleast_1d(np.asarray(distances, dtype=np.float32))
        indices = np.atleast_1d(np.asarray(indices, dtype=np.int64))
        if indices.size != self.num_points:
            raise RuntimeError(
                "KDTree query returned an unexpected number of neighbors. "
                f"frame_idx={frame_idx}, expected={self.num_points}, got={indices.size}, "
                f"source_path={self.dump_file}."
            )

        if self.selection_method == "radius_then_closest":
            assert self.radius is not None
            within = indices[distances <= self.radius]
            if within.size >= self.num_points:
                return within[: self.num_points]
        return indices

    def _query_local_structures(
        self,
        *,
        frame_idx: int,
        centers: np.ndarray,
    ) -> np.ndarray:
        tree = self._get_tree(frame_idx)
        distances, indices = tree.query(centers, k=self.num_points)
        distances = np.asarray(distances, dtype=np.float32)
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim == 1:
            indices = indices.reshape(1, -1)
            distances = distances.reshape(1, -1)
        if indices.shape != (int(centers.shape[0]), self.num_points):
            raise RuntimeError(
                "KDTree batch query returned an unexpected neighbor array shape. "
                f"frame_idx={frame_idx}, centers_shape={tuple(centers.shape)}, "
                f"expected_shape={(int(centers.shape[0]), self.num_points)}, "
                f"got_shape={tuple(indices.shape)}, source_path={self.dump_file}."
            )

        if self.selection_method == "radius_then_closest":
            assert self.radius is not None
            selected = np.empty_like(indices)
            for row_idx in range(indices.shape[0]):
                within = indices[row_idx, distances[row_idx] <= self.radius]
                if within.size >= self.num_points:
                    selected[row_idx] = within[: self.num_points]
                else:
                    selected[row_idx] = indices[row_idx]
            return selected
        return indices

    def _to_local_coordinates(
        self,
        *,
        frame_idx: int,
        points: np.ndarray,
        center: np.ndarray,
    ) -> np.ndarray:
        if not self.center_neighborhoods:
            return points.astype(np.float32, copy=False)
        rel = points - center[None, :]
        box_lengths = np.asarray(self.box_lengths[frame_idx], dtype=np.float32)
        rel = rel - box_lengths[None, :] * np.round(rel / box_lengths[None, :])
        return rel.astype(np.float32, copy=False)

    def _to_local_coordinates_batch(
        self,
        *,
        frame_idx: int,
        points: np.ndarray,
        centers: np.ndarray,
    ) -> np.ndarray:
        if not self.center_neighborhoods:
            return points.astype(np.float32, copy=False)
        rel = points - centers[:, None, :]
        box_lengths = np.asarray(self.box_lengths[frame_idx], dtype=np.float32)
        rel = rel - box_lengths[None, None, :] * np.round(rel / box_lengths[None, None, :])
        return rel.astype(np.float32, copy=False)

    def _normalize_point_cloud_batch(self, points: np.ndarray) -> np.ndarray:
        if self.radius is None:
            raise RuntimeError(
                "Temporal local-structure batch normalization requires an explicit cutoff radius, "
                f"but dataset.radius is None for dump_file={self.dump_file}."
            )
        return points / float(self.radius)

    def _build_batch_from_indices(self, indices: np.ndarray) -> dict[str, Any]:
        index_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        if index_array.size == 0:
            raise ValueError("Temporal batch indices must be non-empty.")
        if np.any(index_array < 0) or np.any(index_array >= len(self)):
            raise IndexError(
                "Temporal batch indices are out of range. "
                f"min_index={int(index_array.min())}, max_index={int(index_array.max())}, len={len(self)}."
            )

        batch_size = int(index_array.size)
        sequence_points = np.empty(
            (batch_size, self.sequence_length, self.num_points, 3),
            dtype=np.float32,
        )
        local_atom_ids_batch = np.empty(
            (batch_size, self.sequence_length, self.num_points),
            dtype=np.int64,
        )
        center_positions = np.empty((batch_size, self.sequence_length, 3), dtype=np.float32)
        timesteps_batch = np.empty((batch_size, self.sequence_length), dtype=np.int64)
        frame_indices_batch = np.empty((batch_size, self.sequence_length), dtype=np.int64)
        center_atom_ids = np.empty((batch_size,), dtype=np.int64)
        anchor_frame_indices = np.empty((batch_size,), dtype=np.int64)
        anchor_timesteps = np.empty((batch_size,), dtype=np.int64)
        precomputed_neighbor_indices = self._load_precomputed_neighbor_indices()

        window_slots = (index_array // self.center_count).astype(np.int64, copy=False)
        center_slots = (index_array % self.center_count).astype(np.int64, copy=False)

        grouped_positions: dict[int, list[int]] = {}
        for batch_pos, window_slot in enumerate(window_slots.tolist()):
            grouped_positions.setdefault(int(window_slot), []).append(int(batch_pos))

        frame_offsets = np.arange(self.sequence_length, dtype=np.int64) * self.frame_stride
        for window_slot, batch_positions_list in grouped_positions.items():
            batch_positions = np.asarray(batch_positions_list, dtype=np.int64)
            start_frame = int(self._window_start_frames[window_slot])
            frame_indices = start_frame + frame_offsets
            timesteps = self.timesteps[frame_indices].astype(np.int64, copy=False)
            center_atom_indices = np.asarray(
                self._center_atom_indices[center_slots[batch_positions]],
                dtype=np.int64,
            )
            center_atom_ids_window = np.asarray(self.atom_ids[center_atom_indices], dtype=np.int64)

            frame_indices_batch[batch_positions] = frame_indices[None, :]
            timesteps_batch[batch_positions] = timesteps[None, :]
            center_atom_ids[batch_positions] = center_atom_ids_window
            anchor_frame_indices[batch_positions] = int(frame_indices[0])
            anchor_timesteps[batch_positions] = int(timesteps[0])

            for local_frame_idx, frame_idx in enumerate(frame_indices.tolist()):
                frame_points = np.asarray(self.positions[frame_idx], dtype=np.float32)
                centers = np.asarray(frame_points[center_atom_indices], dtype=np.float32)
                if precomputed_neighbor_indices is None:
                    selected = self._query_local_structures(frame_idx=frame_idx, centers=centers)
                else:
                    selected = np.asarray(
                        precomputed_neighbor_indices[frame_idx, center_slots[batch_positions]],
                        dtype=np.int64,
                    )
                local_points = np.asarray(frame_points[selected], dtype=np.float32)
                local_points = self._to_local_coordinates_batch(
                    frame_idx=frame_idx,
                    points=local_points,
                    centers=centers,
                )
                if self.normalize:
                    local_points = self._normalize_point_cloud_batch(local_points).astype(np.float32, copy=False)
                sequence_points[batch_positions, local_frame_idx] = local_points
                local_atom_ids_batch[batch_positions, local_frame_idx] = np.asarray(
                    self.atom_ids[selected],
                    dtype=np.int64,
                )
                center_positions[batch_positions, local_frame_idx] = centers + self.box_low[frame_idx]

        return {
            "points": torch.from_numpy(sequence_points),
            "local_atom_ids": torch.from_numpy(local_atom_ids_batch),
            "coords": torch.from_numpy(center_positions[:, 0].copy()),
            "center_positions": torch.from_numpy(center_positions),
            "timesteps": torch.from_numpy(timesteps_batch),
            "frame_indices": torch.from_numpy(frame_indices_batch),
            "center_atom_id": torch.from_numpy(center_atom_ids),
            "instance_id": torch.from_numpy(center_atom_ids.copy()),
            "anchor_frame_index": torch.from_numpy(anchor_frame_indices),
            "anchor_timestep": torch.from_numpy(anchor_timesteps),
            "source_path": [str(self.dump_file)] * batch_size,
        }

    @staticmethod
    def _resolve_position_columns(atom_columns: Sequence[str]) -> tuple[int, int, int]:
        for names in _SUPPORTED_POSITION_COLUMNS:
            if all(name in atom_columns for name in names):
                return tuple(int(atom_columns.index(name)) for name in names)
        raise ValueError(
            "LAMMPS dump must provide one of the supported coordinate column sets "
            f"{_SUPPORTED_POSITION_COLUMNS}, got atom_columns={list(atom_columns)}."
        )

    @staticmethod
    def _resolve_required_column(atom_columns: Sequence[str], name: str) -> int:
        if name not in atom_columns:
            raise ValueError(
                f"LAMMPS dump is missing required atom column {name!r}. "
                f"Available columns: {list(atom_columns)}."
            )
        return int(atom_columns.index(name))

    @staticmethod
    def _resolve_optional_column(atom_columns: Sequence[str], name: str) -> int | None:
        if name not in atom_columns:
            return None
        return int(atom_columns.index(name))

    @classmethod
    def _read_frame_header(
        cls,
        handle,
        *,
        source_path: Path,
    ) -> dict[str, Any] | None:
        line = handle.readline()
        if line == "":
            return None
        if not line.startswith("ITEM: TIMESTEP"):
            raise ValueError(
                "Expected 'ITEM: TIMESTEP' while parsing LAMMPS dump. "
                f"Got line={line.strip()!r}, source_path={source_path}."
            )

        timestep_line = handle.readline()
        if timestep_line == "":
            raise ValueError(
                "Unexpected EOF after 'ITEM: TIMESTEP'. "
                f"source_path={source_path}."
            )
        timestep = int(timestep_line.strip())

        number_header = handle.readline()
        if not number_header.startswith("ITEM: NUMBER OF ATOMS"):
            raise ValueError(
                "Expected 'ITEM: NUMBER OF ATOMS' in LAMMPS dump header. "
                f"Got line={number_header.strip()!r}, timestep={timestep}, source_path={source_path}."
            )

        num_atoms_line = handle.readline()
        if num_atoms_line == "":
            raise ValueError(
                "Unexpected EOF after 'ITEM: NUMBER OF ATOMS'. "
                f"timestep={timestep}, source_path={source_path}."
            )
        num_atoms = int(num_atoms_line.strip())

        box_header = handle.readline()
        if not box_header.startswith("ITEM: BOX BOUNDS"):
            raise ValueError(
                "Expected 'ITEM: BOX BOUNDS' in LAMMPS dump header. "
                f"Got line={box_header.strip()!r}, timestep={timestep}, source_path={source_path}."
            )
        box_mode_tokens = box_header.strip().split()[3:]

        box_low = np.empty((3,), dtype=np.float32)
        box_high = np.empty((3,), dtype=np.float32)
        for axis in range(3):
            bounds_line = handle.readline()
            if bounds_line == "":
                raise ValueError(
                    "Unexpected EOF while reading box bounds. "
                    f"axis={axis}, timestep={timestep}, source_path={source_path}."
                )
            parts = bounds_line.strip().split()
            if len(parts) != 2:
                raise NotImplementedError(
                    "Only orthorhombic LAMMPS boxes with two bounds values per axis are supported. "
                    f"Got bounds_line={bounds_line.strip()!r}, box_mode_tokens={box_mode_tokens}, "
                    f"timestep={timestep}, source_path={source_path}."
                )
            box_low[axis] = float(parts[0])
            box_high[axis] = float(parts[1])

        atoms_header = handle.readline()
        if not atoms_header.startswith("ITEM: ATOMS "):
            raise ValueError(
                "Expected 'ITEM: ATOMS ...' in LAMMPS dump header. "
                f"Got line={atoms_header.strip()!r}, timestep={timestep}, source_path={source_path}."
            )
        atom_columns = tuple(atoms_header.strip().split()[2:])
        if not atom_columns:
            raise ValueError(
                "LAMMPS 'ITEM: ATOMS' header did not include any atom columns. "
                f"timestep={timestep}, source_path={source_path}."
            )

        return {
            "timestep": timestep,
            "num_atoms": num_atoms,
            "box_low": box_low,
            "box_high": box_high,
            "atom_columns": atom_columns,
        }
