from __future__ import annotations

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


def _normalize_point_cloud(points: np.ndarray, radius: float | None) -> np.ndarray:
    if radius is not None:
        return points / float(radius)
    max_norm = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_norm <= 0.0:
        raise ValueError(
            "Cannot normalize a local structure with zero spatial extent. "
            f"points_shape={points.shape}, max_norm={max_norm}."
        )
    return points / max_norm


@dataclass(frozen=True)
class _DumpScanResult:
    frame_count: int
    num_atoms: int
    atom_columns: tuple[str, ...]
    timesteps: np.ndarray
    box_low: np.ndarray
    box_high: np.ndarray


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
        center_atom_ids: Sequence[int] | None = None,
        center_atom_stride: int | None = None,
        max_center_atoms: int | None = None,
        center_selection_seed: int = 0,
        normalize: bool = True,
        center_neighborhoods: bool = True,
        selection_method: str = "closest",
        cache_dir: str | Path | None = None,
        rebuild_cache: bool = False,
        tree_cache_size: int = 4,
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
        self.normalize = bool(normalize)
        self.center_neighborhoods = bool(center_neighborhoods)
        self.center_selection_seed = int(center_selection_seed)
        self.tree_cache_size = int(tree_cache_size)
        self.build_lock_timeout_sec = float(build_lock_timeout_sec)
        self.build_lock_stale_sec = float(build_lock_stale_sec)

        if self.sequence_length <= 0:
            raise ValueError(f"sequence_length must be > 0, got {self.sequence_length}")
        if self.num_points <= 0:
            raise ValueError(f"num_points must be > 0, got {self.num_points}")
        if self.radius is not None and self.radius <= 0.0:
            raise ValueError(f"radius must be > 0 when provided, got {self.radius}")
        if self.frame_stride <= 0:
            raise ValueError(f"frame_stride must be > 0, got {self.frame_stride}")
        if self.window_stride <= 0:
            raise ValueError(f"window_stride must be > 0, got {self.window_stride}")
        if self.frame_start < 0:
            raise ValueError(f"frame_start must be >= 0, got {self.frame_start}")
        if self.tree_cache_size <= 0:
            raise ValueError(f"tree_cache_size must be > 0, got {self.tree_cache_size}")

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

        self._center_atom_indices = self._resolve_center_atom_indices(
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

        self._tree_cache: OrderedDict[int, cKDTree] = OrderedDict()

        logger.print(
            "[temporal-lammps] "
            f"Loaded dataset from {self.dump_file} with "
            f"{self.frame_count} frames, {self.num_atoms} atoms, "
            f"{self._center_atom_indices.size} tracked centers, "
            f"{self._window_start_frames.size} windows, total_samples={len(self)}."
        )

    def __len__(self) -> int:
        return int(self._window_start_frames.size) * int(self._center_atom_indices.size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if index < 0 or index >= len(self):
            raise IndexError(f"index out of range: index={index}, len={len(self)}")

        window_slot = int(index) // int(self._center_atom_indices.size)
        center_slot = int(index) % int(self._center_atom_indices.size)
        start_frame = int(self._window_start_frames[window_slot])
        center_atom_index = int(self._center_atom_indices[center_slot])
        center_atom_id = int(self.atom_ids[center_atom_index])

        frame_indices = start_frame + np.arange(self.sequence_length, dtype=np.int64) * self.frame_stride
        sequence_points = np.empty((self.sequence_length, self.num_points, 3), dtype=np.float32)
        center_positions = np.empty((self.sequence_length, 3), dtype=np.float32)
        timesteps = self.timesteps[frame_indices].astype(np.int64, copy=False)

        for local_frame_idx, frame_idx in enumerate(frame_indices.tolist()):
            frame_points = self.positions[frame_idx]
            center = np.asarray(frame_points[center_atom_index], dtype=np.float32)
            selected = self._query_local_structure(frame_idx=frame_idx, center=center)
            local_points = np.asarray(frame_points[selected], dtype=np.float32)
            local_points = self._to_local_coordinates(
                frame_idx=frame_idx,
                points=local_points,
                center=center,
            )
            if self.normalize:
                local_points = _normalize_point_cloud(local_points, self.radius).astype(np.float32, copy=False)
            sequence_points[local_frame_idx] = local_points
            center_positions[local_frame_idx] = center + self.box_low[frame_idx]

        return {
            "points": torch.from_numpy(sequence_points),
            "center_positions": torch.from_numpy(center_positions),
            "timesteps": torch.as_tensor(timesteps, dtype=torch.int64),
            "frame_indices": torch.as_tensor(frame_indices, dtype=torch.int64),
            "center_atom_id": torch.as_tensor(center_atom_id, dtype=torch.int64),
            "source_path": str(self.dump_file),
        }

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_tree_cache"] = OrderedDict()
        return state

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
    def scan_dump_file(cls, dump_file: str | Path) -> _DumpScanResult:
        dump_path = Path(dump_file).expanduser().resolve()
        if not dump_path.exists():
            raise FileNotFoundError(f"LAMMPS dump file does not exist: {dump_path}")
        if not dump_path.is_file():
            raise ValueError(f"dump_file must be a file, got: {dump_path}")

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
        return _DumpScanResult(
            frame_count=frame_count,
            num_atoms=int(num_atoms_expected),
            atom_columns=atom_columns_expected,
            timesteps=np.asarray(timesteps, dtype=np.int64),
            box_low=np.asarray(box_low, dtype=np.float32),
            box_high=np.asarray(box_high, dtype=np.float32),
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
                positions_memmap[frame_idx] = np.mod(coords - box_low[None, :], box_lengths[None, :]).astype(
                    np.float32,
                    copy=False,
                )

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
        center_atom_ids: Sequence[int] | None,
        center_atom_stride: int | None,
        max_center_atoms: int | None,
    ) -> np.ndarray:
        specified_modes = sum(
            option is not None
            for option in (center_atom_ids, center_atom_stride, max_center_atoms)
        )
        if specified_modes == 0:
            raise ValueError(
                "Explicit center atom selection is required for TemporalLAMMPSDumpDataset. "
                "Set one of center_atom_ids, center_atom_stride, or max_center_atoms to avoid "
                "accidentally creating billions of samples."
            )
        if specified_modes > 1:
            raise ValueError(
                "Use exactly one center atom selection mode. "
                f"Got center_atom_ids={center_atom_ids is not None}, "
                f"center_atom_stride={center_atom_stride}, max_center_atoms={max_center_atoms}."
            )

        if center_atom_ids is not None:
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

        if center_atom_stride is not None:
            stride = int(center_atom_stride)
            if stride <= 0:
                raise ValueError(f"center_atom_stride must be > 0, got {stride}")
            return np.arange(0, self.num_atoms, stride, dtype=np.int64)

        count = int(max_center_atoms)
        if count <= 0:
            raise ValueError(f"max_center_atoms must be > 0, got {count}")
        count = min(count, self.num_atoms)
        rng = np.random.default_rng(self.center_selection_seed)
        picked = rng.choice(self.num_atoms, size=count, replace=False)
        return np.sort(picked.astype(np.int64, copy=False))

    def _resolve_window_start_frames(self) -> np.ndarray:
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

    def _get_tree(self, frame_idx: int) -> cKDTree:
        tree = self._tree_cache.get(frame_idx)
        if tree is not None:
            self._tree_cache.move_to_end(frame_idx)
            return tree

        points = np.asarray(self.positions[frame_idx], dtype=np.float32)
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
