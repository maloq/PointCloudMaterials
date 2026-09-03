"""Memory-mapped binary storage for repository-produced shooting trajectories."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.lib.format import open_memmap

from src.data_utils.shooting_dataset import (
    ShootingFrame,
    ShootingPositionFrame,
)
from src.data_utils.shooting_text_conversion import (
    load_lammps_shooting_frames_for_conversion,
)


FORMAT_NAME = "pointcloudmaterials.shooting_trajectory"
SCHEMA_VERSION = 1
STORAGE_DTYPES = ("float32", "float16")
_ARRAY_FILES = {
    "positions": "positions.npy",
    "velocities": "velocities.npy",
    "timesteps": "timesteps.npy",
    "box_low": "box_low.npy",
    "box_high": "box_high.npy",
    "atom_ids": "atom_ids.npy",
    "atom_types": "atom_types.npy",
}


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required shooting-binary manifest is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(
            f"Expected a JSON object in shooting-binary manifest {path}, "
            f"got {type(value).__name__}."
        )
    return value


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    if values.ndim <= 1:
        digest.update(np.ascontiguousarray(values).tobytes())
    else:
        for index in range(values.shape[0]):
            digest.update(np.ascontiguousarray(values[index]).tobytes())
    return digest.hexdigest()


def _array_description(
    values: np.ndarray, filename: str, *, sha256: str
) -> dict[str, Any]:
    return {
        "file": filename,
        "dtype": values.dtype.name,
        "shape": list(values.shape),
        "sha256": sha256,
    }


@dataclass(frozen=True)
class ShootingBinaryTrajectory:
    """Validated, memory-mapped view of one binary shooting trajectory."""

    root: Path
    manifest: dict[str, Any]
    positions: np.ndarray
    velocities: np.ndarray
    timesteps: np.ndarray
    box_low: np.ndarray
    box_high: np.ndarray
    atom_ids: np.ndarray
    atom_types: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ShootingBinaryTrajectory":
        root = Path(path).expanduser().resolve()
        manifest = _load_json_object(root / "manifest.json")
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError(
                f"Unsupported shooting-binary format in {root / 'manifest.json'}: "
                f"expected={FORMAT_NAME!r}, observed={manifest.get('format')!r}."
            )
        if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported shooting-binary schema in {root / 'manifest.json'}: "
                f"expected={SCHEMA_VERSION}, observed={manifest.get('schema_version')!r}."
            )
        if manifest.get("state") != "complete":
            raise RuntimeError(
                f"Shooting-binary trajectory is not complete: root={root}, "
                f"state={manifest.get('state')!r}."
            )

        arrays: dict[str, np.ndarray] = {}
        descriptions = manifest.get("arrays")
        if not isinstance(descriptions, dict):
            raise TypeError(
                f"Shooting-binary manifest arrays must be a JSON object: {root / 'manifest.json'}."
            )
        for name, default_filename in _ARRAY_FILES.items():
            description = descriptions.get(name)
            if not isinstance(description, dict):
                raise KeyError(
                    f"Shooting-binary manifest is missing array description {name!r}: "
                    f"{root / 'manifest.json'}."
                )
            filename = description.get("file")
            if filename != default_filename:
                raise ValueError(
                    f"Unexpected file for shooting-binary array {name!r}: "
                    f"expected={default_filename!r}, observed={filename!r}, root={root}."
                )
            array_path = root / filename
            if not array_path.is_file():
                raise FileNotFoundError(
                    f"Shooting-binary array {name!r} is missing: {array_path}"
                )
            values = np.load(array_path, mmap_mode="r", allow_pickle=False)
            expected_shape = tuple(int(value) for value in description["shape"])
            expected_dtype = np.dtype(str(description["dtype"]))
            if values.shape != expected_shape or values.dtype != expected_dtype:
                raise RuntimeError(
                    f"Shooting-binary array contract changed for {name!r}: "
                    f"expected_shape={expected_shape}, observed_shape={values.shape}, "
                    f"expected_dtype={expected_dtype.name}, observed_dtype={values.dtype.name}, "
                    f"path={array_path}."
                )
            arrays[name] = values

        frame_count = int(manifest["frame_count"])
        atom_count = int(manifest["atom_count"])
        vector_shape = (frame_count, atom_count, 3)
        frame_vector_shape = (frame_count, 3)
        expected_shapes = {
            "positions": vector_shape,
            "velocities": vector_shape,
            "timesteps": (frame_count,),
            "box_low": frame_vector_shape,
            "box_high": frame_vector_shape,
            "atom_ids": (atom_count,),
            "atom_types": (atom_count,),
        }
        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                raise RuntimeError(
                    f"Shooting-binary semantic shape mismatch for {name!r}: "
                    f"expected={expected_shape}, observed={arrays[name].shape}, root={root}."
                )
        storage_dtype = str(manifest["storage_dtype"])
        if storage_dtype not in STORAGE_DTYPES:
            raise ValueError(
                f"Unsupported shooting-binary storage_dtype={storage_dtype!r} in {root}."
            )
        expected_vector_dtype = np.dtype(storage_dtype)
        if (
            arrays["positions"].dtype != expected_vector_dtype
            or arrays["velocities"].dtype != expected_vector_dtype
        ):
            raise RuntimeError(
                f"Shooting-binary vector dtype disagrees with storage_dtype={storage_dtype}: "
                f"positions={arrays['positions'].dtype}, velocities={arrays['velocities'].dtype}."
            )
        if not np.array_equal(
            arrays["atom_ids"], np.arange(1, atom_count + 1, dtype=np.int64)
        ):
            raise RuntimeError(
                f"Shooting-binary atom IDs are not exactly 1..{atom_count}: {root}."
            )
        if np.any(arrays["box_high"] <= arrays["box_low"]):
            raise RuntimeError(f"Shooting-binary trajectory has non-positive box lengths: {root}.")
        if np.any(np.diff(arrays["timesteps"]) <= 0):
            raise RuntimeError(
                f"Shooting-binary timesteps must be strictly increasing: {root}."
            )
        return cls(root=root, manifest=manifest, **arrays)

    @property
    def atom_count(self) -> int:
        return int(self.manifest["atom_count"])

    @property
    def frame_count(self) -> int:
        return int(self.manifest["frame_count"])

    @property
    def storage_dtype(self) -> np.dtype[Any]:
        return np.dtype(str(self.manifest["storage_dtype"]))

    def _frame_indices(self, timesteps: Sequence[int]) -> tuple[tuple[int, ...], tuple[int, ...]]:
        requested = tuple(sorted({int(value) for value in timesteps}))
        if not requested or requested[0] < 0:
            raise ValueError(
                f"Requested shooting timesteps must be nonempty and nonnegative: {requested}."
            )
        timestep_to_index = {
            int(timestep): index for index, timestep in enumerate(self.timesteps.tolist())
        }
        missing = [timestep for timestep in requested if timestep not in timestep_to_index]
        if missing:
            raise RuntimeError(
                f"Requested timesteps are absent from binary shooting trajectory {self.root}: "
                f"missing={missing}, available={self.timesteps.tolist()}."
            )
        return requested, tuple(timestep_to_index[value] for value in requested)

    def load_position_frames(
        self, timesteps: Sequence[int]
    ) -> dict[int, ShootingPositionFrame]:
        """Return position-only frame views without touching the velocity array."""

        requested, indices = self._frame_indices(timesteps)
        frames: dict[int, ShootingPositionFrame] = {}
        for timestep, index in zip(requested, indices):
            positions = np.asarray(self.positions[index], dtype=np.float32)
            box_low = np.asarray(self.box_low[index], dtype=np.float32)
            box_high = np.asarray(self.box_high[index], dtype=np.float32)
            box_lengths = box_high - box_low
            if np.any(positions < 0.0):
                raise RuntimeError(
                    f"Decoded shooting positions contain negative wrapped coordinates: "
                    f"root={self.root}, timestep={timestep}, minimum={float(positions.min())}."
                )
            # float16 rounding can move a value just below L to exactly L. The text
            # reader has the same half-open [0, L) contract for periodic cKDTree.
            if np.any(positions >= box_lengths[None, :]):
                positions = np.minimum(
                    positions,
                    np.nextafter(box_lengths, np.zeros_like(box_lengths))[None, :],
                )
            frames[timestep] = ShootingPositionFrame(
                timestep=timestep,
                atom_ids=np.asarray(self.atom_ids),
                atom_types=np.asarray(self.atom_types),
                positions=positions,
                box_low=box_low,
                box_high=box_high,
            )
        return frames

    def load_frames(self, timesteps: Sequence[int]) -> dict[int, ShootingFrame]:
        """Load requested frames, including velocities, as float32 consumer views."""

        position_frames = self.load_position_frames(timesteps)
        _, indices = self._frame_indices(tuple(position_frames))
        frames: dict[int, ShootingFrame] = {}
        for (timestep, position_frame), index in zip(position_frames.items(), indices):
            frames[timestep] = ShootingFrame(
                timestep=timestep,
                atom_ids=position_frame.atom_ids,
                atom_types=position_frame.atom_types,
                positions=position_frame.positions,
                box_low=position_frame.box_low,
                box_high=position_frame.box_high,
                velocities=np.asarray(self.velocities[index], dtype=np.float32),
            )
        return frames

    def verify_checksums(self) -> dict[str, str]:
        """Read every stored value and verify the manifest's semantic checksums."""

        observed: dict[str, str] = {}
        for name in _ARRAY_FILES:
            expected = str(self.manifest["arrays"][name].get("sha256", ""))
            if len(expected) != 64:
                raise RuntimeError(
                    f"Shooting-binary manifest has no valid SHA-256 for array {name!r}: "
                    f"{self.root / 'manifest.json'}."
                )
            observed[name] = _array_sha256(getattr(self, name))
            if observed[name] != expected:
                raise RuntimeError(
                    f"Shooting-binary checksum mismatch for array {name!r}: "
                    f"expected={expected}, observed={observed[name]}, root={self.root}."
                )
        return observed


def convert_shooting_trajectory(
    trajectory_path: str | Path,
    output_dir: str | Path,
    *,
    timesteps: Sequence[int],
    atom_count: int,
    storage_dtype: str,
    provenance: dict[str, Any],
) -> ShootingBinaryTrajectory:
    """Convert one complete repository shooting dump into an atomic binary directory."""

    source = Path(trajectory_path).expanduser().resolve()
    target = Path(output_dir).expanduser().resolve()
    dtype_name = str(storage_dtype)
    if dtype_name not in STORAGE_DTYPES:
        raise ValueError(
            f"storage_dtype must be one of {STORAGE_DTYPES}, got {storage_dtype!r}."
        )
    requested = tuple(int(value) for value in timesteps)
    if not requested or tuple(sorted(set(requested))) != requested:
        raise ValueError(
            f"Conversion timesteps must be nonempty, unique, and increasing: {requested}."
        )
    if int(atom_count) <= 0:
        raise ValueError(f"atom_count must be positive, got {atom_count}.")
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"Source shooting trajectory is missing or empty: {source}")
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing binary shooting trajectory: {target}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{os.getpid()}"
    if building.exists():
        raise FileExistsError(
            f"Refusing to reuse an existing interrupted conversion directory: {building}"
        )
    building.mkdir()

    frames = load_lammps_shooting_frames_for_conversion(
        source,
        timesteps=requested,
        atom_count=int(atom_count),
    )
    if tuple(frames) != requested:
        raise RuntimeError(
            f"Parsed shooting timesteps differ from the conversion contract: "
            f"expected={requested}, observed={tuple(frames)}, source={source}."
        )
    first = frames[requested[0]]
    for timestep in requested[1:]:
        frame = frames[timestep]
        if not np.array_equal(frame.atom_ids, first.atom_ids):
            raise RuntimeError(
                f"Atom IDs changed between shooting frames: source={source}, timestep={timestep}."
            )
        if not np.array_equal(frame.atom_types, first.atom_types):
            raise RuntimeError(
                f"Atom types changed between shooting frames: source={source}, timestep={timestep}."
            )

    frame_count = len(requested)
    vector_shape = (frame_count, int(atom_count), 3)
    vector_dtype = np.dtype(dtype_name)
    positions = open_memmap(
        building / _ARRAY_FILES["positions"],
        mode="w+",
        dtype=vector_dtype,
        shape=vector_shape,
    )
    velocities = open_memmap(
        building / _ARRAY_FILES["velocities"],
        mode="w+",
        dtype=vector_dtype,
        shape=vector_shape,
    )
    box_low = np.empty((frame_count, 3), dtype=np.float32)
    box_high = np.empty((frame_count, 3), dtype=np.float32)
    source_position_digest = hashlib.sha256()
    source_velocity_digest = hashlib.sha256()
    stored_position_digest = hashlib.sha256()
    stored_velocity_digest = hashlib.sha256()
    for frame_index, timestep in enumerate(requested):
        frame = frames[timestep]
        if not np.all(np.isfinite(frame.positions)):
            raise RuntimeError(
                f"Shooting positions contain non-finite values: source={source}, "
                f"timestep={timestep}."
            )
        if not np.all(np.isfinite(frame.velocities)):
            raise RuntimeError(
                f"Shooting velocities contain non-finite values: source={source}, "
                f"timestep={timestep}."
            )
        if dtype_name == "float16":
            float16_limit = float(np.finfo(np.float16).max)
            maximum = max(
                float(np.max(np.abs(frame.positions))),
                float(np.max(np.abs(frame.velocities))),
            )
            if maximum > float16_limit:
                raise OverflowError(
                    f"Shooting values exceed the finite float16 range: source={source}, "
                    f"timestep={timestep}, maximum_absolute_value={maximum}, "
                    f"float16_limit={float16_limit}."
                )
        source_position_digest.update(np.ascontiguousarray(frame.positions).tobytes())
        source_velocity_digest.update(np.ascontiguousarray(frame.velocities).tobytes())
        stored_position_digest.update(
            np.ascontiguousarray(frame.positions, dtype=vector_dtype).tobytes()
        )
        stored_velocity_digest.update(
            np.ascontiguousarray(frame.velocities, dtype=vector_dtype).tobytes()
        )
        positions[frame_index] = frame.positions
        velocities[frame_index] = frame.velocities
        box_low[frame_index] = frame.box_low
        box_high[frame_index] = frame.box_high
    positions.flush()
    velocities.flush()
    del positions
    del velocities

    timesteps_array = np.asarray(requested, dtype=np.int64)
    atom_ids = np.asarray(first.atom_ids, dtype=np.int64)
    atom_types = np.asarray(first.atom_types, dtype=np.int32)
    np.save(building / _ARRAY_FILES["timesteps"], timesteps_array, allow_pickle=False)
    np.save(building / _ARRAY_FILES["box_low"], box_low, allow_pickle=False)
    np.save(building / _ARRAY_FILES["box_high"], box_high, allow_pickle=False)
    np.save(building / _ARRAY_FILES["atom_ids"], atom_ids, allow_pickle=False)
    np.save(building / _ARRAY_FILES["atom_types"], atom_types, allow_pickle=False)
    for filename in _ARRAY_FILES.values():
        with (building / filename).open("rb") as handle:
            os.fsync(handle.fileno())

    source_stat = source.stat()
    stored_positions = np.load(
        building / _ARRAY_FILES["positions"], mmap_mode="r", allow_pickle=False
    )
    stored_velocities = np.load(
        building / _ARRAY_FILES["velocities"], mmap_mode="r", allow_pickle=False
    )
    observed_position_sha256 = _array_sha256(stored_positions)
    observed_velocity_sha256 = _array_sha256(stored_velocities)
    if observed_position_sha256 != stored_position_digest.hexdigest():
        raise RuntimeError(
            f"Stored position checksum differs from converted source values: source={source}, "
            f"target={building}."
        )
    if observed_velocity_sha256 != stored_velocity_digest.hexdigest():
        raise RuntimeError(
            f"Stored velocity checksum differs from converted source values: source={source}, "
            f"target={building}."
        )
    arrays = {
        "positions": _array_description(
            stored_positions,
            _ARRAY_FILES["positions"],
            sha256=observed_position_sha256,
        ),
        "velocities": _array_description(
            stored_velocities,
            _ARRAY_FILES["velocities"],
            sha256=observed_velocity_sha256,
        ),
        "timesteps": _array_description(
            timesteps_array,
            _ARRAY_FILES["timesteps"],
            sha256=_array_sha256(timesteps_array),
        ),
        "box_low": _array_description(
            box_low, _ARRAY_FILES["box_low"], sha256=_array_sha256(box_low)
        ),
        "box_high": _array_description(
            box_high, _ARRAY_FILES["box_high"], sha256=_array_sha256(box_high)
        ),
        "atom_ids": _array_description(
            atom_ids, _ARRAY_FILES["atom_ids"], sha256=_array_sha256(atom_ids)
        ),
        "atom_types": _array_description(
            atom_types,
            _ARRAY_FILES["atom_types"],
            sha256=_array_sha256(atom_types),
        ),
    }
    del stored_positions
    del stored_velocities
    manifest = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage_dtype": dtype_name,
        "coordinate_convention": (
            "positions are wrapped float32 consumer coordinates relative to box_low "
            "in the half-open periodic interval [0, box_high-box_low)"
        ),
        "velocity_units": "angstrom_per_ps",
        "atom_count": int(atom_count),
        "frame_count": frame_count,
        "first_timestep": requested[0],
        "last_timestep": requested[-1],
        "source": {
            "trajectory_path": str(source),
            "size_bytes": int(source_stat.st_size),
            "mtime_ns": int(source_stat.st_mtime_ns),
            "semantic_float32_sha256": {
                "positions": source_position_digest.hexdigest(),
                "velocities": source_velocity_digest.hexdigest(),
            },
        },
        "provenance": provenance,
        "arrays": arrays,
    }
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(building, target)
    return ShootingBinaryTrajectory.load(target)


def compose_shooting_binary_trajectories(
    source_trajectories: Sequence[ShootingBinaryTrajectory],
    output_dir: str | Path,
    *,
    timesteps: Sequence[int],
    storage_dtype: str,
    provenance: dict[str, Any],
) -> ShootingBinaryTrajectory:
    """Write selected frames from validated binary trajectory segments.

    Sources are searched in order, so an original uninterrupted segment wins at
    a duplicated restart timestep and a continuation supplies only later frames.
    """

    sources = tuple(source_trajectories)
    if not sources:
        raise ValueError("Binary trajectory composition requires at least one source.")
    requested = tuple(int(value) for value in timesteps)
    if not requested or tuple(sorted(set(requested))) != requested:
        raise ValueError(
            f"Composition timesteps must be nonempty, unique, and increasing: {requested}."
        )
    dtype_name = str(storage_dtype)
    if dtype_name not in STORAGE_DTYPES:
        raise ValueError(
            f"storage_dtype must be one of {STORAGE_DTYPES}, got {storage_dtype!r}."
        )
    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing binary shooting trajectory: {target}"
        )

    first = sources[0]
    first.verify_checksums()
    frame_sources: dict[int, tuple[ShootingBinaryTrajectory, int]] = {}
    for source in sources:
        source.verify_checksums()
        if source.atom_count != first.atom_count:
            raise RuntimeError(
                "Cannot compose shooting binaries with different atom counts: "
                f"first={first.atom_count}, source={source.atom_count}, root={source.root}."
            )
        if not np.array_equal(source.atom_ids, first.atom_ids) or not np.array_equal(
            source.atom_types, first.atom_types
        ):
            raise RuntimeError(
                f"Cannot compose shooting binaries with different atom identity: {source.root}."
            )
        for frame_index, timestep in enumerate(source.timesteps.tolist()):
            frame_sources.setdefault(int(timestep), (source, frame_index))
    missing = [timestep for timestep in requested if timestep not in frame_sources]
    if missing:
        raise RuntimeError(
            "Binary trajectory segments do not cover the requested composition: "
            f"missing={missing}, sources={[str(source.root) for source in sources]}."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{os.getpid()}"
    if building.exists():
        raise FileExistsError(
            f"Refusing to reuse an interrupted composition directory: {building}"
        )
    building.mkdir()
    frame_count = len(requested)
    vector_dtype = np.dtype(dtype_name)
    vector_shape = (frame_count, first.atom_count, 3)
    positions = open_memmap(
        building / _ARRAY_FILES["positions"],
        mode="w+",
        dtype=vector_dtype,
        shape=vector_shape,
    )
    velocities = open_memmap(
        building / _ARRAY_FILES["velocities"],
        mode="w+",
        dtype=vector_dtype,
        shape=vector_shape,
    )
    box_low = np.empty((frame_count, 3), dtype=np.float32)
    box_high = np.empty((frame_count, 3), dtype=np.float32)
    semantic_position_digest = hashlib.sha256()
    semantic_velocity_digest = hashlib.sha256()
    for output_index, timestep in enumerate(requested):
        source, source_index = frame_sources[timestep]
        source_positions = np.asarray(source.positions[source_index], dtype=np.float32)
        source_velocities = np.asarray(source.velocities[source_index], dtype=np.float32)
        if not np.all(np.isfinite(source_positions)) or not np.all(
            np.isfinite(source_velocities)
        ):
            raise RuntimeError(
                f"Cannot compose non-finite frame values: source={source.root}, "
                f"timestep={timestep}."
            )
        if dtype_name == "float16":
            maximum = max(
                float(np.max(np.abs(source_positions))),
                float(np.max(np.abs(source_velocities))),
            )
            if maximum > float(np.finfo(np.float16).max):
                raise OverflowError(
                    f"Composed frame exceeds float16 range: source={source.root}, "
                    f"timestep={timestep}, maximum_absolute_value={maximum}."
                )
        semantic_position_digest.update(np.ascontiguousarray(source_positions).tobytes())
        semantic_velocity_digest.update(np.ascontiguousarray(source_velocities).tobytes())
        positions[output_index] = source_positions
        velocities[output_index] = source_velocities
        box_low[output_index] = source.box_low[source_index]
        box_high[output_index] = source.box_high[source_index]
    positions.flush()
    velocities.flush()
    del positions
    del velocities

    timesteps_array = np.asarray(requested, dtype=np.int64)
    atom_ids = np.asarray(first.atom_ids, dtype=np.int64)
    atom_types = np.asarray(first.atom_types, dtype=np.int32)
    np.save(building / _ARRAY_FILES["timesteps"], timesteps_array, allow_pickle=False)
    np.save(building / _ARRAY_FILES["box_low"], box_low, allow_pickle=False)
    np.save(building / _ARRAY_FILES["box_high"], box_high, allow_pickle=False)
    np.save(building / _ARRAY_FILES["atom_ids"], atom_ids, allow_pickle=False)
    np.save(building / _ARRAY_FILES["atom_types"], atom_types, allow_pickle=False)

    arrays: dict[str, dict[str, Any]] = {}
    for name, filename in _ARRAY_FILES.items():
        values = np.load(building / filename, mmap_mode="r", allow_pickle=False)
        arrays[name] = _array_description(
            values,
            filename,
            sha256=_array_sha256(values),
        )
    manifest = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage_dtype": dtype_name,
        "coordinate_convention": (
            "positions decode to float32 coordinates relative to box_low in the "
            "half-open periodic interval [0, box_high-box_low)"
        ),
        "velocity_units": "angstrom_per_ps",
        "atom_count": first.atom_count,
        "frame_count": frame_count,
        "first_timestep": requested[0],
        "last_timestep": requested[-1],
        "source": {
            "components": [
                {
                    "path": str(source.root),
                    "storage_dtype": source.storage_dtype.name,
                    "first_timestep": int(source.timesteps[0]),
                    "last_timestep": int(source.timesteps[-1]),
                }
                for source in sources
            ],
            "semantic_float32_sha256": {
                "positions": semantic_position_digest.hexdigest(),
                "velocities": semantic_velocity_digest.hexdigest(),
            },
        },
        "provenance": provenance,
        "arrays": arrays,
    }
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(building, target)
    result = ShootingBinaryTrajectory.load(target)
    result.verify_checksums()
    return result


def binary_directory_sizes(path: str | Path) -> dict[str, int]:
    """Return apparent and allocated bytes for one completed binary directory."""

    root = Path(path).expanduser().resolve()
    files = [entry for entry in root.iterdir() if entry.is_file()]
    if not files:
        raise RuntimeError(f"Binary shooting directory contains no files: {root}")
    return {
        "apparent_bytes": sum(entry.stat().st_size for entry in files),
        "allocated_bytes": sum(entry.stat().st_blocks * 512 for entry in files),
    }


__all__ = [
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "STORAGE_DTYPES",
    "ShootingBinaryTrajectory",
    "binary_directory_sizes",
    "compose_shooting_binary_trajectories",
    "convert_shooting_trajectory",
]
