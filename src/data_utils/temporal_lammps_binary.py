"""Verified memory-mapped storage for completed temporal LAMMPS trajectories."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap


FORMAT_NAME = "pointcloudmaterials.temporal_lammps_trajectory"
SCHEMA_VERSION = 1
BINARY_SUFFIX = "_binary_float32"
STORAGE_DTYPES = ("float32", "float16")
_BINARY_SUFFIX_BY_DTYPE = {
    "float32": BINARY_SUFFIX,
    "float16": "_binary_float16",
}
_ARRAY_FILES = {
    "positions": "positions.npy",
    "timesteps": "timesteps.npy",
    "box_low": "box_low.npy",
    "box_high": "box_high.npy",
    "atom_ids": "atom_ids.npy",
    "atom_types": "atom_types.npy",
}


def binary_path_for_dump(
    path: str | Path, *, storage_dtype: str = "float32"
) -> Path:
    dump_path = Path(path).expanduser().resolve()
    if dump_path.suffix != ".lammpstrj":
        raise ValueError(
            f"Expected a .lammpstrj path when deriving a temporal binary path, got {dump_path}."
        )
    dtype_name = str(storage_dtype)
    if dtype_name not in STORAGE_DTYPES:
        raise ValueError(
            f"storage_dtype must be one of {STORAGE_DTYPES}, got {storage_dtype!r}."
        )
    return dump_path.parent / f"{dump_path.stem}{_BINARY_SUFFIX_BY_DTYPE[dtype_name]}"


def resolve_temporal_lammps_artifact(path: str | Path) -> Path:
    """Resolve an existing text dump or its verified binary replacement."""

    requested = Path(path).expanduser().resolve()
    if requested.is_file():
        return requested
    if requested.is_dir():
        manifest_path = requested / "manifest.json"
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("format") == FORMAT_NAME:
                return requested
        raise ValueError(
            f"Directory is not a temporal LAMMPS binary trajectory: {requested}"
        )
    if requested.suffix == ".lammpstrj":
        replacements = [
            binary_path_for_dump(requested, storage_dtype=dtype_name)
            for dtype_name in STORAGE_DTYPES
        ]
        existing = [replacement for replacement in replacements if replacement.is_dir()]
        if len(existing) == 1:
            return existing[0]
        if len(existing) > 1:
            raise RuntimeError(
                "LAMMPS text trajectory is absent and multiple binary replacements exist; "
                f"request one binary directory explicitly: candidates={existing}."
            )
        raise FileNotFoundError(
            "LAMMPS trajectory is absent in both supported forms: "
            f"text={requested}, binary_candidates={replacements}."
        )
    raise FileNotFoundError(f"Temporal LAMMPS trajectory artifact is missing: {requested}")


def _array_sha256(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    if values.ndim <= 1:
        digest.update(np.ascontiguousarray(values).tobytes())
    else:
        for index in range(int(values.shape[0])):
            digest.update(np.ascontiguousarray(values[index]).tobytes())
    return digest.hexdigest()


def _array_description(values: np.ndarray, filename: str) -> dict[str, Any]:
    return {
        "file": filename,
        "dtype": values.dtype.name,
        "shape": list(values.shape),
        "sha256": _array_sha256(values),
    }


@dataclass(frozen=True)
class TemporalLAMMPSBinaryTrajectory:
    root: Path
    manifest: dict[str, Any]
    positions: np.ndarray
    timesteps: np.ndarray
    box_low: np.ndarray
    box_high: np.ndarray
    atom_ids: np.ndarray
    atom_types: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "TemporalLAMMPSBinaryTrajectory":
        root = Path(path).expanduser().resolve()
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Temporal binary manifest is missing: {manifest_path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise TypeError(f"Temporal binary manifest must be a JSON object: {manifest_path}")
        if manifest.get("format") != FORMAT_NAME:
            raise ValueError(
                f"Unsupported temporal binary format in {manifest_path}: "
                f"expected={FORMAT_NAME!r}, observed={manifest.get('format')!r}."
            )
        if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported temporal binary schema in {manifest_path}: "
                f"expected={SCHEMA_VERSION}, observed={manifest.get('schema_version')!r}."
            )
        if manifest.get("state") != "complete":
            raise RuntimeError(
                f"Temporal binary trajectory is not complete: root={root}, "
                f"state={manifest.get('state')!r}."
            )
        storage_dtype = str(manifest.get("storage_dtype"))
        if storage_dtype not in STORAGE_DTYPES:
            raise ValueError(
                f"Unsupported temporal binary storage_dtype={storage_dtype!r}; "
                f"expected one of {STORAGE_DTYPES}."
            )

        descriptions = manifest.get("arrays")
        if not isinstance(descriptions, dict):
            raise TypeError(f"Temporal binary arrays must be a JSON object: {manifest_path}")
        arrays: dict[str, np.ndarray] = {}
        for name, filename in _ARRAY_FILES.items():
            description = descriptions.get(name)
            if not isinstance(description, dict) or description.get("file") != filename:
                raise ValueError(
                    f"Temporal binary array description is invalid for {name!r}: {manifest_path}"
                )
            array_path = root / filename
            if not array_path.is_file():
                raise FileNotFoundError(f"Temporal binary array is missing: {array_path}")
            values = np.load(array_path, mmap_mode="r", allow_pickle=False)
            expected_shape = tuple(int(value) for value in description["shape"])
            expected_dtype = np.dtype(str(description["dtype"]))
            if values.shape != expected_shape or values.dtype != expected_dtype:
                raise RuntimeError(
                    f"Temporal binary array contract changed for {name!r}: "
                    f"expected_shape={expected_shape}, observed_shape={values.shape}, "
                    f"expected_dtype={expected_dtype.name}, observed_dtype={values.dtype.name}, "
                    f"path={array_path}."
                )
            arrays[name] = values

        frame_count = int(manifest["frame_count"])
        atom_count = int(manifest["atom_count"])
        expected_shapes = {
            "positions": (frame_count, atom_count, 3),
            "timesteps": (frame_count,),
            "box_low": (frame_count, 3),
            "box_high": (frame_count, 3),
            "atom_ids": (atom_count,),
            "atom_types": (atom_count,),
        }
        for name, expected_shape in expected_shapes.items():
            if arrays[name].shape != expected_shape:
                raise RuntimeError(
                    f"Temporal binary semantic shape mismatch for {name!r}: "
                    f"expected={expected_shape}, observed={arrays[name].shape}, root={root}."
                )
        expected_dtypes = {
            "positions": np.dtype(storage_dtype),
            "timesteps": np.dtype("int64"),
            "box_low": np.dtype("float32"),
            "box_high": np.dtype("float32"),
            "atom_ids": np.dtype("int64"),
            "atom_types": np.dtype("int32"),
        }
        for name, expected_dtype in expected_dtypes.items():
            if arrays[name].dtype != expected_dtype:
                raise RuntimeError(
                    f"Temporal binary dtype mismatch for {name!r}: expected={expected_dtype.name}, "
                    f"observed={arrays[name].dtype.name}, root={root}."
                )
        if not np.array_equal(
            arrays["atom_ids"], np.arange(1, atom_count + 1, dtype=np.int64)
        ):
            raise RuntimeError(f"Temporal binary atom IDs are not exactly 1..{atom_count}: {root}")
        if np.any(arrays["box_high"] <= arrays["box_low"]):
            raise RuntimeError(f"Temporal binary trajectory has non-positive box lengths: {root}")
        if frame_count > 1 and np.any(np.diff(arrays["timesteps"]) <= 0):
            raise RuntimeError(f"Temporal binary timesteps are not strictly increasing: {root}")
        return cls(root=root, manifest=manifest, **arrays)

    @property
    def frame_count(self) -> int:
        return int(self.manifest["frame_count"])

    @property
    def atom_count(self) -> int:
        return int(self.manifest["atom_count"])

    def verify_checksums(self) -> dict[str, str]:
        observed: dict[str, str] = {}
        for name in _ARRAY_FILES:
            expected = str(self.manifest["arrays"][name].get("sha256", ""))
            if len(expected) != 64:
                raise RuntimeError(
                    f"Temporal binary manifest has no valid checksum for {name!r}: {self.root}"
                )
            observed[name] = _array_sha256(getattr(self, name))
            if observed[name] != expected:
                raise RuntimeError(
                    f"Temporal binary checksum mismatch for {name!r}: expected={expected}, "
                    f"observed={observed[name]}, root={self.root}."
                )
        return observed


def write_temporal_lammps_binary(
    output_dir: str | Path,
    *,
    positions: np.ndarray,
    timesteps: np.ndarray,
    box_low: np.ndarray,
    box_high: np.ndarray,
    atom_ids: np.ndarray,
    atom_types: np.ndarray,
    atom_columns: tuple[str, ...],
    source: dict[str, Any],
    provenance: dict[str, Any],
    storage_dtype: str = "float32",
) -> TemporalLAMMPSBinaryTrajectory:
    """Atomically write one repository trajectory with float32 consumer semantics."""

    target = Path(output_dir).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite temporal binary trajectory: {target}")
    if positions.dtype != np.dtype("float32") or positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(
            f"positions must be repository-produced float32 (frames, atoms, 3), got "
            f"shape={positions.shape}, dtype={positions.dtype}."
        )
    dtype_name = str(storage_dtype)
    if dtype_name not in STORAGE_DTYPES:
        raise ValueError(
            f"storage_dtype must be one of {STORAGE_DTYPES}, got {storage_dtype!r}."
        )
    frame_count, atom_count, _ = positions.shape
    typed_arrays = {
        "timesteps": np.asarray(timesteps, dtype=np.int64),
        "box_low": np.asarray(box_low, dtype=np.float32),
        "box_high": np.asarray(box_high, dtype=np.float32),
        "atom_ids": np.asarray(atom_ids, dtype=np.int64),
        "atom_types": np.asarray(atom_types, dtype=np.int32),
    }
    expected_shapes = {
        "timesteps": (frame_count,),
        "box_low": (frame_count, 3),
        "box_high": (frame_count, 3),
        "atom_ids": (atom_count,),
        "atom_types": (atom_count,),
    }
    for name, expected_shape in expected_shapes.items():
        if typed_arrays[name].shape != expected_shape:
            raise ValueError(
                f"Temporal binary input {name!r} has shape={typed_arrays[name].shape}, "
                f"expected={expected_shape}."
            )
    if not np.array_equal(
        typed_arrays["atom_ids"], np.arange(1, atom_count + 1, dtype=np.int64)
    ):
        raise ValueError(f"Temporal binary atom IDs must be exactly 1..{atom_count}.")
    box_lengths = typed_arrays["box_high"] - typed_arrays["box_low"]
    if np.any(box_lengths <= 0.0):
        raise ValueError("Temporal binary box bounds contain a non-positive length.")
    target.parent.mkdir(parents=True, exist_ok=True)
    building = target.parent / f".{target.name}.building-{os.getpid()}"
    if building.exists():
        raise FileExistsError(f"Interrupted temporal binary build already exists: {building}")
    building.mkdir()
    stored_positions = open_memmap(
        building / _ARRAY_FILES["positions"],
        mode="w+",
        dtype=np.dtype(dtype_name),
        shape=positions.shape,
    )
    source_float32_digest = hashlib.sha256()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    maximum_absolute_error = 0.0
    value_count = 0
    for frame_index in range(frame_count):
        frame = np.asarray(positions[frame_index], dtype=np.float32)
        if not np.all(np.isfinite(frame)):
            raise ValueError(
                f"Temporal binary positions contain non-finite values at frame={frame_index}."
            )
        if np.any(frame < 0.0) or np.any(frame >= box_lengths[frame_index][None, :]):
            raise ValueError(
                "Temporal binary positions must use wrapped coordinates relative to box_low "
                f"in [0, L): frame={frame_index}."
            )
        source_float32_digest.update(np.ascontiguousarray(frame).tobytes())
        if dtype_name == "float16":
            encoded = frame.astype(np.float16)
            decoded = encoded.astype(np.float32)
            error = decoded - frame
            absolute_error = np.abs(error)
            squared_error_sum += float(np.sum(error * error, dtype=np.float64))
            absolute_error_sum += float(np.sum(absolute_error, dtype=np.float64))
            maximum_absolute_error = max(
                maximum_absolute_error, float(np.max(absolute_error))
            )
            value_count += int(error.size)
            stored_positions[frame_index] = encoded
        else:
            stored_positions[frame_index] = frame
    stored_positions.flush()
    del stored_positions
    for name in ("timesteps", "box_low", "box_high", "atom_ids", "atom_types"):
        np.save(building / _ARRAY_FILES[name], typed_arrays[name], allow_pickle=False)
    for filename in _ARRAY_FILES.values():
        with (building / filename).open("rb") as handle:
            os.fsync(handle.fileno())

    arrays: dict[str, dict[str, Any]] = {}
    for name, filename in _ARRAY_FILES.items():
        values = np.load(building / filename, mmap_mode="r", allow_pickle=False)
        arrays[name] = _array_description(values, filename)
    manifest = {
        "format": FORMAT_NAME,
        "schema_version": SCHEMA_VERSION,
        "state": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "storage_dtype": dtype_name,
        "coordinate_convention": (
            "positions decode to float32 wrapped Cartesian coordinates in angstrom relative "
            "to box_low in the half-open periodic interval [0, box_high-box_low)"
        ),
        "atom_count": atom_count,
        "frame_count": frame_count,
        "first_timestep": int(typed_arrays["timesteps"][0]),
        "last_timestep": int(typed_arrays["timesteps"][-1]),
        "atom_columns": list(atom_columns),
        "source": source,
        "provenance": provenance,
        "semantic_float32_sha256": source_float32_digest.hexdigest(),
        "quantization": (
            {
                "value_count": value_count,
                "mean_absolute_error_A": absolute_error_sum / value_count,
                "rmse_A": float(np.sqrt(squared_error_sum / value_count)),
                "maximum_absolute_error_A": maximum_absolute_error,
            }
            if dtype_name == "float16"
            else {
                "value_count": int(positions.size),
                "mean_absolute_error_A": 0.0,
                "rmse_A": 0.0,
                "maximum_absolute_error_A": 0.0,
            }
        ),
        "arrays": arrays,
    }
    with (building / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(building, target)
    return TemporalLAMMPSBinaryTrajectory.load(target)


def binary_directory_sizes(path: str | Path) -> dict[str, int]:
    root = Path(path).expanduser().resolve()
    files = [entry for entry in root.iterdir() if entry.is_file()]
    if not files:
        raise RuntimeError(f"Temporal binary directory contains no files: {root}")
    return {
        "apparent_bytes": sum(entry.stat().st_size for entry in files),
        "allocated_bytes": sum(entry.stat().st_blocks * 512 for entry in files),
    }


__all__ = [
    "BINARY_SUFFIX",
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "STORAGE_DTYPES",
    "TemporalLAMMPSBinaryTrajectory",
    "binary_directory_sizes",
    "binary_path_for_dump",
    "resolve_temporal_lammps_artifact",
    "write_temporal_lammps_binary",
]
