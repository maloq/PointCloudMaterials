#!/usr/bin/env python3
"""Convert the completed Zr/Al/Mg initial-configuration campaign to float16."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from scipy.spatial import cKDTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.temporal_lammps_binary import (  # noqa: E402
    FORMAT_NAME,
    TemporalLAMMPSBinaryTrajectory,
    binary_directory_sizes,
    binary_path_for_dump,
    write_temporal_lammps_binary,
)
from src.data_utils.temporal_lammps_dataset import (  # noqa: E402
    TemporalLAMMPSDumpDataset,
    _sanitize_periodic_points,
)


EXPECTED_COLUMNS = ("id", "type", "x", "y", "z")
TARGET_DIRNAME = "trajectory_binary_float16"
BRANCH_REPORT = "binary_conversion_float16.json"
CAMPAIGN_REPORT = "binary_conversion_float16.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}.")
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(document).__name__}.")
    return document


def _write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Refusing to reuse temporary JSON path: {temporary}.")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _geometry_error(
    reference_positions: np.ndarray,
    candidate: TemporalLAMMPSBinaryTrajectory,
    *,
    center_count: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(20260904)
    centers = np.sort(
        rng.choice(candidate.atom_count, size=min(center_count, candidate.atom_count), replace=False)
    )
    frame_indices = (0, candidate.frame_count // 2, candidate.frame_count - 1)
    neighbor_count = 16
    retained = 0
    compared = 0
    distance_errors: list[np.ndarray] = []
    for frame_index in frame_indices:
        box_lengths = np.asarray(
            candidate.box_high[frame_index] - candidate.box_low[frame_index], dtype=np.float32
        )
        reference = np.asarray(reference_positions[frame_index], dtype=np.float32)
        decoded = _sanitize_periodic_points(
            np.asarray(candidate.positions[frame_index], dtype=np.float32), box_lengths
        )
        reference_tree = cKDTree(reference, boxsize=box_lengths, balanced_tree=False)
        candidate_tree = cKDTree(decoded, boxsize=box_lengths, balanced_tree=False)
        _, reference_neighbors = reference_tree.query(
            reference[centers], k=neighbor_count + 1, workers=1
        )
        _, candidate_neighbors = candidate_tree.query(
            decoded[centers], k=neighbor_count + 1, workers=1
        )
        reference_neighbors = np.asarray(reference_neighbors)[:, 1:]
        candidate_neighbors = np.asarray(candidate_neighbors)[:, 1:]
        for expected, observed in zip(reference_neighbors, candidate_neighbors, strict=True):
            retained += len(set(expected.tolist()).intersection(observed.tolist()))
        compared += int(reference_neighbors.size)

        reference_delta = reference[reference_neighbors] - reference[centers, None, :]
        decoded_delta = decoded[reference_neighbors] - decoded[centers, None, :]
        reference_delta -= box_lengths[None, None, :] * np.round(
            reference_delta / box_lengths[None, None, :]
        )
        decoded_delta -= box_lengths[None, None, :] * np.round(
            decoded_delta / box_lengths[None, None, :]
        )
        errors = np.abs(
            np.linalg.norm(decoded_delta.astype(np.float64), axis=-1)
            - np.linalg.norm(reference_delta.astype(np.float64), axis=-1)
        )
        distance_errors.append(errors.reshape(-1))
    combined = np.concatenate(distance_errors)
    return {
        "sampled_frame_indices": list(frame_indices),
        "sampled_centers_per_frame": int(centers.size),
        "neighbors_per_center": neighbor_count,
        "neighbor_set_retention_fraction": retained / compared,
        "same_neighbor_distance_mean_absolute_error_A": float(np.mean(combined)),
        "same_neighbor_distance_p99_absolute_error_A": float(np.quantile(combined, 0.99)),
        "same_neighbor_distance_maximum_absolute_error_A": float(np.max(combined)),
    }


def _parse_source_to_float32_memmap(
    source: Path,
    metadata: dict[str, Any],
    temporary_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frame_count = int(metadata["expected_frames"])
    atom_count = int(metadata["atom_count"])
    sample_steps = int(metadata["sample_interval_steps"])
    run_steps = int(metadata["duration_steps"])
    expected_timesteps = np.arange(0, run_steps + 1, sample_steps, dtype=np.int64)
    if expected_timesteps.size != frame_count:
        raise RuntimeError(
            f"Branch metadata has inconsistent timestep/frame contract: {metadata}."
        )
    positions = open_memmap(
        temporary_dir / "positions_float32.npy",
        mode="w+",
        dtype=np.float32,
        shape=(frame_count, atom_count, 3),
    )
    box_low = np.empty((frame_count, 3), dtype=np.float32)
    box_high = np.empty((frame_count, 3), dtype=np.float32)
    expected_ids = np.arange(1, atom_count + 1, dtype=np.int64)
    atom_types: np.ndarray | None = None
    with source.open("r", encoding="utf-8") as handle:
        for frame_index, expected_timestep in enumerate(expected_timesteps.tolist()):
            header = TemporalLAMMPSDumpDataset._read_frame_header(handle, source_path=source)
            if header is None:
                raise RuntimeError(
                    f"Unexpected EOF at frame={frame_index}/{frame_count} in {source}."
                )
            if int(header["timestep"]) != expected_timestep:
                raise RuntimeError(
                    f"Timestep changed in {source}: frame={frame_index}, "
                    f"expected={expected_timestep}, observed={header['timestep']}."
                )
            if int(header["num_atoms"]) != atom_count:
                raise RuntimeError(
                    f"Atom count changed in {source}: frame={frame_index}, "
                    f"expected={atom_count}, observed={header['num_atoms']}."
                )
            columns = tuple(str(value) for value in header["atom_columns"])
            if columns != EXPECTED_COLUMNS:
                raise RuntimeError(
                    f"Column contract changed in {source}: expected={EXPECTED_COLUMNS}, "
                    f"observed={columns}."
                )
            values = np.fromfile(
                handle,
                dtype=np.float64,
                count=atom_count * len(EXPECTED_COLUMNS),
                sep=" ",
            )
            if values.size != atom_count * len(EXPECTED_COLUMNS):
                raise RuntimeError(
                    f"Incomplete atom block in {source}: frame={frame_index}, "
                    f"expected_values={atom_count * len(EXPECTED_COLUMNS)}, "
                    f"observed_values={values.size}."
                )
            table = values.reshape(atom_count, len(EXPECTED_COLUMNS))
            observed_ids = table[:, 0].astype(np.int64, copy=False)
            if not np.array_equal(observed_ids, expected_ids):
                raise RuntimeError(
                    f"Atom IDs are not exactly sorted 1..{atom_count} in {source}, "
                    f"frame={frame_index}."
                )
            frame_types = table[:, 1].astype(np.int32, copy=False)
            if atom_types is None:
                atom_types = np.array(frame_types, copy=True)
            elif not np.array_equal(frame_types, atom_types):
                raise RuntimeError(f"Atom types changed in {source}, frame={frame_index}.")
            box_low[frame_index] = np.asarray(header["box_low"], dtype=np.float32)
            box_high[frame_index] = np.asarray(header["box_high"], dtype=np.float32)
            box_lengths = box_high[frame_index] - box_low[frame_index]
            if np.any(box_lengths <= 0.0):
                raise RuntimeError(
                    f"Non-positive box length in {source}, frame={frame_index}: "
                    f"box_low={box_low[frame_index]}, box_high={box_high[frame_index]}."
                )
            coordinates = table[:, 2:5].astype(np.float32, copy=False)
            wrapped = np.mod(coordinates - box_low[frame_index][None, :], box_lengths[None, :])
            positions[frame_index] = _sanitize_periodic_points(wrapped, box_lengths)
            if frame_index == 0 or (frame_index + 1) % 20 == 0 or frame_index + 1 == frame_count:
                print(
                    f"[float16] parsed {source.parent.parent.name}/{source.parent.name} "
                    f"frame={frame_index + 1}/{frame_count}",
                    flush=True,
                )
        extra = TemporalLAMMPSDumpDataset._read_frame_header(handle, source_path=source)
        if extra is not None:
            raise RuntimeError(
                f"Source contains frames beyond the metadata contract: source={source}, "
                f"extra_timestep={extra['timestep']}."
            )
    positions.flush()
    if atom_types is None:
        raise RuntimeError(f"No atom types were parsed from {source}.")
    return positions, expected_timesteps, box_low, box_high, atom_types


def _convert_branch(
    campaign_root: Path,
    material: str,
    snapshot: str,
    *,
    geometry_centers: int,
) -> dict[str, Any]:
    branch_dir = campaign_root / "branches" / material / snapshot
    metadata_path = branch_dir / "metadata.json"
    metadata = _load_json(metadata_path)
    if metadata.get("state") != "complete":
        raise RuntimeError(
            f"Refusing to convert incomplete branch {material}/{snapshot}: "
            f"state={metadata.get('state')!r}."
        )
    source = branch_dir / "trajectory.lammpstrj"
    if not source.is_file() or source.stat().st_size != int(metadata["trajectory"]["size_bytes"]):
        raise RuntimeError(
            f"Source trajectory is missing or changed for {material}/{snapshot}: {source}."
        )
    target = binary_path_for_dump(source, storage_dtype="float16")
    if target.name != TARGET_DIRNAME:
        raise RuntimeError(f"Unexpected float16 target path: {target}.")
    report_path = branch_dir / BRANCH_REPORT
    if target.exists() or report_path.exists():
        if not target.is_dir() or not report_path.is_file():
            raise RuntimeError(
                f"Partial prior conversion exists for {material}/{snapshot}: "
                f"target={target.exists()}, report={report_path.exists()}."
            )
        report = _load_json(report_path)
        if report.get("state") != "complete":
            raise RuntimeError(f"Prior branch conversion is not complete: {report_path}.")
        binary = TemporalLAMMPSBinaryTrajectory.load(target)
        binary.verify_checksums()
        print(f"[float16] verified existing {material}/{snapshot}", flush=True)
        return report

    source_stat = source.stat()
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=".float16-source-", dir=branch_dir) as raw_temporary:
        temporary_dir = Path(raw_temporary)
        positions, timesteps, box_low, box_high, atom_types = _parse_source_to_float32_memmap(
            source, metadata, temporary_dir
        )
        binary = write_temporal_lammps_binary(
            target,
            positions=positions,
            timesteps=timesteps,
            box_low=box_low,
            box_high=box_high,
            atom_ids=np.arange(1, int(metadata["atom_count"]) + 1, dtype=np.int64),
            atom_types=atom_types,
            atom_columns=EXPECTED_COLUMNS,
            source={
                "trajectory_lammpstrj": {
                    "path": str(source.resolve()),
                    "size_bytes": int(source_stat.st_size),
                    "mtime_ns": int(source_stat.st_mtime_ns),
                },
                "campaign_manifest": str((campaign_root / "manifest.json").resolve()),
                "branch_metadata": str(metadata_path.resolve()),
            },
            provenance={
                "conversion_script": str(Path(__file__).resolve()),
                "source_campaign_complete": True,
                "source_coordinate_semantics": "repository temporal LAMMPS float32 decoder",
            },
            storage_dtype="float16",
        )
        checksums = binary.verify_checksums()
        geometry = _geometry_error(
            positions,
            binary,
            center_count=geometry_centers,
        )
    sizes = binary_directory_sizes(target)
    report = {
        "format": FORMAT_NAME,
        "schema_version": 1,
        "state": "complete",
        "completed_at_utc": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "material": material,
        "snapshot": snapshot,
        "source": {
            "path": str(source.resolve()),
            "size_bytes": int(source_stat.st_size),
            "mtime_ns": int(source_stat.st_mtime_ns),
            "preserved": True,
        },
        "binary": {
            "path": str(target.resolve()),
            "storage_dtype": "float16",
            "frame_count": binary.frame_count,
            "atom_count": binary.atom_count,
            **sizes,
            "apparent_ratio_vs_text": sizes["apparent_bytes"] / source_stat.st_size,
            "checksums": checksums,
        },
        "position_quantization": binary.manifest["quantization"],
        "local_geometry": geometry,
    }
    _write_json_atomic(report_path, report)
    print(
        f"[float16] complete {material}/{snapshot}: "
        f"binary_GiB={sizes['apparent_bytes'] / 1024**3:.3f}, "
        f"position_rmse_A={binary.manifest['quantization']['rmse_A']:.6f}, "
        f"neighbor_retention={geometry['neighbor_set_retention_fraction']:.6f}",
        flush=True,
    )
    return report


def convert(campaign_root: Path, *, geometry_centers: int, workers: int) -> None:
    campaign_root = campaign_root.expanduser().resolve()
    manifest = _load_json(campaign_root / "manifest.json")
    if manifest.get("state") != "complete" or int(manifest.get("completed_branch_count", -1)) != 18:
        raise RuntimeError(
            f"Campaign must contain 18 completed branches before conversion: "
            f"state={manifest.get('state')!r}, "
            f"completed={manifest.get('completed_branch_count')!r}."
        )
    if geometry_centers <= 0:
        raise ValueError(f"geometry_centers must be positive, got {geometry_centers}.")
    if workers <= 0:
        raise ValueError(f"workers must be positive, got {workers}.")
    branches = [
        (str(branch["material"]), str(branch["snapshot"]))
        for branch in manifest["branches"]
    ]
    worker_count = min(workers, len(branches))
    expected_binary_bytes = sum(
        (int(_load_json(campaign_root / "branches" / branch["material"] / branch["snapshot"] / "metadata.json")["expected_frames"])
         * int(_load_json(campaign_root / "branches" / branch["material"] / branch["snapshot"] / "metadata.json")["atom_count"])
         * 3 * np.dtype(np.float16).itemsize)
        for branch in manifest["branches"]
    )
    free_bytes = shutil.disk_usage(campaign_root).free
    largest_temporary_bytes = max(
        int(_load_json(campaign_root / "branches" / branch["material"] / branch["snapshot"] / "metadata.json")["expected_frames"])
        * int(_load_json(campaign_root / "branches" / branch["material"] / branch["snapshot"] / "metadata.json")["atom_count"])
        * 3 * np.dtype(np.float32).itemsize
        for branch in manifest["branches"]
    )
    required_bytes = int(
        (expected_binary_bytes + worker_count * largest_temporary_bytes) * 1.05
    )
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient disk for float16 conversion while preserving source text: "
            f"free={free_bytes}, required_with_margin={required_bytes}."
        )

    campaign_report_path = campaign_root / CAMPAIGN_REPORT
    if campaign_report_path.exists():
        existing = _load_json(campaign_report_path)
        if existing.get("state") == "complete":
            raise FileExistsError(f"Campaign conversion is already complete: {campaign_report_path}.")
    campaign_report: dict[str, Any] = {
        "format": FORMAT_NAME,
        "schema_version": 1,
        "state": "running",
        "started_at_utc": _utc_now(),
        "campaign_root": str(campaign_root),
        "storage_dtype": "float16",
        "workers": worker_count,
        "source_text_preserved": True,
        "expected_binary_position_bytes": expected_binary_bytes,
        "free_bytes_at_start": free_bytes,
        "branch_reports": [],
    }
    _write_json_atomic(campaign_report_path, campaign_report)
    started = time.monotonic()
    try:
        reports_by_branch: dict[tuple[str, str], dict[str, Any]] = {}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures: dict[Future[dict[str, Any]], tuple[str, str]] = {}
            for index, (material, snapshot) in enumerate(branches):
                print(
                    f"[float16] queued branch={index + 1}/18 "
                    f"material={material} snapshot={snapshot}",
                    flush=True,
                )
                future = executor.submit(
                    _convert_branch,
                    campaign_root,
                    material,
                    snapshot,
                    geometry_centers=geometry_centers,
                )
                futures[future] = (material, snapshot)
            for future in as_completed(futures):
                material, snapshot = futures[future]
                reports_by_branch[(material, snapshot)] = future.result()
                reports = [
                    reports_by_branch[branch]
                    for branch in branches
                    if branch in reports_by_branch
                ]
                print(
                    f"[float16] campaign progress={len(reports)}/18 "
                    f"last={material}/{snapshot}",
                    flush=True,
                )
                campaign_report["branch_reports"] = [
                    str(
                        campaign_root
                        / "branches"
                        / report["material"]
                        / report["snapshot"]
                        / BRANCH_REPORT
                    )
                    for report in reports
                ]
                _write_json_atomic(campaign_report_path, campaign_report)
        reports = [reports_by_branch[branch] for branch in branches]
    except BaseException as error:
        campaign_report.update(
            {
                "state": "failed",
                "failed_at_utc": _utc_now(),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json_atomic(campaign_report_path, campaign_report)
        raise
    campaign_report.update(
        {
            "state": "complete",
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "branch_count": len(reports),
            "source_text_bytes": sum(int(report["source"]["size_bytes"]) for report in reports),
            "binary_apparent_bytes": sum(
                int(report["binary"]["apparent_bytes"]) for report in reports
            ),
            "mean_position_rmse_A": float(
                np.mean([report["position_quantization"]["rmse_A"] for report in reports])
            ),
            "minimum_neighbor_set_retention_fraction": float(
                min(
                    report["local_geometry"]["neighbor_set_retention_fraction"]
                    for report in reports
                )
            ),
        }
    )
    _write_json_atomic(campaign_report_path, campaign_report)
    print(json.dumps(campaign_report, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--geometry-centers", type=int, default=128)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of branches to convert concurrently (default: 1).",
    )
    args = parser.parse_args()
    convert(
        args.campaign,
        geometry_centers=args.geometry_centers,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
