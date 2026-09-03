#!/usr/bin/env python3
"""Convert complete LAMMPS shooting branches to a fast memory-mapped format."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
from scipy.spatial import cKDTree

from src.data_utils.shooting_binary import (
    ShootingBinaryTrajectory,
    binary_directory_sizes,
    convert_shooting_trajectory,
)
from src.data_utils.shooting_text_conversion import (
    load_lammps_shooting_frames_for_conversion,
)


_EXPECTED_COLUMNS = ("id", "type", "x", "y", "z", "vx", "vy", "vz")


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required JSON file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}, got {type(value).__name__}.")
    return value


def _validated_complete_branches(
    campaign_root: Path,
    *,
    branch_indices: Sequence[int] | None,
) -> tuple[dict[str, Any], tuple[int, ...], list[tuple[dict[str, Any], dict[str, Any], Path]]]:
    manifest_path = campaign_root / "manifest.json"
    manifest = _load_json_object(manifest_path)
    if manifest.get("campaign_type") != "position_conditioned_langevin_nvt_shooting":
        raise ValueError(
            f"Unsupported campaign_type={manifest.get('campaign_type')!r} in {manifest_path}."
        )
    protocol = manifest["protocol"]
    if tuple(protocol["dump_columns"]) != _EXPECTED_COLUMNS:
        raise ValueError(
            f"Binary shooting conversion requires columns {_EXPECTED_COLUMNS}, "
            f"got {tuple(protocol['dump_columns'])} in {manifest_path}."
        )
    sample_interval = int(protocol["sample_interval_steps"])
    run_steps = int(protocol["run_steps"])
    if run_steps % sample_interval != 0:
        raise RuntimeError(
            f"Campaign run_steps={run_steps} is not divisible by "
            f"sample_interval_steps={sample_interval}: {manifest_path}."
        )
    timesteps = tuple(range(0, run_steps + 1, sample_interval))
    if len(timesteps) != int(protocol["expected_frame_count"]):
        raise RuntimeError(
            f"Campaign frame contract is inconsistent: generated={len(timesteps)}, "
            f"expected={protocol['expected_frame_count']}, manifest={manifest_path}."
        )

    by_index = {int(branch["branch_index"]): branch for branch in manifest["branches"]}
    if branch_indices is None:
        selected_indices = tuple(sorted(by_index))
    else:
        selected_indices = tuple(int(value) for value in branch_indices)
        if len(set(selected_indices)) != len(selected_indices):
            raise ValueError(f"Duplicate --branch-index values: {selected_indices}.")
        missing = [value for value in selected_indices if value not in by_index]
        if missing:
            raise ValueError(
                f"Requested branch indices are absent from {manifest_path}: {missing}."
            )

    validated: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for branch_index in selected_indices:
        branch = by_index[branch_index]
        branch_dir = campaign_root / str(branch["branch_dir"])
        outcome_path = branch_dir / "outcome.json"
        outcome = _load_json_object(outcome_path)
        if outcome.get("state") != "complete":
            raise RuntimeError(
                f"Refusing to convert an incomplete branch: branch_index={branch_index}, "
                f"state={outcome.get('state')!r}, outcome={outcome_path}."
            )
        for key in (
            "branch_index",
            "branch_id",
            "branch_dir",
            "parent_index",
            "parent_id",
            "source_run_id",
            "source_split",
            "source_velocity_seed",
            "temperature_K",
            "phase",
            "shot_index",
            "velocity_seed",
            "thermostat_seed",
        ):
            if outcome.get(key) != branch[key]:
                raise RuntimeError(
                    f"Completed outcome disagrees with its campaign manifest: "
                    f"branch_index={branch_index}, key={key!r}, "
                    f"manifest={branch[key]!r}, outcome={outcome.get(key)!r}, "
                    f"path={outcome_path}."
                )
        if (
            int(outcome["frame_count"]) != len(timesteps)
            or int(outcome["first_timestep"]) != timesteps[0]
            or int(outcome["last_timestep"]) != timesteps[-1]
        ):
            raise RuntimeError(
                f"Completed branch violates the campaign frame contract: "
                f"branch_index={branch_index}, outcome={outcome_path}."
            )
        trajectory = branch_dir / "trajectory.lammpstrj"
        if not trajectory.is_file() or trajectory.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Completed branch trajectory is missing or empty: {trajectory}"
            )
        if trajectory.stat().st_size != int(outcome["trajectory_size_bytes"]):
            raise RuntimeError(
                f"Completed branch trajectory size changed: branch_index={branch_index}, "
                f"outcome_size={outcome['trajectory_size_bytes']}, "
                f"observed_size={trajectory.stat().st_size}, path={trajectory}."
            )
        restart = branch_dir / "final.restart.bin"
        if not restart.is_file() or restart.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Completed branch restart is missing or empty: {restart}"
            )
        if restart.stat().st_size != int(outcome["restart_size_bytes"]):
            raise RuntimeError(
                f"Completed branch restart size changed: branch_index={branch_index}, "
                f"outcome_size={outcome['restart_size_bytes']}, "
                f"observed_size={restart.stat().st_size}, path={restart}."
            )
        validated.append((branch, outcome, trajectory))
    return manifest, timesteps, validated


def _vector_error_metrics(
    reference: ShootingBinaryTrajectory,
    candidate: ShootingBinaryTrajectory,
    *,
    field: str,
    sample_count: int,
) -> dict[str, float | int]:
    reference_values = getattr(reference, field)
    candidate_values = getattr(candidate, field)
    if reference_values.shape != candidate_values.shape:
        raise RuntimeError(
            f"Cannot compare binary {field}: reference_shape={reference_values.shape}, "
            f"candidate_shape={candidate_values.shape}."
        )
    rng = np.random.default_rng(20260901)
    samples_per_frame = max(1, int(np.ceil(sample_count / reference.frame_count)))
    sampled_absolute_errors: list[np.ndarray] = []
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    reference_squared_sum = 0.0
    maximum_absolute_error = 0.0
    unchanged_count = 0
    value_count = 0
    for frame_index in range(reference.frame_count):
        ref = np.asarray(reference_values[frame_index], dtype=np.float32)
        low = np.asarray(candidate_values[frame_index], dtype=np.float32)
        error = low - ref
        absolute_error = np.abs(error)
        squared_error_sum += float(np.sum(error * error, dtype=np.float64))
        absolute_error_sum += float(np.sum(absolute_error, dtype=np.float64))
        reference_squared_sum += float(np.sum(ref * ref, dtype=np.float64))
        maximum_absolute_error = max(
            maximum_absolute_error, float(np.max(absolute_error))
        )
        unchanged_count += int(np.count_nonzero(error == 0.0))
        value_count += int(error.size)
        flat = absolute_error.reshape(-1)
        count = min(samples_per_frame, flat.size)
        indices = rng.choice(flat.size, size=count, replace=False)
        sampled_absolute_errors.append(flat[indices])
    sampled = np.concatenate(sampled_absolute_errors)
    rmse = float(np.sqrt(squared_error_sum / value_count))
    reference_rms = float(np.sqrt(reference_squared_sum / value_count))
    return {
        "value_count": value_count,
        "unchanged_value_count": unchanged_count,
        "unchanged_fraction": unchanged_count / value_count,
        "mean_absolute_error": absolute_error_sum / value_count,
        "rmse": rmse,
        "relative_rmse_vs_value_rms": rmse / reference_rms,
        "maximum_absolute_error": maximum_absolute_error,
        "sampled_median_absolute_error": float(np.quantile(sampled, 0.5)),
        "sampled_p95_absolute_error": float(np.quantile(sampled, 0.95)),
        "sampled_p99_absolute_error": float(np.quantile(sampled, 0.99)),
        "sampled_value_count": int(sampled.size),
    }


def _geometry_error_metrics(
    reference: ShootingBinaryTrajectory,
    candidate: ShootingBinaryTrajectory,
    *,
    center_count: int,
) -> dict[str, float | int | list[int]]:
    frame_indices = sorted({0, reference.frame_count // 2, reference.frame_count - 1})
    rng = np.random.default_rng(20260901)
    centers = np.sort(
        rng.choice(
            reference.atom_count,
            size=min(int(center_count), reference.atom_count),
            replace=False,
        )
    )
    neighbor_count = min(16, reference.atom_count - 1)
    if neighbor_count <= 0:
        raise ValueError("Geometry comparison requires at least two atoms.")
    retained = 0
    compared_neighbors = 0
    distance_errors: list[np.ndarray] = []
    for frame_index in frame_indices:
        timestep = int(reference.timesteps[frame_index])
        ref_frame = reference.load_frames([timestep])[timestep]
        low_frame = candidate.load_frames([timestep])[timestep]
        box_lengths = ref_frame.box_lengths.astype(np.float64)
        reference_tree = cKDTree(ref_frame.positions, boxsize=box_lengths)
        candidate_tree = cKDTree(low_frame.positions, boxsize=box_lengths)
        _, reference_neighbors = reference_tree.query(
            ref_frame.positions[centers], k=neighbor_count + 1, workers=1
        )
        _, candidate_neighbors = candidate_tree.query(
            low_frame.positions[centers], k=neighbor_count + 1, workers=1
        )
        reference_neighbors = np.asarray(reference_neighbors)[:, 1:]
        candidate_neighbors = np.asarray(candidate_neighbors)[:, 1:]
        for ref_ids, low_ids in zip(reference_neighbors, candidate_neighbors, strict=True):
            retained += len(set(ref_ids.tolist()).intersection(low_ids.tolist()))
        compared_neighbors += int(reference_neighbors.size)

        reference_delta = (
            ref_frame.positions[reference_neighbors]
            - ref_frame.positions[centers, None, :]
        ).astype(np.float64)
        candidate_delta = (
            low_frame.positions[reference_neighbors]
            - low_frame.positions[centers, None, :]
        ).astype(np.float64)
        reference_delta -= np.rint(reference_delta / box_lengths) * box_lengths
        candidate_delta -= np.rint(candidate_delta / box_lengths) * box_lengths
        reference_distances = np.linalg.norm(reference_delta, axis=-1)
        candidate_distances = np.linalg.norm(candidate_delta, axis=-1)
        distance_errors.append(np.abs(candidate_distances - reference_distances).reshape(-1))
    errors = np.concatenate(distance_errors)
    return {
        "sampled_frame_indices": frame_indices,
        "sampled_center_count_per_frame": int(centers.size),
        "neighbors_per_center": neighbor_count,
        "neighbor_set_retention_fraction": retained / compared_neighbors,
        "same_neighbor_distance_mean_absolute_error_A": float(np.mean(errors)),
        "same_neighbor_distance_p99_absolute_error_A": float(np.quantile(errors, 0.99)),
        "same_neighbor_distance_maximum_absolute_error_A": float(np.max(errors)),
    }


def _consume_frames(frames: dict[int, Any]) -> float:
    checksum = 0.0
    for frame in frames.values():
        checksum += float(np.sum(frame.positions, dtype=np.float64))
        checksum += float(np.sum(frame.velocities, dtype=np.float64))
    return checksum


def _benchmark(
    loader: Callable[[], dict[int, Any]], *, repetitions: int
) -> dict[str, float | int]:
    if repetitions <= 0:
        raise ValueError(f"benchmark repetitions must be positive, got {repetitions}.")
    elapsed: list[float] = []
    checksum: float | None = None
    for _ in range(repetitions):
        gc.collect()
        start = time.perf_counter()
        observed = _consume_frames(loader())
        elapsed.append(time.perf_counter() - start)
        if checksum is None:
            checksum = observed
        elif not np.isclose(observed, checksum, rtol=0.0, atol=1.0e-5):
            raise RuntimeError(
                f"Benchmark loader returned inconsistent checksums: first={checksum}, "
                f"observed={observed}."
            )
    return {
        "repetitions": repetitions,
        "median_seconds": statistics.median(elapsed),
        "minimum_seconds": min(elapsed),
        "maximum_seconds": max(elapsed),
        "checksum": float(checksum),
    }


def _convert(args: argparse.Namespace) -> None:
    campaign_root = Path(args.campaign_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    selected_indices = None if args.all_complete else args.branch_index
    manifest, timesteps, branches = _validated_complete_branches(
        campaign_root,
        branch_indices=selected_indices,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    storage_dtypes = tuple(args.dtype or ("float32", "float16"))
    if len(set(storage_dtypes)) != len(storage_dtypes):
        raise ValueError(f"Duplicate --dtype values: {storage_dtypes}.")
    if int(args.precision_sample_count) <= 0:
        raise ValueError(
            f"--precision-sample-count must be positive, got {args.precision_sample_count}."
        )
    if int(args.geometry_centers) <= 0:
        raise ValueError(
            f"--geometry-centers must be positive, got {args.geometry_centers}."
        )
    if int(args.benchmark_repetitions) <= 0:
        raise ValueError(
            f"--benchmark-repetitions must be positive, got {args.benchmark_repetitions}."
        )

    report: dict[str, Any] = {
        "campaign_root": str(campaign_root),
        "campaign_manifest": str(campaign_root / "manifest.json"),
        "output_root": str(output_root),
        "storage_dtypes": list(storage_dtypes),
        "branch_reports": [],
    }
    for branch, outcome, trajectory in branches:
        branch_output = output_root / str(branch["branch_id"])
        branch_output.mkdir(parents=True, exist_ok=True)
        converted: dict[str, ShootingBinaryTrajectory] = {}
        dtype_reports: dict[str, Any] = {}
        for dtype_name in storage_dtypes:
            target = branch_output / f"trajectory_{dtype_name}"
            print(
                f"[shooting-binary] converting branch={branch['branch_index']} "
                f"dtype={dtype_name} source={trajectory} target={target}",
                flush=True,
            )
            converted[dtype_name] = convert_shooting_trajectory(
                trajectory,
                target,
                timesteps=timesteps,
                atom_count=int(manifest["atom_count"]),
                storage_dtype=dtype_name,
                provenance={
                    "campaign_manifest": str(campaign_root / "manifest.json"),
                    "branch": branch,
                    "outcome_completed_at": outcome.get("completed_at"),
                },
            )
            sizes = binary_directory_sizes(target)
            dtype_reports[dtype_name] = {
                "path": str(target),
                **sizes,
                "apparent_ratio_vs_text": sizes["apparent_bytes"]
                / trajectory.stat().st_size,
                "allocated_ratio_vs_text": sizes["allocated_bytes"]
                / (trajectory.stat().st_blocks * 512),
            }

        branch_report: dict[str, Any] = {
            "branch_index": int(branch["branch_index"]),
            "branch_id": str(branch["branch_id"]),
            "source_trajectory": str(trajectory),
            "source_apparent_bytes": trajectory.stat().st_size,
            "source_allocated_bytes": trajectory.stat().st_blocks * 512,
            "binary": dtype_reports,
        }
        if "float32" in converted and "float16" in converted:
            print(
                f"[shooting-binary] measuring float16 precision branch={branch['branch_index']}",
                flush=True,
            )
            branch_report["float16_precision"] = {
                "positions_A": _vector_error_metrics(
                    converted["float32"],
                    converted["float16"],
                    field="positions",
                    sample_count=int(args.precision_sample_count),
                ),
                "velocities_A_per_ps": _vector_error_metrics(
                    converted["float32"],
                    converted["float16"],
                    field="velocities",
                    sample_count=int(args.precision_sample_count),
                ),
                "local_geometry": _geometry_error_metrics(
                    converted["float32"],
                    converted["float16"],
                    center_count=int(args.geometry_centers),
                ),
            }

        if args.benchmark:
            benchmark_timesteps = tuple(
                timesteps[index]
                for index in np.linspace(0, len(timesteps) - 1, 5, dtype=np.int64)
            )
            print(
                f"[shooting-binary] benchmarking branch={branch['branch_index']} "
                f"timesteps={benchmark_timesteps}",
                flush=True,
            )
            benchmarks = {
                "timesteps": list(benchmark_timesteps),
                "text": _benchmark(
                    lambda: load_lammps_shooting_frames_for_conversion(
                        trajectory,
                        timesteps=benchmark_timesteps,
                        atom_count=int(manifest["atom_count"]),
                    ),
                    repetitions=int(args.benchmark_repetitions),
                ),
            }
            for dtype_name, binary in converted.items():
                benchmarks[dtype_name] = _benchmark(
                    lambda binary=binary: binary.load_frames(benchmark_timesteps),
                    repetitions=int(args.benchmark_repetitions),
                )
                benchmarks[dtype_name]["median_speedup_vs_text"] = (
                    benchmarks["text"]["median_seconds"]
                    / benchmarks[dtype_name]["median_seconds"]
                )
            branch_report["selected_frame_benchmark"] = benchmarks
        report["branch_reports"].append(branch_report)

    report_path = output_root / "conversion_report.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite conversion report: {report_path}")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[shooting-binary] complete report={report_path}", flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert strict-complete repository shooting branches into memory-mapped "
            "binary arrays without modifying the source campaign."
        )
    )
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--output-root", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--branch-index", type=int, action="append")
    selection.add_argument("--all-complete", action="store_true")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float16"),
        action="append",
        help="Storage dtype; repeat to create both. Defaults to float32 and float16.",
    )
    parser.add_argument("--precision-sample-count", type=int, default=1_000_000)
    parser.add_argument("--geometry-centers", type=int, default=512)
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--benchmark-repetitions", type=int, default=3)
    return parser


if __name__ == "__main__":
    _convert(_build_parser().parse_args())
