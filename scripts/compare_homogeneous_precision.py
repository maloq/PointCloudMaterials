#!/usr/bin/env python3
"""Compare completed matched FP32 and BF16 crystallization campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32-campaigns", type=Path, nargs=2, required=True)
    parser.add_argument("--bf16-campaigns", type=Path, nargs=2, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-png", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required comparison artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON mapping.")
    return value


def _load_campaign(root: Path) -> dict[str, object]:
    status = _json(root / "campaign_status.json")
    if status.get("status") != "complete":
        raise RuntimeError(
            f"{root}: comparison requires status='complete', got {status.get('status')!r}."
        )
    replicas = status.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != 1:
        raise RuntimeError(f"{root}: expected exactly one campaign replica.")
    replica_status = replicas[0]
    if (
        not isinstance(replica_status, dict)
        or replica_status.get("md_status") != "complete"
        or replica_status.get("analysis_status") != "complete"
    ):
        raise RuntimeError(f"{root}: MD and offline analysis must both be complete.")
    replica = root / "replicas" / "replica_000"
    metadata = _json(replica / "run_metadata.json")
    with np.load(replica / "trajectory.npz") as trajectory_file:
        trajectory = {key: trajectory_file[key].copy() for key in trajectory_file.files}
    with np.load(replica / "crystallization_progress.npz") as progress_file:
        progress = {key: progress_file[key].copy() for key in progress_file.files}
    event_value = replica_status.get("online_threshold_event_json")
    event = None if event_value is None else json.loads(str(event_value))
    return {
        "root": str(root.resolve()),
        "status": status,
        "metadata": metadata,
        "trajectory": trajectory,
        "progress": progress,
        "event": event,
    }


def _series_summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "final": float(values[-1]),
    }


def _common_values(
    fp32: dict[str, np.ndarray],
    bf16: dict[str, np.ndarray],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_steps, fp32_indices, bf16_indices = np.intersect1d(
        fp32["step"],
        bf16["step"],
        assume_unique=True,
        return_indices=True,
    )
    if common_steps.size != fp32["step"].size:
        raise RuntimeError(
            "BF16 three-samples-per-ps trajectory does not contain every FP32 "
            f"one-sample-per-ps step for field {key!r}."
        )
    return common_steps, fp32[key][fp32_indices], bf16[key][bf16_indices]


def _pair_summary(
    fp32: dict[str, object],
    bf16: dict[str, object],
) -> dict[str, object]:
    fp32_metadata = fp32["metadata"]
    bf16_metadata = bf16["metadata"]
    fp32_trajectory = fp32["trajectory"]
    bf16_trajectory = bf16["trajectory"]
    fp32_progress = fp32["progress"]
    bf16_progress = bf16["progress"]
    if not all(
        isinstance(value, dict)
        for value in (
            fp32_metadata,
            bf16_metadata,
            fp32_trajectory,
            bf16_trajectory,
            fp32_progress,
            bf16_progress,
        )
    ):
        raise TypeError("Loaded comparison payload has an invalid internal type.")

    thermodynamics: dict[str, object] = {}
    for key in (
        "temperature_K",
        "pressure_GPa",
        "volume_A3",
        "potential_energy_eV_per_atom",
    ):
        common_steps, fp32_values, bf16_values = _common_values(
            fp32_trajectory, bf16_trajectory, key
        )
        thermodynamics[key] = {
            "common_frame_count": int(common_steps.size),
            "fp32": _series_summary(fp32_values),
            "bf16": _series_summary(bf16_values),
            "bf16_minus_fp32_mean": float(
                np.mean(bf16_values) - np.mean(fp32_values)
            ),
        }

    _, fp32_crystalline, bf16_crystalline = _common_values(
        fp32_progress, bf16_progress, "crystalline_fraction"
    )
    _, fp32_largest, bf16_largest = _common_values(
        fp32_progress, bf16_progress, "largest_crystalline_cluster_atoms"
    )
    fp32_performance = fp32_metadata["calculator_performance"]
    bf16_performance = bf16_metadata["calculator_performance"]
    if not isinstance(fp32_performance, dict) or not isinstance(
        bf16_performance, dict
    ):
        raise TypeError("run_metadata calculator_performance must be a mapping.")
    bf16_snapshot = bf16_performance["graph_cache"]["worker_metrics_at_snapshot"]
    if not isinstance(bf16_snapshot, dict):
        raise TypeError("BF16 graph-cache snapshot must be a mapping.")

    return {
        "fp32_root": fp32["root"],
        "bf16_root": bf16["root"],
        "random_seed": int(fp32_metadata["random_seed"]),
        "events": {
            "fp32": fp32["event"],
            "bf16": bf16["event"],
        },
        "performance": {
            "fp32_steps_per_second": float(
                fp32_performance["measured_steps_per_second"]
            ),
            "bf16_steps_per_second": float(
                bf16_performance["measured_steps_per_second"]
            ),
            "speedup": float(
                bf16_performance["measured_steps_per_second"]
                / fp32_performance["measured_steps_per_second"]
            ),
            "bf16_peak_allocated_GiB": float(
                bf16_snapshot["cuda_max_memory_allocated_bytes"] / 2**30
            ),
            "bf16_peak_reserved_GiB": float(
                bf16_snapshot["cuda_max_memory_reserved_bytes"] / 2**30
            ),
        },
        "thermodynamics_at_common_integer_ps_frames": thermodynamics,
        "crystallization_at_common_integer_ps_frames": {
            "fp32_final_crystalline_fraction": float(fp32_crystalline[-1]),
            "bf16_final_crystalline_fraction": float(bf16_crystalline[-1]),
            "fp32_maximum_largest_cluster_atoms": int(np.max(fp32_largest)),
            "bf16_maximum_largest_cluster_atoms": int(np.max(bf16_largest)),
        },
    }


def _plot(
    pairs: list[tuple[dict[str, object], dict[str, object]]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for column, (fp32, bf16) in enumerate(pairs):
        seed = int(fp32["metadata"]["random_seed"])
        for campaign, label, color in (
            (fp32, "FP32", "#1f77b4"),
            (bf16, "BF16 autocast", "#d62728"),
        ):
            progress = campaign["progress"]
            axes[0, column].plot(
                progress["time_ps"],
                progress["crystalline_fraction"],
                label=label,
                color=color,
                linewidth=1.5,
            )
            axes[1, column].plot(
                progress["time_ps"],
                progress["largest_crystalline_cluster_atoms"],
                label=label,
                color=color,
                linewidth=1.5,
            )
        axes[0, column].set_title(f"Matched seed {seed}")
        axes[0, column].set_ylabel("Crystalline fraction")
        axes[1, column].set_ylabel("Largest cluster (atoms)")
        axes[1, column].set_xlabel("Measurement time (ps)")
        axes[0, column].legend()
        axes[1, column].legend()
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
    figure.suptitle(
        "70,304-atom MACE crystallization: FP32 versus BF16 autocast",
        fontsize=14,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = _arguments()
    fp32_campaigns = [_load_campaign(path.resolve()) for path in args.fp32_campaigns]
    bf16_campaigns = [_load_campaign(path.resolve()) for path in args.bf16_campaigns]
    pairs = list(zip(fp32_campaigns, bf16_campaigns, strict=True))
    summaries = [
        _pair_summary(fp32, bf16)
        for fp32, bf16 in pairs
    ]
    report = {
        "schema_version": 1,
        "interpretation": (
            "Matched initial source and velocity seeds isolate precision at step zero, "
            "but chaotic MD trajectories should be compared statistically rather than "
            "by pointwise coordinate agreement after divergence."
        ),
        "pairs": summaries,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot(pairs, args.output_png)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
