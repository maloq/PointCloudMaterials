#!/usr/bin/env python3
"""Compare verified in-progress BF16 checkpoints with completed matched FP32 runs."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from plot_homogeneous_checkpoint import (  # noqa: E402
    _latest_verified_snapshot,
    _load_online_arrays,
    _load_trace,
)
from src.data_utils.synthetic.atomistic.homogeneous_campaign_config import (  # noqa: E402
    load_homogeneous_campaign_config,
)
from src.data_utils.synthetic.atomistic.transition_analysis import (  # noqa: E402
    CRYSTALLINE_STRUCTURE_TYPES,
    STRUCTURE_COLORS,
    STRUCTURE_NAMES,
    _ptm_structure_types,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two verified running BF16 checkpoints with their completed "
            "matched-seed FP32 campaigns."
        )
    )
    parser.add_argument("--fp32-campaigns", type=Path, nargs=2, required=True)
    parser.add_argument("--bf16-campaign-configs", type=Path, nargs=2, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def _json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required comparison artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a JSON mapping.")
    return value


def _npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required comparison archive is missing: {path}")
    with np.load(path) as stored:
        return {key: stored[key].copy() for key in stored.files}


def _load_fp32(root: Path) -> dict[str, object]:
    root = root.resolve()
    status = _json(root / "campaign_status.json")
    if status.get("status") != "complete":
        raise RuntimeError(
            f"{root}: FP32 reference must be complete, got {status.get('status')!r}."
        )
    replicas = status.get("replicas")
    if not isinstance(replicas, list) or len(replicas) != 1:
        raise RuntimeError(f"{root}: expected exactly one FP32 replica.")
    replica_status = replicas[0]
    if not isinstance(replica_status, dict):
        raise TypeError(f"{root}: FP32 replica status must be a mapping.")
    replica = root / "replicas" / "replica_000"
    event_json = replica_status.get("online_threshold_event_json")
    return {
        "root": root,
        "metadata": _json(replica / "run_metadata.json"),
        "trajectory": _npz(replica / "trajectory.npz"),
        "progress": _npz(replica / "crystallization_progress.npz"),
        "event": None if event_json is None else json.loads(str(event_json)),
    }


def _load_bf16(config_path: Path) -> dict[str, object]:
    config = load_homogeneous_campaign_config(config_path.resolve())
    checkpoint_directory = config.output_root / "checkpoints" / "replica_000"
    snapshot, completed_global_step = _latest_verified_snapshot(checkpoint_directory)
    metadata = _json(snapshot / "metadata.json")
    if metadata.get("completed_global_step") != completed_global_step:
        raise RuntimeError(
            f"{snapshot}: metadata completed_global_step does not match its verified "
            "snapshot manifest."
        )
    trace = _load_trace(snapshot)
    online = _load_online_arrays(snapshot)
    equilibration_steps = config.homogeneous.equilibration_steps
    measurement_mask = trace.step >= equilibration_steps
    measurement_trace = {
        "step": trace.step[measurement_mask] - equilibration_steps,
        "time_ps": (
            (trace.step[measurement_mask] - equilibration_steps)
            * config.homogeneous.generator.dynamics.timestep_fs
            / 1000.0
        ),
        "temperature_K": trace.temperature_K[measurement_mask],
        "pressure_GPa": trace.pressure_GPa[measurement_mask],
        "volume_A3": trace.volume_A3[measurement_mask],
        "potential_energy_eV_per_atom": trace.potential_energy_eV_per_atom[
            measurement_mask
        ],
        "positions_A": trace.positions_A[measurement_mask],
        "cell_vectors_A": trace.cell_vectors_A[measurement_mask],
    }
    if not measurement_trace["step"].size:
        raise RuntimeError(f"{snapshot}: no post-equilibration trajectory frames exist.")
    expected_measurement_step = completed_global_step - equilibration_steps
    if int(measurement_trace["step"][-1]) != expected_measurement_step:
        raise RuntimeError(
            f"{snapshot}: latest measurement trajectory step="
            f"{int(measurement_trace['step'][-1])}, expected={expected_measurement_step}."
        )
    return {
        "config": config,
        "snapshot": snapshot,
        "completed_global_step": completed_global_step,
        "metadata": metadata,
        "trajectory": measurement_trace,
        "online": online,
    }


def _aligned_fp32_index(fp32_steps: np.ndarray, requested_step: int) -> int:
    indices = np.flatnonzero(fp32_steps == requested_step)
    if indices.size != 1:
        raise RuntimeError(
            f"FP32 trajectory must contain exactly one frame at measurement step "
            f"{requested_step}, found {indices.size}."
        )
    return int(indices[0])


def _event_steps(event: object, *, bf16_checkpoint: bool) -> tuple[int, int] | None:
    if event is None:
        return None
    if not isinstance(event, dict):
        raise TypeError(f"Crystallization event must be a mapping, got {type(event).__name__}.")
    onset_key = "onset_measurement_step" if bf16_checkpoint else "onset_step"
    confirmation_key = (
        "confirmation_measurement_step" if bf16_checkpoint else "confirmation_step"
    )
    return int(event[onset_key]), int(event[confirmation_key])


def _save_figure(figure: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    figure.savefig(temporary, dpi=190, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(output)


def _plot_crystallization(
    pairs: list[tuple[dict[str, object], dict[str, object]]], output: Path
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), sharex="col")
    for column, (fp32, bf16) in enumerate(pairs):
        fp32_progress = fp32["progress"]
        bf16_online = bf16["online"]
        metadata = bf16["metadata"]
        config = bf16["config"]
        if not all(
            isinstance(value, dict)
            for value in (fp32_progress, bf16_online, metadata)
        ):
            raise TypeError("Crystallization plotting payload has an invalid type.")
        timestep_fs = config.homogeneous.generator.dynamics.timestep_fs
        bf16_time_ps = bf16_online["measurement_step"] * timestep_fs / 1000.0
        endpoint_ps = float(bf16_time_ps[-1])
        seed = int(metadata["random_seed"])

        fraction_axis = axes[0, column]
        cluster_axis = axes[1, column]
        fraction_axis.plot(
            fp32_progress["time_ps"],
            fp32_progress["crystalline_fraction"],
            color="#457b9d",
            linewidth=2.0,
            label="FP32 complete (1 frame/ps)",
        )
        fraction_axis.plot(
            bf16_time_ps,
            bf16_online["crystalline_fraction"],
            color="#d1495b",
            linewidth=1.5,
            label="BF16 checkpoint (4 checks/ps)",
        )
        cluster_axis.plot(
            fp32_progress["time_ps"],
            fp32_progress["largest_crystalline_cluster_atoms"],
            color="#457b9d",
            linewidth=2.0,
            label="FP32 complete",
        )
        cluster_axis.plot(
            bf16_time_ps,
            bf16_online["largest_crystalline_cluster_atoms"],
            color="#d1495b",
            linewidth=1.5,
            label="BF16 checkpoint",
        )
        for axis in (fraction_axis, cluster_axis):
            axis.axvline(
                endpoint_ps,
                color="#d1495b",
                linestyle=":",
                linewidth=1.2,
                label="BF16 available through" if axis is fraction_axis else None,
            )
            axis.axvspan(endpoint_ps, 130.0, color="#adb5bd", alpha=0.14)
            axis.grid(alpha=0.25)
            axis.set_xlim(0.0, 130.0)

        fp32_event = _event_steps(fp32["event"], bf16_checkpoint=False)
        bf16_event = _event_steps(metadata.get("online_event"), bf16_checkpoint=True)
        if fp32_event is not None:
            fraction_axis.axvline(
                fp32_event[0] * timestep_fs / 1000.0,
                color="#457b9d",
                linestyle="--",
                linewidth=1.0,
            )
        if bf16_event is not None:
            fraction_axis.axvline(
                bf16_event[0] * timestep_fs / 1000.0,
                color="#d1495b",
                linestyle="--",
                linewidth=1.0,
            )
        speed = float(metadata["calculator_performance"]["measured_steps_per_second"])
        fp32_speed = float(fp32["metadata"]["calculator_performance"]["measured_steps_per_second"])
        fraction_axis.set_title(
            f"velocity seed {seed}: BF16 {endpoint_ps:.0f}/130 ps, "
            f"speedup {speed / fp32_speed:.3f}×"
        )
        fraction_axis.set_ylabel("crystalline fraction")
        cluster_axis.set_ylabel("largest crystalline cluster (atoms)")
        cluster_axis.set_xlabel("measurement time (ps)")
        fraction_axis.legend(loc="upper left")
        cluster_axis.legend(loc="upper left")

    figure.suptitle(
        "70,304-atom MACE crystallization: current selective-BF16 checkpoints "
        "versus matched FP32",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.01,
        "Gray region has no BF16 data yet. Dashed event lines use the configured "
        "persistent 100-atom threshold. Trajectories are chaotic; compare kinetics "
        "and distributions, not pointwise coordinates.",
        ha="center",
        fontsize=9,
        color="#495057",
    )
    figure.subplots_adjust(top=0.91, bottom=0.09, hspace=0.08, wspace=0.18)
    _save_figure(figure, output)


def _plot_thermodynamics(
    pairs: list[tuple[dict[str, object], dict[str, object]]], output: Path
) -> None:
    fields = (
        ("temperature_K", "temperature (K)"),
        ("pressure_GPa", "pressure (GPa)"),
        ("volume_A3", "volume (Å³)"),
        ("potential_energy_eV_per_atom", "potential energy (eV/atom)"),
    )
    figure, axes = plt.subplots(4, 2, figsize=(14.5, 12.0), sharex="col")
    for column, (fp32, bf16) in enumerate(pairs):
        fp32_trace = fp32["trajectory"]
        bf16_trace = bf16["trajectory"]
        metadata = bf16["metadata"]
        if not all(isinstance(value, dict) for value in (fp32_trace, bf16_trace, metadata)):
            raise TypeError("Thermodynamic plotting payload has an invalid type.")
        endpoint_ps = float(bf16_trace["time_ps"][-1])
        for row, (field, ylabel) in enumerate(fields):
            axis = axes[row, column]
            axis.plot(
                fp32_trace["step"] / 1000.0,
                fp32_trace[field],
                color="#457b9d",
                linewidth=1.8,
                label="FP32",
            )
            axis.plot(
                bf16_trace["time_ps"],
                bf16_trace[field],
                color="#d1495b",
                linewidth=1.1,
                label="BF16",
            )
            axis.axvline(endpoint_ps, color="#d1495b", linestyle=":", linewidth=1.0)
            axis.axvspan(endpoint_ps, 130.0, color="#adb5bd", alpha=0.14)
            axis.set_ylabel(ylabel)
            axis.set_xlim(0.0, 130.0)
            axis.grid(alpha=0.23)
            if row == 0:
                axis.set_title(f"velocity seed {int(metadata['random_seed'])}")
                axis.legend(loc="best")
        axes[-1, column].set_xlabel("measurement time (ps)")
    figure.suptitle(
        "Thermodynamics: current selective-BF16 checkpoints versus matched FP32",
        fontsize=14,
    )
    figure.subplots_adjust(top=0.94, hspace=0.08, wspace=0.2)
    _save_figure(figure, output)


def _plot_h100_validation(validation: dict[str, object], output: Path) -> None:
    if validation.get("passed") is not True or validation.get("atom_count") != 70304:
        raise RuntimeError(
            "The H100 comparison plot requires a passed 70,304-atom validation report."
        )
    comparisons = validation.get("comparisons")
    thresholds = validation.get("thresholds")
    fp32 = validation.get("fp32")
    bf16 = validation.get("bf16_autocast")
    if not all(isinstance(value, dict) for value in (comparisons, thresholds, fp32, bf16)):
        raise TypeError("H100 validation report comparison sections must be mappings.")

    fp32_speed = float(comparisons["md_steps_per_second_fp32"])
    bf16_speed = float(comparisons["md_steps_per_second_bf16"])
    fp32_memory = float(fp32["md_benchmark"]["memory"]["peak_reserved_bytes"] / 2**30)
    bf16_memory = float(
        bf16["md_benchmark"]["memory"]["peak_reserved_bytes"] / 2**30
    )
    normalized_errors = np.asarray(
        [
            comparisons["energy_difference_meV_per_atom"]
            / thresholds["maximum_energy_difference_meV_per_atom"],
            comparisons["force_rmse_eV_per_A"]
            / thresholds["maximum_force_rmse_eV_per_A"],
            comparisons["maximum_stress_difference_GPa"]
            / thresholds["maximum_stress_difference_GPa"],
        ],
        dtype=np.float64,
    )

    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    colors = ("#457b9d", "#d1495b")
    axes[0].bar(("FP32", "BF16"), (fp32_speed, bf16_speed), color=colors)
    axes[0].set_ylabel("MD throughput (steps/s)")
    axes[0].set_title(f"throughput: {bf16_speed / fp32_speed:.3f}×")
    for index, value in enumerate((fp32_speed, bf16_speed)):
        axes[0].text(index, value, f"{value:.3f}", ha="center", va="bottom")

    axes[1].bar(("FP32", "BF16"), (fp32_memory, bf16_memory), color=colors)
    axes[1].set_ylabel("peak CUDA reserved memory (GiB)")
    axes[1].set_title(
        f"memory: {100.0 * (1.0 - bf16_memory / fp32_memory):.1f}% lower"
    )
    for index, value in enumerate((fp32_memory, bf16_memory)):
        axes[1].text(index, value, f"{value:.1f}", ha="center", va="bottom")

    error_labels = ("energy/atom", "force RMSE", "stress")
    axes[2].bar(error_labels, 100.0 * normalized_errors, color="#2a9d8f")
    axes[2].axhline(100.0, color="#d62828", linestyle="--", label="gate limit")
    axes[2].set_ylabel("error as % of validation limit")
    axes[2].set_title("selective-BF16 parity gate: passed")
    axes[2].tick_params(axis="x", rotation=18)
    axes[2].legend(loc="upper right")
    for index, value in enumerate(100.0 * normalized_errors):
        axes[2].text(index, value, f"{value:.1f}%", ha="center", va="bottom")

    device = validation.get("device")
    device_name = device.get("name") if isinstance(device, dict) else "H100"
    figure.suptitle(
        f"Matched full-graph validation: 70,304 atoms on {device_name}", fontsize=14
    )
    figure.subplots_adjust(top=0.82, bottom=0.2, wspace=0.32)
    _save_figure(figure, output)


def _plot_structure_slices(
    pairs: list[tuple[dict[str, object], dict[str, object]]], output: Path
) -> list[dict[str, object]]:
    figure, axes = plt.subplots(2, 2, figsize=(13.5, 12.0))
    structure_summaries: list[dict[str, object]] = []
    for row, (fp32, bf16) in enumerate(pairs):
        fp32_trace = fp32["trajectory"]
        bf16_trace = bf16["trajectory"]
        metadata = bf16["metadata"]
        config = bf16["config"]
        if not all(isinstance(value, dict) for value in (fp32_trace, bf16_trace, metadata)):
            raise TypeError("Structure plotting payload has an invalid type.")
        measurement_step = int(bf16_trace["step"][-1])
        fp32_index = _aligned_fp32_index(fp32_trace["step"], measurement_step)
        states = (
            (
                "FP32",
                fp32_trace["positions_A"][fp32_index],
                fp32_trace["cell_vectors_A"][fp32_index],
            ),
            (
                "selective BF16",
                bf16_trace["positions_A"][-1],
                bf16_trace["cell_vectors_A"][-1],
            ),
        )
        seed_summary: dict[str, object] = {
            "random_seed": int(metadata["random_seed"]),
            "measurement_step": measurement_step,
            "measurement_time_ps": float(bf16_trace["time_ps"][-1]),
        }
        for column, (label, positions_A, cell_vectors_A) in enumerate(states):
            atoms = Atoms(
                numbers=np.full(
                    positions_A.shape[0],
                    atomic_numbers[config.homogeneous.generator.system.chemical_symbol],
                    dtype=np.int32,
                ),
                positions=positions_A,
                cell=cell_vectors_A,
                pbc=True,
            )
            structure_types = _ptm_structure_types(
                atoms, config.homogeneous.analysis.ptm_rmsd_cutoff
            )
            crystalline_fraction = float(
                np.mean(np.isin(structure_types, CRYSTALLINE_STRUCTURE_TYPES))
            )
            seed_summary[f"{label.replace(' ', '_').lower()}_ptm_crystalline_fraction"] = (
                crystalline_fraction
            )
            positions = atoms.get_positions(wrap=True)
            scaled = atoms.get_scaled_positions(wrap=True)
            cell_lengths = np.linalg.norm(np.asarray(atoms.cell), axis=1)
            slice_half_width_A = max(2.5, 0.08 * float(cell_lengths[1]))
            slice_mask = (
                np.abs(scaled[:, 1] - 0.5) * cell_lengths[1] <= slice_half_width_A
            )
            axis = axes[row, column]
            for structure_id, (structure_name, color) in enumerate(
                zip(STRUCTURE_NAMES, STRUCTURE_COLORS, strict=True)
            ):
                mask = slice_mask & (structure_types == structure_id)
                axis.scatter(
                    positions[mask, 0],
                    positions[mask, 2],
                    s=4.5,
                    alpha=0.82,
                    linewidths=0.0,
                    color=color,
                    label=structure_name.upper(),
                    rasterized=True,
                )
            axis.set(
                xlim=(0.0, cell_lengths[0]),
                ylim=(0.0, cell_lengths[2]),
                xlabel="x (Å)",
                ylabel="z (Å)",
                title=(
                    f"seed {int(metadata['random_seed'])} — {label} — "
                    f"t={float(bf16_trace['time_ps'][-1]):.0f} ps\n"
                    f"PTM crystalline fraction={crystalline_fraction:.3f}"
                ),
            )
            axis.set_aspect("equal", adjustable="box")
        structure_summaries.append(seed_summary)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(STRUCTURE_NAMES))
    figure.suptitle(
        "Matched-time central-y atom slices: FP32 versus selective BF16\n"
        "PTM classification is evaluated locally for every atom",
        fontsize=14,
    )
    figure.subplots_adjust(top=0.91, bottom=0.09, hspace=0.22, wspace=0.16)
    _save_figure(figure, output)
    return structure_summaries


def _pair_summary(
    fp32: dict[str, object], bf16: dict[str, object]
) -> dict[str, object]:
    fp32_progress = fp32["progress"]
    fp32_trace = fp32["trajectory"]
    fp32_metadata = fp32["metadata"]
    bf16_online = bf16["online"]
    bf16_trace = bf16["trajectory"]
    bf16_metadata = bf16["metadata"]
    if not all(
        isinstance(value, dict)
        for value in (
            fp32_progress,
            fp32_trace,
            fp32_metadata,
            bf16_online,
            bf16_trace,
            bf16_metadata,
        )
    ):
        raise TypeError("Comparison payload has an invalid internal type.")
    seed = int(bf16_metadata["random_seed"])
    if int(fp32_metadata["random_seed"]) != seed:
        raise RuntimeError(
            f"Precision comparison requires matched velocity seeds, got FP32="
            f"{fp32_metadata['random_seed']} and BF16={seed}."
        )
    latest_measurement_step = int(bf16_online["measurement_step"][-1])
    fp32_progress_index = _aligned_fp32_index(
        fp32_progress["step"], latest_measurement_step
    )
    fp32_speed = float(
        fp32_metadata["calculator_performance"]["measured_steps_per_second"]
    )
    bf16_performance = bf16_metadata["calculator_performance"]
    bf16_speed = float(bf16_performance["measured_steps_per_second"])
    bf16_memory = bf16_performance["graph_cache"]["worker_metrics_at_snapshot"]

    common_steps, fp32_indices, bf16_indices = np.intersect1d(
        fp32_trace["step"],
        bf16_trace["step"],
        assume_unique=True,
        return_indices=True,
    )
    if not common_steps.size or int(common_steps[-1]) != latest_measurement_step:
        raise RuntimeError(
            f"Seed {seed}: common thermodynamic frames do not reach the latest verified "
            f"measurement step {latest_measurement_step}."
        )
    thermodynamics: dict[str, object] = {}
    for field in (
        "temperature_K",
        "pressure_GPa",
        "volume_A3",
        "potential_energy_eV_per_atom",
    ):
        fp32_values = fp32_trace[field][fp32_indices]
        bf16_values = bf16_trace[field][bf16_indices]
        thermodynamics[field] = {
            "common_frame_count": int(common_steps.size),
            "fp32_mean": float(np.mean(fp32_values)),
            "bf16_mean": float(np.mean(bf16_values)),
            "bf16_minus_fp32_mean": float(
                np.mean(bf16_values) - np.mean(fp32_values)
            ),
        }
    return {
        "random_seed": seed,
        "fp32_root": str(fp32["root"]),
        "bf16_root": str(bf16["config"].output_root),
        "bf16_verified_snapshot": str(bf16["snapshot"]),
        "bf16_completed_global_step": int(bf16["completed_global_step"]),
        "bf16_latest_measurement_step": latest_measurement_step,
        "bf16_latest_measurement_time_ps": latest_measurement_step / 1000.0,
        "planned_measurement_time_ps": 130.0,
        "events": {
            "fp32": fp32["event"],
            "bf16": bf16_metadata.get("online_event"),
        },
        "performance": {
            "fp32_steps_per_second": fp32_speed,
            "bf16_steps_per_second": bf16_speed,
            "bf16_speedup": bf16_speed / fp32_speed,
            "bf16_peak_allocated_GiB": float(
                bf16_memory["cuda_max_memory_allocated_bytes"] / 2**30
            ),
            "bf16_peak_reserved_GiB": float(
                bf16_memory["cuda_max_memory_reserved_bytes"] / 2**30
            ),
        },
        "crystallization_at_latest_common_online_frame": {
            "fp32_crystalline_fraction": float(
                fp32_progress["crystalline_fraction"][fp32_progress_index]
            ),
            "bf16_crystalline_fraction": float(bf16_online["crystalline_fraction"][-1]),
            "fp32_largest_cluster_atoms": int(
                fp32_progress["largest_crystalline_cluster_atoms"][fp32_progress_index]
            ),
            "bf16_largest_cluster_atoms": int(
                bf16_online["largest_crystalline_cluster_atoms"][-1]
            ),
        },
        "thermodynamics_at_common_integer_ps_frames": thermodynamics,
    }


def _copy_stamped(path: Path, suffix: str) -> Path:
    stamped = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    temporary = stamped.with_name(f".{stamped.stem}.tmp{stamped.suffix}")
    shutil.copy2(path, temporary)
    temporary.replace(stamped)
    return stamped


def main() -> None:
    args = _arguments()
    fp32_campaigns = [_load_fp32(path) for path in args.fp32_campaigns]
    bf16_campaigns = [_load_bf16(path) for path in args.bf16_campaign_configs]
    validation = _json(args.validation_report.resolve())
    pairs = list(zip(fp32_campaigns, bf16_campaigns, strict=True))
    summaries = [_pair_summary(fp32, bf16) for fp32, bf16 in pairs]

    output_directory = args.output_directory.resolve()
    crystallization_path = output_directory / "in_progress_crystallization.png"
    thermodynamics_path = output_directory / "in_progress_thermodynamics.png"
    structures_path = output_directory / "in_progress_matched_structure_slices.png"
    validation_path = output_directory / "full_graph_h100_performance.png"
    report_path = output_directory / "in_progress_comparison.json"
    _plot_crystallization(pairs, crystallization_path)
    _plot_thermodynamics(pairs, thermodynamics_path)
    structure_summaries = _plot_structure_slices(pairs, structures_path)
    _plot_h100_validation(validation, validation_path)

    report = {
        "schema_version": 1,
        "status": "in_progress",
        "interpretation": (
            "Each pair has the same liquid source and velocity seed. Chaotic MD rapidly "
            "destroys pointwise trajectory correspondence, so compare event timing, "
            "growth kinetics, thermodynamic distributions, and matched-time structures. "
            "Gray plot regions explicitly contain no BF16 data yet."
        ),
        "pairs": summaries,
        "matched_structure_ptm": structure_summaries,
        "full_graph_h100_validation_report": str(args.validation_report.resolve()),
        "full_graph_h100_validation": {
            "passed": validation["passed"],
            "atom_count": validation["atom_count"],
            "device": validation["device"],
            "thresholds": validation["thresholds"],
            "comparisons": validation["comparisons"],
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report = report_path.with_name(f".{report_path.stem}.tmp.json")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_report.replace(report_path)

    stamp = "_".join(
        f"seed{summary['random_seed']}_step{summary['bf16_completed_global_step']:012d}"
        for summary in summaries
    )
    outputs = [
        crystallization_path,
        thermodynamics_path,
        structures_path,
        validation_path,
        report_path,
    ]
    stamped_outputs = [_copy_stamped(path, stamp) for path in outputs]
    for path in outputs + stamped_outputs:
        print(f"Wrote {path}")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
