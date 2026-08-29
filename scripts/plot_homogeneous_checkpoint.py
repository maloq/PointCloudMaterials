#!/usr/bin/env python3
"""Plot a verified in-progress homogeneous-crystallization checkpoint."""

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
from ase import units
from ase.data import atomic_masses, atomic_numbers


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data_utils.synthetic.atomistic.homogeneous_analysis import (  # noqa: E402
    first_persistent_threshold_run,
)
from src.data_utils.synthetic.atomistic.homogeneous_campaign_config import (  # noqa: E402
    HomogeneousCampaignConfig,
    load_homogeneous_campaign_config,
)
from src.data_utils.synthetic.atomistic.homogeneous_resumable import (  # noqa: E402
    _load_and_verify_named_snapshot,
)
from src.data_utils.synthetic.atomistic.simulation import (  # noqa: E402
    ThermodynamicTrace,
    validate_thermodynamic_trace,
)


TRACE_KEYS = {
    "step",
    "temperature_K",
    "pressure_GPa",
    "volume_A3",
    "potential_energy_eV_per_atom",
    "positions_A",
    "cell_vectors_A",
}
ONLINE_KEYS = {
    "measurement_step",
    "crystalline_fraction",
    "crystalline_cluster_count",
    "largest_crystalline_cluster_atoms",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render a thermodynamic/crystallinity dashboard from the latest verified "
            "checkpoint without modifying the running campaign."
        )
    )
    parser.add_argument("--campaign-config", required=True, type=Path)
    parser.add_argument("--replica-name", default="replica_000")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "PNG path. By default, write <campaign output>/visualizations/"
            "<replica>_checkpoint_dashboard.png."
        ),
    )
    parser.add_argument(
        "--include-structure-slices",
        action="store_true",
        help=(
            "Also render PTM-colored initial/middle/latest structure slices from the "
            "same hash-verified checkpoint."
        ),
    )
    parser.add_argument(
        "--structure-output",
        type=Path,
        default=None,
        help=(
            "Structure-slice PNG path. Supplying this option implies "
            "--include-structure-slices. By default, write <campaign output>/"
            "visualizations/<replica>_checkpoint_structure_slices.png."
        ),
    )
    parser.add_argument(
        "--step-stamped",
        action="store_true",
        help=(
            "Additionally preserve immutable copies whose names include the verified "
            "global checkpoint step."
        ),
    )
    return parser.parse_args()


def _latest_verified_snapshot(checkpoint_directory: Path) -> tuple[Path, int]:
    latest_path = checkpoint_directory / "LATEST"
    if not latest_path.is_file():
        raise FileNotFoundError(
            f"No committed checkpoint pointer exists at {latest_path}. Wait for the first "
            "chunk checkpoint before plotting."
        )
    snapshot_name = latest_path.read_text(encoding="utf-8").strip()
    if not snapshot_name or Path(snapshot_name).name != snapshot_name:
        raise RuntimeError(
            f"{latest_path}: expected one checkpoint directory name, got "
            f"{snapshot_name!r}."
        )
    snapshot = checkpoint_directory / snapshot_name
    manifest = _load_and_verify_named_snapshot(snapshot)
    return snapshot, int(manifest["completed_global_step"])


def _load_trace(snapshot: Path) -> ThermodynamicTrace:
    path = snapshot / "trace.npz"
    with np.load(path) as stored:
        observed_keys = set(stored.files)
        if observed_keys != TRACE_KEYS:
            raise RuntimeError(
                f"{path}: expected arrays={sorted(TRACE_KEYS)}, got "
                f"{sorted(observed_keys)}."
            )
        trace = ThermodynamicTrace(
            step=stored["step"].copy(),
            temperature_K=stored["temperature_K"].copy(),
            pressure_GPa=stored["pressure_GPa"].copy(),
            volume_A3=stored["volume_A3"].copy(),
            potential_energy_eV_per_atom=stored[
                "potential_energy_eV_per_atom"
            ].copy(),
            positions_A=stored["positions_A"].copy(),
            cell_vectors_A=stored["cell_vectors_A"].copy(),
        )
    validate_thermodynamic_trace(
        trace,
        atom_count=trace.positions_A.shape[1],
        context=f"checkpoint dashboard input {path}",
    )
    return trace


def _load_online_arrays(snapshot: Path) -> dict[str, np.ndarray]:
    path = snapshot / "online_crystallinity.npz"
    with np.load(path) as stored:
        observed_keys = set(stored.files)
        if observed_keys != ONLINE_KEYS:
            raise RuntimeError(
                f"{path}: expected arrays={sorted(ONLINE_KEYS)}, got "
                f"{sorted(observed_keys)}."
            )
        arrays = {name: stored[name].copy() for name in ONLINE_KEYS}
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1 or not arrays["measurement_step"].size:
        raise RuntimeError(
            f"{path}: online crystallinity arrays must have one shared nonzero length; "
            f"got {lengths}."
        )
    if np.any(np.diff(arrays["measurement_step"]) <= 0):
        raise RuntimeError(
            f"{path}: measurement_step must increase strictly, got "
            f"{arrays['measurement_step'].tolist()}."
        )
    return arrays


def _retained_checkpoint_steps(checkpoint_directory: Path) -> tuple[int, ...]:
    steps: list[int] = []
    for path in checkpoint_directory.glob("step_*"):
        if path.is_dir() and path.name.removeprefix("step_").isdigit():
            steps.append(int(path.name.removeprefix("step_")))
    return tuple(sorted(steps))


def _event_steps(
    online: dict[str, np.ndarray],
    *,
    event_cadence_steps: int,
    threshold_atoms: int,
    persistence_frames: int,
) -> tuple[int, int] | None:
    steps = online["measurement_step"]
    on_event_cadence = steps % event_cadence_steps == 0
    event_steps = steps[on_event_cadence]
    event_largest = online["largest_crystalline_cluster_atoms"][on_event_cadence]
    indices = first_persistent_threshold_run(
        event_largest,
        threshold=threshold_atoms,
        persistence_frames=persistence_frames,
    )
    if indices is None:
        return None
    onset_index, confirmation_index = indices
    return int(event_steps[onset_index]), int(event_steps[confirmation_index])


def _plot_dashboard(
    output: Path,
    *,
    trace: ThermodynamicTrace,
    online: dict[str, np.ndarray],
    checkpoint_steps: tuple[int, ...],
    completed_global_step: int,
    replica_name: str,
    model_name: str,
    chemical_symbol: str,
    timestep_fs: float,
    equilibration_steps: int,
    planned_measurement_steps: int,
    sample_interval: int,
    target_temperature_K: float,
    target_pressure_GPa: float,
    maximum_liquid_crystalline_fraction: float | None,
    nucleus_size_threshold_atoms: int,
    threshold_persistence_frames: int,
    simulation_kind: str = "homogeneous crystallization",
) -> None:
    global_time_ps = trace.step * timestep_fs / 1000.0
    measurement_global_steps = equilibration_steps + online["measurement_step"]
    measurement_global_time_ps = measurement_global_steps * timestep_fs / 1000.0
    equilibration_time_ps = equilibration_steps * timestep_fs / 1000.0
    completed_time_ps = completed_global_step * timestep_fs / 1000.0
    checkpoint_times_ps = np.asarray(checkpoint_steps, dtype=np.float64) * (
        timestep_fs / 1000.0
    )

    atom_count = trace.positions_A.shape[1]
    mass_amu = atomic_masses[atomic_numbers[chemical_symbol]]
    density_g_cm3 = (
        atom_count * mass_amu * units._amu / (trace.volume_A3 * 1.0e-30) / 1000.0
    )
    energy_change_meV_per_atom = 1000.0 * (
        trace.potential_energy_eV_per_atom
        - trace.potential_energy_eV_per_atom[0]
    )
    measured_steps = max(0, completed_global_step - equilibration_steps)
    completion_fraction = min(1.0, measured_steps / planned_measurement_steps)
    event = _event_steps(
        online,
        event_cadence_steps=sample_interval,
        threshold_atoms=nucleus_size_threshold_atoms,
        persistence_frames=threshold_persistence_frames,
    )

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.22,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    figure, axes = plt.subplots(
        3,
        2,
        figsize=(15.5, 12.0),
        sharex=True,
        constrained_layout=False,
    )

    temperature_axis = axes[0, 0]
    temperature_axis.plot(
        global_time_ps,
        trace.temperature_K,
        color="#e76f51",
        marker="o",
        markersize=3.5,
        label="checkpoint trace",
    )
    temperature_axis.axhline(
        target_temperature_K,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"target {target_temperature_K:g} K",
    )
    temperature_axis.set_ylabel("temperature (K)")
    temperature_axis.legend(loc="best")

    pressure_axis = axes[0, 1]
    pressure_axis.plot(
        global_time_ps,
        trace.pressure_GPa,
        color="#277da1",
        marker="o",
        markersize=3.5,
    )
    pressure_axis.axhline(
        target_pressure_GPa,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"target {target_pressure_GPa:g} GPa",
    )
    pressure_axis.set_ylabel("pressure (GPa)")
    pressure_axis.legend(loc="best")

    density_axis = axes[1, 0]
    density_axis.plot(
        global_time_ps,
        density_g_cm3,
        color="#2a9d8f",
        marker="o",
        markersize=3.5,
    )
    density_axis.set_ylabel("mass density (g cm$^{-3}$)")

    energy_axis = axes[1, 1]
    energy_axis.plot(
        global_time_ps,
        energy_change_meV_per_atom,
        color="#f4a261",
        marker="o",
        markersize=3.5,
    )
    energy_axis.axhline(0.0, color="black", linewidth=0.8)
    energy_axis.set_ylabel("potential-energy change (meV/atom)")

    fraction_axis = axes[2, 0]
    fraction_axis.plot(
        measurement_global_time_ps,
        100.0 * online["crystalline_fraction"],
        color="#264653",
        marker="o",
        markersize=3.0,
        label="FCC+HCP+BCC",
    )
    observed_crystalline_percent = 100.0 * float(
        np.max(online["crystalline_fraction"])
    )
    fraction_y_max = max(1.0, 1.15 * observed_crystalline_percent)
    if maximum_liquid_crystalline_fraction is not None:
        liquid_limit_percent = 100.0 * maximum_liquid_crystalline_fraction
        if observed_crystalline_percent >= 0.5 * liquid_limit_percent:
            fraction_y_max = max(fraction_y_max, 1.05 * liquid_limit_percent)
            fraction_axis.axhline(
                liquid_limit_percent,
                color="#d62828",
                linestyle="--",
                linewidth=1.0,
                label="source liquid limit",
            )
        else:
            fraction_axis.text(
                0.99,
                0.96,
                f"source liquid limit: {liquid_limit_percent:g}% (off-scale)",
                transform=fraction_axis.transAxes,
                ha="right",
                va="top",
                color="#d62828",
            )
    fraction_axis.set(
        ylabel="crystalline atoms (%)",
        ylim=(0.0, fraction_y_max),
    )
    fraction_axis.legend(loc="best")

    cluster_axis = axes[2, 1]
    cluster_axis.plot(
        measurement_global_time_ps,
        online["largest_crystalline_cluster_atoms"],
        color="#6a4c93",
        marker="o",
        markersize=3.0,
        label="largest cluster",
    )
    cluster_axis.axhline(
        nucleus_size_threshold_atoms,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=(
            f"nucleus threshold ({nucleus_size_threshold_atoms} atoms, "
            f"{threshold_persistence_frames} frames)"
        ),
    )
    cluster_count_axis = cluster_axis.twinx()
    cluster_count_axis.spines["right"].set_visible(True)
    cluster_count_axis.plot(
        measurement_global_time_ps,
        online["crystalline_cluster_count"],
        color="#90be6d",
        alpha=0.65,
        linewidth=1.0,
        label="cluster count",
    )
    cluster_axis.set(
        ylabel="largest crystalline cluster (atoms)",
    )
    cluster_count_axis.set_ylabel("number of crystalline clusters")
    handles, labels = cluster_axis.get_legend_handles_labels()
    count_handles, count_labels = cluster_count_axis.get_legend_handles_labels()
    cluster_axis.legend(handles + count_handles, labels + count_labels, loc="best")

    for axis in axes.flat:
        axis.axvspan(
            0.0,
            equilibration_time_ps,
            color="#adb5bd",
            alpha=0.13,
            linewidth=0.0,
        )
        for checkpoint_time_ps in checkpoint_times_ps:
            axis.axvline(
                checkpoint_time_ps,
                color="#6c757d",
                linestyle=":",
                linewidth=0.75,
                alpha=0.65,
            )
        axis.set_xlim(0.0, max(completed_time_ps, equilibration_time_ps))

    if event is not None:
        onset_step, confirmation_step = event
        onset_time_ps = (equilibration_steps + onset_step) * timestep_fs / 1000.0
        confirmation_time_ps = (
            (equilibration_steps + confirmation_step) * timestep_fs / 1000.0
        )
        for axis in (fraction_axis, cluster_axis):
            axis.axvline(onset_time_ps, color="#d62828", linewidth=1.2)
            axis.axvline(
                confirmation_time_ps,
                color="#d62828",
                linewidth=1.2,
                linestyle="--",
            )
        event_summary = (
            f"persistent nucleus onset={onset_step * timestep_fs / 1000.0:.2f} ps "
            f"measurement time, confirmed="
            f"{confirmation_step * timestep_fs / 1000.0:.2f} ps"
        )
    else:
        event_summary = "no persistent threshold-sized nucleus in this checkpoint"

    temperature_axis.text(
        equilibration_time_ps / 2.0,
        0.03,
        "equilibration",
        transform=temperature_axis.get_xaxis_transform(),
        ha="center",
        color="#495057",
    )
    run_state = (
        "completed"
        if completed_global_step
        >= equilibration_steps + planned_measurement_steps
        else "in-progress"
    )
    figure.suptitle(
        f"{replica_name}: verified {run_state} {simulation_kind} dashboard\n"
        f"{model_name} | checkpoint step {completed_global_step:,} "
        f"({completed_time_ps:.2f} ps global; {100.0 * completion_fraction:.1f}% of "
        f"planned measurement)\n{event_summary}",
        fontsize=13,
        y=0.99,
    )
    figure.supxlabel("global simulation time (ps)", y=0.035)
    figure.text(
        0.5,
        0.009,
        "Gray band: equilibration. Dotted lines: retained hash-verified checkpoints. "
        "Crystallinity begins after equilibration and is sampled more densely than "
        "thermodynamics.",
        ha="center",
        fontsize=9,
        color="#495057",
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.925,
        bottom=0.075,
        top=0.875,
        hspace=0.06,
        wspace=0.14,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.png")
    figure.savefig(temporary, dpi=180, format="png")
    plt.close(figure)
    temporary.replace(output)


def _resolved_output(path: Path) -> Path:
    return path if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _step_stamped_path(path: Path, completed_global_step: int) -> Path:
    return path.with_name(
        f"{path.stem}_step_{completed_global_step:012d}{path.suffix}"
    )


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    shutil.copy2(source, temporary)
    temporary.replace(destination)


def render_checkpoint_visualizations(
    config: HomogeneousCampaignConfig,
    *,
    replica_name: str = "replica_000",
    dashboard_output: Path | None = None,
    include_structure_slices: bool = False,
    structure_output: Path | None = None,
    step_stamped: bool = False,
) -> tuple[Path, int, tuple[Path, ...]]:
    """Render requested views from one hash-verified checkpoint snapshot."""

    checkpoint_directory = config.output_root / "checkpoints" / replica_name
    snapshot, completed_global_step = _latest_verified_snapshot(checkpoint_directory)
    trace = _load_trace(snapshot)
    online = _load_online_arrays(snapshot)
    with (snapshot / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected_metadata = {
        "replica_name": replica_name,
        "completed_global_step": completed_global_step,
        "equilibration_steps": config.homogeneous.equilibration_steps,
        "planned_measurement_steps": config.homogeneous.steps,
    }
    mismatches = {
        key: {"observed": metadata.get(key), "expected": expected}
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"{snapshot / 'metadata.json'}: checkpoint metadata differs from the "
            f"campaign: {mismatches}."
        )
    if dashboard_output is None:
        dashboard_output = (
            config.output_root
            / "visualizations"
            / f"{replica_name}_checkpoint_dashboard.png"
        )
    else:
        dashboard_output = _resolved_output(dashboard_output)

    homogeneous = config.homogeneous
    _plot_dashboard(
        dashboard_output,
        trace=trace,
        online=online,
        checkpoint_steps=_retained_checkpoint_steps(checkpoint_directory),
        completed_global_step=completed_global_step,
        replica_name=replica_name,
        model_name=homogeneous.generator.potential.model_name,
        chemical_symbol=homogeneous.generator.system.chemical_symbol,
        timestep_fs=homogeneous.generator.dynamics.timestep_fs,
        equilibration_steps=homogeneous.equilibration_steps,
        planned_measurement_steps=homogeneous.steps,
        sample_interval=homogeneous.sample_interval,
        target_temperature_K=homogeneous.temperature_K,
        target_pressure_GPa=homogeneous.generator.dynamics.pressure_GPa,
        maximum_liquid_crystalline_fraction=(
            homogeneous.generator.validation.maximum_liquid_crystalline_fraction
        ),
        nucleus_size_threshold_atoms=(
            homogeneous.analysis.nucleus_size_threshold_atoms
        ),
        threshold_persistence_frames=(
            homogeneous.analysis.threshold_persistence_frames
        ),
    )
    outputs = [dashboard_output]

    if include_structure_slices or structure_output is not None:
        from src.data_utils.synthetic.atomistic.transition_analysis import (
            write_structure_slice_visualization,
        )

        if structure_output is None:
            structure_output = (
                config.output_root
                / "visualizations"
                / f"{replica_name}_checkpoint_structure_slices.png"
            )
        else:
            structure_output = _resolved_output(structure_output)
        structure_output.parent.mkdir(parents=True, exist_ok=True)
        temporary_structure_output = structure_output.with_name(
            f".{structure_output.stem}.tmp{structure_output.suffix}"
        )
        write_structure_slice_visualization(
            temporary_structure_output,
            trace=trace,
            chemical_symbol=homogeneous.generator.system.chemical_symbol,
            timestep_fs=homogeneous.generator.dynamics.timestep_fs,
            reference_planes_fractional=(),
            simulation_name=(
                f"{trace.positions_A.shape[1]:,}-atom crystallization {replica_name} "
                f"(checkpoint step {completed_global_step})"
            ),
            temperature_K=homogeneous.temperature_K,
            ptm_rmsd_cutoff=homogeneous.analysis.ptm_rmsd_cutoff,
        )
        temporary_structure_output.replace(structure_output)
        outputs.append(structure_output)

    if step_stamped:
        stamped_outputs = []
        for output in outputs:
            stamped_output = _step_stamped_path(output, completed_global_step)
            _copy_atomic(output, stamped_output)
            stamped_outputs.append(stamped_output)
        outputs.extend(stamped_outputs)

    return snapshot, completed_global_step, tuple(outputs)


def main() -> None:
    args = _arguments()
    config = load_homogeneous_campaign_config(args.campaign_config)
    snapshot, completed_global_step, outputs = render_checkpoint_visualizations(
        config,
        replica_name=args.replica_name,
        dashboard_output=args.output,
        include_structure_slices=args.include_structure_slices,
        structure_output=args.structure_output,
        step_stamped=args.step_stamped,
    )
    for output in outputs:
        print(
            f"Wrote {output} from verified checkpoint {snapshot.name} "
            f"(global step {completed_global_step})."
        )


if __name__ == "__main__":
    main()
