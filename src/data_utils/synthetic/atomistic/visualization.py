from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms

from .artifacts import EnvironmentLabels, PHASE_NAMES
from .config import GeneratorConfig
from .simulation import ThermodynamicTrace
from .validation import SystemDiagnostics


PHASE_COLORS = {
    "solid_bulk": "#277da1",
    "liquid_bulk": "#f8961e",
    "interface": "#d62828",
}


def _central_slice_mask(atoms: Atoms) -> np.ndarray:
    positions = atoms.get_positions(wrap=True)
    cell_lengths = np.asarray(atoms.cell.lengths(), dtype=np.float64)
    half_width = max(2.5, 0.08 * float(cell_lengths[1]))
    mask = np.abs(positions[:, 1] - 0.5 * cell_lengths[1]) <= half_width
    if np.count_nonzero(mask) < min(200, len(atoms)):
        mask = np.abs(positions[:, 1] - 0.5 * cell_lengths[1]) <= 0.2 * cell_lengths[1]
    return mask


def _plot_structure(
    path: Path,
    atoms: Atoms,
    labels: EnvironmentLabels,
    config: GeneratorConfig,
    title: str,
) -> None:
    positions = atoms.get_positions(wrap=True)
    cell_lengths = np.asarray(atoms.cell.lengths(), dtype=np.float64)
    slice_mask = _central_slice_mask(atoms)
    figure, axis = plt.subplots(figsize=(8.0, 6.5), constrained_layout=True)
    for phase_name in PHASE_NAMES:
        mask = slice_mask & (labels.phase_names == phase_name)
        if np.any(mask):
            axis.scatter(
                positions[mask, 0],
                positions[mask, 2],
                s=12.0,
                alpha=0.82,
                linewidths=0.0,
                color=PHASE_COLORS[phase_name],
                label=f"{phase_name} ({np.count_nonzero(mask):,} in slice)",
                rasterized=True,
            )
    if labels.slab_bounds_fractional is not None:
        for boundary in labels.slab_bounds_fractional:
            axis.axhline(
                boundary * cell_lengths[2], color="black", linestyle="--", linewidth=1.2
            )
    axis.set(
        xlim=(0.0, cell_lengths[0]),
        ylim=(0.0, cell_lengths[2]),
        xlabel="x (Å)",
        ylabel="z (Å)",
        title=(
            f"{title}\ncentral y slice, T={config.dynamics.target_temperature_K:.0f} K, "
            f"P={config.dynamics.pressure_GPa:.3f} GPa"
        ),
    )
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="upper right", frameon=True, fontsize=8)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_thermodynamics(
    path: Path,
    trace: ThermodynamicTrace,
    atom_count: int,
    config: GeneratorConfig,
    title: str,
) -> None:
    density = atom_count / trace.volume_A3
    figure, axes = plt.subplots(3, 1, figsize=(8.0, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(trace.step, trace.temperature_K, color="#6a4c93")
    axes[0].axhline(config.dynamics.target_temperature_K, color="black", linestyle="--")
    axes[0].set_ylabel("T (K)")
    axes[1].plot(trace.step, trace.pressure_GPa, color="#1982c4")
    axes[1].axhline(config.dynamics.pressure_GPa, color="black", linestyle="--")
    axes[1].set_ylabel("P (GPa)")
    axes[2].plot(trace.step, density, color="#2a9d8f")
    axes[2].set(xlabel="MD step", ylabel="N/V (atoms Å⁻³)")
    figure.suptitle(f"{title}: recorded target-state trajectory")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_diagnostics(
    path: Path,
    labels: EnvironmentLabels,
    diagnostics: SystemDiagnostics,
    title: str,
) -> None:
    ptm_names = ["fcc", "hcp", "bcc", "ico", "other"]
    ptm_values = [diagnostics.ptm_structure_fractions[name] for name in ptm_names]
    label_values = [np.count_nonzero(labels.phase_names == name) / len(labels.phase_names) for name in PHASE_NAMES]
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.5), constrained_layout=True)
    axes[0].bar(ptm_names, ptm_values, color="#577590")
    axes[0].set(ylim=(0.0, 1.0), ylabel="atom fraction", title="PTM audit (not labels)")
    axes[1].bar(PHASE_NAMES, label_values, color=[PHASE_COLORS[name] for name in PHASE_NAMES])
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set(ylim=(0.0, 1.0), ylabel="atom fraction", title="Preparation/context labels")
    figure.suptitle(
        f"{title}\nρ={diagnostics.number_density_per_A3:.5f} atoms Å⁻³, "
        f"min r={diagnostics.minimum_pair_distance_A:.3f} Å, "
        f"max |F|={diagnostics.maximum_force_eV_per_A:.3f} eV Å⁻¹"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_environment_visualizations(
    directory: Path,
    *,
    name: str,
    atoms: Atoms,
    labels: EnvironmentLabels,
    trace: ThermodynamicTrace,
    diagnostics: SystemDiagnostics,
    config: GeneratorConfig,
) -> None:
    visualization_dir = directory / "visualizations"
    visualization_dir.mkdir()
    _plot_structure(visualization_dir / "structure_slice.png", atoms, labels, config, name)
    _plot_thermodynamics(
        visualization_dir / "thermodynamic_trace.png", trace, len(atoms), config, name
    )
    _plot_diagnostics(
        visualization_dir / "structure_diagnostics.png", labels, diagnostics, name
    )


def write_benchmark_overview(root: Path, environment_names: list[str]) -> None:
    column_count = min(3, len(environment_names))
    row_count = int(np.ceil(len(environment_names) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(7.0 * column_count, 6.0 * row_count),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(-1)
    for axis, name in zip(axes_array, environment_names):
        image = plt.imread(root / name / "visualizations/structure_slice.png")
        axis.imshow(image)
        axis.set_title(name.replace("_", " "))
        axis.axis("off")
    for axis in axes_array[len(environment_names):]:
        axis.axis("off")
    figure.suptitle("Force-driven aluminium phase-context benchmark", fontsize=16)
    figure.savefig(root / "benchmark_overview.png", dpi=160)
    plt.close(figure)
