from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.data import atomic_numbers

from .artifacts import PHASE_NAMES
from .simulation import ThermodynamicTrace
from .transition_config import TransitionBranchConfig


STRUCTURE_NAMES = ("other", "fcc", "hcp", "bcc", "ico")
CRYSTALLINE_STRUCTURE_TYPES = np.array([1, 2, 3], dtype=np.int32)
STRUCTURE_COLORS = (
    "#8d99ae",
    "#2a9d8f",
    "#e76f51",
    "#457b9d",
    "#e9c46a",
)


@dataclass(frozen=True)
class TransitionAnalysis:
    step: np.ndarray
    time_ps: np.ndarray
    structure_fractions: np.ndarray
    crystalline_fraction: np.ndarray
    prepared_liquid_slab_crystalline_fraction: np.ndarray
    prepared_solid_region_crystalline_fraction: np.ndarray
    crystalline_profile: np.ndarray
    profile_bin_centers_fractional: np.ndarray
    front_displacement_A: np.ndarray
    net_crystalline_fraction_change: float
    net_front_displacement_A: float
    average_front_velocity_m_per_s: float


@dataclass(frozen=True)
class PhaseRdfAnalysis:
    step: np.ndarray
    time_ps: np.ndarray
    distance_A: np.ndarray
    phase_names: tuple[str, ...]
    g_r: np.ndarray


def phase_rdf_metadata(
    analysis: PhaseRdfAnalysis,
    *,
    cutoff_A: float,
    bins: int,
) -> dict[str, object]:
    return {
        "definition": (
            "Total-Al radial distribution around central atoms grouped by their initial "
            "prepared phase provenance; neighboring atoms may have any phase label."
        ),
        "normalization": "whole-cell instantaneous number density",
        "backend": "OVITO compiled partial radial distribution function",
        "phase_names": list(analysis.phase_names),
        "cutoff_A": cutoff_A,
        "bins": bins,
        "frames": len(analysis.step),
    }


def write_phase_rdf_archive(path: Path, analysis: PhaseRdfAnalysis) -> None:
    with path.open("wb") as handle:
        np.savez(
            handle,
            step=analysis.step,
            time_ps=analysis.time_ps,
            distance_A=analysis.distance_A,
            phase_names=np.asarray(analysis.phase_names),
            g_r=analysis.g_r,
        )


def _ptm_structure_types(atoms: Atoms) -> np.ndarray:
    try:
        from ovito.io.ase import ase_to_ovito
        from ovito.modifiers import PolyhedralTemplateMatchingModifier
        from ovito.pipeline import Pipeline, StaticSource
    except ImportError as exc:
        raise ImportError(
            "Transition analysis requires OVITO Polyhedral Template Matching. Install the "
            "repository requirements in the pointnet environment."
        ) from exc
    pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(atoms)))
    pipeline.modifiers.append(PolyhedralTemplateMatchingModifier())
    data = pipeline.compute()
    return np.asarray(data.particles["Structure Type"], dtype=np.int32)


def analyze_phase_rdf(
    trace: ThermodynamicTrace,
    *,
    chemical_symbol: str,
    prepared_phase_ids: np.ndarray,
    timestep_fs: float,
    cutoff_A: float,
    bins: int,
    branch_name: str,
    progress: Callable[[str], None] = print,
) -> PhaseRdfAnalysis:
    """Compute a center-conditioned Al RDF for each prepared phase provenance.

    Neighbors may belong to any phase. Selecting only the central atoms avoids treating the
    spatially finite interface and slab regions as independent periodic bulk systems.
    """

    frame_count, atom_count, _ = trace.positions_A.shape
    try:
        from ovito.data import ParticleType
        from ovito.io.ase import ase_to_ovito
        from ovito.modifiers import CoordinationAnalysisModifier
    except ImportError as exc:
        raise ImportError(
            "Per-phase RDF analysis requires OVITO's compiled coordination analysis. "
            "Install the repository requirements in the pointnet environment."
        ) from exc

    phase_fractions = (
        np.bincount(prepared_phase_ids, minlength=len(PHASE_NAMES)) / atom_count
    )
    g_r = np.empty((frame_count, len(PHASE_NAMES), bins), dtype=np.float64)
    numbers = np.full(atom_count, atomic_numbers[chemical_symbol], dtype=np.int32)
    prepared_particle_types = prepared_phase_ids.astype(np.int32) + 1
    phase_pairs = tuple(
        (first, second)
        for first in range(len(PHASE_NAMES))
        for second in range(first, len(PHASE_NAMES))
    )
    expected_components = tuple(
        f"{PHASE_NAMES[first]}-{PHASE_NAMES[second]}"
        for first, second in phase_pairs
    )
    modifier = CoordinationAnalysisModifier(
        cutoff=cutoff_A,
        number_of_bins=bins,
        partial=True,
        type_property="Prepared Phase",
    )
    progress(
        f"{branch_name}: per-phase RDF audit of {frame_count} frames "
        f"to {cutoff_A:.2f} A ({bins} bins, OVITO compiled backend)"
    )

    distance_A: np.ndarray | None = None
    for frame_index, (positions_A, cell_A) in enumerate(
        zip(trace.positions_A, trace.cell_vectors_A)
    ):
        atoms = Atoms(numbers=numbers, positions=positions_A, cell=cell_A, pbc=True)
        data = ase_to_ovito(atoms)
        phase_property = data.particles_.create_property(
            "Prepared Phase", data=prepared_particle_types
        )
        for phase_id, phase_name in enumerate(PHASE_NAMES, start=1):
            phase_property.types.append(ParticleType(id=phase_id, name=phase_name))
        data.apply(modifier)
        table = data.tables["coordination-rdf"]
        components = tuple(table.y.component_names)
        if components != expected_components:
            raise RuntimeError(
                f"{branch_name}: OVITO returned RDF components {components}, expected "
                f"{expected_components}."
            )
        table_values = table.xy()
        if distance_A is None:
            distance_A = table_values[:, 0].copy()
        partial_rdf = table_values[:, 1:].T
        frame_rdf = np.zeros((len(PHASE_NAMES), bins), dtype=np.float64)
        for pair_index, (first, second) in enumerate(phase_pairs):
            frame_rdf[first] += phase_fractions[second] * partial_rdf[pair_index]
            if first != second:
                frame_rdf[second] += phase_fractions[first] * partial_rdf[pair_index]
        g_r[frame_index] = frame_rdf

    if distance_A is None:
        raise RuntimeError(
            f"{branch_name}: RDF analysis received an empty thermodynamic trace."
        )

    return PhaseRdfAnalysis(
        step=trace.step.copy(),
        time_ps=trace.step.astype(np.float64) * timestep_fs / 1000.0,
        distance_A=distance_A,
        phase_names=PHASE_NAMES,
        g_r=g_r,
    )


def analyze_transition(
    trace: ThermodynamicTrace,
    *,
    chemical_symbol: str,
    timestep_fs: float,
    slab_bounds_fractional: tuple[float, float],
    solid_number_density_per_A3: float,
    profile_bins: int,
    branch: TransitionBranchConfig,
    progress: Callable[[str], None] = print,
) -> TransitionAnalysis:
    frame_count, atom_count, _ = trace.positions_A.shape
    progress(
        f"{branch.name}: PTM audit of {frame_count} transition frames "
        f"({atom_count} atoms/frame)"
    )
    structure_fractions = np.empty((frame_count, len(STRUCTURE_NAMES)), dtype=np.float64)
    crystalline_fraction = np.empty(frame_count, dtype=np.float64)
    liquid_slab_fraction = np.empty(frame_count, dtype=np.float64)
    solid_region_fraction = np.empty(frame_count, dtype=np.float64)
    crystalline_profile = np.empty((frame_count, profile_bins), dtype=np.float64)
    front_displacement_A = np.empty(frame_count, dtype=np.float64)
    bin_centers = (np.arange(profile_bins, dtype=np.float64) + 0.5) / profile_bins
    numbers = np.full(atom_count, atomic_numbers[chemical_symbol], dtype=np.int32)
    lower, upper = slab_bounds_fractional
    initial_crystalline_count = 0

    for frame_index, (positions_A, cell_A) in enumerate(
        zip(trace.positions_A, trace.cell_vectors_A)
    ):
        atoms = Atoms(numbers=numbers, positions=positions_A, cell=cell_A, pbc=True)
        structure_types = _ptm_structure_types(atoms)
        counts = np.bincount(structure_types, minlength=len(STRUCTURE_NAMES))[: len(STRUCTURE_NAMES)]
        structure_fractions[frame_index] = counts / atom_count
        crystalline = np.isin(structure_types, CRYSTALLINE_STRUCTURE_TYPES)
        crystalline_count = int(np.count_nonzero(crystalline))
        crystalline_fraction[frame_index] = crystalline_count / atom_count
        if frame_index == 0:
            initial_crystalline_count = crystalline_count

        scaled_z = np.linalg.solve(np.asarray(cell_A).T, np.asarray(positions_A).T).T[:, 2] % 1.0
        liquid_slab = (scaled_z >= lower) & (scaled_z < upper)
        solid_region = ~liquid_slab
        liquid_slab_fraction[frame_index] = float(np.mean(crystalline[liquid_slab]))
        solid_region_fraction[frame_index] = float(np.mean(crystalline[solid_region]))

        bin_indices = np.floor(scaled_z * profile_bins).astype(np.int64)
        bin_counts = np.bincount(bin_indices, minlength=profile_bins)
        if np.any(bin_counts == 0):
            empty_bins = np.flatnonzero(bin_counts == 0).tolist()
            raise RuntimeError(
                f"{branch.name}: empty transition profile bins {empty_bins}; "
                f"profile_bins={profile_bins}, atom_count={atom_count}."
            )
        crystalline_counts = np.bincount(
            bin_indices,
            weights=crystalline.astype(np.float64),
            minlength=profile_bins,
        )
        crystalline_profile[frame_index] = crystalline_counts / bin_counts

        lateral_area_A2 = float(
            np.linalg.norm(np.cross(np.asarray(cell_A)[0], np.asarray(cell_A)[1]))
        )
        front_displacement_A[frame_index] = (
            (crystalline_count - initial_crystalline_count)
            / (2.0 * solid_number_density_per_A3 * lateral_area_A2)
        )

    time_ps = trace.step.astype(np.float64) * timestep_fs / 1000.0
    fraction_change = float(crystalline_fraction[-1] - crystalline_fraction[0])
    front_change_A = float(front_displacement_A[-1])
    duration_ps = float(time_ps[-1] - time_ps[0])
    average_velocity_m_per_s = front_change_A / duration_ps * 100.0
    signed_change = fraction_change if branch.expected_direction == "growth" else -fraction_change
    if signed_change < branch.minimum_crystalline_fraction_change:
        raise RuntimeError(
            f"{branch.name}: direct-coexistence trajectory changed crystalline fraction by "
            f"{fraction_change:+.4f}, but expected_direction={branch.expected_direction!r} "
            f"requires at least {branch.minimum_crystalline_fraction_change:.4f} in the "
            "expected direction. Increase the temperature separation or trajectory length."
        )
    return TransitionAnalysis(
        step=trace.step.copy(),
        time_ps=time_ps,
        structure_fractions=structure_fractions,
        crystalline_fraction=crystalline_fraction,
        prepared_liquid_slab_crystalline_fraction=liquid_slab_fraction,
        prepared_solid_region_crystalline_fraction=solid_region_fraction,
        crystalline_profile=crystalline_profile,
        profile_bin_centers_fractional=bin_centers,
        front_displacement_A=front_displacement_A,
        net_crystalline_fraction_change=fraction_change,
        net_front_displacement_A=front_change_A,
        average_front_velocity_m_per_s=average_velocity_m_per_s,
    )


def write_transition_visualization(
    path: Path,
    *,
    trace: ThermodynamicTrace,
    analysis: TransitionAnalysis,
    branch: TransitionBranchConfig,
    pressure_GPa: float,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    axes[0, 0].plot(
        analysis.time_ps,
        analysis.crystalline_fraction,
        color="#264653",
        label="whole cell",
    )
    axes[0, 0].plot(
        analysis.time_ps,
        analysis.prepared_liquid_slab_crystalline_fraction,
        color="#e76f51",
        label="prepared liquid slab",
    )
    axes[0, 0].plot(
        analysis.time_ps,
        analysis.prepared_solid_region_crystalline_fraction,
        color="#2a9d8f",
        label="prepared solid region",
    )
    axes[0, 0].set(xlabel="time (ps)", ylabel="PTM crystalline fraction", ylim=(0.0, 1.0))
    axes[0, 0].legend()

    axes[0, 1].plot(
        analysis.time_ps,
        analysis.front_displacement_A,
        color="#6a4c93",
    )
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set(xlabel="time (ps)", ylabel="mean displacement per front (Å)")

    temperature_axis = axes[1, 0]
    pressure_axis = temperature_axis.twinx()
    temperature_axis.plot(analysis.time_ps, trace.temperature_K, color="#f4a261")
    temperature_axis.axhline(branch.temperature_K, color="#f4a261", linestyle="--")
    pressure_axis.plot(analysis.time_ps, trace.pressure_GPa, color="#457b9d", alpha=0.8)
    pressure_axis.axhline(pressure_GPa, color="#457b9d", linestyle="--")
    temperature_axis.set(xlabel="time (ps)", ylabel="temperature (K)")
    pressure_axis.set_ylabel("pressure (GPa)")

    axes[1, 1].imshow(
        analysis.crystalline_profile,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(0.0, 1.0, analysis.time_ps[0], analysis.time_ps[-1]),
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )
    axes[1, 1].set(xlabel="fractional position along interface normal", ylabel="time (ps)")
    figure.suptitle(
        f"{branch.name}: direct solid–liquid coexistence at {branch.temperature_K:.0f} K\n"
        f"Δcrystalline={analysis.net_crystalline_fraction_change:+.3f}, "
        f"mean front velocity={analysis.average_front_velocity_m_per_s:+.1f} m/s"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_structure_slice_visualization(
    path: Path,
    *,
    trace: ThermodynamicTrace,
    chemical_symbol: str,
    timestep_fs: float,
    reference_planes_fractional: tuple[float, ...],
    simulation_name: str,
    temperature_K: float,
) -> None:
    selected_frames = (0, len(trace.step) // 2, len(trace.step) - 1)
    numbers = np.full(
        trace.positions_A.shape[1],
        atomic_numbers[chemical_symbol],
        dtype=np.int32,
    )
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 6.2))

    for axis, frame_index in zip(axes, selected_frames):
        atoms = Atoms(
            numbers=numbers,
            positions=trace.positions_A[frame_index],
            cell=trace.cell_vectors_A[frame_index],
            pbc=True,
        )
        positions_A = atoms.get_positions(wrap=True)
        cell_A = np.asarray(atoms.cell, dtype=np.float64)
        cell_lengths_A = np.linalg.norm(cell_A, axis=1)
        scaled_positions = atoms.get_scaled_positions(wrap=True)
        slice_half_width_A = max(2.5, 0.08 * float(cell_lengths_A[1]))
        slice_mask = (
            np.abs(scaled_positions[:, 1] - 0.5) * cell_lengths_A[1]
            <= slice_half_width_A
        )
        structure_types = _ptm_structure_types(atoms)
        for structure_id, (structure_name, color) in enumerate(
            zip(STRUCTURE_NAMES, STRUCTURE_COLORS)
        ):
            mask = slice_mask & (structure_types == structure_id)
            axis.scatter(
                positions_A[mask, 0],
                positions_A[mask, 2],
                s=5.0,
                alpha=0.82,
                linewidths=0.0,
                color=color,
                label=structure_name.upper(),
                rasterized=True,
            )
        for boundary in reference_planes_fractional:
            axis.axhline(
                boundary * cell_lengths_A[2],
                color="black",
                linestyle="--",
                linewidth=1.0,
            )
        crystalline_fraction = float(
            np.mean(np.isin(structure_types, CRYSTALLINE_STRUCTURE_TYPES))
        )
        time_ps = float(trace.step[frame_index] * timestep_fs / 1000.0)
        axis.set(
            xlim=(0.0, cell_lengths_A[0]),
            ylim=(0.0, cell_lengths_A[2]),
            xlabel="x (Å)",
            ylabel="z (Å)",
            title=f"t={time_ps:.2f} ps, crystalline={crystalline_fraction:.3f}",
        )
        axis.set_aspect("equal", adjustable="box")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=len(STRUCTURE_NAMES),
    )
    audit_note = "PTM colors are audit observables"
    if reference_planes_fractional:
        audit_note += "; dashed lines mark the prepared interfaces"
    figure.suptitle(
        f"{simulation_name}: central-y structure slices at {temperature_K:.0f} K\n"
        f"{audit_note}",
        y=0.98,
    )
    figure.subplots_adjust(bottom=0.15, top=0.82, wspace=0.22)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def write_phase_rdf_visualization(
    path: Path,
    *,
    analysis: PhaseRdfAnalysis,
    branch: TransitionBranchConfig,
) -> None:
    figure, axes = plt.subplots(
        2,
        len(analysis.phase_names),
        figsize=(16.0, 8.5),
        sharex=True,
        constrained_layout=True,
    )
    selected_frames = (0, len(analysis.step) // 2, len(analysis.step) - 1)
    colors = ("#264653", "#e9c46a", "#e76f51")
    heatmap_limit = float(np.quantile(analysis.g_r, 0.995))

    for phase_index, phase_name in enumerate(analysis.phase_names):
        curve_axis = axes[0, phase_index]
        for frame_index, color in zip(selected_frames, colors):
            curve_axis.plot(
                analysis.distance_A,
                analysis.g_r[frame_index, phase_index],
                color=color,
                label=f"{analysis.time_ps[frame_index]:.2f} ps",
            )
        curve_axis.set_title(phase_name.replace("_", " "))
        curve_axis.set_ylabel("center-conditioned g(r)")
        curve_axis.legend()

        heatmap_axis = axes[1, phase_index]
        heatmap = heatmap_axis.imshow(
            analysis.g_r[:, phase_index],
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(
                0.0,
                float(analysis.distance_A[-1]),
                float(analysis.time_ps[0]),
                float(analysis.time_ps[-1]),
            ),
            vmin=0.0,
            vmax=heatmap_limit,
            cmap="magma",
        )
        heatmap_axis.set(xlabel="r (A)", ylabel="time (ps)")
        figure.colorbar(heatmap, ax=heatmap_axis, label="g(r)")

    figure.suptitle(
        f"{branch.name}: RDF evolution by initial phase provenance at "
        f"{branch.temperature_K:.0f} K\n"
        "Central atoms follow their prepared labels; neighbors include every Al atom"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_phase_rdf_overview(path: Path, branch_images: dict[str, Path]) -> None:
    figure, axes = plt.subplots(
        1,
        len(branch_images),
        figsize=(16.0, 6.0),
        constrained_layout=True,
    )
    for axis, image_path in zip(np.atleast_1d(axes), branch_images.values()):
        axis.imshow(plt.imread(image_path))
        axis.axis("off")
    figure.suptitle("MACE direct-coexistence RDF evolution by initial phase provenance")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_structure_slice_overview(path: Path, branch_images: dict[str, Path]) -> None:
    figure, axes = plt.subplots(
        len(branch_images),
        1,
        figsize=(17.0, 6.0 * len(branch_images)),
        constrained_layout=True,
    )
    for axis, (branch_name, image_path) in zip(np.atleast_1d(axes), branch_images.items()):
        axis.imshow(plt.imread(image_path))
        axis.set_title(branch_name)
        axis.axis("off")
    figure.suptitle("MACE direct-coexistence atomic structure slices")
    figure.savefig(path, dpi=160)
    plt.close(figure)
