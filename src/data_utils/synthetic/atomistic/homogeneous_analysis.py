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

from .simulation import ThermodynamicTrace
from .transition_analysis import (
    CRYSTALLINE_STRUCTURE_TYPES,
    STRUCTURE_COLORS,
    STRUCTURE_NAMES,
)


@dataclass(frozen=True)
class HomogeneousCrystallizationAnalysis:
    step: np.ndarray
    time_ps: np.ndarray
    structure_fractions: np.ndarray
    crystalline_fraction: np.ndarray
    crystalline_cluster_count: np.ndarray
    largest_crystalline_cluster_atoms: np.ndarray
    rdf_distance_A: np.ndarray
    rdf_g_r: np.ndarray
    nucleus_size_threshold_atoms: int
    nucleation_observed: bool
    nucleation_step: int | None
    nucleation_time_ps: float | None


def analyze_homogeneous_crystallization(
    trace: ThermodynamicTrace,
    *,
    chemical_symbol: str,
    timestep_fs: float,
    crystalline_cluster_cutoff_A: float,
    nucleus_size_threshold_atoms: int,
    rdf_cutoff_A: float,
    rdf_bins: int,
    progress: Callable[[str], None] = print,
) -> HomogeneousCrystallizationAnalysis:
    try:
        from ovito.io.ase import ase_to_ovito
        from ovito.modifiers import (
            ClusterAnalysisModifier,
            CoordinationAnalysisModifier,
            PolyhedralTemplateMatchingModifier,
        )
    except ImportError as exc:
        raise ImportError(
            "Homogeneous crystallization analysis requires OVITO for PTM, connected "
            "cluster analysis, and RDF calculation. Install the repository requirements "
            "in the pointnet environment."
        ) from exc

    frame_count, atom_count, _ = trace.positions_A.shape
    progress(
        f"homogeneous_crystallization: PTM, nucleus, and RDF audit of {frame_count} "
        f"frames ({atom_count} atoms/frame)"
    )
    structure_fractions = np.empty((frame_count, len(STRUCTURE_NAMES)), dtype=np.float64)
    crystalline_fraction = np.empty(frame_count, dtype=np.float64)
    cluster_count = np.empty(frame_count, dtype=np.int64)
    largest_cluster = np.empty(frame_count, dtype=np.int64)
    rdf_g_r = np.empty((frame_count, rdf_bins), dtype=np.float64)
    numbers = np.full(atom_count, atomic_numbers[chemical_symbol], dtype=np.int32)
    ptm = PolyhedralTemplateMatchingModifier()
    clusters = ClusterAnalysisModifier(
        cutoff=crystalline_cluster_cutoff_A,
        only_selected=True,
        sort_by_size=True,
    )
    rdf = CoordinationAnalysisModifier(
        cutoff=rdf_cutoff_A,
        number_of_bins=rdf_bins,
    )
    rdf_distance_A: np.ndarray | None = None

    for frame_index, (positions_A, cell_A) in enumerate(
        zip(trace.positions_A, trace.cell_vectors_A)
    ):
        atoms = Atoms(numbers=numbers, positions=positions_A, cell=cell_A, pbc=True)
        data = ase_to_ovito(atoms)
        data.apply(ptm)
        structure_types = np.asarray(data.particles["Structure Type"], dtype=np.int32)
        counts = np.bincount(
            structure_types, minlength=len(STRUCTURE_NAMES)
        )[: len(STRUCTURE_NAMES)]
        structure_fractions[frame_index] = counts / atom_count
        crystalline = np.isin(structure_types, CRYSTALLINE_STRUCTURE_TYPES)
        crystalline_fraction[frame_index] = float(np.mean(crystalline))

        data.particles_.create_property(
            "Selection", data=crystalline.astype(np.int32)
        )
        data.apply(clusters)
        cluster_count[frame_index] = int(
            data.attributes["ClusterAnalysis.cluster_count"]
        )
        largest_cluster[frame_index] = int(
            data.attributes["ClusterAnalysis.largest_size"]
        )

        data.apply(rdf)
        rdf_values = data.tables["coordination-rdf"].xy()
        if rdf_distance_A is None:
            rdf_distance_A = rdf_values[:, 0].copy()
        rdf_g_r[frame_index] = rdf_values[:, 1]

    if rdf_distance_A is None:
        raise RuntimeError("Homogeneous crystallization analysis received an empty trace.")
    time_ps = trace.step.astype(np.float64) * timestep_fs / 1000.0
    threshold_crossings = np.flatnonzero(
        largest_cluster >= nucleus_size_threshold_atoms
    )
    if len(threshold_crossings):
        onset_index = int(threshold_crossings[0])
        nucleation_step = int(trace.step[onset_index])
        nucleation_time_ps = float(time_ps[onset_index])
    else:
        nucleation_step = None
        nucleation_time_ps = None

    return HomogeneousCrystallizationAnalysis(
        step=trace.step.copy(),
        time_ps=time_ps,
        structure_fractions=structure_fractions,
        crystalline_fraction=crystalline_fraction,
        crystalline_cluster_count=cluster_count,
        largest_crystalline_cluster_atoms=largest_cluster,
        rdf_distance_A=rdf_distance_A,
        rdf_g_r=rdf_g_r,
        nucleus_size_threshold_atoms=nucleus_size_threshold_atoms,
        nucleation_observed=nucleation_step is not None,
        nucleation_step=nucleation_step,
        nucleation_time_ps=nucleation_time_ps,
    )


def write_homogeneous_progress_visualization(
    path: Path,
    *,
    trace: ThermodynamicTrace,
    analysis: HomogeneousCrystallizationAnalysis,
    temperature_K: float,
    pressure_GPa: float,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.0), constrained_layout=True)
    for structure_name, color, fractions in zip(
        STRUCTURE_NAMES,
        STRUCTURE_COLORS,
        analysis.structure_fractions.T,
    ):
        axes[0, 0].plot(
            analysis.time_ps,
            fractions,
            color=color,
            label=structure_name.upper(),
        )
    axes[0, 0].plot(
        analysis.time_ps,
        analysis.crystalline_fraction,
        color="black",
        linewidth=1.5,
        label="FCC+HCP+BCC",
    )
    axes[0, 0].set(xlabel="time (ps)", ylabel="PTM structure fraction", ylim=(0.0, 1.0))
    axes[0, 0].legend(ncol=2)

    axes[0, 1].plot(
        analysis.time_ps,
        analysis.largest_crystalline_cluster_atoms,
        color="#6a4c93",
    )
    axes[0, 1].axhline(
        analysis.nucleus_size_threshold_atoms,
        color="black",
        linestyle="--",
        label=f"analysis threshold ({analysis.nucleus_size_threshold_atoms} atoms)",
    )
    axes[0, 1].set(xlabel="time (ps)", ylabel="largest connected crystalline cluster (atoms)")
    axes[0, 1].legend()

    temperature_axis = axes[1, 0]
    pressure_axis = temperature_axis.twinx()
    temperature_axis.plot(analysis.time_ps, trace.temperature_K, color="#f4a261")
    temperature_axis.axhline(temperature_K, color="#f4a261", linestyle="--")
    pressure_axis.plot(analysis.time_ps, trace.pressure_GPa, color="#457b9d", alpha=0.8)
    pressure_axis.axhline(pressure_GPa, color="#457b9d", linestyle="--")
    temperature_axis.set(xlabel="time (ps)", ylabel="temperature (K)")
    pressure_axis.set_ylabel("pressure (GPa)")

    image = axes[1, 1].imshow(
        analysis.rdf_g_r,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=(
            float(analysis.rdf_distance_A[0]),
            float(analysis.rdf_distance_A[-1]),
            float(analysis.time_ps[0]),
            float(analysis.time_ps[-1]),
        ),
        cmap="magma",
    )
    axes[1, 1].set(xlabel="pair distance (Å)", ylabel="time (ps)")
    figure.colorbar(image, ax=axes[1, 1], label="g(r)")

    if analysis.nucleation_observed:
        result_text = f"threshold first crossed at {analysis.nucleation_time_ps:.2f} ps"
    else:
        result_text = "no threshold-sized crystalline nucleus observed"
    figure.suptitle(
        f"Homogeneous crystallization from supercooled liquid at {temperature_K:.0f} K\n"
        f"{result_text}"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def write_homogeneous_rdf_visualization(
    path: Path,
    *,
    analysis: HomogeneousCrystallizationAnalysis,
    temperature_K: float,
) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 6.0), constrained_layout=True)
    frame_indices = (0, len(analysis.step) // 2, len(analysis.step) - 1)
    colors = ("#457b9d", "#e9c46a", "#e76f51")
    for frame_index, color in zip(frame_indices, colors):
        axis.plot(
            analysis.rdf_distance_A,
            analysis.rdf_g_r[frame_index],
            color=color,
            label=f"t={analysis.time_ps[frame_index]:.2f} ps",
        )
    axis.set(xlabel="pair distance (Å)", ylabel="g(r)", xlim=(0.0, analysis.rdf_distance_A[-1]))
    axis.legend()
    axis.set_title(f"Total Al RDF during homogeneous crystallization at {temperature_K:.0f} K")
    figure.savefig(path, dpi=180)
    plt.close(figure)
