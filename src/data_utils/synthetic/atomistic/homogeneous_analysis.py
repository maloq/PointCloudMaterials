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
    ptm_rmsd_cutoff: float
    nucleus_size_threshold_atoms: int
    threshold_persistence_frames: int
    nucleation_observed: bool
    nucleation_step: int | None
    nucleation_time_ps: float | None
    confirmation_step: int | None
    confirmation_time_ps: float | None


@dataclass(frozen=True)
class ReplicaObservation:
    replica_name: str
    random_seed: int
    event_observed: bool
    observation_time_ps: float


@dataclass(frozen=True)
class HomogeneousSurvivalAnalysis:
    time_ps: np.ndarray
    replicas_at_risk: np.ndarray
    events: np.ndarray
    censored: np.ndarray
    survival_probability: np.ndarray
    observations: tuple[ReplicaObservation, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "replica_count": len(self.observations),
            "observed_event_count": int(sum(item.event_observed for item in self.observations)),
            "right_censored_count": int(
                sum(not item.event_observed for item in self.observations)
            ),
            "observations": [
                {
                    "replica_name": item.replica_name,
                    "random_seed": item.random_seed,
                    "event_observed": item.event_observed,
                    "observation_time_ps": item.observation_time_ps,
                }
                for item in self.observations
            ],
            "kaplan_meier": {
                "time_ps": self.time_ps.tolist(),
                "replicas_at_risk": self.replicas_at_risk.tolist(),
                "events": self.events.tolist(),
                "right_censored": self.censored.tolist(),
                "survival_probability": self.survival_probability.tolist(),
            },
            "rate_estimate": None,
            "rate_estimate_status": "not_computed",
            "rate_estimate_reason": (
                "A non-parametric survival curve does not by itself establish a stationary "
                "Poisson nucleation process. A rate additionally requires validated "
                "metastable-liquid stationarity, a model-specific undercooling, volume and "
                "finite-size convergence, and enough independent events."
            ),
        }


def analyze_replica_survival(
    observations: tuple[ReplicaObservation, ...],
) -> HomogeneousSurvivalAnalysis:
    """Construct a Kaplan-Meier curve from event times and right-censored replicas."""
    if not observations:
        raise ValueError("Survival analysis requires at least one replica observation.")
    if len({item.replica_name for item in observations}) != len(observations):
        raise ValueError("Survival analysis replica names must be unique.")
    if len({item.random_seed for item in observations}) != len(observations):
        raise ValueError("Survival analysis random seeds must be unique.")
    for item in observations:
        if not np.isfinite(item.observation_time_ps) or item.observation_time_ps < 0.0:
            raise ValueError(
                f"Replica {item.replica_name!r} has invalid observation_time_ps="
                f"{item.observation_time_ps}."
            )

    times = np.unique(
        np.asarray([item.observation_time_ps for item in observations], dtype=np.float64)
    )
    at_risk = np.empty(len(times), dtype=np.int64)
    events = np.empty(len(times), dtype=np.int64)
    censored = np.empty(len(times), dtype=np.int64)
    survival = np.empty(len(times), dtype=np.float64)
    remaining = len(observations)
    probability = 1.0
    for index, time_ps in enumerate(times):
        events_at_time = sum(
            item.event_observed and item.observation_time_ps == time_ps
            for item in observations
        )
        censored_at_time = sum(
            (not item.event_observed) and item.observation_time_ps == time_ps
            for item in observations
        )
        at_risk[index] = remaining
        events[index] = events_at_time
        censored[index] = censored_at_time
        probability *= 1.0 - events_at_time / remaining
        survival[index] = probability
        remaining -= events_at_time + censored_at_time

    if remaining != 0:
        raise RuntimeError(
            f"Survival accounting left {remaining} replicas at risk after all observations."
        )
    return HomogeneousSurvivalAnalysis(
        time_ps=times,
        replicas_at_risk=at_risk,
        events=events,
        censored=censored,
        survival_probability=survival,
        observations=observations,
    )


def first_persistent_threshold_run(
    values: np.ndarray,
    *,
    threshold: int,
    persistence_frames: int,
) -> tuple[int, int] | None:
    """Return the inclusive (onset, confirmation) indices of the first sustained crossing."""
    if values.ndim != 1:
        raise ValueError(
            f"Persistent threshold analysis requires a one-dimensional series, got "
            f"shape={values.shape}."
        )
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    if persistence_frames <= 0:
        raise ValueError(
            f"persistence_frames must be positive, got {persistence_frames}."
        )
    run_start: int | None = None
    for index, value in enumerate(values):
        if value >= threshold:
            if run_start is None:
                run_start = index
            if index - run_start + 1 == persistence_frames:
                return run_start, index
        else:
            run_start = None
    return None


def analyze_homogeneous_crystallization(
    trace: ThermodynamicTrace,
    *,
    chemical_symbol: str,
    timestep_fs: float,
    ptm_rmsd_cutoff: float,
    crystalline_cluster_cutoff_A: float,
    nucleus_size_threshold_atoms: int,
    threshold_persistence_frames: int,
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
        f"homogeneous_crystallization: PTM, connected-cluster, and RDF audit of {frame_count} "
        f"frames ({atom_count} atoms/frame)"
    )
    structure_fractions = np.empty((frame_count, len(STRUCTURE_NAMES)), dtype=np.float64)
    crystalline_fraction = np.empty(frame_count, dtype=np.float64)
    cluster_count = np.empty(frame_count, dtype=np.int64)
    largest_cluster = np.empty(frame_count, dtype=np.int64)
    rdf_g_r = np.empty((frame_count, rdf_bins), dtype=np.float64)
    numbers = np.full(atom_count, atomic_numbers[chemical_symbol], dtype=np.int32)
    ptm = PolyhedralTemplateMatchingModifier()
    ptm.rmsd_cutoff = ptm_rmsd_cutoff
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
    persistent_run = first_persistent_threshold_run(
        largest_cluster,
        threshold=nucleus_size_threshold_atoms,
        persistence_frames=threshold_persistence_frames,
    )
    if persistent_run is not None:
        onset_index, confirmation_index = persistent_run
        nucleation_step = int(trace.step[onset_index])
        nucleation_time_ps = float(time_ps[onset_index])
        confirmation_step = int(trace.step[confirmation_index])
        confirmation_time_ps = float(time_ps[confirmation_index])
    else:
        nucleation_step = None
        nucleation_time_ps = None
        confirmation_step = None
        confirmation_time_ps = None

    return HomogeneousCrystallizationAnalysis(
        step=trace.step.copy(),
        time_ps=time_ps,
        structure_fractions=structure_fractions,
        crystalline_fraction=crystalline_fraction,
        crystalline_cluster_count=cluster_count,
        largest_crystalline_cluster_atoms=largest_cluster,
        rdf_distance_A=rdf_distance_A,
        rdf_g_r=rdf_g_r,
        ptm_rmsd_cutoff=ptm_rmsd_cutoff,
        nucleus_size_threshold_atoms=nucleus_size_threshold_atoms,
        threshold_persistence_frames=threshold_persistence_frames,
        nucleation_observed=nucleation_step is not None,
        nucleation_step=nucleation_step,
        nucleation_time_ps=nucleation_time_ps,
        confirmation_step=confirmation_step,
        confirmation_time_ps=confirmation_time_ps,
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
        label=(
            f"analysis threshold ({analysis.nucleus_size_threshold_atoms} atoms for "
            f"{analysis.threshold_persistence_frames} frames)"
        ),
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
        result_text = (
            f"persistent threshold onset at {analysis.nucleation_time_ps:.2f} ps; "
            f"confirmed at {analysis.confirmation_time_ps:.2f} ps"
        )
    else:
        result_text = "no persistent threshold-sized crystalline cluster observed"
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
