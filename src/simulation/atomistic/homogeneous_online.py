from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase import Atoms

from .homogeneous_analysis import first_persistent_threshold_run
from .transition_analysis import CRYSTALLINE_STRUCTURE_TYPES


@dataclass(frozen=True)
class OnlineCrystallinityObservation:
    measurement_step: int
    crystalline_fraction: float
    crystalline_cluster_count: int
    largest_crystalline_cluster_atoms: int


@dataclass(frozen=True)
class OnlineThresholdEvent:
    onset_step: int
    confirmation_step: int


class OnlineCrystallinityDetector:
    """PTM/cluster-only detector used for MD control; it deliberately omits RDF."""

    def __init__(
        self,
        *,
        ptm_rmsd_cutoff: float,
        crystalline_cluster_cutoff_A: float,
    ) -> None:
        try:
            from ovito.io.ase import ase_to_ovito
            from ovito.modifiers import (
                ClusterAnalysisModifier,
                PolyhedralTemplateMatchingModifier,
            )
        except ImportError as exc:
            raise ImportError(
                "Online crystallinity stopping requires OVITO PTM and cluster analysis. "
                "Install the repository requirements in the pointnet environment."
            ) from exc
        self._ase_to_ovito = ase_to_ovito
        self._ptm = PolyhedralTemplateMatchingModifier()
        self._ptm.rmsd_cutoff = ptm_rmsd_cutoff
        self._clusters = ClusterAnalysisModifier(
            cutoff=crystalline_cluster_cutoff_A,
            only_selected=True,
            sort_by_size=True,
        )

    def evaluate(
        self, atoms: Atoms, *, measurement_step: int
    ) -> OnlineCrystallinityObservation:
        # Use the exact float32 wrapped coordinate representation persisted in the
        # trajectory, so the asynchronous full analysis sees byte-equivalent geometry at
        # shared frames rather than a slightly different float64 control observable.
        analysis_atoms = Atoms(
            numbers=atoms.numbers,
            positions=np.asarray(
                atoms.get_positions(wrap=True), dtype=np.float32
            ).astype(np.float64),
            cell=np.asarray(atoms.cell.array, dtype=np.float64),
            pbc=atoms.pbc,
        )
        data = self._ase_to_ovito(analysis_atoms)
        data.apply(self._ptm)
        structure_types = np.asarray(data.particles["Structure Type"], dtype=np.int32)
        crystalline = np.isin(structure_types, CRYSTALLINE_STRUCTURE_TYPES)
        data.particles_.create_property("Selection", data=crystalline.astype(np.int32))
        data.apply(self._clusters)
        return OnlineCrystallinityObservation(
            measurement_step=measurement_step,
            crystalline_fraction=float(np.mean(crystalline)),
            crystalline_cluster_count=int(
                data.attributes["ClusterAnalysis.cluster_count"]
            ),
            largest_crystalline_cluster_atoms=int(
                data.attributes["ClusterAnalysis.largest_size"]
            ),
        )


class OnlineThresholdTracker:
    def __init__(
        self,
        *,
        threshold_atoms: int,
        persistence_frames: int,
        event_cadence_steps: int = 1,
        observations: tuple[OnlineCrystallinityObservation, ...] = (),
    ) -> None:
        self.threshold_atoms = threshold_atoms
        self.persistence_frames = persistence_frames
        if (
            not isinstance(event_cadence_steps, int)
            or isinstance(event_cadence_steps, bool)
            or event_cadence_steps <= 0
        ):
            raise ValueError(
                f"event_cadence_steps must be a positive integer, got "
                f"{event_cadence_steps!r}."
            )
        self.event_cadence_steps = event_cadence_steps
        self._observations = list(observations)
        self._event = self._find_event()

    @property
    def observations(self) -> tuple[OnlineCrystallinityObservation, ...]:
        return tuple(self._observations)

    @property
    def event(self) -> OnlineThresholdEvent | None:
        return self._event

    def _find_event(self) -> OnlineThresholdEvent | None:
        if not self._observations:
            return None
        event_observations = [
            item
            for item in self._observations
            if item.measurement_step % self.event_cadence_steps == 0
        ]
        if not event_observations:
            return None
        values = np.asarray(
            [item.largest_crystalline_cluster_atoms for item in event_observations],
            dtype=np.int64,
        )
        indices = first_persistent_threshold_run(
            values,
            threshold=self.threshold_atoms,
            persistence_frames=self.persistence_frames,
        )
        if indices is None:
            return None
        onset_index, confirmation_index = indices
        return OnlineThresholdEvent(
            onset_step=event_observations[onset_index].measurement_step,
            confirmation_step=event_observations[
                confirmation_index
            ].measurement_step,
        )

    def append(self, observation: OnlineCrystallinityObservation) -> None:
        if self._observations:
            previous_step = self._observations[-1].measurement_step
            if observation.measurement_step <= previous_step:
                raise ValueError(
                    "Online crystallinity measurement steps must increase strictly: "
                    f"previous={previous_step}, new={observation.measurement_step}."
                )
        self._observations.append(observation)
        if self._event is None:
            self._event = self._find_event()


def online_observations_to_arrays(
    observations: tuple[OnlineCrystallinityObservation, ...],
) -> dict[str, np.ndarray]:
    return {
        "measurement_step": np.asarray(
            [item.measurement_step for item in observations], dtype=np.int64
        ),
        "crystalline_fraction": np.asarray(
            [item.crystalline_fraction for item in observations], dtype=np.float64
        ),
        "crystalline_cluster_count": np.asarray(
            [item.crystalline_cluster_count for item in observations], dtype=np.int64
        ),
        "largest_crystalline_cluster_atoms": np.asarray(
            [item.largest_crystalline_cluster_atoms for item in observations],
            dtype=np.int64,
        ),
    }


def online_observations_from_arrays(
    arrays: dict[str, np.ndarray],
) -> tuple[OnlineCrystallinityObservation, ...]:
    lengths = {name: len(values) for name, values in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(
            f"Online crystallinity checkpoint arrays have inconsistent lengths: {lengths}."
        )
    return tuple(
        OnlineCrystallinityObservation(
            measurement_step=int(step),
            crystalline_fraction=float(fraction),
            crystalline_cluster_count=int(cluster_count),
            largest_crystalline_cluster_atoms=int(largest_cluster),
        )
        for step, fraction, cluster_count, largest_cluster in zip(
            arrays["measurement_step"],
            arrays["crystalline_fraction"],
            arrays["crystalline_cluster_count"],
            arrays["largest_crystalline_cluster_atoms"],
        )
    )
