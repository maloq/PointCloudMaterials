from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import TemporalBenchmarkConfig
from .dynamics import LatentTrajectories, SiteLayout
from .graph import TransitionGraph


@dataclass(frozen=True)
class ValidationSummary:
    summary: dict[str, object]


_PTM_STRUCTURE_TYPE_NAMES = {
    0: "other",
    1: "fcc",
    2: "hcp",
    3: "bcc",
    4: "ico",
    5: "simple_cubic",
    6: "cubic_diamond",
    7: "hexagonal_diamond",
    8: "graphene",
}
_PTM_CRYSTALLINE_STRUCTURE_TYPES = np.asarray([1, 2, 3], dtype=np.int32)


@dataclass(frozen=True)
class StructuralAuditFrame:
    frame_index: int
    atom_count: int
    ptm_structure_type_counts: dict[str, int]
    by_true_state: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "atom_count": self.atom_count,
            "ptm_structure_type_counts": self.ptm_structure_type_counts,
            "by_true_state": self.by_true_state,
        }


def audit_rendered_frame(
    *,
    config: TemporalBenchmarkConfig,
    graph: TransitionGraph,
    frame_index: int,
    positions_A: np.ndarray,
    true_state_ids: np.ndarray,
) -> StructuralAuditFrame:
    if frame_index not in config.structural_audit.frame_indices:
        raise ValueError(
            f"Frame {frame_index} was passed to the structural audit but the explicit "
            "structural_audit.frame_indices are "
            f"{config.structural_audit.frame_indices}."
        )
    if positions_A.ndim != 2 or positions_A.shape[1] != 3:
        raise ValueError(
            f"Structural audit frame {frame_index} positions must have shape (N, 3), "
            f"got {positions_A.shape}."
        )
    expected_state_shape = (positions_A.shape[0],)
    if true_state_ids.shape != expected_state_shape:
        raise ValueError(
            f"Structural audit frame {frame_index} true state ids must have shape "
            f"{expected_state_shape}, got {true_state_ids.shape}."
        )
    if not np.isfinite(positions_A).all():
        bad_atom_indices = np.flatnonzero(~np.isfinite(positions_A).all(axis=1))[:20]
        raise ValueError(
            f"Structural audit frame {frame_index} contains non-finite coordinates for "
            f"atom indices {bad_atom_indices.tolist()}."
        )
    invalid_state_mask = (true_state_ids < 0) | (
        true_state_ids >= len(graph.state_names)
    )
    if np.any(invalid_state_mask):
        invalid_state_ids = np.unique(true_state_ids[invalid_state_mask])
        raise ValueError(
            f"Structural audit frame {frame_index} contains state ids outside the "
            f"transition graph: {invalid_state_ids.tolist()}; state_count="
            f"{len(graph.state_names)}."
        )
    chemical_symbol = config.rendering.source_chemical_symbol
    if chemical_symbol is None:
        raise RuntimeError(
            "Structural audit requires rendering.source_chemical_symbol from the verified "
            "force-driven source identity, but it is missing. Reload the temporal YAML "
            "through load_temporal_config instead of constructing an unverified config."
        )
    structure_types = _ptm_structure_types(
        positions_A=positions_A,
        box_size_A=float(config.domain.box_size),
        chemical_symbol=chemical_symbol,
        rmsd_cutoff=float(config.structural_audit.ptm_rmsd_cutoff),
    )
    if structure_types.shape != expected_state_shape:
        raise RuntimeError(
            f"OVITO PTM returned shape {structure_types.shape} for structural audit frame "
            f"{frame_index}, expected {expected_state_shape}."
        )
    unexpected_structure_types = sorted(
        set(np.unique(structure_types).tolist()) - set(_PTM_STRUCTURE_TYPE_NAMES)
    )
    if unexpected_structure_types:
        raise RuntimeError(
            f"OVITO PTM returned unsupported structure type ids "
            f"{unexpected_structure_types} for frame {frame_index}; supported mapping is "
            f"{_PTM_STRUCTURE_TYPE_NAMES}."
        )

    crystalline_mask = np.isin(
        structure_types, _PTM_CRYSTALLINE_STRUCTURE_TYPES
    )
    structure_counts = {
        structure_name: int(np.count_nonzero(structure_types == structure_id))
        for structure_id, structure_name in _PTM_STRUCTURE_TYPE_NAMES.items()
    }
    by_true_state: dict[str, dict[str, Any]] = {}
    for state_id, state_name in enumerate(graph.state_names):
        state_mask = true_state_ids == state_id
        atom_count = int(np.count_nonzero(state_mask))
        crystalline_count = int(np.count_nonzero(crystalline_mask & state_mask))
        state_structure_counts = {
            structure_name: int(
                np.count_nonzero(state_mask & (structure_types == structure_id))
            )
            for structure_id, structure_name in _PTM_STRUCTURE_TYPE_NAMES.items()
        }
        by_true_state[state_name] = {
            "atom_count": atom_count,
            "crystalline_atom_count": crystalline_count,
            "crystalline_fraction": (
                float(crystalline_count / atom_count) if atom_count else None
            ),
            "ptm_structure_type_counts": state_structure_counts,
            "ptm_structure_type_fractions": {
                structure_name: (
                    float(structure_count / atom_count) if atom_count else None
                )
                for structure_name, structure_count in state_structure_counts.items()
            },
        }
    return StructuralAuditFrame(
        frame_index=int(frame_index),
        atom_count=int(positions_A.shape[0]),
        ptm_structure_type_counts=structure_counts,
        by_true_state=by_true_state,
    )


def summarize_structural_audit(
    *,
    config: TemporalBenchmarkConfig,
    graph: TransitionGraph,
    frames: list[StructuralAuditFrame],
) -> dict[str, Any]:
    expected_frame_indices = list(config.structural_audit.frame_indices)
    observed_frame_indices = [frame.frame_index for frame in frames]
    if observed_frame_indices != expected_frame_indices:
        raise RuntimeError(
            "Structural audit did not receive exactly the configured rendered frames in "
            f"order: expected={expected_frame_indices}, observed={observed_frame_indices}."
        )

    aggregate_counts: dict[str, dict[str, Any]] = {
        state_name: {
            "atom_count": 0,
            "crystalline_atom_count": 0,
            "ptm_structure_type_counts": {
                structure_name: 0
                for structure_name in _PTM_STRUCTURE_TYPE_NAMES.values()
            },
        }
        for state_name in graph.state_names
    }
    for frame in frames:
        if set(frame.by_true_state) != set(graph.state_names):
            raise RuntimeError(
                f"Structural audit frame {frame.frame_index} has state records "
                f"{sorted(frame.by_true_state)}, expected {sorted(graph.state_names)}."
            )
        for state_name in graph.state_names:
            state_record = frame.by_true_state[state_name]
            aggregate_counts[state_name]["atom_count"] += int(
                state_record["atom_count"]
            )
            aggregate_counts[state_name]["crystalline_atom_count"] += int(
                state_record["crystalline_atom_count"]
            )
            for structure_name in _PTM_STRUCTURE_TYPE_NAMES.values():
                aggregate_counts[state_name]["ptm_structure_type_counts"][
                    structure_name
                ] += int(
                    state_record["ptm_structure_type_counts"][structure_name]
                )

    minimum_atoms = int(
        config.structural_audit.minimum_aggregate_atoms_per_state
    )
    insufficient_states = {
        state_name: int(aggregate_counts[state_name]["atom_count"])
        for state_name in config.structural_audit.state_names
        if aggregate_counts[state_name]["atom_count"] < minimum_atoms
    }
    if insufficient_states:
        raise RuntimeError(
            "Structural PTM audit has inadequate atom support across the selected frames: "
            f"observed={insufficient_states}, required_minimum_per_state={minimum_atoms}, "
            f"selected_frames={expected_frame_indices}. Select frames containing every "
            "audited state or lower the threshold only with a documented statistical "
            "justification."
        )

    aggregate_by_state: dict[str, dict[str, Any]] = {}
    for state_name, counts in aggregate_counts.items():
        atom_count = int(counts["atom_count"])
        crystalline_count = int(counts["crystalline_atom_count"])
        state_structure_counts = {
            structure_name: int(structure_count)
            for structure_name, structure_count in counts[
                "ptm_structure_type_counts"
            ].items()
        }
        aggregate_by_state[state_name] = {
            "atom_count": atom_count,
            "crystalline_atom_count": crystalline_count,
            "crystalline_fraction": (
                float(crystalline_count / atom_count) if atom_count else 0.0
            ),
            "ptm_structure_type_counts": state_structure_counts,
            "ptm_structure_type_fractions": {
                structure_name: (
                    float(structure_count / atom_count) if atom_count else 0.0
                )
                for structure_name, structure_count in state_structure_counts.items()
            },
        }

    crystal_state_name = config.structural_audit.crystal_state_name
    liquid_state_name = config.structural_audit.liquid_state_name
    crystal_fraction = float(
        aggregate_by_state[crystal_state_name]["crystalline_fraction"]
    )
    liquid_fraction = float(
        aggregate_by_state[liquid_state_name]["crystalline_fraction"]
    )
    observed_margin = crystal_fraction - liquid_fraction
    required_margin = float(
        config.structural_audit.minimum_crystal_liquid_fraction_margin
    )
    if observed_margin < required_margin:
        raise RuntimeError(
            "Structural PTM audit failed the configured crystal/liquid ordering: "
            f"state {crystal_state_name!r} FCC/HCP/BCC fraction={crystal_fraction:.6%}, "
            f"state {liquid_state_name!r} fraction={liquid_fraction:.6%}, observed_margin="
            f"{observed_margin:.6%}, required_margin={required_margin:.6%}. The procedural "
            "renderer did not produce coordinates that support its true state names; do "
            "not use this dataset as structurally qualified crystallization data."
        )
    minimum_crystal_fraction = float(
        config.structural_audit.minimum_crystal_crystalline_fraction
    )
    if crystal_fraction < minimum_crystal_fraction:
        raise RuntimeError(
            "Structural PTM audit failed the absolute crystal-state requirement: "
            f"state {crystal_state_name!r} FCC/HCP/BCC fraction={crystal_fraction:.6%}, "
            f"required_minimum={minimum_crystal_fraction:.6%}. A relative separation from "
            "liquid is insufficient when most true crystal-labelled atoms remain "
            "PTM-other. Redesign the coherent crystal target instead of weakening this "
            "qualification gate."
        )
    crystal_fcc_fraction = float(
        aggregate_by_state[crystal_state_name]["ptm_structure_type_fractions"][
            "fcc"
        ]
    )
    minimum_crystal_fcc_fraction = float(
        config.structural_audit.minimum_crystal_fcc_fraction
    )
    if crystal_fcc_fraction < minimum_crystal_fcc_fraction:
        raise RuntimeError(
            "Structural PTM audit failed the declared FCC crystal-state requirement: "
            f"state {crystal_state_name!r} FCC fraction={crystal_fcc_fraction:.6%}, "
            f"required_minimum={minimum_crystal_fcc_fraction:.6%}. The state declares "
            "crystal_structure='fcc'; HCP or BCC matches cannot substitute for FCC."
        )
    maximum_liquid_fraction = float(
        config.structural_audit.maximum_liquid_crystalline_fraction
    )
    if liquid_fraction > maximum_liquid_fraction:
        raise RuntimeError(
            "Structural PTM audit failed the absolute liquid-state requirement: "
            f"state {liquid_state_name!r} FCC/HCP/BCC fraction={liquid_fraction:.6%}, "
            f"allowed_maximum={maximum_liquid_fraction:.6%}. The procedural liquid labels "
            "contain excessive crystalline order."
        )

    return {
        "passed": True,
        "definition": (
            "OVITO Polyhedral Template Matching is recomputed from each selected rendered "
            "full-box frame. FCC, HCP, and BCC are counted as crystalline, then aggregated "
            "by the renderer's true per-atom state_ids. PTM is an audit observable and does "
            "not replace the procedural labels."
        ),
        "backend": "OVITO PolyhedralTemplateMatchingModifier",
        "chemical_symbol": config.rendering.source_chemical_symbol,
        "periodic_cell_vectors_A": (
            np.eye(3, dtype=np.float64) * float(config.domain.box_size)
        ).tolist(),
        "ptm_normalized_rmsd_cutoff": float(
            config.structural_audit.ptm_rmsd_cutoff
        ),
        "crystalline_structure_types": ["fcc", "hcp", "bcc"],
        "selected_frame_indices": expected_frame_indices,
        "audited_state_names": list(config.structural_audit.state_names),
        "atom_sufficiency_scope": "aggregate across selected_frame_indices",
        "minimum_aggregate_atoms_per_state": minimum_atoms,
        "aggregate_by_true_state": aggregate_by_state,
        "crystal_liquid_check": {
            "crystal_state_name": crystal_state_name,
            "liquid_state_name": liquid_state_name,
            "crystal_crystalline_fraction": crystal_fraction,
            "liquid_crystalline_fraction": liquid_fraction,
            "observed_fraction_margin": observed_margin,
            "minimum_required_fraction_margin": required_margin,
            "passed": True,
        },
        "absolute_phase_checks": {
            "minimum_crystal_crystalline_fraction": minimum_crystal_fraction,
            "minimum_crystal_fcc_fraction": minimum_crystal_fcc_fraction,
            "maximum_liquid_crystalline_fraction": maximum_liquid_fraction,
            "observed_crystal_fcc_fraction": crystal_fcc_fraction,
            "crystal_passed": True,
            "crystal_fcc_passed": True,
            "liquid_passed": True,
        },
        "frames": [frame.to_dict() for frame in frames],
    }


def _ptm_structure_types(
    *,
    positions_A: np.ndarray,
    box_size_A: float,
    chemical_symbol: str,
    rmsd_cutoff: float,
) -> np.ndarray:
    try:
        from ase import Atoms
        from ase.data import atomic_numbers
        from ovito.io.ase import ase_to_ovito
        from ovito.modifiers import PolyhedralTemplateMatchingModifier
        from ovito.pipeline import Pipeline, StaticSource
    except ImportError as exc:
        raise ImportError(
            "Temporal structural validation requires ASE and OVITO Polyhedral Template "
            "Matching. Install the repository requirements in the pointnet environment."
        ) from exc
    if chemical_symbol not in atomic_numbers:
        raise ValueError(
            f"Verified source chemical symbol {chemical_symbol!r} is not recognized by ASE."
        )
    numbers = np.full(
        positions_A.shape[0], atomic_numbers[chemical_symbol], dtype=np.int32
    )
    atoms = Atoms(
        numbers=numbers,
        positions=positions_A,
        cell=np.eye(3, dtype=np.float64) * box_size_A,
        pbc=True,
    )
    pipeline = Pipeline(source=StaticSource(data=ase_to_ovito(atoms)))
    modifier = PolyhedralTemplateMatchingModifier()
    modifier.rmsd_cutoff = rmsd_cutoff
    pipeline.modifiers.append(modifier)
    data = pipeline.compute()
    return np.asarray(data.particles["Structure Type"], dtype=np.int32).copy()


def validate_temporal_dataset(
    config: TemporalBenchmarkConfig,
    graph: TransitionGraph,
    layout: SiteLayout,
    latent: LatentTrajectories,
    local_points_by_frame: np.ndarray,
    point_state_ids_by_frame: np.ndarray,
    point_grain_ids_by_frame: np.ndarray,
    frame_atom_counts: list[int],
    frame_minimum_pair_distances: list[float],
    structural_audit_frames: list[StructuralAuditFrame],
) -> ValidationSummary:
    expected_shape = (config.time.num_frames, config.domain.site_count)
    if latent.state_ids.shape != expected_shape:
        raise ValueError(
            f"latent.state_ids must have shape {expected_shape}, got {latent.state_ids.shape}."
        )
    if local_points_by_frame.shape[:2] != expected_shape:
        raise ValueError(
            f"local_points_by_frame shape prefix must be {expected_shape}, got {local_points_by_frame.shape}."
        )
    expected_point_label_shape = local_points_by_frame.shape[:-1]
    if point_state_ids_by_frame.shape != expected_point_label_shape:
        raise ValueError(
            "point_state_ids_by_frame must label every local point: "
            f"expected {expected_point_label_shape}, got {point_state_ids_by_frame.shape}."
        )
    if point_grain_ids_by_frame.shape != expected_point_label_shape:
        raise ValueError(
            "point_grain_ids_by_frame must label every local point: "
            f"expected {expected_point_label_shape}, got {point_grain_ids_by_frame.shape}."
        )
    if np.any(point_state_ids_by_frame < 0) or np.any(
        point_state_ids_by_frame >= len(graph.state_names)
    ):
        invalid = np.unique(
            point_state_ids_by_frame[
                (point_state_ids_by_frame < 0)
                | (point_state_ids_by_frame >= len(graph.state_names))
            ]
        )
        raise ValueError(
            "point_state_ids_by_frame contains labels outside the transition graph: "
            f"invalid_ids={invalid.tolist()}, state_count={len(graph.state_names)}."
        )
    if not frame_atom_counts:
        raise ValueError("frame_atom_counts is empty; validation requires per-frame atom counts.")
    if any(count <= 0 for count in frame_atom_counts):
        raise ValueError(
            f"Frame atom counts must all be positive, got {frame_atom_counts}."
        )
    if len(frame_minimum_pair_distances) != config.time.num_frames:
        raise ValueError(
            "frame_minimum_pair_distances must contain one value per frame, got "
            f"{len(frame_minimum_pair_distances)} for {config.time.num_frames} frames."
        )
    required_minimum_pair_distance = min(
        float(state.template_params.get("min_pair_distance", 0.75 * config.domain.avg_nn_distance))
        for state in config.states
    )
    observed_minimum_pair_distance = float(np.min(frame_minimum_pair_distances))
    if not np.isfinite(observed_minimum_pair_distance):
        raise ValueError(
            "Periodic minimum-pair-distance validation received a non-finite value: "
            f"{frame_minimum_pair_distances}."
        )
    if observed_minimum_pair_distance < required_minimum_pair_distance:
        raise ValueError(
            "Rendered temporal frames violate the periodic minimum-distance invariant: "
            f"observed_minimum={observed_minimum_pair_distance:.6f} A, required="
            f"{required_minimum_pair_distance:.6f} A."
        )
    for array_name, array_value in {
        "state_ids": latent.state_ids,
        "grain_ids": latent.grain_ids,
        "crystal_variant_ids": latent.crystal_variant_ids,
        "orientation_quaternions": latent.orientation_quaternions,
        "strain": latent.strain,
        "thermal_jitter": latent.thermal_jitter,
        "defect_amplitude": latent.defect_amplitude,
        "local_points_by_frame": local_points_by_frame,
    }.items():
        if np.issubdtype(array_value.dtype, np.floating) and not np.isfinite(
            array_value
        ).all():
            raise ValueError(f"Non-finite values detected in {array_name}.")

    transition_counts: Counter[str] = Counter()
    dwell_by_state: defaultdict[str, list[int]] = defaultdict(list)
    complete_phase_dwell_counts: Counter[str] = Counter()

    for phase_site_id in range(layout.phase_site_count):
        segment_start = 0
        for frame_idx in range(1, config.time.num_frames):
            previous_state = int(latent.phase_state_ids[frame_idx - 1, phase_site_id])
            current_state = int(latent.phase_state_ids[frame_idx, phase_site_id])
            if not graph.is_valid_transition(previous_state, current_state):
                raise ValueError(
                    "Invalid phase-field transition: "
                    f"phase_site_id={phase_site_id}, frame_index={frame_idx}, "
                    f"{graph.name(previous_state)} -> {graph.name(current_state)}."
                )
            if current_state == previous_state:
                continue
            is_seed_intervention = bool(
                latent.phase_seed_site_mask[phase_site_id]
            ) and frame_idx == int(config.dynamics.nucleation_start_frame)
            if not is_seed_intervention:
                state_name = graph.name(previous_state)
                segment_length = frame_idx - segment_start
                dwell = graph.state_config(state_name).dwell
                if not int(dwell.min_steps) <= segment_length <= int(dwell.max_steps):
                    raise ValueError(
                        "Complete semi-Markov dwell violates its configured bounds: "
                        f"phase_site_id={phase_site_id}, state={state_name!r}, "
                        f"segment=[{segment_start}, {frame_idx}), length={segment_length}, "
                        f"configured=[{dwell.min_steps}, {dwell.max_steps}]."
                    )
                complete_phase_dwell_counts[state_name] += 1
            segment_start = frame_idx

    if np.any(latent.phase_seed_site_mask):
        nucleation_state_name = config.dynamics.nucleation_state
        if nucleation_state_name is None:
            nucleation_state_name = config.dynamics.primary_path[1]
        nucleation_state_id = graph.index(nucleation_state_name)
        start_frame = int(config.dynamics.nucleation_start_frame)
        if np.any(
            latent.phase_state_ids[:start_frame, latent.phase_seed_site_mask]
            == nucleation_state_id
        ):
            raise ValueError(
                "Seed phase sites entered the nucleation state before "
                f"dynamics.nucleation_start_frame={start_frame}."
            )
        activated_states = latent.phase_state_ids[
            start_frame, latent.phase_seed_site_mask
        ]
        if not np.all(activated_states == nucleation_state_id):
            raise ValueError(
                f"Not every seed phase site activated as {nucleation_state_name!r} at "
                f"frame {start_frame}."
            )
        activated_grains = latent.phase_grain_ids[
            start_frame, latent.phase_seed_site_mask
        ]
        if np.any(activated_grains < 0) or len(np.unique(activated_grains)) != 1:
            raise ValueError(
                "The activated nucleation seed blob must share exactly one non-negative "
                f"grain id, got {np.unique(activated_grains).tolist()}."
            )

    for site_id in range(config.domain.site_count):
        segment_start = 0
        for frame_idx in range(1, config.time.num_frames):
            prev_state = int(latent.state_ids[frame_idx - 1, site_id])
            curr_state = int(latent.state_ids[frame_idx, site_id])
            if not graph.is_valid_transition(prev_state, curr_state):
                raise ValueError(
                    f"Invalid state transition at site_id={site_id}, frame_index={frame_idx}: "
                    f"{graph.name(prev_state)} -> {graph.name(curr_state)} is not in the transition graph."
                )
            if curr_state != prev_state:
                transition_counts[f"{graph.name(prev_state)}->{graph.name(curr_state)}"] += 1
                dwell_by_state[graph.name(prev_state)].append(frame_idx - segment_start)
                segment_start = frame_idx
        final_state = graph.name(int(latent.state_ids[-1, site_id]))
        dwell_by_state[final_state].append(config.time.num_frames - segment_start)

    state_counts = Counter(graph.name(int(state_idx)) for state_idx in latent.state_ids.reshape(-1))
    mean_dwell_by_state = {
        state_name: float(np.mean(durations))
        for state_name, durations in dwell_by_state.items()
        if durations
    }
    grain_ids = latent.grain_ids[latent.grain_ids >= 0]
    same_state_fraction = np.mean(
        point_state_ids_by_frame == latent.state_ids[:, :, None], axis=2
    )
    structural_audit = summarize_structural_audit(
        config=config,
        graph=graph,
        frames=structural_audit_frames,
    )
    summary = {
        "dataset_name": config.dataset_name,
        "num_frames": int(config.time.num_frames),
        "site_count": int(layout.site_count),
        "points_per_site": int(config.domain.atoms_per_site),
        "frame_atom_count": int(frame_atom_counts[0]),
        "frame_atom_count_min": int(np.min(frame_atom_counts)),
        "frame_atom_count_max": int(np.max(frame_atom_counts)),
        "frame_atom_count_mean": float(np.mean(frame_atom_counts)),
        "minimum_periodic_pair_distance": observed_minimum_pair_distance,
        "required_minimum_pair_distance": required_minimum_pair_distance,
        "num_grains": int(len(np.unique(grain_ids))) if grain_ids.size > 0 else 0,
        "transition_event_count": int(sum(transition_counts.values())),
        "state_frame_counts": dict(state_counts),
        "transition_counts": dict(transition_counts),
        "crystal_variant_fraction": float(np.mean(latent.crystal_variant_ids == 1)),
        "mean_dwell_by_state": mean_dwell_by_state,
        "complete_phase_dwell_counts": dict(complete_phase_dwell_counts),
        "metastable_fraction": float(np.mean(latent.metastable_mask)),
        "seed_site_fraction": float(np.mean(latent.seed_site_mask)),
        "central_context_mean_same_state_fraction": float(np.mean(same_state_fraction)),
        "central_context_minimum_same_state_fraction": float(np.min(same_state_fraction)),
        "central_context_fraction_below_half": float(np.mean(same_state_fraction < 0.5)),
        "structural_audit": structural_audit,
    }
    return ValidationSummary(summary=summary)
