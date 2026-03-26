from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .config import DwellTimeConfig, DynamicsConfig, NuisanceConfig, TemporalBenchmarkConfig
from .geometry import (
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_slerp,
    quaternion_slerp_batch,
    random_rotation_matrix,
    random_unit_vector,
    rotation_matrix_to_quaternion,
)
from .graph import TransitionGraph


@dataclass(frozen=True)
class SiteLayout:
    centers: np.ndarray
    neighbor_indices: np.ndarray
    phase_centers: np.ndarray
    phase_neighbor_indices: np.ndarray
    tracked_to_phase_index: np.ndarray

    @property
    def site_count(self) -> int:
        return int(self.centers.shape[0])

    @property
    def phase_site_count(self) -> int:
        return int(self.phase_centers.shape[0])


@dataclass(frozen=True)
class TransitionEvent:
    frame_index: int
    site_id: int
    source_state: str
    target_state: str
    source_grain_id: int
    target_grain_id: int


@dataclass(frozen=True)
class LatentTrajectories:
    state_ids: np.ndarray
    grain_ids: np.ndarray
    crystal_variant_ids: np.ndarray
    orientation_quaternions: np.ndarray
    strain: np.ndarray
    thermal_jitter: np.ndarray
    defect_amplitude: np.ndarray
    dwell_total: np.ndarray
    dwell_remaining: np.ndarray
    segment_ids: np.ndarray
    transition_mask: np.ndarray
    metastable_mask: np.ndarray
    seed_site_mask: np.ndarray
    transition_events: list[TransitionEvent]
    grain_orientations: dict[int, np.ndarray]
    phase_state_ids: np.ndarray
    phase_grain_ids: np.ndarray
    phase_crystal_variant_ids: np.ndarray
    phase_orientation_quaternions: np.ndarray
    phase_strain: np.ndarray
    phase_thermal_jitter: np.ndarray
    phase_defect_amplitude: np.ndarray
    phase_seed_site_mask: np.ndarray


@dataclass(frozen=True)
class NeighborPhaseContext:
    liquid_fraction: float
    precursor_fraction: float
    interface_fraction: float
    crystal_fraction: float
    grain_boundary_fraction: float
    liquid_like_fraction: float
    solid_fraction: float
    distinct_crystal_grains: int


def build_site_layout(config: TemporalBenchmarkConfig) -> SiteLayout:
    domain = config.domain
    centers = _build_site_centers(domain=domain, seed=int(config.seed))
    tree = cKDTree(centers)
    k = min(domain.site_count, max(1, int(config.dynamics.neighbor_count)) + 1)
    _, indices = tree.query(centers, k=k)
    if k == 1:
        neighbor_indices = np.full((centers.shape[0], 0), -1, dtype=np.int32)
    else:
        neighbor_indices = np.asarray(indices[:, 1:], dtype=np.int32)
    phase_centers = _build_phase_field_centers(config=config, seed=int(config.seed))
    phase_tree = cKDTree(phase_centers)
    phase_k = min(phase_centers.shape[0], max(1, int(config.dynamics.neighbor_count)) + 1)
    _, phase_indices = phase_tree.query(phase_centers, k=phase_k)
    if phase_k == 1:
        phase_neighbor_indices = np.full((phase_centers.shape[0], 0), -1, dtype=np.int32)
    else:
        phase_neighbor_indices = np.asarray(phase_indices[:, 1:], dtype=np.int32)
    _, tracked_to_phase_index = phase_tree.query(centers, k=1)
    return SiteLayout(
        centers=centers.astype(np.float32),
        neighbor_indices=neighbor_indices,
        phase_centers=phase_centers.astype(np.float32),
        phase_neighbor_indices=phase_neighbor_indices.astype(np.int32),
        tracked_to_phase_index=np.asarray(tracked_to_phase_index, dtype=np.int32),
    )


def simulate_latent_trajectories(
    config: TemporalBenchmarkConfig,
    graph: TransitionGraph,
    layout: SiteLayout,
    rng: np.random.Generator,
) -> LatentTrajectories:
    num_frames = int(config.time.num_frames)
    site_count = int(layout.site_count)
    phase_site_count = int(layout.phase_site_count)
    phase_state_ids = np.full((num_frames, phase_site_count), -1, dtype=np.int16)
    phase_grain_ids = np.full((num_frames, phase_site_count), -1, dtype=np.int32)
    phase_crystal_variant_ids = np.zeros((num_frames, phase_site_count), dtype=np.int8)
    phase_orientation_quaternions = np.zeros((num_frames, phase_site_count, 4), dtype=np.float32)
    phase_strain = np.zeros((num_frames, phase_site_count, 3), dtype=np.float32)
    phase_thermal_jitter = np.zeros((num_frames, phase_site_count), dtype=np.float32)
    phase_defect_amplitude = np.zeros((num_frames, phase_site_count), dtype=np.float32)

    states = graph.state_map
    grain_orientations: dict[int, np.ndarray] = {}
    next_grain_id = 0

    phase_seed_site_mask = np.zeros(phase_site_count, dtype=bool)
    nucleation_state_idx = _resolve_nucleation_state(config.dynamics, graph)
    if config.dynamics.mode == "coupled" and config.dynamics.nucleation_seed_fraction > 0.0:
        seed_count = max(1, int(round(config.dynamics.nucleation_seed_fraction * max(layout.site_count, 1))))
        seed_count = min(seed_count, phase_site_count)
        seed_indices = rng.choice(phase_site_count, size=seed_count, replace=False)
        phase_seed_site_mask[np.asarray(seed_indices, dtype=np.int32)] = True
        blob_hops = int(config.dynamics.nucleation_blob_hops)
        if blob_hops > 0:
            phase_seed_site_mask = _expand_seed_mask(
                seed_mask=phase_seed_site_mask,
                neighbor_indices=layout.phase_neighbor_indices,
                hops=blob_hops,
            )

    current_state = np.empty(phase_site_count, dtype=np.int16)
    current_grain = np.full(phase_site_count, -1, dtype=np.int32)
    current_crystal_variant = np.zeros(phase_site_count, dtype=np.int8)
    current_orientation = np.zeros((phase_site_count, 4), dtype=np.float32)
    current_strain = np.zeros((phase_site_count, 3), dtype=np.float32)
    current_thermal = np.zeros(phase_site_count, dtype=np.float32)
    current_defect = np.zeros(phase_site_count, dtype=np.float32)

    # Vectorized initial state sampling
    init_states_list = []
    init_probs_list = []
    for sn, prob in config.dynamics.initial_state_probs.items():
        if prob > 0.0:
            init_states_list.append(graph.index(sn))
            init_probs_list.append(float(prob))
    init_probs_arr = np.array(init_probs_list, dtype=np.float64)
    init_probs_arr /= init_probs_arr.sum()
    init_states_arr = np.array(init_states_list, dtype=np.int16)
    sampled_indices = rng.choice(init_states_arr, size=phase_site_count, p=init_probs_arr)
    if nucleation_state_idx is not None:
        sampled_indices[phase_seed_site_mask] = nucleation_state_idx
    current_state[:] = sampled_indices

    # Initial grain assignment (sequential because it allocates grain IDs)
    for site_id in range(phase_site_count):
        state_name = graph.state_names[int(current_state[site_id])]
        current_grain[site_id], next_grain_id = _assign_initial_grain(
            state_name=state_name,
            states=states,
            is_seed=bool(phase_seed_site_mask[site_id]),
            next_grain_id=next_grain_id,
            grain_orientations=grain_orientations,
            rng=rng,
        )
        current_orientation[site_id] = _initial_orientation(
            grain_id=current_grain[site_id],
            grain_orientations=grain_orientations,
            rng=rng,
        )
    # Vectorized thermal/defect initialization
    for si, sn in enumerate(graph.state_names):
        mask = current_state == si
        current_thermal[mask] = states[sn].base_thermal_jitter
        current_defect[mask] = states[sn].base_defect_amplitude

    # Pre-compute state index lookups for vectorized dynamics
    _state_idx_map = {name: graph.index(name) for name in graph.state_names}
    _L = _state_idx_map.get("L", -1)
    _P = _state_idx_map.get("P", -1)
    _I = _state_idx_map.get("I", -1)
    _C = _state_idx_map.get("C", -1)
    _G = _state_idx_map.get("G", -1)
    _D = _state_idx_map.get("D", -1)

    # Pre-compute template_kind classification arrays (indexed by state_idx)
    num_states = len(graph.state_names)
    _is_liquid_like = np.zeros(num_states, dtype=bool)
    _is_interface = np.zeros(num_states, dtype=bool)
    _is_solid_core = np.zeros(num_states, dtype=bool)
    _is_grain_bearing = np.zeros(num_states, dtype=bool)
    _allows_new_grain = np.zeros(num_states, dtype=bool)
    for _si, _sn in enumerate(graph.state_names):
        _sc = states[_sn]
        _is_liquid_like[_si] = _sc.template_kind in {"liquid", "precursor"}
        _is_interface[_si] = _sc.template_kind == "interface"
        _is_solid_core[_si] = _sc.template_kind in {"crystal", "defective_crystal", "grain_boundary"}
        _is_grain_bearing[_si] = _sc.grain_bearing
        _allows_new_grain[_si] = _sc.grain_bearing and _sc.template_kind in {
            "interface", "crystal", "grain_boundary", "defective_crystal",
        }
    _base_strain_scale_by_state = np.array(
        [states[state_name].base_strain_scale for state_name in graph.state_names],
        dtype=np.float32,
    )
    _base_thermal_by_state = np.array(
        [states[state_name].base_thermal_jitter for state_name in graph.state_names],
        dtype=np.float32,
    )
    _base_defect_by_state = np.array(
        [states[state_name].base_defect_amplitude for state_name in graph.state_names],
        dtype=np.float32,
    )

    # Pre-compute edge weight lookup: edge_weight_matrix[src_idx, tgt_idx]
    edge_weight_matrix = np.zeros((num_states, num_states), dtype=np.float64)
    for _sn in graph.state_names:
        for edge in graph.outgoing_edges(_sn):
            edge_weight_matrix[graph.index(_sn), graph.index(edge.target)] = edge.weight

    # Neighbor indices as padded array (already int32, -1 for invalid)
    _neigh = layout.phase_neighbor_indices  # (phase_site_count, max_neighbors)
    _neigh_valid = (_neigh >= 0)  # (phase_site_count, max_neighbors)
    _n_valid = _neigh_valid.sum(axis=1).astype(np.float64)  # (phase_site_count,)
    _n_valid_safe = np.maximum(_n_valid, 1.0)

    for frame_idx in range(num_frames):
        if frame_idx > 0:
            previous_state = current_state.copy()
            previous_grain = current_grain.copy()

            # --- Vectorized coupled transition sampling ---
            current_state = _sample_transitions_batch(
                previous_state=previous_state,
                previous_grain=previous_grain,
                neigh=_neigh,
                neigh_valid=_neigh_valid,
                n_valid_safe=_n_valid_safe,
                phase_seed_site_mask=phase_seed_site_mask,
                graph=graph,
                dynamics=config.dynamics,
                edge_weight_matrix=edge_weight_matrix,
                is_liquid_like=_is_liquid_like,
                is_interface=_is_interface,
                is_solid_core=_is_solid_core,
                state_indices=(_L, _P, _I, _C, _G, _D),
                frame_idx=frame_idx,
                rng=rng,
            )

            # --- Vectorized grain assignment ---
            current_grain, next_grain_id = _update_grains_batch(
                current_state=current_state,
                previous_grain=previous_grain,
                neigh=_neigh,
                neigh_valid=_neigh_valid,
                phase_seed_site_mask=phase_seed_site_mask,
                is_grain_bearing=_is_grain_bearing,
                allows_new_grain=_allows_new_grain,
                precursor_state_idx=_P,
                next_grain_id=next_grain_id,
                grain_orientations=grain_orientations,
                rng=rng,
            )

        # --- Vectorized crystal variant update ---
        current_crystal_variant = _update_crystal_variant_ids_batch(
            current_variant_ids=current_crystal_variant,
            current_states=current_state,
            graph=graph,
            dynamics=config.dynamics,
            rng=rng,
        )

        phase_state_ids[frame_idx] = current_state
        phase_grain_ids[frame_idx] = current_grain
        phase_crystal_variant_ids[frame_idx] = current_crystal_variant

        # --- Vectorized orientation update ---
        current_orientation = _update_orientations_batch(
            current_orientations=current_orientation,
            grain_ids=current_grain,
            grain_orientations=grain_orientations,
            nuisance=config.nuisance,
            rng=rng,
        )

        # --- Vectorized AR(1) strain update ---
        current_strain = _update_ar1_vectors_batch(
            current=current_strain,
            relaxation=config.nuisance.strain_relaxation,
            noise_scale=config.nuisance.strain_drift_scale,
            mean_scales=_base_strain_scale_by_state[current_state],
            rng=rng,
        )

        # --- Vectorized AR(1) thermal update ---
        current_thermal = _update_ar1_scalars_batch(
            current=current_thermal,
            relaxation=config.nuisance.thermal_relaxation,
            noise_scale=config.nuisance.thermal_noise_scale,
            means=_base_thermal_by_state[current_state],
            rng=rng,
            lower=0.0,
        )

        # --- Vectorized AR(1) defect update ---
        current_defect = _update_ar1_scalars_batch(
            current=current_defect,
            relaxation=config.nuisance.defect_relaxation,
            noise_scale=config.nuisance.defect_noise_scale,
            means=_base_defect_by_state[current_state],
            rng=rng,
            lower=0.0,
        )

        phase_orientation_quaternions[frame_idx] = current_orientation
        phase_strain[frame_idx] = current_strain
        phase_thermal_jitter[frame_idx] = current_thermal
        phase_defect_amplitude[frame_idx] = current_defect

    tracked_phase_ids = layout.tracked_to_phase_index.astype(np.int32)
    state_ids = phase_state_ids[:, tracked_phase_ids]
    grain_ids = phase_grain_ids[:, tracked_phase_ids]
    crystal_variant_ids = phase_crystal_variant_ids[:, tracked_phase_ids]
    orientation_quaternions = phase_orientation_quaternions[:, tracked_phase_ids]
    strain = phase_strain[:, tracked_phase_ids]
    thermal_jitter = phase_thermal_jitter[:, tracked_phase_ids]
    defect_amplitude = phase_defect_amplitude[:, tracked_phase_ids]
    seed_site_mask = phase_seed_site_mask[tracked_phase_ids]

    dwell_total, dwell_remaining, segment_ids, metastable_mask = _derive_segment_annotations(
        state_ids=state_ids,
        graph=graph,
        metastable_min_dwell=int(config.dynamics.metastable_min_dwell),
    )
    transition_mask = np.zeros((num_frames, site_count), dtype=bool)
    transition_mask[0] = False
    if num_frames > 1:
        transition_mask[1:] = state_ids[1:] != state_ids[:-1]
    transition_events = _derive_transition_events(
        state_ids=state_ids,
        grain_ids=grain_ids,
        graph=graph,
    )

    return LatentTrajectories(
        state_ids=state_ids,
        grain_ids=grain_ids,
        crystal_variant_ids=crystal_variant_ids,
        orientation_quaternions=orientation_quaternions,
        strain=strain,
        thermal_jitter=thermal_jitter,
        defect_amplitude=defect_amplitude,
        dwell_total=dwell_total,
        dwell_remaining=dwell_remaining,
        segment_ids=segment_ids,
        transition_mask=transition_mask,
        metastable_mask=metastable_mask,
        seed_site_mask=seed_site_mask,
        transition_events=transition_events,
        grain_orientations=grain_orientations,
        phase_state_ids=phase_state_ids,
        phase_grain_ids=phase_grain_ids,
        phase_crystal_variant_ids=phase_crystal_variant_ids,
        phase_orientation_quaternions=phase_orientation_quaternions,
        phase_strain=phase_strain,
        phase_thermal_jitter=phase_thermal_jitter,
        phase_defect_amplitude=phase_defect_amplitude,
        phase_seed_site_mask=phase_seed_site_mask,
    )


def write_transition_events_csv(events: list[TransitionEvent], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["frame_index", "site_id", "source_state", "target_state", "source_grain_id", "target_grain_id"]
        )
        for event in events:
            writer.writerow(
                [
                    event.frame_index,
                    event.site_id,
                    event.source_state,
                    event.target_state,
                    event.source_grain_id,
                    event.target_grain_id,
                ]
            )


def _build_site_centers(domain, seed: int) -> np.ndarray:
    if domain.layout == "grid":
        return _grid_centers(domain)
    if domain.layout == "random":
        rng = np.random.default_rng(seed + 17)
        return _random_centers(domain, rng)
    raise ValueError(
        f"Unsupported domain.layout={domain.layout!r}. Expected 'grid' or 'random'."
    )


def _build_phase_field_centers(config: TemporalBenchmarkConfig, seed: int) -> np.ndarray:
    domain = config.domain
    rendering = config.rendering
    target_radius = float(rendering.phase_region_radius_nn) * float(domain.avg_nn_distance)
    if target_radius <= 0.0:
        raise ValueError(
            "rendering.phase_region_radius_nn must be positive, "
            f"got {rendering.phase_region_radius_nn}."
        )
    spacing = max(1.35 * float(domain.avg_nn_distance), 2.0 * target_radius)
    if rendering.phase_region_target_atoms is not None:
        target_atoms = int(rendering.phase_region_target_atoms)
        if target_atoms <= 0:
            raise ValueError(
                "rendering.phase_region_target_atoms must be positive when provided, "
                f"got {rendering.phase_region_target_atoms}."
            )
        target_spacing = float((target_atoms / float(rendering.target_density)) ** (1.0 / 3.0))
        spacing = max(spacing, target_spacing)
    jitter = float(np.clip(rendering.phase_field_jitter_fraction, 0.0, 0.45)) * spacing
    axis_values = _phase_axis_values(
        start=float(domain.padding),
        stop=float(domain.box_size) - float(domain.padding),
        spacing=spacing,
    )
    if axis_values.size == 0:
        raise RuntimeError(
            "Phase field center generation produced no axis points. "
            f"box_size={domain.box_size}, padding={domain.padding}, spacing={spacing:.3f}."
        )
    rng = np.random.default_rng(seed + 4_091)
    centers = []
    lower = float(domain.padding)
    upper = float(domain.box_size) - float(domain.padding)
    for x in axis_values:
        for y in axis_values:
            for z in axis_values:
                candidate = np.array([x, y, z], dtype=np.float32)
                if jitter > 0.0:
                    candidate += rng.uniform(-jitter, jitter, size=3).astype(np.float32)
                    candidate = np.clip(candidate, lower, upper)
                centers.append(candidate)
    if not centers:
        raise RuntimeError(
            "Phase field center generation failed to create any centers. "
            f"axis_values={axis_values.tolist()}."
        )
    return np.asarray(centers, dtype=np.float32)


def _phase_axis_values(*, start: float, stop: float, spacing: float) -> np.ndarray:
    if spacing <= 0.0:
        raise ValueError(f"Phase field spacing must be positive, got {spacing}.")
    if stop <= start:
        raise ValueError(
            f"Phase field axis bounds must satisfy stop > start, got start={start}, stop={stop}."
        )
    extent = stop - start
    count = max(1, int(np.floor(extent / spacing)) + 1)
    return np.linspace(start, stop, num=count, dtype=np.float32)


def _expand_seed_mask(
    *,
    seed_mask: np.ndarray,
    neighbor_indices: np.ndarray,
    hops: int,
) -> np.ndarray:
    if hops <= 0:
        return seed_mask.astype(bool, copy=True)
    expanded = seed_mask.astype(bool, copy=True)
    for _ in range(hops):
        frontier_sites = np.flatnonzero(expanded)
        if frontier_sites.size == 0:
            break
        frontier_neighbors = neighbor_indices[frontier_sites]
        valid_neighbors = frontier_neighbors[frontier_neighbors >= 0]
        if valid_neighbors.size == 0:
            continue
        expanded[valid_neighbors.astype(np.int32, copy=False)] = True
    return expanded


def _grid_centers(domain) -> np.ndarray:
    site_count = int(domain.site_count)
    grid_side = max(1, int(math.ceil(site_count ** (1.0 / 3.0))))
    required_extent = (grid_side - 1) * float(domain.site_spacing) + 2.0 * float(domain.padding)
    if required_extent > float(domain.box_size):
        raise ValueError(
            f"Grid site scaffold does not fit in the box: required_extent={required_extent:.3f}, "
            f"box_size={domain.box_size}, site_spacing={domain.site_spacing}, padding={domain.padding}, "
            f"site_count={site_count}."
        )
    origin = float(domain.padding)
    points = []
    for ix in range(grid_side):
        for iy in range(grid_side):
            for iz in range(grid_side):
                points.append(
                    [
                        origin + ix * float(domain.site_spacing),
                        origin + iy * float(domain.site_spacing),
                        origin + iz * float(domain.site_spacing),
                    ]
                )
                if len(points) == site_count:
                    return np.asarray(points, dtype=np.float32)
    raise RuntimeError(f"Failed to allocate {site_count} grid centers with grid_side={grid_side}.")


def _random_centers(domain, rng: np.random.Generator) -> np.ndarray:
    centers: list[np.ndarray] = []
    attempts = 0
    min_distance = float(domain.random_min_site_distance)
    while len(centers) < int(domain.site_count) and attempts < 100_000:
        attempts += 1
        candidate = rng.uniform(float(domain.padding), float(domain.box_size) - float(domain.padding), size=3)
        if centers:
            distances = np.linalg.norm(np.asarray(centers) - candidate, axis=1)
            if np.any(distances < min_distance):
                continue
        centers.append(candidate.astype(np.float32))
    if len(centers) < int(domain.site_count):
        raise RuntimeError(
            f"Failed to sample {domain.site_count} random site centers with min_distance={min_distance} "
            f"inside box_size={domain.box_size} after {attempts} attempts."
        )
    return np.asarray(centers, dtype=np.float32)


def _sample_dwell_duration(dwell: DwellTimeConfig, rng: np.random.Generator) -> np.int16:
    distribution = dwell.distribution
    min_steps = int(dwell.min_steps)
    max_steps = int(dwell.max_steps)
    if min_steps <= 0:
        raise ValueError(f"DwellTimeConfig.min_steps must be positive, got {min_steps}.")
    if max_steps < min_steps:
        raise ValueError(
            f"DwellTimeConfig.max_steps must be >= min_steps, got min_steps={min_steps}, max_steps={max_steps}."
        )

    if distribution == "fixed":
        if dwell.fixed_steps is None:
            raise ValueError("DwellTimeConfig with distribution='fixed' requires fixed_steps.")
        sampled = int(dwell.fixed_steps)
    elif distribution == "uniform":
        sampled = int(rng.integers(min_steps, max_steps + 1))
    elif distribution == "poisson":
        sampled = int(rng.poisson(max(float(dwell.mean_steps), 1.0)))
    elif distribution == "normal":
        sampled = int(round(rng.normal(loc=float(dwell.mean_steps), scale=float(dwell.std_steps))))
    else:
        raise ValueError(
            f"Unsupported dwell distribution {distribution!r}. Expected one of fixed, uniform, poisson, normal."
        )
    clipped = int(np.clip(sampled, min_steps, max_steps))
    return np.int16(clipped)


def _resolve_nucleation_state(dynamics: DynamicsConfig, graph: TransitionGraph) -> int | None:
    if dynamics.nucleation_state:
        if not graph.has_state(dynamics.nucleation_state):
            raise ValueError(
                f"dynamics.nucleation_state={dynamics.nucleation_state!r} is not present in the state graph."
            )
        return graph.index(dynamics.nucleation_state)
    if dynamics.mode == "coupled" and len(dynamics.primary_path) >= 2:
        return graph.index(dynamics.primary_path[1])
    return None


def _resolve_interface_state(graph: TransitionGraph) -> int | None:
    for state in graph.states:
        if state.template_kind == "interface":
            return graph.index(state.name)
    return None


def is_liquid_like_state_config(state_cfg) -> bool:
    return state_cfg.template_kind in {"liquid", "precursor"}


def is_interface_state_config(state_cfg) -> bool:
    return state_cfg.template_kind == "interface"


def is_solid_core_state_config(state_cfg) -> bool:
    return state_cfg.template_kind in {"crystal", "defective_crystal", "grain_boundary"}


def _neighbor_phase_context(
    neighbor_states: np.ndarray,
    neighbor_grains: np.ndarray,
    graph: TransitionGraph,
) -> NeighborPhaseContext:
    valid_states = [int(state_idx) for state_idx in neighbor_states.tolist() if int(state_idx) >= 0]
    if not valid_states:
        return NeighborPhaseContext(
            liquid_fraction=0.0,
            precursor_fraction=0.0,
            liquid_like_fraction=0.0,
            interface_fraction=0.0,
            crystal_fraction=0.0,
            grain_boundary_fraction=0.0,
            solid_fraction=0.0,
            distinct_crystal_grains=0,
        )
    liquid = 0
    precursor = 0
    liquid_like = 0
    interface = 0
    crystal = 0
    grain_boundary = 0
    solid = 0
    crystal_grains: set[int] = set()
    for local_idx, state_idx in enumerate(valid_states):
        state_cfg = graph.state_config(graph.name(state_idx))
        state_name = graph.name(state_idx)
        if state_name == "L":
            liquid += 1
        if state_name == "P":
            precursor += 1
        if is_liquid_like_state_config(state_cfg):
            liquid_like += 1
        if is_interface_state_config(state_cfg):
            interface += 1
        if state_name == "C" or state_cfg.template_kind == "crystal":
            crystal += 1
            grain_id = int(neighbor_grains[local_idx])
            if grain_id >= 0:
                crystal_grains.add(grain_id)
        if state_name == "G" or state_cfg.template_kind == "grain_boundary":
            grain_boundary += 1
        if state_cfg.template_kind in {"crystal", "defective_crystal", "grain_boundary"}:
            solid += 1
    denom = float(len(valid_states))
    return NeighborPhaseContext(
        liquid_fraction=liquid / denom,
        precursor_fraction=precursor / denom,
        liquid_like_fraction=liquid_like / denom,
        interface_fraction=interface / denom,
        crystal_fraction=crystal / denom,
        grain_boundary_fraction=grain_boundary / denom,
        solid_fraction=solid / denom,
        distinct_crystal_grains=len(crystal_grains),
    )


def _sample_next_state(
    current_state_idx: int,
    graph: TransitionGraph,
    neighbor_states: np.ndarray,
    neighbor_grains: np.ndarray,
    rng: np.random.Generator,
    dynamics: DynamicsConfig,
    frame_idx: int,
    is_seed: bool,
) -> int:
    current_state_name = graph.name(current_state_idx)
    if dynamics.mode == "independent":
        return _sample_independent_next_state(
            current_state_idx=current_state_idx,
            graph=graph,
            rng=rng,
        )
    if dynamics.mode != "coupled":
        raise ValueError(
            f"Unsupported dynamics.mode={dynamics.mode!r}. Expected 'independent' or 'coupled'."
        )
    transition_probabilities = _coupled_transition_probabilities(
        current_state_name=current_state_name,
        neighbor_states=neighbor_states,
        neighbor_grains=neighbor_grains,
        graph=graph,
        dynamics=dynamics,
        frame_idx=frame_idx,
        is_seed=is_seed,
    )
    return _sample_from_transition_probabilities(
        current_state_idx=current_state_idx,
        transition_probabilities=transition_probabilities,
        graph=graph,
        rng=rng,
    )


def _assign_initial_grain(
    state_name: str,
    states,
    is_seed: bool,
    next_grain_id: int,
    grain_orientations: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> tuple[int, int]:
    if not _state_allows_new_grain(state_name=state_name, states=states, is_seed=is_seed):
        return -1, next_grain_id
    grain_id = next_grain_id
    grain_orientations[grain_id] = rotation_matrix_to_quaternion(random_rotation_matrix(rng))
    return grain_id, next_grain_id + 1


def _update_grain_assignment(
    site_id: int,
    state_name: str,
    previous_grain: np.ndarray,
    neighbor_indices: np.ndarray,
    current_grain_id: int,
    states,
    is_seed: bool,
    next_grain_id: int,
    grain_orientations: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> tuple[int, int]:
    state_cfg = states[state_name]
    if not state_cfg.grain_bearing:
        return -1, next_grain_id
    if current_grain_id >= 0:
        return current_grain_id, next_grain_id
    neighbor_grains = [
        int(previous_grain[neighbor])
        for neighbor in neighbor_indices
        if neighbor >= 0 and previous_grain[neighbor] >= 0
    ]
    if neighbor_grains:
        values, counts = np.unique(np.asarray(neighbor_grains, dtype=np.int32), return_counts=True)
        return int(values[np.argmax(counts)]), next_grain_id
    if not _state_allows_new_grain(state_name=state_name, states=states, is_seed=is_seed):
        return -1, next_grain_id
    new_grain_id = next_grain_id
    grain_orientations[new_grain_id] = rotation_matrix_to_quaternion(random_rotation_matrix(rng))
    return new_grain_id, next_grain_id + 1


def _sample_independent_next_state(
    *,
    current_state_idx: int,
    graph: TransitionGraph,
    rng: np.random.Generator,
) -> int:
    current_state_name = graph.name(current_state_idx)
    edges = graph.outgoing_edges(current_state_name)
    if not edges:
        return current_state_idx
    transition_probability = min(0.20, 0.06 + 0.03 * len(edges))
    if rng.random() > transition_probability:
        return current_state_idx
    weights = np.asarray([edge.weight for edge in edges], dtype=np.float64)
    probabilities = weights / np.sum(weights)
    return graph.index(str(rng.choice([edge.target for edge in edges], p=probabilities)))


def _coupled_transition_probabilities(
    *,
    current_state_name: str,
    neighbor_states: np.ndarray,
    neighbor_grains: np.ndarray,
    graph: TransitionGraph,
    dynamics: DynamicsConfig,
    frame_idx: int,
    is_seed: bool,
) -> dict[str, float]:
    edge_weights = {edge.target: float(edge.weight) for edge in graph.outgoing_edges(current_state_name)}
    if not edge_weights:
        return {}
    context = _neighbor_phase_context(
        neighbor_states=neighbor_states,
        neighbor_grains=neighbor_grains,
        graph=graph,
    )
    probabilities: dict[str, float] = {}

    if current_state_name == "L":
        _add_transition_probability(
            probabilities=probabilities,
            target="P",
            probability=(
                0.015
                + 0.030 * context.precursor_fraction
                + 0.160 * context.interface_fraction
                + 0.120 * context.crystal_fraction
                + 0.060 * context.grain_boundary_fraction
            ),
            edge_weights=edge_weights,
        )
        return probabilities

    if current_state_name == "P":
        precursor_back_probability = float(
            np.clip(
                (
                    0.055
                    * (
                        1.0
                        - 0.15 * context.precursor_fraction
                        - 0.60 * context.interface_fraction
                        - 0.40 * context.crystal_fraction
                    )
                ),
                0.005,
                0.090,
            )
        )
        interface_probability = (
            0.004
            + 0.060 * context.precursor_fraction
            + 0.45 * context.interface_fraction
            + 0.40 * context.crystal_fraction
            + 0.12 * context.grain_boundary_fraction
        )
        if is_seed and frame_idx >= dynamics.nucleation_start_frame:
            interface_probability = max(interface_probability + 0.100, 0.120)
        if frame_idx < dynamics.nucleation_start_frame and not is_seed:
            interface_probability *= 0.05
        _add_transition_probability(
            probabilities=probabilities,
            target="L",
            probability=precursor_back_probability,
            edge_weights=edge_weights,
        )
        _add_transition_probability(
            probabilities=probabilities,
            target="I",
            probability=interface_probability,
            edge_weights=edge_weights,
        )
        return probabilities

    if current_state_name == "I":
        collision_threshold = max(2, int(dynamics.boundary_min_distinct_grains))
        grain_collision_probability = 0.0
        if context.distinct_crystal_grains >= collision_threshold:
            excess_grains = context.distinct_crystal_grains - collision_threshold
            grain_collision_probability = min(
                0.85,
                float(dynamics.boundary_state_probability) * (1.15 + 0.25 * excess_grains)
                + 0.22 * context.crystal_fraction,
            )
        crystal_probability = (
            0.10
            + 0.65 * context.interface_fraction
            + 0.35 * context.crystal_fraction
            + 0.12 * context.grain_boundary_fraction
        )
        if grain_collision_probability > 0.0:
            crystal_probability *= 0.35
        retreat_scale = float(
            np.clip(
                1.0 - 0.80 * context.interface_fraction - 0.55 * context.crystal_fraction,
                0.15,
                1.0,
            )
        )
        _add_transition_probability(
            probabilities=probabilities,
            target="G",
            probability=grain_collision_probability,
            edge_weights=edge_weights,
        )
        _add_transition_probability(
            probabilities=probabilities,
            target="C",
            probability=crystal_probability,
            edge_weights=edge_weights,
        )
        _add_transition_probability(
            probabilities=probabilities,
            target="P",
            probability=0.020 * retreat_scale,
            edge_weights=edge_weights,
        )
        _add_transition_probability(
            probabilities=probabilities,
            target="L",
            probability=0.006 * retreat_scale,
            edge_weights=edge_weights,
        )
        return probabilities

    if current_state_name == "C":
        if context.liquid_like_fraction > 0.55:
            _add_transition_probability(
                probabilities=probabilities,
                target="I",
                probability=0.006 * context.liquid_like_fraction,
                edge_weights=edge_weights,
            )
        return probabilities

    if current_state_name == "G":
        if context.distinct_crystal_grains <= 1 and context.crystal_fraction > 0.60:
            _add_transition_probability(
                probabilities=probabilities,
                target="C",
                probability=0.010 * context.crystal_fraction,
                edge_weights=edge_weights,
            )
        return probabilities

    if current_state_name == "D":
        _add_transition_probability(
            probabilities=probabilities,
            target="C",
            probability=0.040,
            edge_weights=edge_weights,
        )
        return probabilities

    return probabilities


def _add_transition_probability(
    *,
    probabilities: dict[str, float],
    target: str,
    probability: float,
    edge_weights: dict[str, float],
) -> None:
    if target not in edge_weights:
        return
    adjusted = float(probability) * float(edge_weights[target])
    if adjusted <= 0.0:
        return
    probabilities[target] = adjusted


def _sample_from_transition_probabilities(
    *,
    current_state_idx: int,
    transition_probabilities: dict[str, float],
    graph: TransitionGraph,
    rng: np.random.Generator,
) -> int:
    if not transition_probabilities:
        return current_state_idx
    targets = list(transition_probabilities.keys())
    probabilities = np.asarray([transition_probabilities[target] for target in targets], dtype=np.float64)
    total_probability = float(np.sum(probabilities))
    if total_probability <= 0.0:
        return current_state_idx
    if total_probability > 0.97:
        probabilities *= 0.97 / total_probability
        total_probability = 0.97
    stay_probability = max(0.0, 1.0 - total_probability)
    state_choices = np.asarray([current_state_idx] + [graph.index(target) for target in targets], dtype=np.int32)
    choice_probabilities = np.concatenate(([stay_probability], probabilities))
    choice_probabilities = choice_probabilities / np.sum(choice_probabilities)
    return int(rng.choice(state_choices, p=choice_probabilities))


def _sample_transitions_batch(
    *,
    previous_state: np.ndarray,
    previous_grain: np.ndarray,
    neigh: np.ndarray,
    neigh_valid: np.ndarray,
    n_valid_safe: np.ndarray,
    phase_seed_site_mask: np.ndarray,
    graph: TransitionGraph,
    dynamics: DynamicsConfig,
    edge_weight_matrix: np.ndarray,
    is_liquid_like: np.ndarray,
    is_interface: np.ndarray,
    is_solid_core: np.ndarray,
    state_indices: tuple[int, int, int, int, int, int],
    frame_idx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized coupled transition sampling for all sites in one frame."""
    _L, _P, _I, _C, _G, _D = state_indices
    n_sites = previous_state.shape[0]
    num_states = len(graph.state_names)

    # --- Compute neighbor phase fractions for all sites at once ---
    # Gather neighbor states: (n_sites, max_neighbors), -1 for invalid
    neigh_clamped = np.where(neigh_valid, neigh, 0)
    neigh_states = previous_state[neigh_clamped]  # (n_sites, max_neighbors)
    neigh_grains = previous_grain[neigh_clamped]  # (n_sites, max_neighbors)
    # Mask invalid entries
    neigh_states_masked = np.where(neigh_valid, neigh_states, -1)

    # Count per-state fractions using broadcasting
    # (n_sites, max_neighbors, 1) == (1, 1, num_states) → (n_sites, max_neighbors, num_states)
    # Then sum over neighbors → (n_sites, num_states)
    state_range = np.arange(num_states, dtype=np.int16)[None, None, :]
    state_counts = np.sum(
        neigh_valid[:, :, None] & (neigh_states_masked[:, :, None] == state_range),
        axis=1,
        dtype=np.float64,
    )  # (n_sites, num_states)

    # Compute fractions
    liquid_count = state_counts[:, _L] if _L >= 0 else np.zeros(n_sites, dtype=np.float64)
    precursor_count = state_counts[:, _P] if _P >= 0 else np.zeros(n_sites, dtype=np.float64)
    interface_count = state_counts[:, _I] if _I >= 0 else np.zeros(n_sites, dtype=np.float64)
    crystal_count = state_counts[:, _C] if _C >= 0 else np.zeros(n_sites, dtype=np.float64)
    gb_count = state_counts[:, _G] if _G >= 0 else np.zeros(n_sites, dtype=np.float64)

    liquid_frac = liquid_count / n_valid_safe if _L >= 0 else np.zeros(n_sites)
    precursor_frac = precursor_count / n_valid_safe if _P >= 0 else np.zeros(n_sites)
    interface_frac = interface_count / n_valid_safe if _I >= 0 else np.zeros(n_sites)
    crystal_frac = crystal_count / n_valid_safe if _C >= 0 else np.zeros(n_sites)
    gb_frac = gb_count / n_valid_safe if _G >= 0 else np.zeros(n_sites)
    solidish_count = interface_count + crystal_count + gb_count
    interface_core_count = interface_count + crystal_count

    # Liquid-like = liquid + precursor
    liquid_like_frac = np.zeros(n_sites, dtype=np.float64)
    for si in range(num_states):
        if is_liquid_like[si]:
            liquid_like_frac += state_counts[:, si] / n_valid_safe

    # Distinct crystal grains per site: count unique grain IDs among crystal/interface/GB neighbors
    # This is the hardest to vectorize — use a loop but over neighbors (small), not sites
    crystal_neighbor_mask = neigh_valid.copy()
    for si in range(num_states):
        if not is_solid_core[si] and not is_interface[si]:
            crystal_neighbor_mask &= (neigh_states_masked != si)
    # For distinct_crystal_grains, we need to count unique positive grain IDs per site
    # among solid neighbors. Use a per-site approach with sets — but avoid per-site Python loop.
    # Instead: for each site, collect the grain IDs of crystal neighbors.
    # Use a padded grain array and np.unique-style counting.
    crystal_neigh_grains = np.where(
        crystal_neighbor_mask & (neigh_grains >= 0),
        neigh_grains,
        -1,
    )  # (n_sites, max_neighbors), -1 for non-crystal/invalid
    # Count distinct positive grains per site
    # Fast approach: sort per row, count transitions
    distinct_crystal_grains = np.zeros(n_sites, dtype=np.int32)
    if crystal_neigh_grains.shape[1] > 0:
        sorted_grains = np.sort(crystal_neigh_grains, axis=1)
        # Count unique values > -1 per row
        is_positive = sorted_grains >= 0
        is_new = np.ones_like(is_positive)
        is_new[:, 1:] = sorted_grains[:, 1:] != sorted_grains[:, :-1]
        distinct_crystal_grains = np.sum(is_positive & is_new, axis=1).astype(np.int32)

    # --- Compute transition probability matrix: (n_sites, num_states) ---
    # trans_prob[i, j] = probability that site i transitions to state j
    trans_prob = np.zeros((n_sites, num_states), dtype=np.float64)

    # L sites → P
    if _L >= 0 and _P >= 0:
        mask_L = (previous_state == _L)
        if np.any(mask_L):
            p_LP = (
                0.010
                + 0.025 * precursor_frac
                + 0.060 * interface_frac
                + 0.050 * crystal_frac
                + 0.030 * gb_frac
            )
            front_capture_threshold = max(1, int(dynamics.liquid_to_precursor_front_threshold))
            front_capture_mask = interface_core_count >= float(front_capture_threshold)
            p_LP = np.where(
                front_capture_mask,
                np.maximum(p_LP, float(dynamics.liquid_to_precursor_front_probability)),
                p_LP,
            )
            p_LP = np.clip(p_LP, 0.0, 1.0)
            trans_prob[mask_L, _P] = p_LP[mask_L] * edge_weight_matrix[_L, _P]

    # P sites → L, I
    if _P >= 0:
        mask_P = (previous_state == _P)
        if np.any(mask_P):
            back_prob = np.clip(
                0.060 * (1.0 - 0.25 * precursor_frac - 0.80 * interface_frac - 0.55 * crystal_frac),
                0.0, 0.090,
            )
            iface_prob = (
                0.001
                + 0.015 * precursor_frac
                + 0.10 * interface_frac
                + 0.08 * crystal_frac
                + 0.04 * gb_frac
            )
            blob_threshold = max(1, int(dynamics.precursor_to_interface_blob_threshold))
            strong_interface_blob_mask = interface_core_count >= float(blob_threshold)
            iface_prob = np.where(
                strong_interface_blob_mask,
                np.maximum(iface_prob, float(dynamics.precursor_to_interface_blob_probability)),
                iface_prob,
            )
            # Seed boost
            seed_and_started = phase_seed_site_mask & (frame_idx >= dynamics.nucleation_start_frame)
            iface_prob = np.where(
                seed_and_started,
                np.maximum(iface_prob, float(dynamics.precursor_to_interface_blob_probability)),
                iface_prob,
            )
            # Pre-nucleation suppression
            pre_nucleation = ~phase_seed_site_mask & (frame_idx < dynamics.nucleation_start_frame)
            iface_prob = np.where(pre_nucleation, 0.0, iface_prob)
            back_prob = np.where(interface_core_count >= 1.0, 0.0, back_prob)
            iface_prob = np.clip(iface_prob, 0.0, 1.0)

            if _L >= 0:
                trans_prob[mask_P, _L] = back_prob[mask_P] * edge_weight_matrix[_P, _L]
            if _I >= 0:
                trans_prob[mask_P, _I] = iface_prob[mask_P] * edge_weight_matrix[_P, _I]

    # I sites → G, C, P, L
    if _I >= 0:
        mask_I = (previous_state == _I)
        if np.any(mask_I):
            collision_threshold = max(2, int(dynamics.boundary_min_distinct_grains))
            excess_grains = np.maximum(0, distinct_crystal_grains - collision_threshold)
            grain_collision_mask = distinct_crystal_grains >= collision_threshold
            grain_collision_prob = np.where(
                distinct_crystal_grains >= collision_threshold,
                np.minimum(
                    1.0,
                    float(dynamics.boundary_state_probability) * (1.15 + 0.25 * excess_grains)
                    + 0.22 * crystal_frac,
                ),
                0.0,
            )
            crystal_prob = 0.020 + 0.25 * interface_frac + 0.65 * crystal_frac + 0.10 * gb_frac
            crystal_blob_threshold = max(1, int(dynamics.interface_to_crystal_blob_threshold))
            compact_blob_mask = interface_core_count >= float(crystal_blob_threshold)
            crystal_prob = np.where(
                compact_blob_mask,
                np.maximum(crystal_prob, float(dynamics.interface_to_crystal_blob_probability)),
                crystal_prob,
            )
            crystal_prob = np.where(grain_collision_mask, 0.0, np.clip(crystal_prob, 0.0, 1.0))
            retreat_to_p = np.where(interface_core_count <= 1.0, 0.08 * liquid_like_frac, 0.0)
            retreat_to_l = np.where(interface_core_count <= 0.0, 0.020 * liquid_frac, 0.0)
            if _G >= 0:
                trans_prob[mask_I, _G] = grain_collision_prob[mask_I] * edge_weight_matrix[_I, _G]
            if _C >= 0:
                trans_prob[mask_I, _C] = crystal_prob[mask_I] * edge_weight_matrix[_I, _C]
            if _P >= 0:
                trans_prob[mask_I, _P] = retreat_to_p[mask_I] * edge_weight_matrix[_I, _P]
            if _L >= 0:
                trans_prob[mask_I, _L] = retreat_to_l[mask_I] * edge_weight_matrix[_I, _L]

    # C sites → I (rare)
    if _C >= 0 and _I >= 0:
        mask_C = (previous_state == _C)
        if np.any(mask_C):
            c_to_i = np.where(liquid_like_frac > 0.55, 0.006 * liquid_like_frac, 0.0)
            trans_prob[mask_C, _I] = c_to_i[mask_C] * edge_weight_matrix[_C, _I]

    # G sites → C (rare)
    if _G >= 0 and _C >= 0:
        mask_G = (previous_state == _G)
        if np.any(mask_G):
            g_to_c = np.where(
                (distinct_crystal_grains <= 1) & (crystal_frac > 0.60),
                0.010 * crystal_frac,
                0.0,
            )
            trans_prob[mask_G, _C] = g_to_c[mask_G] * edge_weight_matrix[_G, _C]

    # D sites → C
    if _D >= 0 and _C >= 0:
        mask_D = (previous_state == _D)
        if np.any(mask_D):
            trans_prob[mask_D, _C] = 0.040 * edge_weight_matrix[_D, _C]

    # --- Sample transitions for all sites at once ---
    # Clamp total probability and compute stay probability
    total_prob = np.sum(trans_prob, axis=1)  # (n_sites,)
    # Where total > 0.97, scale down
    scale_down = np.where(total_prob > 0.97, 0.97 / np.maximum(total_prob, 1e-12), 1.0)
    trans_prob *= scale_down[:, None]
    total_prob = np.sum(trans_prob, axis=1)
    stay_prob = np.maximum(0.0, 1.0 - total_prob)

    # Build full probability matrix: (n_sites, num_states + 1) where last column is "stay"
    # Actually, embed stay into the current state's column
    full_prob = trans_prob.copy()
    full_prob[np.arange(n_sites), previous_state] += stay_prob
    # Normalize
    row_sum = np.sum(full_prob, axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    full_prob /= row_sum

    # Sample: cumulative sum then searchsorted
    cum_prob = np.cumsum(full_prob, axis=1)
    draws = rng.random(n_sites)
    new_state = np.argmax(cum_prob >= draws[:, None], axis=1).astype(np.int16)

    return new_state


def _update_grains_batch(
    *,
    current_state: np.ndarray,
    previous_grain: np.ndarray,
    neigh: np.ndarray,
    neigh_valid: np.ndarray,
    phase_seed_site_mask: np.ndarray,
    is_grain_bearing: np.ndarray,
    allows_new_grain: np.ndarray,
    precursor_state_idx: int | None,
    next_grain_id: int,
    grain_orientations: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """Vectorized grain assignment for all sites after state transitions."""
    n_sites = current_state.shape[0]
    updated_grain = previous_grain.copy()

    # Sites that are not grain-bearing get -1
    not_bearing = ~is_grain_bearing[current_state]
    updated_grain[not_bearing] = -1

    # Sites that are grain-bearing AND already have a valid grain: keep it
    # Sites that need a grain (grain_bearing, current_grain < 0): inherit from neighbors or create new
    needs_grain = is_grain_bearing[current_state] & (previous_grain < 0)
    if not np.any(needs_grain):
        return updated_grain, next_grain_id

    needs_grain_indices = np.flatnonzero(needs_grain)

    for site_id in needs_grain_indices:
        # Try to inherit from neighbors
        valid = neigh_valid[site_id]
        if np.any(valid):
            neighbor_ids = neigh[site_id][valid]
            neighbor_grains_vals = previous_grain[neighbor_ids]
            positive = neighbor_grains_vals[neighbor_grains_vals >= 0]
            if positive.size > 0:
                values, counts = np.unique(positive, return_counts=True)
                updated_grain[site_id] = int(values[np.argmax(counts)])
                continue

        # Create new grain if allowed
        state_idx = int(current_state[site_id])
        is_seed = bool(phase_seed_site_mask[site_id])
        can_create = bool(allows_new_grain[state_idx]) or (
            precursor_state_idx is not None and state_idx == int(precursor_state_idx) and is_seed
        )
        if can_create:
            updated_grain[site_id] = next_grain_id
            grain_orientations[next_grain_id] = rotation_matrix_to_quaternion(
                random_rotation_matrix(rng)
            )
            next_grain_id += 1
        else:
            updated_grain[site_id] = -1

    return updated_grain, next_grain_id


def _state_allows_new_grain(
    *,
    state_name: str,
    states,
    is_seed: bool,
) -> bool:
    state_cfg = states[state_name]
    if not state_cfg.grain_bearing:
        return False
    if state_cfg.template_kind in {"interface", "crystal", "grain_boundary", "defective_crystal"}:
        return True
    if state_name == "P" and is_seed:
        return True
    return False


def _derive_segment_annotations(
    *,
    state_ids: np.ndarray,
    graph: TransitionGraph,
    metastable_min_dwell: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    num_frames, site_count = state_ids.shape
    dwell_total = np.zeros((num_frames, site_count), dtype=np.int16)
    dwell_remaining = np.zeros((num_frames, site_count), dtype=np.int16)
    segment_ids = np.zeros((num_frames, site_count), dtype=np.int32)
    metastable_mask = np.zeros((num_frames, site_count), dtype=bool)

    for site_id in range(site_count):
        segment_start = 0
        segment_id = 0
        for frame_idx in range(1, num_frames + 1):
            segment_ended = frame_idx == num_frames or int(state_ids[frame_idx, site_id]) != int(
                state_ids[frame_idx - 1, site_id]
            )
            if not segment_ended:
                continue
            segment_length = frame_idx - segment_start
            state_name = graph.name(int(state_ids[segment_start, site_id]))
            is_metastable = bool(
                graph.state_config(state_name).metastable and segment_length >= metastable_min_dwell
            )
            dwell_total[segment_start:frame_idx, site_id] = np.int16(segment_length)
            dwell_remaining[segment_start:frame_idx, site_id] = np.arange(
                segment_length,
                0,
                -1,
                dtype=np.int16,
            )
            segment_ids[segment_start:frame_idx, site_id] = np.int32(segment_id)
            metastable_mask[segment_start:frame_idx, site_id] = is_metastable
            segment_start = frame_idx
            segment_id += 1

    return dwell_total, dwell_remaining, segment_ids, metastable_mask


def _derive_transition_events(
    *,
    state_ids: np.ndarray,
    grain_ids: np.ndarray,
    graph: TransitionGraph,
) -> list[TransitionEvent]:
    num_frames, site_count = state_ids.shape
    events: list[TransitionEvent] = []
    for frame_idx in range(1, num_frames):
        changed_sites = np.flatnonzero(state_ids[frame_idx] != state_ids[frame_idx - 1])
        for site_id in changed_sites.tolist():
            events.append(
                TransitionEvent(
                    frame_index=frame_idx,
                    site_id=int(site_id),
                    source_state=graph.name(int(state_ids[frame_idx - 1, site_id])),
                    target_state=graph.name(int(state_ids[frame_idx, site_id])),
                    source_grain_id=int(grain_ids[frame_idx - 1, site_id]),
                    target_grain_id=int(grain_ids[frame_idx, site_id]),
                )
            )
    return events


def _initial_orientation(
    grain_id: int,
    grain_orientations: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    if grain_id >= 0:
        return grain_orientations[grain_id].copy()
    return rotation_matrix_to_quaternion(random_rotation_matrix(rng))


def _update_orientation(
    current_orientation: np.ndarray,
    grain_id: int,
    grain_orientations: dict[int, np.ndarray],
    nuisance: NuisanceConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    axis = random_unit_vector(rng)
    drift_quaternion = axis_angle_to_quaternion(axis, math.radians(nuisance.orientation_drift_deg) * rng.normal())
    updated = quaternion_multiply(drift_quaternion, current_orientation)
    if grain_id >= 0:
        if grain_id not in grain_orientations:
            raise KeyError(f"Missing base orientation for grain_id={grain_id}.")
        updated = quaternion_slerp(updated, grain_orientations[grain_id], nuisance.orientation_alignment)
    return updated.astype(np.float32)


def _update_ar1_vector(
    current: np.ndarray,
    relaxation: float,
    noise_scale: float,
    mean_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = rng.normal(scale=noise_scale, size=current.shape)
    updated = relaxation * current + (1.0 - relaxation) * mean_scale + noise
    return updated.astype(np.float32)


def _update_ar1_scalar(
    current: float,
    relaxation: float,
    noise_scale: float,
    mean: float,
    rng: np.random.Generator,
    lower: float,
) -> np.float32:
    updated = relaxation * current + (1.0 - relaxation) * mean + float(rng.normal(scale=noise_scale))
    return np.float32(max(lower, updated))


def _update_crystal_variant_id(
    *,
    current_variant_id: int,
    state_name: str,
    dynamics: DynamicsConfig,
    rng: np.random.Generator,
) -> np.int8:
    if state_name != "C":
        return np.int8(0)
    if current_variant_id == 1:
        if rng.random() <= dynamics.crystal_variant_persistence:
            return np.int8(1)
        return np.int8(0)
    if rng.random() <= dynamics.crystal_variant_probability:
        return np.int8(1)
    return np.int8(0)


def _update_crystal_variant_ids_batch(
    current_variant_ids: np.ndarray,
    current_states: np.ndarray,
    graph: TransitionGraph,
    dynamics: DynamicsConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized crystal variant update for all sites."""
    n = len(current_variant_ids)
    result = np.zeros(n, dtype=np.int8)
    crystal_idx = graph.index("C") if graph.has_state("C") else -1
    if crystal_idx < 0:
        return result
    is_crystal = current_states == crystal_idx
    if not np.any(is_crystal):
        return result
    draws = rng.random(n)
    # Sites in C with variant==1: persist with crystal_variant_persistence
    persist_mask = is_crystal & (current_variant_ids == 1)
    result[persist_mask] = np.where(
        draws[persist_mask] <= dynamics.crystal_variant_persistence,
        np.int8(1),
        np.int8(0),
    )
    # Sites in C with variant!=1: switch with crystal_variant_probability
    switch_mask = is_crystal & (current_variant_ids != 1)
    result[switch_mask] = np.where(
        draws[switch_mask] <= dynamics.crystal_variant_probability,
        np.int8(1),
        np.int8(0),
    )
    return result


def _update_orientations_batch(
    current_orientations: np.ndarray,
    grain_ids: np.ndarray,
    grain_orientations: dict[int, np.ndarray],
    nuisance: NuisanceConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized orientation update for all sites."""
    n = current_orientations.shape[0]
    # Generate random drift axes and apply small-angle rotations
    axes = rng.normal(size=(n, 3)).astype(np.float32)
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    axes = axes / norms

    angles = math.radians(nuisance.orientation_drift_deg) * rng.normal(size=n)
    half_angles = 0.5 * angles
    sin_half = np.sin(half_angles).astype(np.float32)
    cos_half = np.cos(half_angles).astype(np.float32)
    drift_quats = np.column_stack([cos_half, axes * sin_half[:, None]])
    drift_norms = np.linalg.norm(drift_quats, axis=1, keepdims=True)
    drift_norms = np.where(drift_norms > 1e-12, drift_norms, 1.0)
    drift_quats = (drift_quats / drift_norms).astype(np.float32)

    # Batch quaternion multiply: drift * current
    w0, x0, y0, z0 = drift_quats[:, 0], drift_quats[:, 1], drift_quats[:, 2], drift_quats[:, 3]
    w1, x1, y1, z1 = (
        current_orientations[:, 0],
        current_orientations[:, 1],
        current_orientations[:, 2],
        current_orientations[:, 3],
    )
    updated = np.column_stack([
        w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
        w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
        w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
        w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ]).astype(np.float32)
    u_norms = np.linalg.norm(updated, axis=1, keepdims=True)
    u_norms = np.where(u_norms > 1e-12, u_norms, 1.0)
    updated = updated / u_norms

    # Slerp toward grain orientation for sites with valid grain_id
    has_grain = grain_ids >= 0
    if np.any(has_grain):
        grain_mask_indices = np.flatnonzero(has_grain)
        target_quats = np.empty((grain_mask_indices.shape[0], 4), dtype=np.float32)
        for i, site_idx in enumerate(grain_mask_indices):
            gid = int(grain_ids[site_idx])
            target_quats[i] = grain_orientations[gid]
        slerped = quaternion_slerp_batch(
            updated[grain_mask_indices],
            target_quats,
            nuisance.orientation_alignment,
        )
        updated[grain_mask_indices] = slerped

    return updated.astype(np.float32)


def _update_ar1_vectors_batch(
    current: np.ndarray,
    relaxation: float,
    noise_scale: float,
    mean_scales: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Vectorized AR(1) vector update for all sites. mean_scales is (N,)."""
    noise = rng.normal(scale=noise_scale, size=current.shape).astype(np.float32)
    updated = relaxation * current + (1.0 - relaxation) * mean_scales[:, None] + noise
    return updated.astype(np.float32)


def _update_ar1_scalars_batch(
    current: np.ndarray,
    relaxation: float,
    noise_scale: float,
    means: np.ndarray,
    rng: np.random.Generator,
    lower: float,
) -> np.ndarray:
    """Vectorized AR(1) scalar update for all sites. means is (N,)."""
    noise = rng.normal(scale=noise_scale, size=current.shape).astype(np.float32)
    updated = relaxation * current + (1.0 - relaxation) * means + noise
    return np.maximum(lower, updated).astype(np.float32)
