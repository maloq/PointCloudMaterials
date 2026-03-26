from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
try:
    from numba import njit
    _NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    _NUMBA_AVAILABLE = False

from ..atomistic_generator import LiquidMetalGenerator, LiquidStructureConfig
from .config import TemporalBenchmarkConfig
from .dynamics import LatentTrajectories, SiteLayout
from .geometry import quaternion_to_rotation_matrix, quaternion_to_rotation_matrix_batch, random_rotation_matrices, random_unit_vector
from .graph import TransitionGraph
from .templates import TemplateLibrary


@dataclass(frozen=True)
class RenderedFrame:
    frame_index: int
    atoms: np.ndarray
    site_ids: np.ndarray
    state_ids: np.ndarray
    grain_ids: np.ndarray
    local_atom_ids: np.ndarray
    local_points: np.ndarray
    metadata: dict[str, Any]


@dataclass(frozen=True)
class InterfaceContext:
    normal: np.ndarray
    width: float
    plane_position: float
    plane_jitter: float
    solid_fraction: float


@dataclass(frozen=True)
class BoundaryContext:
    normal: np.ndarray
    width: float
    plane_jitter: float
    grain_a: int
    grain_b: int
    rotation_a: np.ndarray
    rotation_b: np.ndarray


_TEMPLATE_KIND_CODES = {
    "liquid": 0,
    "precursor": 1,
    "interface": 2,
    "crystal": 3,
    "grain_boundary": 4,
    "defective_crystal": 5,
}
_NUMBA_CELL_LIST_WARMED = False


class FrameRenderer:
    """
    Dense temporal frame renderer.

    The renderer keeps one persistent, space-filling atom cloud for the whole
    box and evolves it with bounded local displacements. State-dependent local
    targets then pull subsets of atoms toward liquid, precursor, interface,
    crystal, or grain-boundary structure without changing atom count.
    """

    def __init__(
        self,
        config: TemporalBenchmarkConfig,
        graph: TransitionGraph,
        templates: TemplateLibrary,
        layout: SiteLayout,
        latent: LatentTrajectories,
    ) -> None:
        self.config = config
        self.graph = graph
        self.templates = templates
        self.layout = layout
        self.latent = latent

        self.box_size = float(config.domain.box_size)
        self.avg_nn_distance = float(config.domain.avg_nn_distance)
        self.neighborhood_radius = float(config.domain.neighborhood_radius)
        self.points_per_site = int(config.domain.atoms_per_site)
        self.time_delta = float(config.time.delta_t)
        self.fast_mode = bool(config.rendering.fast_mode)
        self.track_site_count = int(layout.site_count)
        self.phase_site_count = int(layout.phase_site_count)
        self.track_centers = self.layout.centers.astype(np.float32, copy=False)
        self.phase_centers = self.layout.phase_centers.astype(np.float32, copy=False)
        self.state_names = list(graph.state_names)
        self._state_count = len(self.state_names)
        self._renderer_rng = np.random.default_rng(int(config.seed) + 17_301)
        self._frame_cursor = -1
        self.parallel_workers = int(config.rendering.parallel_workers)
        if self.parallel_workers <= 0:
            raise ValueError(
                f"rendering.parallel_workers must be positive, got {config.rendering.parallel_workers}."
            )
        self.site_assignment_candidate_count = int(config.rendering.site_assignment_candidate_count)
        if self.site_assignment_candidate_count <= 0:
            raise ValueError(
                "rendering.site_assignment_candidate_count must be positive, "
                f"got {config.rendering.site_assignment_candidate_count}."
            )
        estimated_atom_count = int(round(float(config.rendering.target_density) * (self.box_size**3)))
        self._kdtree_workers = min(self.parallel_workers, _recommended_kdtree_workers(estimated_atom_count))
        self._site_update_workers = min(
            self.parallel_workers,
            self.phase_site_count,
        )
        self._assignment_chunk_size = _recommended_assignment_chunk_size(
            estimated_atom_count=estimated_atom_count,
            query_k=min(self.phase_site_count, self.site_assignment_candidate_count),
        )

        self._site_spacing = self._estimate_site_spacing()
        self._field_rotations, self._field_axis_scales, self._field_bias = self._build_site_field()
        self._field_rotations_scaled = self._field_rotations / self._field_axis_scales[:, None, :]
        self._site_tree = cKDTree(self.phase_centers)
        self._surface_phases = self._renderer_rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(self.phase_site_count, 3),
        ).astype(np.float32)
        self._surface_scales = self._renderer_rng.uniform(
            0.55 * self._site_spacing,
            1.20 * self._site_spacing,
            size=(self.phase_site_count, 2),
        ).astype(np.float32)
        self._ownership_warp_modes = self._build_static_warp_modes(n_modes=4, amplitude_scale=0.16)
        self._flow_modes = self._build_dynamic_warp_modes(n_modes=5, amplitude_scale=0.09)
        self._phase_dwell_total, self._phase_dwell_remaining = self._build_phase_progress_cache()

        self.base_atoms = self._build_base_atom_cloud().astype(np.float32)
        if self.base_atoms.ndim != 2 or self.base_atoms.shape[1] != 3:
            raise ValueError(
                "Persistent base atom cloud must have shape (n_atoms, 3), "
                f"got {self.base_atoms.shape}."
            )
        self.current_atoms = self.base_atoms.copy()
        self._current_site_ids = self._assign_sites(self.current_atoms)
        self._current_site_groups = self._group_atom_indices_by_site(self._current_site_ids)
        self._last_assignment_atoms = self.current_atoms.copy()
        self._atom_reference_directions = self._build_atom_reference_directions(len(self.base_atoms))
        self._hardcore_distance = self._resolve_hardcore_distance()
        self._relaxation_cell_size = float(max(self._hardcore_distance, 2.0 * self._hardcore_distance))
        relaxation_cells_per_axis = max(1, int(np.ceil(self.box_size / self._relaxation_cell_size)))
        self._relaxation_grid_shape = np.full(3, relaxation_cells_per_axis, dtype=np.int32)
        self._relaxation_grid_size = int(relaxation_cells_per_axis**3)
        self._relaxation_grid_strides = np.asarray(
            [
                1,
                relaxation_cells_per_axis,
                relaxation_cells_per_axis * relaxation_cells_per_axis,
            ],
            dtype=np.int64,
        )
        relaxation_neighbor_deltas: list[tuple[int, int, int]] = []
        relaxation_neighbor_flat_offsets: list[int] = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    if dz < 0 or (dz == 0 and dy < 0) or (dz == 0 and dy == 0 and dx <= 0):
                        continue
                    relaxation_neighbor_deltas.append((dx, dy, dz))
                    relaxation_neighbor_flat_offsets.append(
                        int(
                            dx * self._relaxation_grid_strides[0]
                            + dy * self._relaxation_grid_strides[1]
                            + dz * self._relaxation_grid_strides[2]
                        )
                    )
        self._relaxation_neighbor_deltas = np.asarray(relaxation_neighbor_deltas, dtype=np.int32)
        self._relaxation_neighbor_flat_offsets = np.asarray(relaxation_neighbor_flat_offsets, dtype=np.int64)
        # Cached atom KDTree shared between _relax_close_contacts and _extract_local_neighborhoods
        self._cached_atom_tree: cKDTree | None = None
        if _NUMBA_AVAILABLE and not self.fast_mode:
            _warm_cell_list_pair_kernels()

        # Cache base lattice vectors and motif for each recipe to avoid repeated dict lookups
        self._recipe_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._state_phase_recipes = [self.templates.phase_recipe(state_name) for state_name in self.state_names]
        self._primary_crystal_state_idx = next(
            (
                idx
                for idx, state in enumerate(self.graph.states)
                if state.template_kind == "crystal"
            ),
            0,
        )
        self._template_kind_code_by_state_idx = np.asarray(
            [
                _TEMPLATE_KIND_CODES.get(state.template_kind, -1)
                for state in self.graph.states
            ],
            dtype=np.int8,
        )
        if np.any(self._template_kind_code_by_state_idx < 0):
            bad_indices = np.flatnonzero(self._template_kind_code_by_state_idx < 0)
            bad_kinds = [self.graph.states[int(idx)].template_kind for idx in bad_indices]
            raise ValueError(
                "Encountered unsupported template kinds while initializing FrameRenderer. "
                f"indices={bad_indices.tolist()}, kinds={bad_kinds}."
            )
        self._state_kind_is_liquid = self._template_kind_code_by_state_idx == _TEMPLATE_KIND_CODES["liquid"]
        self._state_kind_is_precursor = self._template_kind_code_by_state_idx == _TEMPLATE_KIND_CODES["precursor"]
        self._state_kind_is_interface = self._template_kind_code_by_state_idx == _TEMPLATE_KIND_CODES["interface"]
        self._state_kind_is_crystalish = np.isin(
            self._template_kind_code_by_state_idx,
            np.asarray(
                [
                    _TEMPLATE_KIND_CODES["crystal"],
                    _TEMPLATE_KIND_CODES["defective_crystal"],
                ],
                dtype=np.int8,
            ),
        )
        self._state_kind_is_grain_boundary = (
            self._template_kind_code_by_state_idx == _TEMPLATE_KIND_CODES["grain_boundary"]
        )
        self._state_kind_is_liquidish = self._state_kind_is_liquid | self._state_kind_is_precursor
        self._state_kind_is_solidish = (
            self._state_kind_is_interface
            | self._state_kind_is_crystalish
            | self._state_kind_is_grain_boundary
        )
        self._interface_width_by_state_idx = np.zeros(self._state_count, dtype=np.float32)
        self._interface_plane_jitter_by_state_idx = np.zeros(self._state_count, dtype=np.float32)
        self._interface_solid_fraction_start_by_state_idx = np.zeros(self._state_count, dtype=np.float32)
        self._interface_solid_fraction_end_by_state_idx = np.zeros(self._state_count, dtype=np.float32)
        self._recipe_variant0_by_state_idx: list[dict[str, Any]] = [{} for _ in range(self._state_count)]
        self._recipe_variant1_by_state_idx: list[dict[str, Any] | None] = [None for _ in range(self._state_count)]
        for state_idx, state in enumerate(self.graph.states):
            state_recipe = self._state_phase_recipes[state_idx]
            template_kind = state.template_kind
            if template_kind in {"crystal", "defective_crystal"}:
                self._recipe_variant0_by_state_idx[state_idx] = state_recipe
                if template_kind == "crystal":
                    self._recipe_variant1_by_state_idx[state_idx] = self.templates.crystal_variant_recipe(
                        state.name,
                        1,
                    )
            elif template_kind == "precursor":
                self._recipe_variant0_by_state_idx[state_idx] = self._state_phase_recipes[self._primary_crystal_state_idx]
            elif template_kind == "interface":
                solid_recipe = state_recipe.get("solid_recipe")
                if solid_recipe is None:
                    raise KeyError(
                        f"Interface state {state.name!r} is missing solid_recipe in its template."
                    )
                self._recipe_variant0_by_state_idx[state_idx] = solid_recipe
                self._interface_width_by_state_idx[state_idx] = float(state_recipe["interface_width"])
                self._interface_plane_jitter_by_state_idx[state_idx] = float(state_recipe.get("plane_jitter", 0.0))
                self._interface_solid_fraction_start_by_state_idx[state_idx] = float(
                    state_recipe["solid_fraction_start"]
                )
                self._interface_solid_fraction_end_by_state_idx[state_idx] = float(
                    state_recipe["solid_fraction_end"]
                )
            elif template_kind == "grain_boundary":
                solid_recipe = state_recipe.get("base_solid_recipe")
                if solid_recipe is None:
                    raise KeyError(
                        f"Grain-boundary state {state.name!r} is missing base_solid_recipe in its template."
                    )
                self._recipe_variant0_by_state_idx[state_idx] = solid_recipe
            else:
                self._recipe_variant0_by_state_idx[state_idx] = self._state_phase_recipes[self._primary_crystal_state_idx]

        # Cache for _assign_sites: skip full reassignment when atoms haven't moved much.
        self._assignment_skip_threshold = float(0.45 * self._site_spacing)
        self._assignment_skip_threshold_sq = self._assignment_skip_threshold**2
        self._assignment_check_stride = max(1, len(self.base_atoms) // 32_768)
        self._second_pass_overlap_threshold = float(0.045 * self.avg_nn_distance)
        self._second_pass_pair_threshold = max(2_048, len(self.base_atoms) // 180)
        self._vectorized_lattice_chunk_atoms = max(65_536, min(524_288, estimated_atom_count))

        # Cache for _warp_points_static: the base warp (ownership modes) is position-dependent
        # but does not change between frames for the same atom positions.  Pre-compute the
        # wavevector-position product once for base_atoms and reuse across frames.
        self._base_warp_phase = (
            (2.0 * np.pi / self.box_size)
            * (self.base_atoms @ self._ownership_warp_modes["wavevectors"].T)
            + self._ownership_warp_modes["phases"][None, :]
        ).astype(np.float32)
        self._base_warp_sin = np.sin(self._base_warp_phase).astype(np.float32)

        # Cache for _flow_displacement: pre-compute base_atoms @ wavevectors.T
        self._flow_base_phase_offset = (
            (2.0 * np.pi / self.box_size)
            * (self.base_atoms @ self._flow_modes["wavevectors"].T)
            + self._flow_modes["phases"][None, :]
        ).astype(np.float32)
        self._fast_mode_local_points: np.ndarray | None = None
        self._fast_mode_local_atom_ids: np.ndarray | None = None
        if self.fast_mode:
            self._cached_atom_tree = cKDTree(self.current_atoms, balanced_tree=False)
            self._fast_mode_local_points, self._fast_mode_local_atom_ids = self._extract_local_neighborhoods(
                self.current_atoms
            )

    def close(self) -> None:
        pass

    def render_frame(self, frame_index: int) -> RenderedFrame:
        if frame_index < 0 or frame_index >= int(self.config.time.num_frames):
            raise IndexError(
                f"frame_index must be in [0, {self.config.time.num_frames}), got {frame_index}."
            )
        if frame_index != self._frame_cursor + 1:
            raise RuntimeError(
                "FrameRenderer requires strictly sequential rendering because atoms evolve persistently. "
                f"Expected frame_index={self._frame_cursor + 1}, got frame_index={frame_index}."
            )

        if self.fast_mode:
            site_ids = self._current_site_ids
            displacements = np.zeros(self.current_atoms.shape[0], dtype=np.float32)
            self._frame_cursor = frame_index
            local_points = self._fast_mode_local_points
            local_atom_ids = self._fast_mode_local_atom_ids
            if local_points is None or local_atom_ids is None:
                raise RuntimeError("Fast-mode renderer expected cached neighborhoods, but they were not initialized.")
        else:
            site_groups = self._current_site_groups
            base_flow_displacement = self._flow_displacement(self.base_atoms, frame_index)
            proposed_atoms = self._evolve_atoms(
                frame_index=frame_index,
                site_groups=site_groups,
                base_flow_displacement=base_flow_displacement,
            )
            proposed_atoms = np.clip(proposed_atoms, 0.0, self.box_size).astype(np.float32)
            proposed_atoms = self._relax_close_contacts(proposed_atoms)

            displacements = np.linalg.norm(proposed_atoms - self.current_atoms, axis=1)
            self.current_atoms = proposed_atoms
            self._frame_cursor = frame_index

            if self._last_assignment_atoms is not None:
                sample_slice = slice(None, None, self._assignment_check_stride)
                sampled_delta = self.current_atoms[sample_slice] - self._last_assignment_atoms[sample_slice]
                sampled_dist_sq = np.einsum("ij,ij->i", sampled_delta, sampled_delta, optimize=True)
                sampled_max_sq = float(np.max(sampled_dist_sq)) if sampled_dist_sq.size > 0 else 0.0
                if sampled_max_sq < 0.36 * self._assignment_skip_threshold_sq:
                    should_reassign = False
                else:
                    full_delta = self.current_atoms - self._last_assignment_atoms
                    full_dist_sq = np.einsum("ij,ij->i", full_delta, full_delta, optimize=True)
                    should_reassign = bool(np.max(full_dist_sq) >= self._assignment_skip_threshold_sq)
            else:
                should_reassign = True

            if not should_reassign:
                site_ids = self._current_site_ids
            else:
                site_ids = self._assign_sites(self.current_atoms)
                self._current_site_ids = site_ids
                self._current_site_groups = self._group_atom_indices_by_site(site_ids)
                self._last_assignment_atoms = self.current_atoms.copy()
        state_ids = self.latent.phase_state_ids[frame_index, site_ids].astype(np.int16)
        grain_ids = self.latent.phase_grain_ids[frame_index, site_ids].astype(np.int32)
        if not self.fast_mode:
            local_points, local_atom_ids = self._extract_local_neighborhoods(self.current_atoms)

        metadata = self._build_frame_metadata(
            frame_index=frame_index,
            site_ids=site_ids,
            state_ids=state_ids,
            grain_ids=grain_ids,
            displacements=displacements,
        )
        return RenderedFrame(
            frame_index=frame_index,
            atoms=self.current_atoms,
            site_ids=site_ids,
            state_ids=state_ids,
            grain_ids=grain_ids,
            local_atom_ids=local_atom_ids,
            local_points=local_points,
            metadata=metadata,
        )

    def _build_base_atom_cloud(self) -> np.ndarray:
        liquid_state = next(
            (state for state in self.graph.states if state.template_kind == "liquid"),
            None,
        )
        fallback_state = liquid_state or self.graph.states[0]
        phase_recipe = self.templates.phase_recipe(fallback_state.name)
        liquid_config = LiquidStructureConfig(**phase_recipe.get("liquid_config", {"method": "simple"}))
        min_pair_distance = phase_recipe.get("min_pair_dist")
        generator = LiquidMetalGenerator(
            self.box_size,
            float(self.config.rendering.target_density),
            self.avg_nn_distance,
            config=liquid_config,
            rng=np.random.default_rng(int(self.config.seed) + 4_211),
            min_pair_dist=min_pair_distance,
        )
        atoms = generator.generate().astype(np.float32)
        expected_atoms = int(round(float(self.config.rendering.target_density) * (self.box_size**3)))
        if atoms.shape[0] < int(0.95 * expected_atoms):
            raise RuntimeError(
                "Persistent full-box atom generation underfilled the simulation box. "
                f"Expected about {expected_atoms} atoms but generated {atoms.shape[0]}. "
                f"liquid_method={liquid_config.method!r}, box_size={self.box_size}, "
                f"target_density={self.config.rendering.target_density}."
            )
        return atoms

    def _build_atom_reference_directions(self, atom_count: int) -> np.ndarray:
        directions = self._renderer_rng.normal(size=(atom_count, 3)).astype(np.float32)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.where(norms > 1e-8, norms, 1.0)
        return directions / norms

    def _resolve_hardcore_distance(self) -> float:
        explicit_values = []
        for state in self.graph.states:
            min_pair = state.template_params.get("min_pair_distance")
            if min_pair is not None:
                explicit_values.append(float(min_pair))
        if explicit_values:
            return float(np.min(explicit_values))
        return 0.75 * self.avg_nn_distance

    def _estimate_site_spacing(self) -> float:
        if self.phase_site_count <= 1:
            return 0.5 * self.box_size
        tree = cKDTree(self.phase_centers)
        distances, _ = tree.query(self.phase_centers, k=min(4, self.phase_site_count))
        if distances.ndim == 1:
            return 0.5 * self.box_size
        neighbor_distances = distances[:, 1:]
        return float(np.mean(neighbor_distances))

    def _build_site_field(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rotations = random_rotation_matrices(self._renderer_rng, self.phase_site_count).astype(np.float32)
        if self.phase_site_count == 1:
            local_scale = np.full(1, 0.45 * self.box_size, dtype=np.float32)
        else:
            tree = cKDTree(self.phase_centers)
            distances, _ = tree.query(self.phase_centers, k=min(5, self.phase_site_count))
            neighbor_distances = distances[:, 1:]
            local_scale = 0.58 * np.mean(neighbor_distances, axis=1)
        axis_scales = (
            local_scale[:, None]
            * self._renderer_rng.uniform(0.86, 1.18, size=(self.phase_site_count, 3))
        )
        axis_scales = axis_scales.astype(np.float32)
        bias = self._renderer_rng.normal(loc=0.0, scale=0.08, size=self.phase_site_count).astype(np.float32)
        return rotations, axis_scales, bias

    def _build_static_warp_modes(self, *, n_modes: int, amplitude_scale: float) -> dict[str, np.ndarray]:
        directions = np.vstack([random_unit_vector(self._renderer_rng) for _ in range(n_modes)]).astype(np.float32)
        wavevectors = self._renderer_rng.integers(1, 4, size=(n_modes, 3)).astype(np.float32)
        phases = self._renderer_rng.uniform(0.0, 2.0 * np.pi, size=n_modes).astype(np.float32)
        amplitudes = (
            self._renderer_rng.uniform(0.50, 1.0, size=n_modes).astype(np.float32)
            * amplitude_scale
            * self._site_spacing
        )
        return {
            "directions": directions,
            "wavevectors": wavevectors,
            "phases": phases,
            "amplitudes": amplitudes,
        }

    def _build_dynamic_warp_modes(self, *, n_modes: int, amplitude_scale: float) -> dict[str, np.ndarray]:
        directions = np.vstack([random_unit_vector(self._renderer_rng) for _ in range(n_modes)]).astype(np.float32)
        wavevectors = self._renderer_rng.integers(1, 4, size=(n_modes, 3)).astype(np.float32)
        phases = self._renderer_rng.uniform(0.0, 2.0 * np.pi, size=n_modes).astype(np.float32)
        angular_speeds = self._renderer_rng.uniform(0.06, 0.18, size=n_modes).astype(np.float32)
        amplitudes = (
            self._renderer_rng.uniform(0.40, 1.0, size=n_modes).astype(np.float32)
            * amplitude_scale
            * self.avg_nn_distance
        )
        return {
            "directions": directions,
            "wavevectors": wavevectors,
            "phases": phases,
            "angular_speeds": angular_speeds,
            "amplitudes": amplitudes,
        }

    def _warp_points_static(self, points: np.ndarray) -> np.ndarray:
        phase = (2.0 * np.pi / self.box_size) * (points @ self._ownership_warp_modes["wavevectors"].T)
        phase += self._ownership_warp_modes["phases"][None, :]
        sin_phase = np.sin(phase)
        # Vectorized: (n_atoms, n_modes) * (n_modes,) -> (n_atoms, n_modes), then @ (n_modes, 3)
        weighted_sin = sin_phase * self._ownership_warp_modes["amplitudes"][None, :]
        warp_offset = weighted_sin @ self._ownership_warp_modes["directions"]
        return (points + warp_offset).astype(np.float32)

    def _flow_displacement(self, points: np.ndarray, frame_index: int) -> np.ndarray:
        time_value = float(frame_index) * self.time_delta
        # Use pre-computed base phase offset (positions @ wavevectors + static phases)
        phase = self._flow_base_phase_offset + time_value * self._flow_modes["angular_speeds"][None, :]
        sin_phase = np.sin(phase)
        # Vectorized: (n_atoms, n_modes) * (n_modes,) -> (n_atoms, n_modes), then @ (n_modes, 3)
        weighted_sin = sin_phase * self._flow_modes["amplitudes"][None, :]
        displacement = weighted_sin @ self._flow_modes["directions"]
        return displacement.astype(np.float32)

    def _assign_sites(self, points: np.ndarray) -> np.ndarray:
        chunk_size = self._assignment_chunk_size
        site_ids = np.empty(points.shape[0], dtype=np.int32)
        warped_points = self._warp_points_static(points)
        query_k = min(self.phase_site_count, self.site_assignment_candidate_count)
        for start in range(0, points.shape[0], chunk_size):
            stop = min(points.shape[0], start + chunk_size)
            chunk = warped_points[start:stop]
            _, candidate_ids = self._site_tree.query(
                chunk,
                k=query_k,
                workers=self._kdtree_workers,
            )
            if query_k == 1:
                candidate_ids = candidate_ids[:, None]
            candidate_ids = np.asarray(candidate_ids, dtype=np.int32)
            candidate_centers = self.phase_centers[candidate_ids]
            diff = chunk[:, None, :] - candidate_centers
            candidate_rotations_scaled = self._field_rotations_scaled[candidate_ids]
            local_scaled = np.einsum("nki,nkij->nkj", diff, candidate_rotations_scaled, optimize=True)
            scores = np.sum(local_scaled * local_scaled, axis=2) + self._field_bias[candidate_ids]
            best_candidate = np.argmin(scores, axis=1)
            site_ids[start:stop] = candidate_ids[np.arange(stop - start), best_candidate]
        return site_ids

    def _group_atom_indices_by_site(self, site_ids: np.ndarray) -> list[tuple[int, np.ndarray]]:
        order = np.argsort(site_ids, kind="stable")
        sorted_site_ids = site_ids[order]
        unique_site_ids, start_indices, counts = np.unique(
            sorted_site_ids,
            return_index=True,
            return_counts=True,
        )
        groups: list[tuple[int, np.ndarray]] = []
        for site_id, start_idx, count in zip(unique_site_ids.tolist(), start_indices.tolist(), counts.tolist()):
            groups.append((int(site_id), order[start_idx : start_idx + count]))
        return groups

    def _evolve_atoms(
        self,
        *,
        frame_index: int,
        site_groups: list[tuple[int, np.ndarray]],
        base_flow_displacement: np.ndarray,
    ) -> np.ndarray:
        proposed = self.current_atoms.copy()
        if not site_groups:
            return proposed
        site_ids = np.asarray([site_id for site_id, _ in site_groups], dtype=np.int32)
        state_ids = self.latent.phase_state_ids[frame_index, site_ids]
        # Partition sites by template_kind for batch processing.
        liquid_mask = self._state_kind_is_liquid[state_ids]
        crystal_mask = self._state_kind_is_crystalish[state_ids]
        other_mask = ~(liquid_mask | crystal_mask)
        liquid_sites = [site_groups[idx] for idx in np.flatnonzero(liquid_mask)]
        crystal_sites = [site_groups[idx] for idx in np.flatnonzero(crystal_mask)]
        other_sites = [site_groups[idx] for idx in np.flatnonzero(other_mask)]

        # --- Batch liquid sites ---
        if liquid_sites:
            self._evolve_liquid_batch(
                frame_index=frame_index,
                liquid_sites=liquid_sites,
                base_flow_displacement=base_flow_displacement,
                proposed=proposed,
            )

        # --- Batch crystal sites ---
        if crystal_sites:
            self._evolve_crystal_batch(
                frame_index=frame_index,
                crystal_sites=crystal_sites,
                base_flow_displacement=base_flow_displacement,
                proposed=proposed,
            )

        # --- Sub-partition other_sites by template_kind ---
        precursor_sites: list[tuple[int, np.ndarray]] = []
        interface_sites: list[tuple[int, np.ndarray]] = []
        remaining_sites: list[tuple[int, np.ndarray]] = []
        if other_sites:
            other_site_ids = np.asarray([site_id for site_id, _ in other_sites], dtype=np.int32)
            other_state_ids = self.latent.phase_state_ids[frame_index, other_site_ids]
            precursor_sites = [other_sites[idx] for idx in np.flatnonzero(self._state_kind_is_precursor[other_state_ids])]
            interface_sites = [other_sites[idx] for idx in np.flatnonzero(self._state_kind_is_interface[other_state_ids])]
            remaining_mask = ~(self._state_kind_is_precursor[other_state_ids] | self._state_kind_is_interface[other_state_ids])
            remaining_sites = [other_sites[idx] for idx in np.flatnonzero(remaining_mask)]

        if precursor_sites:
            self._evolve_precursor_batch(
                frame_index=frame_index,
                precursor_sites=precursor_sites,
                base_flow_displacement=base_flow_displacement,
                proposed=proposed,
            )

        if interface_sites:
            self._evolve_interface_batch(
                frame_index=frame_index,
                interface_sites=interface_sites,
                base_flow_displacement=base_flow_displacement,
                proposed=proposed,
            )

        # --- Per-site fallback for grain_boundary only ---
        for site_id, atom_indices in remaining_sites:
            atom_indices_out, proposed_positions = self._compute_site_update(
                frame_index=frame_index,
                site_id=site_id,
                atom_indices=atom_indices,
                base_flow_displacement=base_flow_displacement,
            )
            proposed[atom_indices_out] = proposed_positions

        return proposed

    def _evolve_liquid_batch(
        self,
        *,
        frame_index: int,
        liquid_sites: list[tuple[int, np.ndarray]],
        base_flow_displacement: np.ndarray,
        proposed: np.ndarray,
    ) -> None:
        """Batch-process all liquid sites in one vectorized pass."""
        all_indices = np.concatenate([idx for _, idx in liquid_sites])
        all_site_ids = np.concatenate([
            np.full(idx.shape[0], site_id, dtype=np.int32) for site_id, idx in liquid_sites
        ])
        base_positions = self.base_atoms[all_indices]
        current_positions = self.current_atoms[all_indices]
        flow_disp = base_flow_displacement[all_indices]

        # Gather per-atom thermal values from site-level latent
        thermals = self.latent.phase_thermal_jitter[frame_index, all_site_ids]
        flow_scale = np.clip(0.65 + 2.5 * thermals, 0.65, 1.15).astype(np.float32)
        target = (base_positions + flow_scale[:, None] * flow_disp).astype(np.float32)

        max_step_base = 0.28 * self.avg_nn_distance
        thermal_boost = np.minimum(
            0.12 * self.avg_nn_distance,
            0.55 * thermals * self.avg_nn_distance,
        )
        max_step = (max_step_base + thermal_boost).astype(np.float32)

        displacement = target - current_positions
        step_norm = np.linalg.norm(displacement, axis=1)
        step_scale = np.minimum(1.0, max_step / np.maximum(step_norm, 1e-8))
        result = current_positions + step_scale[:, None] * displacement

        # Thermal noise
        noise_prefactor = 0.14
        noise_scale = np.minimum(
            0.08 * self.avg_nn_distance,
            noise_prefactor * thermals * self.avg_nn_distance,
        )
        has_noise = noise_scale > 0.0
        if np.any(has_noise):
            noise_vals = np.zeros_like(result)
            # Use a single rng seeded per frame for the liquid batch
            batch_rng = np.random.default_rng(
                (int(self.config.seed) * 1_000_003 + frame_index * 65_537 + 31) & 0xFFFFFFFF
            )
            noise_vals[has_noise] = (
                batch_rng.normal(size=(int(np.sum(has_noise)), 3)) * noise_scale[has_noise, None]
            ).astype(np.float32)
            result += noise_vals

        proposed[all_indices] = result.astype(np.float32)

    def _evolve_crystal_batch(
        self,
        *,
        frame_index: int,
        crystal_sites: list[tuple[int, np.ndarray]],
        base_flow_displacement: np.ndarray,
        proposed: np.ndarray,
    ) -> None:
        """Batch-process all crystal/defective_crystal sites with fully vectorized lattice targeting."""
        if not crystal_sites:
            return

        # Group sites by recipe key (state_name, variant_id) to batch lattice targeting
        from collections import defaultdict
        recipe_groups: dict[tuple[int, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
        for site_id, atom_indices in crystal_sites:
            state_idx = int(self.latent.phase_state_ids[frame_index, site_id])
            variant_id = int(self.latent.phase_crystal_variant_ids[frame_index, site_id])
            recipe_groups[(state_idx, variant_id)].append((site_id, atom_indices))

        for (state_idx, variant_id), group_sites in recipe_groups.items():
            crystal_recipe = self._site_crystal_recipe_for_state_idx(state_idx=state_idx, variant_id=variant_id)
            self._evolve_crystal_recipe_group(
                frame_index=frame_index,
                group_sites=group_sites,
                recipe=crystal_recipe,
                template_kind=self.graph.states[state_idx].template_kind,
                proposed=proposed,
            )

    def _evolve_crystal_recipe_group(
        self,
        *,
        frame_index: int,
        group_sites: list[tuple[int, np.ndarray]],
        recipe: dict[str, Any],
        template_kind: str,
        proposed: np.ndarray,
    ) -> None:
        """Process all crystal sites sharing the same recipe in one vectorized pass."""
        site_ids = np.array([s for s, _ in group_sites], dtype=np.int32)
        atoms_per_site = np.array([idx.shape[0] for _, idx in group_sites], dtype=np.int32)

        # Pre-compute all rotation matrices at once: (n_sites, 3, 3)
        quats = self.latent.phase_orientation_quaternions[frame_index, site_ids]
        rotations = quaternion_to_rotation_matrix_batch(quats)
        strains = self.latent.phase_strain[frame_index, site_ids]
        thermals = self.latent.phase_thermal_jitter[frame_index, site_ids]

        all_indices = np.concatenate([idx for _, idx in group_sites])
        all_current = self.current_atoms[all_indices]
        per_atom_site_ids = np.repeat(site_ids, atoms_per_site)
        per_atom_local_site_ids = np.repeat(np.arange(site_ids.shape[0], dtype=np.int32), atoms_per_site)
        all_targets = self._nearest_lattice_targets_batched(
            positions=all_current,
            centers=self.phase_centers[per_atom_site_ids],
            rotations=rotations[per_atom_local_site_ids],
            strains=strains[per_atom_local_site_ids],
            recipe=recipe,
        )

        # Vectorized step clamping and noise for ALL atoms across ALL sites in this group
        # Compute per-atom thermal from per-site thermal
        per_atom_thermal = np.repeat(thermals, atoms_per_site)
        base_step = 0.14 * self.avg_nn_distance  # crystal/defective_crystal base
        thermal_boost = np.minimum(0.12 * self.avg_nn_distance, 0.55 * per_atom_thermal * self.avg_nn_distance)
        max_step = (base_step + thermal_boost).astype(np.float32)

        displacement = all_targets - all_current
        step_norm = np.linalg.norm(displacement, axis=1)
        step_scale = np.minimum(1.0, max_step / np.maximum(step_norm, 1e-8))
        result = all_current + step_scale[:, None] * displacement

        # Vectorized noise
        noise_prefactor = 0.06  # crystal/defective_crystal
        noise_scale = np.minimum(
            0.08 * self.avg_nn_distance,
            noise_prefactor * per_atom_thermal * self.avg_nn_distance,
        )
        has_noise = noise_scale > 0.0
        if np.any(has_noise):
            batch_rng = np.random.default_rng(
                (int(self.config.seed) * 1_000_003 + frame_index * 65_537 + 73) & 0xFFFFFFFF
            )
            noise_vals = np.zeros_like(result)
            noise_vals[has_noise] = (
                batch_rng.normal(size=(int(np.sum(has_noise)), 3)) * noise_scale[has_noise, None]
            ).astype(np.float32)
            result += noise_vals

        proposed[all_indices] = result.astype(np.float32)

    def _evolve_precursor_batch(
        self,
        *,
        frame_index: int,
        precursor_sites: list[tuple[int, np.ndarray]],
        base_flow_displacement: np.ndarray,
        proposed: np.ndarray,
    ) -> None:
        """Batch-process all precursor sites with pre-computed rotations and vectorized blend."""
        if not precursor_sites:
            return

        site_ids = np.array([s for s, _ in precursor_sites], dtype=np.int32)
        atoms_per_site = np.array([idx.shape[0] for _, idx in precursor_sites], dtype=np.int32)
        all_indices = np.concatenate([idx for _, idx in precursor_sites])

        # Pre-compute all rotations, strains, etc. at once
        quats = self.latent.phase_orientation_quaternions[frame_index, site_ids]
        rotations = quaternion_to_rotation_matrix_batch(quats)
        strains = self.latent.phase_strain[frame_index, site_ids]
        thermals = self.latent.phase_thermal_jitter[frame_index, site_ids]

        all_current = self.current_atoms[all_indices]
        all_base = self.base_atoms[all_indices]
        all_flow = base_flow_displacement[all_indices]

        # Per-atom thermal for liquid target
        per_atom_thermal = np.repeat(thermals, atoms_per_site)
        flow_scale = np.clip(0.65 + 2.5 * per_atom_thermal, 0.65, 1.15).astype(np.float32)
        all_liquid_target = (all_base + flow_scale[:, None] * all_flow).astype(np.float32)

        # Get crystal recipe for precursor (all precursors use the same crystal recipe)
        crystal_recipe = self._recipe_variant0_by_state_idx[self._primary_crystal_state_idx]
        per_atom_site_ids = np.repeat(site_ids, atoms_per_site)
        per_atom_local_site_ids = np.repeat(np.arange(site_ids.shape[0], dtype=np.int32), atoms_per_site)
        all_targets = self._nearest_lattice_targets_batched(
            positions=all_current,
            centers=self.phase_centers[per_atom_site_ids],
            rotations=rotations[per_atom_local_site_ids],
            strains=strains[per_atom_local_site_ids],
            recipe=crystal_recipe,
        )

        # Vectorized alpha computation
        totals = self._phase_dwell_total[frame_index, site_ids].astype(np.float32)
        remainings = self._phase_dwell_remaining[frame_index, site_ids].astype(np.float32)
        progress = np.where(
            totals <= 1, 1.0,
            np.clip(1.0 - (remainings - 1) / np.maximum(totals - 1, 1), 0.0, 1.0),
        )
        # All precursor sites share the same ordered_fraction (from the P state config)
        precursor_state_idx = int(self.latent.phase_state_ids[frame_index, site_ids[0]])
        ordered_fraction = float(
            self.graph.states[precursor_state_idx].template_params.get("ordered_fraction", 0.60)
        )
        per_site_alpha = np.clip(0.18 + 0.50 * ordered_fraction + 0.18 * progress, 0.20, 0.88).astype(np.float32)
        per_atom_alpha = np.repeat(per_site_alpha, atoms_per_site)

        # Vectorized blend: target = liquid + alpha * (structured - liquid)
        blended_target = all_liquid_target + per_atom_alpha[:, None] * (all_targets - all_liquid_target)

        # Vectorized step clamping
        base_step = 0.22 * self.avg_nn_distance  # precursor base
        thermal_boost = np.minimum(0.12 * self.avg_nn_distance, 0.55 * per_atom_thermal * self.avg_nn_distance)
        max_step = (base_step + thermal_boost).astype(np.float32)

        displacement = blended_target - all_current
        step_norm = np.linalg.norm(displacement, axis=1)
        step_scale = np.minimum(1.0, max_step / np.maximum(step_norm, 1e-8))
        result = all_current + step_scale[:, None] * displacement

        # Vectorized noise
        noise_prefactor = 0.10  # precursor
        noise_scale = np.minimum(
            0.08 * self.avg_nn_distance,
            noise_prefactor * per_atom_thermal * self.avg_nn_distance,
        )
        has_noise = noise_scale > 0.0
        if np.any(has_noise):
            batch_rng = np.random.default_rng(
                (int(self.config.seed) * 1_000_003 + frame_index * 65_537 + 47) & 0xFFFFFFFF
            )
            noise_vals = np.zeros_like(result)
            noise_vals[has_noise] = (
                batch_rng.normal(size=(int(np.sum(has_noise)), 3)) * noise_scale[has_noise, None]
            ).astype(np.float32)
            result += noise_vals

        proposed[all_indices] = result.astype(np.float32)

    def _evolve_interface_batch(
        self,
        *,
        frame_index: int,
        interface_sites: list[tuple[int, np.ndarray]],
        base_flow_displacement: np.ndarray,
        proposed: np.ndarray,
    ) -> None:
        """Batch-process all interface sites with pre-computed rotations."""
        if not interface_sites:
            return

        site_ids = np.array([s for s, _ in interface_sites], dtype=np.int32)
        atoms_per_site = np.asarray([idx.shape[0] for _, idx in interface_sites], dtype=np.int32)
        all_indices = np.concatenate([idx for _, idx in interface_sites])
        per_atom_local_site_ids = np.repeat(np.arange(site_ids.shape[0], dtype=np.int32), atoms_per_site)
        state_ids = self.latent.phase_state_ids[frame_index, site_ids]

        quats = self.latent.phase_orientation_quaternions[frame_index, site_ids]
        rotations = quaternion_to_rotation_matrix_batch(quats)
        strains = self.latent.phase_strain[frame_index, site_ids]
        thermals = self.latent.phase_thermal_jitter[frame_index, site_ids]
        defects = self.latent.phase_defect_amplitude[frame_index, site_ids]
        centers = self.phase_centers[site_ids]

        all_current = self.current_atoms[all_indices]
        all_base = self.base_atoms[all_indices]
        all_flow = base_flow_displacement[all_indices]

        per_atom_thermal = np.repeat(thermals, atoms_per_site)
        per_atom_centers = centers[per_atom_local_site_ids]
        per_atom_state_ids = np.repeat(state_ids, atoms_per_site)

        # Liquid targets (vectorized)
        flow_scale = np.clip(0.65 + 2.5 * per_atom_thermal, 0.65, 1.15).astype(np.float32)
        all_liquid_target = (all_base + flow_scale[:, None] * all_flow).astype(np.float32)

        all_structured_targets = np.empty_like(all_current)
        for state_idx in np.unique(state_ids):
            state_atom_mask = per_atom_state_ids == state_idx
            all_structured_targets[state_atom_mask] = self._nearest_lattice_targets_batched(
                positions=all_current[state_atom_mask],
                centers=per_atom_centers[state_atom_mask],
                rotations=rotations[per_atom_local_site_ids[state_atom_mask]],
                strains=strains[per_atom_local_site_ids[state_atom_mask]],
                recipe=self._site_crystal_recipe_for_state_idx(state_idx=int(state_idx), variant_id=0),
            )

        (
            interface_normals,
            interface_widths,
            interface_plane_positions,
            interface_plane_jitters,
            interface_solid_fractions,
        ) = self._interface_context_batch(
            frame_index=frame_index,
            site_ids=site_ids,
            rotations=rotations,
        )

        per_atom_normals = interface_normals[per_atom_local_site_ids]
        per_atom_widths = interface_widths[per_atom_local_site_ids]
        per_atom_plane_positions = interface_plane_positions[per_atom_local_site_ids]
        per_atom_plane_jitters = interface_plane_jitters[per_atom_local_site_ids]
        per_atom_solid_fractions = interface_solid_fractions[per_atom_local_site_ids]
        signed_distance = np.einsum(
            "ij,ij->i",
            all_current - per_atom_centers,
            per_atom_normals,
            optimize=True,
        )
        signed_distance += self._surface_modulation_batched(
            points=all_current,
            centers=per_atom_centers,
            normals=per_atom_normals,
            site_ids=site_ids[per_atom_local_site_ids],
            amplitudes=per_atom_plane_jitters,
        )
        solid_alpha = _sigmoid(
            (per_atom_plane_positions - signed_distance)
            / np.maximum(0.22 * per_atom_widths, 1e-5)
        )
        solid_alpha = np.clip(per_atom_solid_fractions * solid_alpha, 0.0, 1.0)
        all_targets = all_liquid_target + solid_alpha[:, None] * (all_structured_targets - all_liquid_target)

        # Vectorized step clamping + noise
        base_step = 0.22 * self.avg_nn_distance  # interface base
        thermal_boost = np.minimum(0.12 * self.avg_nn_distance, 0.55 * per_atom_thermal * self.avg_nn_distance)
        max_step = (base_step + thermal_boost).astype(np.float32)
        displacement = all_targets - all_current
        step_norm = np.linalg.norm(displacement, axis=1)
        step_scale = np.minimum(1.0, max_step / np.maximum(step_norm, 1e-8))
        result = all_current + step_scale[:, None] * displacement

        noise_prefactor = 0.10  # interface
        noise_scale = np.minimum(
            0.08 * self.avg_nn_distance,
            noise_prefactor * per_atom_thermal * self.avg_nn_distance,
        )
        has_noise = noise_scale > 0.0
        if np.any(has_noise):
            batch_rng = np.random.default_rng(
                (int(self.config.seed) * 1_000_003 + frame_index * 65_537 + 59) & 0xFFFFFFFF
            )
            noise_vals = np.zeros_like(result)
            noise_vals[has_noise] = (
                batch_rng.normal(size=(int(np.sum(has_noise)), 3)) * noise_scale[has_noise, None]
            ).astype(np.float32)
            result += noise_vals

        proposed[all_indices] = result.astype(np.float32)

    def _compute_site_update(
        self,
        *,
        frame_index: int,
        site_id: int,
        atom_indices: np.ndarray,
        base_flow_displacement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        state_idx = int(self.latent.phase_state_ids[frame_index, site_id])
        state_name = self.graph.name(state_idx)
        state_cfg = self.graph.state_config(state_name)
        variant_id = int(self.latent.phase_crystal_variant_ids[frame_index, site_id])
        thermal = float(self.latent.phase_thermal_jitter[frame_index, site_id])
        defect = float(self.latent.phase_defect_amplitude[frame_index, site_id])
        center = self.phase_centers[site_id]
        current_positions = self.current_atoms[atom_indices]
        base_positions = self.base_atoms[atom_indices]
        rotation = quaternion_to_rotation_matrix(
            self.latent.phase_orientation_quaternions[frame_index, site_id]
        ).astype(np.float32)
        strain = self.latent.phase_strain[frame_index, site_id].astype(np.float32)
        liquid_target = self._liquid_target(
            base_positions=base_positions,
            base_flow_displacement=base_flow_displacement[atom_indices],
            thermal=thermal,
        )

        if state_cfg.template_kind == "liquid":
            target = liquid_target
        else:
            crystal_recipe = self._site_crystal_recipe(state_name=state_name, variant_id=variant_id)
            structured_target = self._nearest_lattice_targets(
                positions=current_positions,
                center=center,
                rotation=rotation,
                recipe=crystal_recipe,
                strain=strain,
            )
            if state_cfg.template_kind == "precursor":
                ordered_fraction = float(state_cfg.template_params.get("ordered_fraction", 0.60))
                progress = self._state_progress(frame_index, site_id)
                alpha = float(np.clip(0.18 + 0.50 * ordered_fraction + 0.18 * progress, 0.20, 0.88))
                target = liquid_target + alpha * (structured_target - liquid_target)
            elif state_cfg.template_kind == "interface":
                interface_context = self._interface_context(frame_index=frame_index, site_id=site_id)
                signed_distance = np.dot(current_positions - center[None, :], interface_context.normal)
                signed_distance += self._surface_modulation(
                    points=current_positions,
                    center=center,
                    normal=interface_context.normal,
                    site_id=site_id,
                    amplitude=interface_context.plane_jitter,
                )
                solid_alpha = _sigmoid(
                    (interface_context.plane_position - signed_distance)
                    / max(0.22 * interface_context.width, 1e-5)
                )
                solid_alpha = np.clip(
                    interface_context.solid_fraction * solid_alpha,
                    0.0,
                    1.0,
                )
                target = liquid_target + solid_alpha[:, None] * (structured_target - liquid_target)
            elif state_cfg.template_kind == "grain_boundary":
                boundary_context = self._grain_boundary_context(frame_index=frame_index, site_id=site_id)
                target_a = self._nearest_lattice_targets(
                    positions=current_positions,
                    center=center,
                    rotation=boundary_context.rotation_a,
                    recipe=self._site_crystal_recipe(state_name="C", variant_id=0),
                    strain=strain,
                )
                target_b = self._nearest_lattice_targets(
                    positions=current_positions,
                    center=center,
                    rotation=boundary_context.rotation_b,
                    recipe=self._site_crystal_recipe(state_name="C", variant_id=0),
                    strain=strain,
                )
                signed_distance = np.dot(current_positions - center[None, :], boundary_context.normal)
                signed_distance += self._surface_modulation(
                    points=current_positions,
                    center=center,
                    normal=boundary_context.normal,
                    site_id=site_id,
                    amplitude=boundary_context.plane_jitter,
                )
                side_alpha = _sigmoid(signed_distance / max(0.24 * boundary_context.width, 1e-5))
                mixed_target = (
                    (1.0 - side_alpha)[:, None] * target_a + side_alpha[:, None] * target_b
                )
                band_weight = np.exp(
                    -0.5 * (signed_distance / max(0.52 * boundary_context.width, 1e-5)) ** 2
                ).astype(np.float32)
                target = structured_target + band_weight[:, None] * (mixed_target - structured_target)
                target += (
                    band_weight[:, None]
                    * defect
                    * 0.10
                    * self.avg_nn_distance
                    * self._atom_reference_directions[atom_indices]
                )
            else:
                target = structured_target

        max_step = self._max_displacement_per_frame(state_cfg.template_kind, thermal)
        noise_scale = self._thermal_noise_scale(state_cfg.template_kind, thermal)
        displacement = target - current_positions
        step_norm = np.linalg.norm(displacement, axis=1)
        step_scale = np.minimum(1.0, max_step / np.maximum(step_norm, 1e-8))
        proposed_positions = current_positions + step_scale[:, None] * displacement
        if noise_scale > 0.0:
            site_rng = np.random.default_rng(self._site_rng_seed(frame_index=frame_index, site_id=site_id))
            proposed_positions += site_rng.normal(
                loc=0.0,
                scale=noise_scale,
                size=proposed_positions.shape,
            ).astype(np.float32)
        return atom_indices, proposed_positions.astype(np.float32)

    def _site_rng_seed(self, *, frame_index: int, site_id: int) -> int:
        return (
            int(self.config.seed) * 1_000_003
            + int(frame_index) * 65_537
            + int(site_id) * 8_191
            + 97
        ) & 0xFFFFFFFF

    def _site_crystal_recipe_for_state_idx(self, *, state_idx: int, variant_id: int) -> dict[str, Any]:
        if state_idx < 0 or state_idx >= self._state_count:
            raise IndexError(f"state_idx must be in [0, {self._state_count}), got {state_idx}.")
        if variant_id == 0:
            return self._recipe_variant0_by_state_idx[state_idx]
        if variant_id == 1:
            recipe = self._recipe_variant1_by_state_idx[state_idx]
            if recipe is None:
                raise ValueError(
                    f"variant_id=1 is not available for state_idx={state_idx}, state_name={self.state_names[state_idx]!r}."
                )
            return recipe
        raise ValueError(
            f"Unsupported crystal variant_id={variant_id} for state_idx={state_idx}, "
            f"state_name={self.state_names[state_idx]!r}."
        )

    def _site_crystal_recipe(self, *, state_name: str, variant_id: int) -> dict[str, Any]:
        return self._site_crystal_recipe_for_state_idx(
            state_idx=self.graph.index(state_name),
            variant_id=variant_id,
        )

    def _liquid_target(
        self,
        *,
        base_positions: np.ndarray,
        base_flow_displacement: np.ndarray,
        thermal: float,
    ) -> np.ndarray:
        flow_scale = float(np.clip(0.65 + 2.5 * thermal, 0.65, 1.15))
        return (base_positions + flow_scale * base_flow_displacement).astype(np.float32)

    def _get_recipe_arrays(self, recipe: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cache base lattice vectors, motif, and inv_lattice per recipe."""
        cache_key = id(recipe)
        # Use object id as key — recipes come from TemplateLibrary which reuses dicts
        cached = self._recipe_cache.get(cache_key)
        if cached is not None:
            return cached
        # Also try content-based key for safety
        lv = np.asarray(recipe["lattice_vectors"], dtype=np.float32)
        motif = np.asarray(recipe["motif"], dtype=np.float32)
        inv_lv = np.linalg.inv(lv.astype(np.float64)).astype(np.float32)
        result = (lv, motif, inv_lv)
        self._recipe_cache[cache_key] = result
        return result

    def _nearest_lattice_targets(
        self,
        *,
        positions: np.ndarray,
        center: np.ndarray,
        rotation: np.ndarray,
        recipe: dict[str, Any],
        strain: np.ndarray,
    ) -> np.ndarray:
        base_lv, motif, base_inv = self._get_recipe_arrays(recipe)
        scale_factors = np.clip(1.0 + strain, 0.85, 1.15)
        lattice_vectors = base_lv * scale_factors[None, :]  # broadcast (3,3) * (3,)
        motif_offsets = motif @ lattice_vectors  # (M, 3)
        # For small strain, inv ≈ base_inv / scale_factors (diagonal scaling)
        inv_lattice = base_inv / scale_factors[None, :]

        local = (positions - center[None, :]) @ rotation  # (N, 3)
        n_atoms = local.shape[0]

        # Vectorized over all motifs at once: (N, 1, 3) - (1, M, 3) -> (N, M, 3)
        fractional = (local[:, None, :] - motif_offsets[None, :, :]) @ inv_lattice
        nearest_cell = np.rint(fractional).astype(np.float32)
        candidate_local = np.einsum("nmj,jk->nmk", nearest_cell, lattice_vectors) + motif_offsets[None, :, :]
        local_residual = candidate_local - local[:, None, :]
        dist_sq = np.einsum("nmk,nmk->nm", local_residual, local_residual)
        motif_choice = np.argmin(dist_sq, axis=1)
        chosen_local = candidate_local[np.arange(n_atoms), motif_choice]
        target_world = chosen_local @ rotation.T + center[None, :]
        return target_world.astype(np.float32)

    def _nearest_lattice_targets_batched(
        self,
        *,
        positions: np.ndarray,
        centers: np.ndarray,
        rotations: np.ndarray,
        strains: np.ndarray,
        recipe: dict[str, Any],
    ) -> np.ndarray:
        base_lv, motif, base_inv = self._get_recipe_arrays(recipe)
        targets = np.empty_like(positions, dtype=np.float32)
        chunk_atoms = int(self._vectorized_lattice_chunk_atoms)
        motif_count = motif.shape[0]
        for start in range(0, positions.shape[0], chunk_atoms):
            stop = min(positions.shape[0], start + chunk_atoms)
            pos_chunk = positions[start:stop]
            center_chunk = centers[start:stop]
            rotation_chunk = rotations[start:stop]
            strain_chunk = strains[start:stop]

            scale_factors = np.clip(1.0 + strain_chunk, 0.85, 1.15).astype(np.float32)
            lattice_vectors = base_lv[None, :, :] * scale_factors[:, None, :]
            motif_offsets = np.einsum("mj,njk->nmk", motif, lattice_vectors, optimize=True)
            inv_lattice = base_inv[None, :, :] / scale_factors[:, None, :]

            local = np.einsum(
                "ni,nij->nj",
                pos_chunk - center_chunk,
                rotation_chunk,
                optimize=True,
            )
            fractional = np.einsum(
                "nmi,nij->nmj",
                local[:, None, :] - motif_offsets,
                inv_lattice,
                optimize=True,
            )
            nearest_cell = np.rint(fractional).astype(np.float32)
            candidate_local = (
                np.einsum("nmi,nij->nmj", nearest_cell, lattice_vectors, optimize=True)
                + motif_offsets
            )
            local_residual = candidate_local - local[:, None, :]
            dist_sq = np.einsum("nmi,nmi->nm", local_residual, local_residual, optimize=True)
            motif_choice = np.argmin(dist_sq, axis=1)
            chosen_local = candidate_local[np.arange(stop - start), motif_choice]
            if chosen_local.shape != (stop - start, 3):
                raise RuntimeError(
                    "Batched lattice targeting produced an unexpected chosen_local shape. "
                    f"expected={(stop - start, 3)}, got={chosen_local.shape}, motif_count={motif_count}."
                )
            targets[start:stop] = (
                np.einsum("ni,nji->nj", chosen_local, rotation_chunk, optimize=True) + center_chunk
            ).astype(np.float32)
        return targets

    def _interface_context(self, *, frame_index: int, site_id: int) -> InterfaceContext:
        rotation = quaternion_to_rotation_matrix(
            self.latent.phase_orientation_quaternions[frame_index, site_id]
        ).astype(np.float32)
        normals, widths, plane_positions, plane_jitters, solid_fractions = self._interface_context_batch(
            frame_index=frame_index,
            site_ids=np.asarray([site_id], dtype=np.int32),
            rotations=rotation[None, :, :],
        )
        return InterfaceContext(
            normal=normals[0],
            width=float(widths[0]),
            plane_position=float(plane_positions[0]),
            plane_jitter=float(plane_jitters[0]),
            solid_fraction=float(solid_fractions[0]),
        )

    def _interface_context_batch(
        self,
        *,
        frame_index: int,
        site_ids: np.ndarray,
        rotations: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        state_ids = self.latent.phase_state_ids[frame_index, site_ids]
        totals = self._phase_dwell_total[frame_index, site_ids].astype(np.float32)
        remainings = self._phase_dwell_remaining[frame_index, site_ids].astype(np.float32)
        progress = np.where(
            totals <= 1.0,
            1.0,
            np.clip(1.0 - (remainings - 1.0) / np.maximum(totals - 1.0, 1.0), 0.0, 1.0),
        ).astype(np.float32)
        widths = self._interface_width_by_state_idx[state_ids]
        plane_jitters = self._interface_plane_jitter_by_state_idx[state_ids]
        solid_fraction_starts = self._interface_solid_fraction_start_by_state_idx[state_ids]
        solid_fraction_ends = self._interface_solid_fraction_end_by_state_idx[state_ids]
        solid_fractions = solid_fraction_starts + progress * (solid_fraction_ends - solid_fraction_starts)
        plane_positions = (2.0 * solid_fractions - 1.0) * 0.55 * widths

        neighbor_indices = self.layout.phase_neighbor_indices[site_ids]
        valid_neighbor_mask = neighbor_indices >= 0
        safe_neighbor_indices = np.where(valid_neighbor_mask, neighbor_indices, 0)
        neighbor_centers = self.phase_centers[safe_neighbor_indices]
        neighbor_state_ids = self.latent.phase_state_ids[frame_index, safe_neighbor_indices]
        liquid_neighbor_mask = valid_neighbor_mask & self._state_kind_is_liquidish[neighbor_state_ids]
        solid_neighbor_mask = valid_neighbor_mask & self._state_kind_is_solidish[neighbor_state_ids]

        liquid_sums = np.sum(
            neighbor_centers * liquid_neighbor_mask[..., None].astype(np.float32),
            axis=1,
            dtype=np.float32,
        )
        solid_sums = np.sum(
            neighbor_centers * solid_neighbor_mask[..., None].astype(np.float32),
            axis=1,
            dtype=np.float32,
        )
        liquid_counts = np.sum(liquid_neighbor_mask, axis=1, dtype=np.int32)
        solid_counts = np.sum(solid_neighbor_mask, axis=1, dtype=np.int32)

        liquid_means = liquid_sums / np.maximum(liquid_counts[:, None], 1)
        solid_means = solid_sums / np.maximum(solid_counts[:, None], 1)
        centers = self.phase_centers[site_ids]
        normals = np.empty_like(centers, dtype=np.float32)
        both_mask = (liquid_counts > 0) & (solid_counts > 0)
        solid_only_mask = (liquid_counts == 0) & (solid_counts > 0)
        liquid_only_mask = (liquid_counts > 0) & (solid_counts == 0)
        neither_mask = ~(both_mask | solid_only_mask | liquid_only_mask)
        normals[both_mask] = solid_means[both_mask] - liquid_means[both_mask]
        normals[solid_only_mask] = solid_means[solid_only_mask] - centers[solid_only_mask]
        normals[liquid_only_mask] = centers[liquid_only_mask] - liquid_means[liquid_only_mask]
        normals[neither_mask] = rotations[neither_mask, 0, :]
        normals = _normalize_vectors(
            normals,
            fallback_axes=site_ids + 1,
        )
        return normals, widths, plane_positions.astype(np.float32), plane_jitters, solid_fractions.astype(np.float32)

    def _grain_boundary_context(self, *, frame_index: int, site_id: int) -> BoundaryContext:
        state_name = self.graph.name(int(self.latent.phase_state_ids[frame_index, site_id]))
        recipe = self.templates.phase_recipe(state_name)
        width = float(recipe["boundary_width"])
        plane_jitter = float(recipe.get("plane_jitter", 0.0))

        neighbor_ids = [int(site_id)]
        neighbor_ids.extend(int(item) for item in self.layout.phase_neighbor_indices[site_id] if int(item) >= 0)
        grain_to_centers: dict[int, list[np.ndarray]] = {}
        for neighbor_id in neighbor_ids:
            grain_id = int(self.latent.phase_grain_ids[frame_index, neighbor_id])
            if grain_id < 0:
                continue
            neighbor_state_name = self.graph.name(int(self.latent.phase_state_ids[frame_index, neighbor_id]))
            neighbor_cfg = self.graph.state_config(neighbor_state_name)
            if neighbor_cfg.template_kind not in {"interface", "crystal", "grain_boundary", "defective_crystal"}:
                continue
            grain_to_centers.setdefault(grain_id, []).append(self.phase_centers[neighbor_id])

        if len(grain_to_centers) < 2:
            current_grain_id = int(self.latent.phase_grain_ids[frame_index, site_id])
            if current_grain_id < 0:
                current_grain_id = 0
            alternate_grain_id = current_grain_id + 1
            grain_to_centers.setdefault(current_grain_id, []).append(self.phase_centers[site_id])
            grain_to_centers.setdefault(alternate_grain_id, []).append(
                self.phase_centers[site_id] + np.array([1.0, 0.0, 0.0], dtype=np.float32)
            )

        ranked_grains = sorted(
            grain_to_centers.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )
        grain_a = int(ranked_grains[0][0])
        grain_b = int(ranked_grains[1][0])
        center_a = np.mean(np.asarray(ranked_grains[0][1], dtype=np.float32), axis=0)
        center_b = np.mean(np.asarray(ranked_grains[1][1], dtype=np.float32), axis=0)
        normal = _normalize_vector(center_b - center_a, fallback_axis=site_id + 11)
        return BoundaryContext(
            normal=normal,
            width=width,
            plane_jitter=plane_jitter,
            grain_a=grain_a,
            grain_b=grain_b,
            rotation_a=self._grain_rotation_matrix(grain_a),
            rotation_b=self._grain_rotation_matrix(grain_b),
        )

    def _grain_rotation_matrix(self, grain_id: int) -> np.ndarray:
        if grain_id in self.latent.grain_orientations:
            return quaternion_to_rotation_matrix(self.latent.grain_orientations[grain_id]).astype(np.float32)
        return np.eye(3, dtype=np.float32)

    def _surface_modulation(
        self,
        *,
        points: np.ndarray,
        center: np.ndarray,
        normal: np.ndarray,
        site_id: int,
        amplitude: float,
    ) -> np.ndarray:
        if amplitude <= 0.0:
            return np.zeros(points.shape[0], dtype=np.float32)
        tangent_a, tangent_b = _orthonormal_tangent_basis(normal)
        rel = points - center[None, :]
        u_coord = np.dot(rel, tangent_a) / max(float(self._surface_scales[site_id, 0]), 1e-5)
        v_coord = np.dot(rel, tangent_b) / max(float(self._surface_scales[site_id, 1]), 1e-5)
        phases = self._surface_phases[site_id]
        modulation = (
            np.sin(u_coord + phases[0])
            + 0.6 * np.sin(v_coord + phases[1])
            + 0.35 * np.sin(u_coord + v_coord + phases[2])
        )
        return (amplitude * modulation).astype(np.float32)

    def _surface_modulation_batched(
        self,
        *,
        points: np.ndarray,
        centers: np.ndarray,
        normals: np.ndarray,
        site_ids: np.ndarray,
        amplitudes: np.ndarray,
    ) -> np.ndarray:
        if not np.any(amplitudes > 0.0):
            return np.zeros(points.shape[0], dtype=np.float32)
        tangent_a, tangent_b = _orthonormal_tangent_basis_batch(normals)
        rel = points - centers
        surface_scales = self._surface_scales[site_ids]
        phases = self._surface_phases[site_ids]
        u_coord = np.einsum("ij,ij->i", rel, tangent_a, optimize=True) / np.maximum(surface_scales[:, 0], 1e-5)
        v_coord = np.einsum("ij,ij->i", rel, tangent_b, optimize=True) / np.maximum(surface_scales[:, 1], 1e-5)
        modulation = (
            np.sin(u_coord + phases[:, 0])
            + 0.6 * np.sin(v_coord + phases[:, 1])
            + 0.35 * np.sin(u_coord + v_coord + phases[:, 2])
        )
        return (amplitudes * modulation).astype(np.float32)

    def _state_progress(self, frame_index: int, site_id: int) -> float:
        total = int(self._phase_dwell_total[frame_index, site_id])
        remaining = int(self._phase_dwell_remaining[frame_index, site_id])
        if total <= 1:
            return 1.0
        progress = 1.0 - (remaining - 1) / max(total - 1, 1)
        return float(np.clip(progress, 0.0, 1.0))

    def _max_displacement_per_frame(self, template_kind: str, thermal: float) -> float:
        if template_kind == "liquid":
            base = 0.28 * self.avg_nn_distance
        elif template_kind in {"precursor", "interface"}:
            base = 0.22 * self.avg_nn_distance
        else:
            base = 0.14 * self.avg_nn_distance
        thermal_boost = min(0.12 * self.avg_nn_distance, 0.55 * thermal * self.avg_nn_distance)
        return float(base + thermal_boost)

    def _thermal_noise_scale(self, template_kind: str, thermal: float) -> float:
        if thermal <= 0.0:
            return 0.0
        if template_kind == "liquid":
            prefactor = 0.14
        elif template_kind in {"precursor", "interface"}:
            prefactor = 0.10
        else:
            prefactor = 0.06
        return float(min(0.08 * self.avg_nn_distance, prefactor * thermal * self.avg_nn_distance))

    def _find_close_pairs_cell_list(self, points: np.ndarray) -> np.ndarray:
        if not _NUMBA_AVAILABLE:
            tree = cKDTree(points, balanced_tree=False)
            return np.asarray(
                tree.query_pairs(r=self._hardcore_distance, output_type="ndarray"),
                dtype=np.int32,
            )
        atom_count = int(points.shape[0])
        if atom_count < 2:
            return np.empty((0, 2), dtype=np.int32)

        cell_coords = np.floor(points / self._relaxation_cell_size).astype(np.int32)
        np.clip(cell_coords, 0, self._relaxation_grid_shape - 1, out=cell_coords)
        flat_ids = np.sum(cell_coords.astype(np.int64) * self._relaxation_grid_strides[None, :], axis=1)
        order = np.argsort(flat_ids, kind="stable")
        sorted_flat_ids = flat_ids[order]
        unique_flat_ids, start_indices, counts = np.unique(
            sorted_flat_ids,
            return_index=True,
            return_counts=True,
        )
        if unique_flat_ids.size == 0:
            return np.empty((0, 2), dtype=np.int32)

        cell_lookup = np.full(self._relaxation_grid_size, -1, dtype=np.int32)
        cell_lookup[unique_flat_ids] = np.arange(unique_flat_ids.size, dtype=np.int32)
        unique_cell_coords = cell_coords[order[start_indices]]
        pair_count = _count_close_pairs_cell_list_numba(
            points.astype(np.float32, copy=False),
            order.astype(np.int64, copy=False),
            unique_flat_ids.astype(np.int64, copy=False),
            start_indices.astype(np.int64, copy=False),
            counts.astype(np.int64, copy=False),
            unique_cell_coords.astype(np.int32, copy=False),
            cell_lookup,
            self._relaxation_grid_shape.astype(np.int32, copy=False),
            self._relaxation_neighbor_deltas,
            self._relaxation_neighbor_flat_offsets,
            float(self._hardcore_distance * self._hardcore_distance),
        )
        if pair_count == 0:
            return np.empty((0, 2), dtype=np.int32)
        pairs = np.empty((pair_count, 2), dtype=np.int32)
        filled_pair_count = _fill_close_pairs_cell_list_numba(
            points.astype(np.float32, copy=False),
            order.astype(np.int64, copy=False),
            unique_flat_ids.astype(np.int64, copy=False),
            start_indices.astype(np.int64, copy=False),
            counts.astype(np.int64, copy=False),
            unique_cell_coords.astype(np.int32, copy=False),
            cell_lookup,
            self._relaxation_grid_shape.astype(np.int32, copy=False),
            self._relaxation_neighbor_deltas,
            self._relaxation_neighbor_flat_offsets,
            float(self._hardcore_distance * self._hardcore_distance),
            pairs,
        )
        if filled_pair_count != pair_count:
            raise RuntimeError(
                "Cell-list overlap finder filled a different number of pairs than it counted. "
                f"counted={pair_count}, filled={filled_pair_count}."
            )
        return pairs

    def _relax_close_contacts(self, atoms: np.ndarray, iterations: int = 2) -> np.ndarray:
        relaxed = atoms.astype(np.float32, copy=True)
        tree = None
        for iteration in range(iterations):
            pairs = self._find_close_pairs_cell_list(relaxed)
            if len(pairs) == 0:
                break
            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            vec = relaxed[j_idx] - relaxed[i_idx]
            dist = np.linalg.norm(vec, axis=1)
            near_zero = dist < 1e-8
            if np.any(near_zero):
                vec[near_zero] = self._atom_reference_directions[i_idx[near_zero]]
                dist[near_zero] = 1.0
            direction = vec / dist[:, None]
            overlap = np.maximum(0.0, self._hardcore_distance - dist)
            push = 0.52 * overlap[:, None] * direction
            adjustment = np.zeros_like(relaxed)
            np.add.at(adjustment, i_idx, -push)
            np.add.at(adjustment, j_idx, push)
            relaxed += adjustment.astype(np.float32)
            relaxed = np.clip(relaxed, 0.0, self.box_size)
            if (
                iteration == 0
                and (len(pairs) <= self._second_pass_pair_threshold)
                and (float(np.max(overlap)) <= self._second_pass_overlap_threshold)
            ):
                tree = None
                break
            # Invalidate tree since positions changed
            tree = None
        # Cache tree for reuse in _extract_local_neighborhoods (rebuild if positions changed)
        if tree is None:
            tree = cKDTree(relaxed, balanced_tree=False)
        self._cached_atom_tree = tree
        return relaxed.astype(np.float32)

    def _extract_local_neighborhoods(
        self,
        atoms: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # Reuse cached tree from _relax_close_contacts if available
        if self._cached_atom_tree is not None:
            tree = self._cached_atom_tree
            self._cached_atom_tree = None
        else:
            tree = cKDTree(atoms, balanced_tree=False)
        query_k = min(max(self.points_per_site * 4, self.points_per_site), atoms.shape[0])
        distances, indices = tree.query(
            self.track_centers,
            k=query_k,
            workers=self._kdtree_workers,
        )
        if query_k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        local_points = np.zeros((self.track_site_count, self.points_per_site, 3), dtype=np.float32)
        local_atom_ids = np.full(atoms.shape[0], -1, dtype=np.int32)
        local_ranks = np.arange(self.points_per_site, dtype=np.int32)
        for site_id in range(self.track_site_count):
            center = self.track_centers[site_id]
            neighbor_indices = np.asarray(indices[site_id], dtype=np.int32)
            neighbor_distances = np.asarray(distances[site_id], dtype=np.float32)
            within_radius = neighbor_indices[neighbor_distances <= self.neighborhood_radius]
            if within_radius.size >= self.points_per_site:
                selected = within_radius[: self.points_per_site]
            else:
                selected = neighbor_indices[: self.points_per_site]
            if selected.size < self.points_per_site:
                raise RuntimeError(
                    "Local neighborhood extraction could not recover enough atoms. "
                    f"site_id={site_id}, selected={selected.size}, required={self.points_per_site}, "
                    f"n_atoms={atoms.shape[0]}, neighborhood_radius={self.neighborhood_radius}."
                )
            points = atoms[selected]
            if self.config.trajectories.center_neighborhoods:
                points = points - center[None, :]
            local_points[site_id] = points.astype(np.float32)
            local_atom_ids[selected] = local_ranks
        return local_points, local_atom_ids

    def _build_frame_metadata(
        self,
        *,
        frame_index: int,
        site_ids: np.ndarray,
        state_ids: np.ndarray,
        grain_ids: np.ndarray,
        displacements: np.ndarray,
    ) -> dict[str, Any]:
        tracked_site_state_counts = {
            state_name: int(count)
            for state_name, count in zip(
                self.state_names,
                np.bincount(self.latent.state_ids[frame_index], minlength=self._state_count).tolist(),
            )
        }
        phase_site_state_counts = {
            state_name: int(count)
            for state_name, count in zip(
                self.state_names,
                np.bincount(self.latent.phase_state_ids[frame_index], minlength=self._state_count).tolist(),
            )
        }
        atom_state_counts = {
            state_name: int(count)
            for state_name, count in zip(
                self.state_names,
                np.bincount(state_ids, minlength=self._state_count).tolist(),
            )
        }
        atom_counts_per_site = np.bincount(site_ids, minlength=self.phase_site_count).astype(np.int32)
        grain_ids_non_negative = grain_ids[grain_ids >= 0]
        return {
            "frame_index": int(frame_index),
            "box_size": float(self.box_size),
            "num_atoms": int(self.current_atoms.shape[0]),
            "tracked_site_count": int(self.track_site_count),
            "phase_site_count": int(self.phase_site_count),
            "site_state_counts": phase_site_state_counts,
            "tracked_site_state_counts": tracked_site_state_counts,
            "phase_site_state_counts": phase_site_state_counts,
            "atom_state_counts": atom_state_counts,
            "num_grains_in_atoms": int(len(np.unique(grain_ids_non_negative))) if grain_ids_non_negative.size else 0,
            "site_atom_count_min": int(np.min(atom_counts_per_site)),
            "site_atom_count_max": int(np.max(atom_counts_per_site)),
            "site_atom_count_mean": float(np.mean(atom_counts_per_site)),
            "phase_site_atom_count_min": int(np.min(atom_counts_per_site)),
            "phase_site_atom_count_max": int(np.max(atom_counts_per_site)),
            "phase_site_atom_count_mean": float(np.mean(atom_counts_per_site)),
            "mean_frame_displacement": float(np.mean(displacements)),
            "max_frame_displacement": float(np.max(displacements)),
            "hardcore_distance": float(self._hardcore_distance),
        }

    def _build_phase_progress_cache(self) -> tuple[np.ndarray, np.ndarray]:
        phase_state_ids = self.latent.phase_state_ids
        num_frames, phase_site_count = phase_state_ids.shape
        dwell_total = np.zeros((num_frames, phase_site_count), dtype=np.int16)
        dwell_remaining = np.zeros((num_frames, phase_site_count), dtype=np.int16)
        for site_id in range(phase_site_count):
            segment_start = 0
            for frame_idx in range(1, num_frames + 1):
                segment_ended = frame_idx == num_frames or int(phase_state_ids[frame_idx, site_id]) != int(
                    phase_state_ids[frame_idx - 1, site_id]
                )
                if not segment_ended:
                    continue
                segment_length = frame_idx - segment_start
                dwell_total[segment_start:frame_idx, site_id] = np.int16(segment_length)
                dwell_remaining[segment_start:frame_idx, site_id] = np.arange(
                    segment_length,
                    0,
                    -1,
                    dtype=np.int16,
                )
                segment_start = frame_idx
        return dwell_total, dwell_remaining


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _normalize_vector(vector: np.ndarray, *, fallback_axis: int) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-8:
        return (vector / norm).astype(np.float32)
    fallback_rng = np.random.default_rng(int(fallback_axis))
    return random_unit_vector(fallback_rng).astype(np.float32)


def _normalize_vectors(vectors: np.ndarray, *, fallback_axes: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1)
    normalized = np.zeros_like(vectors, dtype=np.float32)
    valid = norms > 1e-8
    if np.any(valid):
        normalized[valid] = (vectors[valid] / norms[valid, None]).astype(np.float32)
    if np.any(~valid):
        fallback_axes = np.asarray(fallback_axes, dtype=np.int64)
        for row_idx in np.flatnonzero(~valid):
            fallback_rng = np.random.default_rng(int(fallback_axes[row_idx]))
            normalized[row_idx] = random_unit_vector(fallback_rng).astype(np.float32)
    return normalized


def _orthonormal_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = _normalize_vector(normal, fallback_axis=71)
    trial = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(normal, trial))) > 0.85:
        trial = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_a = trial - np.dot(trial, normal) * normal
    tangent_a = _normalize_vector(tangent_a, fallback_axis=73)
    tangent_b = np.cross(normal, tangent_a).astype(np.float32)
    tangent_b = _normalize_vector(tangent_b, fallback_axis=79)
    return tangent_a, tangent_b


def _orthonormal_tangent_basis_batch(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normals = _normalize_vectors(normals, fallback_axes=np.arange(normals.shape[0], dtype=np.int64) + 71)
    trials = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=np.float32), (normals.shape[0], 1))
    use_y_axis = np.abs(normals[:, 0]) > 0.85
    if np.any(use_y_axis):
        trials[use_y_axis] = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    tangent_a = trials - np.einsum("ij,ij->i", trials, normals, optimize=True)[:, None] * normals
    tangent_a = _normalize_vectors(tangent_a, fallback_axes=np.arange(normals.shape[0], dtype=np.int64) + 73)
    tangent_b = np.cross(normals, tangent_a).astype(np.float32)
    tangent_b = _normalize_vectors(tangent_b, fallback_axes=np.arange(normals.shape[0], dtype=np.int64) + 79)
    return tangent_a, tangent_b


def _recommended_kdtree_workers(atom_count: int) -> int:
    if atom_count <= 0:
        raise ValueError(f"atom_count must be positive, got {atom_count}.")
    import os
    cpu_count = os.cpu_count() or 4
    if atom_count < 80_000:
        return min(4, cpu_count)
    if atom_count < 260_000:
        return min(16, cpu_count)
    # At 300k+ atoms, KDTree queries benefit from many workers
    return min(32, cpu_count)


def _recommended_site_update_workers(site_count: int) -> int:
    if site_count <= 0:
        raise ValueError(f"site_count must be positive, got {site_count}.")
    if site_count < 32:
        return 2
    if site_count < 96:
        return 4
    if site_count < 192:
        return 6
    return 8


def _recommended_assignment_chunk_size(*, estimated_atom_count: int, query_k: int) -> int:
    if estimated_atom_count <= 0:
        raise ValueError(
            f"estimated_atom_count must be positive, got {estimated_atom_count}."
        )
    if query_k <= 0:
        raise ValueError(f"query_k must be positive, got {query_k}.")
    target_points = int(2_400_000 // query_k)
    return int(np.clip(target_points, 20_000, min(120_000, estimated_atom_count)))


if _NUMBA_AVAILABLE:
    @njit(cache=True)
    def _count_close_pairs_cell_list_numba(
        points: np.ndarray,
        order: np.ndarray,
        unique_flat_ids: np.ndarray,
        start_indices: np.ndarray,
        counts: np.ndarray,
        unique_cell_coords: np.ndarray,
        cell_lookup: np.ndarray,
        grid_shape: np.ndarray,
        neighbor_deltas: np.ndarray,
        neighbor_flat_offsets: np.ndarray,
        radius_sq: float,
    ) -> int:
        pair_count = 0
        for group_idx in range(unique_flat_ids.shape[0]):
            start = int(start_indices[group_idx])
            count_a = int(counts[group_idx])
            base_flat_id = int(unique_flat_ids[group_idx])
            base_x = int(unique_cell_coords[group_idx, 0])
            base_y = int(unique_cell_coords[group_idx, 1])
            base_z = int(unique_cell_coords[group_idx, 2])

            for local_i in range(count_a - 1):
                atom_i = int(order[start + local_i])
                px_i = float(points[atom_i, 0])
                py_i = float(points[atom_i, 1])
                pz_i = float(points[atom_i, 2])
                for local_j in range(local_i + 1, count_a):
                    atom_j = int(order[start + local_j])
                    dx = float(points[atom_j, 0]) - px_i
                    dy = float(points[atom_j, 1]) - py_i
                    dz = float(points[atom_j, 2]) - pz_i
                    if dx * dx + dy * dy + dz * dz <= radius_sq:
                        pair_count += 1

            for neighbor_idx in range(neighbor_deltas.shape[0]):
                nx = base_x + int(neighbor_deltas[neighbor_idx, 0])
                ny = base_y + int(neighbor_deltas[neighbor_idx, 1])
                nz = base_z + int(neighbor_deltas[neighbor_idx, 2])
                if (
                    nx < 0
                    or ny < 0
                    or nz < 0
                    or nx >= int(grid_shape[0])
                    or ny >= int(grid_shape[1])
                    or nz >= int(grid_shape[2])
                ):
                    continue
                neighbor_group_idx = int(cell_lookup[base_flat_id + int(neighbor_flat_offsets[neighbor_idx])])
                if neighbor_group_idx < 0:
                    continue
                neighbor_start = int(start_indices[neighbor_group_idx])
                neighbor_count = int(counts[neighbor_group_idx])
                for local_i in range(count_a):
                    atom_i = int(order[start + local_i])
                    px_i = float(points[atom_i, 0])
                    py_i = float(points[atom_i, 1])
                    pz_i = float(points[atom_i, 2])
                    for local_j in range(neighbor_count):
                        atom_j = int(order[neighbor_start + local_j])
                        dx = float(points[atom_j, 0]) - px_i
                        dy = float(points[atom_j, 1]) - py_i
                        dz = float(points[atom_j, 2]) - pz_i
                        if dx * dx + dy * dy + dz * dz <= radius_sq:
                            pair_count += 1
        return pair_count


    @njit(cache=True)
    def _fill_close_pairs_cell_list_numba(
        points: np.ndarray,
        order: np.ndarray,
        unique_flat_ids: np.ndarray,
        start_indices: np.ndarray,
        counts: np.ndarray,
        unique_cell_coords: np.ndarray,
        cell_lookup: np.ndarray,
        grid_shape: np.ndarray,
        neighbor_deltas: np.ndarray,
        neighbor_flat_offsets: np.ndarray,
        radius_sq: float,
        out_pairs: np.ndarray,
    ) -> int:
        pair_write_index = 0
        for group_idx in range(unique_flat_ids.shape[0]):
            start = int(start_indices[group_idx])
            count_a = int(counts[group_idx])
            base_flat_id = int(unique_flat_ids[group_idx])
            base_x = int(unique_cell_coords[group_idx, 0])
            base_y = int(unique_cell_coords[group_idx, 1])
            base_z = int(unique_cell_coords[group_idx, 2])

            for local_i in range(count_a - 1):
                atom_i = int(order[start + local_i])
                px_i = float(points[atom_i, 0])
                py_i = float(points[atom_i, 1])
                pz_i = float(points[atom_i, 2])
                for local_j in range(local_i + 1, count_a):
                    atom_j = int(order[start + local_j])
                    dx = float(points[atom_j, 0]) - px_i
                    dy = float(points[atom_j, 1]) - py_i
                    dz = float(points[atom_j, 2]) - pz_i
                    if dx * dx + dy * dy + dz * dz <= radius_sq:
                        if atom_i <= atom_j:
                            out_pairs[pair_write_index, 0] = atom_i
                            out_pairs[pair_write_index, 1] = atom_j
                        else:
                            out_pairs[pair_write_index, 0] = atom_j
                            out_pairs[pair_write_index, 1] = atom_i
                        pair_write_index += 1

            for neighbor_idx in range(neighbor_deltas.shape[0]):
                nx = base_x + int(neighbor_deltas[neighbor_idx, 0])
                ny = base_y + int(neighbor_deltas[neighbor_idx, 1])
                nz = base_z + int(neighbor_deltas[neighbor_idx, 2])
                if (
                    nx < 0
                    or ny < 0
                    or nz < 0
                    or nx >= int(grid_shape[0])
                    or ny >= int(grid_shape[1])
                    or nz >= int(grid_shape[2])
                ):
                    continue
                neighbor_group_idx = int(cell_lookup[base_flat_id + int(neighbor_flat_offsets[neighbor_idx])])
                if neighbor_group_idx < 0:
                    continue
                neighbor_start = int(start_indices[neighbor_group_idx])
                neighbor_count = int(counts[neighbor_group_idx])
                for local_i in range(count_a):
                    atom_i = int(order[start + local_i])
                    px_i = float(points[atom_i, 0])
                    py_i = float(points[atom_i, 1])
                    pz_i = float(points[atom_i, 2])
                    for local_j in range(neighbor_count):
                        atom_j = int(order[neighbor_start + local_j])
                        dx = float(points[atom_j, 0]) - px_i
                        dy = float(points[atom_j, 1]) - py_i
                        dz = float(points[atom_j, 2]) - pz_i
                        if dx * dx + dy * dy + dz * dz <= radius_sq:
                            if atom_i <= atom_j:
                                out_pairs[pair_write_index, 0] = atom_i
                                out_pairs[pair_write_index, 1] = atom_j
                            else:
                                out_pairs[pair_write_index, 0] = atom_j
                                out_pairs[pair_write_index, 1] = atom_i
                            pair_write_index += 1
        return pair_write_index


    def _warm_cell_list_pair_kernels() -> None:
        global _NUMBA_CELL_LIST_WARMED
        if _NUMBA_CELL_LIST_WARMED:
            return
        dummy_points = np.zeros((2, 3), dtype=np.float32)
        dummy_order = np.asarray([0, 1], dtype=np.int64)
        dummy_unique_flat_ids = np.asarray([0], dtype=np.int64)
        dummy_start_indices = np.asarray([0], dtype=np.int64)
        dummy_counts = np.asarray([2], dtype=np.int64)
        dummy_unique_cell_coords = np.zeros((1, 3), dtype=np.int32)
        dummy_cell_lookup = np.asarray([0], dtype=np.int32)
        dummy_grid_shape = np.asarray([1, 1, 1], dtype=np.int32)
        dummy_neighbor_deltas = np.zeros((0, 3), dtype=np.int32)
        dummy_neighbor_flat_offsets = np.zeros(0, dtype=np.int64)
        dummy_pairs = np.empty((1, 2), dtype=np.int32)
        _count_close_pairs_cell_list_numba(
            dummy_points,
            dummy_order,
            dummy_unique_flat_ids,
            dummy_start_indices,
            dummy_counts,
            dummy_unique_cell_coords,
            dummy_cell_lookup,
            dummy_grid_shape,
            dummy_neighbor_deltas,
            dummy_neighbor_flat_offsets,
            0.0,
        )
        _fill_close_pairs_cell_list_numba(
            dummy_points,
            dummy_order,
            dummy_unique_flat_ids,
            dummy_start_indices,
            dummy_counts,
            dummy_unique_cell_coords,
            dummy_cell_lookup,
            dummy_grid_shape,
            dummy_neighbor_deltas,
            dummy_neighbor_flat_offsets,
            0.0,
            dummy_pairs,
        )
        _NUMBA_CELL_LIST_WARMED = True
else:
    def _warm_cell_list_pair_kernels() -> None:
        return
