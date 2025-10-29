"""
Synthetic atomistic dataset generation pipeline.

Implements the step-by-step procedure described in the project spec to
produce multi-phase atomistic boxes with configurable perturbations.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import itertools

import numpy as np
import yaml

import sys, os
sys.path.append(os.getcwd())
from src.data_utils.synthetic.visualization import generate_visualizations

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PhasePerturbationConfig:
    sigma_thermal: float
    p_dropout: float
    dropout_relax_radius: float
    dropout_relax_max_fraction: float
    rot_bubble_prob: float
    rot_bubble_radius: float
    rot_bubble_angle_deg: float
    density_bubbles: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PhaseConfig:
    name: str
    phase_type: str
    structural_params: Dict[str, Any]
    perturbations: PhasePerturbationConfig


@dataclass
class GrainAssignmentConfig:
    mode: str
    assignments: Optional[List[str]] = None
    probabilities: Optional[Dict[str, float]] = None


@dataclass
class GlobalConfig:
    L: float
    rho_target: float
    avg_nn_dist: float
    grain_count: int
    intermediate_layer_thickness_factor: float
    random_seed: int
    data_path: pathlib.Path
    additional: Dict[str, Any] = field(default_factory=dict)
    t_layer: float = 0.0


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def load_config(path: str | pathlib.Path) -> Tuple[GlobalConfig, Dict[str, PhaseConfig], GrainAssignmentConfig]:
    """
    Load generator configuration from a YAML file and compute derived globals.
    """
    config_path = pathlib.Path(path)
    with config_path.open("r") as handle:
        raw_cfg = yaml.safe_load(handle)

    global_raw = raw_cfg.get("global", {})
    if "L" not in global_raw:
        raise ValueError("Global configuration must specify box side length 'L'")
    data_path = pathlib.Path(global_raw.get("output_dir", "output/synthetic_data"))
    global_cfg = GlobalConfig(
        L=float(global_raw["L"]),
        rho_target=float(global_raw["rho_target"]),
        avg_nn_dist=float(global_raw["avg_nn_dist"]),
        grain_count=int(global_raw["grain_count"]),
        intermediate_layer_thickness_factor=float(global_raw.get("intermediate_layer_thickness_factor", 0.0)),
        random_seed=int(global_raw.get("random_seed", 0)),
        data_path=data_path,
        additional={k: v for k, v in global_raw.items() if k not in {
            "L",
            "rho_target",
            "avg_nn_dist",
            "grain_count",
            "intermediate_layer_thickness_factor",
            "random_seed",
            "output_dir",
        }},
    )

    global_cfg.t_layer = global_cfg.intermediate_layer_thickness_factor * global_cfg.avg_nn_dist

    phases_section = raw_cfg.get("phases", {})
    phase_configs: Dict[str, PhaseConfig] = {}
    for phase_name, phase_payload in phases_section.items():
        structural_params = phase_payload.get("structural_params", {})
        perturb_raw = phase_payload.get("perturbations", {})
        perturb_cfg = PhasePerturbationConfig(
            sigma_thermal=float(perturb_raw.get("sigma_thermal", 0.0)),
            p_dropout=float(perturb_raw.get("p_dropout", 0.0)),
            dropout_relax_radius=float(perturb_raw.get("dropout_relax_radius", 0.0)),
            dropout_relax_max_fraction=float(perturb_raw.get("dropout_relax_max_fraction", 0.0)),
            rot_bubble_prob=float(perturb_raw.get("rot_bubble_prob", 0.0)),
            rot_bubble_radius=float(perturb_raw.get("rot_bubble_radius", 0.0)),
            rot_bubble_angle_deg=float(perturb_raw.get("rot_bubble_angle_deg", 0.0)),
            density_bubbles=list(perturb_raw.get("density_bubbles", [])),
        )
        phase_configs[phase_name] = PhaseConfig(
            name=phase_name,
            phase_type=str(phase_payload["phase_type"]),
            structural_params=structural_params,
            perturbations=perturb_cfg,
        )

    grain_section = raw_cfg.get("grain_assignment", {})
    if "explicit" in grain_section:
        assignment_cfg = GrainAssignmentConfig(mode="explicit", assignments=list(grain_section["explicit"]))
    else:
        probs = grain_section.get("probabilities")
        if probs is None:
            raise ValueError("grain_assignment must provide either 'explicit' or 'probabilities'")
        total_prob = sum(probs.values())
        if not np.isclose(total_prob, 1.0):
            raise ValueError("grain assignment probabilities must sum to 1.0")
        assignment_cfg = GrainAssignmentConfig(mode="probabilistic", probabilities=dict(probs))

    return global_cfg, phase_configs, assignment_cfg


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return random_unit_vector(rng)
    return vec / norm


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """
    Sample a random rotation matrix using the Haar distribution on SO(3).
    """
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    C = 1 - c
    return np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
    ])


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion (w, x, y, z).
    """
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif (m00 > m11) and (m00 > m22):
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    quat = np.array([qw, qx, qy, qz])
    return quat / np.linalg.norm(quat)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
    ])


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0_norm = q0 / np.linalg.norm(q0)
    q1_norm = q1 / np.linalg.norm(q1)
    dot = np.clip(np.dot(q0_norm, q1_norm), -1.0, 1.0)
    if dot < 0.0:
        q1_norm = -q1_norm
        dot = -dot
    if dot > 0.9995:
        result = q0_norm + alpha * (q1_norm - q0_norm)
        return result / np.linalg.norm(result)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0 * q0_norm) + (s1 * q1_norm)


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


class SyntheticAtomisticDatasetGenerator:
    """
    Full featured synthetic atomistic dataset generator following the spec.
    """

    def __init__(
        self,
        config_path: str | pathlib.Path,
        rng: Optional[np.random.Generator] = None,
        progress: bool = True,
    ):
        self.global_cfg, self.phase_cfgs, self.grain_assignment_cfg = load_config(config_path)
        self.rng = rng or np.random.default_rng(self.global_cfg.random_seed)
        self.progress = progress
        self._start_time = time.perf_counter()

        self.reference_structures: Dict[str, Dict[str, Any]] = {}
        self.grains: List[Dict[str, Any]] = []
        self.intermediate_regions: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.interface_candidates: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.grain_neighbors: Dict[int, set[int]] = {}
        self._grain_pure_pre_indices: Dict[int, List[int]] = {}
        self.atoms: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self._next_pre_index: int = 0
        self._final_atoms_cache: Optional[List[Dict[str, Any]]] = None
        self.seed_positions: Optional[np.ndarray] = None

    def _progress(self, message: str) -> None:
        if not self.progress:
            return
        elapsed = time.perf_counter() - self._start_time
        print(f"[{elapsed:7.2f}s] {message}")

    # ------------------------------------------------------------------ #
    # Step 1 – reference structures
    # ------------------------------------------------------------------ #

    def build_reference_structures(self) -> None:
        self._progress(f"Preparing reference structures for {len(self.phase_cfgs)} phases")
        for phase_name, phase_cfg in self.phase_cfgs.items():
            recipe = self._build_phase_recipe(phase_cfg)
            self.reference_structures[phase_name] = recipe
        self._progress("Reference structures constructed")

    def _build_phase_recipe(self, phase_cfg: PhaseConfig) -> Dict[str, Any]:
        avg_nn = self.global_cfg.avg_nn_dist
        phase_type = phase_cfg.phase_type
        if phase_type == "crystal_fcc":
            lattice_constant = avg_nn * math.sqrt(2.0)
            motif = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ])
            recipe = {
                "phase_type": phase_type,
                "lattice_constant": lattice_constant,
                "lattice_vectors": (np.eye(3) * lattice_constant).tolist(),
                "motif": motif.tolist(),
            }
        elif phase_type == "crystal_bcc":
            lattice_constant = (2.0 / math.sqrt(3.0)) * avg_nn
            motif = np.array([
                [0.0, 0.0, 0.0],
                [0.5, 0.5, 0.5],
            ])
            recipe = {
                "phase_type": phase_type,
                "lattice_constant": lattice_constant,
                "lattice_vectors": (np.eye(3) * lattice_constant).tolist(),
                "motif": motif.tolist(),
            }
        elif phase_type == "amorphous_repeat":
            recipe = self._build_amorphous_repeat_recipe(phase_cfg)
        elif phase_type == "amorphous_random":
            recipe = {
                "phase_type": phase_type,
                "min_pair_dist": float(phase_cfg.structural_params.get("min_pair_dist", 0.8 * avg_nn)),
            }
        elif phase_type == "amorphous_mixed":
            recipe = {
                "phase_type": phase_type,
                "min_pair_dist": float(phase_cfg.structural_params.get("min_pair_dist", 0.8 * avg_nn)),
                "embedded_crystal": phase_cfg.structural_params.get("embedded_crystal", "crystal_fcc"),
                "embedded_probability": float(phase_cfg.structural_params.get("embedded_probability", 0.25)),
                "embedded_radius": float(phase_cfg.structural_params.get("embedded_radius", 2.0 * avg_nn)),
            }
        else:
            raise ValueError(f"Unsupported phase type: {phase_type}")

        recipe["name"] = phase_cfg.name
        return recipe

    def _build_amorphous_repeat_recipe(self, phase_cfg: PhaseConfig) -> Dict[str, Any]:
        params = phase_cfg.structural_params
        avg_nn = self.global_cfg.avg_nn_dist
        cell_size = float(params.get("cell_size", 2.5 * avg_nn))
        n_points = int(params.get("motif_point_count", 12))
        min_sep = float(params.get("min_pair_dist", 0.8 * avg_nn))
        rng = np.random.default_rng(self.rng.integers(0, 1_000_000))
        motif: List[np.ndarray] = []
        attempts = 0
        max_attempts = 20_000
        while len(motif) < n_points and attempts < max_attempts:
            candidate = rng.uniform(0.0, cell_size, size=3)
            if all(np.linalg.norm(candidate - p) >= min_sep for p in motif):
                motif.append(candidate)
            attempts += 1
        if not motif:
            raise RuntimeError("Failed to construct amorphous_repeat motif with given constraints")
        motif_arr = np.array(motif)
        tile_vectors = np.eye(3) * cell_size
        return {
            "phase_type": phase_cfg.phase_type,
            "motif": motif_arr.tolist(),
            "tile_vectors": tile_vectors.tolist(),
            "cell_size": cell_size,
            "min_pair_dist": min_sep,
        }

    # ------------------------------------------------------------------ #
    # Step 2 – grains and base assignment
    # ------------------------------------------------------------------ #

    def sample_grains(self) -> None:
        self._progress("Sampling grain seeds and phases")
        seeds = self.rng.uniform(0.0, self.global_cfg.L, size=(self.global_cfg.grain_count, 3))
        phases = self._assign_grain_phases()
        self.grains = []
        for idx, (seed, phase_id) in enumerate(zip(seeds, phases)):
            rotation = random_rotation_matrix(self.rng)
            grain = {
                "grain_id": idx,
                "seed_position": seed,
                "base_phase_id": phase_id,
                "base_rotation": rotation,
            }
            self.grains.append(grain)
        self.seed_positions = seeds
        self._progress(f"Sampled {len(self.grains)} grains")

    def _assign_grain_phases(self) -> List[str]:
        if self.grain_assignment_cfg.mode == "explicit":
            assignments = self.grain_assignment_cfg.assignments or []
            if len(assignments) != self.global_cfg.grain_count:
                raise ValueError("Explicit grain assignments must match grain_count")
            return assignments
        probabilities = self.grain_assignment_cfg.probabilities or {}
        phase_ids = list(probabilities.keys())
        probs = np.array(list(probabilities.values()), dtype=float)
        sampled = self.rng.choice(phase_ids, size=self.global_cfg.grain_count, p=probs)
        return list(sampled)

    # ------------------------------------------------------------------ #
    # Step 3 is handled implicitly during atom assignment; adjacency data
    # is derived from the atomic band detection output.
    # ------------------------------------------------------------------ #

    # ------------------------------------------------------------------ #
    # Step 4 – populate atoms
    # ------------------------------------------------------------------ #

    def populate_atoms(self) -> None:
        if self.seed_positions is None:
            raise RuntimeError("sample_grains must be called before populate_atoms")
        self._progress("Populating atoms for all grains")
        self.atoms = []
        self.intermediate_regions = {}
        self.interface_candidates: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.grain_neighbors = {grain["grain_id"]: set() for grain in self.grains}
        self._grain_pure_pre_indices = {grain["grain_id"]: [] for grain in self.grains}
        self._next_pre_index = 0
        for grain in self.grains:
            phase_id = grain["base_phase_id"]
            phase_recipe = self.reference_structures[phase_id]
            full_cloud = self._generate_full_box_cloud(grain, phase_recipe)
            self._progress(
                f" - Grain {grain['grain_id']} ({phase_id}), full-box candidate atoms {len(full_cloud)}",
            )
            self._classify_and_register_grain_atoms(grain, full_cloud)
        interface_created = self._generate_interface_atoms()
        if interface_created:
            self._progress(f"   Added {interface_created} blended interface atoms")
        self._progress(f"Finished populating {len(self.atoms)} pre-perturbation atoms")

    def _generate_full_box_cloud(self, grain: Dict[str, Any], phase_recipe: Dict[str, Any]) -> np.ndarray:
        phase_type = phase_recipe["phase_type"]
        if phase_type.startswith("crystal_") or phase_type == "amorphous_repeat":
            return self._generate_structured_phase_cloud(grain, phase_recipe)
        if phase_type == "amorphous_random":
            return self._generate_amorphous_random_cloud(grain, phase_recipe)
        if phase_type == "amorphous_mixed":
            return self._generate_amorphous_mixed_cloud(grain, phase_recipe)
        raise ValueError(f"Unsupported phase type: {phase_type}")

    def _classify_and_register_grain_atoms(self, grain: Dict[str, Any], positions: np.ndarray) -> None:
        grain_id = grain["grain_id"]
        phase_id = grain["base_phase_id"]
        orientation = grain["base_rotation"]
        if positions.size == 0:
            return
        for pos in positions:
            classification = self._classify_position_for_grain(pos, grain_id)
            kind = classification.get("kind")
            if kind == "pure":
                atom_record = self._make_atom_record(
                    position=pos,
                    phase_id=phase_id,
                    grain_id=grain_id,
                    orientation=orientation,
                )
                self.atoms.append(atom_record)
                self._grain_pure_pre_indices[grain_id].append(atom_record["pre_index"])
            elif kind == "interface":
                pair = classification["pair"]
                entry = self.interface_candidates.setdefault(pair, self._init_interface_candidate(pair))
                entry["positions"][grain_id].append({
                    "position": pos,
                    "dist_owner": classification["dist_owner"],
                    "dist_neighbor": classification["dist_neighbor"],
                    "projection": classification["projection"],
                })
                neighbor_id = classification["neighbor"]
                self.grain_neighbors[grain_id].add(neighbor_id)

    def _init_interface_candidate(self, pair: Tuple[int, int]) -> Dict[str, Any]:
        seeds = self.seed_positions
        if seeds is None:
            raise RuntimeError("Grain seeds have not been initialised.")
        g0, g1 = pair
        return {
            "grain_pair": pair,
            "phase_ids": (
                self.grains[g0]["base_phase_id"],
                self.grains[g1]["base_phase_id"],
            ),
            "positions": {
                g0: [],
                g1: [],
            },
            "seeds": (
                np.array(self.grains[g0]["seed_position"], dtype=float),
                np.array(self.grains[g1]["seed_position"], dtype=float),
            ),
        }

    def _classify_position_for_grain(self, position: np.ndarray, grain_id: int) -> Dict[str, Any]:
        if self.seed_positions is None:
            raise RuntimeError("Grain seeds have not been initialised.")
        dists = np.linalg.norm(self.seed_positions - position, axis=1)
        order = np.argsort(dists)
        nearest = int(order[0])
        if nearest != grain_id:
            return {"kind": "other"}
        if len(order) == 1:
            return {"kind": "pure"}
        second = int(order[1])
        dist_nearest = float(dists[nearest])
        dist_second = float(dists[second])
        neighbor_phase = self.grains[second]["base_phase_id"]
        owner_phase = self.grains[grain_id]["base_phase_id"]
        if neighbor_phase == owner_phase or self.global_cfg.t_layer <= 0.0:
            return {
                "kind": "pure",
            }
        diff = abs(dist_second - dist_nearest)
        if diff > self.global_cfg.t_layer:
            return {
                "kind": "pure",
            }
        pair = tuple(sorted((grain_id, second)))
        seeds = self.seed_positions
        center = 0.5 * (seeds[pair[0]] + seeds[pair[1]])
        direction = seeds[pair[1]] - seeds[pair[0]]
        norm = np.linalg.norm(direction)
        projection = 0.0 if norm == 0.0 else float(np.dot(position - center, direction / norm))
        return {
            "kind": "interface",
            "pair": pair,
            "neighbor": second,
            "dist_owner": dist_nearest,
            "dist_neighbor": dist_second,
            "projection": projection,
        }

    def _compute_local_bounds(self, rotation: np.ndarray, seed_position: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        L = self.global_cfg.L
        corners = np.array(list(itertools.product([0.0, L], repeat=3)), dtype=float)
        relative = corners - seed_position
        local = (rotation.T @ relative.T).T
        return local.min(axis=0), local.max(axis=0)

    def _generate_structured_phase_cloud(self, grain: Dict[str, Any], phase_recipe: Dict[str, Any]) -> np.ndarray:
        if phase_recipe["phase_type"] == "amorphous_repeat":
            cell_vectors = np.asarray(phase_recipe.get("tile_vectors"), dtype=float)
            motif = np.asarray(phase_recipe.get("motif"), dtype=float)
            motif_fractional = False
        else:
            cell_vectors = np.asarray(phase_recipe.get("lattice_vectors"), dtype=float)
            motif = np.asarray(phase_recipe.get("motif"), dtype=float)
            motif_fractional = True

        motif_offsets = motif @ cell_vectors if motif_fractional else motif
        rotation = grain["base_rotation"]
        seed_pos = grain["seed_position"]
        local_min, local_max = self._compute_local_bounds(rotation, seed_pos)
        cell_norms = np.linalg.norm(cell_vectors, axis=1)
        pad = float(np.max(cell_norms)) if cell_norms.size else self.global_cfg.avg_nn_dist
        local_min = local_min - pad
        local_max = local_max + pad

        if motif_offsets.size == 0:
            return np.zeros((0, 3), dtype=float)
        motif_min = motif_offsets.min(axis=0)
        motif_max = motif_offsets.max(axis=0)

        ranges: List[range] = []
        for axis in range(3):
            axis_vec = cell_vectors[axis]
            axis_norm = np.linalg.norm(axis_vec)
            if axis_norm <= 1e-8:
                ranges.append(range(0, 1))
                continue
            i_min = math.floor((local_min[axis] - motif_max[axis]) / axis_norm) - 1
            i_max = math.ceil((local_max[axis] - motif_min[axis]) / axis_norm) + 1
            if i_min > i_max:
                i_min, i_max = i_max, i_min
            ranges.append(range(i_min, i_max + 1))

        positions: List[np.ndarray] = []
        for i in ranges[0]:
            base_i = i * cell_vectors[0]
            for j in ranges[1]:
                base_j = base_i + j * cell_vectors[1]
                for k in ranges[2]:
                    lattice_origin = base_j + k * cell_vectors[2]
                    for motif_offset in motif_offsets:
                        local = lattice_origin + motif_offset
                        world = rotation @ local + seed_pos
                        if self._inside_box(world):
                            positions.append(world)
        if not positions:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(positions, dtype=float)

    def _generate_amorphous_random_cloud(self, grain: Dict[str, Any], phase_recipe: Dict[str, Any]) -> np.ndarray:
        L = self.global_cfg.L
        min_pair = float(phase_recipe.get("min_pair_dist", 0.8 * self.global_cfg.avg_nn_dist))
        min_pair_sq = min_pair * min_pair
        cell_size = max(min_pair, 1e-6)
        inv_cell = 1.0 / cell_size
        neighbor_offsets = list(itertools.product((-1, 0, 1), repeat=3))
        rng = np.random.default_rng(self.rng.integers(0, 1_000_000))
        target = max(1, int(round(self.global_cfg.rho_target * (L ** 3))))
        grid: Dict[Tuple[int, int, int], List[np.ndarray]] = defaultdict(list)
        positions: List[np.ndarray] = []
        attempts = 0
        max_attempts = max(target * 50, 100_000)

        while len(positions) < target and attempts < max_attempts:
            candidate = rng.uniform(0.0, L, size=3)
            attempts += 1
            cell = tuple(np.floor(candidate * inv_cell).astype(int))
            too_close = False
            for offset in neighbor_offsets:
                neighbor_cell = (cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2])
                for neighbor_point in grid.get(neighbor_cell, []):
                    diff = candidate - neighbor_point
                    if float(diff.dot(diff)) < min_pair_sq:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                continue
            grid[cell].append(candidate.copy())
            positions.append(candidate)

        if len(positions) < target:
            self._progress(
                f"     Warning: grain {grain['grain_id']} generated {len(positions)} amorphous points (< target {target})",
            )
        if not positions:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(positions, dtype=float)

    def _generate_amorphous_mixed_cloud(self, grain: Dict[str, Any], phase_recipe: Dict[str, Any]) -> np.ndarray:
        base_positions = self._generate_amorphous_random_cloud(grain, phase_recipe)
        embedded_phase = phase_recipe.get("embedded_crystal", "crystal_fcc")
        embed_prob = float(phase_recipe.get("embedded_probability", 0.25))
        embed_radius = float(phase_recipe.get("embedded_radius", 2.0 * self.global_cfg.avg_nn_dist))
        crystal_recipe = self.reference_structures.get(embedded_phase)
        if (
            crystal_recipe is None
            or embed_radius <= 0.0
            or base_positions.size == 0
        ):
            return base_positions
        rng = np.random.default_rng(self.rng.integers(0, 1_000_000))
        if rng.random() > embed_prob:
            return base_positions
        center = base_positions[int(rng.integers(len(base_positions)))]
        embed_rotation = random_rotation_matrix(rng)
        embedded_cloud = self._generate_structured_phase_in_ball(
            center=center,
            rotation=embed_rotation,
            phase_recipe=crystal_recipe,
            radius=embed_radius,
        )
        if embedded_cloud.size == 0:
            return base_positions
        combined = np.concatenate([base_positions, embedded_cloud], axis=0)
        return combined

    def _generate_structured_phase_in_ball(
        self,
        center: np.ndarray,
        rotation: np.ndarray,
        phase_recipe: Dict[str, Any],
        radius: float,
    ) -> np.ndarray:
        if radius <= 0.0:
            return np.zeros((0, 3), dtype=float)
        if phase_recipe["phase_type"] == "amorphous_repeat":
            cell_vectors = np.asarray(phase_recipe.get("tile_vectors"), dtype=float)
            motif = np.asarray(phase_recipe.get("motif"), dtype=float)
            motif_fractional = False
        else:
            cell_vectors = np.asarray(phase_recipe.get("lattice_vectors"), dtype=float)
            motif = np.asarray(phase_recipe.get("motif"), dtype=float)
            motif_fractional = True
        motif_offsets = motif @ cell_vectors if motif_fractional else motif
        if motif_offsets.size == 0:
            return np.zeros((0, 3), dtype=float)

        local_min = np.full(3, -radius - self.global_cfg.avg_nn_dist, dtype=float)
        local_max = np.full(3, radius + self.global_cfg.avg_nn_dist, dtype=float)
        motif_min = motif_offsets.min(axis=0)
        motif_max = motif_offsets.max(axis=0)

        ranges: List[range] = []
        for axis in range(3):
            axis_vec = cell_vectors[axis]
            axis_norm = np.linalg.norm(axis_vec)
            if axis_norm <= 1e-8:
                ranges.append(range(0, 1))
                continue
            i_min = math.floor((local_min[axis] - motif_max[axis]) / axis_norm) - 1
            i_max = math.ceil((local_max[axis] - motif_min[axis]) / axis_norm) + 1
            if i_min > i_max:
                i_min, i_max = i_max, i_min
            ranges.append(range(i_min, i_max + 1))

        radius_sq = radius * radius
        positions: List[np.ndarray] = []
        for i in ranges[0]:
            base_i = i * cell_vectors[0]
            for j in ranges[1]:
                base_j = base_i + j * cell_vectors[1]
                for k in ranges[2]:
                    lattice_origin = base_j + k * cell_vectors[2]
                    for motif_offset in motif_offsets:
                        local = lattice_origin + motif_offset
                        world = rotation @ local + center
                        diff = world - center
                        if diff.dot(diff) > radius_sq:
                            continue
                        if not self._inside_box(world):
                            continue
                        positions.append(world)
        if not positions:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(positions, dtype=float)

    def _evenly_spaced_indices(self, length: int, count: int) -> List[int]:
        if count <= 0 or length <= 0:
            return []
        if count >= length:
            return list(range(length))
        indices: List[int] = []
        step = (length - 1) / max(count - 1, 1)
        for i in range(count):
            idx = int(round(i * step))
            if indices and idx <= indices[-1]:
                idx = min(length - 1, indices[-1] + 1)
            indices.append(min(idx, length - 1))
        return indices

    def _generate_interface_atoms(self) -> int:
        if self.global_cfg.t_layer <= 0.0 or not self.interface_candidates:
            return 0
        created = 0
        for pair, info in self.interface_candidates.items():
            g0, g1 = pair
            positions0 = info["positions"].get(g0, [])
            positions1 = info["positions"].get(g1, [])
            if not positions0 or not positions1:
                continue
            sorted0 = sorted(positions0, key=lambda item: item["projection"])
            sorted1 = sorted(positions1, key=lambda item: item["projection"])
            count = min(len(sorted0), len(sorted1))
            if count == 0:
                continue
            indices0 = self._evenly_spaced_indices(len(sorted0), count)
            indices1 = self._evenly_spaced_indices(len(sorted1), count)
            phase_ids = info["phase_ids"]
            seeds = info["seeds"]
            inter_phase_id = self._register_intermediate_phase(phase_ids[0], phase_ids[1])
            for idx_a, idx_b in zip(indices0, indices1):
                entry_a = sorted0[idx_a]
                entry_b = sorted1[idx_b]
                pos_a = entry_a["position"]
                pos_b = entry_b["position"]
                mid = 0.5 * (pos_a + pos_b)
                dist_mid = self._distance_to_pair(mid, pair)
                if dist_mid is None:
                    continue
                dist_a_mid, dist_b_mid = dist_mid
                if abs(dist_a_mid - dist_b_mid) > self.global_cfg.t_layer:
                    continue
                alpha = self._smooth_alpha(dist_a_mid, dist_b_mid)
                pos_interp = alpha * pos_a + (1.0 - alpha) * pos_b
                if not self._inside_box(pos_interp):
                    continue
                dist_final = self._distance_to_pair(pos_interp, pair)
                if dist_final is None:
                    continue
                if abs(dist_final[0] - dist_final[1]) > self.global_cfg.t_layer:
                    continue
                orientation = self._blend_orientations(
                    self.grains[g0]["base_rotation"],
                    self.grains[g1]["base_rotation"],
                    alpha,
                )
                atom_record = self._make_atom_record(
                    position=pos_interp,
                    phase_id=inter_phase_id,
                    grain_id=None,
                    orientation=orientation,
                    supporting=pair,
                )
                self.atoms.append(atom_record)
                self._register_intermediate_atom(
                    atom_record,
                    grain_a=g0,
                    grain_b=g1,
                    phase_ids=phase_ids,
                    seeds=seeds,
                )
                created += 1
        return created

    def _distance_to_pair(self, position: np.ndarray, pair: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        if self.seed_positions is None:
            return None
        seeds = self.seed_positions[list(pair)]
        dists = np.linalg.norm(seeds - position, axis=1)
        return float(dists[0]), float(dists[1])

    def _smooth_alpha(self, dist_a: float, dist_b: float) -> float:
        if self.global_cfg.t_layer <= 0.0:
            return 0.5
        band = max(self.global_cfg.t_layer, 1e-8)
        normalized = np.clip((dist_b - dist_a) / band, -1.0, 1.0)
        t = 0.5 + 0.5 * normalized
        t = np.clip(t, 0.0, 1.0)
        return float(self._smoothstep(t))

    @staticmethod
    def _smoothstep(t: float) -> float:
        return t * t * (3.0 - 2.0 * t)

    def _inside_box(self, position: np.ndarray) -> bool:
        L = self.global_cfg.L
        eps = 1e-8
        return bool(np.all(position >= -eps) and np.all(position <= L + eps))

    def _make_atom_record(
        self,
        position: np.ndarray,
        phase_id: str,
        grain_id: Optional[int],
        orientation: np.ndarray,
        supporting: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        pre_index = self._next_pre_index
        self._next_pre_index += 1
        record = {
            "pre_index": pre_index,
            "position": np.array(position, dtype=float),
            "phase_id": phase_id,
            "grain_id": grain_id,
            "orientation": np.array(orientation, dtype=float),
            "alive": True,
        }
        if supporting is not None:
            record["supporting_grains"] = supporting
        return record

    def _register_intermediate_phase(self, phase_a: str, phase_b: str) -> str:
        ordered = tuple(sorted((phase_a, phase_b)))
        phase_id = f"intermediate_{ordered[0]}_{ordered[1]}"
        if phase_id not in self.reference_structures:
            self.reference_structures[phase_id] = {
                "phase_type": "intermediate",
                "parents": ordered,
                "thickness": self.global_cfg.t_layer,
                "method": "spatial_blend_fullbox",
            }
        return phase_id

    def _blend_orientations(self, Ra: np.ndarray, Rb: np.ndarray, alpha: float) -> np.ndarray:
        qa = rotation_matrix_to_quaternion(Ra)
        qb = rotation_matrix_to_quaternion(Rb)
        q = quaternion_slerp(qa, qb, alpha)
        return quaternion_to_rotation_matrix(q)

    def _register_intermediate_atom(
        self,
        atom_record: Dict[str, Any],
        grain_a: int,
        grain_b: int,
        phase_ids: Tuple[str, str],
        seeds: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        key = tuple(sorted((grain_a, grain_b)))
        if (grain_a, grain_b) == key:
            ordered_phases = phase_ids
            ordered_seeds = seeds
        else:
            ordered_phases = (phase_ids[1], phase_ids[0])
            ordered_seeds = (seeds[1], seeds[0])
        region = self.intermediate_regions.setdefault(
            key,
            {
                "between_grains": key,
                "phase_id": atom_record["phase_id"],
                "parent_phase_ids": ordered_phases,
                "thickness": self.global_cfg.t_layer,
                "atom_pre_indices": [],
                "seed_positions": (ordered_seeds[0].tolist(), ordered_seeds[1].tolist()),
            },
        )
        if "parent_phase_ids" not in region:
            region["parent_phase_ids"] = ordered_phases
        if "seed_positions" not in region:
            region["seed_positions"] = (ordered_seeds[0].tolist(), ordered_seeds[1].tolist())
        region["atom_pre_indices"].append(atom_record["pre_index"])

    # ------------------------------------------------------------------ #
    # Step 5 – perturbations
    # ------------------------------------------------------------------ #

    def apply_perturbations(self) -> None:
        self.metadata.setdefault("perturbations", {})
        self._progress("   • Rotation bubbles")
        rotation_count = self._apply_rotation_bubbles()
        self._progress(f"     Applied {rotation_count} rotation bubbles")
        self._progress("   • Thermal noise")
        jittered = self._apply_thermal_noise()
        self._progress(f"     Jittered {jittered} atoms with thermal noise")
        self._progress("   • Dropouts with relaxation")
        dropouts = self._apply_dropout_with_relaxation()
        self._progress(f"     Processed {dropouts} dropout events")
        self._progress("   • Density bubbles")
        density_events = self._apply_density_bubbles()
        self._progress(f"     Processed {density_events} density bubble events")

    def _apply_rotation_bubbles(self) -> int:
        records = []
        for grain in self.grains:
            phase_id = grain["base_phase_id"]
            phase_cfg = self.phase_cfgs[phase_id]
            prob = phase_cfg.perturbations.rot_bubble_prob
            if prob <= 0.0:
                continue
            if self.rng.random() > prob:
                continue
            grain_atoms = [atom for atom in self.atoms if atom["grain_id"] == grain["grain_id"] and atom["alive"]]
            if not grain_atoms:
                continue
            center_atom = grain_atoms[int(self.rng.integers(len(grain_atoms)))]
            center = center_atom["position"].copy()
            radius = phase_cfg.perturbations.rot_bubble_radius
            angle_deg = phase_cfg.perturbations.rot_bubble_angle_deg
            angle_rad = math.radians(self.rng.normal(angle_deg, angle_deg * 0.1) if angle_deg > 0 else 0.0)
            axis = random_unit_vector(self.rng)
            R = rotation_matrix_from_axis_angle(axis, angle_rad)
            affected = []
            for atom in grain_atoms:
                if not atom["alive"]:
                    continue
                offset = atom["position"] - center
                dist = np.linalg.norm(offset)
                if dist <= radius:
                    rotated = center + R @ offset
                    atom["position"] = rotated
                    atom["orientation"] = R @ atom["orientation"]
                    affected.append(atom["pre_index"])
            if affected:
                records.append({
                    "grain_id": grain["grain_id"],
                    "center": center.tolist(),
                    "radius": radius,
                    "rotation_axis": axis.tolist(),
                    "rotation_angle_deg": math.degrees(angle_rad),
                    "affected_pre_indices": affected,
                })
        if records:
            self.metadata["perturbations"]["rotation_bubbles"] = records
        return len(records)

    def _apply_thermal_noise(self) -> int:
        per_phase_records = {}
        total_jittered = 0
        for phase_id, phase_cfg in self.phase_cfgs.items():
            sigma = phase_cfg.perturbations.sigma_thermal
            if sigma <= 0.0:
                continue
            phase_seed = int(self.rng.integers(0, 1_000_000))
            noise_rng = np.random.default_rng(phase_seed)
            affected_atoms = [atom for atom in self.atoms if atom["phase_id"] == phase_id and atom["alive"]]
            if not affected_atoms:
                continue
            noise = noise_rng.normal(scale=sigma, size=(len(affected_atoms), 3))
            for atom, jitter in zip(affected_atoms, noise):
                atom["position"] += jitter
            total_jittered += len(affected_atoms)
            per_phase_records[phase_id] = {"sigma_thermal": sigma, "seed": phase_seed}
        if per_phase_records:
            self.metadata["perturbations"]["thermal_noise"] = per_phase_records
        return total_jittered

    def _apply_dropout_with_relaxation(self) -> int:
        dropout_events = []
        for phase_id, phase_cfg in self.phase_cfgs.items():
            p_drop = phase_cfg.perturbations.p_dropout
            if p_drop <= 0.0:
                continue
            drop_seed = int(self.rng.integers(0, 1_000_000))
            drop_rng = np.random.default_rng(drop_seed)
            candidate_atoms = [atom for atom in self.atoms if atom["phase_id"] == phase_id and atom["alive"]]
            if not candidate_atoms:
                continue
            for atom in candidate_atoms:
                if not atom["alive"]:
                    continue
                if drop_rng.random() > p_drop:
                    continue
                vac_pos = atom["position"].copy()
                neighbors = self._find_neighbors(vac_pos, radius=phase_cfg.perturbations.dropout_relax_radius)
                neighbor_indices = []
                neighbor_displacements = []
                for neighbor in neighbors:
                    if neighbor["pre_index"] == atom["pre_index"] or not neighbor["alive"]:
                        continue
                    dir_vec = vac_pos - neighbor["position"]
                    dist = np.linalg.norm(dir_vec)
                    if dist == 0.0 or dist > phase_cfg.perturbations.dropout_relax_radius:
                        continue
                    frac = phase_cfg.perturbations.dropout_relax_max_fraction
                    displacement_mag = frac * (1.0 - dist / phase_cfg.perturbations.dropout_relax_radius) * dist
                    if displacement_mag <= 0.0:
                        continue
                    unit_dir = dir_vec / dist
                    displacement = unit_dir * displacement_mag
                    neighbor["position"] += displacement
                    neighbor_indices.append(neighbor["pre_index"])
                    neighbor_displacements.append(displacement.tolist())
                atom["alive"] = False
                dropout_events.append({
                    "removed_atom_pre_index": atom["pre_index"],
                    "removed_atom_position": vac_pos.tolist(),
                    "neighbor_atom_pre_indices": neighbor_indices,
                    "neighbor_displacements": neighbor_displacements,
                })
            if dropout_events:
                self.metadata.setdefault("perturbations", {}).setdefault("dropouts", {
                    "phase_seed_records": {},
                    "events": [],
                })
                self.metadata["perturbations"]["dropouts"]["phase_seed_records"][phase_id] = drop_seed
        if dropout_events:
            self.metadata["perturbations"].setdefault("dropouts", {"events": []})
            self.metadata["perturbations"]["dropouts"].setdefault("events", [])
            self.metadata["perturbations"]["dropouts"]["events"].extend(dropout_events)
        return len(dropout_events)

    def _find_neighbors(self, position: np.ndarray, radius: float) -> List[Dict[str, Any]]:
        if radius <= 0.0:
            return []
        neighbors = []
        radius_sq = radius * radius
        for atom in self.atoms:
            if not atom["alive"]:
                continue
            diff = atom["position"] - position
            if diff.dot(diff) <= radius_sq:
                neighbors.append(atom)
        return neighbors

    def _apply_density_bubbles(self) -> int:
        bubble_records = []
        for phase_id, phase_cfg in self.phase_cfgs.items():
            bubbles = phase_cfg.perturbations.density_bubbles
            if not bubbles:
                continue
            phase_atoms = [atom for atom in self.atoms if atom["phase_id"] == phase_id and atom["alive"]]
            if not phase_atoms:
                continue
            for bubble_cfg in bubbles:
                expected_count = float(bubble_cfg.get("expected_count", 0.0))
                radius = float(bubble_cfg.get("radius", 0.0))
                alpha = float(bubble_cfg.get("alpha", 0.0))
                if radius <= 0.0:
                    continue
                count = int(self.rng.poisson(expected_count)) if expected_count > 0 else 0
                if count == 0:
                    continue
                for _ in range(count):
                    center_atom = self.rng.choice(phase_atoms)
                    center = center_atom["position"].copy()
                    affected = [atom for atom in phase_atoms if np.linalg.norm(atom["position"] - center) <= radius and atom["alive"]]
                    affected_indices = [atom["pre_index"] for atom in affected]
                    record = {
                        "phase_id": phase_id,
                        "center": center.tolist(),
                        "radius": radius,
                        "alpha": alpha,
                        "affected_pre_indices_before": affected_indices,
                        "removed_pre_indices": [],
                        "cloned_atom_map": {},
                    }
                    if alpha < 0.0:
                        removal_rng = np.random.default_rng(self.rng.integers(0, 1_000_000))
                        for atom in affected:
                            if not atom["alive"]:
                                continue
                            r = np.linalg.norm(atom["position"] - center)
                            prob = min(1.0, -alpha * (1.0 - (r / radius) ** 2))
                            if removal_rng.random() < prob:
                                atom["alive"] = False
                                record["removed_pre_indices"].append(atom["pre_index"])
                    elif alpha > 0.0:
                        clone_rng = np.random.default_rng(self.rng.integers(0, 1_000_000))
                        for atom in affected:
                            if not atom["alive"]:
                                continue
                            r = np.linalg.norm(atom["position"] - center)
                            prob = min(1.0, alpha * (1.0 - (r / radius) ** 2))
                            if clone_rng.random() < prob:
                                jitter = clone_rng.normal(scale=0.25 * self.global_cfg.avg_nn_dist, size=3)
                                clone_position = atom["position"] + jitter
                                clone = self._make_atom_record(
                                    position=clone_position,
                                    phase_id=phase_id,
                                    grain_id=atom["grain_id"],
                                    orientation=atom["orientation"],
                                )
                                self.atoms.append(clone)
                                record["cloned_atom_map"].setdefault(atom["pre_index"], []).append(clone["pre_index"])
                    bubble_records.append(record)
        if bubble_records:
            self.metadata.setdefault("perturbations", {})["density_bubbles"] = bubble_records
        return len(bubble_records)

    # ------------------------------------------------------------------ #
    # Step 6 – saving outputs (after renumbering)
    # ------------------------------------------------------------------ #

    def save_outputs(self) -> None:
        output_dir = self.global_cfg.data_path
        output_dir.mkdir(parents=True, exist_ok=True)
        final_atoms, pre_to_final = self._finalize_atoms()
        self._final_atoms_cache = final_atoms
        self._finalize_metadata(final_atoms, pre_to_final)
        atoms_array = np.array([atom["position"] for atom in final_atoms], dtype=np.float32)
        np.save(output_dir / "atoms.npy", atoms_array)
        np.save(output_dir / "reference_structures.npy", self.reference_structures, allow_pickle=True)
        metadata_path = output_dir / "metadata.json"
        with metadata_path.open("w") as handle:
            json.dump(self.metadata, handle, indent=2)
        self._progress(f"Saved outputs to {output_dir}")

    def _finalize_atoms(self) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
        final_atoms = []
        mapping: Dict[int, int] = {}
        for new_idx, atom in enumerate(atom for atom in self.atoms if atom["alive"]):
            mapping[atom["pre_index"]] = new_idx
            final_atom = {
                "final_index": new_idx,
                "pre_index": atom["pre_index"],
                "position": atom["position"],
                "phase_id": atom["phase_id"],
                "grain_id": atom["grain_id"],
                "orientation": atom["orientation"],
            }
            if "supporting_grains" in atom:
                final_atom["supporting_grains"] = atom["supporting_grains"]
            final_atoms.append(final_atom)
        return final_atoms, mapping

    def _finalize_metadata(self, final_atoms: List[Dict[str, Any]], pre_to_final: Dict[int, int]) -> None:
        L = self.global_cfg.L
        phase_ids_sorted = sorted(self.reference_structures.keys())
        self.metadata["global"] = {
            "box_size": float(L),
            "volume": float(L ** 3),
            "rho_target": float(self.global_cfg.rho_target),
            "avg_nn_dist": float(self.global_cfg.avg_nn_dist),
            "t_layer": float(self.global_cfg.t_layer),
            "intermediate_layer_thickness_factor": float(self.global_cfg.intermediate_layer_thickness_factor),
            "random_seed": int(self.global_cfg.random_seed),
            "grain_count": len(self.grains),
            "phases": phase_ids_sorted,
            "N_final": len(final_atoms),
        }

        final_lookup = {atom["final_index"]: atom for atom in final_atoms}

        grain_records = []
        for grain in self.grains:
            grain_id = grain["grain_id"]
            grain_atoms = [atom for atom in final_atoms if atom["grain_id"] == grain_id]
            if grain_atoms:
                positions = np.array([atom["position"] for atom in grain_atoms])
                bbox_min = positions.min(axis=0).tolist()
                bbox_max = positions.max(axis=0).tolist()
                atom_indices = [atom["final_index"] for atom in grain_atoms]
            else:
                bbox_min = bbox_max = [float(x) for x in grain["seed_position"]]
                atom_indices = []
            atom_index_range = [min(atom_indices), max(atom_indices)] if atom_indices else []
            neighbors = sorted(self.grain_neighbors.get(grain_id, set())) if hasattr(self, "grain_neighbors") else []
            grain_records.append({
                "grain_id": grain_id,
                "base_phase_id": grain["base_phase_id"],
                "seed_position": grain["seed_position"].tolist(),
                "orientation_matrix": grain["base_rotation"].tolist(),
                "orientation_quaternion": rotation_matrix_to_quaternion(grain["base_rotation"]).tolist(),
                "region_aabb": {"min": bbox_min, "max": bbox_max},
                "region_mask_info": {
                    "type": "voronoi_cell_minus_interfaces",
                    "seed_position": grain["seed_position"].tolist(),
                    "neighbor_grain_ids": neighbors,
                    "interface_thickness": float(self.global_cfg.t_layer),
                },
                "atom_indices": atom_indices,
                "atom_index_range": atom_index_range,
            })
        self.metadata["grains"] = grain_records

        inter_records = []
        for key, region in self.intermediate_regions.items():
            final_indices = [
                pre_to_final[idx] for idx in region["atom_pre_indices"] if idx in pre_to_final
            ]
            if not final_indices:
                continue
            atoms_positions = np.array([final_lookup[idx]["position"] for idx in final_indices])
            bbox_min = atoms_positions.min(axis=0).tolist()
            bbox_max = atoms_positions.max(axis=0).tolist()
            atom_index_range = [min(final_indices), max(final_indices)] if final_indices else []
            seed_positions = region.get("seed_positions", (None, None))
            seed_positions_list: Optional[List[List[float]]] = None
            normal_vec = None
            center = None
            if seed_positions[0] is not None and seed_positions[1] is not None:
                seed_a = np.array(seed_positions[0], dtype=float)
                seed_b = np.array(seed_positions[1], dtype=float)
                diff = seed_b - seed_a
                norm_val = np.linalg.norm(diff)
                normal_vec = (diff / norm_val).tolist() if norm_val > 0 else [0.0, 0.0, 0.0]
                center = (0.5 * (seed_a + seed_b)).tolist()
                seed_positions_list = [seed_a.tolist(), seed_b.tolist()]
            inter_records.append({
                "interface_id": f"{key[0]}_{key[1]}",
                "grain_A_id": key[0],
                "grain_B_id": key[1],
                "parent_phase_A_id": region.get("parent_phase_ids", [None, None])[0],
                "parent_phase_B_id": region.get("parent_phase_ids", [None, None])[1],
                "intermediate_phase_id": region["phase_id"],
                "t_layer_used": region.get("thickness", self.global_cfg.t_layer),
                "atom_indices": final_indices,
                "atom_index_range": atom_index_range,
                "bounding_box": {"min": bbox_min, "max": bbox_max},
                "spatial_band_info": {
                    "type": "voronoi_interface_band",
                    "seed_positions": seed_positions_list,
                    "interface_thickness": region.get("thickness", self.global_cfg.t_layer),
                    "normal": normal_vec,
                    "center": center,
                },
            })
        self.metadata["intermediate_regions"] = inter_records

        # Perturbation metadata indexes
        perturb = self.metadata.get("perturbations", {})
        if "rotation_bubbles" in perturb:
            for bubble in perturb["rotation_bubbles"]:
                bubble["affected_atom_indices"] = [
                    pre_to_final[idx] for idx in bubble.pop("affected_pre_indices", []) if idx in pre_to_final
                ]
        if "dropouts" in perturb:
            for event in perturb["dropouts"].get("events", []):
                event["removed_atom_final_index"] = pre_to_final.get(event["removed_atom_pre_index"], None)
                event["neighbor_atom_indices"] = [
                    pre_to_final[idx] for idx in event.get("neighbor_atom_pre_indices", []) if idx in pre_to_final
                ]
                event.pop("neighbor_atom_pre_indices", None)
        if "density_bubbles" in perturb:
            for bubble in perturb["density_bubbles"]:
                bubble["affected_atom_indices_before"] = [
                    pre_to_final[idx] for idx in bubble.get("affected_pre_indices_before", []) if idx in pre_to_final
                ]
                bubble.pop("affected_pre_indices_before", None)
                bubble["removed_atom_indices"] = [
                    pre_to_final[idx] for idx in bubble.get("removed_pre_indices", []) if idx in pre_to_final
                ]
                bubble.pop("removed_pre_indices", None)
                converted_map = {}
                for parent_pre, clones in bubble.get("cloned_atom_map", {}).items():
                    converted_map[str(pre_to_final[parent_pre])] = [
                        pre_to_final[clone] for clone in clones if clone in pre_to_final
                    ]
                bubble["cloned_atom_map"] = converted_map

        # Phase registry
        phase_registry = {}
        for phase_id, recipe in self.reference_structures.items():
            entry: Dict[str, Any] = {
                "phase_type": recipe["phase_type"],
            }
            if recipe["phase_type"] == "intermediate":
                entry["kind"] = "intermediate"
                entry["parents"] = list(recipe["parents"])
                entry["interface_thickness"] = recipe.get("thickness", self.global_cfg.t_layer)
                entry["method"] = recipe.get("method", "spatial_blend_fullbox")
            else:
                entry["kind"] = "pure"
                for key in ("lattice_vectors", "motif", "lattice_constant", "tile_vectors", "min_pair_dist", "cell_size"):
                    if key in recipe:
                        entry[key] = recipe[key]
            phase_registry[phase_id] = entry
        self.metadata["phase_registry"] = phase_registry

    # ------------------------------------------------------------------ #
    # Step 7 – diagnostic visualisations
    # ------------------------------------------------------------------ #

    def create_visualizations(self, final_atoms: Optional[List[Dict[str, Any]]] = None) -> None:
        output_dir = self.global_cfg.data_path
        output_dir.mkdir(parents=True, exist_ok=True)
        if final_atoms is not None:
            self._final_atoms_cache = final_atoms
        elif self._final_atoms_cache is None:
            cached_atoms, _ = self._finalize_atoms()
            self._final_atoms_cache = cached_atoms
        if self._final_atoms_cache is None:
            return
        generate_visualizations(
            global_cfg=self.global_cfg,
            grains=self.grains,
            atoms=self._final_atoms_cache,
            metadata=self.metadata,
            rng=self.rng,
            output_dir=output_dir,
        )

    # ------------------------------------------------------------------ #
    # High-level orchestration
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        self._start_time = time.perf_counter()
        self._progress("Step 1/6: Building reference structures")
        self.build_reference_structures()
        self._progress("Step 2/6: Sampling grains")
        self.sample_grains()
        self._progress("Step 3/6: Populating atoms")
        self.populate_atoms()
        self._progress("Step 4/6: Applying perturbations")
        self.apply_perturbations()
        self._progress("Step 5/6: Saving outputs")
        self.save_outputs()
        self._progress("Step 6/6: Generating visualisations")
        self.create_visualizations()
        self._progress("Generation complete")


__all__ = [
    "SyntheticAtomisticDatasetGenerator",
    "load_config",
]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic atomistic datasets from YAML config.")
    parser.add_argument("config", type=str, nargs="?", default="configs/data/data_synth_no_perturb.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--quiet", action="store_true", help="Disable progress printing")
    args = parser.parse_args()

    generator = SyntheticAtomisticDatasetGenerator(args.config, progress=not args.quiet)
    generator.run()


if __name__ == "__main__":
    main()
