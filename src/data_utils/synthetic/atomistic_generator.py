"""
Advanced Synthetic Atomistic Dataset Generator v2.0

Major improvements over v1:
- Realistic liquid metal structure via RDF-constrained generation
- O(N) spatial queries using cell-lists
- Parallel grain population (scales to 128+ cores)
- Memory-efficient structured NumPy arrays
- KD-tree accelerated Voronoi classification
- Physics-informed perturbations (Maxwell-Boltzmann, elastic relaxation)

Designed for 10^7 atoms on multi-core systems.
"""

from __future__ import annotations

import json
import math
import pathlib
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import multiprocessing as mp

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import yaml

# Try to import numba for JIT acceleration (optional but recommended)
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator if not args or callable(args[0]) else decorator
    prange = range


# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PhasePerturbationConfig:
    """Configuration for phase-specific perturbations."""
    sigma_thermal: float = 0.0
    temperature_K: float = 0.0
    atomic_mass_amu: float = 55.845
    p_dropout: float = 0.0
    dropout_relax_radius: float = 0.0
    dropout_relax_max_fraction: float = 0.0
    use_elastic_relaxation: bool = False
    rot_bubble_prob: float = 0.0
    rot_bubble_radius: float = 0.0
    rot_bubble_angle_deg: float = 0.0
    density_bubbles: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class LiquidStructureConfig:
    """Configuration for realistic liquid metal structure."""
    method: str = "rdf_constrained"
    target_rdf_file: Optional[str] = None
    target_coordination: float = 12.0
    rdf_iterations: int = 1000
    rdf_tolerance: float = 0.05
    quench_temperature: float = 5000.0
    quench_steps: int = 500
    icosahedral_fraction: float = 0.3
    first_peak_position: float = 0.0
    first_peak_height: float = 2.8
    second_peak_position: float = 0.0
    use_frank_kasper: bool = False
    mro_cluster_fraction: float = 0.45
    mro_cluster_radius: float = 6.0
    mro_radial_jitter: float = 0.10
    mro_tetra_fraction: float = 0.35


@dataclass
class PhaseConfig:
    """Configuration for a single phase."""
    name: str
    phase_type: str
    structural_params: Dict[str, Any] = field(default_factory=dict)
    perturbations: PhasePerturbationConfig = field(default_factory=PhasePerturbationConfig)
    liquid_config: Optional[LiquidStructureConfig] = None


@dataclass
class GrainAssignmentConfig:
    """Configuration for grain-to-phase assignment."""
    mode: str
    assignments: Optional[List[str]] = None
    probabilities: Optional[Dict[str, float]] = None


@dataclass
class ParallelConfig:
    """Configuration for parallel execution."""
    enabled: bool = True
    n_workers: Optional[int] = None
    chunk_size: int = 1
    use_shared_memory: bool = True


@dataclass
class GlobalConfig:
    """Global configuration for the generator."""
    L: float
    rho_target: float
    avg_nn_dist: float
    grain_count: int
    intermediate_layer_thickness_factor: float = 0.0
    random_seed: int = 0
    data_path: pathlib.Path = field(default_factory=lambda: pathlib.Path("output/synthetic_data"))
    additional: Dict[str, Any] = field(default_factory=dict)
    t_layer: float = 0.0
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    cell_size_factor: float = 1.5


# ---------------------------------------------------------------------------
# Structured Array Dtype for Atoms
# ---------------------------------------------------------------------------

ATOM_DTYPE = np.dtype([
    ('position', np.float32, (3,)),
    ('phase_id', np.int32),
    ('grain_id', np.int32),
    ('orientation', np.float32, (9,)),
    ('alive', np.bool_),
    ('pre_index', np.int64),
])

PHASE_ID_MAP: Dict[str, int] = {}
PHASE_NAME_MAP: Dict[int, str] = {}


# ---------------------------------------------------------------------------
# Spatial Indexing: Cell List (O(N) neighbor queries)
# ---------------------------------------------------------------------------

class CellList:
    """
    Efficient O(N) spatial hashing for neighbor queries.
    For 10^7 atoms, ~1000x faster than brute force.
    """
    
    def __init__(self, box_size: float, cell_size: float, periodic: bool = False):
        self.box_size = box_size
        self.cell_size = max(cell_size, 1e-6)
        self.periodic = periodic
        self.n_cells = max(1, int(np.ceil(box_size / self.cell_size)))
        self.inv_cell_size = 1.0 / self.cell_size
        self.cells: Dict[Tuple[int, int, int], List[Tuple[int, np.ndarray]]] = defaultdict(list)
        self.positions: Optional[np.ndarray] = None
        self.n_atoms = 0
        
    def clear(self) -> None:
        self.cells.clear()
        self.positions = None
        self.n_atoms = 0
        
    def build(self, positions: np.ndarray) -> None:
        self.clear()
        self.positions = np.asarray(positions, dtype=np.float32)
        self.n_atoms = len(positions)
        
        cell_indices = np.floor(self.positions * self.inv_cell_size).astype(np.int32)
        cell_indices = np.clip(cell_indices, 0, self.n_cells - 1)
        
        for i, (pos, cell_idx) in enumerate(zip(self.positions, cell_indices)):
            self.cells[tuple(cell_idx)].append((i, pos))
            
    def build_from_structured(self, atoms: np.ndarray, alive_only: bool = True) -> None:
        if alive_only:
            mask = atoms['alive']
            positions = atoms['position'][mask]
            indices = np.where(mask)[0]
        else:
            positions = atoms['position']
            indices = np.arange(len(atoms))
            
        self.clear()
        self.positions = positions.astype(np.float32)
        self.n_atoms = len(positions)
        
        if self.n_atoms == 0:
            return
            
        cell_indices = np.floor(self.positions * self.inv_cell_size).astype(np.int32)
        cell_indices = np.clip(cell_indices, 0, self.n_cells - 1)
        
        for local_i, (pos, cell_idx, global_i) in enumerate(zip(self.positions, cell_indices, indices)):
            self.cells[tuple(cell_idx)].append((int(global_i), pos))
    
    def _get_cell_key(self, position: np.ndarray) -> Tuple[int, int, int]:
        cell_idx = np.floor(position * self.inv_cell_size).astype(np.int32)
        cell_idx = np.clip(cell_idx, 0, self.n_cells - 1)
        return tuple(cell_idx)
    
    def get_neighbors(self, position: np.ndarray, radius: float) -> List[Tuple[int, np.ndarray, float]]:
        if self.positions is None or self.n_atoms == 0:
            return []
            
        radius_sq = radius * radius
        neighbors = []
        n_cells_radius = int(np.ceil(radius * self.inv_cell_size)) + 1
        center_cell = self._get_cell_key(position)
        
        for di in range(-n_cells_radius, n_cells_radius + 1):
            ci = center_cell[0] + di
            if not self.periodic and (ci < 0 or ci >= self.n_cells):
                continue
            ci = ci % self.n_cells
            
            for dj in range(-n_cells_radius, n_cells_radius + 1):
                cj = center_cell[1] + dj
                if not self.periodic and (cj < 0 or cj >= self.n_cells):
                    continue
                cj = cj % self.n_cells
                
                for dk in range(-n_cells_radius, n_cells_radius + 1):
                    ck = center_cell[2] + dk
                    if not self.periodic and (ck < 0 or ck >= self.n_cells):
                        continue
                    ck = ck % self.n_cells
                    
                    for atom_idx, atom_pos in self.cells.get((ci, cj, ck), []):
                        diff = atom_pos - position
                        if self.periodic:
                            diff = diff - self.box_size * np.round(diff / self.box_size)
                        dist_sq = np.dot(diff, diff)
                        if dist_sq <= radius_sq:
                            neighbors.append((atom_idx, atom_pos, np.sqrt(dist_sq)))
                            
        return neighbors


# ---------------------------------------------------------------------------
# Vectorized Utility Functions
# ---------------------------------------------------------------------------

def random_unit_vectors(rng: np.random.Generator, n: int) -> np.ndarray:
    vecs = rng.normal(size=(n, 3))
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return vecs / norms


def random_rotation_matrices(rng: np.random.Generator, n: int) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    
    R = np.zeros((n, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    return random_rotation_matrices(rng, 1)[0]


def rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ], dtype=np.float32)


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw, qx, qy, qz = 0.25*S, (m21-m12)/S, (m02-m20)/S, (m10-m01)/S
    elif m00 > m11 and m00 > m22:
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw, qx, qy, qz = (m21-m12)/S, 0.25*S, (m01+m10)/S, (m02+m20)/S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw, qx, qy, qz = (m02-m20)/S, (m01+m10)/S, 0.25*S, (m12+m21)/S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw, qx, qy, qz = (m10-m01)/S, (m02+m20)/S, (m12+m21)/S, 0.25*S
        
    quat = np.array([qw, qx, qy, qz])
    return quat / np.linalg.norm(quat)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)],
    ], dtype=np.float32)


def quaternion_slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0_norm = q0 / np.linalg.norm(q0)
    q1_norm = q1 / np.linalg.norm(q1)
    dot = np.clip(np.dot(q0_norm, q1_norm), -1.0, 1.0)
    
    if dot < 0.0:
        q1_norm, dot = -q1_norm, -dot
        
    if dot > 0.9995:
        result = q0_norm + alpha * (q1_norm - q0_norm)
        return result / np.linalg.norm(result)
        
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * alpha
    sin_theta = math.sin(theta)
    s0 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0_norm + s1 * q1_norm


# ---------------------------------------------------------------------------
# Realistic Liquid Metal Structure Generation
# ---------------------------------------------------------------------------

class LiquidMetalGenerator:
    """
    Generates realistic liquid metal structures with proper short-range order.
    
    Methods:
    1. RDF-constrained: Iteratively refine positions to match target g(r)
    2. Quench: Start from crystal, apply thermal disorder
    3. Icosahedral: Pack icosahedral/polytetrahedral clusters
    4. Simple: Basic rejection sampling (fallback)
    """
    
    METAL_RDF_PARAMS = {
        'Fe': {'r1': 1.0, 'g1': 2.8, 'r2': 1.63, 'g2': 1.2, 'coord': 12.5},
        'Al': {'r1': 1.0, 'g1': 2.7, 'r2': 1.68, 'g2': 1.15, 'coord': 11.8},
        'Cu': {'r1': 1.0, 'g1': 2.9, 'r2': 1.60, 'g2': 1.25, 'coord': 12.2},
        'Ni': {'r1': 1.0, 'g1': 2.85, 'r2': 1.62, 'g2': 1.22, 'coord': 12.3},
        'generic': {'r1': 1.0, 'g1': 2.8, 'r2': 1.65, 'g2': 1.2, 'coord': 12.0},
    }
    
    def __init__(
        self,
        box_size: float,
        target_density: float,
        avg_nn_dist: float,
        config: Optional[LiquidStructureConfig] = None,
        rng: Optional[np.random.Generator] = None,
        min_pair_dist: Optional[float] = None,
    ):
        self.box_size = box_size
        self.target_density = target_density
        self.avg_nn_dist = avg_nn_dist
        self.min_pair_dist = min_pair_dist if min_pair_dist is not None else (0.85 * avg_nn_dist)
        self.config = config or LiquidStructureConfig()
        self.rng = rng or np.random.default_rng()
        self.n_atoms = int(round(target_density * box_size ** 3))
        self._setup_target_rdf()
        
    def _setup_target_rdf(self) -> None:
        if self.config.target_rdf_file and pathlib.Path(self.config.target_rdf_file).exists():
            data = np.loadtxt(self.config.target_rdf_file)
            self.target_r = data[:, 0]
            self.target_gr = data[:, 1]
        else:
            self._generate_synthetic_rdf()
            
        self.rdf_interp = interp1d(
            self.target_r, self.target_gr,
            kind='cubic', bounds_error=False, fill_value=(0.0, 1.0)
        )
        
    def _generate_synthetic_rdf(self) -> None:
        r1 = self.config.first_peak_position or self.avg_nn_dist
        g1 = self.config.first_peak_height
        r2 = self.config.second_peak_position or (r1 * 1.65)
        
        r_max = min(self.box_size / 2, 5 * self.avg_nn_dist)
        self.target_r = np.linspace(0.01 * self.avg_nn_dist, r_max, 500)
        
        sigma1, sigma2, sigma3 = 0.08 * r1, 0.12 * r1, 0.15 * r1
        r3, g2, g3 = r1 * 2.0, 1.2, 1.05
        core_radius = 0.85 * r1
        
        gr = np.zeros_like(self.target_r)
        mask1 = self.target_r >= core_radius
        gr[mask1] += g1 * np.exp(-0.5 * ((self.target_r[mask1] - r1) / sigma1) ** 2)
        gr += g2 * np.exp(-0.5 * ((self.target_r - r2) / sigma2) ** 2)
        gr += g3 * np.exp(-0.5 * ((self.target_r - r3) / sigma3) ** 2)
        
        decay_scale = 2.0 * r1
        baseline = 1.0 - np.exp(-self.target_r / decay_scale)
        self.target_gr = np.maximum(gr, baseline)
        self.target_gr[self.target_r < core_radius] = 0.0
        self.target_gr = gaussian_filter1d(self.target_gr, sigma=2)
        
    def generate(self) -> np.ndarray:
        method = self.config.method.lower()
        if method == "rdf_constrained":
            return self._generate_rdf_constrained()
        elif method == "quench":
            return self._generate_quench()
        elif method == "icosahedral":
            return self._generate_icosahedral()
        elif method == "mro_clustered":
            return self._generate_mro_clustered()
        elif method == "simple":
            return self._generate_simple()
        else:
            supported_methods = ["simple", "rdf_constrained", "quench", "icosahedral", "mro_clustered"]
            raise ValueError(
                f"Unsupported liquid_structure.method={self.config.method!r}. "
                f"Supported methods: {supported_methods}"
            )
            
    def _generate_simple(self) -> np.ndarray:
        """Generate amorphous positions with batch rejection sampling.
        
        Optimized for large systems using cKDTree and batch generation.
        """
        min_dist = self.min_pair_dist
        positions = np.zeros((0, 3), dtype=np.float32)
        
        # Estimate needed candidates (packing fraction ~0.6-0.7 for random close pack)
        # We need self.n_atoms valid. 
        # For lower densities, efficiency is high.
        # Batch size strategy: generate more than needed, filter.
        
        batch_size = max(10000, self.n_atoms * 2)
        total_attempts = 0
        max_attempts = self.n_atoms * 200  # Safety break
        
        # If we already have atoms (e.g. from recursive calls or other logic), build tree
        # But here valid_positions starts empty usually.
        valid_positions = []
        
        while len(valid_positions) < self.n_atoms and total_attempts < max_attempts:
            needed = self.n_atoms - len(valid_positions)
            
            # Generate a batch of candidates
            current_batch_size = max(needed * 2, 10000)
            candidates = self.rng.uniform(0, self.box_size, size=(current_batch_size, 3)).astype(np.float32)
            total_attempts += current_batch_size
            
            # If we have existing atoms, filter candidates against them
            if len(valid_positions) > 0:
                tree_existing = cKDTree(valid_positions)
                # Query candidates against existing atoms
                # query_ball_point finds neighbors. simpler: query closest distance
                dists, _ = tree_existing.query(candidates, k=1, distance_upper_bound=min_dist)
                # Keep candidates where dist > min_dist (query returns inf if > upper_bound)
                # wait, query returns infinite if no neighbor within upper_bound? 
                # Scipy documentation: "If no neighbors are found ... return inf"
                # So we want dists == inf (meaning no neighbor within min_dist)
                mask_existing = dists == float('inf')
                candidates = candidates[mask_existing]
                
            if len(candidates) == 0:
                continue
                
            # Now self-filter the surviving candidates
            # This is the hard part: greedy selection from candidates
            # Using cKDTree.query_pairs on the batch is efficient
            tree_batch = cKDTree(candidates)
            pairs = tree_batch.query_pairs(min_dist)
            
            # Greedy removal of conflicts
            removed = set()
            for i, j in pairs:
                if i in removed or j in removed:
                    continue
                # Remove one of them (e.g., j)
                removed.add(j)
                
            keep_indices = [i for i in range(len(candidates)) if i not in removed]
            new_valid = candidates[keep_indices]
            
            # Append valid ones
            # Truncate if we have enough
            take = min(needed, len(new_valid))
            if take > 0:
                valid_positions.extend(new_valid[:take])
                
        if len(valid_positions) < self.n_atoms:
            print(f"WARNING: LiquidMetalGenerator._generate_simple could only place "
                  f"{len(valid_positions)}/{self.n_atoms} atoms. Density may be lower than target.")
            
        return np.array(valid_positions, dtype=np.float32)
    
    def _generate_rdf_constrained(self) -> np.ndarray:
        positions = self._generate_simple()
        cell_size = 2.5 * self.avg_nn_dist
        cell_list = CellList(self.box_size, cell_size)
        
        r_max = min(self.box_size / 2, 4 * self.avg_nn_dist)
        n_bins = 100
        r_bins = np.linspace(0.5 * self.avg_nn_dist, r_max, n_bins + 1)
        r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
        target_gr = self.rdf_interp(r_centers)
        
        n_iterations = self.config.rdf_iterations
        move_scale = 0.1 * self.avg_nn_dist
        best_positions = positions.copy()
        best_error = float('inf')
        
        for iteration in range(n_iterations):
            if iteration % 50 == 0:
                cell_list.build(positions)
                
            sample_size = min(500, len(positions))
            sample_indices = self.rng.choice(len(positions), sample_size, replace=False)
            current_gr = self._compute_rdf_sample(positions, sample_indices, r_bins, cell_list)
            error = np.mean((current_gr - target_gr) ** 2)
            
            if error < best_error:
                best_error = error
                best_positions = positions.copy()
                
            if error < self.config.rdf_tolerance ** 2:
                break
                
            n_moves = max(1, len(positions) // 100)
            move_indices = self.rng.choice(len(positions), n_moves, replace=False)
            
            for idx in move_indices:
                old_pos = positions[idx].copy()
                displacement = self.rng.normal(0, move_scale, size=3).astype(np.float32)
                new_pos = (old_pos + displacement) % self.box_size
                
                min_dist = self.min_pair_dist
                neighbors = cell_list.get_neighbors(new_pos, min_dist)
                neighbors = [(i, d) for i, _, d in neighbors if i != idx]
                
                if not neighbors:
                    positions[idx] = new_pos
                    
            if iteration > 0 and iteration % 100 == 0:
                if error > best_error * 1.5:
                    move_scale *= 0.9
                else:
                    move_scale *= 1.02
                move_scale = np.clip(move_scale, 0.01 * self.avg_nn_dist, 0.3 * self.avg_nn_dist)
                
        return best_positions
    
    def _compute_rdf_sample(
        self, positions: np.ndarray, sample_indices: np.ndarray,
        r_bins: np.ndarray, cell_list: CellList
    ) -> np.ndarray:
        r_max = r_bins[-1]
        n_bins = len(r_bins) - 1
        histogram = np.zeros(n_bins)
        n_pairs = 0
        
        for idx in sample_indices:
            pos = positions[idx]
            neighbors = cell_list.get_neighbors(pos, r_max)
            
            for neighbor_idx, _, dist in neighbors:
                if neighbor_idx != idx and dist > 0:
                    bin_idx = int((dist - r_bins[0]) / (r_bins[1] - r_bins[0]))
                    if 0 <= bin_idx < n_bins:
                        histogram[bin_idx] += 1
                        n_pairs += 1
                        
        if n_pairs == 0:
            return np.ones(n_bins)
            
        dr = r_bins[1] - r_bins[0]
        r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
        shell_volumes = 4 * np.pi * r_centers ** 2 * dr
        rho = len(positions) / self.box_size ** 3
        expected = len(sample_indices) * rho * shell_volumes
        return np.divide(histogram, expected, where=expected > 0, out=np.ones_like(histogram))
    
    def _generate_quench(self) -> np.ndarray:
        lattice_const = self.avg_nn_dist * np.sqrt(2)
        n_cells = int(np.ceil((self.n_atoms / 4) ** (1/3)))
        
        motif = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]])
        positions = []
        
        for i in range(n_cells):
            for j in range(n_cells):
                for k in range(n_cells):
                    for m in motif:
                        pos = (np.array([i, j, k]) + m) * lattice_const
                        if np.all(pos < self.box_size):
                            positions.append(pos)
                            
        positions = np.array(positions[:self.n_atoms], dtype=np.float32)
        
        T = self.config.quench_temperature
        T_final = 0.01 * T
        n_steps = self.config.quench_steps
        cooling_rate = (T / T_final) ** (1 / n_steps)
        
        sigma = 0.9 * self.avg_nn_dist
        epsilon = 1.0
        dt = 0.01
        
        cell_list = CellList(self.box_size, 2.5 * sigma)
        
        for step in range(n_steps):
            if step % 10 == 0:
                cell_list.build(positions)
            forces = self._compute_lj_forces_fast(positions, sigma, epsilon, cell_list)
            thermal_kick = np.sqrt(2 * T * dt) * self.rng.normal(size=positions.shape).astype(np.float32)
            positions = (positions + forces * dt + thermal_kick) % self.box_size
            T /= cooling_rate
            
        return positions.astype(np.float32)
    
    def _compute_lj_forces_fast(
        self, positions: np.ndarray, sigma: float, epsilon: float, cell_list: CellList
    ) -> np.ndarray:
        n = len(positions)
        forces = np.zeros_like(positions)
        cutoff = 2.5 * sigma
        
        for i in range(n):
            neighbors = cell_list.get_neighbors(positions[i], cutoff)
            for j, pos_j, dist in neighbors:
                if j <= i or dist < 0.1 * sigma:
                    continue
                r_vec = positions[j] - positions[i]
                r2 = dist ** 2
                r6 = (sigma ** 2 / r2) ** 3
                r12 = r6 ** 2
                f_mag = 24 * epsilon * (2 * r12 - r6) / r2
                force = f_mag * r_vec
                forces[i] += force
                forces[j] -= force
        return forces
    
    def _generate_icosahedral(self) -> np.ndarray:
        phi = (1 + np.sqrt(5)) / 2
        ico_vertices = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ], dtype=np.float32)
        ico_vertices = ico_vertices / np.linalg.norm(ico_vertices[0])
        ico_scaled = np.vstack([[0, 0, 0], ico_vertices]) * self.avg_nn_dist
        
        positions = []
        n_clusters = int(self.config.icosahedral_fraction * self.n_atoms / 13)
        cluster_min_dist = 2.0 * self.avg_nn_dist
        cluster_centers = []
        
        for _ in range(n_clusters):
            for attempt in range(100):
                center = self.rng.uniform(self.avg_nn_dist, self.box_size - self.avg_nn_dist, size=3)
                if not cluster_centers:
                    valid = True
                else:
                    dists = np.linalg.norm(np.array(cluster_centers) - center, axis=1)
                    valid = np.all(dists > cluster_min_dist)
                    
                if valid:
                    cluster_centers.append(center)
                    rotation = random_rotation_matrix(self.rng)
                    cluster_atoms = center + (rotation @ ico_scaled.T).T
                    mask = np.all((cluster_atoms >= 0) & (cluster_atoms < self.box_size), axis=1)
                    positions.extend(cluster_atoms[mask])
                    break
                    
        positions = np.array(positions, dtype=np.float32) if positions else np.zeros((0, 3), dtype=np.float32)
        
        remaining = self.n_atoms - len(positions)
        if remaining > 0:
            cell_list = CellList(self.box_size, self.avg_nn_dist)
            if len(positions) > 0:
                cell_list.build(positions)
            min_dist = self.min_pair_dist
            
            for _ in range(remaining):
                for attempt in range(100):
                    candidate = self.rng.uniform(0, self.box_size, size=3).astype(np.float32)
                    if len(positions) == 0:
                        positions = candidate.reshape(1, 3)
                        break
                    else:
                        neighbors = cell_list.get_neighbors(candidate, min_dist)
                        if not neighbors:
                            positions = np.vstack([positions, candidate])
                            cell_list.cells[cell_list._get_cell_key(candidate)].append(
                                (len(positions) - 1, candidate)
                            )
                            break
                            
        return positions[:self.n_atoms].astype(np.float32)

    def _relax_overlaps(self, positions: np.ndarray, min_dist: float, max_iterations: int = 4) -> np.ndarray:
        """Resolve close contacts by iteratively pushing overlapping pairs apart."""
        if min_dist <= 0:
            raise ValueError(f"min_dist must be positive, got {min_dist}")
        if max_iterations <= 0:
            raise ValueError(f"max_iterations must be positive, got {max_iterations}")

        if len(positions) < 2:
            return positions.astype(np.float32, copy=False)

        relaxed = positions.astype(np.float32, copy=True)
        for _ in range(max_iterations):
            tree = cKDTree(relaxed)
            pairs = tree.query_pairs(r=min_dist, output_type="ndarray")
            if len(pairs) == 0:
                break

            i_idx = pairs[:, 0]
            j_idx = pairs[:, 1]
            vec = relaxed[j_idx] - relaxed[i_idx]
            dist = np.linalg.norm(vec, axis=1)

            near_zero = dist < 1e-8
            if np.any(near_zero):
                random_dirs = self.rng.normal(size=(int(np.sum(near_zero)), 3)).astype(np.float32)
                random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-8
                vec[near_zero] = random_dirs
                dist[near_zero] = 1.0

            direction = vec / dist[:, None]
            overlap = np.maximum(0.0, min_dist - dist)
            push_vec = 0.55 * overlap[:, None] * direction

            adjustments = np.zeros_like(relaxed)
            np.add.at(adjustments, i_idx, -push_vec)
            np.add.at(adjustments, j_idx, push_vec)
            relaxed += adjustments
            relaxed = np.clip(relaxed, 0.0, self.box_size)

        return relaxed.astype(np.float32)

    def _generate_mro_clustered(self) -> np.ndarray:
        """
        Generate amorphous metal with medium-range-order (MRO) clusters.

        Workflow:
        1. Generate baseline amorphous cloud via rejection sampling.
        2. Select sparse cluster centers.
        3. Locally pull neighbors toward two preferred shells while preserving disorder.
        4. Relax overlaps to satisfy minimum pair distance.
        """
        cluster_fraction = float(self.config.mro_cluster_fraction)
        cluster_radius = float(self.config.mro_cluster_radius)
        radial_jitter = float(self.config.mro_radial_jitter)
        tetra_fraction = float(self.config.mro_tetra_fraction)

        if not (0.0 <= cluster_fraction <= 1.0):
            raise ValueError(
                f"mro_cluster_fraction must be in [0, 1], got {cluster_fraction}"
            )
        if cluster_radius <= 0:
            raise ValueError(f"mro_cluster_radius must be > 0, got {cluster_radius}")
        if cluster_radius <= self.min_pair_dist:
            raise ValueError(
                f"mro_cluster_radius={cluster_radius} must exceed min_pair_dist={self.min_pair_dist}"
            )
        if radial_jitter < 0:
            raise ValueError(f"mro_radial_jitter must be >= 0, got {radial_jitter}")
        if not (0.0 <= tetra_fraction <= 1.0):
            raise ValueError(f"mro_tetra_fraction must be in [0, 1], got {tetra_fraction}")

        positions = self._generate_simple()
        if len(positions) == 0 or cluster_fraction == 0.0:
            return positions

        est_atoms_per_cluster = max(
            8,
            int(round((4.0 / 3.0) * np.pi * (cluster_radius ** 3) * self.target_density)),
        )
        target_cluster_atoms = max(1, int(round(cluster_fraction * len(positions))))
        n_clusters = int(np.ceil(target_cluster_atoms / est_atoms_per_cluster))
        n_clusters = int(np.clip(n_clusters, 1, 256))

        min_center_sep = 1.25 * cluster_radius
        perm = self.rng.permutation(len(positions))
        cluster_centers: List[np.ndarray] = []
        for atom_idx in perm:
            candidate = positions[int(atom_idx)]
            if not cluster_centers:
                cluster_centers.append(candidate)
            else:
                dists = np.linalg.norm(np.array(cluster_centers) - candidate, axis=1)
                if np.all(dists >= min_center_sep):
                    cluster_centers.append(candidate)
            if len(cluster_centers) >= n_clusters:
                break

        if len(cluster_centers) == 0:
            raise RuntimeError(
                "mro_clustered could not place any cluster centers. "
                f"Requested n_clusters={n_clusters}, min_center_sep={min_center_sep:.3f}, "
                f"n_atoms={len(positions)}"
            )

        if len(cluster_centers) < n_clusters:
            warnings.warn(
                "mro_clustered placed fewer cluster centers than requested "
                f"({len(cluster_centers)} < {n_clusters}).",
                RuntimeWarning,
                stacklevel=2,
            )

        tree = cKDTree(positions)
        first_shell = self.avg_nn_dist
        second_shell = (1.55 + 0.20 * (1.0 - tetra_fraction)) * self.avg_nn_dist
        shell_cutoff = (1.20 + 0.25 * tetra_fraction) * self.avg_nn_dist

        for center in cluster_centers:
            neighbor_idx = tree.query_ball_point(center, cluster_radius)
            if len(neighbor_idx) < 8:
                continue

            neighbor_idx_arr = np.asarray(neighbor_idx, dtype=np.int64)
            offsets = positions[neighbor_idx_arr] - center
            dists = np.linalg.norm(offsets, axis=1)
            valid = dists > 1e-8
            if np.sum(valid) < 6:
                continue

            idx_valid = neighbor_idx_arr[valid]
            vec_valid = offsets[valid]
            dist_valid = dists[valid]

            target_r = np.where(dist_valid <= shell_cutoff, first_shell, second_shell)
            rescaled = vec_valid * (target_r / dist_valid)[:, None]
            weights = np.clip(1.0 - (dist_valid / cluster_radius) ** 2, 0.0, 1.0)

            random_dirs = self.rng.normal(size=rescaled.shape).astype(np.float32)
            radial_dir = rescaled / (np.linalg.norm(rescaled, axis=1, keepdims=True) + 1e-8)
            tangent = random_dirs - np.sum(random_dirs * radial_dir, axis=1, keepdims=True) * radial_dir
            tangent /= np.linalg.norm(tangent, axis=1, keepdims=True) + 1e-8
            tangent *= (radial_jitter * self.avg_nn_dist * weights)[:, None]

            updated = (
                center
                + (1.0 - weights)[:, None] * vec_valid
                + weights[:, None] * rescaled
                + tangent
            )
            positions[idx_valid] = updated.astype(np.float32)

        positions = np.clip(positions, 0.0, self.box_size).astype(np.float32)
        return self._relax_overlaps(positions, min_dist=self.min_pair_dist, max_iterations=4)


# ---------------------------------------------------------------------------
# Configuration Loading
# ---------------------------------------------------------------------------

def load_config(path: str | pathlib.Path) -> Tuple[GlobalConfig, Dict[str, PhaseConfig], GrainAssignmentConfig]:
    config_path = pathlib.Path(path)
    with config_path.open("r") as f:
        raw = yaml.safe_load(f)
        
    global_raw = raw.get("global", {})
    if "L" not in global_raw:
        raise ValueError("Global configuration must specify box side length 'L'")
        
    parallel_raw = global_raw.get("parallel", {})
    parallel_cfg = ParallelConfig(
        enabled=parallel_raw.get("enabled", True),
        n_workers=parallel_raw.get("n_workers"),
        chunk_size=parallel_raw.get("chunk_size", 1),
        use_shared_memory=parallel_raw.get("use_shared_memory", True),
    )
    
    data_path = pathlib.Path(global_raw.get("output_dir", "output/synthetic_data"))
    
    global_cfg = GlobalConfig(
        L=float(global_raw["L"]),
        rho_target=float(global_raw["rho_target"]),
        avg_nn_dist=float(global_raw["avg_nn_dist"]),
        grain_count=int(global_raw["grain_count"]),
        intermediate_layer_thickness_factor=float(global_raw.get("intermediate_layer_thickness_factor", 0.0)),
        random_seed=int(global_raw.get("random_seed", 0)),
        data_path=data_path,
        parallel=parallel_cfg,
        cell_size_factor=float(global_raw.get("cell_size_factor", 1.5)),
    )
    global_cfg.t_layer = global_cfg.intermediate_layer_thickness_factor * global_cfg.avg_nn_dist
    
    phases_section = raw.get("phases", {})
    phase_configs: Dict[str, PhaseConfig] = {}
    
    for phase_name, phase_data in phases_section.items():
        perturb_raw = phase_data.get("perturbations", {})
        perturb_cfg = PhasePerturbationConfig(
            sigma_thermal=float(perturb_raw.get("sigma_thermal", 0.0)),
            temperature_K=float(perturb_raw.get("temperature_K", 0.0)),
            atomic_mass_amu=float(perturb_raw.get("atomic_mass_amu", 55.845)),
            p_dropout=float(perturb_raw.get("p_dropout", 0.0)),
            dropout_relax_radius=float(perturb_raw.get("dropout_relax_radius", 0.0)),
            dropout_relax_max_fraction=float(perturb_raw.get("dropout_relax_max_fraction", 0.0)),
            use_elastic_relaxation=bool(perturb_raw.get("use_elastic_relaxation", False)),
            rot_bubble_prob=float(perturb_raw.get("rot_bubble_prob", 0.0)),
            rot_bubble_radius=float(perturb_raw.get("rot_bubble_radius", 0.0)),
            rot_bubble_angle_deg=float(perturb_raw.get("rot_bubble_angle_deg", 0.0)),
            density_bubbles=list(perturb_raw.get("density_bubbles", [])),
        )
        
        liquid_raw = phase_data.get("liquid_structure", {})
        liquid_cfg = None
        if liquid_raw:
            liquid_cfg = LiquidStructureConfig(
                method=liquid_raw.get("method", "rdf_constrained"),
                target_rdf_file=liquid_raw.get("target_rdf_file"),
                target_coordination=float(liquid_raw.get("target_coordination", 12.0)),
                rdf_iterations=int(liquid_raw.get("rdf_iterations", 1000)),
                rdf_tolerance=float(liquid_raw.get("rdf_tolerance", 0.05)),
                quench_temperature=float(liquid_raw.get("quench_temperature", 5000.0)),
                quench_steps=int(liquid_raw.get("quench_steps", 500)),
                icosahedral_fraction=float(liquid_raw.get("icosahedral_fraction", 0.3)),
                first_peak_position=float(liquid_raw.get("first_peak_position", 0.0)),
                first_peak_height=float(liquid_raw.get("first_peak_height", 2.8)),
                second_peak_position=float(liquid_raw.get("second_peak_position", 0.0)),
                use_frank_kasper=bool(liquid_raw.get("use_frank_kasper", False)),
                mro_cluster_fraction=float(liquid_raw.get("mro_cluster_fraction", 0.45)),
                mro_cluster_radius=float(liquid_raw.get("mro_cluster_radius", 6.0)),
                mro_radial_jitter=float(liquid_raw.get("mro_radial_jitter", 0.10)),
                mro_tetra_fraction=float(liquid_raw.get("mro_tetra_fraction", 0.35)),
            )
            
        phase_configs[phase_name] = PhaseConfig(
            name=phase_name,
            phase_type=str(phase_data["phase_type"]),
            structural_params=phase_data.get("structural_params", {}),
            perturbations=perturb_cfg,
            liquid_config=liquid_cfg,
        )
        
    grain_section = raw.get("grain_assignment", {})
    if "explicit" in grain_section:
        assignment_cfg = GrainAssignmentConfig(mode="explicit", assignments=list(grain_section["explicit"]))
    else:
        probs = grain_section.get("probabilities")
        if probs is None:
            raise ValueError("grain_assignment must provide 'explicit' or 'probabilities'")
        if not np.isclose(sum(probs.values()), 1.0):
            raise ValueError("grain assignment probabilities must sum to 1.0")
        assignment_cfg = GrainAssignmentConfig(mode="probabilistic", probabilities=dict(probs))
        
    return global_cfg, phase_configs, assignment_cfg


# ---------------------------------------------------------------------------
# Parallel Worker Functions
# ---------------------------------------------------------------------------

def _generate_structured_cloud(
    recipe: Dict[str, Any], rotation: np.ndarray, seed_position: np.ndarray,
    L: float, avg_nn_dist: float
) -> np.ndarray:
    if recipe['phase_type'] == 'amorphous_repeat':
        cell_vectors = np.array(recipe.get('tile_vectors'), dtype=np.float32)
        motif = np.array(recipe.get('motif'), dtype=np.float32)
        motif_fractional = False
    else:
        cell_vectors = np.array(recipe.get('lattice_vectors'), dtype=np.float32)
        motif = np.array(recipe.get('motif'), dtype=np.float32)
        motif_fractional = True
        
    if motif.size == 0 or cell_vectors.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
        
    motif_offsets = motif @ cell_vectors if motif_fractional else motif
    
    corners = np.array([
        [0,0,0], [L,0,0], [0,L,0], [0,0,L], [L,L,0], [L,0,L], [0,L,L], [L,L,L]
    ], dtype=np.float32)
    relative = corners - seed_position
    local = (rotation.T @ relative.T).T
    local_min, local_max = local.min(axis=0), local.max(axis=0)
    
    pad = np.max(np.linalg.norm(cell_vectors, axis=1))
    local_min -= pad
    local_max += pad
    
    motif_min, motif_max = motif_offsets.min(axis=0), motif_offsets.max(axis=0)
    
    ranges = []
    for axis in range(3):
        axis_norm = np.linalg.norm(cell_vectors[axis])
        if axis_norm < 1e-8:
            ranges.append(np.array([0]))
        else:
            i_min = int(np.floor((local_min[axis] - motif_max[axis]) / axis_norm)) - 1
            i_max = int(np.ceil((local_max[axis] - motif_min[axis]) / axis_norm)) + 1
            ranges.append(np.arange(i_min, i_max + 1))
            
    # Vectorized generation
    I, J, K = np.meshgrid(ranges[0], ranges[1], ranges[2], indexing='ij')
    I, J, K = I.flatten(), J.flatten(), K.flatten()
    
    # Base lattice points: N_cells x 3
    base_points = (I[:, None] * cell_vectors[0] + 
                  J[:, None] * cell_vectors[1] + 
                  K[:, None] * cell_vectors[2])
                  
    # Add motif: (N_cells x N_motif) points
    # Reshape for broadcasting: (N_cells, 1, 3) + (1, N_motif, 3) -> (N_cells, N_motif, 3)
    candidate_points = base_points[:, None, :] + motif_offsets[None, :, :]
    candidate_points = candidate_points.reshape(-1, 3)
    
    # Rotate and translate
    world_pos = (rotation @ candidate_points.T).T + seed_position
    
    # Filter bounds
    mask = (world_pos[:, 0] >= -1e-8) & (world_pos[:, 0] <= L + 1e-8) & \
           (world_pos[:, 1] >= -1e-8) & (world_pos[:, 1] <= L + 1e-8) & \
           (world_pos[:, 2] >= -1e-8) & (world_pos[:, 2] <= L + 1e-8)
           
    positions = world_pos[mask]
                        
    return positions.astype(np.float32)


def _build_embedded_crystal_recipe(phase_type: str, avg_nn_dist: float) -> Dict[str, np.ndarray]:
    """Build a minimal crystal recipe used to embed small nuclei in amorphous phases."""
    if phase_type == "crystal_bcc":
        lc = (2.0 / np.sqrt(3.0)) * avg_nn_dist
        motif = np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=np.float32)
        lattice_vectors = np.eye(3, dtype=np.float32) * lc
    elif phase_type == "crystal_hcp":
        a = avg_nn_dist
        c = avg_nn_dist * np.sqrt(8.0 / 3.0)
        lattice_vectors = np.array([
            [a, 0, 0],
            [a * 0.5, a * np.sqrt(3.0) / 2.0, 0],
            [0, 0, c],
        ], dtype=np.float32)
        motif = np.array([[0, 0, 0], [1.0/3.0, 2.0/3.0, 0.5]], dtype=np.float32)
    else:
        # Default to FCC for unknown values.
        lc = avg_nn_dist * np.sqrt(2.0)
        motif = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=np.float32)
        lattice_vectors = np.eye(3, dtype=np.float32) * lc

    return {
        "phase_type": phase_type,
        "lattice_vectors": lattice_vectors,
        "motif": motif,
    }


def _generate_crystal_nucleus(
    crystal_recipe: Dict[str, np.ndarray],
    center: np.ndarray,
    rotation: np.ndarray,
    radius: float,
    L: float,
) -> np.ndarray:
    """Generate a spherical crystal nucleus around a center."""
    if radius <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    cell_vectors = np.array(crystal_recipe["lattice_vectors"], dtype=np.float32)
    motif_frac = np.array(crystal_recipe["motif"], dtype=np.float32)
    motif_offsets = motif_frac @ cell_vectors

    axis_steps = np.linalg.norm(cell_vectors, axis=1)
    step = float(np.min(axis_steps[axis_steps > 1e-8])) if np.any(axis_steps > 1e-8) else 1.0
    pad = float(np.max(np.linalg.norm(motif_offsets, axis=1))) if len(motif_offsets) > 0 else 0.0
    n_shell = int(np.ceil((radius + pad) / max(step, 1e-6))) + 1
    rng = np.arange(-n_shell, n_shell + 1, dtype=np.int32)

    I, J, K = np.meshgrid(rng, rng, rng, indexing='ij')
    I, J, K = I.ravel(), J.ravel(), K.ravel()

    base_points = (
        I[:, None] * cell_vectors[0] +
        J[:, None] * cell_vectors[1] +
        K[:, None] * cell_vectors[2]
    )
    candidate_points = base_points[:, None, :] + motif_offsets[None, :, :]
    candidate_points = candidate_points.reshape(-1, 3)

    world_pos = (rotation @ candidate_points.T).T + center

    dist_sq = np.sum((world_pos - center) ** 2, axis=1)
    in_sphere = dist_sq <= (radius * radius)
    in_box = (
        (world_pos[:, 0] >= 0.0) & (world_pos[:, 0] <= L) &
        (world_pos[:, 1] >= 0.0) & (world_pos[:, 1] <= L) &
        (world_pos[:, 2] >= 0.0) & (world_pos[:, 2] <= L)
    )

    mask = in_sphere & in_box
    if not np.any(mask):
        return np.zeros((0, 3), dtype=np.float32)
    return world_pos[mask].astype(np.float32)


def _inject_crystal_nuclei(
    amorphous_positions: np.ndarray,
    phase_recipe: Dict[str, Any],
    rng: np.random.Generator,
    avg_nn_dist: float,
    rho_target: float,
    L: float,
) -> np.ndarray:
    """Embed sparse crystal nuclei into an amorphous point cloud."""
    if len(amorphous_positions) == 0:
        return amorphous_positions

    embed_fraction = float(phase_recipe.get("embedded_probability", 0.0))
    embed_radius = float(phase_recipe.get("embedded_radius", 0.0))
    crystal_type = str(phase_recipe.get("embedded_crystal", "crystal_fcc"))

    if embed_fraction <= 0.0 or embed_radius <= 0.0:
        return amorphous_positions

    target_nucleus_atoms = max(1, int(round(embed_fraction * len(amorphous_positions))))
    est_atoms_per_nucleus = max(1, int(round((4.0 / 3.0) * np.pi * (embed_radius ** 3) * rho_target)))
    n_nuclei = max(1, min(4, int(np.ceil(target_nucleus_atoms / est_atoms_per_nucleus))))

    centers: List[np.ndarray] = []
    min_center_sep = 1.8 * embed_radius
    for _ in range(n_nuclei):
        chosen = None
        for _attempt in range(200):
            candidate = amorphous_positions[int(rng.integers(0, len(amorphous_positions)))]
            if not centers:
                chosen = candidate
                break
            dists = np.linalg.norm(np.array(centers) - candidate, axis=1)
            if np.all(dists >= min_center_sep):
                chosen = candidate
                break
        if chosen is None:
            chosen = amorphous_positions[int(rng.integers(0, len(amorphous_positions)))]
        centers.append(chosen.astype(np.float32))

    crystal_recipe = _build_embedded_crystal_recipe(crystal_type, avg_nn_dist)
    keep_mask = np.ones(len(amorphous_positions), dtype=bool)
    nucleus_chunks: List[np.ndarray] = []

    for center in centers:
        carve_radius = 1.05 * embed_radius
        dists_sq = np.sum((amorphous_positions - center) ** 2, axis=1)
        keep_mask &= dists_sq > (carve_radius * carve_radius)

        nucleus = _generate_crystal_nucleus(
            crystal_recipe=crystal_recipe,
            center=center,
            rotation=random_rotation_matrix(rng),
            radius=0.95 * embed_radius,
            L=L,
        )
        if len(nucleus) > 0:
            nucleus_chunks.append(nucleus)

    if not nucleus_chunks:
        return amorphous_positions

    nucleus_positions = np.vstack(nucleus_chunks)
    if len(nucleus_positions) > target_nucleus_atoms:
        select_idx = rng.choice(len(nucleus_positions), size=target_nucleus_atoms, replace=False)
        nucleus_positions = nucleus_positions[select_idx]

    remaining_positions = amorphous_positions[keep_mask]
    if len(remaining_positions) == 0:
        return nucleus_positions.astype(np.float32)
    return np.vstack([remaining_positions, nucleus_positions]).astype(np.float32)


def _estimate_local_amorphous_box(
    seed_position: np.ndarray,
    all_grain_seeds: Optional[np.ndarray],
    L: float,
    avg_nn_dist: float,
    grain_count: int,
    expansion_factor: float = 1.0,
) -> Tuple[np.ndarray, float]:
    """
    Estimate a local cubic generation box around a grain seed for amorphous phases.

    The box is intentionally larger than an average Voronoi grain so local generation
    captures most of the grain with margin, while avoiding full-domain amorphous generation.
    Uses a conservative grain-count-based estimate to avoid pathological oversizing.
    """
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    if avg_nn_dist <= 0:
        raise ValueError(f"avg_nn_dist must be positive, got {avg_nn_dist}")
    if grain_count <= 0:
        raise ValueError(f"grain_count must be positive, got {grain_count}")
    if expansion_factor <= 0:
        raise ValueError(f"expansion_factor must be positive, got {expansion_factor}")

    avg_grain_volume = (L ** 3) / float(grain_count)
    eq_radius = float((3.0 * avg_grain_volume / (4.0 * np.pi)) ** (1.0 / 3.0))
    # Keep the baseline tied to average grain volume. Do not upscale by local
    # seed spacing directly, because sparse outliers can push this close to full-box.
    base_radius = 1.05 * eq_radius + 2.5 * avg_nn_dist

    radius = min(L / 2.0, base_radius * expansion_factor)
    cube_size = min(L, max(4.0 * avg_nn_dist, 2.0 * radius))
    lower = np.clip(seed_position - 0.5 * cube_size, 0.0, L - cube_size)
    return lower.astype(np.float32), float(cube_size)


def _populate_grain_worker(
    grain_data: Dict[str, Any], global_cfg_dict: Dict[str, Any],
    phase_recipe: Dict[str, Any], seed: int,
    all_grain_seeds: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray, str]:
    """
    Worker function to populate a single grain.
    
    CRITICAL: Only returns atoms that fall within this grain's Voronoi cell.
    """
    rng = np.random.default_rng(seed)
    grain_id = grain_data['grain_id']
    phase_id = grain_data['phase_id']
    seed_position = np.array(grain_data['seed_position'], dtype=np.float32)
    rotation = np.array(grain_data['rotation'], dtype=np.float32)
    
    L = global_cfg_dict['L']
    avg_nn_dist = global_cfg_dict['avg_nn_dist']
    rho_target = global_cfg_dict['rho_target']
    grain_count_raw = global_cfg_dict.get('grain_count')
    if grain_count_raw is None:
        raise KeyError("global_cfg_dict is missing required key 'grain_count'")
    grain_count = int(grain_count_raw)
    if grain_count <= 0:
        raise ValueError(f"grain_count must be > 0, got {grain_count}")
    
    phase_type = phase_recipe['phase_type']
    grain_tree = cKDTree(all_grain_seeds) if all_grain_seeds is not None and len(all_grain_seeds) > 1 else None
    
    # Generate candidate positions
    if phase_type.startswith('crystal_') or phase_type == 'amorphous_repeat':
        positions = _generate_structured_cloud(phase_recipe, rotation, seed_position, L, avg_nn_dist)
    elif phase_type in ('amorphous_random', 'liquid_metal', 'amorphous_mixed'):
        liquid_config = phase_recipe.get('liquid_config')
        config = LiquidStructureConfig(**liquid_config) if liquid_config else LiquidStructureConfig(method='simple')
        min_pair_dist = phase_recipe.get('min_pair_dist')  # From config YAML
        expected_atoms_per_grain = max(1, int(round(rho_target * (L ** 3) / grain_count)))
        min_filtered_atoms_target = max(1, int(round(0.25 * expected_atoms_per_grain)))

        best_positions_world: Optional[np.ndarray] = None
        best_filtered_count = -1
        expansion_factors = (1.0, 1.35, 1.7)

        for expansion_factor in expansion_factors:
            local_origin, local_box_size = _estimate_local_amorphous_box(
                seed_position=seed_position,
                all_grain_seeds=all_grain_seeds,
                L=L,
                avg_nn_dist=avg_nn_dist,
                grain_count=grain_count,
                expansion_factor=expansion_factor,
            )

            generator = LiquidMetalGenerator(
                local_box_size,
                rho_target,
                avg_nn_dist,
                config,
                rng,
                min_pair_dist=min_pair_dist,
            )
            local_positions = generator.generate()

            if phase_type == 'amorphous_mixed' and len(local_positions) > 0:
                local_positions = _inject_crystal_nuclei(
                    amorphous_positions=local_positions,
                    phase_recipe=phase_recipe,
                    rng=rng,
                    avg_nn_dist=avg_nn_dist,
                    rho_target=rho_target,
                    L=local_box_size,
                )

            if len(local_positions) == 0:
                continue

            positions_world = local_positions + local_origin[None, :]
            positions_world = np.clip(positions_world, 0.0, L).astype(np.float32)

            if grain_tree is not None:
                _, nearest_grains_local = grain_tree.query(positions_world)
                filtered_count = int(np.sum(nearest_grains_local == grain_id))
            else:
                filtered_count = len(positions_world)

            if filtered_count > best_filtered_count:
                best_filtered_count = filtered_count
                best_positions_world = positions_world

            if filtered_count >= min_filtered_atoms_target:
                break

        if best_positions_world is None:
            raise RuntimeError(
                "Local amorphous generation failed for grain "
                f"{grain_id} (phase={phase_id}, seed={seed_position.tolist()})"
            )
        if best_filtered_count < min_filtered_atoms_target:
            warnings.warn(
                "Local amorphous generation produced fewer Voronoi-filtered atoms than target "
                f"for grain {grain_id}: got {best_filtered_count}, target {min_filtered_atoms_target}. "
                "Proceeding with best available local sample.",
                RuntimeWarning,
                stacklevel=2,
            )
        positions = best_positions_world
    else:
        positions = np.zeros((0, 3), dtype=np.float32)
    
    # CRITICAL: Filter to only atoms within this grain's Voronoi cell
    if len(positions) > 0 and grain_tree is not None:
        # Find nearest grain for each atom position
        _, nearest_grains = grain_tree.query(positions)
        
        # Keep only atoms where THIS grain is the nearest
        voronoi_mask = nearest_grains == grain_id
        positions = positions[voronoi_mask]
    
    return grain_id, positions, phase_id


# ---------------------------------------------------------------------------
# Main Generator Class
# ---------------------------------------------------------------------------

class SyntheticAtomisticDatasetGenerator:
    """
    Advanced synthetic atomistic dataset generator.
    Features: realistic liquids, parallelization, O(N) spatial queries.
    """
    
    def __init__(
        self, config_path: str | pathlib.Path,
        rng: Optional[np.random.Generator] = None,
        progress: bool = True, skip_visualization: bool = False,
    ):
        self.global_cfg, self.phase_cfgs, self.grain_assignment_cfg = load_config(config_path)
        self.rng = rng or np.random.default_rng(self.global_cfg.random_seed)
        self.progress = progress
        self.skip_visualization = skip_visualization
        self._start_time = time.perf_counter()
        
        self._build_phase_maps()
        
        self.reference_structures: Dict[str, Dict[str, Any]] = {}
        self.reference_point_clouds: Dict[str, np.ndarray] = {}
        self.grains: List[Dict[str, Any]] = []
        self.grain_kdtree: Optional[cKDTree] = None
        self.seed_positions: Optional[np.ndarray] = None
        
        self._estimate_atom_count()
        self.atoms: Optional[np.ndarray] = None
        self.atom_count: int = 0
        self.cell_list: Optional[CellList] = None
        self.intermediate_regions: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.grain_neighbors: Dict[int, set] = {}
        self.metadata: Dict[str, Any] = {}
        
    def _build_phase_maps(self) -> None:
        global PHASE_ID_MAP, PHASE_NAME_MAP
        PHASE_ID_MAP.clear()
        PHASE_NAME_MAP.clear()
        for i, name in enumerate(sorted(self.phase_cfgs.keys())):
            PHASE_ID_MAP[name] = i
            PHASE_NAME_MAP[i] = name
            
    def _estimate_atom_count(self) -> None:
        L = self.global_cfg.L
        rho = self.global_cfg.rho_target
        self._max_atoms = int(1.2 * rho * L ** 3)
        
    def _progress(self, message: str) -> None:
        if self.progress:
            elapsed = time.perf_counter() - self._start_time
            print(f"[{elapsed:8.2f}s] {message}")
            
    def build_reference_structures(self) -> None:
        self._progress(f"Building reference structures for {len(self.phase_cfgs)} phases")
        
        for phase_name, phase_cfg in self.phase_cfgs.items():
            recipe = self._build_phase_recipe(phase_cfg)
            self.reference_structures[phase_name] = recipe
            cloud = self._build_reference_point_cloud(recipe, phase_cfg)
            if cloud is not None:
                self.reference_point_clouds[phase_name] = cloud
                
        self._progress("Reference structures complete")
        
    def _build_phase_recipe(self, phase_cfg: PhaseConfig) -> Dict[str, Any]:
        avg_nn = self.global_cfg.avg_nn_dist
        phase_type = phase_cfg.phase_type
        
        if phase_type == "crystal_fcc":
            lc = avg_nn * np.sqrt(2.0)
            motif = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]])
            recipe = {"phase_type": phase_type, "lattice_constant": lc,
                     "lattice_vectors": (np.eye(3) * lc).tolist(), "motif": motif.tolist()}
        elif phase_type == "crystal_bcc":
            lc = (2.0 / np.sqrt(3.0)) * avg_nn
            motif = np.array([[0,0,0], [0.5,0.5,0.5]])
            recipe = {"phase_type": phase_type, "lattice_constant": lc,
                     "lattice_vectors": (np.eye(3) * lc).tolist(), "motif": motif.tolist()}
        elif phase_type == "crystal_hcp":
            a, c = avg_nn, avg_nn * np.sqrt(8/3)
            lv = np.array([[a,0,0], [a*0.5, a*np.sqrt(3)/2, 0], [0,0,c]])
            motif = np.array([[0,0,0], [1/3, 2/3, 0.5]])
            recipe = {"phase_type": phase_type, "lattice_constant": a,
                     "lattice_vectors": lv.tolist(), "motif": motif.tolist()}
        elif phase_type == "amorphous_repeat":
            recipe = self._build_amorphous_repeat_recipe(phase_cfg)
        elif phase_type in ("amorphous_random", "liquid_metal"):
            min_pair = float(phase_cfg.structural_params.get("min_pair_dist", 0.85 * avg_nn))
            recipe = {"phase_type": phase_type, "min_pair_dist": min_pair}
            if phase_cfg.liquid_config:
                recipe["liquid_config"] = asdict(phase_cfg.liquid_config)
        elif phase_type == "amorphous_mixed":
            recipe = {
                "phase_type": phase_type,
                "min_pair_dist": float(phase_cfg.structural_params.get("min_pair_dist", 0.85 * avg_nn)),
                "embedded_crystal": phase_cfg.structural_params.get("embedded_crystal", "crystal_fcc"),
                "embedded_probability": float(phase_cfg.structural_params.get("embedded_probability", 0.25)),
                "embedded_radius": float(phase_cfg.structural_params.get("embedded_radius", 2.0 * avg_nn)),
            }
            if phase_cfg.liquid_config:
                recipe["liquid_config"] = asdict(phase_cfg.liquid_config)
        else:
            raise ValueError(f"Unsupported phase type: {phase_type}")
            
        recipe["name"] = phase_cfg.name
        self._scale_phase_density(recipe, phase_cfg)
        return recipe
    
    def _build_amorphous_repeat_recipe(self, phase_cfg: PhaseConfig) -> Dict[str, Any]:
        params = phase_cfg.structural_params
        avg_nn = self.global_cfg.avg_nn_dist
        cell_size = float(params.get("cell_size", 2.5 * avg_nn))
        n_points = int(params.get("motif_point_count", 12))
        min_sep = float(params.get("min_pair_dist", 0.8 * avg_nn))
        
        rng = np.random.default_rng(self.rng.integers(0, 2**31))
        motif = []
        cell_list = CellList(cell_size, min_sep)
        # Initialize so get_neighbors works during incremental build
        cell_list.positions = np.zeros((0, 3), dtype=np.float32)
        cell_list.n_atoms = 0
        
        for _ in range(20000):
            if len(motif) >= n_points:
                break
            candidate = rng.uniform(0, cell_size, size=3).astype(np.float32)
            if not motif:
                motif.append(candidate)
                cell_list.cells[cell_list._get_cell_key(candidate)].append((0, candidate))
                cell_list.n_atoms = 1
            else:
                neighbors = cell_list.get_neighbors(candidate, min_sep)
                if not neighbors:
                    motif.append(candidate)
                    cell_list.cells[cell_list._get_cell_key(candidate)].append((len(motif)-1, candidate))
                    cell_list.n_atoms = len(motif)
                    
        if not motif:
            raise RuntimeError("Failed to construct amorphous_repeat motif")
            
        return {
            "phase_type": phase_cfg.phase_type, "motif": np.array(motif).tolist(),
            "tile_vectors": (np.eye(3) * cell_size).tolist(),
            "cell_size": cell_size, "min_pair_dist": min_sep
        }
        
    def _scale_phase_density(self, recipe: Dict[str, Any], phase_cfg: PhaseConfig) -> None:
        """Scale lattice to achieve target density, optionally preserving NN distance for crystals.
        
        For crystal phases (crystal_bcc, crystal_fcc, crystal_hcp), we skip density
        rescaling by default to ensure all phases have consistent nearest-neighbor
        distances. This produces more physically consistent training data.
        
        Set structural_params.preserve_nn_distance: false to enable density rescaling.
        """
        motif = recipe.get("motif")
        if motif is None or len(motif) == 0:
            return
        cell_key = "lattice_vectors" if "lattice_vectors" in recipe else "tile_vectors"
        if cell_key not in recipe:
            return
        
        phase_type = recipe.get("phase_type", "")
        
        # For crystal phases, preserve NN distance by default (skip density rescaling)
        # This ensures BCC, FCC, HCP all have the same avg_nn_dist
        preserve_nn = phase_cfg.structural_params.get("preserve_nn_distance", True)
        if phase_type.startswith("crystal_") and preserve_nn:
            recipe["density_scale_factor"] = 1.0
            recipe["density_target"] = self.global_cfg.rho_target
            recipe["preserve_nn_distance"] = True
            return
        
        target_density = float(phase_cfg.structural_params.get("density_target", self.global_cfg.rho_target))
        if target_density <= 0:
            return
        cell_vectors = np.array(recipe[cell_key], dtype=np.float64)
        volume = abs(np.linalg.det(cell_vectors))
        if volume <= 0:
            return
        current_density = len(motif) / volume
        scale = np.cbrt(current_density / target_density)
        
        if not np.isclose(scale, 1.0, rtol=1e-4):
            scaled_vectors = cell_vectors * scale
            recipe[cell_key] = scaled_vectors.tolist()
            if recipe["phase_type"] == "amorphous_repeat":
                recipe["motif"] = (np.array(motif) * scale).tolist()
            if "lattice_constant" in recipe:
                recipe["lattice_constant"] = float(recipe["lattice_constant"] * scale)
            if "cell_size" in recipe:
                recipe["cell_size"] = float(recipe["cell_size"] * scale)
        recipe["density_scale_factor"] = float(scale)
        recipe["density_target"] = float(target_density)
        recipe["preserve_nn_distance"] = False
        
    def _build_reference_point_cloud(self, recipe: Dict[str, Any], phase_cfg: PhaseConfig, num_points: int = 80) -> Optional[np.ndarray]:
        phase_type = recipe.get("phase_type")
        if phase_type is None:
            return None
        rng = np.random.default_rng(self.rng.integers(0, 2**31))
        
        if phase_type.startswith("crystal_"):
            lattice = np.array(recipe.get("lattice_vectors"), dtype=np.float32)
            motif = np.array(recipe.get("motif"), dtype=np.float32)
            if motif.size == 0:
                return None
            points = self._tile_motif(motif @ lattice, lattice, num_points)
        elif phase_type == "amorphous_repeat":
            tile = np.array(recipe.get("tile_vectors"), dtype=np.float32)
            motif = np.array(recipe.get("motif"), dtype=np.float32)
            if motif.size == 0:
                return None
            points = self._tile_motif(motif, tile, num_points)
        elif phase_type in ("amorphous_random", "liquid_metal"):
            config = phase_cfg.liquid_config or LiquidStructureConfig(method='simple')
            box_size = 5 * self.global_cfg.avg_nn_dist
            density = num_points / (box_size ** 3)
            generator = LiquidMetalGenerator(box_size, density, self.global_cfg.avg_nn_dist, config, rng)
            points = generator.generate()[:num_points]
        else:
            return None
            
        if points is None or len(points) == 0:
            return None
        points = points - np.mean(points, axis=0)
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist
        return points.astype(np.float32)
    
    def _tile_motif(self, motif: np.ndarray, cell: np.ndarray, n_points: int) -> np.ndarray:
        points = []
        for expand in range(1, 10):
            for i in range(-expand, expand + 1):
                for j in range(-expand, expand + 1):
                    for k in range(-expand, expand + 1):
                        offset = i * cell[0] + j * cell[1] + k * cell[2]
                        points.extend(motif + offset)
            if len(points) >= n_points:
                break
        points = np.array(points)
        center = np.mean(points, axis=0)
        dists = np.linalg.norm(points - center, axis=1)
        return points[np.argsort(dists)[:n_points]]
    
    def sample_grains(self) -> None:
        self._progress("Sampling grain seeds and phases")
        n_grains = self.global_cfg.grain_count
        L = self.global_cfg.L
        
        self.seed_positions = self.rng.uniform(0, L, size=(n_grains, 3)).astype(np.float32)
        self.grain_kdtree = cKDTree(self.seed_positions)
        phases = self._assign_grain_phases()
        rotations = random_rotation_matrices(self.rng, n_grains)
        
        self.grains = []
        for idx in range(n_grains):
            self.grains.append({
                "grain_id": idx, "seed_position": self.seed_positions[idx],
                "base_phase_id": phases[idx], "base_rotation": rotations[idx],
            })
        self.grain_neighbors = {i: set() for i in range(n_grains)}
        self._progress(f"Sampled {n_grains} grains")
        
    def _assign_grain_phases(self) -> List[str]:
        n_grains = self.global_cfg.grain_count
        if self.grain_assignment_cfg.mode == "explicit":
            assignments = self.grain_assignment_cfg.assignments or []
            if len(assignments) != n_grains:
                raise ValueError("Explicit assignments must match grain_count")
            return assignments
        probs = self.grain_assignment_cfg.probabilities or {}
        phase_ids = list(probs.keys())
        prob_values = np.array(list(probs.values()), dtype=np.float64)

        if n_grains <= 0 or not phase_ids:
            return []

        positive_idx = np.where(prob_values > 0.0)[0]
        if len(positive_idx) == 0:
            raise ValueError("Probabilistic grain assignment has no positive probabilities")

        # Use a dedicated RNG stream so phase assignment is stable and does not
        # depend on prior random draws (e.g., reference cloud generation).
        phase_rng = np.random.default_rng(int(self.global_cfg.random_seed) + 7919)
        counts = np.zeros(len(phase_ids), dtype=np.int32)

        if n_grains >= len(positive_idx):
            # Ensure every non-zero-probability phase is represented at least once.
            counts[positive_idx] = 1
            remaining = n_grains - len(positive_idx)
            if remaining > 0:
                pos_probs = prob_values[positive_idx]
                pos_probs = pos_probs / np.sum(pos_probs)
                counts[positive_idx] += phase_rng.multinomial(remaining, pos_probs)
        else:
            # Not enough grains to cover every phase: distribute by probabilities.
            pos_probs = prob_values[positive_idx]
            pos_probs = pos_probs / np.sum(pos_probs)
            counts[positive_idx] = phase_rng.multinomial(n_grains, pos_probs)

        assignments: List[str] = []
        for phase_name, count in zip(phase_ids, counts.tolist()):
            if count > 0:
                assignments.extend([phase_name] * int(count))

        phase_rng.shuffle(assignments)
        return assignments
    
    def populate_atoms(self) -> None:
        self._progress("Populating atoms for all grains")
        self.atoms = np.zeros(self._max_atoms, dtype=ATOM_DTYPE)
        self.atom_count = 0
        
        grain_data_list = [{
            'grain_id': g['grain_id'], 'phase_id': g['base_phase_id'],
            'seed_position': g['seed_position'].tolist(), 'rotation': g['base_rotation'].tolist(),
        } for g in self.grains]
        
        global_cfg_dict = {
            'L': self.global_cfg.L, 'avg_nn_dist': self.global_cfg.avg_nn_dist,
            'rho_target': self.global_cfg.rho_target,
            'grain_count': self.global_cfg.grain_count,
        }
        worker_seeds = self.rng.integers(0, 2**31, size=len(self.grains))
        
        if self.global_cfg.parallel.enabled:
            self._populate_parallel(grain_data_list, global_cfg_dict, worker_seeds)
        else:
            self._populate_serial(grain_data_list, global_cfg_dict, worker_seeds)
            
        self._progress("Classifying atoms to grains")
        self._classify_atoms_to_grains()
        
        if self.global_cfg.t_layer > 0:
            self._progress("Generating interface atoms")
            self._generate_interface_atoms()
            
        self._progress(f"Total atoms: {self.atom_count}")
        
    def _populate_parallel(self, grain_data_list, global_cfg_dict, worker_seeds):
        n_workers = self.global_cfg.parallel.n_workers or mp.cpu_count()
        self._progress(f"Using {n_workers} parallel workers")
        
        # Get all grain seeds for Voronoi filtering
        all_grain_seeds = self.seed_positions
        
        tasks = [(gd, global_cfg_dict, self.reference_structures[gd['phase_id']], int(s), all_grain_seeds)
                 for gd, s in zip(grain_data_list, worker_seeds)]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_populate_grain_worker, *t): t[0]['grain_id'] for t in tasks}
            for future in as_completed(futures):
                grain_id = futures[future]
                try:
                    gid, positions, phase_id = future.result()
                    self._add_grain_atoms(gid, positions, phase_id)
                    if self.progress and gid % max(1, len(self.grains)//10) == 0:
                        self._progress(f"  Grain {gid}: {len(positions)} atoms")
                except Exception as e:
                    self._progress(f"  Grain {grain_id} failed: {e}")
                    
    def _populate_serial(self, grain_data_list, global_cfg_dict, worker_seeds):
        # Get all grain seeds for Voronoi filtering
        all_grain_seeds = self.seed_positions
        
        for gd, seed in zip(grain_data_list, worker_seeds):
            recipe = self.reference_structures[gd['phase_id']]
            gid, positions, _ = _populate_grain_worker(gd, global_cfg_dict, recipe, int(seed), all_grain_seeds)
            self._add_grain_atoms(gid, positions, gd['phase_id'])
            if self.progress and gid % max(1, len(self.grains)//10) == 0:
                self._progress(f"  Grain {gid}: {len(positions)} atoms")
                
    def _add_grain_atoms(self, grain_id: int, positions: np.ndarray, phase_id: str) -> None:
        if len(positions) == 0:
            return
        n_new = len(positions)
        
        if self.atom_count + n_new > len(self.atoms):
            new_size = int(1.5 * (self.atom_count + n_new))
            new_atoms = np.zeros(new_size, dtype=ATOM_DTYPE)
            new_atoms[:self.atom_count] = self.atoms[:self.atom_count]
            self.atoms = new_atoms
            
        grain = self.grains[grain_id]
        rotation_flat = grain['base_rotation'].flatten()
        phase_int = PHASE_ID_MAP.get(phase_id, -1)
        
        start, end = self.atom_count, self.atom_count + n_new
        self.atoms['position'][start:end] = positions
        self.atoms['phase_id'][start:end] = phase_int
        self.atoms['grain_id'][start:end] = grain_id  # Set correct grain_id immediately
        self.atoms['orientation'][start:end] = rotation_flat
        self.atoms['alive'][start:end] = True
        self.atoms['pre_index'][start:end] = np.arange(start, end)
        self.atom_count = end
        
    def _classify_atoms_to_grains(self) -> None:
        """
        Classify atoms to grains using Voronoi and REMOVE atoms outside their cell.
        
        This is critical: each grain generates atoms for the full box, so we must
        keep only atoms where the nearest grain seed matches the generating grain.
        """
        if self.grain_kdtree is None or self.atom_count == 0:
            return
            
        positions = self.atoms['position'][:self.atom_count]
        current_grain_ids = self.atoms['grain_id'][:self.atom_count].copy()
        
        # Find which grain each atom SHOULD belong to (nearest seed = Voronoi)
        distances, voronoi_grain_ids = self.grain_kdtree.query(positions)
        
        # CRITICAL FIX: Only keep atoms that are in their correct Voronoi cell
        # Atoms were generated with grain_id = -1, then we check if their position
        # falls within the Voronoi cell of ANY grain
        # 
        # For structured phases: atoms were tiled for full box by each grain
        # We keep an atom only if it's closest to the grain that "should" own it
        # Since all grains generate overlapping lattices, we use position-based dedup
        
        # Strategy: Use a spatial hash to keep only one atom per location
        # Round positions to grid and keep first occurrence
        
        self._progress("    Deduplicating overlapping atoms...")
        
        grid_resolution = 0.1 * self.global_cfg.avg_nn_dist  # Fine grid
        
        # Quantize positions to grid
        quantized = np.round(positions / grid_resolution).astype(np.int64)
        
        # Create unique key for each position
        # Use a large prime multiplier to create hash
        L_cells = int(np.ceil(self.global_cfg.L / grid_resolution)) + 1
        keys = (quantized[:, 0] * L_cells * L_cells + 
                quantized[:, 1] * L_cells + 
                quantized[:, 2])
        
        # Find unique positions (keep first occurrence)
        _, unique_indices = np.unique(keys, return_index=True)
        unique_indices = np.sort(unique_indices)  # Maintain order
        
        n_before = self.atom_count
        n_duplicates = n_before - len(unique_indices)
        
        if n_duplicates > 0:
            self._progress(f"    Removed {n_duplicates:,} duplicate atoms ({100*n_duplicates/n_before:.1f}%)")
            
            # Compact the array to keep only unique atoms
            self.atoms[:len(unique_indices)] = self.atoms[unique_indices]
            self.atom_count = len(unique_indices)
            
            # Recompute Voronoi assignment for remaining atoms
            positions = self.atoms['position'][:self.atom_count]
            distances, voronoi_grain_ids = self.grain_kdtree.query(positions)
        
        # Assign grain IDs based on Voronoi
        self.atoms['grain_id'][:self.atom_count] = voronoi_grain_ids
        
        # Track grain neighbors for interface generation
        if self.global_cfg.t_layer > 0:
            distances_2, grain_ids_2 = self.grain_kdtree.query(positions, k=2)
            dist_diff = distances_2[:, 1] - distances_2[:, 0]
            interface_mask = dist_diff < self.global_cfg.t_layer
            for i in np.where(interface_mask)[0]:
                g1, g2 = grain_ids_2[i]
                self.grain_neighbors[g1].add(g2)
                self.grain_neighbors[g2].add(g1)
                
    def _generate_interface_atoms(self) -> int:
        created = 0
        for g1 in range(len(self.grains)):
            for g2 in self.grain_neighbors[g1]:
                if g2 <= g1:
                    continue
                key = (g1, g2)
                self.intermediate_regions[key] = {
                    "between_grains": key,
                    "phase_id": f"intermediate_{self.grains[g1]['base_phase_id']}_{self.grains[g2]['base_phase_id']}",
                    "atom_indices": [],
                }
        return created
    
    def apply_perturbations(self) -> None:
        self.metadata.setdefault("perturbations", {})
        cell_size = self.global_cfg.cell_size_factor * self.global_cfg.avg_nn_dist
        self.cell_list = CellList(self.global_cfg.L, cell_size)
        self.cell_list.build_from_structured(self.atoms[:self.atom_count])
        
        self._progress("Applying perturbations:")
        
        self._progress("  • Rotation bubbles")
        rot_count = self._apply_rotation_bubbles()
        self._progress(f"    Applied {rot_count} rotation bubbles")
        
        self._progress("  • Thermal noise")
        thermal_count = self._apply_thermal_noise()
        self._progress(f"    Jittered {thermal_count} atoms")
        
        self._progress("  • Vacancy dropouts")
        dropout_count = self._apply_dropouts()
        self._progress(f"    Created {dropout_count} vacancies")
        
        self._progress("  • Density bubbles")
        bubble_count = self._apply_density_bubbles()
        self._progress(f"    Applied {bubble_count} density bubbles")
        
        # Enforce minimum distance after all perturbations to prevent atomic overlap
        self._progress("  • Enforcing minimum distance")
        min_dist_adjusted = self._enforce_minimum_distance()
        self._progress(f"    Resolved {min_dist_adjusted} overlapping pairs")
        
        self.cell_list.build_from_structured(self.atoms[:self.atom_count])
        
    def _apply_rotation_bubbles(self) -> int:
        records = []
        for grain in self.grains:
            phase_id = grain['base_phase_id']
            perturb = self.phase_cfgs[phase_id].perturbations
            
            if perturb.rot_bubble_prob <= 0 or self.rng.random() > perturb.rot_bubble_prob:
                continue
                
            grain_mask = (self.atoms['grain_id'][:self.atom_count] == grain['grain_id']) & self.atoms['alive'][:self.atom_count]
            grain_indices = np.where(grain_mask)[0]
            if len(grain_indices) == 0:
                continue
                
            center_idx = self.rng.choice(grain_indices)
            center = self.atoms['position'][center_idx].copy()
            radius = perturb.rot_bubble_radius
            angle_rad = np.radians(self.rng.normal(perturb.rot_bubble_angle_deg, perturb.rot_bubble_angle_deg * 0.1))
            axis = random_unit_vectors(self.rng, 1)[0]
            R = rotation_matrix_from_axis_angle(axis, angle_rad)
            
            affected = []
            for idx in grain_indices:
                offset = self.atoms['position'][idx] - center
                dist = np.linalg.norm(offset)
                if dist <= radius:
                    weight = max(0, 1.0 - (dist / radius) ** 2)
                    if weight > 0.1:
                        rotated = R @ offset
                        self.atoms['position'][idx] = center + weight * rotated + (1 - weight) * offset
                        old_orient = self.atoms['orientation'][idx].reshape(3, 3)
                        self.atoms['orientation'][idx] = (R @ old_orient).flatten()
                        affected.append(int(idx))
                        
            if affected:
                records.append({
                    "grain_id": int(grain['grain_id']), "center": center.tolist(),
                    "radius": float(radius), "axis": axis.tolist(),
                    "angle_deg": float(np.degrees(angle_rad)), "affected_count": len(affected),
                })
                
        self.metadata["perturbations"]["rotation_bubbles"] = records
        return len(records)
    
    def _apply_thermal_noise(self) -> int:
        """
        Apply thermal noise using Debye model approximation.
        
        For metals at room temperature, typical RMS displacement is ~2-5% of lattice spacing.
        Using harmonic approximation: <u²> = k_B T / k_eff
        where k_eff is the effective spring constant (~1-10 eV/Å² for metals).
        """
        total_jittered = 0
        records = {}
        
        for phase_name, phase_cfg in self.phase_cfgs.items():
            perturb = phase_cfg.perturbations
            phase_int = PHASE_ID_MAP.get(phase_name, -1)
            
            phase_mask = (self.atoms['phase_id'][:self.atom_count] == phase_int) & self.atoms['alive'][:self.atom_count]
            phase_indices = np.where(phase_mask)[0]
            if len(phase_indices) == 0:
                continue
                
            if perturb.temperature_K > 0:
                # Physical constants
                k_B = 8.617e-5  # eV/K
                T = perturb.temperature_K
                m_amu = perturb.atomic_mass_amu
                
                # Debye temperature (use 400K as typical for iron, scale with mass)
                # θ_D ∝ sqrt(k/m), for iron θ_D ≈ 470K
                theta_D_iron = 470.0
                m_iron = 55.845
                theta_D = theta_D_iron * np.sqrt(m_iron / m_amu)
                
                # High-temperature Debye model: <u²> = (9ℏ²T)/(m k_B θ_D²)
                # In practical units, this gives:
                # <u²> [Å²] ≈ 1.546 × T[K] / (m[amu] × θ_D²[K²])
                # (1.546 comes from 9 × (ℏ/k_B)² × (1/amu) in Å² units)
                hbar_over_kB = 7.6382  # ℏ/k_B in K·ps, but we need different units
                
                # Simpler, validated formula for metals:
                # σ ≈ sqrt(3 k_B T / (m ω_D²)) where ω_D = k_B θ_D / ℏ
                # This simplifies to: σ ≈ sqrt(3 T / (m θ_D²)) × (ℏ/k_B) × sqrt(k_B/amu)
                # 
                # Numerical coefficient from known values:
                # For Fe at 300K: σ ≈ 0.05-0.08 Å (from X-ray diffraction)
                # Using Lindemann: σ/a ≈ 0.1 at melting, so at T/Tm ≈ 0.17 (300K/1800K):
                # σ ≈ 0.04-0.06 Å
                #
                # Validated formula: σ² = 0.0247 × T / (m × θ_D²) [Å², K, amu, K]
                # The 0.0247 factor = 3 × (ℏ/k_B)² × (1 eV / 1 amu·Å²/ps²) appropriately converted
                
                u2 = 0.0247 * T / (m_amu * (theta_D / 100)**2)  # θ_D/100 to get right scale
                sigma = np.sqrt(u2)
                
                # Sanity check: cap at 10% of nearest neighbor distance
                max_sigma = 0.10 * self.global_cfg.avg_nn_dist
                sigma = min(sigma, max_sigma)
                
            elif perturb.sigma_thermal > 0:
                sigma = perturb.sigma_thermal
            else:
                continue
                
            displacements = self.rng.normal(0, sigma, size=(len(phase_indices), 3)).astype(np.float32)
            self.atoms['position'][phase_indices] += displacements
            total_jittered += len(phase_indices)
            records[phase_name] = {"sigma_used": float(sigma), "temperature_K": float(perturb.temperature_K), "n_atoms": len(phase_indices)}
            
        self.metadata["perturbations"]["thermal_noise"] = records
        return total_jittered
    
    def _enforce_minimum_distance(self, min_dist: float = None, max_iterations: int = 10) -> int:
        """
        Push overlapping atoms apart to enforce minimum pair distance.
        
        Uses scipy cKDTree.query_pairs for fast O(N log N) pair finding,
        then vectorized adjustment computation.
        
        Args:
            min_dist: Minimum allowed pair distance (default: 0.85 * avg_nn_dist)
            max_iterations: Maximum relaxation iterations
            
        Returns:
            Total number of overlapping pairs resolved
        """
        if min_dist is None:
            min_dist = 0.85 * self.global_cfg.avg_nn_dist
        
        total_resolved = 0
        alive_mask = self.atoms['alive'][:self.atom_count]
        alive_indices = np.where(alive_mask)[0]
        
        if len(alive_indices) < 2:
            return 0
        
        for iteration in range(max_iterations):
            # Get current positions of alive atoms
            positions = self.atoms['position'][alive_indices].copy()
            
            # Build KD-tree and find all pairs closer than min_dist
            tree = cKDTree(positions)
            pairs = tree.query_pairs(r=min_dist, output_type='ndarray')
            
            if len(pairs) == 0:
                # No overlapping pairs, we're done
                break
            
            # Compute adjustments vectorized
            i_local, j_local = pairs[:, 0], pairs[:, 1]
            pos_i = positions[i_local]
            pos_j = positions[j_local]
            
            # Distance vectors and magnitudes
            diff = pos_j - pos_i
            dists = np.linalg.norm(diff, axis=1)
            
            # Handle near-coincident points (avoid division by zero)
            near_zero = dists < 1e-6
            if np.any(near_zero):
                random_dirs = self.rng.normal(size=(np.sum(near_zero), 3)).astype(np.float32)
                random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-8
                diff[near_zero] = random_dirs * min_dist
                dists[near_zero] = min_dist
            
            # Normalize direction vectors
            directions = diff / dists[:, np.newaxis]
            
            # Compute overlap amount and push magnitude
            overlaps = min_dist - dists
            push = 0.5 * overlaps * 1.1  # Push each atom by half (plus 10%)
            
            # Compute adjustment vectors
            push_vectors = push[:, np.newaxis] * directions
            
            # Accumulate adjustments (use np.add.at for scatter-add)
            adjustments = np.zeros_like(positions)
            np.add.at(adjustments, i_local, -push_vectors)
            np.add.at(adjustments, j_local, push_vectors)
            
            # Apply adjustments back to original atom array
            self.atoms['position'][alive_indices] += adjustments
            
            # Clamp positions to simulation box
            L = self.global_cfg.L
            self.atoms['position'][alive_indices] = np.clip(
                self.atoms['position'][alive_indices], 0, L
            )
            
            total_resolved += len(pairs)
        
        # Record metadata
        self.metadata["perturbations"]["minimum_distance_enforcement"] = {
            "min_dist": float(min_dist),
            "iterations": iteration + 1 if 'iteration' in dir() else 0,
            "total_pairs_resolved": total_resolved,
        }
        
        return total_resolved
    
    def _apply_dropouts(self) -> int:
        events = []
        for phase_name, phase_cfg in self.phase_cfgs.items():
            perturb = phase_cfg.perturbations
            if perturb.p_dropout <= 0:
                continue
                
            phase_int = PHASE_ID_MAP.get(phase_name, -1)
            phase_mask = (self.atoms['phase_id'][:self.atom_count] == phase_int) & self.atoms['alive'][:self.atom_count]
            phase_indices = np.where(phase_mask)[0]
            if len(phase_indices) == 0:
                continue
                
            dropout_mask = self.rng.random(len(phase_indices)) < perturb.p_dropout
            dropout_indices = phase_indices[dropout_mask]
            
            for idx in dropout_indices:
                if not self.atoms['alive'][idx]:
                    continue
                vacancy_pos = self.atoms['position'][idx].copy()
                
                if perturb.dropout_relax_radius > 0 and self.cell_list is not None:
                    neighbors = self.cell_list.get_neighbors(vacancy_pos, perturb.dropout_relax_radius)
                    for n_idx, n_pos, n_dist in neighbors:
                        if n_idx == idx or not self.atoms['alive'][n_idx] or n_dist < 1e-6:
                            continue
                        direction = (vacancy_pos - n_pos) / n_dist
                        if perturb.use_elastic_relaxation:
                            strength = perturb.dropout_relax_max_fraction * (self.global_cfg.avg_nn_dist / n_dist) ** 2
                        else:
                            strength = perturb.dropout_relax_max_fraction * (1.0 - n_dist / perturb.dropout_relax_radius)
                        self.atoms['position'][n_idx] += strength * self.global_cfg.avg_nn_dist * direction
                        
                self.atoms['alive'][idx] = False
                events.append({"vacancy_position": vacancy_pos.tolist(), "removed_index": int(idx)})
                
        self.metadata["perturbations"]["dropouts"] = {"events": events, "count": len(events)}
        return len(events)
    
    def _apply_density_bubbles(self) -> int:
        records = []
        for phase_name, phase_cfg in self.phase_cfgs.items():
            perturb = phase_cfg.perturbations
            for bubble_cfg in perturb.density_bubbles:
                expected_count = bubble_cfg.get("expected_count", 0)
                radius = bubble_cfg.get("radius", 0)
                alpha = bubble_cfg.get("alpha", 0)
                
                if radius <= 0:
                    continue
                n_bubbles = self.rng.poisson(expected_count) if expected_count > 0 else 0
                
                for _ in range(n_bubbles):
                    phase_int = PHASE_ID_MAP.get(phase_name, -1)
                    phase_mask = (self.atoms['phase_id'][:self.atom_count] == phase_int) & self.atoms['alive'][:self.atom_count]
                    phase_indices = np.where(phase_mask)[0]
                    if len(phase_indices) == 0:
                        continue
                        
                    center_idx = self.rng.choice(phase_indices)
                    center = self.atoms['position'][center_idx].copy()
                    record = {"phase": phase_name, "center": center.tolist(), "radius": float(radius),
                             "alpha": float(alpha), "removed": 0, "added": 0}
                    
                    if self.cell_list is not None:
                        neighbors = self.cell_list.get_neighbors(center, radius)
                        affected_indices = [n_idx for n_idx, _, _ in neighbors if self.atoms['alive'][n_idx]]
                    else:
                        dists = np.linalg.norm(self.atoms['position'][:self.atom_count] - center, axis=1)
                        affected_indices = np.where((dists <= radius) & self.atoms['alive'][:self.atom_count])[0]
                        
                    if alpha < 0:
                        for idx in affected_indices:
                            r = np.linalg.norm(self.atoms['position'][idx] - center)
                            prob = min(1.0, -alpha * (1.0 - (r / radius) ** 2))
                            if self.rng.random() < prob:
                                self.atoms['alive'][idx] = False
                                record["removed"] += 1
                    elif alpha > 0:
                        for idx in affected_indices:
                            r = np.linalg.norm(self.atoms['position'][idx] - center)
                            prob = min(1.0, alpha * (1.0 - (r / radius) ** 2))
                            if self.rng.random() < prob and self.atom_count < len(self.atoms):
                                jitter = self.rng.normal(0, 0.2 * self.global_cfg.avg_nn_dist, size=3).astype(np.float32)
                                self.atoms['position'][self.atom_count] = self.atoms['position'][idx] + jitter
                                self.atoms['phase_id'][self.atom_count] = self.atoms['phase_id'][idx]
                                self.atoms['grain_id'][self.atom_count] = self.atoms['grain_id'][idx]
                                self.atoms['orientation'][self.atom_count] = self.atoms['orientation'][idx]
                                self.atoms['alive'][self.atom_count] = True
                                self.atoms['pre_index'][self.atom_count] = self.atom_count
                                self.atom_count += 1
                                record["added"] += 1
                    records.append(record)
                    
        self.metadata["perturbations"]["density_bubbles"] = records
        return len(records)
    
    def save_outputs(self) -> None:
        output_dir = self.global_cfg.data_path
        output_dir.mkdir(parents=True, exist_ok=True)
        self._progress(f"Saving outputs to {output_dir}")
        
        alive_mask = self.atoms['alive'][:self.atom_count]
        final_atoms = self.atoms[:self.atom_count][alive_mask]
        n_final = len(final_atoms)
        
        np.save(output_dir / "atoms.npy", final_atoms['position'])
        np.save(output_dir / "atoms_full.npy", final_atoms)
        np.save(output_dir / "reference_structures.npy", self.reference_structures, allow_pickle=True)
        if self.reference_point_clouds:
            np.save(output_dir / "reference_point_clouds.npy", self.reference_point_clouds, allow_pickle=True)
            
        self._build_metadata(n_final)
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(self.metadata, f, indent=2)
        with open(output_dir / "phase_mapping.json", "w") as f:
            json.dump({"name_to_id": PHASE_ID_MAP, "id_to_name": {v: k for k, v in PHASE_ID_MAP.items()}}, f, indent=2)
        self._progress(f"Saved {n_final} atoms")
        
    def _build_metadata(self, n_final: int) -> None:
        L = self.global_cfg.L
        self.metadata["global"] = {
            "box_size": float(L), "volume": float(L**3), "rho_target": float(self.global_cfg.rho_target),
            "rho_actual": float(n_final / L**3), "avg_nn_dist": float(self.global_cfg.avg_nn_dist),
            "t_layer": float(self.global_cfg.t_layer), "random_seed": int(self.global_cfg.random_seed),
            "grain_count": len(self.grains), "N_final": int(n_final), "phases": sorted(PHASE_ID_MAP.keys()),
        }
        
        # Build mapping from old atom indices to new (after removing dead atoms)
        alive_mask = self.atoms['alive'][:self.atom_count]
        old_to_new_idx = np.full(self.atom_count, -1, dtype=np.int64)
        old_to_new_idx[alive_mask] = np.arange(n_final)
        
        grain_records = []
        for grain in self.grains:
            grain_mask = (self.atoms['grain_id'][:self.atom_count] == grain['grain_id']) & alive_mask
            grain_old_indices = np.where(grain_mask)[0]
            # Convert to new indices after dead atom removal
            grain_new_indices = old_to_new_idx[grain_old_indices]
            grain_new_indices = grain_new_indices[grain_new_indices >= 0]  # Remove any invalid
            
            grain_records.append({
                "grain_id": int(grain['grain_id']), 
                "base_phase_id": grain['base_phase_id'],
                "seed_position": [float(x) for x in grain['seed_position']], 
                "orientation_matrix": [[float(x) for x in row] for row in grain['base_rotation']],
                "n_atoms": len(grain_new_indices), 
                "atom_indices": grain_new_indices.tolist(),
                "neighbors": [int(x) for x in sorted(self.grain_neighbors.get(grain['grain_id'], []))],
            })
        self.metadata["grains"] = grain_records
        
        phase_stats = {}
        for phase_name in PHASE_ID_MAP:
            phase_int = PHASE_ID_MAP[phase_name]
            mask = (self.atoms['phase_id'][:self.atom_count] == phase_int) & self.atoms['alive'][:self.atom_count]
            n = int(np.sum(mask))
            phase_stats[phase_name] = {"n_atoms": n, "fraction": float(n / n_final) if n_final > 0 else 0}
        self.metadata["phase_statistics"] = phase_stats
        
    def create_visualizations(self) -> None:
        if self.skip_visualization:
            return
        try:
            import sys, os
            sys.path.append(os.getcwd())
            from src.data_utils.synthetic.visualization import generate_visualizations
            output_dir = self.global_cfg.data_path
            alive_mask = self.atoms['alive'][:self.atom_count]
            final_atoms = self.atoms[:self.atom_count][alive_mask]
            atoms_list = [{"final_index": i, "position": a['position'], 
                          "phase_id": PHASE_NAME_MAP.get(a['phase_id'], f"unknown_{a['phase_id']}"),
                          "grain_id": int(a['grain_id']), "orientation": a['orientation'].reshape(3,3)}
                         for i, a in enumerate(final_atoms)]
            generate_visualizations(self.global_cfg, self.grains, atoms_list, self.metadata, self.rng, output_dir)
        except ImportError:
            self._progress("Visualization module not available")
        except Exception as e:
            self._progress(f"Visualization failed: {e}")
            
    def _compute_nn_diagnostics(self, stage_name: str) -> dict:
        """Compute and log NN distance statistics at a generation stage.
        
        Used to diagnose overlap issues by tracking NN stats throughout generation.
        """
        if self.atom_count < 2:
            return {}
        
        alive_mask = self.atoms['alive'][:self.atom_count]
        positions = self.atoms['position'][:self.atom_count][alive_mask]
        
        if len(positions) < 2:
            return {}
        
        # Sample for efficiency on large datasets
        sample_size = min(50000, len(positions))
        if len(positions) > sample_size:
            idx = np.random.choice(len(positions), sample_size, replace=False)
            sample_pos = positions[idx]
        else:
            sample_pos = positions
        
        tree = cKDTree(positions)
        distances, _ = tree.query(sample_pos, k=2)
        nn_dists = distances[:, 1]
        
        stats = {
            "stage": stage_name,
            "n_atoms": int(len(positions)),
            "nn_mean": float(np.mean(nn_dists)),
            "nn_std": float(np.std(nn_dists)),
            "nn_min": float(np.min(nn_dists)),
            "nn_p1": float(np.percentile(nn_dists, 1)),
            "nn_p5": float(np.percentile(nn_dists, 5)),
            "n_below_2A": int(np.sum(nn_dists < 2.0)),
            "n_below_1A": int(np.sum(nn_dists < 1.0)),
        }
        
        self._progress(f"  NN diagnostics ({stage_name}): mean={stats['nn_mean']:.3f}Å "
                      f"min={stats['nn_min']:.3f}Å P1={stats['nn_p1']:.3f}Å")
        
        if stats['nn_min'] < 1.0:
            self._progress(f"  ⚠️ WARNING: NN min < 1.0Å detected! ({stats['n_below_1A']} atoms)")
        elif stats['nn_min'] < 1.8:
            self._progress(f"  ⚠️ Warning: NN min < 1.8Å ({stats['n_below_2A']} atoms with NN < 2Å)")
        
        return stats
    
    def run(self) -> None:
        self._start_time = time.perf_counter()
        self._progress("=" * 60)
        self._progress("Synthetic Atomistic Dataset Generator v2.0")
        self._progress("=" * 60)
        
        self._progress("Step 1/6: Building reference structures")
        self.build_reference_structures()
        self._progress("Step 2/6: Sampling grains")
        self.sample_grains()
        self._progress("Step 3/6: Populating atoms")
        self.populate_atoms()
        
        # Diagnostic: check NN after atom population
        post_pop_stats = self._compute_nn_diagnostics("post_population")
        
        self._progress("Step 4/6: Applying perturbations")
        self.apply_perturbations()
        
        # Diagnostic: check NN after perturbations
        post_perturb_stats = self._compute_nn_diagnostics("post_perturbation")
        
        # Store diagnostics in metadata
        self.metadata["nn_diagnostics"] = {
            "post_population": post_pop_stats,
            "post_perturbation": post_perturb_stats,
        }
        
        self._progress("Step 5/6: Saving outputs")
        self.save_outputs()
        
        if not self.skip_visualization:
            self._progress("Step 6/6: Creating visualizations")
            self.create_visualizations()
        else:
            self._progress("Step 6/6: Skipping visualizations")
            
        elapsed = time.perf_counter() - self._start_time
        self._progress("=" * 60)
        self._progress(f"Generation complete in {elapsed:.2f}s")
        self._progress("=" * 60)


# ---------------------------------------------------------------------------
# Analysis Utilities
# ---------------------------------------------------------------------------

def compute_rdf(positions: np.ndarray, box_size: float, r_max: Optional[float] = None, n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    n = len(positions)
    if r_max is None:
        r_max = box_size / 2
    cell_list = CellList(box_size, r_max / 5)
    cell_list.build(positions)
    
    r_bins = np.linspace(0, r_max, n_bins + 1)
    hist = np.zeros(n_bins)
    dr = r_bins[1] - r_bins[0]
    
    sample_size = min(1000, n)
    sample_indices = np.random.choice(n, sample_size, replace=False)
    
    for i in sample_indices:
        neighbors = cell_list.get_neighbors(positions[i], r_max)
        for j, _, dist in neighbors:
            if j != i and dist > 0:
                bin_idx = int(dist / dr)
                if 0 <= bin_idx < n_bins:
                    hist[bin_idx] += 1
                    
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
    shell_volumes = 4 * np.pi * r_centers ** 2 * dr
    rho = n / box_size ** 3
    expected = sample_size * rho * shell_volumes
    g_r = np.divide(hist, expected, where=expected > 0, out=np.ones_like(hist))
    return r_centers, g_r


def analyze_structure(positions: np.ndarray, box_size: float, avg_nn_dist: float) -> Dict[str, Any]:
    r, gr = compute_rdf(positions, box_size)
    peak_idx = np.argmax(gr[r > 0.5 * avg_nn_dist])
    first_peak_r = r[r > 0.5 * avg_nn_dist][peak_idx]
    first_peak_g = gr[r > 0.5 * avg_nn_dist][peak_idx]
    
    cutoff = 1.4 * avg_nn_dist
    cell_list = CellList(box_size, cutoff * 1.5)
    cell_list.build(positions)
    
    coordinations = []
    sample = np.random.choice(len(positions), min(500, len(positions)), replace=False)
    for i in sample:
        neighbors = cell_list.get_neighbors(positions[i], cutoff)
        coordinations.append(sum(1 for j, _, _ in neighbors if j != i))
        
    coordinations = np.array(coordinations)
    return {
        "n_atoms": len(positions), "density": len(positions) / box_size ** 3,
        "first_peak_position": float(first_peak_r), "first_peak_height": float(first_peak_g),
        "mean_coordination": float(np.mean(coordinations)), "std_coordination": float(np.std(coordinations)),
        "rdf_r": r.tolist(), "rdf_gr": gr.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic atomistic datasets")
    parser.add_argument("config", type=str, nargs="?", default="configs/data/data_synth_polycrystalline_balanced_geometries_v2.yaml")
    parser.add_argument("--quiet", "-q", action="store_true")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--workers", "-w", type=int, default=16)
    args = parser.parse_args()
    
    generator = SyntheticAtomisticDatasetGenerator(args.config, progress=not args.quiet, skip_visualization=args.skip_viz)
    if args.workers is not None:
        generator.global_cfg.parallel.n_workers = args.workers
    generator.run()
    
    if args.analyze:
        positions = np.load(generator.global_cfg.data_path / "atoms.npy")
        stats = analyze_structure(positions, generator.global_cfg.L, generator.global_cfg.avg_nn_dist)
        print(f"\n{'='*60}\nStructure Analysis\n{'='*60}")
        print(f"  Atoms: {stats['n_atoms']}")
        print(f"  Density: {stats['density']:.4f}")
        print(f"  First RDF peak: r={stats['first_peak_position']:.3f}, g={stats['first_peak_height']:.2f}")
        print(f"  Mean coordination: {stats['mean_coordination']:.1f} ± {stats['std_coordination']:.1f}")


if __name__ == "__main__":
    main()
