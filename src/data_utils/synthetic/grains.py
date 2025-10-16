"""Grain seeding, spatial indexing, and phase assignment utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple
import math

import numpy as np

from scipy.spatial import cKDTree as _SciPyKDTree, Voronoi


class _KDTree:
    """Lightweight KDTree wrapper with optional SciPy acceleration."""

    def __init__(self, data: np.ndarray) -> None:
        self.data = np.asarray(data, dtype=float)
        self._tree = _SciPyKDTree(self.data)


    def query(self, x: np.ndarray, k: int = 1):
        x = np.asarray(x, dtype=float)
        dist, idx = self._tree.query(x, k=k)

        return dist, idx

    def query_ball_point(self, x: np.ndarray, r: float):
        x = np.asarray(x, dtype=float)
        if self._tree is not None:
            return self._tree.query_ball_point(x, r)
        if x.ndim == 1:
            diff = self.data - x
            dist = np.linalg.norm(diff, axis=1)
            return np.where(dist <= r)[0].tolist()
        indices = []
        for point in x:
            diff = self.data - point
            dist = np.linalg.norm(diff, axis=1)
            indices.append(np.where(dist <= r)[0].tolist())
        return indices

from .config import DatasetConfig, GrainRadiusDistSpec, PhaseName, sample_radius, weighted_choice


@dataclass
class GrainSeed:
    id: int
    center: np.ndarray
    radius: float
    parent_id: int | None


@dataclass
class GrainRegionIndex:
    seeds: List[GrainSeed]
    adjacency: Dict[int, List[int]]
    L: float

    def __post_init__(self) -> None:
        centers = [seed.center for seed in self.seeds]
        self._tree = _KDTree(np.stack(centers)) if centers else None
        self._id_lookup = {seed.id: idx for idx, seed in enumerate(self.seeds)}

    def lookup(self, x: np.ndarray) -> int:
        if self._tree is None:
            raise ValueError("Region index has no seeds.")
        dist, idx = self._tree.query(x, k=1)
        return self.seeds[int(idx)].id

    def boundary_distance(self, x: np.ndarray) -> float:
        if self._tree is None:
            raise ValueError("Region index has no seeds.")
        dist, idx = self._tree.query(x, k=1)
        seed = self.seeds[int(idx)]
        return float(max(seed.radius - dist, 0.0))

    def neighbors(self, grain_id: int) -> List[int]:
        return self.adjacency.get(grain_id, [])

    def get_seed(self, grain_id: int) -> GrainSeed:
        return self.seeds[self._id_lookup[grain_id]]


def _clip_to_box(point: np.ndarray, L: float) -> np.ndarray:
    return np.clip(point, 0.0, L)


def _build_adjacency_from_tree(seeds: List[GrainSeed], k: int = 8) -> Dict[int, List[int]]:
    if not seeds:
        return {}
    centers = np.stack([seed.center for seed in seeds])
    tree = _KDTree(centers)
    adjacency: Dict[int, set[int]] = {seed.id: set() for seed in seeds}
    distances, indices = tree.query(centers, k=min(k + 1, len(seeds)))
    for idx, nbrs in enumerate(indices):
        grain_id = seeds[idx].id
        for nbr_idx in nbrs:
            if nbr_idx == idx:
                continue
            adjacency[grain_id].add(seeds[nbr_idx].id)
    return {gid: sorted(list(nbrs)) for gid, nbrs in adjacency.items()}


def sample_grain_seeds_thomas(cfg: DatasetConfig, rng: np.random.Generator) -> List[GrainSeed]:
    cfg.validate()
    parent_lambda = cfg.thomas_parent_intensity or 0.0
    n_parents = rng.poisson(parent_lambda * (cfg.L ** 3))
    if n_parents == 0:
        n_parents = 1
    parent_centers = rng.uniform(0.0, cfg.L, size=(n_parents, 3))
    sigma = (cfg.thomas_dispersion_sigma or 0.05) * cfg.L
    seeds: List[GrainSeed] = []
    next_id = 0
    radius_spec = cfg.grain_radius_dist or GrainRadiusDistSpec(kind="lognormal", params={"median": 0.1, "gstd": 1.5})

    for parent_id, parent_center in enumerate(parent_centers):
        mean_children = cfg.thomas_child_mean or 1.0
        n_children = max(1, rng.poisson(mean_children))
        displacements = rng.normal(scale=sigma, size=(n_children, 3))
        child_centers = parent_center + displacements
        child_centers = _clip_to_box(child_centers, cfg.L)
        for center in child_centers:
            radius = sample_radius(radius_spec, rng) * cfg.L
            seeds.append(GrainSeed(id=next_id, center=center, radius=radius, parent_id=parent_id))
            next_id += 1
    return seeds


def _finite_voronoi_vertices(vor: Voronoi, L: float) -> None:
    for point in vor.points:
        for axis in range(3):
            if not (0 <= point[axis] <= L):
                raise ValueError("Voronoi seeds must lie within box bounds.")


def sample_grain_seeds_voronoi(cfg: DatasetConfig, rng: np.random.Generator) -> List[GrainSeed]:
    cfg.validate()
    count = cfg.voronoi_seed_count or 100
    seeds_centers = rng.uniform(0.0, cfg.L, size=(count, 3))
    seeds: List[GrainSeed] = []
    radius_spec = cfg.grain_radius_dist or GrainRadiusDistSpec(kind="uniform", params={"low": 0.05, "high": 0.15})
    if Voronoi is not None:
        vor = Voronoi(seeds_centers)
        _finite_voronoi_vertices(vor, cfg.L)
        for idx, center in enumerate(seeds_centers):
            if vor.regions[vor.point_region[idx]]:
                vertex_idx = [v for v in vor.regions[vor.point_region[idx]] if v != -1]
                if vertex_idx:
                    region_vertices = vor.vertices[vertex_idx, :]
                    distances = np.linalg.norm(region_vertices - center, axis=1)
                    radius = np.nanmedian(distances)
                    if not np.isfinite(radius) or radius <= 0:
                        radius = sample_radius(radius_spec, rng) * cfg.L
                else:
                    radius = sample_radius(radius_spec, rng) * cfg.L
            else:
                radius = sample_radius(radius_spec, rng) * cfg.L
            seeds.append(GrainSeed(id=idx, center=center, radius=radius, parent_id=None))
    else:  # pragma: no cover - fallback when SciPy is unavailable
        tree = _KDTree(seeds_centers)
        neighbor_dist, _ = tree.query(seeds_centers, k=min(4, count))
        if np.ndim(neighbor_dist) == 1:
            neighbor_dist = neighbor_dist[:, None]
        median_neighbor = np.median(neighbor_dist[:, 1:], axis=1)  # skip self distance
        for idx, center in enumerate(seeds_centers):
            radius = float(max(median_neighbor[idx], 1e-3))
            seeds.append(GrainSeed(id=idx, center=center, radius=radius, parent_id=None))
    return seeds


def assign_phase_to_grains(
    seeds: List[GrainSeed],
    cfg: DatasetConfig,
    rng: np.random.Generator,
) -> Dict[int, PhaseName]:
    phase_mix = cfg.phase_mix
    phase_names = list(phase_mix.keys())
    weights = np.array(list(phase_mix.values()), dtype=float)
    weights /= weights.sum()
    grain_to_phase: Dict[int, PhaseName] = {}
    if not seeds:
        return grain_to_phase

    if cfg.grain_model == "thomas":
        # assign per parent for clumping
        parent_to_phase: Dict[int, PhaseName] = {}
        for seed in seeds:
            pid = seed.parent_id if seed.parent_id is not None else seed.id
            if pid not in parent_to_phase:
                parent_to_phase[pid] = weighted_choice(phase_names, weights, rng)
            grain_to_phase[seed.id] = parent_to_phase[pid]
    else:
        # initial assignment
        for seed in seeds:
            grain_to_phase[seed.id] = weighted_choice(phase_names, weights, rng)
        # smooth using adjacency majority vote
        adjacency = _build_adjacency_from_tree(seeds)
        beta = 1.5
        for seed in seeds:
            neighbors = adjacency.get(seed.id, [])
            if not neighbors:
                continue
            counts = {phase: 0.0 for phase in phase_names}
            for nbr in neighbors:
                counts[grain_to_phase[nbr]] += 1.0
            neighbor_weights = np.array([math.exp(beta * counts[p]) for p in phase_names])
            grain_to_phase[seed.id] = weighted_choice(phase_names, neighbor_weights, rng)
    return grain_to_phase


def build_grain_regions(seeds: List[GrainSeed], cfg: DatasetConfig) -> GrainRegionIndex:
    adjacency = _build_adjacency_from_tree(seeds)
    return GrainRegionIndex(seeds=seeds, adjacency=adjacency, L=cfg.L)


def compute_phase_autocorrelation(centers: np.ndarray, phases: Iterable[PhaseName]) -> float:
    """Compute a simple Moran's I score over provided samples."""

    centers = np.asarray(centers)
    if centers.ndim != 2 or centers.shape[1] != 3:
        raise ValueError("centers must have shape (N, 3)")
    phase_list = list(phases)
    if len(phase_list) != len(centers):
        raise ValueError("centers and phases must align")
    unique_phases = {p: idx for idx, p in enumerate(sorted(set(phase_list)))}
    values = np.array([unique_phases[p] for p in phase_list], dtype=float)
    tree = cKDTree(centers)
    distances, indices = tree.query(centers, k=min(8, len(centers)))
    weights = np.zeros_like(distances)
    with np.errstate(divide="ignore"):
        weights = 1.0 / (distances + 1e-6)
    weights[:, 0] = 0.0  # exclude self
    W = weights.sum()
    mean_val = values.mean()
    numerator = 0.0
    denominator = ((values - mean_val) ** 2).sum()
    for i in range(len(centers)):
        diff_i = values[i] - mean_val
        for w, j in zip(weights[i], indices[i]):
            if j == i:
                continue
            numerator += w * diff_i * (values[j] - mean_val)
    if denominator == 0 or W == 0:
        return 0.0
    return float((len(centers) / W) * (numerator / denominator))
