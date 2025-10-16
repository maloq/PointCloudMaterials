"""Environment sampling, ground-truth lookup, and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import DatasetConfig, EnvCenterSamplerSpec, NoiseSpec
from .grains import GrainRegionIndex
from .orientation import OrientationField, apply_rotation, matrix_to_quaternion
from .phases import MotifParams, PhaseLibrary, PhasePrototype, instantiate_motif


@dataclass
class EnvironmentGT:
    phase_id: int
    phase_name: str
    grain_id: int
    orientation: np.ndarray
    boundary_distance: float


def _phase_id_map(cfg: DatasetConfig) -> Dict[str, int]:
    names = list(cfg.phase_mix.keys())
    return {name: idx for idx, name in enumerate(names)}


def sample_environment_centers(
    cfg: DatasetConfig,
    region_index: GrainRegionIndex,
    *,
    rng: np.random.Generator,
    N_env: int,
) -> np.ndarray:
    spec: EnvCenterSamplerSpec = cfg.env_center_sampler
    centers: list[np.ndarray] = []
    if spec.name == "uniform":
        return rng.uniform(0.0, cfg.L, size=(N_env, 3))
    if spec.name == "grain_weighted":
        seeds = region_index.seeds
        if not seeds:
            return rng.uniform(0.0, cfg.L, size=(N_env, 3))
        volumes = np.array([s.radius ** 3 for s in seeds], dtype=float)
        volumes /= volumes.sum()
        for _ in range(N_env):
            idx = rng.choice(len(seeds), p=volumes)
            seed = seeds[int(idx)]
            center = seed.center + rng.normal(scale=seed.radius / 3, size=3)
            centers.append(np.clip(center, 0.0, cfg.L))
        return np.asarray(centers)
    if spec.name == "boundary_band":
        band_fraction = spec.boundary_band_fraction or cfg.splits.boundary_band_fraction
        oversample = max(spec.oversample_factor, 1.0)
        target_band = int(min(N_env, round(N_env * band_fraction)))
        while len(centers) < target_band:
            candidate = rng.uniform(0.0, cfg.L, size=3)
            grain_id = region_index.lookup(candidate)
            seed = region_index.get_seed(grain_id)
            threshold = band_fraction * seed.radius
            if region_index.boundary_distance(candidate) <= threshold:
                centers.append(candidate)
        remaining = N_env - len(centers)
        if remaining > 0:
            centers.extend(rng.uniform(0.0, cfg.L, size=(remaining, 3)))
        centers_arr = np.asarray(centers)
        rng.shuffle(centers_arr, axis=0)
        return centers_arr
    raise ValueError(f"Unknown environment sampler {spec.name}")


def lookup_ground_truth(
    center: np.ndarray,
    region_index: GrainRegionIndex,
    phase_map: Dict[int, str],
    orientation_field: OrientationField,
    phase_ids: Dict[str, int],
) -> EnvironmentGT:
    grain_id = region_index.lookup(center)
    phase_name = phase_map[grain_id]
    orientation = orientation_field.R(grain_id, center)
    boundary = region_index.boundary_distance(center)
    return EnvironmentGT(
        phase_id=phase_ids[phase_name],
        phase_name=phase_name,
        grain_id=grain_id,
        orientation=orientation,
        boundary_distance=boundary,
    )


def _ensure_length(points: np.ndarray, target: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] == target:
        return points
    if points.shape[0] > target:
        idx = rng.choice(points.shape[0], size=target, replace=False)
        return points[idx]
    idx = rng.choice(points.shape[0], size=target - points.shape[0], replace=True)
    return np.concatenate([points, points[idx]], axis=0)


def apply_noise(points: np.ndarray, noise: NoiseSpec, *, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=float).copy()
    target_len = points.shape[0]
    centroid = points.mean(axis=0)
    if noise.anisotropic_scale is not None:
        s_min, s_max = noise.anisotropic_scale
        scales = rng.uniform(s_min, s_max, size=3)
        points = (points - centroid) * scales + centroid
    radial = points - centroid
    radii = np.linalg.norm(radial, axis=1)
    max_r = max(radii.max(), 1e-6)
    if noise.jitter_sigma > 0:
        points += rng.normal(scale=noise.jitter_sigma * max_r, size=points.shape)
    weights = np.ones(points.shape[0], dtype=float)
    if noise.density_gradient:
        grad = noise.density_gradient
        weights = np.clip(1.0 + grad * (radii / max_r - 0.5), 1e-3, None)
    keep_mask = np.ones(points.shape[0], dtype=bool)
    if noise.missing_rate > 0:
        drop_prob = noise.missing_rate * (weights / weights.max())
        rand = rng.random(points.shape[0])
        keep_mask = rand > drop_prob
        if keep_mask.sum() == 0:
            keep_mask[rng.integers(0, points.shape[0])] = True
        points = points[keep_mask]
        weights = weights[keep_mask]
    if noise.outlier_rate > 0:
        n_out = int(round(noise.outlier_rate * points.shape[0]))
        if n_out > 0:
            idx = rng.choice(points.shape[0], size=n_out, replace=False)
            outliers = rng.uniform(-max_r, max_r, size=(n_out, 3)) + centroid
            points[idx] = outliers
    points = _ensure_length(points, target_len, rng)
    return points


def render_environment(
    center: np.ndarray,
    gt: EnvironmentGT,
    phase_library: PhaseLibrary,
    cfg: DatasetConfig,
    *,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, object]]:
    prototype: PhasePrototype = phase_library.get(gt.phase_name)
    params: MotifParams = prototype.default_params
    canonical = instantiate_motif(
        prototype,
        params,
        cfg.M,
        rng=rng,
        jitter=0.0,
    )
    local = apply_rotation(canonical, gt.orientation)
    world = local + center
    noisy = apply_noise(world, cfg.noise, rng=rng)
    noisy = _ensure_length(noisy, cfg.M, rng)
    orientation_quat = matrix_to_quaternion(gt.orientation)
    meta = {
        "orientation_quaternion": orientation_quat,
        "boundary_distance": gt.boundary_distance,
        "missing_rate": cfg.noise.missing_rate,
        "outlier_rate": cfg.noise.outlier_rate,
        "anisotropic_scale": cfg.noise.anisotropic_scale,
    }
    return noisy.astype(np.float32), meta
