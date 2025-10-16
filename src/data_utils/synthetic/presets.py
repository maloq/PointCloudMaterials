"""Ready-to-use dataset configuration presets."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

import numpy as np

from .config import (
    DatasetConfig,
    EnvCenterSamplerSpec,
    GrainRadiusDistSpec,
    NoiseSpec,
    SplitSpec,
    normalize_phase_mix,
)

BASELINE_PRESET = DatasetConfig(
    L=1.0,
    num_phase_types=5,
    phase_mix={"fcc": 0.50, "amorphous": 0.25, "plate": 0.25},
    grain_model="thomas",
    thomas_parent_intensity=25.0,
    thomas_child_mean=6.0,
    thomas_dispersion_sigma=0.08,
    grain_radius_dist=GrainRadiusDistSpec(kind="lognormal", params={"median": 0.10, "gstd": 1.5}),
    orientation_intra_grain_kappa=np.deg2rad(3.0),
    M=64,
    env_center_sampler=EnvCenterSamplerSpec(name="boundary_band", boundary_band_fraction=0.02, oversample_factor=1.2),
    noise=NoiseSpec(jitter_sigma=0.01, missing_rate=0.000, outlier_rate=0.01, anisotropic_scale=(0.98, 1.02)),
    splits=SplitSpec(ratios={"train": 0.7, "val": 0.15, "test": 0.15}),
    seed=42,
)


def imbalanced_phase_preset(rare_phase: str = "icosa", rare_fraction: float = 0.05) -> DatasetConfig:
    cfg = deepcopy(BASELINE_PRESET)
    mix = dict(cfg.phase_mix)
    if rare_phase not in mix:
        raise ValueError(f"Phase {rare_phase} not in baseline mix")
    leftover = 1.0 - rare_fraction
    other_phases = [p for p in mix.keys() if p != rare_phase]
    even_share = leftover / len(other_phases)
    for phase in other_phases:
        mix[phase] = even_share
    mix[rare_phase] = rare_fraction
    cfg.phase_mix = normalize_phase_mix(mix)
    cfg.imbalance = {rare_phase: rare_fraction}
    return cfg


def voronoi_anisotropic_preset() -> DatasetConfig:
    cfg = deepcopy(BASELINE_PRESET)
    cfg.grain_model = "voronoi"
    cfg.voronoi_seed_count = 320
    cfg.grain_radius_dist = GrainRadiusDistSpec(kind="uniform", params={"low": 0.06, "high": 0.14})
    cfg.env_center_sampler = EnvCenterSamplerSpec(name="grain_weighted")
    cfg.noise = NoiseSpec(jitter_sigma=0.008, missing_rate=0.07, outlier_rate=0.01, anisotropic_scale=(0.9, 1.1))
    return cfg


def temporal_growth_sequence(steps: int, growth_rate: float = 0.12) -> List[DatasetConfig]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    cfgs: List[DatasetConfig] = []
    base = deepcopy(BASELINE_PRESET)
    for t in range(steps):
        cfg = deepcopy(base)
        factor = (1.0 + growth_rate) ** t
        if cfg.grain_radius_dist is not None:
            params = dict(cfg.grain_radius_dist.params)
            if "median" in params:
                params["median"] *= factor
            cfg.grain_radius_dist = GrainRadiusDistSpec(kind=cfg.grain_radius_dist.kind, params=params)
        cfg.seed = base.seed + t
        cfgs.append(cfg)
    return cfgs
