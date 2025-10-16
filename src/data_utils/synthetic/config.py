"""Configuration objects and random number helpers for synthetic point-cloud datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Literal, Optional, Tuple
import math
import random

import numpy as np
import torch

PhaseName = Literal["fcc", "bcc", "hcp", "icosa", "amorphous", "rod", "plate"]
GrainModel = Literal["thomas", "voronoi"]
EnvSamplerName = Literal["uniform", "grain_weighted", "boundary_band"]
ExportFormat = Literal["npz", "h5"]


@dataclass
class GrainRadiusDistSpec:
    """Specification of the grain radius distribution."""

    kind: Literal["lognormal", "gamma", "uniform", "fixed"]
    params: Dict[str, float]


@dataclass
class NoiseSpec:
    jitter_sigma: float
    anisotropic_scale: Optional[Tuple[float, float]] = None
    missing_rate: float = 0.0
    outlier_rate: float = 0.0
    density_gradient: Optional[float] = None


@dataclass
class SplitSpec:
    ratios: Dict[str, float]
    holdout_unseen_rotations: bool = True
    boundary_band_fraction: float = 0.02


@dataclass
class EnvCenterSamplerSpec:
    name: EnvSamplerName
    boundary_band_fraction: Optional[float] = None
    oversample_factor: float = 1.0


@dataclass
class DatasetConfig:
    L: float
    num_phase_types: int
    phase_mix: Dict[PhaseName, float]
    grain_model: GrainModel
    thomas_parent_intensity: Optional[float] = None
    thomas_child_mean: Optional[float] = None
    thomas_dispersion_sigma: Optional[float] = None
    voronoi_seed_count: Optional[int] = None
    grain_radius_dist: Optional[GrainRadiusDistSpec] = None
    orientation_intra_grain_kappa: float = math.radians(3.0)
    M: int = 64
    env_center_sampler: EnvCenterSamplerSpec = field(default_factory=lambda: EnvCenterSamplerSpec(name="uniform"))
    noise: NoiseSpec = field(default_factory=lambda: NoiseSpec(jitter_sigma=0.01))
    imbalance: Optional[Dict[PhaseName, float]] = None
    splits: SplitSpec = field(default_factory=lambda: SplitSpec(ratios={"train": 0.7, "val": 0.15, "test": 0.15}))
    seed: int = 42

    def validate(self) -> None:
        total = sum(self.phase_mix.values())
        if not np.isclose(total, 1.0, atol=1e-4):
            raise ValueError(f"phase_mix must sum to 1 (got {total}).")
        split_total = sum(self.splits.ratios.values())
        if not np.isclose(split_total, 1.0, atol=1e-4):
            raise ValueError(f"Split ratios must sum to 1 (got {split_total}).")
        if self.grain_model == "thomas":
            if self.thomas_parent_intensity is None or self.thomas_child_mean is None:
                raise ValueError("Thomas process requires parent intensity and child mean.")
        if self.grain_model == "voronoi" and self.voronoi_seed_count is None:
            raise ValueError("Voronoi model requires voronoi_seed_count.")


class RNGManager:
    """Convenience holder for numpy/torch RNGs tied to a root seed."""

    def __init__(self, seed: int) -> None:
        self._root_seed = seed
        self._np_rng = np.random.default_rng(seed)
        self._torch_rng = torch.Generator()
        self._torch_rng.manual_seed(seed)

    @property
    def seed(self) -> int:
        return self._root_seed

    def np_rng(self) -> np.random.Generator:
        return self._np_rng

    def torch_rng(self) -> torch.Generator:
        return self._torch_rng

    def reseed(self, seed: int) -> None:
        self._root_seed = seed
        self._np_rng = np.random.default_rng(seed)
        self._torch_rng.manual_seed(seed)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sample_radius(spec: GrainRadiusDistSpec, rng: np.random.Generator) -> float:
    """Sample a radius according to the provided specification."""

    if spec.kind == "fixed":
        return float(spec.params["value"])
    if spec.kind == "lognormal":
        median = spec.params.get("median", 0.1)
        gstd = spec.params.get("gstd", 1.5)
        mu = math.log(median)
        sigma = math.log(gstd)
        return float(rng.lognormal(mean=mu, sigma=sigma))
    if spec.kind == "gamma":
        shape = spec.params.get("shape", 2.0)
        scale = spec.params.get("scale", 0.1)
        return float(rng.gamma(shape, scale))
    if spec.kind == "uniform":
        low = spec.params.get("low", 0.05)
        high = spec.params.get("high", 0.15)
        return float(rng.uniform(low, high))
    raise ValueError(f"Unsupported grain radius distribution: {spec.kind}")


def normalize_phase_mix(phase_mix: Dict[PhaseName, float]) -> Dict[PhaseName, float]:
    total = sum(phase_mix.values())
    if total <= 0:
        raise ValueError("phase_mix must have positive weights.")
    return {k: v / total for k, v in phase_mix.items()}


def weighted_choice(labels: Iterable[PhaseName], weights: Iterable[float], rng: np.random.Generator) -> PhaseName:
    labels_list = list(labels)
    weights_list = np.array(list(weights), dtype=float)
    if weights_list.sum() == 0:
        raise ValueError("Weights must sum to > 0.")
    probs = weights_list / weights_list.sum()
    idx = rng.choice(len(labels_list), p=probs)
    return labels_list[idx]
