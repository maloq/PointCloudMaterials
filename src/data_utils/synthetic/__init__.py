"""Synthetic dataset generation utilities for rotation-equivariant point-cloud models."""

from .config import (
    DatasetConfig,
    EnvCenterSamplerSpec,
    GrainRadiusDistSpec,
    NoiseSpec,
    RNGManager,
    SplitSpec,
    set_random_seed,
)
from .presets import (
    BASELINE_PRESET,
    imbalanced_phase_preset,
    temporal_growth_sequence,
    voronoi_anisotropic_preset,
)
from .scene import generate_scene, make_splits, export_scene, load_scene, SyntheticPointCloudDataset
from .metrics import clustering_scores, rotation_errors, phase_iou, robustness_curve, boundary_stratified_metrics
from .sanity_checks import (
    moran_i,
    ripley_k,
    motif_separability,
    check_equivariance,
    boundary_band_fraction,
    phase_cardinality,
)

__all__ = [
    "DatasetConfig",
    "EnvCenterSamplerSpec",
    "GrainRadiusDistSpec",
    "NoiseSpec",
    "RNGManager",
    "SplitSpec",
    "set_random_seed",
    "generate_scene",
    "make_splits",
    "export_scene",
    "load_scene",
    "SyntheticPointCloudDataset",
    "BASELINE_PRESET",
    "imbalanced_phase_preset",
    "voronoi_anisotropic_preset",
    "temporal_growth_sequence",
    "clustering_scores",
    "rotation_errors",
    "phase_iou",
    "robustness_curve",
    "boundary_stratified_metrics",
    "moran_i",
    "ripley_k",
    "motif_separability",
    "check_equivariance",
    "boundary_band_fraction",
    "phase_cardinality",
]
