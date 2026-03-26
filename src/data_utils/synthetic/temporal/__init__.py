"""
Temporal synthetic atomistic benchmark generation.

The temporal generator is intentionally separate from the legacy static
generator. It builds latent site-state trajectories first, then renders each
frame and tracked neighborhood sequence from that latent process.
"""

from .api import generate_temporal_dataset
from .config import TemporalBenchmarkConfig, dump_temporal_config, load_temporal_config
from .sharded import generate_temporal_dataset_shards
from .visualization import generate_temporal_visualizations

__all__ = [
    "TemporalBenchmarkConfig",
    "dump_temporal_config",
    "generate_temporal_dataset",
    "generate_temporal_dataset_shards",
    "generate_temporal_visualizations",
    "load_temporal_config",
]
