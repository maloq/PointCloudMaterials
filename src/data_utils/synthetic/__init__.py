"""
Synthetic atomistic dataset generation utilities.

The :mod:`synthetic.atomistic_generator` module exposes the main entrypoint
for building synthetic multi-phase 3D atomistic boxes based on YAML configs.
"""

from .atomistic_generator import SyntheticAtomisticDatasetGenerator, load_config
from .visualization import generate_visualizations

__all__ = [
    "SyntheticAtomisticDatasetGenerator",
    "load_config",
    "generate_visualizations",
]
