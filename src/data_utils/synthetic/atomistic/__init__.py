"""Force-driven atomistic benchmark generation.

The package intentionally exposes one workflow: equilibrate a solid and a
liquid at a declared thermodynamic state, then create a solid--liquid
coexistence cell by melting a slab of the solid.  Phase density is measured
from the simulated volume; it is never supplied as a packing parameter.
"""

from .config import GeneratorConfig, load_config
from .generator import GenerationResult, generate_dataset

__all__ = ["GenerationResult", "GeneratorConfig", "generate_dataset", "load_config"]
