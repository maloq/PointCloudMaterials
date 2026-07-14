"""Force-driven atomistic benchmark generation.

The package exposes the endpoint/interface generator, direct-coexistence phase
transformations, and homogeneous crystallization from the validated bulk liquid.
Phase density is measured from the simulated volume; it is never supplied as a
packing parameter.
"""

from .config import GeneratorConfig, load_config
from .generator import GenerationResult, generate_dataset
from .homogeneous_config import (
    HomogeneousCrystallizationConfig,
    load_homogeneous_crystallization_config,
)
from .homogeneous_generator import (
    HomogeneousCrystallizationResult,
    generate_homogeneous_crystallization_dataset,
)
from .transition_config import TransitionConfig, load_transition_config
from .transition_generator import TransitionGenerationResult, generate_transition_dataset
from .transition_rdf import add_phase_rdf_to_transition_dataset

__all__ = [
    "GenerationResult",
    "GeneratorConfig",
    "HomogeneousCrystallizationConfig",
    "HomogeneousCrystallizationResult",
    "TransitionConfig",
    "TransitionGenerationResult",
    "add_phase_rdf_to_transition_dataset",
    "generate_dataset",
    "generate_homogeneous_crystallization_dataset",
    "generate_transition_dataset",
    "load_config",
    "load_homogeneous_crystallization_config",
    "load_transition_config",
]
