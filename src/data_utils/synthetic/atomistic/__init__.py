"""Force-driven atomistic benchmark generation.

The package exposes the endpoint/interface generator, direct-coexistence phase
transformations, and homogeneous crystallization from the validated bulk liquid.
Phase density is measured from the simulated volume; it is never supplied as a
packing parameter.
"""

from .config import (
    GeneratorConfig,
    PotentialQualification,
    load_config,
    validate_potential_qualification,
)
from .generator import GenerationResult, generate_dataset
from .homogeneous_config import (
    HomogeneousCrystallizationConfig,
    load_homogeneous_crystallization_config,
)
from .homogeneous_campaign_config import (
    HomogeneousCampaignConfig,
    load_homogeneous_campaign_config,
)
from .homogeneous_generator import (
    HomogeneousCrystallizationReplicaResult,
    HomogeneousCrystallizationResult,
    generate_homogeneous_crystallization_dataset,
)
from .homogeneous_liquid_source import (
    HomogeneousLiquidSourceResult,
    generate_homogeneous_liquid_source,
)
from .transition_config import TransitionConfig, load_transition_config
from .transition_generator import TransitionGenerationResult, generate_transition_dataset
from .transition_rdf import add_phase_rdf_to_transition_dataset

__all__ = [
    "GenerationResult",
    "GeneratorConfig",
    "HomogeneousCrystallizationConfig",
    "HomogeneousCrystallizationReplicaResult",
    "HomogeneousCrystallizationResult",
    "HomogeneousCampaignConfig",
    "HomogeneousLiquidSourceResult",
    "PotentialQualification",
    "TransitionConfig",
    "TransitionGenerationResult",
    "add_phase_rdf_to_transition_dataset",
    "generate_dataset",
    "generate_homogeneous_crystallization_dataset",
    "generate_homogeneous_liquid_source",
    "generate_transition_dataset",
    "load_config",
    "load_homogeneous_crystallization_config",
    "load_homogeneous_campaign_config",
    "load_transition_config",
    "validate_potential_qualification",
]
