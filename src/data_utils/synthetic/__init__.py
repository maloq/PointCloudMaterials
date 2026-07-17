"""Synthetic atomistic dataset generation utilities."""

__all__ = [
    "GenerationResult",
    "GeneratorConfig",
    "generate_dataset",
    "load_config",
    "generate_visualizations",
]


def __getattr__(name):
    if name in {"GenerationResult", "GeneratorConfig", "generate_dataset", "load_config"}:
        from .atomistic_generator import (
            GenerationResult,
            GeneratorConfig,
            generate_dataset,
            load_config,
        )

        namespace = {
            "GenerationResult": GenerationResult,
            "GeneratorConfig": GeneratorConfig,
            "generate_dataset": generate_dataset,
            "load_config": load_config,
        }
        return namespace[name]
    if name == "generate_visualizations":
        from .visualization import generate_visualizations

        return generate_visualizations
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
