"""Post-training analysis pipeline utilities."""


def __getattr__(name: str):
    """Lazy imports to avoid conflicts when running pipeline.py as __main__."""
    if name in ("build_datamodule", "load_vicreg_model", "run_post_training_analysis"):
        from .pipeline import build_datamodule, load_vicreg_model, run_post_training_analysis
        return {"build_datamodule": build_datamodule, "load_vicreg_model": load_vicreg_model,
                "run_post_training_analysis": run_post_training_analysis}[name]
    if name in ("load_checkpoint_analysis_config", "load_checkpoint_training_config"):
        from .config import load_checkpoint_analysis_config, load_checkpoint_training_config
        return {"load_checkpoint_analysis_config": load_checkpoint_analysis_config,
                "load_checkpoint_training_config": load_checkpoint_training_config}[name]
    if name == "AnalyzableModel":
        from .utils import AnalyzableModel
        return AnalyzableModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnalyzableModel",
    "build_datamodule",
    "load_vicreg_model",
    "load_checkpoint_analysis_config",
    "load_checkpoint_training_config",
    "run_post_training_analysis",
]
