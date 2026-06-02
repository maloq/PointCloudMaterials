import os
import traceback

from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from src.training_methods.registry import resolve_training_method
from src.training_methods.trainer import train_model
from src.utils.logging_config import setup_logging


logger = setup_logging()


@rank_zero_only
def run_post_training_analysis_safe(
    checkpoint_path: str,
    output_dir: str,
    cuda_device: int = 0,
):
    try:
        from src.analysis.pipeline import run_post_training_analysis

        logger.print("\n" + "=" * 60)
        logger.print("Starting contrastive analysis...")
        logger.print("=" * 60)

        run_post_training_analysis(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            cuda_device=cuda_device,
        )
        logger.print("Post-training analysis completed successfully!")
    except Exception as exc:
        logger.print(f"\nWarning: Post-training analysis failed with error: {exc}")
        logger.print("Training completed successfully, but analysis could not be run.")
        traceback.print_exc()


def _first_cuda_device(cfg: DictConfig) -> int:
    devices = getattr(cfg, "devices", None)
    if isinstance(devices, ListConfig):
        return int(list(devices)[0]) if devices else 0
    if isinstance(devices, (list, tuple)):
        return int(devices[0]) if devices else 0
    return 0


def _run_registered_post_training_analysis(
    cfg: DictConfig,
    *,
    checkpoint_callbacks,
    enabled_by_default: bool,
    requested: bool,
) -> None:
    run_analysis = bool(getattr(cfg, "run_post_training_analysis", requested and enabled_by_default))
    if not run_analysis:
        return

    best_ckpt = checkpoint_callbacks[0].best_model_path if checkpoint_callbacks else ""
    if not best_ckpt or not os.path.exists(best_ckpt):
        logger.print("Warning: No best checkpoint found, skipping post-training analysis")
        return

    output_dir = os.path.join(os.path.dirname(best_ckpt), "analysis")
    run_post_training_analysis_safe(best_ckpt, output_dir, _first_cuda_device(cfg))


def train(
    cfg: DictConfig,
    *,
    method_name: str | None = None,
    run_analysis: bool = True,
):
    method = resolve_training_method(cfg, method_name=method_name)
    model_class = method.load_module_class()
    run_test = bool(getattr(cfg, "run_test_after_training", True))

    trainer, model, dm, checkpoint_callbacks = train_model(
        cfg,
        model_class,
        run_test=run_test,
    )

    _run_registered_post_training_analysis(
        cfg,
        checkpoint_callbacks=checkpoint_callbacks,
        enabled_by_default=method.run_post_training_analysis,
        requested=run_analysis,
    )
    return trainer, model, dm, checkpoint_callbacks


__all__ = ["run_post_training_analysis_safe", "train"]
