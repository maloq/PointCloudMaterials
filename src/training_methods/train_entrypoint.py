import os
from pathlib import Path
import sys

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, ListConfig, OmegaConf
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
        raise RuntimeError(f'Training finished, but requested analysis failed for {checkpoint_path}') from exc


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
        raise FileNotFoundError(f'Requested post-training analysis has no retained best checkpoint: {best_ckpt!r}')

    checkpoint_dir = Path(best_ckpt).parent
    default_analysis = (checkpoint_dir.parent if checkpoint_dir.name == 'technical' else
                        checkpoint_dir / ('analysis_standard' if cfg.data.kind == 'relaxed_histories' else 'analysis'))
    output_dir = OmegaConf.select(cfg, 'analysis_output_dir', default=str(default_analysis))
    run_post_training_analysis_safe(best_ckpt, output_dir, _first_cuda_device(cfg))
    if not bool(OmegaConf.select(cfg, 'checkpoint_keep_last_after_analysis', default=True)):
        last = Path(best_ckpt).parent/'last.ckpt'
        if last.exists() and last.resolve() != Path(best_ckpt).resolve():
            last.unlink()


def _train(
    cfg: DictConfig,
    *,
    method_name: str | None = None,
    run_analysis: bool = True,
):
    model_class, default_analysis = resolve_training_method(
        cfg, method_name=method_name
    )
    run_test = bool(getattr(cfg, "run_test_after_training", True))

    trainer, model, dm, checkpoint_callbacks = train_model(
        cfg,
        model_class,
        run_test=run_test,
    )

    _run_registered_post_training_analysis(
        cfg,
        checkpoint_callbacks=checkpoint_callbacks,
        enabled_by_default=default_analysis,
        requested=run_analysis,
    )
    return trainer, model, dm, checkpoint_callbacks


def train(cfg: DictConfig, *, method_name: str | None = None, run_analysis: bool = True):
    # Hydra owns these entry points; only rank zero writes shared provenance.
    if rank_zero_only.rank != 0:
        return _train(cfg, method_name=method_name, run_analysis=run_analysis)
    from src.experiment_runner.tracking import tracked_run
    output = Path(HydraConfig.get().runtime.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if output.name == 'technical':
        from src.experiment_runner.artifacts import result_folders
        result_folders(output.parent)
        (output.parent / 'README.md').write_text('# Training run\n\n'
            'Technical files contain the resolved config, execution record and checkpoints.\n'
            'The requested post-training analysis will populate plots/ and tables/.\n')
    config = output / 'resolved_config.yaml'
    OmegaConf.save(cfg, config, resolve=True)
    with tracked_run(output, kind='training', configs=[config], command=[sys.executable, *sys.argv]):
        return _train(cfg, method_name=method_name, run_analysis=run_analysis)


__all__ = ["run_post_training_analysis_safe", "train"]
