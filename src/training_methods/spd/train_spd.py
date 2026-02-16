import os
# Hack to fix multi-GPU training on this server (NCCL P2P hang)
os.environ["NCCL_P2P_DISABLE"] = "1"

import sys
import time
import traceback
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from datetime import datetime
from omegaconf import DictConfig, ListConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import wandb


sys.path.append(os.getcwd())
from src.utils.logging_config import setup_logging
from src.training_methods.spd.spd_module import ShapePoseDisentanglement
from src.data_utils.data_module import (
    RealPointCloudDataModule,
    SyntheticPointCloudDataModule,
)
torch.set_float32_matmul_precision('high')

logger = setup_logging()


def _resolve_checkpoint_path(cfg: DictConfig, field_names: tuple[str, ...]) -> str | None:
    """Resolve an optional checkpoint path from one of *field_names*."""
    checkpoint_path = None
    for field_name in field_names:
        value = getattr(cfg, field_name, None)
        if value is None:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue
        checkpoint_path = value_str
        break

    if checkpoint_path is None:
        return None

    checkpoint_path = os.path.expanduser(checkpoint_path)
    if not os.path.isabs(checkpoint_path):
        base_dir = os.getcwd()
        try:
            base_dir = HydraConfig.get().runtime.cwd
        except Exception as exc:
            logger.warning(
                "Hydra runtime cwd is unavailable; resolving checkpoint "
                "relative to current directory '%s'. Error: %s",
                base_dir,
                exc,
            )
        checkpoint_path = os.path.join(base_dir, checkpoint_path)
    checkpoint_path = os.path.abspath(checkpoint_path)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return checkpoint_path


def _resolve_resume_checkpoint(cfg: DictConfig) -> str | None:
    """Resolve optional training resume checkpoint from config."""
    return _resolve_checkpoint_path(
        cfg,
        ("resume_from_checkpoint", "resume_checkpoint_path"),
    )


def _resolve_init_checkpoint(cfg: DictConfig) -> str | None:
    """Resolve optional model-initialization checkpoint from config."""
    return _resolve_checkpoint_path(
        cfg,
        (
            "init_from_checkpoint",
            "init_weights_from_checkpoint",
            "weights_checkpoint_path",
        ),
    )


def _extract_state_dict(checkpoint) -> dict:
    """Extract model weights dictionary from a Lightning/PyTorch checkpoint payload."""
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict")
        if isinstance(state_dict, dict):
            return state_dict
        model_state_dict = checkpoint.get("model_state_dict")
        if isinstance(model_state_dict, dict):
            return model_state_dict
        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint
    raise ValueError(
        "Checkpoint does not contain a recognized state dictionary "
        "(`state_dict` or `model_state_dict`)."
    )


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    """Strip common wrapper prefixes (e.g. model./module.) from state dict keys."""
    prefixes = ("model.", "module.")
    normalized = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        normalized[new_key] = value
    return normalized


def _compatible_state_dict_for_model(state_dict: dict, model_state: dict) -> tuple[dict, list[str]]:
    """Keep only tensors whose names and shapes match the current model."""
    compatible = {}
    shape_mismatch = []
    for key, value in state_dict.items():
        target = model_state.get(key)
        if target is None:
            continue
        if tuple(target.shape) != tuple(value.shape):
            shape_mismatch.append(key)
            continue
        compatible[key] = value
    return compatible, shape_mismatch


def _load_model_weights_from_checkpoint(model: pl.LightningModule, checkpoint_path: str, *, strict: bool = False) -> None:
    """Load model weights from *checkpoint_path* without restoring trainer state."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    source_state_dict = _extract_state_dict(checkpoint)
    model_state = model.state_dict()
    ckpt_hp = checkpoint.get("hyper_parameters") if isinstance(checkpoint, dict) else None
    ckpt_enc = getattr(getattr(ckpt_hp, "encoder", None), "name", None)
    model_enc = getattr(getattr(getattr(model, "hparams", None), "encoder", None), "name", None)
    if ckpt_enc and model_enc and str(ckpt_enc) != str(model_enc):
        logger.print(
            f"Warning: init checkpoint encoder differs from current config: "
            f"checkpoint={ckpt_enc}, current={model_enc}. "
            "Only tensors with matching names and shapes will be loaded."
        )

    raw_compatible, raw_shape_mismatch = _compatible_state_dict_for_model(source_state_dict, model_state)
    stripped_state_dict = _strip_state_dict_prefixes(source_state_dict)
    stripped_compatible, stripped_shape_mismatch = _compatible_state_dict_for_model(stripped_state_dict, model_state)

    use_stripped = len(stripped_compatible) > len(raw_compatible)
    selected_state_dict = stripped_compatible if use_stripped else raw_compatible
    selected_shape_mismatch = stripped_shape_mismatch if use_stripped else raw_shape_mismatch

    if not selected_state_dict:
        raise RuntimeError(
            f"No compatible tensors found when loading checkpoint '{checkpoint_path}'. "
            "Check that the checkpoint matches the current architecture."
        )

    missing_keys, unexpected_keys = model.load_state_dict(selected_state_dict, strict=strict)

    source_encoder_keys = sum(1 for k in source_state_dict if k.startswith("encoder."))
    model_encoder_keys = sum(1 for k in model_state if k.startswith("encoder."))
    loaded_encoder_keys = sum(1 for k in selected_state_dict if k.startswith("encoder."))

    logger.print(f"Initialized model weights from checkpoint: {checkpoint_path}")
    logger.print(
        "Checkpoint load summary: "
        f"loaded={len(selected_state_dict)} "
        f"/ model_tensors={len(model_state)}, "
        f"shape_mismatch_skipped={len(selected_shape_mismatch)}, "
        f"missing_after_load={len(missing_keys)}, "
        f"unexpected_after_load={len(unexpected_keys)}, "
        f"strict={strict}"
    )

    if source_encoder_keys > 0 and model_encoder_keys > 0 and loaded_encoder_keys == 0:
        logger.print(
            "Warning: No encoder weights were loaded from init checkpoint. "
            "Current encoder will remain randomly initialized."
        )


def _resolve_accumulate_grad_batches(cfg: DictConfig) -> int:
    """Resolve gradient accumulation from config with backward-compatible aliases."""
    value = getattr(cfg, "accumulate_grad_batches", None)
    if value is None:
        value = getattr(cfg, "gradient_accumulation_steps", None)
    if value is None:
        return 1

    accum = int(value)
    if accum < 1:
        raise ValueError(
            "Gradient accumulation must be >= 1. "
            f"Got accumulate_grad_batches={value!r}."
        )
    return accum

@rank_zero_only
def get_rundir_name() -> str:
    now = datetime.now()
    return str(f"output/{now:%Y-%m-%d}/{now:%H-%M-%S}")

@rank_zero_only
def init_wandb(cfg: DictConfig, run_dir):
    os.environ['WANDB_MODE'] = cfg.wandb_mode
    os.environ['WANDB_DIR'] = 'output/wandb'
    os.environ['WANDB_CONFIG_DIR'] = 'output/wandb'
    os.environ['WANDB_CACHE_DIR'] = 'output/wandb'
    wandb.init(project='PointCloudMaterials', name=cfg.experiment_name)
    return WandbLogger(save_dir=os.path.join(os.getcwd(), run_dir),
                       project=cfg.project_name,
                       name=cfg.experiment_name,
                       log_model=False)


def train_model(cfg: DictConfig, model_class, run_dir=None, checkpoint_callbacks=None,
                devices=None, run_test=True):
    """
    Generic training function that can be used with any PyTorch Lightning model.

    Args:
        cfg: Hydra configuration
        model_class: The model class to instantiate (e.g., ShapePoseDisentanglement, EquivariantAutoencoder)
        run_dir: Optional custom output directory (default: auto-generated from timestamp)
        checkpoint_callbacks: Optional list of checkpoint callbacks (default: single checkpoint monitoring val/loss)
        devices: Optional device list (default: from cfg.devices or [0])
        run_test: Whether to run test phase after training (default: True)

    Returns:
        tuple: (trainer, model, datamodule, checkpoint_callbacks) for post-training processing
    """
    logger.print(f"Starting in {os.getcwd()}")

    if run_dir is None:
        try:
            run_dir = HydraConfig.get().runtime.output_dir
        except Exception as exc:
            run_dir = get_rundir_name()
            logger.warning(
                "Hydra runtime output_dir is unavailable; falling back to "
                "timestamped run directory '%s'. Error: %s",
                run_dir,
                exc,
            )

    wandb_logger = init_wandb(cfg, run_dir)

    if cfg.data.kind == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(cfg)
    model = model_class(cfg)

    checkpoint_monitor = str(getattr(cfg, "checkpoint_monitor", "val/loss"))
    checkpoint_mode = str(getattr(cfg, "checkpoint_mode", "min")).strip().lower()
    if checkpoint_mode not in {"min", "max"}:
        raise ValueError(
            f"checkpoint_mode must be 'min' or 'max', got {checkpoint_mode!r}"
        )
    checkpoint_save_top_k = int(getattr(cfg, "checkpoint_save_top_k", 3))
    checkpoint_save_top_k = max(1, checkpoint_save_top_k)

    # Set up checkpoint callbacks
    if checkpoint_callbacks is None:
        checkpoint_callbacks = [ModelCheckpoint(
            dirpath=run_dir,
            monitor=checkpoint_monitor,
            filename=f'{cfg.experiment_name}-{{epoch:02d}}',
            save_top_k=checkpoint_save_top_k,
            mode=checkpoint_mode,
        )]

    lr_monitor = LearningRateMonitor(logging_interval='step')

    callbacks = list(checkpoint_callbacks) + [lr_monitor]

    # Set up devices
    if devices is None:
        if isinstance(cfg.devices, ListConfig):
            devices = list(cfg.devices)
        elif isinstance(cfg.devices, (list, tuple)):
            devices = list(cfg.devices) if len(cfg.devices) > 0 else [0]
        else:
            devices = [0]

    ddp_strategy = None
    if cfg.gpu and len(devices) > 1 and hasattr(cfg, 'ddp_find_unused_parameters') and cfg.ddp_find_unused_parameters:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)

    precision = getattr(cfg, "precision", "bf16-mixed")
    accumulate_grad_batches = _resolve_accumulate_grad_batches(cfg)
    trainer_kwargs = dict(
        default_root_dir=run_dir,
        max_epochs=cfg.epochs,
        accelerator='gpu' if cfg.gpu else 'cpu',
        devices=devices,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.log_every_n_steps,
        precision=precision,
        benchmark=True,
        check_val_every_n_epoch=30,
        accumulate_grad_batches=accumulate_grad_batches,
    )

    if accumulate_grad_batches > 1:
        logger.print(
            "Gradient accumulation enabled: "
            f"accumulate_grad_batches={accumulate_grad_batches}"
        )
    
    if hasattr(cfg, 'gradient_clip_val'):
        trainer_kwargs['gradient_clip_val'] = cfg.gradient_clip_val
        trainer_kwargs['gradient_clip_algorithm'] = 'norm'

    if ddp_strategy is not None:
        trainer_kwargs["strategy"] = ddp_strategy

    trainer = pl.Trainer(**trainer_kwargs)

    resume_ckpt_path = _resolve_resume_checkpoint(cfg)
    init_ckpt_path = _resolve_init_checkpoint(cfg)

    if resume_ckpt_path is not None and init_ckpt_path is not None:
        raise ValueError(
            "Both resume and init checkpoints are set. Use only one mode:\n"
            "- resume_from_checkpoint (full-state resume)\n"
            "- init_from_checkpoint (weights-only initialization)"
        )

    if init_ckpt_path is not None:
        init_strict = bool(getattr(cfg, "init_from_checkpoint_strict", False))
        _load_model_weights_from_checkpoint(model, init_ckpt_path, strict=init_strict)
        logger.print("Starting fresh training from loaded model weights (epoch/optimizer reset).")
        trainer.fit(model, dm)
    else:
        if resume_ckpt_path is not None:
            logger.print(f"Resuming training from checkpoint: {resume_ckpt_path}")
        trainer.fit(model, dm, ckpt_path=resume_ckpt_path)

    # Resolve the checkpoint used for final test.
    test_ckpt_path = None
    for callback in checkpoint_callbacks:
        path = getattr(callback, "best_model_path", "")
        if isinstance(path, str) and path and os.path.exists(path):
            test_ckpt_path = path
            break
    if test_ckpt_path is None:
        for callback in checkpoint_callbacks:
            path = getattr(callback, "last_model_path", "")
            if isinstance(path, str) and path and os.path.exists(path):
                test_ckpt_path = path
                break

    # Run test after training completes
    if run_test:
        logger.print("Starting test phase...")
        run_test_single_device = bool(getattr(cfg, "test_single_device", True))
        if cfg.gpu and len(devices) > 1 and run_test_single_device:
            logger.print("Running test on a single device to avoid duplicated DDP samples.")
            test_trainer_kwargs = dict(
                default_root_dir=run_dir,
                accelerator='gpu',
                devices=1,
                logger=wandb_logger,
                precision=precision,
                benchmark=True,
                log_every_n_steps=cfg.log_every_n_steps,
            )
            if hasattr(cfg, 'gradient_clip_val'):
                test_trainer_kwargs['gradient_clip_val'] = cfg.gradient_clip_val
                test_trainer_kwargs['gradient_clip_algorithm'] = 'norm'
            test_trainer = pl.Trainer(**test_trainer_kwargs)
            test_trainer.test(model, dm, ckpt_path=test_ckpt_path)
        else:
            trainer.test(model, dm, ckpt_path=test_ckpt_path)

    return trainer, model, dm, checkpoint_callbacks


@rank_zero_only
def run_post_training_analysis_safe(checkpoint_path: str, output_dir: str, cuda_device: int = 0, cfg: DictConfig | None = None):
    """Run post-training analysis with error handling.
    
    This function wraps the analysis in try/except to prevent analysis failures
    from crashing the training pipeline.
    """
    try:
        from src.training_methods.spd.predict_and_visualize import run_post_training_analysis
        
        logger.print("\n" + "=" * 60)
        logger.print("Starting post-training analysis...")
        logger.print("=" * 60)
        
        run_post_training_analysis(
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            cuda_device=cuda_device,
            max_samples=5000,
            k_range=range(2, 7),
            run_dbscan=False,
            run_hdbscan=False,
            force_recompute=True,
            cfg=cfg,
        )
        
        logger.print("Post-training analysis completed successfully!")
        
    except Exception as e:
        logger.print(f"\nWarning: Post-training analysis failed with error: {e}")
        logger.print("Training completed successfully, but analysis could not be run.")
        logger.print("You can run the analysis manually using:")
        logger.print(f"  python src/training_methods/spd/predict_and_visualize.py")
        traceback.print_exc()


def train(cfg: DictConfig, run_analysis: bool = True):
    """SPD-specific training function.
    
    Args:
        cfg: Hydra configuration
        run_analysis: Whether to run post-training analysis (default: True)
    """
    trainer, model, dm, checkpoint_callbacks = train_model(cfg, ShapePoseDisentanglement)
    
    # Run post-training analysis if enabled and we have synthetic data
    if run_analysis and cfg.data.kind == "synthetic":
        # Get the best checkpoint path
        best_ckpt = checkpoint_callbacks[0].best_model_path
        if best_ckpt and os.path.exists(best_ckpt):
            # Output directory is the parent of the checkpoint
            output_dir = os.path.join(os.path.dirname(best_ckpt), "analysis")
            
            # Get CUDA device from config
            if isinstance(cfg.devices, ListConfig):
                cuda_device = list(cfg.devices)[0] if cfg.devices else 0
            elif isinstance(cfg.devices, (list, tuple)):
                cuda_device = cfg.devices[0] if cfg.devices else 0
            else:
                cuda_device = 0
            
            run_post_training_analysis_safe(best_ckpt, output_dir, cuda_device, cfg)
        else:
            logger.print("Warning: No best checkpoint found, skipping post-training analysis")
    
    return trainer, model, dm, checkpoint_callbacks


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='spd_synth_small')
def main(cfg: DictConfig):
    train(cfg)

if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
