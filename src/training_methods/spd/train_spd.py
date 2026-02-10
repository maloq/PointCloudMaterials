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


def _resolve_resume_checkpoint(cfg: DictConfig) -> str | None:
    """Resolve optional training resume checkpoint from config."""
    resume_ckpt = getattr(cfg, "resume_from_checkpoint", None)
    if resume_ckpt is None:
        resume_ckpt = getattr(cfg, "resume_checkpoint_path", None)
    if resume_ckpt is None:
        return None

    resume_ckpt = str(resume_ckpt).strip()
    if not resume_ckpt:
        return None

    resume_ckpt = os.path.expanduser(resume_ckpt)
    if not os.path.isabs(resume_ckpt):
        base_dir = os.getcwd()
        try:
            base_dir = HydraConfig.get().runtime.cwd
        except Exception:
            pass
        resume_ckpt = os.path.join(base_dir, resume_ckpt)
    resume_ckpt = os.path.abspath(resume_ckpt)

    if not os.path.exists(resume_ckpt):
        raise FileNotFoundError(f"Resume checkpoint not found: {resume_ckpt}")

    return resume_ckpt

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
        except Exception:
            run_dir = get_rundir_name()

    wandb_logger = init_wandb(cfg, run_dir)

    if cfg.data.kind == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(cfg)
    model = model_class(cfg)

    # Set up checkpoint callbacks
    if checkpoint_callbacks is None:
        checkpoint_callbacks = [ModelCheckpoint(
            dirpath=run_dir,
            monitor='val/loss',
            filename=f'{cfg.experiment_name}-{{epoch:02d}}',
            save_top_k=3,
            mode='min',
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
        check_val_every_n_epoch=4,
    )
    
    if hasattr(cfg, 'gradient_clip_val'):
        trainer_kwargs['gradient_clip_val'] = cfg.gradient_clip_val
        trainer_kwargs['gradient_clip_algorithm'] = 'norm'

    if ddp_strategy is not None:
        trainer_kwargs["strategy"] = ddp_strategy

    trainer = pl.Trainer(**trainer_kwargs)

    resume_ckpt_path = _resolve_resume_checkpoint(cfg)
    if resume_ckpt_path is not None:
        logger.print(f"Resuming training from checkpoint: {resume_ckpt_path}")
    trainer.fit(model, dm, ckpt_path=resume_ckpt_path)

    # Run test after training completes
    if run_test:
        logger.print("Starting test phase...")
        trainer.test(model, dm, ckpt_path='best')

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
