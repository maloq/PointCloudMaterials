import os
import sys
import time
import hydra
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
from src.utils.curriculum_callback import CurriculumLearningCallback
from src.data_utils.data_module import (
    RealPointCloudDataModule,
    SyntheticPointCloudDataModule,
)
torch.set_float32_matmul_precision('high')

logger = setup_logging()

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

    # Set up curriculum learning callback if enabled
    callbacks = list(checkpoint_callbacks) + [lr_monitor]
    if hasattr(cfg, 'curriculum_learning') and cfg.curriculum_learning.enable:
        curriculum_callback = CurriculumLearningCallback(
            start_fraction=cfg.curriculum_learning.start_fraction,
            end_fraction=cfg.curriculum_learning.end_fraction,
            start_epoch=cfg.curriculum_learning.start_epoch,
            end_epoch=cfg.curriculum_learning.end_epoch,
        )
        callbacks.append(curriculum_callback)
        logger.print("Curriculum learning callback enabled")

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

    trainer_kwargs = dict(
        default_root_dir=run_dir,
        max_epochs=cfg.epochs,
        accelerator='gpu' if cfg.gpu else 'cpu',
        devices=devices,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=cfg.log_every_n_steps,
        precision='bf16-mixed',
        benchmark=True,
        check_val_every_n_epoch=4,
    )

    # Reload dataloaders every epoch when curriculum learning is active
    # This is necessary because the dataset size changes between epochs
    if hasattr(cfg, 'curriculum_learning') and cfg.curriculum_learning.enable:
        trainer_kwargs["reload_dataloaders_every_n_epochs"] = 1
        # Disable automatic DistributedSampler when using curriculum learning
        # because the dataset size changes and DistributedSampler expects fixed size
        if len(devices) > 1:
            trainer_kwargs["use_distributed_sampler"] = False
            logger.print("Disabled automatic DistributedSampler for curriculum learning")
        logger.print("Dataloader will be reloaded every epoch for curriculum learning")

    if ddp_strategy is not None:
        trainer_kwargs["strategy"] = ddp_strategy

    trainer = pl.Trainer(**trainer_kwargs)

    trainer.fit(model, dm)

    # Run test after training completes
    if run_test:
        logger.print("Starting test phase...")
        trainer.test(model, dm, ckpt_path='best')

    return trainer, model, dm, checkpoint_callbacks


def train(cfg: DictConfig):
    """SPD-specific training function."""
    train_model(cfg, ShapePoseDisentanglement)


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='spd_synth_small')
def main(cfg: DictConfig):
    train(cfg)

if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
