import os
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
try:
    from src.utils.logging_config import setup_logging
    from src.training_methods.spd.spd_module import ShapePoseDisentanglement
    from src.utils.curriculum_callback import CurriculumLearningCallback
    from src.data_utils.data_module import (
        RealPointCloudDataModule,
        SyntheticPointCloudDataModule,
    )
except ImportError:
    # If running from src/training_methods/spd/ substitute path
    sys.path.append(os.path.join(os.getcwd(), '..', '..', '..'))
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
    return str(f"output/overfit/{now:%Y-%m-%d}/{now:%H-%M-%S}")

@rank_zero_only
def init_wandb(cfg: DictConfig, run_dir):
    os.environ['WANDB_MODE'] = cfg.wandb_mode
    os.environ['WANDB_DIR'] = 'output/wandb'
    os.environ['WANDB_CONFIG_DIR'] = 'output/wandb'
    os.environ['WANDB_CACHE_DIR'] = 'output/wandb'
    wandb.init(project='PointCloudMaterials', name=f"OVERFIT_{cfg.experiment_name}")
    return WandbLogger(save_dir=os.path.join(os.getcwd(), run_dir),
                       project=cfg.project_name,
                       name=f"OVERFIT_{cfg.experiment_name}",
                       log_model=False)


def train_model(cfg: DictConfig, model_class, run_dir=None, checkpoint_callbacks=None,
                devices=None, run_test=False):
    """
    Overfit training function.
    """
    logger.print(f"Starting OVERFIT run in {os.getcwd()}")

    if run_dir is None:
        run_dir = get_rundir_name()

    wandb_logger = init_wandb(cfg, run_dir)

    if cfg.data.kind == "synthetic":
        dm = SyntheticPointCloudDataModule(cfg)
    else:
        dm = RealPointCloudDataModule(cfg)
    model = model_class(cfg)

    lr_monitor = LearningRateMonitor(logging_interval='step')

    # No curriculum learning for overfitting
    callbacks = [lr_monitor]
    logger.print("Curriculum learning DISABLED for overfitting")

    # Set up devices
    if devices is None:
        if isinstance(cfg.devices, ListConfig):
            devices = list(cfg.devices)
        elif isinstance(cfg.devices, (list, tuple)):
            devices = list(cfg.devices) if len(cfg.devices) > 0 else [0]
        else:
            devices = [0]

    # For overfitting we usually don't need DDP, simplified to single device usually better but let's keep it robust
    ddp_strategy = None
    if cfg.gpu and len(devices) > 1 and hasattr(cfg, 'ddp_find_unused_parameters') and cfg.ddp_find_unused_parameters:
        ddp_strategy = DDPStrategy(find_unused_parameters=True)

    # Force relevant params for overfitting
    overfit_epochs = 5000 # Much larger to ensure convergence
    
    trainer_kwargs = dict(
        default_root_dir=run_dir,
        max_epochs=overfit_epochs,
        accelerator='gpu' if cfg.gpu else 'cpu',
        devices=devices,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1, # Log every step
        precision='bf16-mixed',
        benchmark=True,
        check_val_every_n_epoch=1, # Check every epoch
        overfit_batches=1, # THE KEY CHANGE
    )
    
    if hasattr(cfg, 'gradient_clip_val'):
        trainer_kwargs['gradient_clip_val'] = cfg.gradient_clip_val
        trainer_kwargs['gradient_clip_algorithm'] = 'norm'

    if ddp_strategy is not None:
        trainer_kwargs["strategy"] = ddp_strategy

    trainer = pl.Trainer(**trainer_kwargs)

    # We might need to manually ensure dataloader has data if overfit_batches interacts weirdly with custom DataModules
    # But usually pl handles it.
    
    trainer.fit(model, dm)

    return trainer, model, dm, checkpoint_callbacks


def train(cfg: DictConfig):
    """SPD-specific overfit training function."""
    trainer, model, dm, checkpoint_callbacks = train_model(cfg, ShapePoseDisentanglement)
    return trainer, model, dm, checkpoint_callbacks


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='spd_synth_small')
def main(cfg: DictConfig):
    train(cfg)

if __name__ == '__main__':
    # Override some defaults for overfitting
    sys.argv.append('hydra.run.dir=output/overfit/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
