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
from src.training_methods.spd.supervised_encoder_module import SupervisedEncoder
from src.data_utils.data_module import SyntheticPointCloudDataModule

torch.set_float32_matmul_precision('medium')

logger = setup_logging()


@rank_zero_only
def get_rundir_name() -> str:
    now = datetime.now()
    return str(f"output/supervised_encoder/{now:%Y-%m-%d}/{now:%H-%M-%S}")


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


def get_checkpoint_filename(cfg: DictConfig) -> str:
    """Generate checkpoint filename with hyperparameters."""
    encoder_name = cfg.encoder.name
    latent_size = cfg.latent_size
    batch_size = cfg.batch_size
    lr = cfg.learning_rate
    rotation_mode = cfg.rotation_mode

    filename = (f"supervised_encoder_{encoder_name}_"
               f"l{latent_size}_bs{batch_size}_lr{lr:.4f}_"
               f"rot{rotation_mode}_epoch{{epoch:02d}}")
    return filename


def train(cfg: DictConfig):
    logger.print(f"Starting supervised encoder pretraining in {os.getcwd()}")
    run_dir = get_rundir_name()
    wandb_logger = init_wandb(cfg, run_dir)

    # Use synthetic data module
    dm = SyntheticPointCloudDataModule(cfg)

    # Build model
    model = SupervisedEncoder(cfg)

    # Checkpoint callback with hyperparameters in filename
    checkpoint_filename = get_checkpoint_filename(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        monitor='val/accuracy',
        filename=checkpoint_filename,
        save_top_k=3,
        mode='max',  # Maximize validation accuracy
    )

    # Also save best loss checkpoint
    checkpoint_callback_loss = ModelCheckpoint(
        dirpath=run_dir,
        monitor='val/loss',
        filename=checkpoint_filename.replace('epoch', 'loss_epoch'),
        save_top_k=1,
        mode='min',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')



    trainer_kwargs = dict(
        default_root_dir=run_dir,
        max_epochs=cfg.epochs,
        accelerator='gpu' if cfg.gpu else 'cpu',
        devices=[0],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, checkpoint_callback_loss, lr_monitor],
        log_every_n_steps=cfg.log_every_n_steps,
        precision='bf16-mixed',
        benchmark=True,
    )
    trainer = pl.Trainer(**trainer_kwargs)

    # Train
    trainer.fit(model, dm)

    logger.print(f"Training completed. Checkpoints saved to {run_dir}")
    logger.print(f"Best checkpoint (accuracy): {checkpoint_callback.best_model_path}")
    logger.print(f"Best checkpoint (loss): {checkpoint_callback_loss.best_model_path}")


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='supervised_encoder_synth')
def main(cfg: DictConfig):
    train(cfg)


if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=output/supervised_encoder/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
