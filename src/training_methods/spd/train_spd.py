import os
import sys
import time
import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from datetime import datetime
from omegaconf import DictConfig
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import wandb


sys.path.append(os.getcwd())
from src.utils.logging_config import setup_logging
from src.training_methods.spd.spd_module import ShapePoseDisentanglement
from src.data_utils.data_module import PointCloudDataModule
torch.set_float32_matmul_precision('medium')

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


def train(cfg: DictConfig):
    logger.print(f"Starting in {os.getcwd()}")
    run_dir = get_rundir_name()
    wandb_logger = init_wandb(cfg, run_dir)

    dm = PointCloudDataModule(cfg)
    model = ShapePoseDisentanglement(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=run_dir,
        monitor='val_loss',
        filename=f'{cfg.experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=3,
        mode='min',
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=cfg.epochs,
        accelerator='gpu' if cfg.gpu else 'cpu',
        devices=[0],
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=cfg.log_every_n_steps,
        precision='bf16-mixed',
        benchmark=True,
    )

    trainer.fit(model, dm)


@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(), 'configs'), config_name='spd_synth')
def main(cfg: DictConfig):
    train(cfg)

if __name__ == '__main__':
    sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
