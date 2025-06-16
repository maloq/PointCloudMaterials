import sys,os
sys.path.append(os.getcwd())
from src.utils.logging_config import setup_logging
logger = setup_logging()
import torch
import numpy as np
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import wandb
import time
from src.autoencoder.autoencoder_module import PointNetAutoencoder
from src.data_utils.data_module import PointCloudDataModule
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from typing import Union

import warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('medium')

@rank_zero_only
def log_config(cfg):
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

@rank_zero_only
def get_rundir_name() -> str:
    now = datetime.now()
    return str(f'output/{now:%Y-%m-%d}/{now:%H-%M-%S}')

@rank_zero_only
def init_wandb(cfg: DictConfig, run_dir):
    os.environ['WANDB_MODE'] = cfg.wandb_mode
    os.environ['WANDB_DIR'] = 'output/wandb'
    os.environ['WANDB_CONFIG_DIR'] = 'output/wandb'
    os.environ['WANDB_CACHE_DIR'] = 'output/wandb'
    wandb.init(project='PointCloudMaterials', name=cfg.experiment_name)
    wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), run_dir),
                               project=cfg.project_name,
                               name=cfg.experiment_name,
                               num_points=cfg.data.num_points,
                               overlap_fraction=cfg.data.overlap_fraction,
                               model_log_interval=None,
                               log_dataset_dir=None,
                               log_best_dir=None,
                               log_latest_dir=None)
    return wandb_logger


class AdaptiveSWA(StochasticWeightAveraging):
    """Start SWA from whatever LR the optimizer is using at swa_epoch_start."""
    def __init__(self, swa_epoch_start=0.8, **kwargs):
        # pass a dummy lr to satisfy the parent __init__
        super().__init__(swa_lrs=1e-3, swa_epoch_start=swa_epoch_start, **kwargs)

    def on_train_epoch_start(self, trainer, pl_module):
        # When we reach the first SWA epoch (but before Lightning initialises SWA)
        if (not self._initialized) and (self.swa_start <= trainer.current_epoch <= self.swa_end):
            # Grab the live LRs from the (single) optimizer
            self._swa_lrs = [pg["lr"] for pg in trainer.optimizers[0].param_groups]
        # Let the parent class do the normal SWA setup
        super().on_train_epoch_start(trainer, pl_module)


def train(cfg: DictConfig):

    start_time = time.process_time()
    logger.print(f"Starting in {os.getcwd()}")
    logger.print(f"torch version {torch.__version__ }")
    run_dir = get_rundir_name()
    wandb_logger = init_wandb(cfg, run_dir)

    default_root_dir = run_dir
    checkpoint_loc = run_dir   
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=False) 
    
    dm = PointCloudDataModule(cfg)
    model = PointNetAutoencoder(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_loc,
        monitor='val_loss',
        filename=f'{cfg.experiment_name}-{{epoch:02d}}-{{val_loss:.2f}}',
        save_top_k=3,
        mode='min',
    )
    
    if cfg.enable_swa:
        assert cfg.swa_epoch_start < cfg.epochs, "SWA epoch start must be less than epochs"
        swa_callback = AdaptiveSWA(swa_epoch_start=cfg.swa_epoch_start)
        callbacks=[checkpoint_callback, lr_monitor, swa_callback]
    else:
        callbacks=[checkpoint_callback, lr_monitor]

    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        max_epochs=cfg.epochs,
        accelerator='gpu' if cfg.gpu else 'cpu',
        callbacks=callbacks,
        precision='16-mixed',
        devices=[0],
        log_every_n_steps=cfg.log_every_n_steps,
        logger=wandb_logger,
        benchmark=True,
        check_val_every_n_epoch=10,
        profiler='simple',
        gradient_clip_val=0.5,
        
    )
    log_config(cfg)
    logger.print(f"Time to start script {time.process_time() - start_time} seconds")
    trainer.fit(model, dm)



@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="autoencoder_16")
def main(cfg: DictConfig):
    logger.print(f"torch.version.cuda: {torch.version.cuda}")
    logger.print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logger.print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    try:
        train(cfg)
        logger.print('Train finished!')
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            logger.print('Keyboard interrupt detected, finishing training...')
            wandb.finish()
        else:
            logger.error(f"An unexpected error occurred: {e}")
            raise e


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn', force=True)
    sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
    wandb.finish()
