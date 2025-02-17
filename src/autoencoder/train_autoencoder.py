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


import warnings
warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision('high')


def log_config(cfg):
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))


def get_rundir_name() -> str:
    now = datetime.now()
    return str(f'output/{now:%Y-%m-%d}/{now:%H-%M-%S}')


def train(cfg: DictConfig):

    start_time = time.process_time()
    logger.print(f"Starting in {os.getcwd()}")
    logger.print(f"torch version {torch.__version__ }")
    os.environ['WANDB_MODE'] = cfg.wandb_mode
    os.environ['WANDB_DIR'] = 'output/wandb'
    os.environ['WANDB_CONFIG_DIR'] = 'output/wandb'
    os.environ['WANDB_CACHE_DIR'] = 'output/wandb'
    run_dir = get_rundir_name()
    wandb.finish()
    wandb.init(project='PointCloudMaterials', name=cfg.experiment_name)
    wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), run_dir),
                               project=cfg.project_name,
                               name=cfg.experiment_name,
                               model_log_interval=None,
                               log_dataset_dir=None,
                               log_best_dir=None,
                               log_latest_dir=None)
    default_root_dir = run_dir
    checkpoint_loc = run_dir   
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=False) 
    
    dm = PointCloudDataModule(cfg)
    model = PointNetAutoencoder(cfg)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_loc,
        monitor='val_loss',
        filename='pointnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )
    
    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        max_epochs=cfg.training.epochs,
        accelerator='gpu' if cfg.training.gpu else 'cpu',
        callbacks=[checkpoint_callback, lr_monitor, StochasticWeightAveraging(swa_lrs=0.001)],
        precision='16-mixed',
        devices=[0],
        log_every_n_steps=cfg.training.log_every_n_steps,
        logger=wandb_logger,
        benchmark=True,
        check_val_every_n_epoch=10,
        profiler='simple',
    )
    log_config(cfg)
    logger.print(f"Time to start script {time.process_time() - start_time} seconds")
    trainer.fit(model, dm)



@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="Al_autoencoder")
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
    wandb.finish()
    sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
    wandb.finish()
