import sys,os
sys.path.append(os.getcwd())

import torch
import numpy as np
import hydra
import pytorch_lightning as pl
from src.autoencoder.autoencoder_module import PointNetAutoencoder
from src.data_utils.data_module import PointCloudDataModule
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import logging
import wandb
import time
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
torch.set_float32_matmul_precision('high')


def log_config(cfg):
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))


def get_rundir_name() -> str:
    now = datetime.now()
    return str(f'output/{now:%Y-%m-%d}/{now:%H-%M-%S}')


def train_classification(cfg: DictConfig):

    start_time = time.process_time()
    logging.info(f"Starting in {os.getcwd()}")
    logging.info(f"torch version {torch.__version__ }")
    os.environ['WANDB_MODE'] = cfg.wandb_mode
    os.environ['WANDB_DIR'] = 'output/wandb'
    os.environ['WANDB_CONFIG_DIR'] = 'output/wandb'
    os.environ['WANDB_CACHE_DIR'] = 'output/wandb'
    run_dir = get_rundir_name()
    wandb.finish()
    wandb.init()
    wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), run_dir),
                               project=cfg.project_name,
                               name=cfg.experiment_name,
                               log_model='all')
    
    default_root_dir = run_dir
    checkpoint_loc = run_dir   
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=False) 
    
    dm = PointCloudDataModule(cfg)
    model = PointNetAutoencoder(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_loc,
        monitor='val_loss',
        filename='pointnet-{epoch:02d}-{val_loss:.2f}',
        save_top_k=2,
        mode='min',
    )
    
    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        max_epochs=cfg.training.epochs,
        accelerator='gpu' if cfg.training.gpu else 'cpu',
        callbacks=[checkpoint_callback, lr_monitor],
        precision='32',
        log_every_n_steps=cfg.training.log_every_n_steps,
        logger=wandb_logger,
        benchmark=True,
        check_val_every_n_epoch=10,
        # profiler='simple',
    )
    log_config(cfg)
    logging.info(f"Time to start train {time.process_time() - start_time} seconds")
    trainer.fit(model, dm)



@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="Al_autoencoder")
def main(cfg: DictConfig):
    print("torch.__version__", torch.__version__)
    print("torch.version.cuda", torch.version.cuda)
    print("torch.cuda.is_available()", torch.cuda.is_available())
    print("torch.cuda.device_count()", torch.cuda.device_count())
    try:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
        train_classification(cfg)
        logging.info('Train finished!')
    except Exception as e:
        if isinstance(e, KeyboardInterrupt):
            logging.info('Keyboard interrupt detected, finishing training...')
            wandb.finish()
        else:
            logging.error(f"An unexpected error occurred: {e}")
            raise e


if __name__ == "__main__":
    wandb.finish()
    sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
    wandb.finish()
