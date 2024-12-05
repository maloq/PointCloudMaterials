import sys,os
sys.path.append(os.getcwd())
import torch
import numpy as np
import pytorch_lightning as pl
from src.cls.classification_module import PointNetClassifier
from src.cls.data_module import PointCloudDataModule
import hydra
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
    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_DIR'] = 'output/wandb'
    os.environ['WANDB_CONFIG_DIR'] = 'output/wandb'
    os.environ['WANDB_CACHE_DIR'] = 'output/wandb'
    run_dir = get_rundir_name()
    wandb.init()
    wandb_logger = WandbLogger(save_dir=os.path.join(os.getcwd(), run_dir),
                               project=cfg.project_name,
                               name=cfg.experiment_name,
                               log_model='all')
    
    default_root_dir = run_dir
    checkpoint_loc = run_dir   
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=False) 
    
    dm = PointCloudDataModule(cfg)
    
    model = PointNetClassifier(
        lr=cfg.training.learning_rate,
        use_normals=False,
        decay_rate=cfg.training.decay_rate
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_loc,
        monitor='val_acc',
        filename='pointnet-{epoch:02d}-{val_acc:.2f}',
        save_top_k=2,
        mode='max',
    )
    
    trainer = pl.Trainer(
        default_root_dir=default_root_dir,
        max_epochs=cfg.training.epochs,
        accelerator='gpu' if cfg.training.gpu else 'cpu',
        callbacks=[checkpoint_callback, lr_monitor],
        precision='16-mixed',
        log_every_n_steps=cfg.training.log_every_n_steps,
        logger=wandb_logger,
        benchmark=True,
        profiler='simple',
    )
    log_config(cfg)
    logging.info(f"Time to start train {time.process_time() - start_time} seconds")
    trainer.fit(model, dm)



@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="Al_classification")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    train_classification(cfg)
    logging.info('Train finished!')


if __name__ == "__main__":
    sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
    wandb.finish()
