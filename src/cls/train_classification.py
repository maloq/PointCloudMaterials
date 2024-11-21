import sys,os
sys.path.append(os.getcwd())
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.cls.lightning_module import PointNetClassifier
from src.cls.data_module import PointCloudDataModule
import hydra
from omegaconf import DictConfig
import logging
import wandb
torch.set_float32_matmul_precision('medium')


def train_classification(cfg: DictConfig):
    dm = PointCloudDataModule(batch_size=cfg.training.batch_size)
    
    model = PointNetClassifier(
        lr=cfg.training.learning_rate,
        use_normals=False,
        decay_rate=cfg.training.decay_rate
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',
        dirpath='checkpoints',
        filename='pointnet-{epoch:02d}-{val_acc:.2f}',
        save_top_k=1,
        mode='max',
    )
    
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator='gpu' if cfg.training.gpu else 'cpu',
        callbacks=[checkpoint_callback],
        precision='16-mixed'
    )
    trainer.fit(model, dm)



@hydra.main(version_base=None, config_path=os.path.join(os.getcwd(),"configs"), config_name="train_cls")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s-%(message)s')
    train_classification(cfg)
    logging.info('Train finished!')


if __name__ == "__main__":
    sys.argv.append('hydra.run.dir=output/${now:%Y-%m-%d}/${now:%H-%M-%S}')
    main()
    wandb.finish()
