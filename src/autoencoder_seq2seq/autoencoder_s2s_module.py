import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from src.loss.reconstruction_loss_s2s import mse_loss, mse_kl_divergence_loss, mse_loss_l1reg
from src.models.autoencoders_nn.seq2seq_autoencoder import build_model
from src.utils.logging_config import setup_logging  
from src.utils.optimizer_utils import get_optimizers_and_scheduler
logger = setup_logging()


class AutoencoderSeq2Seq(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = build_model(cfg)
        self.sphere_radius = cfg.data.radius
        self.num_points = cfg.data.num_points
        self.dr = cfg.data.dr
        self.reconstruction_loss_scale = cfg.reconstruction_loss_scale
        self.feature_transform_loss_scale = cfg.feature_transform_loss_scale
        if cfg.torch_compile:
            self.model = torch.compile(self.model)

        if cfg.loss == 'chamfer_kl_divergence_loss':
            self.criterion = mse_kl_divergence_loss
        elif cfg.loss == 'mse_loss':
            self.criterion = mse_loss
        elif cfg.loss == 'mse_loss_l1reg':
            self.criterion = mse_loss_l1reg
        else:
            raise ValueError(f"Loss function {cfg.loss} not supported")
        self.density = self.compute_density()

        # try:
        #     logger.print(f"Loss: {self.criterion.__name__}")
        # except:
        #     pass


    def compute_density(self):
        volume = (4/3) * torch.pi * (self.sphere_radius**3)
        density = self.num_points / volume
        return density
    
    def forward(self, x, return_latent: bool = False):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        points, _ = batch
        pred, latent = self(points)
        loss, aux_loss = self.criterion(points.float(),
                                        pred.float(),
                                        latent=latent,
                                        density=self.density,
                                        dr=self.dr,
                                        sphere_radius=self.sphere_radius,
                                        reconstruction_loss_scale=self.reconstruction_loss_scale)
        if len(aux_loss) == 1:
            self.log('train_aux_rec_loss', aux_loss[0], prog_bar=True)
        else:
            pass
            
        self.log('train_loss', loss, prog_bar=True)
        return {'loss': loss}  
      

    def validation_step(self, batch, batch_idx):
        points, _ = batch
        pred, latent = self(points)
        loss, aux_loss = self.criterion(points.float(),
                                        pred.float(),
                                        latent=latent,
                                        density=self.density,
                                        dr=self.dr,
                                        sphere_radius=self.sphere_radius,
                                        reconstruction_loss_scale=self.reconstruction_loss_scale)
        if len(aux_loss) == 1:
            self.log('val_aux_rec_loss', aux_loss[0], prog_bar=True)
        else:
            pass

        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss}    
    
    
    def configure_optimizers(self):
        return get_optimizers_and_scheduler(self)
