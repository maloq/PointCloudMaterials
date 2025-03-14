import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from src.loss.reconstruction_loss_s2s import *
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
        self.reconstruction_loss_scale = cfg.training.reconstruction_loss_scale
        self.feature_transform_loss_scale = cfg.training.feature_transform_loss_scale

        if cfg.model.torch_compile:
            self.model = torch.compile(self.model)
        if cfg.loss == 'chamfer_kl_divergence_loss':
            self.criterion = chamfer_kl_divergence_loss
        elif cfg.loss == 'mse_loss':
            self.criterion == mse_loss
        self.density = self.compute_density()
        logger.print(f"Loss: {self.criterion.__name__}")


    def compute_density(self):
        volume = (4/3) * torch.pi * (self.sphere_radius**3)
        density = self.num_points / volume
        return density
    
    def forward(self, x, return_latent: bool = False):
        if return_latent:
            latent = self.model.encoder(x)[0]
            return self.model(x)[0], latent
        else:
            return self.model(x)
    
    def training_step(self, batch, batch_idx):
        points, _ = batch
        pred, latent = self(points)
        loss, aux_loss = self.criterion(points.float(),
                                        pred.float(),
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
