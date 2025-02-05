import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from src.loss.reconstruction_loss import *
from src.models.autoencoders_nn.pointnet_autoencoder import build_model
from src.utils.logging_config import setup_logging  
logger = setup_logging()


class PointNetAutoencoder(pl.LightningModule):

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
        self.criterion = chamfer_regularized_encoder_loss
        self.density = self.compute_density()
        logger.print(f"Loss: {self.criterion.__name__}")


    def compute_density(self):
        volume = (4/3) * torch.pi * (self.sphere_radius**3)
        density = self.num_points / volume
        return density
    
    def forward(self, x):
        return self.model(x)        
    
    def training_step(self, batch, batch_idx):
        points, _ = batch
        pred, trans_feat_list = self(points.permute(0,2,1))
        loss, aux_loss = self.criterion(points.float(),
                                        pred.float(),
                                        trans_feat_list=trans_feat_list,
                                        density=self.density,
                                        reconstruction_loss_scale=self.reconstruction_loss_scale,
                                        feature_transform_loss_scale=self.feature_transform_loss_scale)
        if len(aux_loss) == 2:
            rec_loss, feature_transform_loss = aux_loss
            self.log('train_aux_rec_loss', rec_loss, prog_bar=True)
        elif len(aux_loss) == 1:
            feature_transform_loss = aux_loss[0]
        else:
            pass
            
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_ft_loss', feature_transform_loss, prog_bar=False)
        return {'loss': loss}  
      

    def validation_step(self, batch, batch_idx):
        points, _ = batch
        pred, trans_feat_list = self(points.permute(0,2,1))
        loss, aux_loss = self.criterion(points.float(),
                                        pred.float(),
                                        trans_feat_list=trans_feat_list,
                                        density=self.density,
                                        reconstruction_loss_scale=self.reconstruction_loss_scale,
                                        feature_transform_loss_scale=self.feature_transform_loss_scale)
        if len(aux_loss) == 2:
            rec_loss, feature_transform_loss = aux_loss
            self.log('val_aux_rec_loss', rec_loss, prog_bar=True)
        else:
            feature_transform_loss = aux_loss[0]

        self.log('val_loss', loss, prog_bar=True)
        self.log('val_ft_loss', feature_transform_loss, prog_bar=True)
        return {'val_loss': loss}    
    
    
    def configure_optimizers(self):
        if self.hparams.training.scheduler_name == 'step':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.training.learning_rate,
                weight_decay=self.hparams.training.decay_rate
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=self.hparams.training.scheduler_gamma)
            return [optimizer], [scheduler]
        elif self.hparams.training.scheduler_name == 'onecycle':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.training.learning_rate,
                weight_decay=self.hparams.training.decay_rate
            )
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.hparams.training.learning_rate, steps_per_epoch=5, epochs=self.hparams.training.epochs)
            return [optimizer], [scheduler]
        else:
            raise ValueError(f"Scheduler {self.hparams.training.scheduler_name} not found")
