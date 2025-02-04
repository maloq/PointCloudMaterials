import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy, Precision, Recall, AUROC
from src.loss.reconstruction_loss import *
from models.autoencoders_nn.pointnet_autoencoder import build_model


class PointNetAutoencoder(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.model = build_model(cfg)
        self.criterion = chamfer_wasserstein_loss

            
    def forward(self, x):
        return self.model(x)        
    

    def training_step(self, batch, batch_idx):
        points, _ = batch
        pred, trans_feat_list = self(points.permute(0,2,1))
        loss, rec_loss, feature_transform_loss = self.criterion(points.float(), pred.float(), trans_feat_list=trans_feat_list)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_rec_loss', rec_loss, prog_bar=True)
        self.log('train_feature_transform_loss', feature_transform_loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        points, _ = batch
        pred, trans_feat_list = self(points.permute(0,2,1))
        loss, rec_loss, feature_transform_loss = self.criterion(points.float(), pred.float(), trans_feat_list=trans_feat_list)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_rec_loss', rec_loss, prog_bar=True)
        self.log('val_feature_transform_loss', feature_transform_loss, prog_bar=True)
        return {'val_loss': loss}    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.training.learning_rate,
            weight_decay=self.hparams.training.decay_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return [optimizer], [scheduler]
        