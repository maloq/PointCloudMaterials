import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy, Precision, Recall, AUROC
from src.models.point_net.pointnet_autoencoder import DummyPointCloudAE, PointNetAE, PointNetAE_MLP
from src.loss.reconstruction_loss import point_reconstruction_loss_regularized_AE
from src.loss.regularization_loss import feature_transform_reguliarzer



class PointNetAutoencoder(pl.LightningModule):

    def __init__(self,lr=0.006, decay_rate=1e-4, latent_size=64):
            super().__init__()
            self.save_hyperparameters()
            # self.model = DummyPointCloudAE(point_size=64, latent_size=latent_size)
            self.model = PointNetAE(point_size=64, latent_size=latent_size)
            self.criterion = point_reconstruction_loss_regularized_AE
            

    def forward(self, x):
        return self.model(x)        
    

    def training_step(self, batch, batch_idx):
        points, _ = batch
        pred, trans_feat_list = self(points.permute(0,2,1))
        loss = self.criterion(points.float(), pred.float(), trans_feat_list)
        self.log('train_loss', loss, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        points, _ = batch
        pred, trans_feat_list = self(points.permute(0,2,1))
        loss = self.criterion(points.float(), pred.float(), trans_feat_list)
        self.log('val_loss', loss, prog_bar=True)
        return {'val_loss': loss}    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.decay_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        return [optimizer], [scheduler]
        