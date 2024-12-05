import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy, Precision, Recall, AUROC
from src.models.point_net.pointnet_autoencoder import PointCloudAE


class PointNetAutoencoder(pl.LightningModule):


    def __init__(self, num_classes=2, lr=0.006, use_normals=False, decay_rate=1e-4):
            super().__init__()
            self.save_hyperparameters()
            self.model = PointCloudAE(num_classes, normal_channel=use_normals)
            self.criterion = point_loss()
            
            # Metrics for each stage
            self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.train_precision = Precision(task="multiclass", num_classes=num_classes)
            self.train_recall = Recall(task="multiclass", num_classes=num_classes)
            self.train_auroc = AUROC(task="multiclass", num_classes=num_classes)
            
            self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
            self.val_precision = Precision(task="multiclass", num_classes=num_classes)
            self.val_recall = Recall(task="multiclass", num_classes=num_classes)
            self.val_auroc = AUROC(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)        
    
    def training_step(self, batch, batch_idx):
        points, target = batch
        points = points.transpose(2, 1)
        pred, trans_feat = self(points)
        loss = self.criterion(pred, target.long(), trans_feat)
        
        # Calculate metrics
        acc = self.train_accuracy(pred, target.long())
        prec = self.train_precision(pred, target.long())
        rec = self.train_recall(pred, target.long())
        auroc = self.train_auroc(pred, target.long())
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=False)
        self.log('train_precision', prec, prog_bar=False)
        self.log('train_recall', rec, prog_bar=False)
        self.log('train_auroc', auroc, prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        points, target = batch
        points = points.transpose(2, 1)
        pred, trans_feat = self(points)
        loss = self.criterion(pred, target.long(), trans_feat)
        
        # Calculate metrics
        acc = self.val_accuracy(pred, target.long())
        prec = self.val_precision(pred, target.long())
        rec = self.val_recall(pred, target.long())
        auroc = self.val_auroc(pred, target.long())
        
        self.log('val_loss', loss, prog_bar=False)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', prec, prog_bar=False)
        self.log('val_recall', rec, prog_bar=False)
        self.log('val_auroc', auroc, prog_bar=True)
        
        return {'val_loss': loss, 'val_acc': acc}    
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.decay_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        return [optimizer], [scheduler]
        