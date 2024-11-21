import os
import sys
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.models.point_net.pointnet_cls import PointNet, point_loss
from src.models.point_net2.pointnet2_cls_ssg import PointNet2, point_loss2


class PointNetClassifier(pl.LightningModule):
    def __init__(self, num_classes=2, lr=0.006, use_normals=False, decay_rate=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = PointNet(num_classes, normal_channel=use_normals)
        self.criterion = point_loss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        points, target = batch
        points = points.transpose(2, 1)
        pred, trans_feat = self(points)
        loss = self.criterion(pred, target.long(), trans_feat)
        
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        acc = correct.item() / float(points.size()[0])
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        points, target = batch
        points = points.transpose(2, 1)
        pred, trans_feat = self(points)
        loss = self.criterion(pred, target.long(), trans_feat)
        
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        acc = correct.item() / float(points.size()[0])
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.decay_rate
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
        return [optimizer], [scheduler]