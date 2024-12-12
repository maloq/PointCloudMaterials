import torch
import torch.nn as nn
import torch.nn.functional as F
import sys,os
sys.path.append(os.getcwd())
from src.loss.regularization_loss import feature_transform_reguliarzer


class point_loss_classification(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(point_loss_classification, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


