import sys,os
sys.path.append(os.getcwd())
import torch
from pytorch3d.loss import chamfer_distance
from src.loss.regularization_loss import feature_transform_reguliarzer


def point_reconstruction_loss_regularized(pred, target, trans_feat, feature_transform_loss_scale=0.001):

    loss, _ = chamfer_distance(pred, target) 
    feature_transform_loss = feature_transform_reguliarzer(trans_feat)
    total_loss = loss + feature_transform_loss * feature_transform_loss_scale
    return total_loss


def point_reconstruction_loss(pred, target, trans_feat, feature_transform_loss_scale=0.001):
    loss, _ = chamfer_distance(pred, target) 
    return loss