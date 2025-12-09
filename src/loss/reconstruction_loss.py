import sys, os
sys.path.append(os.getcwd())
import torch

import geomloss


import torch.nn as nn
from src.utils.logging_config import setup_logging
logger = setup_logging()
EPS = 1e-10


def chamfer_distance(pred: torch.Tensor,
                        target: torch.Tensor,
                        *,
                        squared: bool = True,
                        point_reduction: str = 'mean'):
    """
    Pure-PyTorch implementation of the symmetric Chamfer distance.

    Args
    ----
    pred, target : (B, N, 3) or (B, 3, N) tensors
        Point-clouds of the same batch size B.  N/M may differ.
    squared : bool, default=True
        If True (default) uses squared Euclidean distances—matching the
        behaviour of PyTorch3D.  Set to False for plain L2.
    point_reduction : str, default='mean'
        Reduction mode for the points dimension: 'mean' or 'sum'.

    Returns
    -------
    cd : torch.Tensor
        Scalar Chamfer distance averaged over the batch.
    aux : None
        Placeholder to stay API-compatible with PyTorch3D.
    """
    # Ensure both clouds are (B, *, 3)
    pred   = _to_B_N_3(pred)
    target = _to_B_N_3(target)

    # ------------------------------------------------------------------
    # fast squared Euclidean distances:  ‖x‖² + ‖y‖² − 2 x·yᵀ
    # ------------------------------------------------------------------
    B, N, _ = pred.shape
    M       = target.shape[1]

    # ‖x‖²   (B, N, 1)
    pred_sq   = (pred   ** 2).sum(-1, keepdim=True)
    # ‖y‖²   (B, 1, M)
    target_sq = (target ** 2).sum(-1).unsqueeze(1)

    # -2 x·yᵀ   (B, N, M)
    cross = -2.0 * torch.bmm(pred, target.transpose(1, 2))

    dists2 = pred_sq + target_sq + cross                      # (B, N, M)

    # Numerical stability – make sure we do not go below zero
    dists2 = dists2.clamp_min(0.)

    # If the caller wants plain L2, take the sqrt **once**
    dists = torch.sqrt(dists2 + 1e-8) if not squared else dists2
    # ------------------------------------------------------------------

    # Closest distances each way
    min_pred2gt = dists.min(dim=2)[0]     # (B, N)
    min_gt2pred = dists.min(dim=1)[0]     # (B, M)

    if point_reduction == 'sum':
        cd = min_pred2gt.sum(1) + min_gt2pred.sum(1)          # (B,)
    else:
        cd = min_pred2gt.mean(1) + min_gt2pred.mean(1)        # (B,)
    return cd.mean(), None
        

def sinkhorn_distance(pred: torch.Tensor,
                        target: torch.Tensor,
                        *,
                        blur=.02, p=2, scaling=.5):
    """
    Batched Sinkhorn distance between two point clouds.
    
    This is a wrapper around the `geomloss` library.

    Args
    ----
    pred, target : (B, N, 3) or (B, 3, N) tensors
        Point-clouds of the same batch size B.  N/M may differ.
    blur : float, default .05
        Sinkhorn softness parameter.
    p : int, default 2
        Power of the Euclidean distance for the cost.
    scaling : float, default .5
        Scaling of the cost matrix.

    Returns
    -------
    shd : torch.Tensor
        Scalar Sinkhorn distance averaged over the batch.
    aux : None
        Placeholder to stay API-compatible with PyTorch3D.
    """

    pred = _to_B_N_3(pred)
    target = _to_B_N_3(target)

    loss = geomloss.SamplesLoss(loss="sinkhorn", p=p, blur=blur,
                                scaling=scaling, backend="tensorized")
    
    return loss(pred, target).mean(), None


def chamfer_loss(pred, target, **kwargs):
    """
    Computes the Chamfer distance loss between predictions and targets.
    Optionally includes L1 regularization on the latent space.

    Args:
        pred (Tensor): Predicted point cloud.
        target (Tensor): Ground truth point cloud.
        **kwargs: Additional keyword arguments, can include 'latent' and 'l1_latent_loss_scale'.

    Returns:
        tuple: (Total loss, dictionary for auxiliary losses).
    """
    point_reduction = kwargs.get('point_reduction', 'mean')
    loss, _ = chamfer_distance(pred, target, point_reduction=point_reduction)
    aux_loss_dict = {}
    return _add_optional_losses(loss, aux_loss_dict, **kwargs)


def sinkhorn_loss(pred, target, **kwargs):
    """
    Computes the Sinkhorn distance loss between predictions and targets.
    Optionally includes L1/KL regularization on the latent space.

    Args:
        pred (Tensor): Predicted point cloud.
        target (Tensor): Ground truth point cloud.
        **kwargs: Additional keyword arguments, can include 'latent', 
                  'l1_latent_loss_scale', and 'kl_latent_loss_scale'.

    Returns:
        tuple: (Total loss, dictionary for auxiliary losses).
    """
    loss, _ = sinkhorn_distance(pred, target)
    aux_loss_dict = {}
    return _add_optional_losses(loss, aux_loss_dict, **kwargs)


def chamfer_regularized_encoder_loss(pred, target, **kwargs):
    """
    Computes the Chamfer distance loss with feature transform regularization for the encoder.
    Optionally includes L1 regularization on the latent space.

    Args:
        pred (Tensor): Predicted point clouds.
        target (Tensor): Ground truth point clouds.
        trans_feat_list (list or tuple): Contains the encoder's feature transform.
        feature_transform_loss_scale (float): Scale factor for the feature transform regularization.
        **kwargs: Can also include 'latent' and 'l1_latent_loss_scale'.

    Returns:
        tuple: (Total loss, dictionary containing auxiliary losses).
    """
    point_reduction = kwargs.get('point_reduction', 'mean')
    loss, _ = chamfer_distance(pred, target, point_reduction=point_reduction)
    encoder_trans = kwargs['trans_feat_list'][0]
    reg_loss = feature_transform_regularizer(encoder_trans)
    total_loss = loss + kwargs['feature_transform_loss_scale'] * reg_loss
    aux_loss_dict = {'ft_loss': reg_loss}
    return _add_optional_losses(total_loss, aux_loss_dict, **kwargs)


def feature_transform_regularizer(trans):
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device, dtype=trans.dtype)[None, :, :] # (1, d, d)

    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss


def chamfer_regularized_encoder_loss_repulsion(pred, target, **kwargs):
    """
    Computes the Chamfer distance loss with feature transform regularization for the encoder.
    Optionally includes repulsion loss.

    Args:
        pred (Tensor): Predicted point clouds.
        target (Tensor): Ground truth point clouds.
        trans_feat_list (list or tuple): Contains at least the encoder's feature transform as the first element.
        feature_transform_loss_scale (float): Scale factor for the feature transform regularization loss.
        **kwargs: Can include 'repulsion_h' (float, default 0.05) and 
                  'repulsion_scale' (float, default 0.1).


    Returns:
        tuple: (Total loss, dictionary containing {'ft_loss': regularization_loss, 'repulsion_loss': repulsion_loss_val}).
    """
    point_reduction = kwargs.get('point_reduction', 'mean')
    loss, _ = chamfer_distance(pred, target, point_reduction=point_reduction)
    encoder_trans = kwargs['trans_feat_list'][0]
    reg_loss = feature_transform_regularizer(encoder_trans)
    total_loss = loss + kwargs['feature_transform_loss_scale'] * reg_loss
    
    repulsion_h = kwargs.get('repulsion_h', 0.05)
    repulsion_scale = kwargs.get('repulsion_scale', 0.1)
    
    repulsion_loss_val = repulsion_loss(pred, h=repulsion_h)
    total_loss += repulsion_scale * repulsion_loss_val
    
    aux_loss_dict = {'ft_loss': reg_loss, 'repulsion_loss': repulsion_loss_val}
    return _add_optional_losses(total_loss, aux_loss_dict, **kwargs)


def _to_B_N_3(pc: torch.Tensor) -> torch.Tensor:
    """Ensure tensor shape is (B, N, 3). Accepts (B, 3, N)."""
    if pc.dim() != 3:
        raise ValueError("Point cloud must be (B,N,3) or (B,3,N)")
    if pc.shape[1] == 3 and pc.shape[2] != 3:
        pc = pc.transpose(1, 2).contiguous()
    return pc



def sliced_wasserstein_distance(pc1: torch.Tensor, pc2: torch.Tensor,
                                num_projections: int = 64, p: int = 2) -> torch.Tensor:
    """Compute batched SWD‑p (default p=2)."""
    pc1 = _to_B_N_3(pc1)
    pc2 = _to_B_N_3(pc2)
    B, N, _ = pc1.shape

    device = pc1.device
    dirs = torch.randn(num_projections, 3, device=device)
    dirs = dirs / (dirs.norm(dim=1, keepdim=True) + EPS)

    proj1 = torch.matmul(pc1, dirs.t())   # (B,N,K)
    proj2 = torch.matmul(pc2, dirs.t())
    proj1, _ = torch.sort(proj1, dim=1)
    proj2, _ = torch.sort(proj2, dim=1)

    if p == 1:
        dist = torch.abs(proj1 - proj2).mean(dim=1)   # (B,K)
    else:
        dist = ((proj1 - proj2) ** p).mean(dim=1)

    return dist.mean()                                # scalar



def repulsion_loss(pc: torch.Tensor, h: float = 0.05) -> torch.Tensor:
    pc = _to_B_N_3(pc)
    B, N, _ = pc.shape
    dists = torch.cdist(pc, pc, p=2)                  # (B,N,N)
    eye = torch.eye(N, device=pc.device).bool()
    dists = dists.masked_fill(eye.unsqueeze(0), 1e9)
    return torch.exp(- (dists ** 2) / (h ** 2)).mean()


def swd_repulsion_loss(pred: torch.Tensor, target: torch.Tensor, *,
                       num_projections: int = 64,
                       repulsion_h: float = 0.05,
                       repulsion_scale: float = 0.1):
    """Sliced‑Wasserstein + repulsion."""
    swd = sliced_wasserstein_distance(pred, target, num_projections=num_projections)
    rep = repulsion_loss(pred, h=repulsion_h)
    return swd + repulsion_scale * rep, {'swd': swd, 'repulsion': rep}


def kl_latent_regularizer(latent: torch.Tensor) -> torch.Tensor:
    """
    Deterministic β-VIB penalty.
    KL( N(latent, I) ‖ N(0, I) )  =  0.5 · ‖latent‖²  (up to a constant)
    """
    return 0.5 * torch.mean(torch.sum(latent ** 2, dim=1))


def l1_latent_regularizer(latent):
    """Computes the L1 regularization loss on a latent vector."""
    return torch.mean(torch.abs(latent))

def _add_optional_losses(total_loss, aux_loss_dict, **kwargs):
    """Helper to add optional losses like L1 latent regularization."""
    l1_latent_loss_scale = kwargs.get('l1_latent_loss_scale', 0.0)
    if l1_latent_loss_scale > 0 and 'latent' in kwargs and kwargs['latent'] is not None:
        l1_loss = l1_latent_regularizer(kwargs['latent'])
        total_loss += l1_latent_loss_scale * l1_loss
        aux_loss_dict['l1_latent_loss'] = l1_loss

    kl_latent_loss_scale = kwargs.get('kl_latent_loss_scale', 0.0)
    if kl_latent_loss_scale > 0 and 'latent' in kwargs and kwargs['latent'] is not None:
        kl_loss = kl_latent_regularizer(kwargs['latent'])
        total_loss += kl_latent_loss_scale * kl_loss
        aux_loss_dict['kl_latent_loss'] = kl_loss
        
    return total_loss, aux_loss_dict


def rotation_geodesic_kabsch_loss(rot_pred: torch.Tensor,
                                  cano_points: torch.Tensor,
                                  target_points: torch.Tensor,
                                  detach_teacher: bool = True,
                                  eps: float = 1e-6) -> torch.Tensor:
    """
    Geodesic SO(3) loss between predicted rotation R and Kabsch teacher R*.
    Args:
        rot_pred:      (B, 3, 3)  predicted rotation (need not be perfect SO(3), you can orthogonalize upstream)
        cano_points:   (B, N, 3)  canonical reconstruction (pre-rotation)
        target_points: (B, N, 3)  target point cloud
        detach_teacher: if True, stop gradient through R* (recommended)
    Returns:
        Mean geodesic angle (radians) over the batch.
    """
    with torch.no_grad() if detach_teacher else torch.enable_grad():
        R_star = _kabsch_so3_teacher(cano_points, target_points)
    if detach_teacher:
        R_star = R_star.detach()

    # geodesic angle between R and R*:  theta = arccos( (trace(R^T R*) - 1)/2 )
    RtR = torch.matmul(rot_pred.transpose(-1, -2).to(torch.float32), R_star.to(torch.float32))  # (B,3,3)
    tr = RtR.diagonal(dim1=-2, dim2=-1).sum(-1)                                                 # (B,)
    # numeric guards
    tr = tr.clamp(min=-1.0, max=3.0)
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + eps, 1.0 - eps)
    theta = torch.arccos(cos_theta)                                                             # (B,)
    return theta.mean()