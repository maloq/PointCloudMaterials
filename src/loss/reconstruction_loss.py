import sys, os
import math
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
                        point_reduction: str = 'mean'):
    """
    Pure-PyTorch implementation of the symmetric Chamfer distance.

    Args
    ----
    pred, target : (B, N, 3) or (B, 3, N) tensors
        Point-clouds of the same batch size B.  N/M may differ.
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
    # dists = torch.sqrt(dists2 + 1e-8) if not squared else dists2
    dists = torch.sqrt(dists2 + 1e-8)
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

    blur = _coerce_positive_float(blur, 0.02)
    scaling = _coerce_float(scaling, 0.5)
    if not (0.0 < scaling < 1.0):
        scaling = 0.5
    diameter = _estimate_diameter(pred, target, blur, eps=1e-6)

    loss = geomloss.SamplesLoss(loss="sinkhorn", p=p, blur=blur,
                                scaling=scaling, diameter=diameter,
                                backend="tensorized")
    
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


def _pairwise_distances(
    pc: torch.Tensor,
    *,
    use_upper: bool = True,
    exclude_self: bool = True,
) -> torch.Tensor:
    pc = _to_B_N_3(pc)
    B, N, _ = pc.shape
    if N < 2:
        return pc.new_zeros((B, 0))
    dists = torch.cdist(pc, pc, p=2)
    if use_upper:
        mask = torch.triu(torch.ones(N, N, device=pc.device, dtype=torch.bool), diagonal=1)
        return dists[:, mask]
    if exclude_self:
        mask = ~torch.eye(N, device=pc.device, dtype=torch.bool)
        return dists[:, mask]
    return dists.reshape(B, -1)


def _soft_histogram(
    distances: torch.Tensor,
    bin_centers: torch.Tensor,
    sigma: float,
    *,
    normalize: bool = True,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if distances.numel() == 0:
        return distances.new_zeros((distances.shape[0], bin_centers.shape[0]))
    diff = distances.unsqueeze(-1) - bin_centers.view(1, 1, -1)
    sigma = max(float(sigma), eps)
    weights = torch.exp(-0.5 * (diff / sigma) ** 2)
    if normalize:
        weights = weights / (weights.sum(dim=-1, keepdim=True) + eps)
    if mask is not None:
        weights = weights * mask.unsqueeze(-1).to(weights.dtype)
        denom = mask.sum(dim=1, keepdim=True).clamp_min(eps)
    else:
        denom = float(distances.shape[1])
    return weights.sum(dim=1) / denom


def _resolve_r_max(
    r_max: float | None,
    pred_dists: torch.Tensor,
    target_dists: torch.Tensor,
    *,
    default: float,
) -> float:
    if r_max is None:
        if pred_dists.numel() == 0 and target_dists.numel() == 0:
            return default
        with torch.no_grad():
            combined = torch.cat([pred_dists, target_dists], dim=1)
            r_max = float(combined.max().item())
    else:
        r_max = _coerce_float(r_max, default)
    if not math.isfinite(r_max) or r_max <= 0.0:
        return default
    return r_max


def pairwise_distance_distribution_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    n_bins: int = 64,
    r_max: float | None = None,
    r_min: float = 0.0,
    sigma: float | None = None,
    normalize: bool = True,
    loss_type: str = "l2",
    use_upper: bool = True,
    clamp: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = _to_B_N_3(pred)
    target = _to_B_N_3(target)
    if pred.shape[1] < 2 or target.shape[1] < 2:
        return pred.new_tensor(0.0)

    pred_d = _pairwise_distances(pred, use_upper=use_upper, exclude_self=True)
    target_d = _pairwise_distances(target, use_upper=use_upper, exclude_self=True)
    r_max_val = _resolve_r_max(r_max, pred_d, target_d, default=2.0)
    if r_max_val <= r_min:
        return pred.new_tensor(0.0)

    n_bins = max(int(n_bins), 1)
    dr = (r_max_val - r_min) / float(n_bins)
    bin_centers = torch.linspace(
        r_min + 0.5 * dr,
        r_max_val - 0.5 * dr,
        n_bins,
        device=pred.device,
        dtype=pred.dtype,
    )
    if sigma is None or sigma <= 0:
        sigma = 0.5 * dr

    if clamp:
        pred_d = pred_d.clamp(min=r_min, max=r_max_val)
        target_d = target_d.clamp(min=r_min, max=r_max_val)

    hist_pred = _soft_histogram(pred_d, bin_centers, sigma, normalize=normalize, eps=eps)
    hist_target = _soft_histogram(target_d, bin_centers, sigma, normalize=normalize, eps=eps)

    if loss_type.lower() in {"l1", "mae"}:
        return torch.mean(torch.abs(hist_pred - hist_target))
    return torch.mean((hist_pred - hist_target) ** 2)


def rdf_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    n_bins: int = 64,
    r_max: float | None = None,
    r_min: float = 0.0,
    sigma: float | None = None,
    reference: str = "origin",
    exclude_self: bool = True,
    shell_normalize: bool = True,
    normalize: bool = True,
    loss_type: str = "l2",
    clamp: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = _to_B_N_3(pred)
    target = _to_B_N_3(target)
    if pred.shape[1] < 2 or target.shape[1] < 2:
        return pred.new_tensor(0.0)

    if reference == "centroid":
        pred_center = pred.mean(dim=1, keepdim=True)
        target_center = target.mean(dim=1, keepdim=True)
    else:
        pred_center = pred.new_zeros((pred.shape[0], 1, 3))
        target_center = target.new_zeros((target.shape[0], 1, 3))

    pred_r = torch.linalg.norm(pred - pred_center, dim=-1)
    target_r = torch.linalg.norm(target - target_center, dim=-1)

    pred_mask = pred_r > eps if exclude_self else None
    target_mask = target_r > eps if exclude_self else None

    r_max_val = _resolve_r_max(r_max, pred_r, target_r, default=1.0)
    if r_max_val <= r_min:
        return pred.new_tensor(0.0)

    n_bins = max(int(n_bins), 1)
    dr = (r_max_val - r_min) / float(n_bins)
    bin_centers = torch.linspace(
        r_min + 0.5 * dr,
        r_max_val - 0.5 * dr,
        n_bins,
        device=pred.device,
        dtype=pred.dtype,
    )
    if sigma is None or sigma <= 0:
        sigma = 0.5 * dr

    if clamp:
        pred_r = pred_r.clamp(min=r_min, max=r_max_val)
        target_r = target_r.clamp(min=r_min, max=r_max_val)

    hist_pred = _soft_histogram(pred_r, bin_centers, sigma, normalize=False, mask=pred_mask, eps=eps)
    hist_target = _soft_histogram(target_r, bin_centers, sigma, normalize=False, mask=target_mask, eps=eps)

    if shell_normalize:
        shell = (4.0 * math.pi) * (bin_centers ** 2) * dr
        hist_pred = hist_pred / (shell + eps)
        hist_target = hist_target / (shell + eps)

    if normalize:
        hist_pred = hist_pred / (hist_pred.sum(dim=1, keepdim=True) + eps)
        hist_target = hist_target / (hist_target.sum(dim=1, keepdim=True) + eps)

    if loss_type.lower() in {"l1", "mae"}:
        return torch.mean(torch.abs(hist_pred - hist_target))
    return torch.mean((hist_pred - hist_target) ** 2)


def _coerce_float(value, default):
    if torch.is_tensor(value):
        value = value.item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(value):
        return default
    return value


def _coerce_positive_float(value, default):
    value = _coerce_float(value, default)
    if value <= 0.0:
        return default
    return value


def _estimate_diameter(pred: torch.Tensor, target: torch.Tensor, blur: float, eps: float = 1e-6) -> float:
    mins = torch.minimum(pred.amin(dim=1), target.amin(dim=1))
    maxs = torch.maximum(pred.amax(dim=1), target.amax(dim=1))
    diag = torch.linalg.norm(maxs - mins, dim=-1)
    diameter = float(diag.max().item())
    if not math.isfinite(diameter) or diameter <= 0.0:
        return max(blur, eps)
    return max(diameter, blur, eps)



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
