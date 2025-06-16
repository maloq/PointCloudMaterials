import sys, os
sys.path.append(os.getcwd())
import torch
try:
    from pytorch3d.loss import chamfer_distance as chamfer_distance  
    pytorch3d_available = True
except Exception:                         
    pytorch3d_available = False

import torch.nn as nn
from src.utils.logging_config import setup_logging
logger = setup_logging()


if not pytorch3d_available:

    def chamfer_distance(pred: torch.Tensor,
                         target: torch.Tensor,
                         *,
                         squared: bool = True):
        """
        Pure-PyTorch implementation of the symmetric Chamfer distance.

        Args
        ----
        pred, target : (B, N, 3) or (B, 3, N) tensors
            Point-clouds of the same batch size B.  N/M may differ.
        squared : bool, default=True
            If True (default) uses squared Euclidean distances—matching the
            behaviour of PyTorch3D.  Set to False for plain L2.

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

        cd = min_pred2gt.mean(1) + min_gt2pred.mean(1)            # (B,)
        return cd.mean(), None
        


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
    loss, _ = chamfer_distance(pred, target)
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
    loss, _ = chamfer_distance(pred, target)
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
    loss, _ = chamfer_distance(pred, target)
    encoder_trans = kwargs['trans_feat_list'][0]
    reg_loss = feature_transform_regularizer(encoder_trans)
    total_loss = loss + kwargs['feature_transform_loss_scale'] * reg_loss
    
    repulsion_h = kwargs.get('repulsion_h', 0.05)
    repulsion_scale = kwargs.get('repulsion_scale', 0.1)
    
    repulsion_loss_val = repulsion_loss(pred, h=repulsion_h)
    total_loss += repulsion_scale * repulsion_loss_val
    
    aux_loss_dict = {'ft_loss': reg_loss, 'repulsion_loss': repulsion_loss_val}
    return _add_optional_losses(total_loss, aux_loss_dict, **kwargs)


def chamfer_elbo_loss(pred, target, mu, logvar, **kwargs):
    """
    Computes the VAE ELBO loss: Chamfer reconstruction loss + KL divergence.
    Optionally includes feature transform regularization and L1 latent regularization.

    Args:
        pred (Tensor): Predicted point clouds.
        target (Tensor): Ground truth point clouds.
        mu (Tensor): Latent space mean.
        logvar (Tensor): Latent space log variance.
        **kwargs: Must include 'kl_beta'. Can include 'trans_feat_list', 
                  'feature_transform_loss_scale', 'latent', 'l1_latent_loss_scale'.

    Returns:
        tuple: (Total ELBO loss, dictionary of auxiliary losses).
    """
    # Reconstruction loss
    recon_loss, _ = chamfer_distance(pred, target)

    # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # Sum over latent dimensions, then mean over batch
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kld_loss = torch.mean(kld_loss)

    kl_beta = kwargs.get('kl_beta', 1.0)
    
    total_loss = recon_loss + kl_beta * kld_loss
    aux_loss_dict = {'kld_loss': kld_loss}

    # Optional: Feature transform regularization
    feature_transform_loss = torch.tensor(0.0, device=pred.device)
    if 'trans_feat_list' in kwargs and kwargs['trans_feat_list'] and \
       'feature_transform_loss_scale' in kwargs and kwargs['feature_transform_loss_scale'] > 0:
        
        # Ensure trans_feat_list is not empty before trying to access its elements
        if isinstance(kwargs['trans_feat_list'], (list, tuple)) and len(kwargs['trans_feat_list']) > 0:
            # Assuming the relevant transform is the first, typically from the encoder
            feature_transform_loss = feature_transform_regularizer(kwargs['trans_feat_list'][0])
            total_loss += kwargs['feature_transform_loss_scale'] * feature_transform_loss
            aux_loss_dict['ft_loss'] = feature_transform_loss
        else:
            logger.debug("trans_feat_list provided but empty or not a list/tuple, skipping ft_loss for ELBO.")
            
    return _add_optional_losses(total_loss, aux_loss_dict, **kwargs)


EPS = 1e-10

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




def elbo_swd_repulsion_loss(pred: torch.Tensor, target: torch.Tensor,
                            mu: torch.Tensor, logvar: torch.Tensor, *,
                            kl_beta: float = 1.0,
                            num_projections: int = 64,
                            repulsion_h: float = 0.05,
                            repulsion_scale: float = 0.1,
                            **kwargs):
    """
    ELBO loss using SWD and repulsion. Optionally L1 regularized.
    """
    swd_loss = sliced_wasserstein_distance(pred, target, num_projections=num_projections)
    rep_loss = repulsion_loss(pred, h=repulsion_h)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    total_loss = swd_loss + repulsion_scale * rep_loss + kl_beta * kld_loss
    
    aux_loss_dict = {
        'swd_loss': swd_loss,
        'repulsion_loss': rep_loss,
        'kld_loss': kld_loss
    }
    
    return _add_optional_losses(total_loss, aux_loss_dict, **kwargs)


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
    return total_loss, aux_loss_dict