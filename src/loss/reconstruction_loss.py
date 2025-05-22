import sys,os
sys.path.append(os.getcwd())
import torch
from pytorch3d.loss import chamfer_distance
import torch.nn as nn
from src.utils.logging_config import setup_logging
logger = setup_logging()



def chamfer_loss(pred, target, **kwargs):
    """
    Computes the Chamfer distance loss between predictions and targets.

    Args:
        pred (Tensor): Predicted point cloud.
        target (Tensor): Ground truth point cloud.
        **kwargs: Additional keyword arguments (currently ignored).

    Returns:
        tuple: (Chamfer distance loss, empty dictionary for auxiliary losses).
    """
    loss, _ = chamfer_distance(pred, target)
    return loss, {} # Return empty dict for aux losses


def chamfer_regularized_encoder_loss(pred, target, **kwargs):
    """
    Computes the Chamfer distance loss with feature transform regularization for the encoder.

    Args:
        pred (Tensor): Predicted point clouds.
        target (Tensor): Ground truth point clouds.
        trans_feat_list (list or tuple): Contains at least the encoder's feature transform as the first element.
        feature_transform_loss_scale (float): Scale factor for the feature transform regularization loss.

    Returns:
        tuple: (Total loss, dictionary containing {'ft_loss': regularization_loss}).
    """
    loss, _ = chamfer_distance(pred, target)
    encoder_trans = kwargs['trans_feat_list'][0]
    reg_loss = feature_transform_regularizer(encoder_trans)
    total_loss = loss + kwargs['feature_transform_loss_scale'] * reg_loss
    aux_loss_dict = {'ft_loss': reg_loss}
    return total_loss, aux_loss_dict


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
    return total_loss, aux_loss_dict


def chamfer_elbo_loss(pred, target, mu, logvar, **kwargs):
    """
    Computes the VAE ELBO loss: Chamfer reconstruction loss + KL divergence.
    Optionally includes feature transform regularization.

    Args:
        pred (Tensor): Predicted point clouds.
        target (Tensor): Ground truth point clouds.
        mu (Tensor): Latent space mean.
        logvar (Tensor): Latent space log variance.
        **kwargs: Must include 'kl_beta'. Can include 'trans_feat_list', 
                  'feature_transform_loss_scale'.

    Returns:
        tuple: (Total ELBO loss, dictionary of auxiliary losses {'kld_loss': unscaled_kld, 'ft_loss': unscaled_ft_loss}).
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
            
    return total_loss, aux_loss_dict


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
    """ELBO with SWD reconstruction + repulsion term and annealed KL."""
    swd = sliced_wasserstein_distance(pred, target, num_projections=num_projections)
    rep = repulsion_loss(pred, h=repulsion_h)
    recon = swd + repulsion_scale * rep

    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total = recon + kl_beta * kld
    return total, {'swd': swd, 'repulsion': rep, 'kld': kld}