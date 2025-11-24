"""
Pairwise Distance Loss for Point Clouds.

This loss ensures that the pairwise distances between points in the predicted
point cloud match those in the target point cloud, helping preserve local
geometric structure.
"""
import torch
import torch.nn as nn


def _to_B_N_3(pc: torch.Tensor) -> torch.Tensor:
    """Ensure tensor shape is (B, N, 3). Accepts (B, 3, N)."""
    if pc.dim() != 3:
        raise ValueError("Point cloud must be (B,N,3) or (B,3,N)")
    if pc.shape[1] == 3 and pc.shape[2] != 3:
        pc = pc.transpose(1, 2).contiguous()
    return pc


def pairwise_distance_matrix(pc: torch.Tensor, squared: bool = False) -> torch.Tensor:
    """
    Compute the pairwise distance matrix for a batch of point clouds.
    
    Args:
        pc: Point cloud tensor of shape (B, N, 3)
        squared: If True, return squared distances (default: False)
    
    Returns:
        Distance matrix of shape (B, N, N)
    """
    pc = _to_B_N_3(pc)
    # Using efficient computation: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x·y
    pc_sq = (pc ** 2).sum(dim=-1, keepdim=True)  # (B, N, 1)
    cross = torch.bmm(pc, pc.transpose(1, 2))     # (B, N, N)
    dist_sq = pc_sq + pc_sq.transpose(1, 2) - 2 * cross  # (B, N, N)
    dist_sq = dist_sq.clamp(min=0.0)  # Numerical stability
    
    if squared:
        return dist_sq
    return torch.sqrt(dist_sq + 1e-8)


def pairwise_distance_loss(pred: torch.Tensor, 
                           target: torch.Tensor,
                           *,
                           squared: bool = False,
                           normalize: bool = True,
                           p: int = 2) -> tuple[torch.Tensor, None]:
    """
    Compute the pairwise distance loss between predicted and target point clouds.
    
    This loss encourages the predicted point cloud to preserve the pairwise
    distance structure of the target point cloud.
    
    Args:
        pred: Predicted point cloud (B, N, 3) or (B, 3, N)
        target: Target point cloud (B, N, 3) or (B, 3, N)
        squared: If True, compare squared distances (default: False)
        normalize: If True, normalize distances by the max distance (default: True)
        p: Norm type for comparison (1 for L1, 2 for L2) (default: 2)
    
    Returns:
        loss: Scalar loss value averaged over the batch
        aux: None (for API compatibility)
    """
    pred = _to_B_N_3(pred)
    target = _to_B_N_3(target)
    
    # Compute pairwise distance matrices
    pred_dist = pairwise_distance_matrix(pred, squared=squared)
    target_dist = pairwise_distance_matrix(target, squared=squared)
    
    # Optional normalization to make loss scale-invariant
    if normalize:
        pred_max = pred_dist.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        target_max = target_dist.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        pred_dist = pred_dist / (pred_max + 1e-8)
        target_dist = target_dist / (target_max + 1e-8)
    
    # Compute difference
    diff = pred_dist - target_dist
    
    if p == 1:
        loss = torch.abs(diff).mean()
    else:
        loss = (diff ** 2).mean()
        if p == 2:
            loss = torch.sqrt(loss + 1e-8)
    
    return loss, None


def sampled_pairwise_distance_loss(pred: torch.Tensor,
                                    target: torch.Tensor,
                                    *,
                                    n_samples: int = 256,
                                    squared: bool = False,
                                    normalize: bool = True,
                                    p: int = 2) -> tuple[torch.Tensor, None]:
    """
    Compute pairwise distance loss using random point sampling for efficiency.
    
    For large point clouds, computing full N×N distance matrices is expensive.
    This version randomly samples a subset of points to reduce computation.
    
    Args:
        pred: Predicted point cloud (B, N, 3) or (B, 3, N)
        target: Target point cloud (B, N, 3) or (B, 3, N)
        n_samples: Number of points to sample (default: 256)
        squared: If True, compare squared distances (default: False)
        normalize: If True, normalize distances (default: True)
        p: Norm type for comparison (default: 2)
    
    Returns:
        loss: Scalar loss value averaged over the batch
        aux: None (for API compatibility)
    """
    pred = _to_B_N_3(pred)
    target = _to_B_N_3(target)
    
    B, N, _ = pred.shape
    
    # If we have fewer points than requested samples, use all points
    if N <= n_samples:
        return pairwise_distance_loss(pred, target, squared=squared, 
                                       normalize=normalize, p=p)
    
    # Random sampling (same indices for pred and target)
    indices = torch.randperm(N, device=pred.device)[:n_samples]
    pred_sampled = pred[:, indices, :]
    target_sampled = target[:, indices, :]
    
    return pairwise_distance_loss(pred_sampled, target_sampled, 
                                   squared=squared, normalize=normalize, p=p)


def local_pairwise_distance_loss(pred: torch.Tensor,
                                  target: torch.Tensor,
                                  *,
                                  k: int = 16,
                                  normalize: bool = True,
                                  p: int = 2) -> tuple[torch.Tensor, None]:
    """
    Compute pairwise distance loss using only k-nearest neighbors.
    
    This focuses on preserving local geometric structure by only comparing
    distances to the k nearest neighbors of each point.
    
    Args:
        pred: Predicted point cloud (B, N, 3) or (B, 3, N)
        target: Target point cloud (B, N, 3) or (B, 3, N)
        k: Number of nearest neighbors to consider (default: 16)
        normalize: If True, normalize distances (default: True)
        p: Norm type for comparison (default: 2)
    
    Returns:
        loss: Scalar loss value averaged over the batch
        aux: None (for API compatibility)
    """
    pred = _to_B_N_3(pred)
    target = _to_B_N_3(target)
    
    B, N, _ = pred.shape
    k = min(k, N - 1)  # Can't have more neighbors than points - 1
    
    # Compute full distance matrices
    pred_dist = pairwise_distance_matrix(pred, squared=False)
    target_dist = pairwise_distance_matrix(target, squared=False)
    
    # Get k smallest distances for each point (excluding self-distance)
    # Add large value to diagonal to exclude self-connections
    eye_mask = torch.eye(N, device=pred.device, dtype=pred.dtype).unsqueeze(0) * 1e9
    pred_dist_masked = pred_dist + eye_mask
    target_dist_masked = target_dist + eye_mask
    
    # Get k-nearest distances
    pred_knn, _ = torch.topk(pred_dist_masked, k, dim=-1, largest=False)
    target_knn, _ = torch.topk(target_dist_masked, k, dim=-1, largest=False)
    
    # Optional normalization
    if normalize:
        pred_max = pred_knn.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        target_max = target_knn.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        pred_knn = pred_knn / (pred_max + 1e-8)
        target_knn = target_knn / (target_max + 1e-8)
    
    # Compute difference
    diff = pred_knn - target_knn
    
    if p == 1:
        loss = torch.abs(diff).mean()
    else:
        loss = (diff ** 2).mean()
        if p == 2:
            loss = torch.sqrt(loss + 1e-8)
    
    return loss, None


class PairwiseDistanceLoss(nn.Module):
    """
    PyTorch Module wrapper for pairwise distance loss.
    
    Args:
        mode: Loss computation mode - 'full', 'sampled', or 'local' (default: 'sampled')
        n_samples: Number of samples for 'sampled' mode (default: 256)
        k: Number of neighbors for 'local' mode (default: 16)
        squared: Compare squared distances (default: False)
        normalize: Normalize distances (default: True)
        p: Norm type (default: 2)
    """
    
    def __init__(self,
                 mode: str = 'sampled',
                 n_samples: int = 256,
                 k: int = 16,
                 squared: bool = False,
                 normalize: bool = True,
                 p: int = 2):
        super().__init__()
        self.mode = mode.lower()
        self.n_samples = n_samples
        self.k = k
        self.squared = squared
        self.normalize = normalize
        self.p = p
        
        if self.mode not in ('full', 'sampled', 'local'):
            raise ValueError(f"Unknown mode: {mode}. Must be 'full', 'sampled', or 'local'")
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, None]:
        if self.mode == 'full':
            return pairwise_distance_loss(pred, target, 
                                          squared=self.squared,
                                          normalize=self.normalize,
                                          p=self.p)
        elif self.mode == 'sampled':
            return sampled_pairwise_distance_loss(pred, target,
                                                   n_samples=self.n_samples,
                                                   squared=self.squared,
                                                   normalize=self.normalize,
                                                   p=self.p)
        else:  # local
            return local_pairwise_distance_loss(pred, target,
                                                 k=self.k,
                                                 normalize=self.normalize,
                                                 p=self.p)


def compute_pdist_loss(pred: torch.Tensor,
                       target: torch.Tensor,
                       *,
                       mode: str = "sampled",
                       n_samples: int = 256,
                       k: int = 16,
                       normalize: bool = True,
                       p: int = 2,
                       squared: bool = False) -> torch.Tensor:
    """
    Compute pairwise distance loss with configurable mode and parameters.
    
    This is a convenience function that dispatches to the appropriate
    pairwise distance loss variant based on the mode parameter.
    
    Args:
        pred: Predicted point cloud (B, N, 3) or (B, 3, N)
        target: Target point cloud (B, N, 3) or (B, 3, N)
        mode: Loss computation mode - 'full', 'sampled', or 'local' (default: 'sampled')
        n_samples: Number of points to sample for 'sampled' mode (default: 256)
        k: Number of nearest neighbors for 'local' mode (default: 16)
        normalize: If True, normalize distances (default: True)
        p: Norm type for comparison (1=L1, 2=L2) (default: 2)
        squared: If True, compare squared distances (default: False)
    
    Returns:
        loss: Scalar loss value
    """
    mode = mode.lower()
    
    if mode == "full":
        val, _ = pairwise_distance_loss(
            pred, target,
            squared=squared,
            normalize=normalize,
            p=p
        )
    elif mode == "local":
        val, _ = local_pairwise_distance_loss(
            pred, target,
            k=k,
            normalize=normalize,
            p=p
        )
    else:  # sampled (default)
        val, _ = sampled_pairwise_distance_loss(
            pred, target,
            n_samples=n_samples,
            squared=squared,
            normalize=normalize,
            p=p
        )
    return val

