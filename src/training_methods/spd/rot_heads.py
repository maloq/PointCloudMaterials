import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Any, Dict

import os, sys
sys.path.append(os.getcwd())

from src.models.autoencoders.encoders.vn import VNLinearLeakyReLU, VNLinear, VNMaxPool



def gram_schmidt_rotation(vectors: torch.Tensor) -> torch.Tensor:
    """
    Converts 2 vectors (B, 2, 3) into a Rotation Matrix (B, 3, 3)
    using Gram-Schmidt orthogonalization.
    v1 becomes the X-axis (or first col), v2 helps define the Y-axis.
    """
    v1 = vectors[:, 0, :]  # (B, 3)
    v2 = vectors[:, 1, :]  # (B, 3)

    # Normalize v1 -> x_axis
    x_axis = F.normalize(v1, dim=-1, eps=1e-6)

    # Project v2 onto plane perpendicular to v1
    # v2_new = v2 - (v2 . x_axis) * x_axis
    dot = (v2 * x_axis).sum(dim=-1, keepdim=True)
    y_axis = v2 - dot * x_axis
    y_axis = F.normalize(y_axis, dim=-1, eps=1e-6)

    # z_axis = x cross y
    z_axis = torch.cross(x_axis, y_axis, dim=-1)

    # Stack columns
    R = torch.stack([x_axis, y_axis, z_axis], dim=-1) # (B, 3, 3)
    return R


class VNRotationHead(nn.Module):
    """
    Strictly Equivariant Rotation Head.
    Uses Vector Neurons to reduce features to 2 vectors, then builds R.
    """
    def __init__(self, in_features: int, hidden: int = 128, **kwargs):
        super().__init__()
        
        in_channels = in_features // 3
        hidden_channels = hidden

        # Reduce equivariant features down to 2 vectors per sample
        self.net = nn.Sequential(
            VNLinearLeakyReLU(in_channels, hidden_channels, dim=3),
            VNLinearLeakyReLU(hidden_channels, hidden_channels, dim=3),
            # Final projection to exactly 2 vectors (the "basis" candidates)
            VNLinear(hidden_channels, 2) 
        )

    def forward(self, eq_z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eq_z: (B, C, 3) - The pooled equivariant embedding from Encoder
        Returns:
            R: (B, 3, 3) - Rotation matrix
        """
        # eq_z is (B, C, 3). VN layers expect (B, C, 3) or (B, 1, C, 3) depending on impl.
        # Based on your vn.py, VNLinear expects (B, C, 3).
        
        # 1. Predict 2 vectors using VN layers
        # Output shape: (B, 2, 3)
        vectors = self.net(eq_z)
        
        # 2. Gram-Schmidt to get valid Rotation Matrix
        R = gram_schmidt_rotation(vectors)
        
        return R


def sixd_to_so3(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Map a 6D vector to a proper rotation matrix using a continuous
    Gram–Schmidt process (Zhou et al., CVPR'19).
    Args:
        x: (..., 6) tensor
    Returns:
        R: (..., 3, 3) rotation (det=+1, orthonormal columns)
    """
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]

    b1 = F.normalize(a1, dim=-1, eps=eps)
    # remove component of a2 along b1
    a2_proj = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2_proj, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)

    R = torch.stack([b1, b2, b3], dim=-1)  # columns
    return R


def _orthogonalize(mat: torch.Tensor) -> torch.Tensor:
    """Return the closest orthogonal matrix using SVD."""
    orig_dtype = mat.dtype
    u, _, v = torch.linalg.svd(mat.to(torch.float32))
    return (u @ v.transpose(-1, -2)).to(orig_dtype)


def _autocast_disabled_context(tensor: torch.Tensor):
    """Return a context manager that disables autocast when it is active."""
    device_type = tensor.device.type
    if device_type == "cuda" and torch.is_autocast_enabled():
        return torch.autocast(device_type=device_type, enabled=False)
    if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled") and torch.is_autocast_cpu_enabled():
        return torch.autocast(device_type=device_type, enabled=False)
    return nullcontext()


def _det3x3(mat: torch.Tensor) -> torch.Tensor:
    """Fast batched determinant for 3x3 matrices using triple product.
    Expects `mat` of shape (..., 3, 3) in float32.
    """
    a = mat[..., 0, :]
    b = mat[..., 1, :]
    c = mat[..., 2, :]
    return torch.sum(torch.cross(a, b, dim=-1) * c, dim=-1)


def kabsch_rotation(cano: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute optimal rotation aligning canonical output to target using Kabsch algorithm.
    Args:
        cano: predicted canonical points (B, N, 3) or (B, 3, N)
        target: reference points (matching shape permutations)
    """

    with _autocast_disabled_context(cano):
        orig_dtype = cano.dtype
        target_points = target.to(dtype=torch.float32, device=cano.device)
        cano_points = cano.to(torch.float32)

        cano_centered = cano_points - cano_points.mean(dim=1, keepdim=True)
        target_centered = target_points - target_points.mean(dim=1, keepdim=True)

        cov = cano_centered.transpose(1, 2) @ target_centered
        U, _, Vh = torch.linalg.svd(cov, full_matrices=False)
        # Diagonal fix: R = Vh^T * diag(1,1,s) * U^T, where s = sign(det(VU^T)) == sign(det(cov)).
        s = torch.sign(_det3x3(cov))
        Sfix = torch.eye(3, device=cov.device, dtype=cov.dtype).expand(cov.shape[0], -1, -1).clone()
        Sfix[:, -1, -1] = torch.where(s < 0, torch.tensor(-1.0, dtype=cov.dtype, device=cov.device), torch.tensor(1.0, dtype=cov.dtype, device=cov.device))
        R = Vh.transpose(-1, -2) @ Sfix @ U.transpose(-1, -2)

        return R.to(dtype=orig_dtype)


class _BaseRotHead(nn.Module):
    """Shared pooling logic for rotation heads."""

    def __init__(self, in_features: int, hidden: int = 256, use_attention: bool = True):
        super().__init__()
        if in_features <= 0:
            raise ValueError(f"in_features must be positive, got {in_features}")
        self.in_features = in_features
        self.use_attention = use_attention
        self.hidden = hidden

    def _pool_equivariant(self, eq_full: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eq_full: (B, C, 3, N)
        Returns:
            pooled: (B, C, 3)
        """
        if self.use_attention:
            scores = eq_full.norm(dim=2).mean(dim=1)          # (B, N)
            weights = torch.softmax(scores, dim=-1)           # (B, N)
            pooled = (eq_full * weights[:, None, None, :]).sum(dim=-1)
        else:
            pooled = eq_full.mean(dim=-1)
        return pooled


class Rot6DHead(_BaseRotHead):
    """
    Rotation head that consumes per-point equivariant features (B, C, 3, N)
    and outputs a proper rotation matrix via a 6D continuous parameterization.

    - Uses a simple learned pooling over points (optional attention)
    - MLP -> 6D -> sixd_to_so3
    """
    def __init__(self, in_features: int, hidden: int = 256, use_attention: bool = True):
        super().__init__(in_features, hidden, use_attention)
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 6),
        )

    def forward(self, eq_full: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eq_full: (B, C, 3) equivariant features
        Returns:
            R: (B, 3, 3)
        """
        B, C, _ = eq_full.shape
        h = eq_full.reshape(B, C * 3)
        sixd = self.mlp(h)
        R = sixd_to_so3(sixd)
        return R


class RotMatrixHead(_BaseRotHead):
    """
    Rotation head that predicts rotation matrices directly (optionally orthogonalized).
    """

    def __init__(
        self,
        in_features: int,
        hidden: int = 256,
        use_attention: bool = True,
        orthogonalize: bool = False,
    ):
        super().__init__(in_features, hidden, use_attention)
        self.orthogonalize = orthogonalize
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 9),
        )

    def forward(self, eq_full: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eq_full: (B, C, 3)
        Returns:
            R: (B, 3, 3)
        """
        B, C, _ = eq_full.shape
        h = eq_full.reshape(B, C * 3)
        mat = self.mlp(h).view(B, 3, 3)
        if self.orthogonalize:
            mat = _orthogonalize(mat)
        return mat


def build_rot_head(cfg: Any, in_features: int) -> nn.Module:
    """
    Factory helper to initialize rotation heads from configuration.

    Args:
        cfg: configuration object (OmegaConf DictConfig or similar) with
            - rot_net.name
            - rot_net.kwargs (optional)
        in_features: flattened equivariant feature dimension (C * 3)
    """
    rot_net = getattr(cfg, "rot_net", None)
    if rot_net is None:
        raise ValueError("cfg.rot_net is required to build the rotation head")

    # Handle both dict-like and object-like configs
    def _get(obj, key, default):
        return obj.get(key, default) if hasattr(obj, "get") else getattr(obj, key, default)

    name = _get(rot_net, "name", "Rot6DHead")
    kwargs = _get(rot_net, "kwargs", {}) or {}

    registry = {
        "rot6dhead": Rot6DHead,
        "sixd_head": Rot6DHead,
        "rotmatrixhead": RotMatrixHead,
        "matrix_head": RotMatrixHead,
        "vn_rotation_head": VNRotationHead,
    }

    head_cls = registry.get(name.lower())
    if head_cls is None:
        raise ValueError(f"Unknown rotation head '{name}'. Supported: {list(registry.keys())}")

    return head_cls(in_features=in_features, **kwargs)
