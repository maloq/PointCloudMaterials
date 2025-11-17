import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from typing import Any, Dict


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
    if cfg is None:
        raise ValueError("Rotation head configuration `cfg` must be provided")

    rot_net_cfg = getattr(cfg, "rot_net", None)
    if rot_net_cfg is None:
        raise ValueError("cfg.rot_net is required to build the rotation head")

    name = rot_net_cfg.get("name", "Rot6DHead") if hasattr(rot_net_cfg, "get") else getattr(rot_net_cfg, "name", "Rot6DHead")
    kwargs: Dict[str, Any] = rot_net_cfg.get("kwargs", {}) if hasattr(rot_net_cfg, "get") else getattr(rot_net_cfg, "kwargs", {})

    name = name.lower()
    if name == "rot6dhead" or name == "sixd_head":
        return Rot6DHead(in_features=in_features, **kwargs)
    if name == "rotmatrixhead" or name == "matrix_head":
        return RotMatrixHead(in_features=in_features, **kwargs)

    raise ValueError(f"Unknown rotation head '{rot_net_cfg.get('name', name)}'")
