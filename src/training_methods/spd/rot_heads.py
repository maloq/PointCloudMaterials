import torch
import torch.nn as nn
import torch.nn.functional as F


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
            eq_full: (B, C, 3, N) per-point equivariant features
        Returns:
            R: (B, 3, 3)
        """
        B, C, _, _ = eq_full.shape
        pooled = self._pool_equivariant(eq_full)
        h = pooled.reshape(B, C * 3)
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
            eq_full: (B, C, 3, N)
        Returns:
            R: (B, 3, 3)
        """
        B, C, _, _ = eq_full.shape
        pooled = self._pool_equivariant(eq_full)
        h = pooled.reshape(B, C * 3)
        mat = self.mlp(h).view(B, 3, 3)
        if self.orthogonalize:
            mat = _orthogonalize(mat)
        return mat
