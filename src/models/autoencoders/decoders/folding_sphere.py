from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F

from ..base     import Decoder
from ..registry import register_decoder


class _TransposedBatchNormRelu(nn.Module):
    """BN + ReLU for (B,N,C) by transposing to (B,C,N) for BatchNorm1d."""

    def __init__(self, num_features: int, dropout_rate: float = 0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B,N,C)
        x = x.transpose(1, 2)                                # (B,C,N)
        x = F.relu(self.bn(x))
        x = x.transpose(1, 2)                                # (B,N,C)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


def _fibonacci_sphere_points(n: int, radius: float) -> torch.Tensor:
    """Approx. uniform *n* points on sphere of given *radius* (Fibonacci spiral)."""
    if n <= 0:
        return torch.empty((0, 3))

    device = "cpu"
    idx = torch.arange(n, dtype=torch.float32, device=device)
    phi = (1 + 5 ** 0.5) / 2
    theta = 2 * math.pi * idx / phi
    z = 1.0 - 2.0 * (idx + 0.5) / n
    r = torch.sqrt(1.0 - z * z)
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return radius * torch.stack((x, y, z), dim=-1)          # (n,3)


# ---------------------------------------------------------------------------
#  Two‑shell Folding‑sphere decoder
# ---------------------------------------------------------------------------
@register_decoder("FoldingSphere")
class FoldingDecoderSphereTwoShell(Decoder):
    def __init__(
        self,
        num_points : int,
        latent_size: int,
        *,
        n_shells   : int   = 2,
        R1         : float = 0.5,
        R2         : float = 1.0,
        hidden_dim : int   = 512,
        dropout_rate: float = 0.0,
        learnable_template: bool = False,
    ) -> None:
        super().__init__()
        if n_shells not in (1, 2):
            raise ValueError("`n_shells` must be 1 or 2.")
        if num_points < 2:
            raise ValueError("Need ≥2 points (centre + shell).")

        # ----------------  template ----------------
        shell_pts = num_points - 1
        if n_shells == 1:
            templ = torch.cat([torch.zeros(1, 3),
                               _fibonacci_sphere_points(shell_pts, R1)], dim=0)
        else:
            ratio = R1**2 / (R1**2 + R2**2)
            n1 = max(1, round(shell_pts * ratio))
            n2 = shell_pts - n1 or 1
            templ = torch.cat([
                torch.zeros(1, 3),
                _fibonacci_sphere_points(n1, R1),
                _fibonacci_sphere_points(n2, R2),
            ], dim=0)

            if learnable_template:
                self.template = nn.Parameter(templ)   # learnable
            else:
                self.register_buffer("template", templ)
            self._n = num_points

        def make_stage(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                _TransposedBatchNormRelu(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                _TransposedBatchNormRelu(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, 3),
                _TransposedBatchNormRelu(3),
                nn.Linear(3, 3),
            )

        self.stage1 = make_stage(latent_size + 3)
        self.stage2 = make_stage(latent_size + 3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        z_exp = z.unsqueeze(1).expand(B, self._n, -1)
        templ = self.template.expand(B, -1, -1)
        coarse  = self.stage1(torch.cat((z_exp, templ), dim=-1))
        refined = self.stage2(torch.cat((z_exp, coarse), dim=-1))
        return refined


# ---------------------------------------------------------------------------
#  Folding‑sphere decoder + local attention
# ---------------------------------------------------------------------------
@register_decoder("FoldingSphereAttn")
class FoldingDecoderSphereTwoShellAttn(FoldingDecoderSphereTwoShell):
    """Adds 2 local‑attention Folding stages on top of the sphere decoder."""

    def __init__(
        self,
        num_points   : int,
        latent_size  : int,
        *,
        n_shells     : int   = 2,
        R1           : float = 1.0,
        R2           : float = 1.5,
        R_att        : float = 0.20,
        hidden_dim   : int   = 512,
        attn_dim     : int   = 128,
        dropout_rate : float = 0.0,
        learnable_template: bool = False,
    ):
        super().__init__(
            num_points=num_points,
            latent_size=latent_size,
            n_shells=n_shells,
            R1=R1,
            R2=R2,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.R_att    = R_att
        self.sqrt_d   = math.sqrt(attn_dim)
        self.q_proj   = nn.Linear(3, attn_dim, bias=False)
        self.k_proj   = nn.Linear(3, attn_dim, bias=False)
        self.v_proj   = nn.Linear(3, attn_dim, bias=False)

        def make_stage(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                _TransposedBatchNormRelu(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                _TransposedBatchNormRelu(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, 3),
            )

        self.stage0 = make_stage(latent_size + 3)                 # (latent + templ)
        self.stage1 = make_stage(latent_size + 3 + attn_dim)
        self.stage2 = make_stage(latent_size + 3 + attn_dim)

    # -----------------------  forward -----------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        z_exp = z.unsqueeze(1).expand(B, self._n, -1)
        templ = self.template.expand(B, -1, -1)

        coords0 = self.stage0(torch.cat((z_exp, templ), dim=-1))
        attn1   = self._local_attention(coords0)
        coords1 = self.stage1(torch.cat((z_exp, coords0, attn1), dim=-1))
        attn2   = self._local_attention(coords1)
        coords2 = self.stage2(torch.cat((z_exp, coords1, attn2), dim=-1))
        return coords2

    # ----------------  local attention ----------------
    def _local_attention(self, coords: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(coords)
        K = self.k_proj(coords)
        V = self.v_proj(coords)

        d2 = torch.cdist(coords, coords) ** 2
        mask = d2 <= (self.R_att ** 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d
        scores = scores.masked_fill(~mask, float('-inf'))
        W = torch.softmax(scores, dim=-1).masked_fill(~mask, 0.0)
        return torch.matmul(W, V)



@register_decoder("FoldingSphereAttnRes")
class FoldingDecoderSphereTwoShellAttnRes(FoldingDecoderSphereTwoShell):
    """Adds 2 local‑attention Folding stages on top of the sphere decoder with residual connections."""

    def __init__(
        self,
        num_points   : int,
        latent_size  : int,
        *,
        n_shells     : int   = 2,
        R1           : float = 1.0,
        R2           : float = 1.5,
        R_att        : float = 0.20,
        hidden_dim   : int   = 512,
        attn_dim     : int   = 128,
        dropout_rate : float = 0.0,
        learnable_template: bool = False,
    ):
        super().__init__(
            num_points=num_points,
            latent_size=latent_size,
            n_shells=n_shells,
            R1=R1,
            R2=R2,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        self.R_att    = R_att
        self.sqrt_d   = math.sqrt(attn_dim)
        self.q_proj   = nn.Linear(3, attn_dim, bias=False)
        self.k_proj   = nn.Linear(3, attn_dim, bias=False)
        self.v_proj   = nn.Linear(3, attn_dim, bias=False)

        def make_stage(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                _TransposedBatchNormRelu(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                _TransposedBatchNormRelu(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, 3),
            )

        self.stage0 = make_stage(latent_size + 3)                 # (latent + templ)
        self.stage1 = make_stage(latent_size + 3 + attn_dim)
        self.stage2 = make_stage(latent_size + 3 + attn_dim)

    # ----------------  local attention ----------------
    def _local_attention(self, coords: torch.Tensor) -> torch.Tensor:
        Q = self.q_proj(coords)
        K = self.k_proj(coords)
        V = self.v_proj(coords)

        d2 = torch.cdist(coords, coords) ** 2
        mask = d2 <= (self.R_att ** 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d
        scores = scores.masked_fill(~mask, float('-inf'))
        W = torch.softmax(scores, dim=-1).masked_fill(~mask, 0.0)
        return torch.matmul(W, V)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)

        # ---- Infer the latent width this decoder was built with from stage0 ----
        # stage0 takes (latent + 3) inputs
        first_linear = None
        for m in self.stage0.modules():
            if isinstance(m, nn.Linear):
                first_linear = m
                break
        assert first_linear is not None, "stage0 must start with a Linear layer"
        expected_latent = first_linear.in_features - 3

        # Pad/trim z so it matches what the decoder expects (no new params)
        zdim = z.size(-1)
        if zdim != expected_latent:
            if zdim < expected_latent:
                z = F.pad(z, (0, expected_latent - zdim))
            else:
                z = z[..., :expected_latent]

        # ---- (B,N,3) template and per-point latent ----
        templ = self.template
        if templ.dim() == 2:              # (N,3) -> (1,N,3)
            templ = templ.unsqueeze(0)
        templ = templ.expand(B, -1, -1)   # (B,N,3)
        z_exp = z.unsqueeze(1).expand(B, templ.size(1), -1)  # (B,N,latent)

        # ---------------- stage 0: offsets from template ----------------
        x0 = torch.cat((z_exp, templ), dim=-1)   # (B,N,latent+3)
        delta0  = self.stage0(x0)                # (B,N,3)
        coarse0 = templ + delta0                 # residual add

        # ---------------- stage 1: include local attention ----------------
        attn1   = self._local_attention(coarse0) # (B,N,attn_dim)
        x1 = torch.cat((z_exp, coarse0, attn1), dim=-1)  # (B,N,latent+3+attn_dim)
        delta1  = self.stage1(x1)                # (B,N,3)
        coarse1 = coarse0 + delta1

        # ---------------- stage 2: include local attention ----------------
        attn2   = self._local_attention(coarse1)
        x2 = torch.cat((z_exp, coarse1, attn2), dim=-1)  # (B,N,latent+3+attn_dim)
        delta2  = self.stage2(x2)                # (B,N,3)
        refined = coarse1 + delta2

        return refined 