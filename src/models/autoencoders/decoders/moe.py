import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import Decoder
from ..registry import register_decoder


def mlp(sizes: List[int], act=nn.ReLU, bn: bool = True, last_bn: bool = False) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if bn:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(act(inplace=True))
        else:
            if last_bn and bn:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
    return nn.Sequential(*layers)


def spherical_fibonacci_points(n: int, device: torch.device) -> torch.Tensor:
    """
    Deterministic quasi-uniform directions on S^2. Returns (n, 3).
    """

    i = torch.arange(n, dtype=torch.float32, device=device) + 0.5
    phi = 2.0 * math.pi * i / ((1.0 + 5.0 ** 0.5) * 0.5)  # 2π i / φ
    cos_theta = 1.0 - 2.0 * i / n
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=0.0))
    x = torch.cos(phi) * sin_theta
    y = torch.sin(phi) * sin_theta
    z = cos_theta
    dirs = torch.stack([x, y, z], dim=-1)  # (n, 3)
    return dirs



def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    x: (B, N, 3)
    returns idx: (B, N, k) indices of k nearest neighbors (excluding self).
    """
    B, N, _ = x.shape
    with torch.no_grad():
        # Pairwise squared distances
        d2 = torch.cdist(x, x, p=2.0) ** 2  # (B, N, N)
        # Exclude self by adding a large number to diagonal
        eye = torch.eye(N, device=x.device).unsqueeze(0)
        d2 = d2 + eye * 1e6
        _, idx = torch.topk(d2, k=k, dim=-1, largest=False, sorted=False)
    return idx  # (B, N, k)


class GatedRouter(nn.Module):
    def __init__(self, input_dim, n_experts, hidden=256):
        super().__init__()
        self.net = mlp([input_dim, hidden, hidden, n_experts], bn=True, last_bn=False)

    def forward(self, z, hard=False):
        logits = self.net(z)
        # Gumbel-Softmax allows differentiable "hard" selection
        # distinct_prob forces the network to choose ONE expert, not a mix.
        if self.training:
            weights = F.gumbel_softmax(logits, tau=1.0, hard=hard)
        else:
            # Inference: Hard argmax
            idx = torch.argmax(logits, dim=-1)
            weights = F.one_hot(idx, num_classes=logits.size(-1)).float()
        return weights


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # The codebook
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        # z: (B, D)
        # Calculate distances
        distances = (torch.sum(z**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(z, self.embeddings.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight Through Estimator
        quantized = z + (quantized - z).detach()
        
        return quantized, loss, encoding_indices

class AmorphousExpert(nn.Module):
    """
    Decodes a radially-parameterized amorphous template:
      - Predicts per-point radii r_i >= 0 (B,N)
      - Projects onto deterministic spherical-fibonacci directions (N,3)
      - Adds small Gaussian noise only during training
    """
    def __init__(
        self,
        latent_size: int,
        num_points: int,
        hidden: int = 256,
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.n = num_points
        self.noise_std = noise_std

        self.head_r = mlp([latent_size, hidden, hidden, self.n], bn=True, last_bn=False)
        self.head_scale = mlp([latent_size, hidden, 1], bn=True, last_bn=False)

        # directions buffer is lazily created on first forward (so it picks the right device)

        self.register_buffer("_dirs_ready", torch.tensor([0], dtype=torch.uint8), persistent=False)
        self.register_buffer("_dirs", torch.empty(0), persistent=False)

    def _ensure_dirs(self, device):
        if self._dirs.numel() == 0 or self._dirs.device != device:
            dirs = spherical_fibonacci_points(self.n, device=device)  # (N,3)
            self._dirs = dirs
            self._dirs_ready = torch.tensor([1], dtype=torch.uint8, device=device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (B, D) -> pts: (B, N, 3)
        """
        B = z.size(0)
        self._ensure_dirs(z.device)

        r = F.softplus(self.head_r(z), beta=1.0) + 1e-4     # (B,N)
        s = F.softplus(self.head_scale(z), beta=1.0).clamp_min(0.1)  # (B,1)

        dirs = self._dirs.unsqueeze(0).expand(B, -1, -1)      # (B,N,3)
        pts = dirs * r.unsqueeze(-1) * s.unsqueeze(-1)        # (B,N,3)

        # Small noise during training only (keeps eval deterministic)
        if self.training and self.noise_std > 0:
            pts = pts + torch.randn_like(pts) * self.noise_std

        pts = pts - pts.mean(dim=1, keepdim=True)
        return pts


class TransformerRefiner(nn.Module):
    """
    Replaces KNN GNN with a Transformer. 
    Global attention is critical for crystals to align periodic boundaries.
    """
    def __init__(self, latent_dim, hidden_dim=128, n_heads=4, n_layers=2):
        super().__init__()
        self.embedding = nn.Linear(3, hidden_dim)
        
        # Conditional Layer Norm or AdaGN is better, but concatenation is simple
        self.z_proj = nn.Linear(latent_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, dim_feedforward=256, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.out_proj = nn.Linear(hidden_dim, 3)

    def forward(self, x, z):
        # x: (B, N, 3)
        # z: (B, D)
        B, N, _ = x.shape
        
        h = self.embedding(x) # (B, N, H)
        z_feat = self.z_proj(z).unsqueeze(1) # (B, 1, H)
        
        # Add Z context to every point token
        h = h + z_feat 
        
        # Self-attention captures global symmetry
        h = self.transformer(h)
        
        delta = self.out_proj(h)
        return x + delta, delta.norm(dim=-1).mean(dim=-1) # Return refined X and residual magnitude

class DifferentiableLatticeExpert(nn.Module):
    """
    Improved Lattice Expert. 
    Instead of cropping (non-differentiable selection), we fold a fixed 
    grid into the desired shape.
    """
    def __init__(self, latent_size, num_points, hidden=256):
        super().__init__()
        self.n = num_points
        
        # 1. Learn the primitive unit cell basis vectors (3 vectors)
        self.head_basis = mlp([latent_size, hidden, 9], bn=True)
        
        # 2. Learn a displacement field for a fixed grid to handle N points
        # We start with a cube grid of N points
        side = int(math.ceil(num_points**(1/3)))
        t = torch.linspace(0, 1, side)
        grid = torch.stack(torch.meshgrid(t, t, t, indexing='ij'), dim=-1).reshape(-1, 3)
        grid = grid[:num_points] # Take exactly N points
        self.register_buffer("base_grid", grid) # (N, 3)

    def forward(self, z):
        B = z.shape[0]
        
        # Predict Basis Matrix B (Transformation from unit cube to Crystal Cell)
        basis = self.head_basis(z).view(B, 3, 3) # (B, 3, 3)
        
        # Apply lattice transformation to the base grid
        # This ensures all points move COHERENTLY respecting the predicted symmetry
        base_grid = self.base_grid.unsqueeze(0).expand(B, -1, -1) # (B, N, 3)
        
        # X_crystal = Grid @ Basis
        pts = torch.bmm(base_grid, basis)
        
        pts = pts - pts.mean(dim=1, keepdim=True)
        return pts

@register_decoder("VQMoE")
class VQMoEDecoder(nn.Module):
    def __init__(self, 
                 num_points=80, 
                 latent_size=48, 
                 vq_size=128, # Size of codebook (number of motifs)
                 use_vq=True,
                 router_hidden=64,
                 transformer_hidden=64,
                 transformer_layers=2,
                 transformer_heads=4,
                 lattice_hidden=64,
                 amorphous_hidden=64,                 
                 ):
        super().__init__()
        
        self.use_vq = use_vq
        
        # 1. VQ-VAE Layer
        if self.use_vq:
            self.vq = VectorQuantizer(num_embeddings=vq_size, embedding_dim=latent_size)
        
        # 2. Experts
        self.expert_lattice = DifferentiableLatticeExpert(latent_size, num_points, hidden=lattice_hidden)
        self.expert_amorphous = AmorphousExpert(latent_size, num_points, hidden=amorphous_hidden) # Your existing one is fine
        
        # 3. Gated Router (Gumbel Softmax)
        self.router = mlp([latent_size, router_hidden, 2], bn=False)
        
        # 4. Transformer Refiner (Global context)
        self.refiner = TransformerRefiner(latent_size, hidden_dim=transformer_hidden, n_layers=transformer_layers)

    def forward(self, z):
        # z: (B, D)
        
        # 1. VQ Bottleneck
        vq_loss = torch.tensor(0.0, device=z.device)
        if self.use_vq:
            # z_q is the "clean" motif embedding
            z_q, vq_loss, indices = self.vq(z)
            
            # Structure branch sees the Clean Motif (z_q)
            # Refiner sees the specific noise (z) or z_q depending on strategy.
            # Usually passing 'z' to refiner allows it to model the noise.
            z_structure = z_q
        else:
            z_structure = z

        # 2. Generate Templates
        out_lattice = self.expert_lattice(z_structure)
        out_amorphous = self.expert_amorphous(z_structure)
        
        # 3. Hard Gating (Gumbel Softmax)
        # During training, this samples. During eval, it argmaxes.
        # This prevents "averaging" coordinates.
        logits = self.router(z_structure)
        weights = F.gumbel_softmax(logits, tau=1.0, hard=True) # (B, 2)
        
        # Select output without blending positions
        # weights[:, 0] is (B,), expand to (B, N, 3)
        w_lat = weights[:, 0].view(-1, 1, 1)
        w_amo = weights[:, 1].view(-1, 1, 1)
        
        mixed = w_lat * out_lattice + w_amo * out_amorphous
        
        # 4. Refine
        # The refiner adds the "Continuity" and small perturbations.
        # It takes the generated template and the original (possibly noisy) Z.
        final_pts, residual_mag = self.refiner(mixed, z)
        
        return final_pts, vq_loss, weights


# ----- Minimal sanity check -----
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, D = 4, 80, 96
    z = torch.randn(B, D)

    dec = MoETemplateResidualDecoder(
        num_points=N,
        latent_size=D,
        n_experts=2,              # {Lattice, Amorphous}
        topk=1,                   # force decisive gating
        gate_temp=0.7,
        lattice_kwargs=dict(basis_size=6, replicate_radius=2),
        amorphous_kwargs=dict(noise_std=0.01),
        residual_kwargs=dict(k=12, n_layers=2, hidden_ctx=64, edge_hidden=64, step_size=0.7),
    )

    Xhat = dec(z)  # (B,N,3)
    print("Output shape:", Xhat.shape)
    print("Last aux keys:", dec.last_aux.keys())
