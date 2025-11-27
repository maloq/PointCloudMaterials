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
            layers.append(act(inplace=False))  # Avoid inplace for DDP compatibility
        else:
            if last_bn and bn:
                layers.append(nn.BatchNorm1d(sizes[i + 1]))
    return nn.Sequential(*layers)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        distances = (torch.sum(z**2, dim=1, keepdim=True) 
                    + torch.sum(self.embeddings.weight**2, dim=1)
                    - 2 * torch.matmul(z, self.embeddings.weight.t()))
            
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self.embeddings.weight)
        
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = z + (quantized - z).detach()
        
        return quantized, loss, encoding_indices


class SimpleLatticeExpert(nn.Module):
    """
    A robust, simplified expert that ONLY learns the Lattice Matrix.
    It assumes atoms are arranged in a simple grid (0,0,0), (0,0,1)...
    and deforms the whole grid via matrix multiplication.
    """
    def __init__(self, latent_size, num_points, hidden=256):
        super().__init__()
        self.num_points = num_points
        
        # 1. Pre-calculate a fixed integer grid (0,0,0) to (N,N,N)
        # We generate more points than needed and slice later
        grid_dim = math.ceil(num_points ** (1/3))
        x = torch.arange(grid_dim, dtype=torch.float32)
        y = torch.arange(grid_dim, dtype=torch.float32)
        z = torch.arange(grid_dim, dtype=torch.float32)
        # Create grid
        grid = torch.stack(torch.meshgrid(x, y, z, indexing='ij'), dim=-1).reshape(-1, 3)
        # Center the grid around 0
        grid = grid - grid.mean(dim=0)
        
        # Register as buffer (fixed, not learnable)
        self.register_buffer('base_grid', grid[:num_points, :])

        # 2. Predict ONLY the 6 lattice parameters (3 lengths, 3 angles)
        self.net = mlp([latent_size, hidden, hidden, 6], bn=True, last_bn=False)

    def _params_to_matrix(self, params):
        # Lengths: ensure positive and non-zero. Initialize around 1.0
        lengths = F.softplus(params[:, :3]) + 0.5 
        
        # Angles: constrain to realistic crystal angles (60 to 120 degrees)
        # 0 -> 60 deg, 1 -> 120 deg
        angles = torch.sigmoid(params[:, 3:]) * (math.pi/3) + (math.pi/3)
        
        a, b, c = lengths[:, 0], lengths[:, 1], lengths[:, 2]
        alpha, beta, gamma = angles[:, 0], angles[:, 1], angles[:, 2]
        
        zeros = torch.zeros_like(a)
        
        # Matrix rows
        v1 = torch.stack([a, zeros, zeros], dim=1)
        v2 = torch.stack([b * torch.cos(gamma), b * torch.sin(gamma), zeros], dim=1)
        
        cx = c * torch.cos(beta)
        cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / (torch.sin(gamma) + 1e-6)
        cz = torch.sqrt(torch.clamp(c**2 - cx**2 - cy**2, min=1e-6))
        v3 = torch.stack([cx, cy, cz], dim=1)
        
        # Stack to (B, 3, 3)
        return torch.stack([v1, v2, v3], dim=1)

    def forward(self, z):
        B = z.shape[0]
        
        # Predict Lattice params
        params = self.net(z)
        matrix = self._params_to_matrix(params) # (B, 3, 3)
        
        # Apply deformation: Points = Grid * Matrix
        # Grid: (N, 3), Matrix: (B, 3, 3) -> Result: (B, N, 3)
        base = self.base_grid.unsqueeze(0).expand(B, -1, -1)
        pts = torch.bmm(base, matrix)
        
        return pts

class MultiCrystalExpert(nn.Module):
    """
    Multiple learnable crystal prototypes.
    Each prototype has its own unit cell shape and atom positions.
    The latent selects and deforms a prototype.
    """
    def __init__(
        self,
        latent_size: int,
        num_points: int,
        num_prototypes: int = 4,
        atoms_per_cell: int = 4,
        hidden: int = 256,
    ):
        super().__init__()
        self.n = num_points
        self.num_prototypes = num_prototypes
        self.atoms_per_cell = atoms_per_cell
        
        # Learnable prototype parameters (ideal crystal templates)
        # Cell params: 3 lengths + 3 angles
                # Initialize cell params to be non-collapsed (lengths ~1.0, angles ~90 deg)
        # params: [l1, l2, l3, a1, a2, a3]
        # Softplus(0.5) + 0.5 ≈ 1.2 length
        self.prototype_cell_params = nn.Parameter(torch.ones(num_prototypes, 6) * 0.5)
        # Add some noise so they aren't identical
        self.prototype_cell_params.data += torch.randn_like(self.prototype_cell_params) * 0.2
        # Fractional coordinates within unit cell
        self.prototype_frac_coords = nn.Parameter(torch.rand(num_prototypes, atoms_per_cell, 3))
        
        # Tiling setup
        self.n_cells = math.ceil(num_points / atoms_per_cell)
        self.cells_per_dim = math.ceil(self.n_cells ** (1/3))
        offsets = []
        for i in range(self.cells_per_dim):
            for j in range(self.cells_per_dim):
                for k in range(self.cells_per_dim):
                    offsets.append([i, j, k])
        offsets = torch.tensor(offsets[:self.n_cells], dtype=torch.float32)
        self.register_buffer("cell_offsets", offsets)
        
        # Prediction heads
        self.head_prototype_logits = mlp([latent_size, hidden, num_prototypes], bn=True)
        self.head_cell_deform = mlp([latent_size, hidden, 6], bn=True)
        self.head_frac_deform = mlp([latent_size, hidden, atoms_per_cell * 3], bn=True)
        self.head_scale = mlp([latent_size, hidden, 1], bn=True)

    def _cell_params_to_matrix(self, params: torch.Tensor) -> torch.Tensor:
        """Convert 6 cell parameters (lengths + angles) to 3x3 lattice matrix"""
        lengths = F.softplus(params[..., :3]) + 0.5
        angles = torch.sigmoid(params[..., 3:]) * (2*math.pi/3) + (math.pi/3)
        
        a, b, c = lengths[..., 0], lengths[..., 1], lengths[..., 2]
        alpha, beta, gamma = angles[..., 0], angles[..., 1], angles[..., 2]
        
        # Crystallographic convention
        v1 = torch.stack([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
        v2 = torch.stack([b * torch.cos(gamma), b * torch.sin(gamma), torch.zeros_like(b)], dim=-1)
        
        cx = c * torch.cos(beta)
        cy = c * (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / (torch.sin(gamma) + 1e-6)
        cz = torch.sqrt(torch.clamp(c**2 - cx**2 - cy**2, min=1e-6))
        v3 = torch.stack([cx, cy, cz], dim=-1)
        
        return torch.stack([v1, v2, v3], dim=-2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        
        # 1. Prototype selection
        logits = self.head_prototype_logits(z)
        if self.training:
            weights = F.gumbel_softmax(logits, tau=0.5, hard=True)
        else:
            weights = F.one_hot(logits.argmax(-1), self.num_prototypes).float()
        
        # 2. Weighted prototype parameters
        base_cell = torch.einsum('bk,kp->bp', weights, self.prototype_cell_params)
        base_frac = torch.einsum('bk,kac->bac', weights, self.prototype_frac_coords)
        
        # 3. Add deformations from latent
        cell_deform = self.head_cell_deform(z) * 0.1
        frac_deform = self.head_frac_deform(z).view(B, self.atoms_per_cell, 3) * 0.1
        
        cell_params = base_cell + cell_deform
        frac_coords = torch.sigmoid(base_frac + frac_deform)
        
        # 4. Build lattice matrix
        lattice_matrix = self._cell_params_to_matrix(cell_params)
        
        # 5. Tile unit cell
        all_frac = []
        for cell_idx in range(self.n_cells):
            offset = self.cell_offsets[cell_idx]
            tiled = frac_coords + offset.view(1, 1, 3)
            all_frac.append(tiled)
        all_frac = torch.cat(all_frac, dim=1)[:, :self.n, :]
        
        # 6. Fractional to Cartesian
        pts = torch.bmm(all_frac, lattice_matrix)
        
        # 7. Scale and center
        scale = F.softplus(self.head_scale(z)).clamp(min=0.1)
        pts = pts * scale.unsqueeze(-1)
        pts = pts - pts.mean(dim=1, keepdim=True)
        
        return pts


class ImprovedAmorphousExpert(nn.Module):
    """
    Full 3D coordinate prediction with learnable point queries.
    More expressive than radial-only parameterization.
    """
    def __init__(
        self,
        latent_size: int,
        num_points: int,
        hidden: int = 256,
    ):
        super().__init__()
        self.n = num_points
        
        # Learnable point identities
        self.point_embeddings = nn.Parameter(torch.randn(num_points, hidden // 2) * 0.02)
        
        # Project latent
        self.z_proj = nn.Linear(latent_size, hidden // 2)
        
        # Coordinate MLP
        self.coord_mlp = mlp([hidden, hidden, hidden, 3], bn=True)
        
        # Scale
        self.head_scale = mlp([latent_size, hidden // 2, 1], bn=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.shape[0]
        
        z_feat = self.z_proj(z).unsqueeze(1).expand(-1, self.n, -1)
        pt_emb = self.point_embeddings.unsqueeze(0).expand(B, -1, -1)
        combined = torch.cat([z_feat, pt_emb], dim=-1)
        
        combined_flat = combined.view(B * self.n, -1)
        coords_flat = self.coord_mlp(combined_flat)
        pts = coords_flat.view(B, self.n, 3)
        
        scale = F.softplus(self.head_scale(z)).clamp(min=0.1)
        pts = pts * scale.unsqueeze(-1)
        pts = pts - pts.mean(dim=1, keepdim=True)
        
        return pts


# =============================================================================
# FiLM-CONDITIONED TRANSFORMER REFINER
# =============================================================================

class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation for strong conditioning"""
    def __init__(self, feature_dim: int, condition_dim: int):
        super().__init__()
        self.scale = nn.Linear(condition_dim, feature_dim)
        self.shift = nn.Linear(condition_dim, feature_dim)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale = self.scale(condition).unsqueeze(1)
        shift = self.shift(condition).unsqueeze(1)
        return x * (1 + scale) + shift


class FiLMTransformerRefiner(nn.Module):
    """
    Transformer with FiLM conditioning at each layer.
    Bounded displacement to force good templates.
    """
    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        max_displacement: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_displacement = max_displacement
        
        self.input_proj = nn.Linear(3, hidden_dim)
        self.pos_encoder = nn.Linear(3, hidden_dim)
        
        self.film_layers = nn.ModuleList([
            FiLMLayer(hidden_dim, latent_dim) for _ in range(n_layers)
        ])
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
            for _ in range(n_layers)
        ])
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(n_layers)
        ])
        self.norms1 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(n_layers)])
        
        self.output_proj = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        
        h = self.input_proj(x) + self.pos_encoder(x)
        
        for i in range(len(self.attention_layers)):
            h = self.film_layers[i](h, z)
            
            h_norm = self.norms1[i](h)
            attn_out, _ = self.attention_layers[i](h_norm, h_norm, h_norm)
            h = h + attn_out
            
            h = h + self.ffn_layers[i](self.norms2[i](h))
        
        delta = self.output_proj(h)
        delta = torch.tanh(delta) * self.max_displacement
        
        refined = x + delta
        refined = refined - refined.mean(dim=1, keepdim=True)
        
        residual_mag = delta.norm(dim=-1).mean()
        
        return refined, residual_mag


# =============================================================================
# MAIN DECODER
# =============================================================================

@register_decoder("VQMoE")
class VQMoEDecoder(Decoder):
    """
    Improved VQ-MoE Decoder with:
    - Multi-crystal expert with learnable prototypes
    - Improved amorphous expert with full 3D freedom  
    - FiLM-conditioned transformer refiner
    - Bounded refinement to force good templates
    
    Original interface preserved: forward returns (points, vq_loss, weights)
    """
    def __init__(
        self,
        num_points: int = 80,
        latent_size: int = 48,
        vq_size: int = 128,
        use_vq: bool = True,
        router_hidden: int = 64,
        transformer_hidden: int = 64,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        lattice_hidden: int = 64,
        amorphous_hidden: int = 64,
        # New parameters with defaults
        num_crystal_prototypes: int = 4,
        atoms_per_cell: int = 4,
        max_refinement_displacement: float = 0.3,
        use_refiner: bool = True,
    ):
        super().__init__()
        
        self.use_vq = use_vq
        self.num_points = num_points
        self.use_refiner = use_refiner
        
        # 1. VQ-VAE Layer
        if self.use_vq:
            self.vq = VectorQuantizer(num_embeddings=vq_size, embedding_dim=latent_size)
        
        # 2. Experts
        # self.expert_lattice = MultiCrystalExpert(
        #     latent_size=latent_size,
        #     num_points=num_points,
        #     num_prototypes=num_crystal_prototypes,
        #     atoms_per_cell=atoms_per_cell,
        #     hidden=lattice_hidden,
        # )

        self.expert_lattice = SimpleLatticeExpert(
            latent_size=latent_size,
            num_points=num_points,
            hidden=lattice_hidden,
        )

        self.expert_amorphous = ImprovedAmorphousExpert(
            latent_size=latent_size,
            num_points=num_points,
            hidden=amorphous_hidden,
        )
        
        # 3. Router
        self.router = mlp([latent_size, router_hidden, 2], bn=False)
        
        # 4. Transformer Refiner with FiLM
        self.refiner = FiLMTransformerRefiner(
            latent_dim=latent_size,
            hidden_dim=transformer_hidden,
            n_heads=transformer_heads,
            n_layers=transformer_layers,
            max_displacement=max_refinement_displacement,
        )
        
        # Store last forward info for debugging/analysis
        self.last_aux = {}

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, D) latent vectors
            
        Returns:
            final_pts: (B, N, 3) reconstructed point clouds
            vq_loss: scalar VQ commitment loss
            weights: (B, 2) routing weights [crystal, amorphous]
        """
        B = z.shape[0]
        
        # 1. VQ Bottleneck
        vq_loss = torch.tensor(0.0, device=z.device)
        if self.use_vq:
            z_q, vq_loss, indices = self.vq(z)
            z_structure = z_q
        else:
            z_structure = z
            indices = None

        # 2. Generate Templates from both experts
        out_lattice = self.expert_lattice(z_structure)
        out_amorphous = self.expert_amorphous(z_structure)
        
        # 3. Hard Gating
        logits = self.router(z_structure)
        if self.training:
            weights = F.gumbel_softmax(logits, tau=1.0, hard=True)
        else:
            idx = torch.argmax(logits, dim=-1)
            weights = F.one_hot(idx, num_classes=2).float()
        
        # Select output (hard selection due to hard=True)
        w_lat = weights[:, 0].view(-1, 1, 1)
        w_amo = weights[:, 1].view(-1, 1, 1)
        mixed = w_lat * out_lattice + w_amo * out_amorphous
        
        # 4. Refine with original z (captures instance-specific details)
        if self.use_refiner:
            final_pts, residual_mag = self.refiner(mixed, z)
        else:
            final_pts = mixed
            residual_mag = torch.tensor(0.0, device=z.device)
        
        # Store auxiliary info
        self.last_aux = {
            'template': mixed.detach(),
            'lattice_output': out_lattice.detach(),
            'amorphous_output': out_amorphous.detach(),
            'vq_indices': indices,
            'residual_mag': residual_mag.detach() if self.use_refiner else residual_mag,
            'route_logits': logits.detach(),
        }
        
        return final_pts, vq_loss, weights

def spherical_fibonacci_points(n: int, device: torch.device) -> torch.Tensor:
    """Deterministic quasi-uniform directions on S^2."""
    i = torch.arange(n, dtype=torch.float32, device=device) + 0.5
    phi = 2.0 * math.pi * i / ((1.0 + 5.0 ** 0.5) * 0.5)
    cos_theta = 1.0 - 2.0 * i / n
    sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=0.0))
    x = torch.cos(phi) * sin_theta
    y = torch.sin(phi) * sin_theta
    z = cos_theta
    return torch.stack([x, y, z], dim=-1)


class AmorphousExpert(nn.Module):
    """Original amorphous expert - kept for compatibility"""
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

        # Pre-compute directions on CPU, will be moved to correct device by Lightning/DDP
        self.register_buffer("_dirs", spherical_fibonacci_points(num_points, device=torch.device('cpu')), persistent=True)

    def _ensure_dirs(self, device):
        # Buffer will be automatically moved to the correct device by DDP
        # Only need to check if device mismatch (shouldn't happen with proper DDP setup)
        if self._dirs.device != device:
            self._dirs = self._dirs.to(device)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0)
        self._ensure_dirs(z.device)

        r = F.softplus(self.head_r(z), beta=1.0) + 1e-4
        s = F.softplus(self.head_scale(z), beta=1.0).clamp_min(0.1)

        dirs = self._dirs.unsqueeze(0).expand(B, -1, -1)
        pts = dirs * r.unsqueeze(-1) * s.unsqueeze(-1)

        if self.training and self.noise_std > 0:
            pts = pts + torch.randn_like(pts) * self.noise_std

        pts = pts - pts.mean(dim=1, keepdim=True)
        return pts


@register_decoder("VQMoE_v2")
class VQMoEDecoderV2(Decoder):
    """
    Version 2: Uses improved lattice expert but keeps original amorphous expert.
    Useful if your amorphous reconstruction is already working well.
    """
    def __init__(
        self,
        num_points: int = 80,
        latent_size: int = 48,
        vq_size: int = 128,
        use_vq: bool = True,
        router_hidden: int = 64,
        transformer_hidden: int = 64,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        lattice_hidden: int = 64,
        amorphous_hidden: int = 64,
        num_crystal_prototypes: int = 4,
        atoms_per_cell: int = 4,
        max_refinement_displacement: float = 0.3,
        amorphous_noise_std: float = 0.02,
        force_crystal: bool = False,
        use_refiner: bool = True,
    ):
        super().__init__()
        
        self.use_vq = use_vq
        self.num_points = num_points
        self.force_crystal = force_crystal
        self.use_refiner = use_refiner

        if self.use_vq:
            self.vq = VectorQuantizer(num_embeddings=vq_size, embedding_dim=latent_size)
        
        # Improved lattice expert
        self.expert_lattice = MultiCrystalExpert(
            latent_size=latent_size,
            num_points=num_points,
            num_prototypes=num_crystal_prototypes,
            atoms_per_cell=atoms_per_cell,
            hidden=lattice_hidden,
        )
        
        # Improved amorphous expert (full 3D capacity)
        self.expert_amorphous = ImprovedAmorphousExpert(
            latent_size=latent_size,
            num_points=num_points,
            hidden=amorphous_hidden,
        )
        
        self.router = mlp([latent_size, router_hidden, 2], bn=False)
        
        self.refiner = FiLMTransformerRefiner(
            latent_dim=latent_size,
            hidden_dim=transformer_hidden,
            n_heads=transformer_heads,
            n_layers=transformer_layers,
            max_displacement=max_refinement_displacement,
        )
        
        self.last_aux = {}

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = z.shape[0]
        
        vq_loss = torch.tensor(0.0, device=z.device)
        if self.use_vq:
            z_q, vq_loss, indices = self.vq(z)
            z_structure = z_q
        else:
            z_structure = z
            indices = None

        out_lattice = self.expert_lattice(z_structure)
        out_amorphous = self.expert_amorphous(z_structure)

        logits = self.router(z_structure)
        
        if self.force_crystal:
            # Force 100% weight on Crystal (index 0)
            weights = torch.zeros_like(logits)
            weights[:, 0] = 1.0
        elif self.training:
             weights = F.gumbel_softmax(logits, tau=1.0, hard=True)
        else:
             idx = torch.argmax(logits, dim=-1)
             weights = F.one_hot(idx, num_classes=2).float()
        
        w_lat = weights[:, 0].view(-1, 1, 1)
        w_amo = weights[:, 1].view(-1, 1, 1)
            
        mixed = w_lat * out_lattice + w_amo * out_amorphous
        
        if self.use_refiner:
            final_pts, residual_mag = self.refiner(mixed, z)
        else:
            final_pts = mixed
            residual_mag = torch.tensor(0.0, device=z.device)
        
        self.last_aux = {
            'template': mixed.detach(),
            'lattice_output': out_lattice.detach(),
            'amorphous_output': out_amorphous.detach(),
            'vq_indices': indices,
            'residual_mag': residual_mag.detach() if self.use_refiner else residual_mag,
            'route_logits': logits.detach(),
        }
        
        return final_pts, vq_loss, weights


# =============================================================================
# MINIMAL SANITY CHECK
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    B, N, D = 4, 80, 96
    z = torch.randn(B, D)

    # Test original interface
    dec = VQMoEDecoder(
        num_points=N,
        latent_size=D,
        vq_size=64,
        use_vq=True,
        num_crystal_prototypes=4,
        atoms_per_cell=4,
    )

    Xhat, vq_loss, weights = dec(z)
    print("Output shape:", Xhat.shape)
    print("VQ loss:", vq_loss.item())
    print("Routing weights shape:", weights.shape)
    print("Routing weights (first 2):", weights[:2])
    print("Last aux keys:", dec.last_aux.keys())
    
    # Test V2
    dec_v2 = VQMoEDecoderV2(
        num_points=N,
        latent_size=D,
        vq_size=64,
    )
    Xhat2, vq_loss2, weights2 = dec_v2(z)
    print("\nV2 Output shape:", Xhat2.shape)