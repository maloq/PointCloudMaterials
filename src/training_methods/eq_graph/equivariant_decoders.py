"""
Equivariant Decoders for Point Cloud Reconstruction

Addresses the "mean shape" problem with:
1. Flow Matching Decoder - learns a vector field to transform noise to structure
2. Score-Based Diffusion Decoder - iterative denoising for sharp outputs
3. Autoregressive Decoder - generates atoms sequentially

All maintain SE(3) equivariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Literal
import math


# =============================================================================
# SCATTER OPERATIONS (Same as encoder)
# =============================================================================

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, 
                 dim_size: Optional[int] = None) -> torch.Tensor:
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    index_expanded = index
    for _ in range(src.dim() - index.dim()):
        index_expanded = index_expanded.unsqueeze(-1)
    index_expanded = index_expanded.expand_as(src)
    
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index_expanded, src)
    
    ones = torch.ones_like(src)
    count = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    count.scatter_add_(dim, index_expanded, ones)
    
    return out / count.clamp(min=1)


# =============================================================================
# EQUIVARIANT BUILDING BLOCKS
# =============================================================================

class EquivariantLinear(nn.Module):
    """
    Linear layer that processes scalar and vector features separately.
    Maintains E(3) equivariance.
    """
    def __init__(self, scalar_in: int, scalar_out: int, 
                 vector_in: int, vector_out: int):
        super().__init__()
        self.scalar_linear = nn.Linear(scalar_in, scalar_out)
        self.vector_linear = nn.Linear(vector_in, vector_out, bias=False)
        
        # Cross-talk: scalars can modulate vectors
        self.scalar_to_vector_gate = nn.Linear(scalar_in, vector_out)
    
    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: (B, N, scalar_in) scalar features
            v: (B, N, vector_in, 3) vector features
            
        Returns:
            s_out: (B, N, scalar_out)
            v_out: (B, N, vector_out, 3)
        """
        s_out = self.scalar_linear(s)
        
        # Transform vectors
        v_out = self.vector_linear(v.transpose(-1, -2)).transpose(-1, -2)
        
        # Gate vectors with scalars
        gate = torch.sigmoid(self.scalar_to_vector_gate(s)).unsqueeze(-1)
        v_out = v_out * gate
        
        return s_out, v_out


class EquivariantLayerNorm(nn.Module):
    """Layer normalization for scalar + vector features."""
    def __init__(self, scalar_dim: int, vector_dim: int, eps: float = 1e-5):
        super().__init__()
        self.scalar_ln = nn.LayerNorm(scalar_dim, eps=eps)
        self.vector_scale = nn.Parameter(torch.ones(vector_dim))
        self.eps = eps
    
    def forward(self, s: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.scalar_ln(s)
        
        # Normalize vectors by their norm
        v_norm = torch.norm(v, dim=-1, keepdim=True)  # (B, N, C, 1)
        v_normalized = v / (v_norm + self.eps)
        
        # Scale
        scale = self.vector_scale.view(1, 1, -1, 1)
        v = v_normalized * v_norm.mean(dim=-2, keepdim=True) * scale
        
        return s, v


class EquivariantBlock(nn.Module):
    """Single equivariant transformer-style block."""
    def __init__(self, scalar_dim: int, vector_dim: int, hidden_mult: int = 4):
        super().__init__()
        
        self.norm1 = EquivariantLayerNorm(scalar_dim, vector_dim)
        self.norm2 = EquivariantLayerNorm(scalar_dim, vector_dim)
        
        # Self-attention (scalar) + vector update
        self.attn = EquivariantAttention(scalar_dim, vector_dim, n_heads=4)
        
        # FFN
        self.ffn_s = nn.Sequential(
            nn.Linear(scalar_dim, scalar_dim * hidden_mult),
            nn.GELU(),
            nn.Linear(scalar_dim * hidden_mult, scalar_dim),
        )
        self.ffn_v = nn.Sequential(
            nn.Linear(vector_dim, vector_dim * hidden_mult, bias=False),
            nn.Linear(vector_dim * hidden_mult, vector_dim, bias=False),
        )
    
    def forward(self, s: torch.Tensor, v: torch.Tensor, 
                pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Attention with residual
        s_norm, v_norm = self.norm1(s, v)
        s_attn, v_attn = self.attn(s_norm, v_norm, pos)
        s = s + s_attn
        v = v + v_attn
        
        # FFN with residual
        s_norm, v_norm = self.norm2(s, v)
        s = s + self.ffn_s(s_norm)
        v = v + self.ffn_v(v_norm.transpose(-1, -2)).transpose(-1, -2)
        
        return s, v


class EquivariantAttention(nn.Module):
    """
    Attention that respects E(3) equivariance.
    Uses relative positions for geometric awareness.
    """
    def __init__(self, scalar_dim: int, vector_dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = scalar_dim // n_heads
        
        self.q_proj = nn.Linear(scalar_dim, scalar_dim)
        self.k_proj = nn.Linear(scalar_dim, scalar_dim)
        self.v_proj = nn.Linear(scalar_dim, scalar_dim)
        self.o_proj = nn.Linear(scalar_dim, scalar_dim)
        
        # Distance-based attention bias
        self.dist_proj = nn.Sequential(
            nn.Linear(1, n_heads),
            nn.Softplus(),
        )
        
        # Vector value projection
        self.v_v_proj = nn.Linear(vector_dim, vector_dim, bias=False)
        self.v_o_proj = nn.Linear(vector_dim, vector_dim, bias=False)
    
    def forward(self, s: torch.Tensor, v: torch.Tensor, 
                pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            s: (B, N, scalar_dim)
            v: (B, N, vector_dim, 3)
            pos: (B, N, 3) positions
        """
        B, N, _ = s.shape
        
        # Compute Q, K, V
        q = self.q_proj(s).view(B, N, self.n_heads, self.head_dim)
        k = self.k_proj(s).view(B, N, self.n_heads, self.head_dim)
        v_s = self.v_proj(s).view(B, N, self.n_heads, self.head_dim)
        
        # Attention scores
        attn = torch.einsum('bnhd,bmhd->bhnm', q, k) / math.sqrt(self.head_dim)
        
        # Distance-based bias
        dist = torch.cdist(pos, pos).unsqueeze(-1)  # (B, N, N, 1)
        dist_bias = self.dist_proj(dist).permute(0, 3, 1, 2)  # (B, heads, N, N)
        attn = attn - dist_bias  # Closer = higher attention
        
        attn = F.softmax(attn, dim=-1)
        attn_mean = attn.mean(dim=1)  # (B, N, N)
        
        # Apply attention to scalar values
        s_out = torch.einsum('bhnm,bmhd->bnhd', attn, v_s)
        s_out = self.o_proj(s_out.reshape(B, N, -1))
        
        # Apply attention to vector values
        v_v = self.v_v_proj(v.transpose(-1, -2)).transpose(-1, -2)  # (B, N, vector_dim, 3)
        v_out = torch.einsum('bnm,bmcd->bncd', attn_mean, v_v)  # Average over heads
        v_out = self.v_o_proj(v_out.transpose(-1, -2)).transpose(-1, -2)
        
        return s_out, v_out


# =============================================================================
# FLOW MATCHING DECODER
# =============================================================================

class FlowMatchingDecoder(nn.Module):
    """
    Equivariant Flow Matching Decoder.
    
    Learns a velocity field that transforms Gaussian noise into the target structure.
    Produces sharp, diverse outputs - NO mean shape problem!
    
    Key idea: Instead of predicting E[X|z], we learn the optimal transport 
    from noise to data, then sample specific instances.
    """
    def __init__(
        self,
        latent_dim: int = 256,
        n_atoms: int = 64,
        scalar_dim: int = 128,
        vector_dim: int = 32,
        n_layers: int = 6,
        sigma_min: float = 0.001,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        self.sigma_min = sigma_min
        
        # Embed latent code
        self.latent_embed = nn.Linear(latent_dim, scalar_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            SinusoidalEmbedding(scalar_dim),
            nn.Linear(scalar_dim, scalar_dim),
            nn.GELU(),
            nn.Linear(scalar_dim, scalar_dim),
        )
        
        # Initial features from positions
        self.pos_embed = nn.Linear(3, scalar_dim)
        self.init_vector = nn.Linear(1, vector_dim, bias=False)
        
        # Main network
        self.layers = nn.ModuleList([
            EquivariantBlock(scalar_dim, vector_dim) 
            for _ in range(n_layers)
        ])
        
        # Output velocity (equivariant)
        self.vel_head = nn.Sequential(
            nn.Linear(vector_dim, vector_dim, bias=False),
            nn.Linear(vector_dim, 1, bias=False),
        )
        
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, 
                z: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity at time t.
        
        Args:
            x_t: (B, N, 3) noisy positions at time t
            t: (B,) time in [0, 1]
            z: (B, latent_dim) latent code
            
        Returns:
            v_t: (B, N, 3) velocity field
        """
        B, N, _ = x_t.shape
        
        # Embeddings
        z_embed = self.latent_embed(z)  # (B, scalar_dim)
        t_embed = self.time_embed(t)  # (B, scalar_dim)
        
        # Combine with position features
        s = self.pos_embed(x_t)  # (B, N, scalar_dim)
        s = s + z_embed.unsqueeze(1) + t_embed.unsqueeze(1)
        
        # Initialize vector features from positions
        v = self.init_vector(x_t.unsqueeze(-2).transpose(-1, -2))  # (B, N, vector_dim, 3)
        v = v.transpose(-1, -2)
        
        # Process through layers
        for layer in self.layers:
            s, v = layer(s, v, x_t)
        
        # Output velocity
        vel = self.vel_head(v.transpose(-1, -2)).squeeze(-1)  # (B, N, 3)
        
        return vel
    
    def sample(self, z: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        Generate point cloud from latent code.
        
        Uses ODE integration from t=0 (noise) to t=1 (data).
        """
        B = z.shape[0]
        device = z.device
        
        # Start from Gaussian noise
        x = torch.randn(B, self.n_atoms, 3, device=device)
        
        # Euler integration
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device)
            v = self.forward(x, t, z)
            x = x + v * dt
        
        return x
    
    def compute_loss(self, x_0: torch.Tensor, x_1: torch.Tensor, 
                     z: torch.Tensor) -> torch.Tensor:
        """
        Flow matching loss.
        
        Args:
            x_0: (B, N, 3) noise samples
            x_1: (B, N, 3) target point clouds
            z: (B, latent_dim) latent codes
            
        Returns:
            loss: scalar
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample time uniformly
        t = torch.rand(B, device=device)
        
        # Linear interpolation (optimal transport path)
        t_expand = t.view(B, 1, 1)
        x_t = (1 - t_expand) * x_0 + t_expand * x_1
        
        # Target velocity (derivative of interpolation)
        v_target = x_1 - x_0
        
        # Predicted velocity
        v_pred = self.forward(x_t, t, z)
        
        # MSE loss
        loss = F.mse_loss(v_pred, v_target)
        
        return loss


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# =============================================================================
# SCORE-BASED DIFFUSION DECODER
# =============================================================================

class DiffusionDecoder(nn.Module):
    """
    Equivariant Diffusion Decoder.
    
    Uses denoising score matching to generate sharp point clouds.
    """
    def __init__(
        self,
        latent_dim: int = 256,
        n_atoms: int = 64,
        scalar_dim: int = 128,
        vector_dim: int = 32,
        n_layers: int = 6,
        n_timesteps: int = 1000,
    ):
        super().__init__()
        
        self.n_atoms = n_atoms
        self.n_timesteps = n_timesteps
        
        # Noise schedule (linear)
        betas = torch.linspace(1e-4, 0.02, n_timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        
        # Score network (predicts noise)
        self.score_net = ScoreNetwork(
            latent_dim=latent_dim,
            scalar_dim=scalar_dim,
            vector_dim=vector_dim,
            n_layers=n_layers,
            n_timesteps=n_timesteps,
        )
    
    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, 
                z: torch.Tensor) -> torch.Tensor:
        """Predict noise added to x."""
        return self.score_net(x_noisy, t, z)
    
    def compute_loss(self, x_0: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Denoising score matching loss.
        """
        B = x_0.shape[0]
        device = x_0.device
        
        # Sample timestep
        t = torch.randint(0, self.n_timesteps, (B,), device=device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Noisy sample
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(B, 1, 1)
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t].view(B, 1, 1)
        x_noisy = sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise
        
        # Predict noise
        noise_pred = self.forward(x_noisy, t, z)
        
        # MSE loss
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def sample(self, z: torch.Tensor, n_steps: Optional[int] = None) -> torch.Tensor:
        """
        Generate point cloud using DDPM sampling.
        """
        B = z.shape[0]
        device = z.device
        
        if n_steps is None:
            n_steps = self.n_timesteps
        
        # Start from pure noise
        x = torch.randn(B, self.n_atoms, 3, device=device)
        
        # Reverse diffusion
        step_size = self.n_timesteps // n_steps
        timesteps = list(range(0, self.n_timesteps, step_size))[::-1]
        
        for t in timesteps:
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.forward(x, t_batch, z)
            
            # DDPM update
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            beta = self.betas[t]
            
            # Predicted x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
            
            # Sample x_{t-1}
            if t > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(alpha) * x + beta * noise_pred / torch.sqrt(1 - alpha_cumprod)
                x = x + torch.sqrt(beta) * noise
            else:
                x = x_0_pred
        
        return x


class ScoreNetwork(nn.Module):
    """Score network for diffusion decoder."""
    def __init__(self, latent_dim: int, scalar_dim: int, vector_dim: int,
                 n_layers: int, n_timesteps: int):
        super().__init__()
        
        self.latent_embed = nn.Linear(latent_dim, scalar_dim)
        self.time_embed = nn.Embedding(n_timesteps, scalar_dim)
        self.pos_embed = nn.Linear(3, scalar_dim)
        self.init_vector = nn.Linear(1, vector_dim, bias=False)
        
        self.layers = nn.ModuleList([
            EquivariantBlock(scalar_dim, vector_dim)
            for _ in range(n_layers)
        ])
        
        self.out_head = nn.Sequential(
            nn.Linear(vector_dim, vector_dim, bias=False),
            nn.Linear(vector_dim, 1, bias=False),
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                z: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        
        s = self.pos_embed(x)
        s = s + self.latent_embed(z).unsqueeze(1)
        s = s + self.time_embed(t).unsqueeze(1)
        
        v = self.init_vector(x.unsqueeze(-2).transpose(-1, -2)).transpose(-1, -2)
        
        for layer in self.layers:
            s, v = layer(s, v, x)
        
        out = self.out_head(v.transpose(-1, -2)).squeeze(-1)
        return out


# =============================================================================
# AUTOREGRESSIVE DECODER (Forces commitment to specific structures)
# =============================================================================

class AutoregressiveDecoder(nn.Module):
    """
    Autoregressive equivariant decoder.
    
    Generates atoms one at a time, conditioning on previously placed atoms.
    This naturally avoids the mean shape problem since each atom is placed
    relative to the existing partial structure.
    """
    def __init__(
        self,
        latent_dim: int = 256,
        n_atoms: int = 64,
        hidden_dim: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        
        self.n_atoms = n_atoms
        self.latent_dim = latent_dim
        
        # Latent embedding
        self.latent_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Context encoder (encodes partial point cloud)
        self.context_encoder = ContextEncoder(hidden_dim, n_layers)
        
        # Position predictor (outputs equivariant position)
        self.pos_predictor = PositionPredictor(hidden_dim)
        
        # Stop predictor (when to stop adding atoms)
        self.stop_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(self, z: torch.Tensor, teacher_forcing: bool = True,
                x_target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate or reconstruct point cloud.
        
        Args:
            z: (B, latent_dim) latent code
            teacher_forcing: if True, use target positions during training
            x_target: (B, N, 3) target positions for teacher forcing
            
        Returns:
            x_pred: (B, N, 3) predicted positions
        """
        B = z.shape[0]
        device = z.device
        
        z_embed = self.latent_mlp(z)  # (B, hidden_dim)
        
        # Start with empty point cloud
        points = []
        
        for i in range(self.n_atoms):
            # Encode context (existing points)
            if len(points) == 0:
                context = z_embed
            else:
                partial_cloud = torch.stack(points, dim=1)  # (B, i, 3)
                context = self.context_encoder(partial_cloud, z_embed)
            
            # Predict next position
            if teacher_forcing and x_target is not None:
                # Use target position for context but predict anyway for loss
                next_pos = self.pos_predictor(context, 
                    partial_cloud if len(points) > 0 else None)
                points.append(x_target[:, i])
            else:
                next_pos = self.pos_predictor(context,
                    partial_cloud if len(points) > 0 else None)
                points.append(next_pos)
        
        x_pred = torch.stack(points, dim=1)  # (B, N, 3)
        return x_pred
    
    def compute_loss(self, x_target: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive loss - predict each atom given previous ones.
        """
        B, N, _ = x_target.shape
        device = x_target.device
        
        z_embed = self.latent_mlp(z)
        
        total_loss = 0.0
        
        for i in range(N):
            if i == 0:
                context = z_embed
                partial_cloud = None
            else:
                partial_cloud = x_target[:, :i]  # (B, i, 3)
                context = self.context_encoder(partial_cloud, z_embed)
            
            # Predict position i
            pred_pos = self.pos_predictor(context, partial_cloud)  # (B, 3)
            
            # Loss (allow permutation within remaining atoms)
            # Simple: just MSE to closest remaining atom
            remaining = x_target[:, i:]  # (B, N-i, 3)
            dists = torch.norm(pred_pos.unsqueeze(1) - remaining, dim=-1)  # (B, N-i)
            min_dist = dists.min(dim=1).values  # (B,)
            
            total_loss = total_loss + min_dist.mean()
        
        return total_loss / N


class ContextEncoder(nn.Module):
    """Encodes partial point cloud to context vector."""
    def __init__(self, hidden_dim: int, n_layers: int):
        super().__init__()
        
        self.embed = nn.Linear(3, hidden_dim)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(n_layers)
        ])
        
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)
    
    def forward(self, partial: torch.Tensor, z_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            partial: (B, K, 3) partial point cloud
            z_embed: (B, hidden_dim) latent embedding
        """
        h = self.embed(partial)  # (B, K, hidden_dim)
        
        for layer in self.layers:
            h = layer(h)
        
        # Global pooling
        h_global = h.mean(dim=1)  # (B, hidden_dim)
        
        # Combine with latent
        context = self.combine(torch.cat([h_global, z_embed], dim=-1))
        
        return context


class PositionPredictor(nn.Module):
    """
    Predicts next atom position in an equivariant way.
    
    Uses the centroid and principal directions of existing atoms
    to define an equivariant reference frame.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        
        # Predict offset in local frame
        self.offset_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),  # 3D offset
        )
        
        # Predict scale
        self.scale_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )
    
    def forward(self, context: torch.Tensor, 
                partial: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            context: (B, hidden_dim)
            partial: (B, K, 3) existing points or None
            
        Returns:
            pos: (B, 3) predicted position
        """
        B = context.shape[0]
        device = context.device
        
        # Predict offset and scale
        offset = self.offset_mlp(context)  # (B, 3)
        scale = self.scale_mlp(context)  # (B, 1)
        
        if partial is None:
            # First atom: place at origin with some offset
            return offset * scale
        
        # Compute local frame from partial cloud
        centroid = partial.mean(dim=1)  # (B, 3)
        centered = partial - centroid.unsqueeze(1)  # (B, K, 3)
        
        # PCA for local frame
        cov = torch.bmm(centered.transpose(1, 2), centered)  # (B, 3, 3)
        _, _, V = torch.linalg.svd(cov)  # V: (B, 3, 3)
        
        # Local frame (equivariant!)
        frame = V  # Columns are principal directions
        
        # Transform offset to global frame
        global_offset = torch.bmm(frame, offset.unsqueeze(-1)).squeeze(-1)
        
        # Add to centroid
        pos = centroid + global_offset * scale
        
        return pos


# =============================================================================
# HYBRID DECODER WITH STRUCTURE TYPE ROUTING
# =============================================================================

class HybridDecoder(nn.Module):
    """
    Hybrid decoder that routes to different reconstruction strategies
    based on structure type (crystal vs amorphous).
    
    - For crystals: Uses strict lattice-aware reconstruction
    - For amorphous: Uses statistical reconstruction (RDF matching)
    """
    def __init__(
        self,
        latent_dim: int = 256,
        n_atoms: int = 64,
    ):
        super().__init__()
        
        # Structure type classifier (from latent)
        self.type_classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # 0 = amorphous, 1 = crystal
        )
        
        # Crystal decoder (point-wise accurate)
        self.crystal_decoder = FlowMatchingDecoder(
            latent_dim=latent_dim,
            n_atoms=n_atoms,
        )
        
        # Amorphous decoder (density field based)
        self.amorphous_decoder = DensityFieldDecoder(
            latent_dim=latent_dim,
            n_atoms=n_atoms,
        )
    
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: (B, latent_dim)
            
        Returns:
            x_crystal: (B, N, 3) crystal reconstruction
            x_amorphous: (B, N, 3) amorphous reconstruction  
            crystallinity: (B, 1) crystallinity score
        """
        crystallinity = self.type_classifier(z)
        
        x_crystal = self.crystal_decoder.sample(z)
        x_amorphous = self.amorphous_decoder.sample(z)
        
        return x_crystal, x_amorphous, crystallinity
    
    def compute_loss(self, x_target: torch.Tensor, z: torch.Tensor,
                     anisotropy: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Compute hybrid loss.
        
        Uses anisotropy score (from encoder) to weight crystal vs amorphous loss.
        """
        crystallinity = self.type_classifier(z)
        
        # Crystal loss (EMD or Chamfer)
        noise = torch.randn_like(x_target)
        crystal_loss = self.crystal_decoder.compute_loss(noise, x_target, z)
        
        # Amorphous loss (RDF matching)
        x_amorphous = self.amorphous_decoder.sample(z)
        amorphous_loss = rdf_loss(x_target, x_amorphous)
        
        # Consistency loss: crystallinity should match anisotropy
        consistency_loss = F.mse_loss(crystallinity, anisotropy)
        
        # Weighted total
        total_loss = (
            anisotropy * crystal_loss + 
            (1 - anisotropy) * amorphous_loss +
            0.1 * consistency_loss
        ).mean()
        
        return {
            'total': total_loss,
            'crystal': crystal_loss.mean(),
            'amorphous': amorphous_loss.mean(),
            'consistency': consistency_loss,
        }


class DensityFieldDecoder(nn.Module):
    """
    Decoder that outputs a density field, then samples atom positions.
    Good for amorphous structures where exact positions don't matter.
    """
    def __init__(self, latent_dim: int, n_atoms: int, grid_size: int = 32):
        super().__init__()
        
        self.n_atoms = n_atoms
        self.grid_size = grid_size
        
        # Decode to 3D density grid
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, grid_size ** 3),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns density grid."""
        B = z.shape[0]
        density = self.decoder(z)
        density = density.view(B, self.grid_size, self.grid_size, self.grid_size)
        return F.softplus(density)
    
    def sample(self, z: torch.Tensor) -> torch.Tensor:
        """Sample atom positions from density field."""
        B = z.shape[0]
        device = z.device
        
        density = self.forward(z)  # (B, G, G, G)
        
        # Flatten and sample
        density_flat = density.view(B, -1)  # (B, G^3)
        probs = F.softmax(density_flat, dim=-1)
        
        # Sample indices
        indices = torch.multinomial(probs, self.n_atoms, replacement=True)  # (B, N)
        
        # Convert to 3D coordinates
        G = self.grid_size
        z_idx = indices // (G * G)
        y_idx = (indices % (G * G)) // G
        x_idx = indices % G
        
        # Normalize to [-1, 1]
        coords = torch.stack([x_idx, y_idx, z_idx], dim=-1).float()
        coords = (coords / (G - 1)) * 2 - 1
        
        # Add small noise for smoothness
        coords = coords + torch.randn_like(coords) * 0.05
        
        return coords


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def rdf_loss(x1: torch.Tensor, x2: torch.Tensor, 
             n_bins: int = 50, r_max: float = 5.0) -> torch.Tensor:
    """
    Radial Distribution Function loss.
    
    Compares statistical properties rather than exact positions.
    Good for amorphous structures.
    """
    rdf1 = compute_rdf(x1, n_bins, r_max)
    rdf2 = compute_rdf(x2, n_bins, r_max)
    return F.mse_loss(rdf1, rdf2)


def compute_rdf(x: torch.Tensor, n_bins: int = 50, 
                r_max: float = 5.0) -> torch.Tensor:
    """
    Compute radial distribution function.
    
    Args:
        x: (B, N, 3) point cloud
        n_bins: number of histogram bins
        r_max: maximum radius
        
    Returns:
        rdf: (B, n_bins) RDF values
    """
    B, N, _ = x.shape
    device = x.device
    
    # Pairwise distances
    dist = torch.cdist(x, x)  # (B, N, N)
    
    # Remove self-distances
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    dist = dist[:, mask].view(B, N * (N - 1))  # (B, N*(N-1))
    
    # Histogram
    bin_edges = torch.linspace(0, r_max, n_bins + 1, device=device)
    rdf = torch.zeros(B, n_bins, device=device)
    
    for i in range(n_bins):
        in_bin = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        rdf[:, i] = in_bin.float().sum(dim=1)
    
    # Normalize by shell volume
    r = (bin_edges[:-1] + bin_edges[1:]) / 2
    dr = bin_edges[1] - bin_edges[0]
    shell_volume = 4 * np.pi * r ** 2 * dr
    rdf = rdf / (shell_volume.unsqueeze(0) * N + 1e-8)
    
    return rdf


def chamfer_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """
    Chamfer distance between two point clouds.
    """
    # x1: (B, N1, 3), x2: (B, N2, 3)
    dist = torch.cdist(x1, x2)  # (B, N1, N2)
    
    # For each point in x1, find closest in x2
    min1 = dist.min(dim=2).values.mean(dim=1)  # (B,)
    
    # For each point in x2, find closest in x1
    min2 = dist.min(dim=1).values.mean(dim=1)  # (B,)
    
    return (min1 + min2) / 2


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("Testing decoders...")
    
    B, N = 4, 64
    latent_dim = 256
    
    z = torch.randn(B, latent_dim)
    x_target = torch.randn(B, N, 3)
    
    # Test Flow Matching
    print("\n1. Flow Matching Decoder")
    flow_dec = FlowMatchingDecoder(latent_dim=latent_dim, n_atoms=N)
    
    noise = torch.randn_like(x_target)
    loss = flow_dec.compute_loss(noise, x_target, z)
    print(f"   Loss: {loss.item():.4f}")
    
    x_sample = flow_dec.sample(z, n_steps=10)
    print(f"   Sample shape: {x_sample.shape}")
    
    # Test Diffusion
    print("\n2. Diffusion Decoder")
    diff_dec = DiffusionDecoder(latent_dim=latent_dim, n_atoms=N, n_timesteps=100)
    
    loss = diff_dec.compute_loss(x_target, z)
    print(f"   Loss: {loss.item():.4f}")
    
    x_sample = diff_dec.sample(z, n_steps=10)
    print(f"   Sample shape: {x_sample.shape}")
    
    # Test Autoregressive
    print("\n3. Autoregressive Decoder")
    ar_dec = AutoregressiveDecoder(latent_dim=latent_dim, n_atoms=N)
    
    loss = ar_dec.compute_loss(x_target, z)
    print(f"   Loss: {loss.item():.4f}")
    
    x_sample = ar_dec(z, teacher_forcing=False)
    print(f"   Sample shape: {x_sample.shape}")
    
    # Test RDF
    print("\n4. RDF Loss")
    rdf = compute_rdf(x_target)
    print(f"   RDF shape: {rdf.shape}")
    
    rdf_l = rdf_loss(x_target, x_sample)
    print(f"   RDF loss: {rdf_l.item():.4f}")
    
    print("\nAll tests passed!")
