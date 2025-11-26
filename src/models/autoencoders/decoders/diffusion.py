import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ..base import Decoder
from ..registry import register_decoder

# ---------------------------------------------------------------------------
# E(3) Equivariant Layer (Adapted from EGNN)
# ---------------------------------------------------------------------------
class E_GCL(nn.Module):
    """
    Equivariant Graph Convolutional Layer.
    Updates node features (h) and coordinates (x) based on relative distances.
    """
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, act_fn=nn.SiLU(), residual=True, attention=False):
        super(E_GCL, self).__init__()
        self.residual = residual
        self.attention = attention
        input_edge = input_nf * 2 + edges_in_d
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 1, hidden_nf), # +1 for radial distance
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )
        
        # Coordinate update net: weights the relative difference (x_i - x_j)
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False) # Outputs scalar weight
        )
        
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = torch.zeros(x.size(0), edge_attr.size(1), device=x.device, dtype=edge_attr.dtype)
        agg.index_add_(0, row, edge_attr)
        
        # Normalize by degree to prevent explosion in dense graphs
        num_edges = edge_index.shape[1]
        num_nodes = x.shape[0]
        avg_degree = num_edges / max(num_nodes, 1)
        if avg_degree > 1:
            agg = agg / avg_degree
        
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.residual:
            out = x + out
        return out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        # Use tanh to bound coordinate update weights for stability
        coord_weights = torch.tanh(self.coord_mlp(edge_feat))
        trans = coord_diff * coord_weights
        
        # Aggregate updates: x_i_new = x_i + sum_j (x_i - x_j) * weight
        agg_x = torch.zeros_like(coord)
        agg_x.index_add_(0, row, trans)
        
        # Normalize by number of neighbors to prevent explosion in dense graphs
        num_edges = edge_index.shape[1]
        num_nodes = coord.shape[0]
        avg_degree = num_edges / max(num_nodes, 1)
        if avg_degree > 1:
            agg_x = agg_x / avg_degree
        
        return coord + agg_x

    def forward(self, h, x, edge_index, edge_attr=None):
        row, col = edge_index
        coord_diff = x[row] - x[col]
        radial = torch.sum(coord_diff**2, 1, keepdim=True) + 1e-8  # Squared distance with eps for stability
        
        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(x, edge_index, coord_diff, edge_feat)
        h = self.node_model(h, edge_index, edge_feat, None)
        return h, coord

# ---------------------------------------------------------------------------
# The Denoiser Network
# ---------------------------------------------------------------------------
class EGNN_Denoiser(nn.Module):
    def __init__(self, latent_dim, hidden_dim=64, n_layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.time_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Initial node feature embedding (constant 1s or random)
        self.node_embedding = nn.Embedding(1, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(E_GCL(
                input_nf=hidden_dim, 
                output_nf=hidden_dim, 
                hidden_nf=hidden_dim, 
                edges_in_d=0, # No edge attributes
                residual=True
            ))

    def forward(self, x, t, z):
        """
        x: (B, N, 3) noisy coordinates
        t: (B,) timesteps
        z: (B, D) invariant latent conditioning
        """
        B, N, _ = x.shape
        
        # 1. Create fully connected graph (or KNN)
        # For small N (<200), fully connected is fast and best for gradients
        # We construct edge indices on the fly
        rows, cols = torch.meshgrid(
            torch.arange(N, device=x.device), 
            torch.arange(N, device=x.device), 
            indexing='ij'
        )
        rows = rows.reshape(-1)
        cols = cols.reshape(-1)
        # Remove self-loops
        mask = rows != cols
        rows, cols = rows[mask], cols[mask]
        
        # Batch offset
        batch_offset = torch.arange(B, device=x.device) * N
        batch_rows = (rows.unsqueeze(0) + batch_offset.unsqueeze(1)).view(-1)
        batch_cols = (cols.unsqueeze(0) + batch_offset.unsqueeze(1)).view(-1)
        edge_index = torch.stack([batch_rows, batch_cols], dim=0)

        # 2. Prepare features
        # Flatten x to (B*N, 3)
        x_flat = x.reshape(B*N, 3)
        
        # Node features h: combine time + latent + base embedding
        t_emb = self.time_embedding(t.unsqueeze(-1)).unsqueeze(1).repeat(1, N, 1) # (B, N, H)
        z_emb = self.latent_proj(z).unsqueeze(1).repeat(1, N, 1) # (B, N, H)
        h_base = self.node_embedding(torch.zeros(B*N, dtype=torch.long, device=x.device)).view(B, N, -1)
        
        h = h_base + t_emb + z_emb
        h_flat = h.reshape(B*N, -1)
        
        # 3. Message Passing Steps
        for layer in self.layers:
            h_flat, x_flat = layer(h_flat, x_flat, edge_index)
            
        # 4. Output (Predicted Noise / Velocity)
        # We output the updated x as the "denoised direction"
        # Since standard EGNN outputs updated coords, we treat (x_new - x_in) as the predicted noise direction
        pred_noise = x_flat.reshape(B, N, 3) - x
        
        return pred_noise

# ---------------------------------------------------------------------------
# The Main Decoder Wrapper
# ---------------------------------------------------------------------------
@register_decoder("DiffusionEGNN")
class DiffusionDecoder(Decoder):
    def __init__(
        self,
        latent_size: int,
        num_points: int,
        hidden_dim: int = 128,
        layers: int = 4,
        timesteps: int = 100, # Keep low for speed during prototyping
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ):
        super().__init__()
        self.num_points = num_points
        self.timesteps = timesteps
        
        # Denoiser
        self.denoiser = EGNN_Denoiser(latent_size, hidden_dim, layers)
        
        # Diffusion parameters (DDPM)
        self.register_buffer('betas', torch.linspace(beta_start, beta_end, timesteps))
        self.register_buffer('alphas', 1. - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def get_loss(self, x_start, z):
        """
        Forward diffusion (Training)
        x_start: (B, N, 3) Real data
        z: (B, D) Latent
        """
        B = x_start.shape[0]
        dtype = x_start.dtype
        
        t = torch.randint(0, self.timesteps, (B,), device=x_start.device)
        
        # Add noise (match dtype)
        noise = torch.randn_like(x_start)
        
        # Extract alpha_bar for specific t (cast to input dtype for mixed precision)
        a_bar = self.alphas_cumprod[t].to(dtype).view(B, 1, 1)
        
        # Compute noisy input
        sqrt_a_bar = torch.sqrt(a_bar)
        sqrt_one_minus_a_bar = torch.sqrt(1 - a_bar)
        x_noisy = sqrt_a_bar * x_start + sqrt_one_minus_a_bar * noise
        
        # Predict noise
        # Note: Normalize t to [0,1] for network
        t_norm = t.to(dtype) / self.timesteps
        noise_pred = self.denoiser(x_noisy, t_norm, z)
        
        # Simple MSE Loss on noise
        # This is rotation invariant in expectation if the network is equivariant!
        # Because if X rotates, Noise rotates, Prediction rotates -> MSE stays same.
        loss = F.mse_loss(noise_pred, noise)
        
        # Safety check for NaN
        if torch.isnan(loss):
            loss = torch.tensor(0.0, device=x_start.device, dtype=dtype, requires_grad=True)
        
        return loss

    @torch.no_grad()
    def sample(self, z):
        """
        Reverse diffusion (Inference)
        """
        B = z.shape[0]
        device = z.device
        dtype = z.dtype
        
        # Start from pure noise (match dtype for mixed precision)
        x = torch.randn(B, self.num_points, 3, device=device, dtype=dtype)
        
        # Loop backwards
        for i in reversed(range(self.timesteps)):
            t = torch.tensor([i] * B, device=device, dtype=dtype)
            t_norm = t / self.timesteps
            
            # Predict noise
            noise_pred = self.denoiser(x, t_norm, z)
            
            # Step parameters (cast to dtype for mixed precision)
            alpha = self.alphas[i].to(dtype)
            alpha_cumprod = self.alphas_cumprod[i].to(dtype)
            beta = self.betas[i].to(dtype)
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0
                
            # Update x (Standard DDPM update)
            x = (1 / torch.sqrt(alpha)) * (
                x - ((1 - alpha) / torch.sqrt(1 - alpha_cumprod)) * noise_pred
            ) + torch.sqrt(beta) * noise
            
            # Clamp to prevent numerical instability (especially early in training)
            x = torch.clamp(x, -10.0, 10.0)
        
        # Final safety check: replace any NaN/Inf with zeros
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        return x

    def forward(self, z, gt_pts=None):
        """
        Dual behavior:
        - If gt_pts is provided (Training): Returns (None, loss, None)
        - If gt_pts is None (Inference): Returns (generated_pts, 0, None)
        """
        if self.training and gt_pts is not None:
            # Training Mode
            loss = self.get_loss(gt_pts, z)
            return gt_pts, loss, None # Return dummy pts
        else:
            # Inference Mode
            generated = self.sample(z)
            return generated, torch.tensor(0.0).to(z.device), None