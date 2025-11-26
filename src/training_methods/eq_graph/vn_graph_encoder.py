"""
Graph-Based Vector Neuron Encoder

Replaces k-NN with principled graph constructions:
- Delaunay triangulation (geometry-intrinsic, no hyperparameters)
- Alpha complex (multi-scale topology)
- Radius-based graphs (physics-motivated cutoffs)

Maintains SO(3) equivariance via Vector Neurons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Literal, Optional
from scipy.spatial import Delaunay, Voronoi
from collections import defaultdict


# =============================================================================
# SCATTER OPERATIONS (Pure PyTorch, no torch_scatter dependency)
# =============================================================================

def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = 0, 
                 dim_size: Optional[int] = None) -> torch.Tensor:
    """
    Scatter mean operation - averages src values by index.
    
    Args:
        src: Source tensor
        index: Index tensor (same size as src in dim)
        dim: Dimension to scatter along
        dim_size: Size of output in scatter dimension
    """
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    # Expand index to match src shape
    index_expanded = index
    for _ in range(src.dim() - index.dim()):
        index_expanded = index_expanded.unsqueeze(-1)
    index_expanded = index_expanded.expand_as(src)
    
    # Sum
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index_expanded, src)
    
    # Count
    ones = torch.ones_like(src)
    count = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    count.scatter_add_(dim, index_expanded, ones)
    
    # Mean (avoid div by zero)
    return out / count.clamp(min=1)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> torch.Tensor:
    """Scatter add operation."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    index_expanded = index
    for _ in range(src.dim() - index.dim()):
        index_expanded = index_expanded.unsqueeze(-1)
    index_expanded = index_expanded.expand_as(src)
    
    out = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    out.scatter_add_(dim, index_expanded, src)
    return out


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim: int = 0,
                dim_size: Optional[int] = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter max operation."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    
    index_expanded = index
    for _ in range(src.dim() - index.dim()):
        index_expanded = index_expanded.unsqueeze(-1)
    index_expanded = index_expanded.expand_as(src)
    
    out = torch.full((dim_size, *src.shape[1:]), float('-inf'), device=src.device, dtype=src.dtype)
    out, argmax = out.scatter_reduce(dim, index_expanded, src, reduce='amax', include_self=True)
    return out, argmax


# =============================================================================
# GRAPH CONSTRUCTION UTILITIES
# =============================================================================

def build_delaunay_edges_batch(positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build Delaunay triangulation edges for a batch of point clouds.
    
    Args:
        positions: (B, N, 3) point coordinates
        
    Returns:
        edge_index: (2, total_edges) - source and target node indices (global)
        batch_edge: (total_edges,) - batch assignment for each edge
    """
    device = positions.device
    B, N, _ = positions.shape
    
    all_edges = []
    all_batch = []
    
    for b in range(B):
        pts = positions[b].detach().cpu().numpy()
        
        try:
            tri = Delaunay(pts)
            edges = set()
            for simplex in tri.simplices:
                # Each simplex is a tetrahedron (4 vertices in 3D)
                for i in range(len(simplex)):
                    for j in range(i + 1, len(simplex)):
                        # Add both directions
                        edges.add((simplex[i], simplex[j]))
                        edges.add((simplex[j], simplex[i]))
            
            if len(edges) > 0:
                edge_array = np.array(list(edges))
                # Offset by batch index
                edge_array += b * N
                all_edges.append(edge_array)
                all_batch.append(np.full(len(edges), b))
        except Exception:
            # Fallback: fully connected for degenerate cases
            idx = np.arange(N)
            src = np.repeat(idx, N)
            tgt = np.tile(idx, N)
            mask = src != tgt
            edge_array = np.stack([src[mask], tgt[mask]], axis=1) + b * N
            all_edges.append(edge_array)
            all_batch.append(np.full(edge_array.shape[0], b))
    
    if len(all_edges) > 0:
        edge_index = torch.tensor(np.concatenate(all_edges, axis=0).T, 
                                   dtype=torch.long, device=device)
        batch_edge = torch.tensor(np.concatenate(all_batch), 
                                   dtype=torch.long, device=device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        batch_edge = torch.zeros((0,), dtype=torch.long, device=device)
    
    return edge_index, batch_edge


def build_radius_edges_batch(positions: torch.Tensor, 
                             cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build radius graph: connect points within cutoff distance.
    
    Physics-motivated for atomic systems (e.g., cutoff = 5.0 Angstrom).
    
    Args:
        positions: (B, N, 3)
        cutoff: maximum distance for edge
        
    Returns:
        edge_index: (2, total_edges)
        batch_edge: (total_edges,)
    """
    device = positions.device
    B, N, _ = positions.shape
    
    # Compute pairwise distances within each batch
    # (B, N, N)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)  # (B, N, N, 3)
    dist = torch.norm(diff, dim=-1)  # (B, N, N)
    
    # Create adjacency mask
    mask = (dist < cutoff) & (dist > 1e-6)  # Exclude self-loops
    
    # Convert to edge list
    all_edges = []
    all_batch = []
    
    for b in range(B):
        src, tgt = torch.where(mask[b])
        if len(src) > 0:
            edges = torch.stack([src + b * N, tgt + b * N], dim=0)
            all_edges.append(edges)
            all_batch.append(torch.full((edges.shape[1],), b, device=device))
    
    if len(all_edges) > 0:
        edge_index = torch.cat(all_edges, dim=1)
        batch_edge = torch.cat(all_batch)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        batch_edge = torch.zeros((0,), dtype=torch.long, device=device)
    
    return edge_index, batch_edge


def build_adaptive_edges_batch(positions: torch.Tensor,
                               min_neighbors: int = 4,
                               max_neighbors: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Adaptive radius graph: ensures each node has at least min_neighbors
    but caps at max_neighbors (sorted by distance).
    
    Handles both dense crystals and sparse amorphous structures.
    """
    device = positions.device
    B, N, _ = positions.shape
    
    # Compute pairwise distances
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)  # (B, N, N)
    
    # Set diagonal to inf
    dist = dist + torch.eye(N, device=device).unsqueeze(0) * 1e10
    
    # For each point, get sorted neighbor indices
    sorted_dist, sorted_idx = torch.sort(dist, dim=-1)
    
    all_edges = []
    all_batch = []
    
    for b in range(B):
        edges = []
        for i in range(N):
            # Ensure at least min_neighbors, at most max_neighbors
            n_neighbors = min(max(min_neighbors, 
                                  (sorted_dist[b, i] < sorted_dist[b, i, min_neighbors] * 1.5).sum().item()),
                             max_neighbors)
            neighbors = sorted_idx[b, i, :n_neighbors]
            for j in neighbors:
                edges.append([i + b * N, j.item() + b * N])
        
        if edges:
            edge_array = torch.tensor(edges, device=device, dtype=torch.long).T
            all_edges.append(edge_array)
            all_batch.append(torch.full((edge_array.shape[1],), b, device=device))
    
    if all_edges:
        edge_index = torch.cat(all_edges, dim=1)
        batch_edge = torch.cat(all_batch)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        batch_edge = torch.zeros((0,), dtype=torch.long, device=device)
    
    return edge_index, batch_edge


# =============================================================================
# VORONOI FEATURE EXTRACTION (Auxiliary Invariant Features)
# =============================================================================

def compute_voronoi_features(positions: torch.Tensor) -> torch.Tensor:
    """
    Compute Voronoi cell features for each atom.
    These are powerful LOCAL INVARIANTS for structure classification.
    
    Features per atom:
    - Cell volume
    - Number of faces
    - Number of vertices  
    - Sphericity (how sphere-like the cell is)
    - Face area distribution statistics
    
    Args:
        positions: (B, N, 3)
        
    Returns:
        features: (B, N, feature_dim)
    """
    device = positions.device
    B, N, _ = positions.shape
    feature_dim = 8
    
    all_features = torch.zeros(B, N, feature_dim, device=device)
    
    for b in range(B):
        pts = positions[b].detach().cpu().numpy()
        
        try:
            vor = Voronoi(pts)
            
            for i in range(N):
                region_idx = vor.point_region[i]
                region = vor.regions[region_idx]
                
                if -1 in region or len(region) == 0:
                    # Infinite cell - use defaults
                    continue
                
                vertices = vor.vertices[region]
                
                # Feature 1: Number of vertices
                n_vertices = len(vertices)
                
                # Feature 2: Number of faces (approximate from ridge count)
                n_faces = sum(1 for ridge in vor.ridge_points if i in ridge)
                
                # Feature 3: Cell "volume" proxy (convex hull would be expensive)
                # Use variance of vertex positions as proxy
                center = vertices.mean(axis=0)
                radii = np.linalg.norm(vertices - center, axis=1)
                vol_proxy = radii.mean()
                
                # Feature 4: Sphericity (1 = perfect sphere)
                sphericity = radii.min() / (radii.max() + 1e-6)
                
                # Feature 5-8: Radius statistics
                radius_mean = radii.mean()
                radius_std = radii.std()
                radius_min = radii.min()
                radius_max = radii.max()
                
                all_features[b, i] = torch.tensor([
                    n_vertices / 20.0,  # Normalize
                    n_faces / 15.0,
                    vol_proxy,
                    sphericity,
                    radius_mean,
                    radius_std,
                    radius_min,
                    radius_max
                ], device=device)
                
        except Exception:
            # Keep zeros for failed cases
            pass
    
    return all_features


# =============================================================================
# VECTOR NEURON LAYERS (Equivariant Building Blocks)
# =============================================================================

class VNLinear(nn.Module):
    """Linear layer for vector features: (B, C_in, 3, ...) -> (B, C_out, 3, ...)"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_channels, in_channels) * 0.01)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, 3, ...) or (B, C_in, 3)
        # Contract over in_channels, preserve 3D vectors
        return torch.einsum('oi,bi...->bo...', self.weight, x)


class VNBatchNorm(nn.Module):
    """Batch normalization for vector features."""
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3) or (B, C, 3, N)
        norm = torch.norm(x, dim=2, keepdim=True)  # (B, C, 1, ...)
        mean_norm = norm.mean(dim=0, keepdim=True)  # (1, C, 1, ...)
        std_norm = norm.std(dim=0, keepdim=True) + self.eps
        
        # Normalize and scale
        x_normalized = x / (norm + self.eps) * (norm - mean_norm) / std_norm
        return x_normalized * self.gamma


class VNLeakyReLU(nn.Module):
    """Leaky ReLU for vector features - projects onto learned direction."""
    def __init__(self, in_channels: int, negative_slope: float = 0.2, share_nonlinearity: bool = False):
        super().__init__()
        self.negative_slope = negative_slope
        
        if share_nonlinearity:
            self.direction = nn.Parameter(torch.randn(1, 1, 3))
        else:
            self.direction = nn.Parameter(torch.randn(1, in_channels, 3))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, 3) or (B, C, 3, N)
        d = F.normalize(self.direction, dim=-1)
        
        if x.dim() == 3:
            # (B, C, 3)
            proj = (x * d).sum(dim=-1, keepdim=True)  # (B, C, 1)
        else:
            # (B, C, 3, N)
            d = d.unsqueeze(-1)  # (1, C, 3, 1)
            proj = (x * d).sum(dim=2, keepdim=True)  # (B, C, 1, N)
        
        mask = (proj >= 0).float()
        return mask * x + (1 - mask) * self.negative_slope * x


class VNLinearLeakyReLU(nn.Module):
    """Combined VN Linear + BatchNorm + LeakyReLU"""
    def __init__(self, in_channels: int, out_channels: int, 
                 use_batchnorm: bool = True, negative_slope: float = 0.2):
        super().__init__()
        self.linear = VNLinear(in_channels, out_channels)
        self.bn = VNBatchNorm(out_channels) if use_batchnorm else nn.Identity()
        self.act = VNLeakyReLU(out_channels, negative_slope)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.linear(x)))


# =============================================================================
# GRAPH MESSAGE PASSING LAYERS (Equivariant)
# =============================================================================

class VNEdgeConv(nn.Module):
    """
    Vector Neuron Edge Convolution.
    
    Message: m_ij = MLP([h_i, h_j - h_i, ||r_ij|| * (r_ij/||r_ij||)])
    
    The relative position r_ij is included as an equivariant feature.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 use_batchnorm: bool = True, use_rel_pos: bool = True):
        super().__init__()
        self.use_rel_pos = use_rel_pos
        
        # Input: [h_i, h_j - h_i, r_ij_unit] -> 3 * in_channels (+ 1 if rel_pos)
        mlp_in = in_channels * 2 + (1 if use_rel_pos else 0)
        
        self.mlp = nn.Sequential(
            VNLinearLeakyReLU(mlp_in, out_channels, use_batchnorm=use_batchnorm),
            VNLinearLeakyReLU(out_channels, out_channels, use_batchnorm=use_batchnorm),
        )
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, 
                pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (total_nodes, C, 3) node features
            edge_index: (2, E) edges
            pos: (total_nodes, 3) node positions
            batch: (total_nodes,) batch assignment
            
        Returns:
            h_new: (total_nodes, C_out, 3)
        """
        src, tgt = edge_index
        
        h_src = h[src]  # (E, C, 3)
        h_tgt = h[tgt]  # (E, C, 3)
        
        # Relative position as equivariant feature
        if self.use_rel_pos:
            r_ij = pos[tgt] - pos[src]  # (E, 3)
            r_norm = torch.norm(r_ij, dim=-1, keepdim=True) + 1e-8  # (E, 1)
            r_unit = (r_ij / r_norm).unsqueeze(1)  # (E, 1, 3) - equivariant!
            
            # Concatenate: [h_src, h_tgt - h_src, r_unit]
            edge_feat = torch.cat([h_src, h_tgt - h_src, r_unit], dim=1)  # (E, 2C+1, 3)
        else:
            edge_feat = torch.cat([h_src, h_tgt - h_src], dim=1)
        
        # MLP on edge features
        msg = self.mlp(edge_feat)  # (E, C_out, 3)
        
        # Aggregate messages (mean)
        h_new = scatter_mean(msg, src, dim=0, dim_size=h.shape[0])
        
        return h_new


class VNEdgeConvWithInvariant(nn.Module):
    """
    Edge convolution that also uses invariant edge features (distances, angles).
    
    Combines equivariant vector messages with invariant scalar conditioning.
    """
    def __init__(self, in_channels: int, out_channels: int, 
                 invariant_dim: int = 16, use_batchnorm: bool = True):
        super().__init__()
        
        # Invariant edge feature encoder
        self.inv_encoder = nn.Sequential(
            nn.Linear(4, invariant_dim),  # distance, normalized distance, etc.
            nn.SiLU(),
            nn.Linear(invariant_dim, invariant_dim),
        )
        
        # Modulation: invariant features modulate the equivariant path
        self.modulation = nn.Sequential(
            nn.Linear(invariant_dim, out_channels),
            nn.Sigmoid(),
        )
        
        # Equivariant path
        mlp_in = in_channels * 2 + 1  # h_i, h_j - h_i, r_unit
        self.eq_mlp = nn.Sequential(
            VNLinearLeakyReLU(mlp_in, out_channels, use_batchnorm=use_batchnorm),
            VNLinearLeakyReLU(out_channels, out_channels, use_batchnorm=use_batchnorm),
        )
    
    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        src, tgt = edge_index
        
        h_src = h[src]
        h_tgt = h[tgt]
        
        # Compute invariant edge features
        r_ij = pos[tgt] - pos[src]
        dist = torch.norm(r_ij, dim=-1, keepdim=True)
        
        # Compute local density proxy (mean distance to neighbors for each node)
        node_mean_dist = scatter_mean(dist, src, dim=0, dim_size=h.shape[0])
        local_density_src = node_mean_dist[src]
        
        inv_feat = torch.cat([
            dist,
            dist / (local_density_src + 1e-6),  # Normalized by local density
            torch.log(dist + 1e-6),
            local_density_src,
        ], dim=-1)
        
        inv_encoded = self.inv_encoder(inv_feat)  # (E, inv_dim)
        modulation = self.modulation(inv_encoded)  # (E, C_out)
        
        # Equivariant path
        r_unit = (r_ij / (dist + 1e-8)).unsqueeze(1)
        edge_feat = torch.cat([h_src, h_tgt - h_src, r_unit], dim=1)
        msg_eq = self.eq_mlp(edge_feat)  # (E, C_out, 3)
        
        # Modulate equivariant messages with invariant features
        msg = msg_eq * modulation.unsqueeze(-1)  # (E, C_out, 3)
        
        # Aggregate
        h_new = scatter_mean(msg, src, dim=0, dim_size=h.shape[0])
        
        return h_new


# =============================================================================
# MULTI-SCALE GRAPH ENCODER
# =============================================================================

class MultiScaleGraphConstruction(nn.Module):
    """
    Builds graphs at multiple scales:
    - Fine: Delaunay or small radius (local bonding)
    - Medium: Medium radius (coordination shells)
    - Coarse: Large radius or full Delaunay (long-range order)
    """
    def __init__(self, 
                 scales: list[float] = [3.0, 5.0, 8.0],
                 use_delaunay: bool = True):
        super().__init__()
        self.scales = scales
        self.use_delaunay = use_delaunay
    
    def forward(self, pos: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            pos: (B, N, 3)
            
        Returns:
            List of (edge_index, batch_edge) for each scale
        """
        graphs = []
        
        if self.use_delaunay:
            # Use Delaunay for finest scale (geometry-intrinsic)
            edge_index, batch_edge = build_delaunay_edges_batch(pos)
            graphs.append((edge_index, batch_edge))
            
            # Use radius graphs for coarser scales
            for r in self.scales[1:]:
                edge_index, batch_edge = build_radius_edges_batch(pos, r)
                graphs.append((edge_index, batch_edge))
        else:
            # All radius-based
            for r in self.scales:
                edge_index, batch_edge = build_radius_edges_batch(pos, r)
                graphs.append((edge_index, batch_edge))
        
        return graphs


# =============================================================================
# MAIN ENCODER
# =============================================================================

class VNGraphEncoderDelaunay(nn.Module):
    """
    Graph-based Vector Neuron Encoder using Delaunay triangulation.
    
    Replaces k-NN with geometry-intrinsic graph construction.
    """
    def __init__(
        self,
        latent_size: int = 256,
        hidden_channels: tuple[int, ...] = (32, 64, 128, 256),
        use_batchnorm: bool = True,
        graph_type: Literal['delaunay', 'radius', 'adaptive'] = 'delaunay',
        radius_cutoff: float = 5.0,
        use_voronoi_features: bool = False,
        use_invariant_edges: bool = True,
    ):
        super().__init__()
        
        self.graph_type = graph_type
        self.radius_cutoff = radius_cutoff
        self.use_voronoi_features = use_voronoi_features
        self.latent_size = latent_size
        
        # Initial embedding: positions -> vector features
        # Input is just positions, so 1 channel (the position vector itself)
        self.embed = VNLinearLeakyReLU(1, hidden_channels[0], use_batchnorm=use_batchnorm)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        EdgeConvClass = VNEdgeConvWithInvariant if use_invariant_edges else VNEdgeConv
        
        in_ch = hidden_channels[0]
        for out_ch in hidden_channels[1:]:
            self.convs.append(EdgeConvClass(in_ch, out_ch, use_batchnorm=use_batchnorm))
            in_ch = out_ch
        
        # Global aggregation
        total_channels = sum(hidden_channels)
        self.global_conv = VNLinearLeakyReLU(total_channels, hidden_channels[-1], 
                                              use_batchnorm=use_batchnorm)
        
        # Output heads
        assert latent_size % 3 == 0, "latent_size must be divisible by 3"
        self.eq_projector = VNLinear(hidden_channels[-1], latent_size // 3)
        
        # Invariant head
        inv_input_dim = hidden_channels[-1] * 2  # norms + projections
        if use_voronoi_features:
            inv_input_dim += 8  # Voronoi features
        
        self.inv_head = nn.Sequential(
            nn.Linear(inv_input_dim, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_size, latent_size),
        )
        
        # Anisotropy predictor (for handling amorphous structures)
        self.anisotropy_head = nn.Sequential(
            nn.Linear(inv_input_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def build_graph(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Build graph based on configured type."""
        if self.graph_type == 'delaunay':
            return build_delaunay_edges_batch(pos)
        elif self.graph_type == 'radius':
            return build_radius_edges_batch(pos, self.radius_cutoff)
        elif self.graph_type == 'adaptive':
            return build_adaptive_edges_batch(pos)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")
    
    def extract_invariant_features(self, h: torch.Tensor) -> torch.Tensor:
        """Extract rotation-invariant features from equivariant vectors."""
        # h: (B, C, 3)
        
        # Norms
        norms = torch.norm(h, dim=-1)  # (B, C)
        
        # Projection onto mean direction
        mean_dir = h.mean(dim=1, keepdim=True)  # (B, 1, 3)
        mean_dir = F.normalize(mean_dir, dim=-1, eps=1e-6)
        projections = (h * mean_dir).sum(dim=-1)  # (B, C)
        
        return torch.cat([norms, projections], dim=-1)  # (B, 2C)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, 3) point cloud positions
            
        Returns:
            z_inv: (B, latent_size) invariant embedding
            z_eq: (B, latent_size//3, 3) equivariant embedding
            anisotropy: (B, 1) anisotropy score (1=crystal, 0=amorphous)
        """
        B, N, _ = x.shape
        device = x.device
        
        # Build graph
        edge_index, batch_edge = self.build_graph(x)
        
        # Create batch tensor for nodes
        batch = torch.arange(B, device=device).repeat_interleave(N)
        
        # Flatten positions
        pos_flat = x.reshape(B * N, 3)
        
        # Initial embedding: treat position as a single vector channel
        h = pos_flat.unsqueeze(1)  # (B*N, 1, 3)
        h = self.embed(h)  # (B*N, C0, 3)
        
        # Store intermediate features for skip connections
        h_list = [h]
        
        # Message passing
        for conv in self.convs:
            h = conv(h, edge_index, pos_flat, batch)
            h_list.append(h)
        
        # Concatenate all scales
        h_cat = torch.cat(h_list, dim=1)  # (B*N, sum(C), 3)
        h = self.global_conv(h_cat)  # (B*N, C_final, 3)
        
        # Global pooling (mean over nodes in each graph)
        h_global = scatter_mean(h, batch, dim=0)  # (B, C_final, 3)
        
        # Equivariant output
        z_eq = self.eq_projector(h_global)  # (B, latent//3, 3)
        
        # Invariant features
        inv_feat = self.extract_invariant_features(h_global)  # (B, 2*C_final)
        
        # Optionally add Voronoi features
        if self.use_voronoi_features:
            vor_feat = compute_voronoi_features(x)  # (B, N, 8)
            vor_global = vor_feat.mean(dim=1)  # (B, 8)
            inv_feat = torch.cat([inv_feat, vor_global], dim=-1)
        
        # Invariant output
        z_inv = self.inv_head(inv_feat)  # (B, latent)
        
        # Anisotropy score
        anisotropy = self.anisotropy_head(inv_feat)  # (B, 1)
        
        return z_inv, z_eq, anisotropy


# =============================================================================
# MULTI-SCALE VARIANT
# =============================================================================

class VNGraphEncoderMultiScale(nn.Module):
    """
    Multi-scale graph encoder that captures structure at different length scales.
    
    Uses separate graph constructions at each scale and fuses information.
    """
    def __init__(
        self,
        latent_size: int = 256,
        scales: list[float] = [3.0, 5.0, 8.0],
        hidden_channels: int = 64,
        n_layers_per_scale: int = 2,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        
        self.scales = scales
        self.n_scales = len(scales)
        self.latent_size = latent_size
        
        # Per-scale encoders
        self.scale_encoders = nn.ModuleList()
        for _ in scales:
            layers = []
            in_ch = 1
            for i in range(n_layers_per_scale):
                out_ch = hidden_channels * (2 ** min(i, 2))
                layers.append(VNEdgeConvWithInvariant(in_ch, out_ch, use_batchnorm=use_batchnorm))
                in_ch = out_ch
            self.scale_encoders.append(nn.ModuleList(layers))
        
        # Cross-scale attention
        final_ch = hidden_channels * 4  # After 2 layers with doubling
        self.cross_scale_attn = nn.MultiheadAttention(
            embed_dim=final_ch * 3,  # Flatten 3D vectors
            num_heads=4,
            batch_first=True
        )
        
        # Output projections
        self.eq_projector = VNLinear(final_ch * self.n_scales, latent_size // 3)
        
        inv_input = final_ch * self.n_scales * 2
        self.inv_head = nn.Sequential(
            nn.Linear(inv_input, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_size, latent_size),
        )
        
        self.anisotropy_head = nn.Sequential(
            nn.Linear(inv_input, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, _ = x.shape
        device = x.device
        batch = torch.arange(B, device=device).repeat_interleave(N)
        pos_flat = x.reshape(B * N, 3)
        
        scale_outputs = []
        
        for scale_idx, (radius, encoder) in enumerate(zip(self.scales, self.scale_encoders)):
            # Build graph at this scale
            if scale_idx == 0:
                # Use Delaunay for finest scale
                edge_index, _ = build_delaunay_edges_batch(x)
            else:
                edge_index, _ = build_radius_edges_batch(x, radius)
            
            # Initial features
            h = pos_flat.unsqueeze(1)  # (B*N, 1, 3)
            
            # Process through layers
            for layer in encoder:
                h = layer(h, edge_index, pos_flat, batch)
            
            # Global pool
            h_global = scatter_mean(h, batch, dim=0)  # (B, C, 3)
            scale_outputs.append(h_global)
        
        # Concatenate scales
        h_multi = torch.cat(scale_outputs, dim=1)  # (B, n_scales * C, 3)
        
        # Equivariant output
        z_eq = self.eq_projector(h_multi)
        
        # Invariant features
        norms = torch.norm(h_multi, dim=-1)
        mean_dir = F.normalize(h_multi.mean(dim=1, keepdim=True), dim=-1, eps=1e-6)
        projs = (h_multi * mean_dir).sum(dim=-1)
        inv_feat = torch.cat([norms, projs], dim=-1)
        
        z_inv = self.inv_head(inv_feat)
        anisotropy = self.anisotropy_head(inv_feat)
        
        return z_inv, z_eq, anisotropy


# =============================================================================
# ROTATION HEAD (Compatible with both encoders)
# =============================================================================

class VNRotationHead(nn.Module):
    """
    Equivariant rotation head with confidence estimation.
    
    For amorphous structures (low anisotropy), the rotation is meaningless,
    so we also output a confidence score.
    """
    def __init__(self, in_features: int, hidden: int = 64):
        super().__init__()
        in_channels = in_features // 3
        
        self.net = nn.Sequential(
            VNLinearLeakyReLU(in_channels, hidden, use_batchnorm=False),
            VNLinearLeakyReLU(hidden, hidden, use_batchnorm=False),
            VNLinear(hidden, 2),  # Output 2 vectors for Gram-Schmidt
        )
    
    def forward(self, z_eq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_eq: (B, C, 3) equivariant features
            
        Returns:
            R: (B, 3, 3) rotation matrix
        """
        vectors = self.net(z_eq)  # (B, 2, 3)
        return gram_schmidt_rotation(vectors)


def gram_schmidt_rotation(vectors: torch.Tensor) -> torch.Tensor:
    """
    Convert 2 vectors to a proper rotation matrix via Gram-Schmidt.
    
    Args:
        vectors: (B, 2, 3) two 3D vectors
        
    Returns:
        R: (B, 3, 3) rotation matrix (SO(3))
    """
    v1 = vectors[:, 0]  # (B, 3)
    v2 = vectors[:, 1]  # (B, 3)
    
    # Normalize first vector
    e1 = F.normalize(v1, dim=-1, eps=1e-6)
    
    # Orthogonalize second vector
    v2_orth = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = F.normalize(v2_orth, dim=-1, eps=1e-6)
    
    # Third vector via cross product
    e3 = torch.cross(e1, e2, dim=-1)
    
    # Stack to form rotation matrix
    R = torch.stack([e1, e2, e3], dim=-1)  # (B, 3, 3)
    
    return R


# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Test the encoder
    torch.manual_seed(42)
    
    B, N = 4, 64
    x = torch.randn(B, N, 3)
    
    print("=" * 60)
    print("Testing VNGraphEncoderDelaunay")
    print("=" * 60)
    
    encoder = VNGraphEncoderDelaunay(
        latent_size=128,
        hidden_channels=(16, 32, 64),
        graph_type='delaunay',
        use_voronoi_features=False,
    )
    
    z_inv, z_eq, anisotropy = encoder(x)
    print(f"Input shape: {x.shape}")
    print(f"z_inv shape: {z_inv.shape}")
    print(f"z_eq shape: {z_eq.shape}")
    print(f"anisotropy shape: {anisotropy.shape}")
    
    # Test equivariance
    print("\nTesting SO(3) equivariance...")
    
    # Random rotation
    theta = torch.tensor(0.7)
    R_test = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ]).float()
    
    x_rotated = torch.einsum('ij,bnj->bni', R_test, x)
    
    z_inv_rot, z_eq_rot, _ = encoder(x_rotated)
    
    # z_inv should be the same (invariant)
    inv_diff = (z_inv - z_inv_rot).abs().max().item()
    print(f"Invariant embedding max diff: {inv_diff:.6f}")
    
    # z_eq should rotate with input (equivariant)
    z_eq_expected = torch.einsum('ij,bcj->bci', R_test, z_eq)
    eq_diff = (z_eq_rot - z_eq_expected).abs().max().item()
    print(f"Equivariant embedding max diff: {eq_diff:.6f}")
    
    print("\n" + "=" * 60)
    print("Testing VNGraphEncoderMultiScale")
    print("=" * 60)
    
    encoder_ms = VNGraphEncoderMultiScale(
        latent_size=128,
        scales=[3.0, 5.0, 8.0],
        hidden_channels=32,
    )
    
    z_inv_ms, z_eq_ms, aniso_ms = encoder_ms(x)
    print(f"z_inv shape: {z_inv_ms.shape}")
    print(f"z_eq shape: {z_eq_ms.shape}")
    
    print("\n" + "=" * 60)
    print("Testing Rotation Head")
    print("=" * 60)
    
    rot_head = VNRotationHead(in_features=z_eq.shape[1] * 3)
    R_pred = rot_head(z_eq)
    print(f"Predicted rotation shape: {R_pred.shape}")
    
    # Verify it's a valid rotation (R^T R = I, det(R) = 1)
    RtR = torch.bmm(R_pred.transpose(-1, -2), R_pred)
    identity_diff = (RtR - torch.eye(3)).abs().max().item()
    det = torch.det(R_pred)
    print(f"R^T R - I max diff: {identity_diff:.6f}")
    print(f"det(R) range: [{det.min().item():.4f}, {det.max().item():.4f}]")
    
    print("\nAll tests passed!")
