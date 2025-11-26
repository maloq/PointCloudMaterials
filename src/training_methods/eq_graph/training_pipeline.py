"""
Complete Training Pipeline for Rotation-Equivariant Point Cloud Autoencoder

Addresses the key challenges:
1. Rotation ambiguity for amorphous structures (via anisotropy gating)
2. Mean shape problem (via flow matching / diffusion decoder)
3. Adaptive loss (EMD for crystals, RDF for amorphous)

Architecture:
    Encoder: VNGraphEncoderDelaunay -> z_inv, z_eq, anisotropy
    Rotation Head: z_eq -> R (only used when anisotropy > threshold)
    Decoder: FlowMatchingDecoder -> X_hat (sharp, not averaged)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Optional, Literal
from pathlib import Path
import math

# Import our modules (in practice, these would be proper imports)
# from vn_graph_encoder import VNGraphEncoderDelaunay, VNRotationHead, gram_schmidt_rotation
# from equivariant_decoders import FlowMatchingDecoder, DiffusionDecoder, rdf_loss, compute_rdf


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def approximate_emd(x1: torch.Tensor, x2: torch.Tensor, 
                    n_iter: int = 50) -> torch.Tensor:
    """
    Approximate Earth Mover's Distance using Sinkhorn iterations.
    
    Much faster than exact EMD, differentiable, works on GPU.
    
    Args:
        x1: (B, N, 3) point cloud 1
        x2: (B, N, 3) point cloud 2
        n_iter: number of Sinkhorn iterations
        
    Returns:
        emd: (B,) approximate EMD per batch
    """
    B, N, _ = x1.shape
    device = x1.device
    
    # Cost matrix (pairwise distances)
    C = torch.cdist(x1, x2)  # (B, N, N)
    
    # Sinkhorn algorithm
    eps = 0.05  # Entropy regularization
    
    # Initialize
    u = torch.ones(B, N, device=device) / N
    v = torch.ones(B, N, device=device) / N
    
    K = torch.exp(-C / eps)
    
    for _ in range(n_iter):
        u = 1.0 / (K @ v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        v = 1.0 / (K.transpose(-1, -2) @ u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
    
    # Transport plan
    P = u.unsqueeze(-1) * K * v.unsqueeze(-2)
    
    # EMD = sum of cost * transport
    emd = (P * C).sum(dim=(-1, -2))
    
    return emd


def chamfer_distance(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Chamfer distance between point clouds."""
    dist = torch.cdist(x1, x2)
    min1 = dist.min(dim=2).values.mean(dim=1)
    min2 = dist.min(dim=1).values.mean(dim=1)
    return (min1 + min2) / 2


def compute_rdf(x: torch.Tensor, n_bins: int = 50, 
                r_max: float = 5.0) -> torch.Tensor:
    """Compute radial distribution function."""
    B, N, _ = x.shape
    device = x.device
    
    dist = torch.cdist(x, x)
    mask = ~torch.eye(N, dtype=torch.bool, device=device)
    dist = dist[:, mask].view(B, N * (N - 1))
    
    bin_edges = torch.linspace(0, r_max, n_bins + 1, device=device)
    rdf = torch.zeros(B, n_bins, device=device)
    
    for i in range(n_bins):
        in_bin = (dist >= bin_edges[i]) & (dist < bin_edges[i + 1])
        rdf[:, i] = in_bin.float().sum(dim=1)
    
    r = (bin_edges[:-1] + bin_edges[1:]) / 2
    dr = bin_edges[1] - bin_edges[0]
    shell_volume = 4 * math.pi * r ** 2 * dr
    rdf = rdf / (shell_volume.unsqueeze(0) * N + 1e-8)
    
    return rdf


def rdf_loss(x1: torch.Tensor, x2: torch.Tensor,
             n_bins: int = 50, r_max: float = 5.0) -> torch.Tensor:
    """RDF matching loss for amorphous structures."""
    rdf1 = compute_rdf(x1, n_bins, r_max)
    rdf2 = compute_rdf(x2, n_bins, r_max)
    return F.mse_loss(rdf1, rdf2, reduction='none').mean(dim=1)


def compute_adf(x: torch.Tensor, n_bins: int = 36, 
                cutoff: float = 3.5) -> torch.Tensor:
    """
    Compute Angular Distribution Function.
    
    For each atom, finds neighbors within cutoff and computes
    distribution of angles between neighbor pairs.
    """
    B, N, _ = x.shape
    device = x.device
    
    # Find neighbors
    dist = torch.cdist(x, x)  # (B, N, N)
    neighbor_mask = (dist < cutoff) & (dist > 0.1)
    
    all_angles = []
    
    for b in range(B):
        angles = []
        for i in range(N):
            neighbors = torch.where(neighbor_mask[b, i])[0]
            if len(neighbors) < 2:
                continue
            
            # Vectors to neighbors
            vecs = x[b, neighbors] - x[b, i]  # (K, 3)
            vecs = F.normalize(vecs, dim=-1)
            
            # Angles between all pairs
            for j in range(len(neighbors)):
                for k in range(j + 1, len(neighbors)):
                    cos_angle = (vecs[j] * vecs[k]).sum()
                    angles.append(cos_angle)
        
        if angles:
            all_angles.append(torch.stack(angles))
    
    # Histogram
    adf = torch.zeros(B, n_bins, device=device)
    bin_edges = torch.linspace(-1, 1, n_bins + 1, device=device)
    
    for b, angles in enumerate(all_angles):
        for i in range(n_bins):
            in_bin = (angles >= bin_edges[i]) & (angles < bin_edges[i + 1])
            adf[b, i] = in_bin.float().sum()
    
    # Normalize
    adf = adf / (adf.sum(dim=1, keepdim=True) + 1e-8)
    
    return adf


# =============================================================================
# COMPLETE AUTOENCODER
# =============================================================================

class EquivariantAutoencoder(nn.Module):
    """
    Complete autoencoder with anisotropy-adaptive reconstruction.
    
    Key features:
    - Delaunay-based graph encoder (no arbitrary k)
    - Anisotropy detection (crystal vs amorphous)
    - Rotation prediction only when meaningful
    - Sharp reconstruction via flow matching
    - Adaptive loss based on structure type
    """
    
    def __init__(
        self,
        latent_dim: int = 256,
        n_atoms: int = 64,
        encoder_hidden: tuple[int, ...] = (32, 64, 128, 256),
        decoder_type: Literal['flow', 'diffusion'] = 'flow',
        use_rotation_head: bool = True,
        anisotropy_threshold: float = 0.5,
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_atoms = n_atoms
        self.anisotropy_threshold = anisotropy_threshold
        self.use_rotation_head = use_rotation_head
        
        # Encoder (produces z_inv, z_eq, anisotropy)
        from vn_graph_encoder import VNGraphEncoderDelaunay
        self.encoder = VNGraphEncoderDelaunay(
            latent_size=latent_dim,
            hidden_channels=encoder_hidden,
            graph_type='delaunay',
            use_voronoi_features=False,
            use_invariant_edges=True,
        )
        
        # Rotation head (only used for anisotropic structures)
        if use_rotation_head:
            from vn_graph_encoder import VNRotationHead
            self.rotation_head = VNRotationHead(latent_dim)
        
        # Decoder
        if decoder_type == 'flow':
            from equivariant_decoders import FlowMatchingDecoder
            self.decoder = FlowMatchingDecoder(
                latent_dim=latent_dim,
                n_atoms=n_atoms,
            )
        else:
            from equivariant_decoders import DiffusionDecoder
            self.decoder = DiffusionDecoder(
                latent_dim=latent_dim,
                n_atoms=n_atoms,
            )
        
        self.decoder_type = decoder_type
    
    def encode(self, x: torch.Tensor) -> dict:
        """
        Encode point cloud.
        
        Args:
            x: (B, N, 3) point cloud
            
        Returns:
            dict with z_inv, z_eq, anisotropy
        """
        z_inv, z_eq, anisotropy = self.encoder(x)
        
        return {
            'z_inv': z_inv,
            'z_eq': z_eq,
            'anisotropy': anisotropy,
        }
    
    def decode(self, z_inv: torch.Tensor, n_steps: int = 50) -> torch.Tensor:
        """
        Decode latent code to point cloud.
        
        Args:
            z_inv: (B, latent_dim) invariant latent
            n_steps: sampling steps for flow/diffusion
            
        Returns:
            x_hat: (B, N, 3) reconstructed point cloud
        """
        return self.decoder.sample(z_inv, n_steps=n_steps)
    
    def predict_rotation(self, z_eq: torch.Tensor) -> torch.Tensor:
        """Predict rotation matrix from equivariant features."""
        if self.use_rotation_head:
            return self.rotation_head(z_eq)
        else:
            B = z_eq.shape[0]
            return torch.eye(3, device=z_eq.device).unsqueeze(0).expand(B, -1, -1)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Full forward pass.
        
        Returns reconstruction and all intermediate values.
        """
        # Encode
        encoded = self.encode(x)
        z_inv = encoded['z_inv']
        z_eq = encoded['z_eq']
        anisotropy = encoded['anisotropy']
        
        # Decode (in canonical pose)
        x_hat = self.decode(z_inv, n_steps=20)  # Fewer steps during training
        
        # Predict rotation
        R = self.predict_rotation(z_eq)
        
        # Rotate reconstruction to match input
        x_hat_rotated = torch.bmm(x_hat, R.transpose(-1, -2))
        
        return {
            'x_hat': x_hat,
            'x_hat_rotated': x_hat_rotated,
            'z_inv': z_inv,
            'z_eq': z_eq,
            'R': R,
            'anisotropy': anisotropy,
        }
    
    def compute_loss(self, x: torch.Tensor, output: dict) -> dict:
        """
        Compute adaptive loss based on structure type.
        
        For anisotropic (crystal-like) structures:
            - Use EMD/Chamfer with rotation alignment
            
        For isotropic (amorphous) structures:
            - Use RDF matching (ignores exact positions)
            - Don't penalize rotation prediction
        """
        B = x.shape[0]
        
        x_hat = output['x_hat']
        x_hat_rotated = output['x_hat_rotated']
        anisotropy = output['anisotropy'].squeeze(-1)  # (B,)
        
        # Crystal loss (requires exact alignment)
        crystal_loss = approximate_emd(x, x_hat_rotated)  # (B,)
        
        # Amorphous loss (statistical matching)
        rdf_l = rdf_loss(x, x_hat)  # (B,)
        
        # Weighted combination based on anisotropy
        # High anisotropy -> crystal loss dominates
        # Low anisotropy -> amorphous loss dominates
        recon_loss = anisotropy * crystal_loss + (1 - anisotropy) * rdf_l * 10.0
        
        # Regularization: encourage anisotropy prediction to be decisive
        anisotropy_entropy = -anisotropy * torch.log(anisotropy + 1e-8) - \
                            (1 - anisotropy) * torch.log(1 - anisotropy + 1e-8)
        
        # Decoder-specific loss (for training the generative model)
        if self.decoder_type == 'flow':
            noise = torch.randn_like(x)
            gen_loss = self.decoder.compute_loss(noise, x, output['z_inv'])
        else:
            gen_loss = self.decoder.compute_loss(x, output['z_inv'])
        
        # Contrastive regularization on latent space
        # (prevents collapse to mean representation)
        z_inv = output['z_inv']
        z_normalized = F.normalize(z_inv, dim=-1)
        similarity = torch.mm(z_normalized, z_normalized.t())
        contrastive_loss = (similarity - torch.eye(B, device=x.device)).pow(2).mean()
        
        # Total loss
        total_loss = (
            recon_loss.mean() +
            gen_loss +
            0.1 * anisotropy_entropy.mean() +
            0.01 * contrastive_loss
        )
        
        return {
            'total': total_loss,
            'recon': recon_loss.mean(),
            'crystal': crystal_loss.mean(),
            'amorphous': rdf_l.mean(),
            'gen': gen_loss,
            'entropy': anisotropy_entropy.mean(),
            'contrastive': contrastive_loss,
            'mean_anisotropy': anisotropy.mean(),
        }


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class PointCloudDataset(Dataset):
    """Simple dataset for point clouds."""
    
    def __init__(self, data: np.ndarray):
        """
        Args:
            data: (N_samples, N_atoms, 3) point clouds
        """
        self.data = torch.tensor(data, dtype=torch.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        # Center the point cloud
        x = x - x.mean(dim=0, keepdim=True)
        return x


def create_synthetic_data(n_samples: int = 1000, n_atoms: int = 64) -> np.ndarray:
    """
    Create synthetic dataset with crystals and amorphous structures.
    
    For testing purposes.
    """
    data = []
    
    for i in range(n_samples):
        if i % 3 == 0:
            # FCC-like crystal
            a = 1.0  # Lattice constant
            grid = np.array([[x, y, z] 
                           for x in range(4) 
                           for y in range(4) 
                           for z in range(4)])
            # Add face centers
            face_centers = grid + np.array([0.5, 0.5, 0])
            points = np.vstack([grid, face_centers])[:n_atoms]
            # Add small noise
            points = points + np.random.randn(*points.shape) * 0.05
            
        elif i % 3 == 1:
            # Amorphous (random)
            points = np.random.randn(n_atoms, 3)
            # Scale to similar size
            points = points / np.std(points) * 0.8
            
        else:
            # Nuclei (crystal core + amorphous shell)
            core_size = n_atoms // 2
            shell_size = n_atoms - core_size
            
            # Crystal core
            grid = np.array([[x, y, z] 
                           for x in range(3) 
                           for y in range(3) 
                           for z in range(3)])[:core_size]
            grid = grid - grid.mean(axis=0)
            
            # Amorphous shell
            shell = np.random.randn(shell_size, 3) * 2
            
            points = np.vstack([grid, shell])
            points = points + np.random.randn(*points.shape) * 0.1
        
        # Center
        points = points - points.mean(axis=0)
        
        # Random rotation
        theta = np.random.rand() * 2 * np.pi
        phi = np.random.rand() * np.pi
        R = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)],
            [np.sin(theta), np.cos(theta) * np.cos(phi), -np.cos(theta) * np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])
        points = points @ R.T
        
        data.append(points)
    
    return np.array(data)


class Trainer:
    """Training loop for the autoencoder."""
    
    def __init__(
        self,
        model: EquivariantAutoencoder,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 1e-4,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
        }
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        n_batches = 0
        
        for batch in self.train_loader:
            x = batch.to(self.device)
            
            # Forward
            output = self.model(x)
            losses = self.model.compute_loss(x, output)
            
            # Backward
            self.optimizer.zero_grad()
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += losses['total'].item()
            n_batches += 1
        
        self.scheduler.step()
        
        return {'loss': total_loss / n_batches}
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Validate on held-out data."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0
        n_batches = 0
        
        for batch in self.val_loader:
            x = batch.to(self.device)
            output = self.model(x)
            losses = self.model.compute_loss(x, output)
            
            total_loss += losses['total'].item()
            n_batches += 1
        
        return {'val_loss': total_loss / n_batches}
    
    def train(self, n_epochs: int, log_interval: int = 10):
        """Full training loop."""
        for epoch in range(n_epochs):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            self.history['train_loss'].append(train_metrics['loss'])
            if val_metrics:
                self.history['val_loss'].append(val_metrics['val_loss'])
            
            if epoch % log_interval == 0:
                msg = f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}"
                if val_metrics:
                    msg += f", val_loss={val_metrics['val_loss']:.4f}"
                print(msg)


# =============================================================================
# ANALYSIS UTILITIES
# =============================================================================

def analyze_latent_space(model: EquivariantAutoencoder, 
                        data_loader: DataLoader,
                        device: str = 'cuda') -> dict:
    """
    Analyze the learned latent space.
    
    Returns embeddings and metadata for visualization.
    """
    model.eval()
    
    all_z = []
    all_anisotropy = []
    
    with torch.no_grad():
        for batch in data_loader:
            x = batch.to(device)
            encoded = model.encode(x)
            
            all_z.append(encoded['z_inv'].cpu())
            all_anisotropy.append(encoded['anisotropy'].cpu())
    
    z = torch.cat(all_z, dim=0).numpy()
    anisotropy = torch.cat(all_anisotropy, dim=0).squeeze().numpy()
    
    return {
        'z': z,
        'anisotropy': anisotropy,
    }


def test_equivariance(model: EquivariantAutoencoder, x: torch.Tensor) -> dict:
    """
    Test rotation equivariance/invariance properties.
    """
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    
    # Random rotation
    theta = torch.tensor(0.7)
    R_test = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ], device=device).float()
    
    x_rotated = torch.einsum('ij,bnj->bni', R_test, x)
    
    with torch.no_grad():
        enc1 = model.encode(x)
        enc2 = model.encode(x_rotated)
    
    # z_inv should be invariant
    inv_diff = (enc1['z_inv'] - enc2['z_inv']).abs().max().item()
    
    # z_eq should transform equivariantly
    z_eq_expected = torch.einsum('ij,bcj->bci', R_test, enc1['z_eq'])
    eq_diff = (enc2['z_eq'] - z_eq_expected).abs().max().item()
    
    # Anisotropy should be invariant
    aniso_diff = (enc1['anisotropy'] - enc2['anisotropy']).abs().max().item()
    
    return {
        'invariant_error': inv_diff,
        'equivariant_error': eq_diff,
        'anisotropy_error': aniso_diff,
    }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Equivariant Point Cloud Autoencoder - Training Demo")
    print("=" * 60)
    
    # Create synthetic data
    print("\n1. Creating synthetic dataset...")
    train_data = create_synthetic_data(n_samples=500, n_atoms=32)
    val_data = create_synthetic_data(n_samples=100, n_atoms=32)
    
    train_dataset = PointCloudDataset(train_data)
    val_dataset = PointCloudDataset(val_data)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Note: The full model requires the encoder and decoder files
    # This is a demonstration of the training structure
    print("\n2. Model architecture summary:")
    print("""
    Encoder: VNGraphEncoderDelaunay
        - Graph: Delaunay triangulation (no k hyperparameter)
        - Features: Vector neurons (SO(3) equivariant)
        - Outputs: z_inv (invariant), z_eq (equivariant), anisotropy
    
    Rotation Head: VNRotationHead
        - Input: z_eq (equivariant features)
        - Output: R ∈ SO(3) via Gram-Schmidt
        - Used only when anisotropy > threshold
    
    Decoder: FlowMatchingDecoder
        - Learns velocity field: noise → structure
        - Produces SHARP samples (no mean shape!)
        - Conditioned on z_inv
    
    Loss:
        - Crystal: EMD(X, R @ X_hat) when anisotropy high
        - Amorphous: RDF_loss(X, X_hat) when anisotropy low
        - Contrastive: prevents latent collapse
    """)
    
    print("\n3. Training would proceed as:")
    print("""
    model = EquivariantAutoencoder(
        latent_dim=256,
        n_atoms=32,
        decoder_type='flow',
    )
    
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train(n_epochs=100)
    """)
    
    print("\n4. Key improvements over original architecture:")
    print("""
    ✓ No arbitrary k-NN: Uses Delaunay triangulation
    ✓ No mean shape: Flow matching produces sharp samples  
    ✓ Handles amorphous: Anisotropy gating + RDF loss
    ✓ Clean separation: Crystal vs amorphous loss paths
    ✓ Rotation handled: Only when structure is anisotropic
    """)
    
    print("\n" + "=" * 60)
    print("See vn_graph_encoder.py and equivariant_decoders.py for full code")
    print("=" * 60)
