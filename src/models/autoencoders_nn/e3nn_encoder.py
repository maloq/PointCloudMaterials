from e3nn import o3
from e3nn.o3 import Irreps
from src.utils.logging_config import setup_logging
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional

logger = setup_logging()


def build_model(cfg: DictConfig):
    model_type = cfg.type
    num_points = cfg.data.num_points
    latent_size = cfg.latent_size


    if model_type == "SO3DenseAutoEncoder":
        logger.print("SO3DenseAutoEncoder")
        # Get parameters from cfg, with defaults if not specified
        radial_dim = cfg.get('radial_dim', 8)
        l_max = cfg.get('l_max', 3)
        decoder_hidden_dim = cfg.get('decoder_hidden_dim', 512)
        return SO3DenseAutoEncoder(num_points, latent_size, 
                                    radial_dim=radial_dim, 
                                    l_max=l_max, 
                                    decoder_hidden_dim=decoder_hidden_dim)



# ------------------------------------------------------------------
#  Decoder : two‑step FoldingNet variant
# ------------------------------------------------------------------
class TwoStepFoldingDecoder(nn.Module):
    """Grid → coarse → refined folding decoder (FoldingNet‑style).

    Parameters
    ----------
    num_points : int
        Number of output points (must be a square number or will be padded).
    latent_size : int
        Dimension of the latent code *z* (assumed scalar, ``0e``).
    hidden_dim : int
        Hidden dimension for the MLP layers in the decoder.
    """

    def __init__(self, num_points: int, latent_size: int, hidden_dim: int = 512):
        super().__init__()
        self.num_points = num_points
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim

        # build a fixed 2‑D grid on [‑1,1]²
        side = int(math.ceil(num_points ** 0.5))
        xs = torch.linspace(-1.0, 1.0, side)
        ys = torch.linspace(-1.0, 1.0, side)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
        if grid.size(0) < num_points:  # pad if square was too small
            pad = grid[: num_points - grid.size(0)]
            grid = torch.cat((grid, pad), dim=0)
        self.register_buffer("grid", grid[:num_points])  # (P, 2)

        # MLP blocks for the two folds -------------------------------------------------
        def mlp(in_dim: int, current_hidden_dim: int, out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, current_hidden_dim), nn.ReLU(), nn.BatchNorm1d(current_hidden_dim),
                nn.Linear(current_hidden_dim, current_hidden_dim), nn.ReLU(), nn.BatchNorm1d(current_hidden_dim),
                nn.Linear(current_hidden_dim, out_dim),
            )

        self.fold1 = mlp(latent_size + 2, self.hidden_dim, 3)      # grid → coarse xyz
        self.fold2 = mlp(latent_size + 3, self.hidden_dim, 3)      # (coarse, z) → residual

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:  # (B, D) → (B, P, 3)
        B = z.size(0)
        z_exp = z.unsqueeze(1).expand(B, self.num_points, -1)          # (B,P,D)
        grid  = self.grid.unsqueeze(0).expand(B, -1, -1)               # (B,P,2)

        # ----------------  step‑1 : grid → coarse  ------------------- #
        h1 = torch.cat((grid, z_exp), dim=-1).view(-1, self.latent_size + 2)
        coarse = self.fold1(h1).view(B, self.num_points, 3)            # (B,P,3)

        # ----------------  step‑2 : coarse → refined  ---------------- #
        h2 = torch.cat((coarse, z_exp), dim=-1).view(-1, self.latent_size + 3)
        delta = self.fold2(h2).view(B, self.num_points, 3)
        return coarse + delta


# ------------------------------------------------------------------
#  Encoder : dense SO(3) equivariant encoder
# ------------------------------------------------------------------
class DenseEquivariantEncoder(nn.Module):
    """Simple equivariant encoder mapping *N×3* point clouds to a latent scalar code.

    The architecture mirrors e3nn's official "dense" example but strips
    it down to the essentials for molecular / crystalline clouds.  It
    remains fully rotation‑equivariant and *translation‑invariant* via a
    centering step.
    """

    def __init__(
        self,
        latent_irreps: Irreps,
        radial_dim: int = 8,
        l_max: int = 2,
    ) -> None:
        super().__init__()
        self.latent_irreps = latent_irreps
        self.l_max = l_max
        self.radial_dim = radial_dim

        # ----- point‑wise feature irreps: scalar radial basis ⊕ spherical harmonics
        scalar_irreps = Irreps(f"{radial_dim}x0e")
        sh_irreps     = o3.Irreps.spherical_harmonics(l_max)
        self.point_irreps = (scalar_irreps + sh_irreps).simplify()

        # linear map from point features → latent scalars (internally‑weighted TP)
        # Replace the thin linear layer with an N-layer MLP
        layers = []
        # First layer from point features to hidden dimension
        hidden_dim = 256  # You can adjust this hidden dimension as needed
        layers.append(o3.Linear(self.point_irreps, Irreps(f"{hidden_dim}x0e")))
        layers.append(nn.ReLU())
        
        # Middle layers (if needed)
        layers.append(o3.Linear(Irreps(f"{hidden_dim}x0e"), Irreps(f"{hidden_dim}x0e")))
        layers.append(nn.ReLU())
        
        # Final layer to latent representation
        layers.append(o3.Linear(Irreps(f"{hidden_dim}x0e"), latent_irreps))
        
        self.to_latent = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, 3) → (B, latent)
        B, N, _ = x.shape
        device = x.device

        # make representation translation‑invariant by centring cloud
        x_centered = x - x.mean(dim=1, keepdim=True)  # (B,N,3)

        #  radial scalar basis  (here: simple R, R², … up to R^8)
        r = x_centered.norm(dim=-1, keepdim=True)                      # (B,N,1)
        r_basis = torch.cat([r ** (k + 1) for k in range(self.radial_dim)], dim=-1)  # (B,N,radial_dim)

        #  spherical harmonics Yₗₘ(x̂)
        sh = o3.spherical_harmonics(list(range(self.l_max + 1)), x_centered, normalize=True, normalization='component')  # (B,N,sh_dim)

        #  concatenate features & wrap as IrrepsArray
        feat = torch.cat([r_basis, sh], dim=-1)                        # (B,N,feature_dim)

        # aggregate by mean (sum would scale with N)
        pooled = feat.mean(dim=1)                                      # (B,point_irreps)

        #  linear → latent scalar code
        latent = self.to_latent(pooled)                                # (B, latent_irreps)
        return latent                                                  # plain tensor (B, latent_dim)


# ------------------------------------------------------------------
#  Full auto‑encoder
# ------------------------------------------------------------------
class SO3DenseAutoEncoder(nn.Module):
    """SO(3)‑equivariant auto‑encoder with two‑stage folding decoder.
    Interface aligned with PointNetAE.
    """

    def __init__(
        self,
        num_points: int,
        latent_size: int,
        radial_dim: int = 8,
        l_max: int = 2,
        decoder_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.num_points = num_points
        self.latent_size = latent_size

        # latent_irreps are scalar (0e)
        latent_irreps = Irreps(f"{latent_size}x0e")
        self.encoder = DenseEquivariantEncoder(latent_irreps, radial_dim, l_max)
        self.decoder = TwoStepFoldingDecoder(num_points, latent_size, hidden_dim=decoder_hidden_dim)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], list]:
        """
        Forward pass of the SO3DenseAutoEncoder.

        Args:
            x (torch.Tensor): Input point cloud, expected shape (B, 3, N)
                              where B is batch_size, N is num_points.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], list]:
                - reconstructed_x (torch.Tensor): Reconstructed point cloud (B, N, 3).
                - latent_code (torch.Tensor): Latent representation (B, latent_size).
                - None: Placeholder for logvar (for VAE compatibility).
                - aux_outputs (list): Empty list, as no auxiliary outputs like STN matrices.
        """
        # DenseEquivariantEncoder expects input of shape (B, N, 3)
        # Assuming x is (B, 3, N) like PointNetAE
        if x.size(1) != 3:
            # This basic check can be expanded if channel dimension could vary.
            # For now, we proceed assuming 3 channels for coordinates.
            logger.warning(f"Input tensor x to SO3DenseAutoEncoder has {x.size(1)} channels, expected 3. Transposing anyway.")

        x_transposed = x.transpose(1, 2)  # Convert (B, 3, N) to (B, N, 3)
        
        latent_code = self.encoder(x_transposed)      # (B, latent_size)
        reconstructed_x = self.decoder(latent_code)  # (B, num_points, 3)
        
        # Match PointNetAE's forward signature
        aux_outputs = [] 
        
        return reconstructed_x, latent_code, None, aux_outputs

