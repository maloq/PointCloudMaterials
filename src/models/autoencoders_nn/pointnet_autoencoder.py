from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3
from e3nn.o3 import Irreps

from pytorch3d.loss import chamfer_distance
import torch
import torch.nn as nn
from omegaconf import DictConfig
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.point_net.pointnet_cls import STNkd, STN3d
from src.utils.logging_config import setup_logging

logger = setup_logging()



def build_model(cfg: DictConfig):
    """
    Factory method to create the model based on OmegaConf configuration.

    Args:
        cfg (DictConfig): The configuration containing model parameters.

    Returns:
        nn.Module: Instantiated model.
    """
    # Use cfg.type instead of cfg.data.type
    model_type = cfg.type
    num_points = cfg.data.num_points
    latent_size = cfg.latent_size


    if model_type == "PointNetAE_MLP":
        logger.print("PointNetAE_MLP")
        return PointNetAE_MLP(num_points, latent_size)
    elif model_type == "PointNetAE_Folding":
        logger.print("PointNetAE_Folding")
        return PointNetAE_Folding(num_points, latent_size)
    elif model_type == "PointNetVAE_Folding":
        logger.print("PointNetVAE_Folding")
        return PointNetVAE_Folding(num_points, latent_size)
    elif model_type == "SO3DenseAutoEncoder":
        logger.print("SO3DenseAutoEncoder")
        # Get parameters from cfg, with defaults if not specified
        radial_dim = cfg.get('radial_dim', 8)
        l_max = cfg.get('l_max', 3)
        decoder_hidden_dim = cfg.get('decoder_hidden_dim', 512)
        return SO3DenseAutoEncoder(num_points, latent_size, 
                                   radial_dim=radial_dim, 
                                   l_max=l_max, 
                                   decoder_hidden_dim=decoder_hidden_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 

class MLP_AE(nn.Module):
    def __init__(self, num_points, latent_size):
        super(MLP_AE, self).__init__()
        self.encoder = MLPEncoder(num_points, latent_size)
        self.decoder = MLPDecoder(num_points, latent_size)

    def forward(self, x):
        x, trans, trans_feat = self.encoder(x)
        x = self.decoder(x)
        # Return reconstructed pointcloud and any transformation features in a list
        return x, [trans_feat] if trans_feat is not None else []


        
class PointNetAE(nn.Module):
    def __init__(self, num_points, latent_size):
        super(PointNetAE, self).__init__()
        
        self.latent_size = latent_size
        self.num_points = num_points

        self.features_encoder = PointNetEncoder(feature_transform=True, channel=3)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            # nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            # nn.Dropout(0.2),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            # nn.Dropout(0.2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.latent_size)
        )
        # self.decoder will be initialized by subclasses.
        # Removed: self.decoder =  nn.Linear(self.latent_size, num_points * 3).reshape(-1, self.num_points, 3)

    def encoder(self, x): 
        """
        Encodes the input point cloud to a latent vector.
        
        Args:
            x: input point cloud of shape (batch_size, 3, num_points)
            
        Returns:
            tuple: (latent_vector, transform_matrix, transform_feature_matrix)
                   where latent_vector is the encoded representation of shape (batch_size, latent_size)
        """
        # x: input point cloud of shape (batch_size, 3, num_points)
        x_features, trans, trans_feat = self.features_encoder(x) # x_features is (B, 1024)
        latent_vector = self.encoder_mlp(x_features) # latent_vector is (B, latent_size)
        return latent_vector, trans, trans_feat
    
    def forward(self, x_input):
        # Encode the input point cloud
        latent_code, _, trans_feat_encoder = self.encoder(x_input)
        
        # Decode the latent code using the decoder provided by the subclass
        if not hasattr(self, 'decoder') or not isinstance(self.decoder, nn.Module):
            raise NotImplementedError(
                "Subclasses of PointNetAE must define 'self.decoder' as an nn.Module instance in their __init__ method."
            )
            
        reconstructed_x = self.decoder(latent_code)
        
        # Collect auxiliary outputs
        aux_outputs = []
        if trans_feat_encoder is not None:
            aux_outputs.append(trans_feat_encoder)
            
        # Return: reconstructed_x, latent_representation (latent_code), logvar_or_none (None for AE), aux_outputs
        return reconstructed_x, latent_code, None, aux_outputs



class PointNetAE_MLP(PointNetAE):

    def __init__(self, num_points, latent_size):
        super().__init__(num_points, latent_size)
        self.decoder = MLPDecoder(num_points, latent_size)
    


class PointNetAE_Folding(PointNetAE):
    def __init__(self, num_points, latent_size):
        super().__init__(num_points, latent_size)
        self.decoder = FoldingDecoderTwoStep(num_points, latent_size)



class MLPEncoder(nn.Module):
    """
    Point cloud encoder using a simple MLP.
    
    Returns:
        tuple: (latent, transform_matrix, transform_feature_matrix) - standardized output format
               where latent is the encoded representation and the transform matrices may be None
    """
    def __init__(self, num_points, latent_size):
        super(MLPEncoder, self).__init__()
        self.num_points = num_points
        self.latent_size = latent_size
        
        self.encoder = nn.Sequential(
            nn.Linear(num_points * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size)
        )
        
    def forward(self, x):
        # desired shape: (batch_size, num_points * 3)
        x = x.reshape(-1, self.num_points * 3)
        latent = self.encoder(x)
        # Return standardized tuple format (latent, trans, trans_feat)
        return latent, None, None
    

class PointNetEncoder(nn.Module):
    """
    Point cloud encoder using PointNet architecture.
    
    Returns:
        tuple: (latent, transform_matrix, transform_feature_matrix) - standardized output format
               where latent is the encoded representation and transform matrices are from spatial 
               transformers if feature_transform=True
    """
    def __init__(self, feature_transform=False, channel=3):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 96, 1)
        self.conv2 = nn.Conv1d(96, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, 1024, 1)
        self.bn1 = nn.BatchNorm1d(96)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(1024)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=96)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
   
        return x, trans, trans_feat


class MLPDecoder(nn.Module):
    """
    Point cloud decoder using a simple MLP
    """
    def __init__(self, num_points, latent_size):
        super().__init__()
        self.num_points = num_points
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, num_points * 3)
        )
        
    def forward(self, x):
        x = self.decoder(x)
        return x.reshape(-1, self.num_points, 3)


class FoldingDecoderTwoStep(nn.Module):
    """
    Two-stage folding decoder (grid → coarse → refined).
    Stage-1 is identical to the original FoldingDecoder,
    Stage-2 folds the coarse surface once more using the same latent vector.
    """
    def __init__(self, num_points: int, latent_size: int, hidden_dim: int = 512):
        super().__init__()
        self.num_points  = num_points
        self.latent_size = latent_size
        self.hidden_dim = hidden_dim

        # -----  build the fixed 2-D grid  -----
        side = int(num_points ** 0.5)
        xs, ys = torch.linspace(-1, 1, side), torch.linspace(-1, 1, side)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        if grid.size(0) < num_points:      # pad or trim to N points
            grid = torch.cat((grid, grid[: num_points - grid.size(0)]), dim=0)
        self.register_buffer("grid", grid[:num_points])

        # --------  Stage-1 MLP stack  --------
        self.s1_linear1 = nn.Linear(latent_size + 2, hidden_dim)
        self.s1_bn1     = nn.BatchNorm1d(hidden_dim)
        self.s1_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.s1_bn2     = nn.BatchNorm1d(hidden_dim)
        self.s1_linear3 = nn.Linear(hidden_dim, 3)
        self.s1_bn3     = nn.BatchNorm1d(3)
        self.s1_linear4 = nn.Linear(3, 3)
        self.s1_bn4     = nn.BatchNorm1d(3)

        # --------  Stage-2 MLP stack  --------
        #  (latent + coarse-xyz  →  refined-xyz)
        self.s2_linear1 = nn.Linear(latent_size + 3, hidden_dim)
        self.s2_bn1     = nn.BatchNorm1d(hidden_dim)
        self.s2_linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.s2_bn2     = nn.BatchNorm1d(hidden_dim)
        self.s2_linear3 = nn.Linear(hidden_dim, 3)
        self.s2_bn3     = nn.BatchNorm1d(3)
        self.s2_linear4 = nn.Linear(3, 3)
        self.s2_bn4     = nn.BatchNorm1d(3)

    # --------------------------- helpers --------------------------- #
    def _fold(self, h: torch.Tensor, mlp):
        # mlp = (l1,bn1,l2,bn2,l3,bn3,l4)
        x = mlp[0](h);  x = F.relu(mlp[1](x.transpose(1, 2))).transpose(1, 2)
        x = mlp[2](x);  x = F.relu(mlp[3](x.transpose(1, 2))).transpose(1, 2)
        x = mlp[4](x);  x = F.relu(mlp[5](x.transpose(1, 2))).transpose(1, 2)
        return mlp[6](x)                             # (B, N, 3)

    # --------------------------- forward --------------------------- #
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        z : (B, latent_size)      latent vector
        Returns
        -------
        refined point cloud  (B, N, 3)
        """
        B = z.size(0)
        z_expand = z.unsqueeze(1).expand(B, self.num_points, -1)      # (B,N,L)
        grid     = self.grid.unsqueeze(0).expand(B, -1, -1)           # (B,N,2)

        # ---------- Stage-1 : grid → coarse ---------- #
        h1 = torch.cat((z_expand, grid), dim=-1)                       # (B,N,L+2)
        coarse = self._fold(h1, (self.s1_linear1, self.s1_bn1,
                                 self.s1_linear2, self.s1_bn2,
                                 self.s1_linear3, self.s1_bn3,
                                 self.s1_linear4))                    # (B,N,3)

        # ---------- Stage-2 : coarse → refined ---------- #
        h2 = torch.cat((z_expand, coarse), dim=-1)                     # (B,N,L+3)
        refined = self._fold(h2, (self.s2_linear1, self.s2_bn1,
                                  self.s2_linear2, self.s2_bn2,
                                  self.s2_linear3, self.s2_bn3,
                                  self.s2_linear4))                   # (B,N,3)
        return refined
    
    
class PointNetVAEBase(nn.Module):
    def __init__(self, num_points, latent_size):
        super(PointNetVAEBase, self).__init__()
        
        self.latent_size = latent_size
        self.num_points = num_points

        # Encoder part (similar to PointNetAE's encoder)
        self.features_encoder = PointNetEncoder(feature_transform=True, channel=3) # From existing PointNetEncoder
        self.encoder_mlp = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.latent_size * 2) # Outputs for mu and logvar
        )
        
        # A simple linear decoder for PointNetVAEBase itself, for completeness.
        # Subclasses like PointNetVAE_Folding will override this with a more complex decoder.
        self.base_decoder_linear = nn.Linear(self.latent_size, num_points * 3)

    def encoder(self, x): 
        # x shape: (batch_size, 3, num_points)
        x_feat, trans, trans_feat = self.features_encoder(x) # x_feat: (B, 1024)
        latents_concat = self.encoder_mlp(x_feat) # (B, latent_size * 2)
        
        mu = latents_concat[:, :self.latent_size]
        logvar = latents_concat[:, self.latent_size:]
        return mu, logvar, trans, trans_feat # trans is input transform, trans_feat is feature transform
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) # Sample epsilon from N(0, I)
        return mu + eps * std

    def decode_base(self, z): # Method for the base_decoder_linear
        reconstructed_flat = self.base_decoder_linear(z)
        reconstructed_points = reconstructed_flat.view(-1, self.num_points, 3)
        return reconstructed_points

    def forward(self, x):
        # This forward method is for PointNetVAEBase itself, using its simple linear decoder.
        # Subclasses will typically override this or at least the decoding part.
        mu, logvar, _, trans_feat_encoder = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        reconstructed_x = self.decode_base(z) 
        
        aux_outputs = []
        if trans_feat_encoder is not None:
            aux_outputs.append(trans_feat_encoder)
            
        return reconstructed_x, mu, logvar, aux_outputs


class PointNetVAE_Folding(PointNetVAEBase):
    def __init__(self, num_points, latent_size):
        super().__init__(num_points, latent_size) # Initialize PointNetVAEBase (encoder, reparameterize)
        
        # Override the decoder with the FoldingDecoder
        self.decoder = FoldingDecoderTwoStep(num_points, latent_size) 

    def forward(self, x):
        # Use encoder and reparameterize from PointNetVAEBase
        mu, logvar, _, trans_feat_encoder = self.encoder(x) 
        z = self.reparameterize(mu, logvar)
        
        # Use the FoldingDecoder for reconstruction
        reconstructed_x = self.decoder(z) # FoldingDecoder's forward method
        
        aux_outputs = []
        if trans_feat_encoder is not None:
            aux_outputs.append(trans_feat_encoder)
        
        # Return reconstruction, mu, logvar, and auxiliary outputs (e.g., transform features)
        return reconstructed_x, mu, logvar, aux_outputs





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




