from __future__ import annotations

import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    dropout_rate = cfg.get("dropout_rate", 0.2)
    if num_points <24:
        R1 = cfg.get("R1", 0.54)
        R2 = cfg.get("R2", 0.75) # ignored in this case
        n_shells = cfg.get("n_shells", 1)
    else:
        R1 = cfg.get("R1", 0.38)
        R2 = cfg.get("R2", 0.54)
        n_shells = cfg.get("n_shells", 2)

    if model_type == "PointNetAE_MLP":
        logger.print("PointNetAE_MLP")
        return PointNetAE_MLP(num_points, latent_size, dropout_rate=dropout_rate)
    elif model_type == "PointNetAE_Folding":
        logger.print("PointNetAE_Folding")
        return PointNetAE_Folding(num_points, latent_size, dropout_rate=dropout_rate)
    elif model_type == "PointNetAE_Folding_Small":
        logger.print("PointNetAE_Folding_Small")
        return PointNetAE_Folding_Small(num_points, latent_size)  
    elif model_type == "PointNetAE_Folding_Sphere":
        logger.print("PointNetAE_Folding_Sphere")
        return PointNetAE_Folding_Sphere(num_points, latent_size, dropout_rate=dropout_rate, R1=R1, R2=R2, n_shells=n_shells)
    elif model_type == "FoldingDecoderSphereTwoShellAttn":
        logger.print("FoldingDecoderSphereTwoShellAttn")
        return FoldingDecoderSphereTwoShellAttn(num_points, latent_size, dropout_rate=dropout_rate, R1=R1, R2=R2, n_shells=n_shells)
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
    def __init__(self, num_points, latent_size, dropout_rate: float = 0.2):
        super(PointNetAE, self).__init__()
        
        self.latent_size = latent_size
        self.num_points = num_points

        self.features_encoder = PointNetEncoder(feature_transform=True, channel=3, dropout_rate=dropout_rate)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.latent_size)
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

    def __init__(self, num_points, latent_size, dropout_rate: float = 0.2):
        super().__init__(num_points, latent_size, dropout_rate=dropout_rate)
        self.decoder = MLPDecoder(num_points, latent_size)
    


class PointNetAE_Folding(PointNetAE):
    def __init__(self, num_points, latent_size, dropout_rate: float = 0.2):
        super().__init__(num_points, latent_size, dropout_rate=dropout_rate)
        self.decoder = FoldingDecoderTwoStep(num_points, latent_size, dropout_rate=dropout_rate)


class PointNetAE_Folding_Sphere(PointNetAE):
    def __init__(self, num_points, latent_size, dropout_rate: float = 0.2, R1: float = 0.5, R2: float = 1.0, n_shells: int = 2):
        super().__init__(num_points, latent_size, dropout_rate=dropout_rate)
        self.decoder = FoldingDecoderSphereTwoShell(num_points, latent_size, dropout_rate=dropout_rate, R1=R1, R2=R2, n_shells=n_shells)


class PointNetAE_Small(PointNetAE):
    def __init__(self, num_points, latent_size, dropout_rate: float = 0.2):
        super(PointNetAE_Small, self).__init__(num_points, latent_size, dropout_rate=dropout_rate)
        
        self.latent_size = latent_size
        self.num_points = num_points

        self.features_encoder = PointNetEncoderSmall(feature_transform=True, channel=3)
        self.encoder_mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, self.latent_size)
        )

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
        x_features, trans, trans_feat = self.features_encoder(x) # x_features is (B, 512)
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


class PointNetAE_Folding_Small(PointNetAE_Small):
    def __init__(self, num_points, latent_size):
        super().__init__(num_points, latent_size)
        self.decoder = FoldingDecoderTwoStepSmall(num_points, latent_size)



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
    def __init__(self, feature_transform=False, channel=3, dropout_rate: float = 0.2):
        super(PointNetEncoder, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 96, 1)
        self.conv2 = nn.Conv1d(96, 256, 1)
        self.conv3 = nn.Conv1d(256, 512, 1)
        self.conv4 = nn.Conv1d(512, 512, 1)
        self.bn1 = nn.BatchNorm1d(96)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout3 = nn.Dropout(dropout_rate)

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
        x = self.dropout1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout3(x)
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)
   
        return x, trans, trans_feat


class PointNetEncoderSmall(nn.Module):
    """
    Smaller version of PointNetEncoder.
    
    Returns:
        tuple: (latent, transform_matrix, transform_feature_matrix) - standardized output format
               where latent is the encoded representation and transform matrices are from spatial 
               transformers if feature_transform=True
    """
    def __init__(self, feature_transform=False, channel=3):
        super(PointNetEncoderSmall, self).__init__()
        self.stn = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

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
        x = x.view(-1, 512)
   
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
    def __init__(self, num_points: int, latent_size: int, hidden_dim: int = 512, dropout_rate: float = 0.0):
        super().__init__()
        self.num_points  = num_points
        self.latent_size = latent_size

        # -----  build the fixed 2-D grid  -----
        side = int(math.sqrt(num_points))
        if side * side != num_points:
            side += 1
        
        xs, ys = torch.linspace(-1, 1, side), torch.linspace(-1, 1, side)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        self.register_buffer("grid", grid[:num_points])

        # --------  Stage-1 MLP stack  --------
        self.stage1 = nn.Sequential(
            nn.Linear(latent_size + 2, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, 3),
            _TransposedBatchNormRelu(3),
            nn.Linear(3, 3)
        )

        # --------  Stage-2 MLP stack  --------
        self.stage2 = nn.Sequential(
            nn.Linear(latent_size + 3, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, 3),
            _TransposedBatchNormRelu(3),
            nn.Linear(3, 3)
        )

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
        z_expand = z.unsqueeze(1).expand(B, self.num_points, -1)
        grid     = self.grid.unsqueeze(0).expand(B, -1, -1)

        # ---------- Stage-1 : grid → coarse ---------- #
        h1 = torch.cat((z_expand, grid), dim=-1)
        coarse = self.stage1(h1)

        # ---------- Stage-2 : coarse → refined ---------- #
        h2 = torch.cat((z_expand, coarse), dim=-1)
        refined = self.stage2(h2)
        return refined
    
    


class FoldingDecoderTwoStepSmall(nn.Module):
    """
    Smaller version of FoldingDecoderTwoStep.
    """
    def __init__(self, num_points: int, latent_size: int, hidden_dim: int = 256, dropout_rate: float = 0.0):
        super().__init__()
        self.num_points  = num_points
        self.latent_size = latent_size

        # -----  build the fixed 2-D grid  -----
        side = int(math.sqrt(num_points))
        if side * side != num_points:
            side += 1
        
        xs, ys = torch.linspace(-1, 1, side), torch.linspace(-1, 1, side)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        self.register_buffer("grid", grid[:num_points])

        # --------  Stage-1 MLP stack  --------
        self.stage1 = nn.Sequential(
            nn.Linear(latent_size + 2, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, 3),
            _TransposedBatchNormRelu(3),
            nn.Linear(3, 3)
        )

        # --------  Stage-2 MLP stack  --------
        self.stage2 = nn.Sequential(
            nn.Linear(latent_size + 3, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, 3),
            _TransposedBatchNormRelu(3),
            nn.Linear(3, 3)
        )

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
        z_expand = z.unsqueeze(1).expand(B, self.num_points, -1)
        grid     = self.grid.unsqueeze(0).expand(B, -1, -1)

        # ---------- Stage-1 : grid → coarse ---------- #
        h1 = torch.cat((z_expand, grid), dim=-1)
        coarse = self.stage1(h1)

        # ---------- Stage-2 : coarse → refined ---------- #
        h2 = torch.cat((z_expand, coarse), dim=-1)
        refined = self.stage2(h2)
        return refined
    

class _TransposedBatchNormRelu(nn.Module):
    """BN + ReLU for tensors of shape (B, N, C).

    The tensor is temporarily transposed to (B, C, N) so that ``BatchNorm1d``
    can be applied over the *channel* dimension C.
    """

    def __init__(self, num_features: int, dropout_rate: float = 0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, N, C)
        x = x.transpose(1, 2)            # (B, C, N)
        x = F.relu(self.bn(x))
        x = x.transpose(1, 2)            # (B, N, C)
        if self.dropout is not None:
            x = self.dropout(x)
        return x



def _fibonacci_sphere_points(n: int, radius: float) -> torch.Tensor:
    """Generate *n* approximately uniformly distributed points on a sphere.

    Uses the *Fibonacci sphere* / *golden‑section spiral* algorithm, which is
    deterministic and O(n).  Points lie on the sphere of the given *radius*.
    Returns a tensor of shape *(n, 3)*.
    """
    if n <= 0:
        return torch.empty((0, 3))

    device = "cpu"  # will be moved later by the caller via register_buffer
    i = torch.arange(n, dtype=torch.float32, device=device)  # [0, 1, …, n‑1]

    phi = (1 + 5 ** 0.5) / 2  # golden ratio
    theta = 2 * math.pi * i / phi

    z = 1.0 - 2.0 * (i + 0.5) / n           # linear in [1‑1/n, ‑1+1/n]
    r_xy = torch.sqrt(1.0 - z * z)

    x = r_xy * torch.cos(theta)
    y = r_xy * torch.sin(theta)

    return radius * torch.stack((x, y, z), dim=-1)  # (n, 3)


# -----------------------------------------------------------------------------
#  Main decoder
# -----------------------------------------------------------------------------

class FoldingDecoderSphereTwoShell(nn.Module):
    r"""FoldingNet-style decoder initialised from one **or** two spherical shells
    plus a central point.

    Parameters
    ----------
    num_points : int
        Total number of points *N* (including the centre).
    latent_size : int
        Dimensionality of the latent vector.
    n_shells : {1, 2}, optional
        How many spherical shells to use (default **2**).
    R1, R2 : float, optional
        Radii of the shells.  If *n_shells* = 1 only *R1* is used.
    hidden_dim : int, optional
        Width of the MLP stacks (default **512**).
    dropout_rate : float, optional
        Dropout after each hidden layer (default **0.0**).
    """

    def __init__(
        self,
        num_points: int,
        latent_size: int,
        *,
        n_shells: int = 2,
        R1: float = 0.5,
        R2: float = 1.0,
        hidden_dim: int = 512,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if n_shells not in (1, 2):
            raise ValueError("`n_shells` must be 1 or 2.")
        if R1 <= 0 or (n_shells == 2 and (R2 <= 0 or R1 >= R2)):
            raise ValueError("Require R1 > 0 and (if two shells) 0 < R1 < R2.")

        # --------------------------------------------------------------
        #  Build the fixed 3-D template  (centre point  +  shell points)
        # --------------------------------------------------------------
        # Reserve exactly one point for the centre
        if num_points < 2:
            raise ValueError("`num_points` must be at least 2 (centre + shell).")
        n_shell_pts = num_points - 1          # remaining points go on shell(s)

        if n_shells == 1:
            # All shell points on radius R1
            pts_shell = _fibonacci_sphere_points(n_shell_pts, R1)
            template = torch.cat(
                [torch.zeros(1, 3),          # centre (0,0,0)
                 pts_shell],
                dim=0,
            )                                # (N, 3)
        else:  # n_shells == 2
            # Allocate points proportional to surface areas (R²)
            area_ratio = R1 ** 2 / (R1 ** 2 + R2 ** 2)
            n1 = max(1, round(n_shell_pts * area_ratio))
            n2 = n_shell_pts - n1            # ensure total matches exactly
            if n2 == 0:                      # pathological but guard anyway
                n1 -= 1
                n2 = 1

            pts_shell1 = _fibonacci_sphere_points(n1, R1)
            pts_shell2 = _fibonacci_sphere_points(n2, R2)

            template = torch.cat(
                [torch.zeros(1, 3),          # centre
                 pts_shell1,
                 pts_shell2],
                dim=0,
            )                                # (N, 3)

        # Register so the buffer moves with the module & is saved in checkpoints
        self.register_buffer("template", template)

        self.num_points  = num_points
        self.latent_size = latent_size

        # --------------------------------------------------------------
        #  Two-stage folding MLPs  (unchanged from original implementation)
        # --------------------------------------------------------------
        self.stage1 = nn.Sequential(
            nn.Linear(latent_size + 3, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, 3),
            _TransposedBatchNormRelu(3),
            nn.Linear(3, 3),
        )

        self.stage2 = nn.Sequential(
            nn.Linear(latent_size + 3, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            _TransposedBatchNormRelu(hidden_dim, dropout_rate=dropout_rate),
            nn.Linear(hidden_dim, 3),
            _TransposedBatchNormRelu(3),
            nn.Linear(3, 3),
        )

    # ------------------------------------------------------------------
    #  Forward pass
    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:     # (B, latent)
        B = z.size(0)

        z_exp  = z.unsqueeze(1).expand(B, self.num_points, -1)   # (B, N, latent)
        templ  = self.template.unsqueeze(0).expand(B, -1, -1)    # (B, N, 3)

        # -------- Stage-1 : template → coarse deformation -------- #
        h1 = torch.cat((z_exp, templ), dim=-1)                  # (B, N, latent+3)
        coarse = self.stage1(h1)                                # (B, N, 3)

        # -------- Stage-2 : coarse → refined output ------------- #
        h2 = torch.cat((z_exp, coarse), dim=-1)
        refined = self.stage2(h2)                               # (B, N, 3)

        return refined


class FoldingDecoderSphereTwoShellAttn(nn.Module):
    r"""FoldingNet-style decoder with three folding steps **plus local self-attention**.

    Parameters
    ----------
    num_points  : int     Total #points **N** (incl. centre).
    latent_size : int     Dimensionality of the latent code.
    n_shells    : {1,2}   One or two initial spherical shells (default 2).
    R1, R2      : float   Radii of the shells.  *R2* ignored if *n_shells==1*.
    R_att       : float   Cut-off radius for attention neighbourhoods.
    hidden_dim  : int     Width of per-point MLPs (default 512).
    attn_dim    : int     Dimension of Q/K/V projections (default 128).
    dropout_rate: float   Dropout rate after hidden layers (default 0).
    """

    def __init__(
        self,
        num_points      : int,
        latent_size     : int,
        *,
        n_shells        : int   = 2,
        R1              : float = 1.0,
        R2              : float = 1.5,
        R_att           : float = 0.20,
        hidden_dim      : int   = 512,
        attn_dim        : int   = 128,
        dropout_rate    : float = 0.0,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------
        #  Template construction  (identical to previous implementation)
        # ------------------------------------------------------------------
        if n_shells not in (1, 2):
            raise ValueError("`n_shells` must be 1 or 2.")
        if num_points < 2:
            raise ValueError("Need at least 2 points (centre + shell).")

        n_shell_pts = num_points - 1
        if n_shells == 1:
            shell_pts = _fibonacci_sphere_points(n_shell_pts, R1)
            template  = torch.cat([torch.zeros(1, 3), shell_pts], dim=0)
        else:
            # Allocate by surface-area ratio
            area_ratio = R1**2 / (R1**2 + R2**2)
            n1 = max(1, round(n_shell_pts * area_ratio))
            n2 = n_shell_pts - n1
            if n2 == 0: n1, n2 = n1 - 1, 1
            t1 = _fibonacci_sphere_points(n1, R1)
            t2 = _fibonacci_sphere_points(n2, R2)
            template = torch.cat([torch.zeros(1, 3), t1, t2], dim=0)

        self.register_buffer("template", template)      # (N, 3)
        self.num_points   = num_points
        self.latent_size  = latent_size
        self.R_att        = R_att
        self.sqrt_d       = math.sqrt(attn_dim)

        # ------------------------------------------------------------------
        #  Shared Q/K/V projections for attention
        # ------------------------------------------------------------------
        self.q_proj = nn.Linear(3, attn_dim, bias=False)
        self.k_proj = nn.Linear(3, attn_dim, bias=False)
        self.v_proj = nn.Linear(3, attn_dim, bias=False)

        # ------------------------------------------------------------------
        #  Three folding stages
        #  Each stage:  (latent + coords + attn) → per-point MLP → new coords
        # ------------------------------------------------------------------
        def make_stage(in_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                _TransposedBatchNormRelu(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, hidden_dim),
                _TransposedBatchNormRelu(hidden_dim, dropout_rate),
                nn.Linear(hidden_dim, 3),
            )

        # Stage-0 takes latent + template  (no attn yet)
        self.stage0 = make_stage(latent_size + 3)
        # Stage-1 and Stage-2 take latent + prev_coords + attn_out
        self.stage1 = make_stage(latent_size + 3 + attn_dim)
        self.stage2 = make_stage(latent_size + 3 + attn_dim)

    # ======================================================================
    #                            F O R W A R D
    # ======================================================================
    def forward(self, z: torch.Tensor) -> torch.Tensor:               # (B,L)
        B = z.shape[0]
        z_exp = z.unsqueeze(1).expand(B, self.num_points, -1)         # (B,N,L)

        # ---------- Stage-0 : template → coarse-0 (no attention) ------------ #
        coords0 = self.stage0(torch.cat((z_exp, self.template.expand(B, -1, -1)), dim=-1))

        # ---------- Stage-1 : attention on coords0 → coords1 ---------------- #
        attn1   = self._local_attention(coords0)                      # (B,N,A)
        coords1 = self.stage1(torch.cat((z_exp, coords0, attn1), dim=-1))

        # ---------- Stage-2 : attention on coords1 → refined --------------- #
        attn2   = self._local_attention(coords1)
        coords2 = self.stage2(torch.cat((z_exp, coords1, attn2), dim=-1))

        return coords2                                                # (B,N,3)

    # ------------------------------------------------------------------
    #  Local self-attention with radius-based masking
    # ------------------------------------------------------------------
    def _local_attention(self, coords: torch.Tensor) -> torch.Tensor:
        """
        coords : (B, N, 3)  — current point coordinates

        Returns
        -------
        out    : (B, N, attn_dim) — aggregated value vectors
        """
        # Q,K,V  from coordinates
        Q = self.q_proj(coords)                                       # (B,N,A)
        K = self.k_proj(coords)                                       # (B,N,A)
        V = self.v_proj(coords)                                       # (B,N,A)

        # Pair-wise squared distances
        D2 = torch.cdist(coords, coords, p=2.0) ** 2                  # (B,N,N)

        # Build mask:  True = keep, False = ignore
        mask = D2 <= (self.R_att ** 2)                                # (B,N,N)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.sqrt_d   # (B,N,N)
        scores = scores.masked_fill(~mask, float("-inf"))
        W = F.softmax(scores, dim=-1)                                 # (B,N,N)
        W = W.masked_fill(~mask, 0.0)

        # Attention output
        out = torch.matmul(W, V)                                      # (B,N,A)
        return out
