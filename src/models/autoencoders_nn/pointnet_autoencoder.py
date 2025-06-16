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


    if model_type == "PointNetAE_MLP":
        logger.print("PointNetAE_MLP")
        return PointNetAE_MLP(num_points, latent_size, dropout_rate=dropout_rate)
    elif model_type == "PointNetAE_Folding":
        logger.print("PointNetAE_Folding")
        return PointNetAE_Folding(num_points, latent_size, dropout_rate=dropout_rate)
    elif model_type == "PointNetAE_Folding_Small":
        logger.print("PointNetAE_Folding_Small")
        return PointNetAE_Folding_Small(num_points, latent_size)  
    elif model_type == "PointNetVAE_Folding":
        logger.print("PointNetVAE_Folding")
        return PointNetVAE_Folding(num_points, latent_size, dropout_rate=dropout_rate)
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
        # self.decoder will be initialized by subclasses.

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


class _TransposedBatchNormRelu(nn.Module):
    def __init__(self, num_features, dropout_rate: float = 0.0):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.relu(self.bn(x))
        x = x.transpose(1, 2)
        if self.dropout:
            x = self.dropout(x)
        return x


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
    
    
class PointNetVAEBase(nn.Module):
    def __init__(self, num_points, latent_size, dropout_rate: float = 0.2):
        super(PointNetVAEBase, self).__init__()
        
        self.latent_size = latent_size
        self.num_points = num_points

        # Encoder part (similar to PointNetAE's encoder)
        self.features_encoder = PointNetEncoder(feature_transform=True, channel=3, dropout_rate=dropout_rate) # From existing PointNetEncoder
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
    def __init__(self, num_points, latent_size, dropout_rate: float = 0.2):
        super().__init__(num_points, latent_size, dropout_rate=dropout_rate) # Initialize PointNetVAEBase (encoder, reparameterize)
        
        # Override the decoder with the FoldingDecoder
        self.decoder = FoldingDecoderTwoStep(num_points, latent_size, dropout_rate=dropout_rate) 

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
