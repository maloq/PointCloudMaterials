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
        self.decoder = FoldingDecoder(num_points, latent_size)



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



    

class FoldingDecoder(nn.Module):
    """
    Point cloud decoder using folding.
    """
    def __init__(self, num_points, latent_size):
        super(FoldingDecoder, self).__init__()
        self.num_points = num_points
        self.latent_size = latent_size
        
        # Build a fixed 2D grid as a prior (assuming num_points is a perfect square)
        side = int(num_points ** 0.5)
        xs = torch.linspace(-1, 1, steps=side)
        ys = torch.linspace(-1, 1, steps=side)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
        grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        
        # Ensure grid has exactly num_points elements (pad or trim if necessary)
        if grid.size(0) < num_points:
            pad = num_points - grid.size(0)
            grid = torch.cat([grid, grid[:pad]], dim=0)
        elif grid.size(0) > num_points:
            grid = grid[:num_points]
        
        self.register_buffer('grid', grid)
        
        self.mlp1 = nn.Linear(latent_size + 2, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.mlp2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.mlp3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.mlp4 = nn.Linear(256, 3)
    
    def forward(self, x):
        # x: (batch_size, latent_size)
        batch_size = x.size(0)
        
        # Expand latent vector to (batch_size, num_points, latent_size)
        x_expanded = x.unsqueeze(1).expand(-1, self.num_points, -1)
        # Expand grid (batch_size, num_points, 2)
        grid = self.grid.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate latent vector with grid coordinates: shape (B, num_points, latent_size+2)
        x_in = torch.cat([x_expanded, grid], dim=-1)
        
        # MLP1: output shape (B, num_points, 1024)
        x = self.mlp1(x_in)
        # Transpose so that feature dimension is second: (B, 1024, num_points)
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(x))
        # Transpose back: (B, num_points, 1024)
        x = x.transpose(1, 2)
        
        # For MLP2:
        x = self.mlp2(x)
        x = x.transpose(1, 2)
        x = F.relu(self.bn2(x))
        x = x.transpose(1, 2)
        
        # For MLP3:
        x = self.mlp3(x)
        x = x.transpose(1, 2)
        x = F.relu(self.bn3(x))
        x = x.transpose(1, 2)
        
        x = self.mlp4(x)
        
        return x


class FoldingDecoderTwoStep(nn.Module):
    """
    Two-stage folding decoder (grid → coarse → refined).
    Stage-1 is identical to the original FoldingDecoder,
    Stage-2 folds the coarse surface once more using the same latent vector.
    """
    def __init__(self, num_points: int, latent_size: int):
        super().__init__()
        self.num_points  = num_points
        self.latent_size = latent_size

        # -----  build the fixed 2-D grid  -----
        side = int(num_points ** 0.5)
        xs, ys = torch.linspace(-1, 1, side), torch.linspace(-1, 1, side)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        if grid.size(0) < num_points:      # pad or trim to N points
            grid = torch.cat((grid, grid[: num_points - grid.size(0)]), dim=0)
        self.register_buffer("grid", grid[:num_points])

        # --------  Stage-1 MLP stack  --------
        self.s1_linear1 = nn.Linear(latent_size + 2, 1024)
        self.s1_bn1     = nn.BatchNorm1d(1024)
        self.s1_linear2 = nn.Linear(1024, 512)
        self.s1_bn2     = nn.BatchNorm1d(512)
        self.s1_linear3 = nn.Linear(512, 256)
        self.s1_bn3     = nn.BatchNorm1d(256)
        self.s1_linear4 = nn.Linear(256, 3)          # coarse output

        # --------  Stage-2 MLP stack  --------
        #  (latent + coarse-xyz  →  refined-xyz)
        self.s2_linear1 = nn.Linear(latent_size + 3, 512)
        self.s2_bn1     = nn.BatchNorm1d(512)
        self.s2_linear2 = nn.Linear(512, 256)
        self.s2_bn2     = nn.BatchNorm1d(256)
        self.s2_linear3 = nn.Linear(256, 128)
        self.s2_bn3     = nn.BatchNorm1d(128)
        self.s2_linear4 = nn.Linear(128, 3)          # refined output

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
        z_expand = z.unsqueeze(1).expand(-1, self.num_points, -1)      # (B,N,L)
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






