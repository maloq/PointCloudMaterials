from pytorch3d.loss import chamfer_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.point_net.pointnet_cls import PointNetEncoder, STNkd, STN3d
from src.models.autoencoder.encoders import MLPEncoder
from src.models.autoencoder.decoders import MLPDecoder, PointNetDecoder, TransformerDecoder, FoldingDecoder
from omegaconf import DictConfig


def build_model(cfg: DictConfig):
    """
    Factory method to create the model based on OmegaConf configuration.

    Args:
        cfg (DictConfig): The configuration containing model parameters.

    Returns:
        nn.Module: Instantiated model.
    """
    model_type = cfg.model.type
    point_size = cfg.data.point_size
    latent_size = cfg.model.latent_size

    if model_type == "PointNetAE":
        return PointNetAE(point_size, latent_size)
    elif model_type == "PointNetAE_MLP":
        return PointNetAE_MLP(point_size, latent_size)
    elif model_type == "PointNetAE_Transformer":
        return PointNetAE_Transformer(point_size, latent_size)
    elif model_type == "PointNetAE_Folding":
        return PointNetAE_Folding(point_size, latent_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 

class MLP_AE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(MLP_AE, self).__init__()
        self.encoder = MLPEncoder(point_size, latent_size)
        self.decoder = MLPDecoder(point_size, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, []


        
class PointNetAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(PointNetAE, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size

        self.features_encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=3)
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
            nn.Linear(256, self.latent_size)
        )
        self.decoder = PointNetDecoder(point_size, latent_size, feature_transform=True)

    def encoder(self, x): 
        x, trans, trans_feat = self.features_encoder(x)
        x = self.encoder_mlp(x)
        x = x.view(-1, self.latent_size)
        return x, trans, trans_feat
    
    def forward(self, x):
        x, _, trans_feat_encoder = self.encoder(x)
        x, _, trans_feat_decoder = self.decoder(x)
        return x, [trans_feat_encoder, trans_feat_decoder]



class PointNetAE_MLP(PointNetAE):

    def __init__(self, point_size, latent_size):
        super().__init__(point_size, latent_size)
        self.decoder = MLPDecoder(point_size, latent_size)
    
    def forward(self, x):
        x, _ , trans_feat_encoder = self.encoder(x)
        x = self.decoder(x)
        return x, [trans_feat_encoder,]



class PointNetAE_Transformer(PointNetAE):

    def __init__(self, point_size, latent_size):
        super().__init__(point_size, latent_size)
        self.decoder = TransformerDecoder(point_size, latent_size)
        
    def forward(self, x):
        x, _ , trans_feat_encoder = self.encoder(x)
        x = self.decoder(x)
        return x, [trans_feat_encoder,]

    
class PointNetAE_Folding(PointNetAE):
    def __init__(self, point_size, latent_size):
        super().__init__(point_size, latent_size)
        self.decoder = FoldingDecoder(point_size, latent_size)

    def forward(self, x):
        x, _ , trans_feat_encoder = self.encoder(x)
        x = self.decoder(x)
        return x, [trans_feat_encoder,]




