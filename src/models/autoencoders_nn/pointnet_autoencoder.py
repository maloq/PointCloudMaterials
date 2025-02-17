from pytorch3d.loss import chamfer_distance
import torch
import torch.nn as nn
from src.models.autoencoders_nn.encoders import MLPEncoder, PointNetEncoder
from src.models.autoencoders_nn.decoders import MLPDecoder, PointNetDecoder, TransformerDecoder, FoldingDecoder, TransformerDecoderFolding
from omegaconf import DictConfig
import logging
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
    model_type = cfg.model.type
    num_points = cfg.data.num_points
    latent_size = cfg.model.latent_size

    if model_type == "PointNetAE":
        logger.print("PointNetAE")
        return PointNetAE(num_points, latent_size)
    elif model_type == "PointNetAE_MLP":
        logger.print("PointNetAE_MLP")
        return PointNetAE_MLP(num_points, latent_size)
    elif model_type == "PointNetAE_Transformer":
        logger.print("PointNetAE_Transformer")
        return PointNetAE_Transformer(num_points, latent_size)
    elif model_type == "PointNetAE_Folding":
        logger.print("PointNetAE_Folding")
        return PointNetAE_Folding(num_points, latent_size)
    elif model_type == "PointNetAE_Transformer_Folding":
        logger.print("PointNetAE_Transformer_Folding")
        return PointNetAE_Transformer_Folding(num_points, latent_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}") 

class MLP_AE(nn.Module):
    def __init__(self, num_points, latent_size):
        super(MLP_AE, self).__init__()
        self.encoder = MLPEncoder(num_points, latent_size)
        self.decoder = MLPDecoder(num_points, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, []


        
class PointNetAE(nn.Module):
    def __init__(self, num_points, latent_size):
        super(PointNetAE, self).__init__()
        
        self.latent_size = latent_size
        self.num_points = num_points

        self.features_encoder = PointNetEncoder(feature_transform=True, channel=3)
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
        self.decoder = PointNetDecoder(num_points, latent_size, feature_transform=True)

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

    def __init__(self, num_points, latent_size):
        super().__init__(num_points, latent_size)
        self.decoder = MLPDecoder(num_points, latent_size)
    
    def forward(self, x):
        x, _ , trans_feat_encoder = self.encoder(x)
        x = self.decoder(x)
        return x, [trans_feat_encoder,]



class PointNetAE_Transformer(PointNetAE):

    def __init__(self, num_points, latent_size):
        super().__init__(num_points, latent_size)
        self.decoder = TransformerDecoder(num_points, latent_size)
        
    def forward(self, x):
        x, _ , trans_feat_encoder = self.encoder(x)
        x = self.decoder(x)
        return x, [trans_feat_encoder,]

    
class PointNetAE_Folding(PointNetAE):
    def __init__(self, num_points, latent_size):
        super().__init__(num_points, latent_size)
        self.decoder = FoldingDecoder(num_points, latent_size)

    def forward(self, x):
        x, _ , trans_feat_encoder = self.encoder(x)
        x = self.decoder(x)
        return x, [trans_feat_encoder,]
    

class PointNetAE_Transformer_Folding(PointNetAE):
    def __init__(self, num_points, latent_size):
        super().__init__(num_points, latent_size)
        self.decoder = TransformerDecoderFolding(num_points, latent_size)

    def forward(self, x):
        x, _ , trans_feat_encoder = self.encoder(x)
        x = self.decoder(x)
        return x, [trans_feat_encoder,]




