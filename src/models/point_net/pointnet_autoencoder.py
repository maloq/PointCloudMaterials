from pytorch3d.loss import chamfer_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.point_net.pointnet_cls import PointNetEncoder, STNkd, STN3d



class MLPEncoder(nn.Module):
    def __init__(self, point_size, latent_size):
        super(MLPEncoder, self).__init__()
        self.point_size = point_size
        self.latent_size = latent_size
        
        self.encoder = nn.Sequential(
            nn.Linear(point_size * 3, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, latent_size)
        )
        
    def forward(self, x):
        # desired shape: (batch_size, point_size * 3)
        x = x.reshape(-1, self.point_size * 3)
        return self.encoder(x)


class MLPDecoder(nn.Module):
    def __init__(self, point_size, latent_size):
        super(MLPDecoder, self).__init__()
        self.point_size = point_size
        self.latent_size = latent_size
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, point_size * 3)
        )
        
    def forward(self, x):
        x = self.decoder(x)
        return x.reshape(-1, self.point_size, 3)



class MLP_AE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(MLP_AE, self).__init__()
        self.encoder = MLPEncoder(point_size, latent_size)
        self.decoder = MLPDecoder(point_size, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x, []



class PointNetDecoder(nn.Module):
    def __init__(self, point_size, latent_size, feature_transform=True):
        super(PointNetDecoder, self).__init__()
        self.point_size = point_size
        self.latent_size = latent_size
        self.feature_transform = feature_transform
        
        # First expand latent vector to match encoder's max-pooled features
        self.fc1 = nn.Linear(latent_size, 1024)
        
        # Expand to point_size features
        self.fc2 = nn.Linear(1024, point_size * 64)
        
        # Convolution layers mirroring encoder (in reverse)
        self.conv1 = torch.nn.Conv1d(64, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 3, 1)  # Output 3D points
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(64)

        # Feature transform networks
        if self.feature_transform:
            self.fstn = STNkd(k=64)  # Feature transform for 64-dim features
            self.stn = STN3d(channel=3)  # Spatial transform for 3D points
        
    def forward(self, x):
        # Expand from latent vector
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Reshape to (batch_size, 64, point_size)
        x = x.view(-1, 64, self.point_size)
        
        # First feature transformation
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        
        # Apply convolutions with batch norm
        x = F.relu(self.bn2(self.conv1(x)))
        x = F.relu(self.bn3(self.conv2(x)))
        x = self.conv3(x)
        
        # Final spatial transformation for output points
        if self.feature_transform:
            trans = self.stn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
        
        # Output shape: (batch_size, 3, point_size)
        # Transpose to match expected shape (batch_size, point_size, 3)
        return x.transpose(2, 1), trans, trans_feat

class TransformerDecoder(nn.Module):
    def __init__(self, point_size, latent_size, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.point_size = point_size
        self.latent_size = latent_size
        self.d_model = d_model
        
        # Project latent vector to the transformer model dimension.
        self.input_proj = nn.Linear(latent_size, d_model)
        
        # Learnable query embeddings for each output point.
        # These queries will be used as input to the transformer decoder.
        self.query_embed = nn.Parameter(torch.randn(point_size, d_model))
        
        # Build a stack of transformer decoder layers.
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final projection from transformer output dimension to 3 (for x, y, z).
        self.output_proj = nn.Linear(d_model, 3)
    
    def forward(self, x):
        # x is expected to be of shape (batch_size, latent_size)
        batch_size = x.size(0)
        
        # Create transformer memory from the latent vector:
        # shape: (1, batch_size, d_model)
        memory = self.input_proj(x).unsqueeze(0)
        
        # Expand the learned query embeddings to the current batch:
        # shape: (point_size, batch_size, d_model)
        queries = self.query_embed.unsqueeze(1).expand(self.point_size, batch_size, self.d_model)
        
        # Decode using the transformer decoder:
        # output shape: (point_size, batch_size, d_model)
        out = self.transformer_decoder(tgt=queries, memory=memory)
        
        # Project each token to 3D coordinates:
        # (point_size, batch_size, 3) then transpose to (batch_size, point_size, 3)
        out = self.output_proj(out).transpose(0, 1)
        
        return out

        
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


    