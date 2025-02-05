import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.point_net.pointnet_cls import STNkd, STN3d

class MLPDecoder(nn.Module):
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


class PointNetDecoder(nn.Module):
    def __init__(self, num_points, latent_size, feature_transform=True):
        super(PointNetDecoder, self).__init__()
        self.num_points = num_points
        self.latent_size = latent_size
        self.feature_transform = feature_transform
        
        # First expand latent vector to match encoder's max-pooled features
        self.fc1 = nn.Linear(latent_size, 1024)
        
        # Expand to num_points features
        self.fc2 = nn.Linear(1024, num_points * 64)
        
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
        
        # Reshape to (batch_size, 64, num_points)
        x = x.view(-1, 64, self.num_points)
        
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
        
        # Output shape: (batch_size, 3, num_points)
        # Transpose to match expected shape (batch_size, num_points, 3)
        return x.transpose(2, 1), trans, trans_feat

class TransformerDecoder(nn.Module):
    def __init__(self, num_points, latent_size, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.num_points = num_points
        self.latent_size = latent_size
        self.d_model = d_model
        
        # Project latent vector to the transformer model dimension.
        self.input_proj = nn.Linear(latent_size, d_model)
        
        # Learnable query embeddings for each output point.
        # These queries will be used as input to the transformer decoder.
        self.query_embed = nn.Parameter(torch.randn(num_points, d_model))
        
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
        # shape: (num_points, batch_size, d_model)
        queries = self.query_embed.unsqueeze(1).expand(self.num_points, batch_size, self.d_model)
        
        # Decode using the transformer decoder:
        # output shape: (num_points, batch_size, d_model)
        out = self.transformer_decoder(tgt=queries, memory=memory)
        
        # Project each token to 3D coordinates:
        # (num_points, batch_size, 3) then transpose to (batch_size, num_points, 3)
        out = self.output_proj(out).transpose(0, 1)
        
        return out
    

class FoldingDecoder(nn.Module):
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
        
        self.mlp1 = nn.Linear(latent_size + 2, 512)
        self.mlp2 = nn.Linear(512, 512)
        self.mlp3 = nn.Linear(512, 256)
        self.mlp4 = nn.Linear(256, 3)
    
    def forward(self, x):
        # x: (batch_size, latent_size)
        batch_size = x.size(0)
        
        # Expand latent vector to (batch_size, num_points, latent_size)
        x_expanded = x.unsqueeze(1).expand(-1, self.num_points, -1)
        # Expand grid (batch_size, num_points, 2)
        grid = self.grid.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Concatenate latent vector with grid coordinates
        x_in = torch.cat([x_expanded, grid], dim=-1)
        x = F.relu(self.mlp1(x_in))
        x = F.relu(self.mlp2(x))
        x = F.relu(self.mlp3(x))
        x = self.mlp4(x)
        
        return x