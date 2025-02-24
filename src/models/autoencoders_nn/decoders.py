import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.point_net.pointnet_cls import STNkd, STN3d


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


class PointNetDecoder(nn.Module):
    """
    Point cloud decoder using PointNet architecture.
    """
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
    """
    Point cloud decoder using a transformer.
    """
    def __init__(self, num_points, latent_size, d_model=256, nhead=8, num_layers=3, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.num_points = num_points
        self.latent_size = latent_size
        self.d_model = d_model
        
        self.input_proj = nn.Linear(latent_size, d_model)
        
        self.query_embed = nn.Parameter(torch.randn(num_points, d_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 3)
    
    def forward(self, x):
        # x is expected to be of shape (batch_size, latent_size)
        batch_size = x.size(0)
        # shape: (1, batch_size, d_model)
        memory = self.input_proj(x).unsqueeze(0)
        # shape: (num_points, batch_size, d_model)
        queries = self.query_embed.unsqueeze(1).expand(self.num_points, batch_size, self.d_model)
        # output shape: (num_points, batch_size, d_model)
        out = self.transformer_decoder(tgt=queries, memory=memory)
        # (num_points, batch_size, 3) then transpose to (batch_size, num_points, 3)
        out = self.output_proj(out).transpose(0, 1)
        
        return out
    

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
        
        self.mlp1 = nn.Linear(latent_size + 2, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.mlp2 = nn.Linear(512, 512)
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
    

class FoldingDecoderRefined(nn.Module):
    """
    A two-stage decoder where the first stage produces a coarse output and a second stage refines it.
    """
    def __init__(self, num_points, latent_size):
        super(FoldingDecoderRefined, self).__init__()
        self.coarse_decoder = FoldingDecoder(num_points, latent_size)
        # Refinement MLP that works on the concatenation of the coarse output and the latent code.
        self.refine_mlp1 = nn.Linear(3 + latent_size, 512)
        self.refine_bn1 = nn.BatchNorm1d(512)
        self.refine_mlp2 = nn.Linear(512, 256)
        self.refine_bn2 = nn.BatchNorm1d(256)
        self.refine_mlp3 = nn.Linear(256, 3)

    def forward(self, x):
        # Coarse output: shape (B, num_points, 3)
        coarse_output = self.coarse_decoder(x)
        batch_size, num_points, _ = coarse_output.size()
          
        # Expand the latent code to match the number of points: shape (B, num_points, latent_size)
        x_expanded = x.unsqueeze(1).expand(-1, num_points, -1)
          
        # Concatenate coarse output and latent features: shape (B, num_points, latent_size + 3)
        refinement_input = torch.cat([coarse_output, x_expanded], dim=2)
          
        # Apply the refinement MLP:
        # 1. Process with the linear layer (operates on the last dimension)
        x_refined = self.refine_mlp1(refinement_input)        # (B, num_points, 512)
        # 2. Transpose to (B, 512, num_points) for BatchNorm1d
        x_refined = x_refined.transpose(1, 2)
        x_refined = F.relu(self.refine_bn1(x_refined))
        # Transpose back to (B, num_points, 512)
        x_refined = x_refined.transpose(1, 2)
          
        x_refined = self.refine_mlp2(x_refined)                # (B, num_points, 256)
        x_refined = x_refined.transpose(1, 2)
        x_refined = F.relu(self.refine_bn2(x_refined))
        x_refined = x_refined.transpose(1, 2)                  # (B, num_points, 256)
          
        # Final linear layer to produce refined 3D points
        refined = self.refine_mlp3(x_refined)                  # (B, num_points, 3)
        return refined
      


class TransformerDecoderFolding(nn.Module):
    """
    Point cloud decoder using a transformer with folding.
    """
    def __init__(self, num_points, latent_size, d_model=512, nhead=8, num_layers=3):
        super(TransformerDecoderFolding, self).__init__()
        self.num_points = num_points
        self.latent_size = latent_size
        self.d_model = d_model

        # Build a fixed 2D grid (assuming num_points is a perfect square)
        side = int(num_points ** 0.5)
        xs = torch.linspace(-1, 1, steps=side)
        ys = torch.linspace(-1, 1, steps=side)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
        grid = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        # Ensure grid has exactly num_points elements
        if grid.size(0) < num_points:
            pad = num_points - grid.size(0)
            grid = torch.cat([grid, grid[:pad]], dim=0)
        elif grid.size(0) > num_points:
            grid = grid[:num_points]
        self.register_buffer("grid", grid)  # shape: (num_points, 2)

        # Project grid coordinates (2D) to d_model dimension (queries)
        self.query_proj = nn.Linear(2, d_model, bias=False)
        # Project the latent code to d_model (memory)
        self.latent_proj = nn.Linear(latent_size, d_model)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.out_proj = nn.Linear(d_model, 3)
    
    def forward(self, x):
        """
        x: latent vector of shape (batch_size, latent_size)
        """
        batch_size = x.size(0)

        # Create query tokens from grid:
        # grid shape: (num_points, 2) -> (num_points, d_model)
        queries = self.query_proj(self.grid)  # (num_points, d_model)
        # Expand queries for batch and transpose to shape (num_points, batch_size, d_model)
        queries = queries.unsqueeze(1).expand(-1, batch_size, -1)

        # Create memory tokens from latent vector:
        # First, project latent x: (batch_size, latent_size) -> (batch_size, d_model).
        memory = self.latent_proj(x)  # (batch_size, d_model)
        # Option 1: Use the single memory token (sequence length 1)
        memory = memory.unsqueeze(0)   # (1, batch_size, d_model)

        # Pass through the transformer decoder.
        # The transformer decoder uses cross-attention between queries and memory.
        decoded = self.transformer_decoder(queries, memory)  # (num_points, batch_size, d_model)

        # Permute to (batch_size, num_points, d_model)
        decoded = decoded.permute(1, 0, 2)
        out = self.out_proj(decoded)  # (batch_size, num_points, 3)
        return out