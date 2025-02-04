import torch.nn as nn
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