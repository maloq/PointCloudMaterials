from pytorch3d.loss import chamfer_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.point_net.pointnet_utils import PointNetEncoder, feature_transform_reguliarzer



class DummyPointCloudAE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(DummyPointCloudAE, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size
        
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, self.latent_size, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(self.latent_size)
        
        self.dec1 = nn.Linear(self.latent_size,256)
        self.dec2 = nn.Linear(256,256)
        self.dec3 = nn.Linear(256,self.point_size*3)

    def encoder(self, x): 
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, self.latent_size)
        return x
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



class PointNetAE(nn.Module):
    def __init__(self, point_size, latent_size, normal_channel=False):
        super(PointNetAE, self).__init__()
        
        self.latent_size = latent_size
        self.point_size = point_size

        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.features_encoder = PointNetEncoder(global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.latent_size)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.2)
        self.dec1 = nn.Linear(self.latent_size,512)
        self.dec2 = nn.Linear(512,256)
        self.dec3 = nn.Linear(256,self.point_size*3)

    def encoder(self, x): 
        x, trans, trans_feat = self.features_encoder(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)  
        x = x.view(-1, self.latent_size)
        return x, trans_feat
    
    def decoder(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)
    
    def forward(self, x):
        x, trans_feat = self.encoder(x)
        x = self.decoder(x)
        return x, trans_feat