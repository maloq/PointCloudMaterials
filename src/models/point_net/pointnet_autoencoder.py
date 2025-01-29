from pytorch3d.loss import chamfer_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.point_net.pointnet_cls import PointNetEncoder, STNkd, STN3d



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



class MLPEncoder(nn.Module):
    def __init__(self, point_size, latent_size):
        super(MLPEncoder, self).__init__()
        self.point_size = point_size
        self.latent_size = latent_size
        
        self.enc1 = nn.Linear(self.point_size * 3, 512)
        self.enc2 = nn.Linear(512, 256)
        self.enc3 = nn.Linear(256, self.latent_size)
        
    def forward(self, x):
        x = x.view(-1, self.point_size * 3)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = self.enc3(x)
        return x



class MLPDecoder(nn.Module):
    def __init__(self, point_size, latent_size):
        super(MLPDecoder, self).__init__()
        self.point_size = point_size
        self.latent_size = latent_size
        
        self.dec1 = nn.Linear(self.latent_size, 512)
        self.dec2 = nn.Linear(512, 256)
        self.dec3 = nn.Linear(256, self.point_size * 3)
        
    def forward(self, x):
        x = F.relu(self.dec1(x))
        x = F.relu(self.dec2(x))
        x = self.dec3(x)
        return x.view(-1, self.point_size, 3)



class MLP_AE(nn.Module):
    def __init__(self, point_size, latent_size):
        super(MLP_AE, self).__init__()
        self.encoder = MLPEncoder(point_size, latent_size)
        self.decoder = MLPDecoder(point_size, latent_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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