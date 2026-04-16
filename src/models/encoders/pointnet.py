from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .base import Encoder
from .registry import register_encoder


class STN3d(nn.Module):
    def __init__(self, channel):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 512, 1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(128)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x_orig_device = x.device
        x_orig_dtype = x.dtype
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
        #     batchsize, 1)
        # if x.is_cuda: # x is on x_orig_device
        #     iden = iden.cuda()
        
        # Modernized identity matrix creation
        iden_matrix = torch.eye(self.k, device=x_orig_device, dtype=x_orig_dtype)
        iden = iden_matrix.view(1, self.k * self.k).repeat(batchsize, 1)
        
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x



class _PointNetBackbone(nn.Module):
    def __init__(self, *, channel: int, feature_transform: bool, widths: tuple[int, int, int, int], dropout: float):
        super().__init__()
        c1, c2, c3, c4 = widths
        self.stn   = STN3d(channel)
        self.conv1 = nn.Conv1d(channel, c1, 1)
        self.conv2 = nn.Conv1d(c1,      c2, 1)
        self.conv3 = nn.Conv1d(c2,      c3, 1)
        self.conv4 = nn.Conv1d(c3,      c4, 1)
        self.bn1, self.bn2 = nn.BatchNorm1d(c1), nn.BatchNorm1d(c2)
        self.bn3, self.bn4 = nn.BatchNorm1d(c3), nn.BatchNorm1d(c4)
        self.drop1 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.feature_transform = feature_transform
        if feature_transform:
            self.fstn = STNkd(k=c1)

    def forward(self, x: torch.Tensor):
        B, D, N = x.size()
        # B, 3, N for point clouds
        assert D == 3, "PointNet encoder expects 3 channels"
        trans = self.stn(x)

        x = x.transpose(2, 1)                       # (B,N,C)
        if D > 3:                                   # extra channels?
            other = x[:, :, 3:]
            x     = x[:, :, :3]
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, other], dim=2)
        x = x.transpose(2, 1)                       # (B,C,N)

        x = F.relu(self.bn1(self.conv1(x)))
        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = torch.bmm(x.transpose(2, 1), trans_feat).transpose(2, 1)
        else:
            trans_feat = None

        x = self.drop1(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(F.relu(self.bn4(self.conv4(x))))
        x = torch.max(x, 2, keepdim=False)[0]       # global max‑pool → (B,C4)
        return x, trans, trans_feat


# ---------------------------------------------------------------------------
#  Big (default) variant  – widths = (96,256,512,512)
# ---------------------------------------------------------------------------
@register_encoder("PnE_L")
class PointNetEncoder(Encoder):
    expects_channel_first = True
    def __init__(
        self,
        latent_size: int = 128,
        *,
        channel: int = 3,
        feature_transform: bool = True,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.invariant_dim = int(latent_size)
        self.backbone = _PointNetBackbone(
            channel=channel,
            feature_transform=feature_transform,
            widths=(96, 256, 512, 512),
            dropout=dropout_rate,
        )

        self.mlp = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, latent_size),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        feat, trans, trans_feat = self.backbone(x)
        latent = self.mlp(feat)
        return latent, trans, trans_feat


# ---------------------------------------------------------------------------
#  Small variant  – widths = (64,128,256,512)
# ---------------------------------------------------------------------------
@register_encoder("PnE_S")
class PointNetEncoderSmall(Encoder):
    expects_channel_first = True
    def __init__(
        self,
        latent_size: int = 128,
        *,
        channel: int = 3,
        feature_transform: bool = False,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.invariant_dim = int(latent_size)
        self.backbone = _PointNetBackbone(
            channel=channel,
            feature_transform=feature_transform,
            widths=(64, 128, 256, 512),
            dropout=dropout_rate,
        )
        self.mlp = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, latent_size),
        )

    def forward(self, x: torch.Tensor):
        feat, trans, trans_feat = self.backbone(x)
        latent = self.mlp(feat)
        return latent, trans, trans_feat
