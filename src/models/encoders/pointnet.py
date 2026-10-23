from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x.view(-1, 3, 3) + torch.eye(3, device=x.device, dtype=x.dtype)


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
        identity = torch.eye(self.k, device=x.device, dtype=x.dtype)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 512)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        return x.view(-1, self.k, self.k) + identity



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
        trans = self.stn(x)

        x = x.transpose(2, 1)                       # (B,N,C)
        x = torch.bmm(x, trans)
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
    input_layout = "b3n"
    output_contract = "invariant"
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, _, _ = self.backbone(x)
        return self.mlp(feat)


# ---------------------------------------------------------------------------
#  Small variant  – widths = (64,128,256,512)
# ---------------------------------------------------------------------------
@register_encoder("PnE_S")
class PointNetEncoderSmall(Encoder):
    input_layout = "b3n"
    output_contract = "invariant"
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
        feat, _, _ = self.backbone(x)
        return self.mlp(feat)
