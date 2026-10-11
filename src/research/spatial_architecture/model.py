"""Keep heads and the historical MACE implementation fixed across architectures."""
import torch
from torch import nn
from src.models.encoders.spatial_esen import SpatialESEN
from src.models.encoders.spatial_tensornet import SpatialTensorNet
from src.research.structural_response.model import Geometry
from src.research.structural_response.data import spatial_target_names


class Spatial(nn.Module):
    def __init__(self, c, architecture):
        super().__init__()
        if architecture == 'esen':
            self.geometry = SpatialESEN(**c['encoder'])
        elif architecture == 'tensornet':
            self.geometry = SpatialTensorNet(**c['encoder'])
        elif architecture == 'mace':
            self.geometry = Geometry(dict(c, encoder=c['reference_encoder']))
        else:
            raise ValueError(f'Unknown spatial architecture: {architecture}')
        self.descriptors = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, len(spatial_target_names(c))))
        self.masked_predictor = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, 128))


def teacher(c, architecture, geometry):
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        target = Spatial(c, architecture).geometry.cuda()
    target.load_state_dict(geometry.state_dict())
    return target.eval().requires_grad_(False)
