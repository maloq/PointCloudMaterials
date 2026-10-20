"""Invariant measurement decoder retaining pair relations in vector latents."""
import math
import torch
from torch import nn
from .history_model import HistoryProcess


class AtomAttention(nn.Module):
    def __init__(self, width=64, heads=4):
        super().__init__()
        self.width, self.heads = width, heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3*width)
        self.pair_weight = nn.Parameter(torch.ones(heads)*.1)
        self.output = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2*width), nn.SiLU(), nn.Linear(2*width, width))

    def forward(self, x, pair):
        b, n, _ = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, self.heads, self.width//self.heads).unbind(2)
        q, k, v = [a.transpose(1, 2) for a in (q, k, v)]
        weights = (q @ k.transpose(-1, -2))/math.sqrt(self.width/self.heads)
        weights = (weights+self.pair_weight[None, :, None, None]*pair[:, None]).softmax(-1)
        x = x+self.output((weights @ v).transpose(1, 2).reshape(b, n, self.width))
        return x+self.ff(x)


class MeasurementHead(nn.Module):
    def __init__(self, channels=8, width=64):
        super().__init__()
        a, b = torch.triu_indices(channels, channels)
        self.register_buffer('a', a); self.register_buffer('b', b)
        self.input = nn.Linear(len(a), width)
        self.layers = nn.ModuleList([AtomAttention(width) for _ in range(2)])
        self.pool = nn.Linear(width, 1)
        self.output = nn.Sequential(nn.Linear(3*width, 128), nn.SiLU(), nn.Linear(128, 244))

    def forward(self, z):
        gram = z @ z.transpose(-1, -2)
        x = self.input(torch.asinh(gram[..., self.a, self.b]))
        # Dot products across atoms retain angular relations discarded by atomwise Gram pooling.
        flat = z.flatten(-2)
        pair = torch.asinh((flat @ flat.transpose(-1, -2))/z.shape[-2])
        for layer in self.layers:
            x = layer(x, pair)
        pooled = (self.pool(x).softmax(1)*x).sum(1)
        return self.output(torch.cat((x[:, 0], x[:, 1:].mean(1), pooled), -1))


class MeasuredHistoryProcess(HistoryProcess):
    def __init__(self, c, arm):
        super().__init__(c, arm)
        self.measurement = MeasurementHead(c['model']['channels'], c['measurement_head']['width'])
        self.fit_update = 0

    def measured_path(self, bound, initial, steps):
        _, codes = bound(initial, steps, latents=True)
        return self.measurement(codes.flatten(0, 1)).reshape(len(initial), steps, 244)
