"""Causal state models on frozen smooth atomic densities."""
import numpy as np
import torch
from torch import nn


class SmoothTemporalState(nn.Module):
    def __init__(self, encoder, mean, std, *, mode, hidden_dim=64, state_dim=32, horizons=4):
        super().__init__()
        if mode not in ("gru", "gated_no_transport", "gated_kabsch", "gated_smooth"):
            raise ValueError(f"Unknown temporal state model: {mode}")
        self.mode = mode
        self.encoder = encoder.requires_grad_(False)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.max_ell = encoder.density.max_ell
        self.radial = encoder.density.radial_channels
        gate_dim = self.radial*(self.max_ell+1)
        self.gate = nn.Sequential(nn.Linear(256+gate_dim, 64), nn.SiLU(), nn.Linear(64, gate_dim))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, np.log(np.exp(-.1/.5)/(1-np.exp(-.1/.5))))
        if mode == "gru":
            self.gate.requires_grad_(False)
        self.fuse = nn.Sequential(nn.Linear(256, 128), nn.SiLU())
        self.recurrent = nn.GRU(128, hidden_dim, batch_first=True)
        self.state_head = nn.Linear(hidden_dim, state_dim)
        self.future_head = nn.Sequential(nn.Linear(state_dim+hidden_dim+3, 128), nn.SiLU(), nn.Linear(128, horizons*32))
        self.current_head = nn.Linear(state_dim, 8)

    def forward(self, q, raw, matrices, material):
        # q: B,H,R,(L+1)^2; raw: standardized frozen spatial features B,H,128.
        if self.mode == "gru":
            memory_features = raw
        else:
            memory = q[:, 0]
            features = [raw[:, 0]]
            for k in range(1, q.shape[1]):
                if self.mode != "gated_no_transport":
                    memory = torch.cat([memory[..., ell*ell:(ell+1)**2] @ block[:, k].transpose(-1, -2)
                                        for ell,block in enumerate(matrices)], -1)
                encoded = (self.encoder.forward_moments(memory)-self.mean)/self.std
                difference = q[:, k]-memory
                innovation = torch.stack([difference[..., ell*ell:(ell+1)**2].square().mean(-1)
                                          for ell in range(self.max_ell+1)], -1)
                innovation = innovation/self.encoder.moment_scale[None, :, None].square()
                gates = torch.sigmoid(self.gate(torch.cat((raw[:, k], encoded, innovation.flatten(1)), 1)))
                gates = gates.reshape(len(q), self.radial, self.max_ell+1)
                expanded = torch.cat([gates[..., ell:ell+1].expand(-1, -1, 2*ell+1) for ell in range(self.max_ell+1)], -1)
                memory = expanded*memory+(1-expanded)*q[:, k]
                features.append((self.encoder.forward_moments(memory)-self.mean)/self.std)
            memory_features = torch.stack(features, 1)
        recurrent_input = self.fuse(torch.cat((raw, memory_features), -1))
        hidden, _ = self.recurrent(recurrent_input)
        states = self.state_head(hidden)
        conditions = torch.nn.functional.one_hot(material, 3).to(raw.dtype)
        future = self.future_head(torch.cat((states[:, -1], hidden[:, -1], conditions), -1)).reshape(len(q), -1, 32)
        current = self.current_head(states[:, -1])
        return future, states, current
