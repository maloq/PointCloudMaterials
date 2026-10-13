"""Equivariant encoder/decoder attention to fixed-identity observed history."""
import math
import torch
from torch import nn
from .model import PhaseProcess


class TemporalBlock(nn.Module):
    """Scalar invariant attention weights mix equivariant vector values."""
    def __init__(self, channels, width, heads=4):
        super().__init__()
        self.channels, self.heads, self.width = channels, heads, width
        a, b = torch.triu_indices(channels, channels)
        self.register_buffer('a', a); self.register_buffer('b', b)
        self.query = nn.Linear(len(a), width)
        self.key = nn.Linear(len(a), width)
        self.value = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)
        self.slot_bias = nn.Parameter(torch.zeros(heads, 4))
        self.ff = nn.Sequential(nn.Linear(len(a), width), nn.SiLU(), nn.Linear(width, channels*channels))

    def scalars(self, z):
        return torch.asinh((z @ z.transpose(-1, -2))[..., self.a, self.b])

    def prepare(self, memory):
        tokens = memory.transpose(1, 2)
        batch, atoms, frames = tokens.shape[:3]
        keys = self.key(self.scalars(tokens)).reshape(batch, atoms, frames, self.heads, self.width//self.heads)
        values = self.value(tokens.transpose(-1, -2)).transpose(-1, -2)
        values = values.reshape(batch, atoms, frames, self.heads, self.channels//self.heads, 3)
        return keys, values

    def forward(self, z, prepared):
        batch, atoms = z.shape[:2]
        past_key, past_value = prepared
        repeats = batch//len(past_key)
        past_key = past_key.repeat_interleave(repeats, dim=0)
        past_value = past_value.repeat_interleave(repeats, dim=0)
        scalar = self.scalars(z)
        q = self.query(scalar).reshape(batch, atoms, self.heads, -1)
        k = torch.cat((past_key, self.key(scalar).reshape(batch, atoms, 1, self.heads, -1)), 2)
        frames = k.shape[2]
        logits = torch.einsum('bnhd,bnthd->bnht', q, k)/math.sqrt(self.width/self.heads)
        weights = (logits+self.slot_bias[:, -frames:]).softmax(-1)
        current = self.value(z.transpose(-1, -2)).transpose(-1, -2)
        values = torch.cat((past_value, current.reshape(batch, atoms, 1, self.heads, self.channels//self.heads, 3)), 2)
        context = torch.einsum('bnht,bnthdc->bnhdc', weights, values).reshape(batch, atoms, self.channels, 3)
        z = z+.1*self.output(context.transpose(-1, -2)).transpose(-1, -2)
        mix = self.ff(self.scalars(z)).reshape(batch, atoms, self.channels, self.channels)
        return z+.1*torch.tanh(mix) @ z


class HistoryProcess(nn.Module):
    def __init__(self, c, arm):
        super().__init__()
        self.process = PhaseProcess(c, dict(arm, model='latent'))
        self.frames = arm['history_frames']
        self.architecture = arm['architecture']
        if self.architecture == 'attention':
            channels, width = c['model']['channels'], c['model']['width']
            self.encoder_attention = nn.ModuleList([TemporalBlock(channels, width) for _ in range(2)])
            self.decoder_attention = nn.ModuleList([TemporalBlock(channels, width) for _ in range(2)])
            self.readout_mix = nn.Sequential(nn.Linear(channels*(channels+1)//2, width),
                nn.SiLU(), nn.Linear(width, 2*channels))
            nn.init.zeros_(self.readout_mix[-1].weight); nn.init.zeros_(self.readout_mix[-1].bias)

    def condition(self, history):
        if self.architecture == 'classic':
            return self.process
        batch = len(history)
        past = history[:, -self.frames:-1]
        if self.frames > 1:
            memory = self.process.encode(past.flatten(0, 1)).reshape(batch, self.frames-1, 80, -1, 3)
        else:
            memory = history.new_empty(batch, 0, 80, self.process.field.channels, 3)
        return BoundProcess(self, memory)


class BoundProcess:
    """One anchor's observed past stays fixed throughout its autonomous future.

    Reencoding true future states for training uses this same fixed context.
    Derivative queries perturb current Q,V only, matching the saved MD tangents.
    """
    latent = True

    def __init__(self, model, memory):
        self.model, self.memory = model, memory
        self.encoder_memory = [layer.prepare(memory) for layer in model.encoder_attention]
        self.decoder_memory = [layer.prepare(memory) for layer in model.decoder_attention]

    def attend(self, z, layers, prepared):
        for layer, tokens in zip(layers, prepared):
            z = layer(z, tokens)
        return z

    def encode(self, h):
        z = self.model.process.encode(h)
        return self.attend(z, self.model.encoder_attention, self.encoder_memory)

    def decode(self, z):
        z = self.attend(z, self.model.decoder_attention, self.decoder_memory)
        h = self.model.process.readout(z.transpose(-1, -2)).transpose(-1, -2)
        descriptor = self.model.decoder_attention[-1].scalars(z)
        weights = self.model.readout_mix(descriptor).reshape(*z.shape[:2], 2, z.shape[-2])
        h = h+.1*torch.tanh(weights) @ z
        return h-h[:, :1]

    def step(self, z, h=None):
        h = self.decode(z) if h is None else h
        ballistic = torch.stack((.01*h[..., 1, :], torch.zeros_like(h[..., 1, :])), -2)
        return z+self.model.process.linear_encode(ballistic)+.05*self.model.process.field(z, h)

    def __call__(self, initial, steps, *, latents=False):
        z = self.encode(initial); decoded0 = self.decode(z)
        anchor = initial.reshape(-1, 80, 2, 3); values = []; codes = []; decoded = decoded0
        for _ in range(steps):
            z = self.step(z, decoded); decoded = self.decode(z)
            values.append(anchor+decoded-decoded0); codes.append(z)
        value = torch.stack(values, 1).flatten(-2)
        return (value, torch.stack(codes, 1)) if latents else value
