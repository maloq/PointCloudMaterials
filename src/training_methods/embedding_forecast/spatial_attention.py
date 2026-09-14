"""Learned neighbor attention with rotation-invariant relative-geometry summaries."""

import math
import torch
from torch import nn

from .context_mixture import ContextMixtureForecaster


class NeighborAttention(nn.Module):
    def __init__(self, dim, output_width, config):
        super().__init__()
        width, heads = config['width'], config['heads']
        if width % heads or config['blocks'] < 1 or config['radial_bins'] < 2:
            raise ValueError(f'Invalid spatial-attention dimensions: {config}')
        if config['precision'] not in ('float32','bfloat16'):
            raise ValueError(f'Invalid spatial-attention precision: {config}')
        self.config = config; self.heads = heads; self.head_width = width//heads
        bins = config['radial_bins']
        self.register_buffer('radial_centers', torch.linspace(0,config['radial_max_A'],bins))
        self.radial_spacing = config['radial_max_A']/(bins-1)
        self.features = nn.Sequential(nn.Linear(dim,width),nn.SiLU(),nn.Linear(width,width))
        self.radial = nn.Sequential(nn.Linear(bins,width),nn.SiLU(),nn.Linear(width,width))
        self.key = nn.Linear(width,width,bias=False)
        self.value = nn.Linear(width,width,bias=False)
        self.radial_bias = nn.Linear(bins,heads,bias=False)
        self.queries = nn.ModuleList([nn.Linear(output_width,width) for _ in range(config['blocks'])])
        self.outputs = nn.ModuleList([nn.Sequential(nn.Linear(width+heads*heads+2,width*2),
            nn.SiLU(),nn.Linear(width*2,output_width)) for _ in range(config['blocks'])])
        for output in self.outputs:
            nn.init.normal_(output[-1].weight,std=.001)
            nn.init.zeros_(output[-1].bias)

    def forward(self, tokens, history, neighbors, relative_A):
        if neighbors.shape[:-1] != relative_A.shape[:-1] or relative_A.shape[-1] != 3:
            raise ValueError('Spatial attention requires paired neighbor embeddings and relative xyz vectors.')
        b,t,k,_ = neighbors.shape
        distance = relative_A.norm(dim=-1)
        # Distinct cached atoms cannot coincide; this floor only defines the zero-vector limit in tests.
        direction = relative_A/distance.clamp_min(1e-8)[...,None]
        radial = torch.exp(-.5*((distance[...,None]-self.radial_centers)/self.radial_spacing).square())
        radii = torch.stack((distance.mean(-1),distance.amax(-1)),-1)/10.
        enabled = tokens.device.type == 'cuda' and self.config['precision'] == 'bfloat16'
        with torch.autocast(device_type=tokens.device.type,dtype=torch.bfloat16,enabled=enabled):
            encoded = self.features(neighbors-history[:,:,None])+self.radial(radial)
            key = self.key(encoded).reshape(b,t,k,self.heads,self.head_width)
            value = self.value(encoded).reshape(b,t,k,self.heads,self.head_width)
            bias = self.radial_bias(radial).permute(0,1,3,2).float()
            for query_layer, output_layer in zip(self.queries,self.outputs):
                query = query_layer(tokens).reshape(b,t,self.heads,self.head_width)
                logits = (key.float()*query[:,:,None].float()).sum(-1).permute(0,1,3,2)/math.sqrt(self.head_width)+bias
                weight = logits.softmax(-1)
                pooled = torch.einsum('bthk,btkhd->bthd',weight.to(value.dtype),value).flatten(-2)
                # Equivariant vector moments retain directional arrangement; their Gram matrix
                # is invariant to a joint rotation/reflection of all relative positions.
                vector = torch.einsum('bthk,btkd->bthd',weight,direction)
                gram = torch.einsum('bthd,btjd->bthj',vector.float(),vector.float()).flatten(-2)
                tokens = tokens+output_layer(torch.cat((pooled.float(),gram,radii),-1)).float()
        entropy = -(weight*weight.clamp_min(1e-12).log()).sum(-1)
        diagnostics = dict(spatial_attention_entropy=entropy.mean((1,2)).detach(),
            spatial_attention_effective_neighbors=entropy.exp().mean((1,2)).detach(),
            spatial_attention_max_weight=weight.amax(-1).mean((1,2)).detach(),
            spatial_attention_distance_A=(weight*distance[:,:,None]).sum(-1).mean((1,2)).detach())
        return tokens,diagnostics


class SpatialAttentionForecaster(ContextMixtureForecaster):
    def __init__(self, dim, history_steps, cadence_ps, horizons_ps, config):
        super().__init__(dim,history_steps,cadence_ps,horizons_ps,config)
        if not self.spatial_neighbors:
            raise ValueError('Learned spatial attention requires individual neighbors.')
        del self.spatial_input
        self.neighbor_attention = NeighborAttention(dim,config['width'],config['spatial_attention'])

    def forward(self, history, neighbors, relative_A):
        if neighbors.shape[2] != self.spatial_neighbors:
            raise ValueError('Spatial attention neighbor count differs from its fitted configuration.')
        delta = torch.cat((torch.zeros_like(history[:,:1]),history[:,1:]-history[:,:-1]),1)
        time = self.past_time[None,:,None].expand(len(history),-1,-1)
        tokens = self.input(torch.cat((history,delta,time),-1))
        tokens,diagnostics = self.neighbor_attention(tokens,history,neighbors,relative_A)
        return dict(self.decode_history(tokens,history),**diagnostics)
