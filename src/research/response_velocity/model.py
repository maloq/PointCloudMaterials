"""Capacity-matched invariant velocity channels on the existing MACE graph."""
import torch
from torch import nn
from src.research.local_response.model import Predictor as GeometryPredictor


class Predictor(GeometryPredictor):
    def __init__(self,c,use_velocity):
        super().__init__(c)
        self.use_velocity=use_velocity
        self.velocity_scale=c['velocity_scale_A_fs']
        # This module exists in BOTH arms; Q-only replaces velocities by zeros.
        self.encoder.velocity_projection=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,6))
        nn.init.normal_(self.encoder.velocity_projection[-1].weight,std=.001)
        nn.init.zeros_(self.encoder.velocity_projection[-1].bias)
        self.head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,128*len(c['horizons_fs'])))
        nn.init.normal_(self.head[-1].weight,std=.001);nn.init.zeros_(self.head[-1].bias)

    def graph(self,state):
        if state.shape[1:] != (80,6):raise ValueError(f'Expected [B,80,xyz+velocity]: {state.shape}')
        q=state[...,:3];v=state[...,3:]
        v=(v-v[:,:1])/self.velocity_scale if self.use_velocity else torch.zeros_like(v)
        graph=super().graph(q);i,j=graph['edge'];flat=v.flatten(0,1)
        delta=q.flatten(0,1)[j]-q.flatten(0,1)[i];r=delta.norm(dim=-1);unit=delta/r[:,None]
        a,b=flat[i],flat[j]
        invariants=torch.stack((a.square().sum(-1),b.square().sum(-1),(a*b).sum(-1),
            (a*unit).sum(-1),(b*unit).sum(-1)),-1)
        taper=(1-r/self.encoder.cutoff).clamp_min(0).pow(4)*(1+4*r/self.encoder.cutoff)
        graph['radial']=graph['radial']+self.encoder.velocity_projection(invariants)*taper[:,None]
        return graph


def derivatives(model,x,basis,create_graph=False):
    return torch.stack([torch.autograd.functional.jvp(model,x,basis[...,j],
        create_graph=create_graph,strict=True)[1] for j in range(basis.shape[-1])],-1)


@torch.no_grad()
def initialize(model,x,micro):
    pooled=torch.cat([model.encoder.pooled_graph(model.graph(b)) for b in x.split(micro)])
    model.encoder.pooled_mean.copy_(pooled.mean(0))
    model.encoder.pooled_scale.copy_(pooled.std(0,unbiased=False).clamp_min(1e-5))
