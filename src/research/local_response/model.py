"""Center-aware local student; live geometry supports response-only gradients."""
import torch
from torch import nn
from src.research.supervised_onset.model import CapacityEncoder
from src.research.response_training.model import initialize, responses


class Predictor(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.encoder=CapacityEncoder(d0=2.8,n_ref=80.,radius=8.,**c['encoder'])
        self.head=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,256))
        nn.init.normal_(self.head[-1].weight,std=.001);nn.init.zeros_(self.head[-1].bias)

    def graph(self, positions):
        batch, atoms, xyz=positions.shape
        if (atoms,xyz)!=(80,3):raise ValueError(f'Expected local80 geometry: {positions.shape}')
        positions=positions-positions[:,:1]
        with torch.no_grad():
            delta=positions[:,None]-positions[:,:,None]
            valid=positions.norm(dim=-1)<8
            allowed=(delta.norm(dim=-1)<self.encoder.cutoff)&valid[:,:,None]&valid[:,None,:]
            allowed &= ~torch.eye(atoms,dtype=torch.bool,device=positions.device)[None]
            b,i,j=allowed.nonzero(as_tuple=True)
        x=positions.flatten(0,1);edge=torch.stack((b*atoms+i,b*atoms+j));vectors=x[edge[1]]-x[edge[0]]
        attrs=x.new_ones(len(x),1)
        radial,cutoff=self.encoder.radial_embedding(vectors.norm(dim=-1,keepdim=True),attrs,edge,self.encoder.atomic_numbers)
        if cutoff is not None:raise ValueError('Expected smooth radial-embedded cutoff')
        # Smooth support taper; avoid the norm derivative at the fixed center.
        r=(positions.square().sum(-1)+torch.nn.functional.one_hot(
            torch.zeros(batch,dtype=torch.long,device=x.device),atoms).to(x.dtype)).sqrt()/8
        weight=(1-r).clamp_min(0).pow(4)*(1+4*r)
        weight=torch.cat((weight.new_ones(batch,1),weight[:,1:]),1).flatten()
        center=torch.zeros_like(attrs);center[torch.arange(batch,device=x.device)*atoms]=1
        return dict(attrs=attrs,center=center,weight=weight,edge=edge,
            angular=self.encoder.spherical_harmonics(vectors),radial=radial,
            group=torch.arange(batch,device=x.device).repeat_interleave(atoms),
            centers=torch.arange(batch,device=x.device)*atoms,size=batch)

    def encode(self,q):return self.encoder(self.graph(q))

    def forward(self,q):return self.head(self.encode(q))
