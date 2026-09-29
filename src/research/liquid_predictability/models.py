"""Matched distance mixtures and a distance-only equivariant context readout."""
import math
import torch
from torch import nn
from src.research.crystal_vector.model import JointCrystalVector
from src.research.equivariant_context.model import geometry,radial_basis
from src.research.spatial_distance.model import parameters,component_cdf

class DescriptorMixture(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        self.register_buffer('mean',torch.zeros(224));self.register_buffer('scale',torch.ones(224))
        if kind=='prior':self.raw=nn.Parameter(torch.zeros(25,4))
        elif kind=='linear':self.net=nn.Linear(224,100)
        elif kind=='mlp':self.net=nn.Sequential(nn.Linear(224,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),nn.Linear(128,100))
        else:raise ValueError(kind)
    def forward(self,batch):
        n=len(batch['distance'])
        raw=self.raw[None].expand(n,-1,-1) if self.kind=='prior' else self.net((batch['features']-self.mean)/self.scale).reshape(n,25,4)
        return dict(parts=parameters(raw[...,:3].float(),raw[...,3].float()))

class DistanceMACE(JointCrystalVector):
    def forward(self,batch):
        if 'z' in batch:z,v=batch['z'],batch['v']
        else:
            z,v=self.encode(batch['positions']);z=z[batch['inverse']];v=v[batch['inverse']]
        g=geometry(batch['actual'],batch['actual']);s=self.stem(z)+self.geometry(g['node']);fields={1:v}
        for block in self.blocks:s,fields=block(s,fields,g)
        raw=self.distance_head(s).float();risk=component_cdf(parameters(raw,torch.zeros_like(raw[...,0])),raw.new_tensor([4,8,12,20,32,64]))
        scores=self.combine(torch.cat((risk,radial_basis(batch['actual'].norm(dim=-1),28.)),-1)).squeeze(-1).float()
        return dict(parts=parameters(raw,scores),z=z.float(),v=v.float(),state=s.float())

def initialize_descriptor(model,features,distance,weights):
    import numpy as np
    if model.kind!='prior':
        mean=weights@features;std=np.sqrt(weights@(features-mean)**2).clip(1e-4)
        model.mean.copy_(torch.as_tensor(mean,device=model.mean.device));model.scale.copy_(torch.as_tensor(std,device=model.scale.device))
    order=np.argsort(distance);quant=np.interp(np.linspace(.02,.98,25),np.cumsum(weights[order]),distance[order])
    raw=torch.zeros((25,4),device=model.mean.device);raw[:,0]=-12
    raw[:,1]=4*torch.atanh(torch.as_tensor((np.log(quant)-math.log(16))/4,device=raw.device).clamp(-.99,.99))
    raw[:,2]=math.log((.3-.15)/(2-.3))
    with torch.no_grad():
        if model.kind=='prior':model.raw.copy_(raw)
        else:
            last=model.net if model.kind=='linear' else model.net[-1]
            last.weight.mul_(.01);last.bias.copy_(raw.flatten())
