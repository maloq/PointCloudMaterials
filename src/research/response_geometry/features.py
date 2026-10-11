"""Three future-only targets from exactly the same trajectories and tangents."""
import math
import torch
from e3nn import o3
from src.research.local_response.oracle import LocalFeatures, LocalOracle, OpenPotential
from src.research.local_response.data import environment


def observed(q):
    x=q.reshape(len(q),-1,3);v=x[:,1:]-x[:,:1];distance=v.norm(dim=-1)
    r=distance/5.5;w=(1-r).clamp_min(0).pow(4)*(1+4*r)
    centers=torch.linspace(1.5,5.25,12,device=q.device,dtype=q.dtype)
    radial=(torch.exp(-.5*((distance[...,None]-centers)/.35).square())*w[...,None]).sum(1)
    spherical=o3.spherical_harmonics(list(range(1,7)),v,normalize=True,normalization='component')
    powers=[]
    for center in [2.7,4.5]:
        weight=w*torch.exp(-.5*((distance-center)/.65).square())
        means=(spherical*weight[...,None]).sum(1)/weight.sum(1).clamp_min(1e-8)[:,None]
        powers.extend(block.square().mean(-1)for block in means.split([3,5,7,9,11,13],dim=-1))
    return torch.cat([radial,torch.stack(powers,-1)],-1)


class Features:
    def __init__(self,config,normalizer,dtype):
        self.old=LocalFeatures(config,dtype)
        self.mean=normalizer['mean'].to(device='cuda',dtype=dtype)
        self.scale=normalizer['scale'].to(device='cuda',dtype=dtype)
        generator=torch.Generator(device='cuda').manual_seed(202610031)
        self.maps={}
        for mode,dim in [('rich_delta',24),('rich_joint',48)]:
            self.maps[mode]=[(torch.randn(dim*(h+1),128,generator=generator,device='cuda',dtype=torch.float64).to(dtype)/math.sqrt(dim*(h+1)),
                (2*math.pi*torch.rand(128,generator=generator,device='cuda',dtype=torch.float64)).to(dtype))for h in range(2)]

    def __call__(self,initial,path):
        anchor=observed(initial)
        endpoint=torch.stack([observed(path[:,h])for h in range(2)],1)
        delta=(endpoint-anchor[:,None])/self.scale
        absolute=(endpoint-self.mean)/self.scale
        outputs=[self.old(initial,path)]
        for mode,coordinates in [('rich_delta',delta),('rich_joint',torch.cat([absolute,delta],-1))]:
            outputs.append(torch.cat([math.sqrt(2/128)*torch.cos(coordinates[:,:h+1].flatten(1)@omega+phase)
                for h,(omega,phase)in enumerate(self.maps[mode])],-1))
        return torch.cat(outputs,-1)


def make(config,state,radius,teacher,normalizer):
    dtype=next(teacher.model.parameters()).dtype
    q,basis,rows=environment(state,radius,dtype)
    return LocalOracle(OpenPotential(teacher,len(rows),config['skin_A']),
        Features(config,normalizer,dtype),config,rows,state['original_atoms']),q,basis
