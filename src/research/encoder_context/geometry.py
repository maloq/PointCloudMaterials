"""Vectorized, on-device graphs for fixed 80-candidate physical patches."""
import torch
from e3nn import o3
from src.training_methods.bcr.data import taper


@torch.no_grad()
def graph(positions,encoder):
    batch,atoms,_=positions.shape
    radius=positions.norm(dim=-1);valid=radius<encoder.radius
    delta=positions[:,:,None]-positions[:,None,:]
    distance=delta.norm(dim=-1)
    allowed=(distance<encoder.cutoff)&valid[:,:,None]&valid[:,None,:]
    allowed&=~torch.eye(atoms,device=positions.device,dtype=torch.bool)[None]
    b,i,j=allowed.nonzero(as_tuple=True)
    edge=torch.stack((b*atoms+i,b*atoms+j))
    x=positions.flatten(0,1);attrs=valid.flatten().float()[:,None]
    weight=taper(radius.flatten(),encoder.radius)*attrs[:,0]
    center=torch.zeros_like(attrs);centers=torch.arange(batch,device=x.device)*atoms
    center[centers]=1
    vectors=x[edge[1]]-x[edge[0]]
    radial,cutoff=encoder.radial_embedding(vectors.norm(dim=-1,keepdim=True),attrs,edge,encoder.atomic_numbers)
    if cutoff is not None:raise ValueError('Expected radial-embedded cutoff')
    radial=radial*(weight[edge[0]]*weight[edge[1]])[:,None]
    return dict(attrs=attrs,center=center,weight=weight,edge=edge,
        angular=encoder.spherical_harmonics(vectors),radial=radial,cutoff=None,
        group=torch.arange(batch,device=x.device).repeat_interleave(atoms),centers=centers,size=batch)


@torch.no_grad()
def physical_targets(positions):
    """24 smooth radial counts, two weighted counts and six bond-order powers."""
    r=positions.norm(dim=-1)
    present=torch.ones_like(r);present[:,0]=0
    support=taper(r,8.)*present
    centers=torch.linspace(.5,7.5,24,device=r.device)
    radial=(torch.exp(-.5*((r[:,:,None]-centers)/.3)**2)*support[:,:,None]).sum(1)/80
    weights=torch.stack((taper(r,5.),taper(r,8.)),-1)*present[:,:,None]
    counts=weights.sum(1)/80
    power=[]
    for degree in (2,4,6):
        y=o3.spherical_harmonics(degree,positions,normalize=True,normalization='component')
        field=torch.einsum('bnr,bnm->brm',weights,y)/weights.sum(1).clamp_min(1e-6)[:,:,None]
        power.append(field.square().mean(-1))
    return torch.cat((radial,counts,*power),-1)
