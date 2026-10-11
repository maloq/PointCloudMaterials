"""CUDA topology and exact differentiable MEAM energy/force derivatives."""
import functools
import hashlib
import math
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from src.research.response_meam.potential import Lee2003Al,RC,EBOUND
from .energy import energy


@functools.lru_cache(maxsize=1)
def extension():
    source=Path(__file__).with_name('topology.cu')
    identity=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load('pcm_meam_topology_'+identity,[str(source)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)


class Topology:
    def __init__(self,atoms,skin,box):
        self.atoms,self.skin,self.box=atoms,skin,box
        self.cutoff=RC*math.sqrt(EBOUND)+skin
        self.reference=None;self.rebuilds=0;self.native=extension()

    @torch.no_grad()
    def __call__(self,q):
        x=q.detach().reshape(-1,3).contiguous()
        if len(x)%self.atoms:raise ValueError('Whole disconnected replicas required')
        if self.reference is None or self.reference.shape!=x.shape:
            rebuild=True
        else:
            d=x-self.reference
            if self.box is not None:d=d-self.box*torch.round(d/self.box)
            rebuild=bool(2*d.norm(dim=-1).max()>=self.skin)
        if rebuild:self.build(x)
        self.counts.zero_()
        self.native.screening(x,self.length,self.box is not None,self.neighbors,self.nneigh,
            self.pair_i,self.pair_j,self.npairs,self.out_i,self.out_j,self.out_pair,self.out_k,self.counts)
        pairs,triplets,error=self.counts.tolist()
        if error:raise RuntimeError(f'MEAM screening capacity exceeded: pairs={pairs}, triplets={triplets}; no truncation applied')
        return self.out_i[:pairs],self.out_j[:pairs],self.out_pair[:triplets],self.out_k[:triplets]

    def build(self,x):
        batches=len(x)//self.atoms
        if self.box is not None:
            origin=torch.zeros(3,device=x.device,dtype=x.dtype);length=self.box.to(x)
            grid=(length/self.cutoff).floor().long().tolist()
            if min(grid)<3:raise ValueError('Periodic cell requires at least three neighbor bins per axis')
        else:
            origin=x.amin(0)-self.skin
            extent=x.amax(0)-origin+self.skin
            grid=(extent/self.cutoff).floor().long().add(1).tolist()
            length=torch.ones(3,device=x.device,dtype=x.dtype)
        integer=dict(device=x.device,dtype=torch.int32)
        index=dict(device=x.device,dtype=torch.int64)
        cells=math.prod(grid)*batches
        bin_counts=torch.zeros(cells,**integer);bins=torch.empty(cells,128,**integer)
        atom_cells=torch.empty(len(x),3,**integer)
        self.neighbors=torch.empty(len(x),128,**integer);self.nneigh=torch.empty(len(x),**integer)
        capacity=len(x)*64
        self.pair_i=torch.empty(capacity,**integer);self.pair_j=torch.empty(capacity,**integer)
        pair_count=torch.zeros(1,**integer);error=torch.zeros(1,**integer)
        self.native.build_neighbors(x,origin,length,*grid,self.atoms,self.cutoff,self.box is not None,
            bin_counts,bins,atom_cells,self.neighbors,self.nneigh,self.pair_i,self.pair_j,pair_count,error)
        if int(error):raise RuntimeError(f'MEAM neighbor capacity/bounds error {int(error)}; bin capacity128, neighbor capacity128')
        self.npairs=int(pair_count)
        self.out_i=torch.empty(capacity,**index);self.out_j=torch.empty(capacity,**index)
        self.out_pair=torch.empty(capacity*8,**index);self.out_k=torch.empty(capacity*8,**index)
        self.counts=torch.zeros(3,**integer);self.length=length
        self.reference=x.clone();self.rebuilds+=1


class FastAl(Lee2003Al):
    """Open replicas or one periodic cell, with compiled analytic force/HVPs.

    q may remain float64 for integration while the declared force dtype is
    float32. Topology uses the same rounded coordinates as the energy. Forces
    and tangents come from AD through that energy, not finite differences.
    """
    def __init__(self,*args,box=None,compile_tangents=True,**kwargs):
        super().__init__(*args,**kwargs)
        self.register_buffer('periodic_box',None if box is None else torch.as_tensor(box,device=self.table.device,dtype=self.table.dtype))
        self.topology=Topology(self.atoms,self.skin,self.periodic_box)
        self.gradient=torch.func.grad_and_value(self.energy_with_support)
        self.compiled_gradient=torch.compile(self.gradient,dynamic=True,fullgraph=True)
        self.compiled_tangent=torch.compile(self._force_tangent,dynamic=True,fullgraph=True) if compile_tangents else self._force_tangent

    def support(self,q):return self.topology(q.to(self.table.dtype))

    def energy_with_support(self,q,support):return energy(self,q,support)

    def forward(self,q):return self.energy_with_support(q,self.support(q))

    def force(self,q):
        grad,e=self.compiled_gradient(q,self.support(q))
        return e.detach(),-grad.detach()

    def _force_tangent(self,q,dq,support):
        def force(x):return -self.gradient(x,support)[0]
        # The force Jacobian is minus a symmetric energy Hessian. torch.func
        # differentiates the continuous energy; topology stays fixed locally.
        force_value,pullback=torch.func.vjp(force,q)
        columns=[pullback(dq[...,j].contiguous())[0] for j in range(dq.shape[-1])]
        tangent=torch.stack(columns,-1) if columns else q.new_empty(*q.shape,0)
        return force_value,tangent

    def force_tangent(self,q,dq):
        f,h=self.compiled_tangent(q,dq,self.support(q))
        return f.detach(),h.detach()

    def virial_trace(self,q):
        scale=q.new_ones((),requires_grad=True)
        e=energy(self,q,self.support(q),scale)
        return -torch.autograd.grad(e,scale)[0].detach()
