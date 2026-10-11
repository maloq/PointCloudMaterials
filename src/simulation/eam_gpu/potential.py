"""Fast pure-Ni EAM/FS force and analytic force-direction derivatives.

Float64 integration inputs/outputs with explicit float32 or float64 force
arithmetic. Scalar energy/virial reductions use float64. No autograd tape,
finite-difference production derivatives, screening, or angular intermediates.
"""
import functools
import hashlib
import math
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
from .tables import read


@functools.lru_cache(maxsize=1)
def extension():
    p=Path(__file__).with_name('kernels.cu')
    name='pcm_ni_eam_'+hashlib.sha256(p.read_bytes()).hexdigest()[:12]
    return load(name,[str(p)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=False)


class FastNi:
    """Fixed orthorhombic cell, or disconnected equal-size open replicas.

    force(q) -> (total energy, forces), force_tangent(q,dq) -> (forces,J*dq).
    q has shape (N,3) or (batch,3*N); dq appends a direction axis (1..4).
    Returned tensors own their storage. Box changes must use set_box().
    Only first force derivatives are exposed; arbitrary autograd is unsupported.
    """
    def __init__(self,file,atoms,*,box=None,dtype=torch.float32,device='cuda',skin=.6,
                 expected_sha256="acc8bd87f61dd67ecd8f72a6f181d6aafe52ca45b29fe6c15cec7c39c0bf76df",cuda_graph=True):
        if dtype not in (torch.float32,torch.float64):raise ValueError('Force dtype must be float32 or float64')
        if atoms<1 or skin<=0:raise ValueError('Positive atom count and Verlet skin required')
        self.data=read(file,expected_sha256);self.atoms=atoms;self.dtype=dtype;self.device=torch.device(device)
        self.skin=skin;self.cutoff=self.data.cutoff+skin;self.native=extension();self.cuda_graph=cuda_graph
        self.rho=torch.tensor(self.data.density,device=device,dtype=dtype)
        self.device=self.rho.device
        self.embed=torch.tensor(self.data.embedding,device=device,dtype=dtype)
        self.pair=torch.tensor(self.data.pair_r,device=device,dtype=dtype)
        self.box=None if box is None else torch.as_tensor(box,device=device,dtype=dtype).clone()
        if self.box is not None and (self.box.shape!=(3,) or not bool(torch.isfinite(self.box).all())):
            raise ValueError('Box must contain three finite orthorhombic lengths')
        self.reference=None;self.rebuilds=0;self.graphs={}

    def set_box(self,box):
        if self.box is None:raise ValueError('Cannot change an open system to periodic')
        value=torch.as_tensor(box,device=self.device,dtype=self.dtype)
        if value.shape!=(3,) or not bool(torch.isfinite(value).all()):raise ValueError('Invalid orthorhombic box')
        self.box.copy_(value);self.reference=None;self.graphs.clear()

    @torch.no_grad()
    def support(self,q):
        valid_shape=q.ndim==2 and (q.shape==(self.atoms,3) or q.shape[1]==3*self.atoms)
        if not valid_shape or not q.numel() or q.device!=self.device or q.dtype not in (torch.float32,torch.float64):
            raise ValueError('Whole float32/float64 Ni replicas on the model CUDA device required, shape (N,3) or (B,3*N)')
        x=q.detach().reshape(-1,3).to(self.dtype).contiguous()
        if not bool(torch.isfinite(x).all()):raise ValueError('Nonfinite coordinates')
        if self.reference is None or self.reference.shape!=x.shape:rebuild=True
        else:
            d=x-self.reference
            if self.box is not None:d-=self.box*torch.round(d/self.box)
            rebuild=bool(d.square().sum(-1).max()>=self.skin**2/4)
        if rebuild:self._build(x)
        return x

    def _build(self,x):
        batches=len(x)//self.atoms
        if self.box is not None:
            origin=torch.zeros(3,device=x.device,dtype=x.dtype);length=self.box
            grid=(length/self.cutoff).floor().long().tolist()
            if min(grid)<3:raise ValueError('Periodic lengths must each be >= 3*(cutoff+skin); multiple images unsupported')
        else:
            origin=x.amin(0)-self.skin;extent=x.amax(0)-origin+self.skin
            grid=(extent/self.cutoff).floor().long().add(1).tolist();length=torch.ones_like(origin)
        if self.reference is None or self.reference.shape!=x.shape:
            self.neighbors=torch.empty(len(x),256,device=x.device,dtype=torch.int32)
            self.count=torch.empty(len(x),device=x.device,dtype=torch.int32);self.graphs.clear()
        integer=dict(device=x.device,dtype=torch.int32);cells=math.prod(grid)*batches
        bins=torch.empty(cells,128,**integer);counts=torch.zeros(cells,**integer)
        atom_cells=torch.empty(len(x),3,**integer);error=torch.zeros(1,**integer)
        self.native.build_neighbors(x,origin,length,*grid,self.atoms,self.cutoff,self.box is not None,
                                    counts,bins,atom_cells,self.neighbors,self.count,error)
        code=int(error)
        if code:raise RuntimeError(f'Ni EAM neighbor error {code}; bin capacity128 / neighbor capacity256; no truncation accepted')
        self.length=length;self.reference=x.clone();self.rebuilds+=1

    def _buffers(self,x,dirs):
        # Rebuilds preserve neighbor pointers; graph caches are invalidated if shape changes.
        key=(len(x),dirs)
        if key in self.graphs:return self.graphs[key]
        b=dict(x=torch.empty_like(x),dq=torch.empty(len(x),3,dirs,device=x.device,dtype=x.dtype),
               fp=torch.empty(len(x),device=x.device,dtype=x.dtype),
               dfp=torch.empty(len(x),dirs,device=x.device,dtype=x.dtype),
               e=torch.empty(len(x),device=x.device,dtype=torch.float64),
               w=torch.empty(len(x),device=x.device,dtype=torch.float64),f=torch.empty_like(x),
               h=torch.empty(len(x),3,dirs,device=x.device,dtype=x.dtype),
               error=torch.zeros(1,device=x.device,dtype=torch.int32))
        b['x'].copy_(x);b['dq'].zero_()
        def evaluate():
            b['error'].zero_()
            self.native.evaluate(b['x'],b['dq'],self.length,self.box is not None,self.neighbors,self.count,
                self.rho,self.embed,self.pair,self.data.dr,self.data.drho,self.data.cutoff,
                b['fp'],b['dfp'],b['e'],b['w'],b['f'],b['h'],b['error'])
            b['energy']=b['e'].sum();b['virial']=b['w'].sum()
        b['call']=evaluate
        if self.cuda_graph:
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):evaluate()
            torch.cuda.current_stream().wait_stream(stream)
            b['graph']=torch.cuda.CUDAGraph()
            with torch.cuda.graph(b['graph']):evaluate()
        self.graphs[key]=b
        return b

    @torch.no_grad()
    def evaluate(self,q,dq=None):
        if dq is not None and (dq.shape[:-1]!=q.shape or not 1<=dq.shape[-1]<=4 or dq.device!=q.device or dq.dtype!=q.dtype):
            raise ValueError('Tangents must append 1..4 directions to q.shape')
        x=self.support(q);dirs=0 if dq is None else dq.shape[-1];b=self._buffers(x,dirs)
        b['x'].copy_(x)
        if dirs:
            if not bool(torch.isfinite(dq).all()):raise ValueError('Nonfinite tangents')
            b['dq'].copy_(dq.reshape(len(x),3,dirs))
        if self.cuda_graph:b['graph'].replay()
        else:b['call']()
        code=int(b['error'])
        if code:raise RuntimeError(f'Ni EAM error {code}: overlapping atoms or density outside fitted table; result rejected')
        return b['energy'].clone(),b['f'].reshape(q.shape).to(q.dtype).clone(),b['virial'].clone(),b['h'].reshape(*q.shape,dirs).to(q.dtype).clone()

    def force(self,q):
        e,f,_,_=self.evaluate(q);return e,f

    def force_tangent(self,q,dq):
        _,f,_,h=self.evaluate(q,dq);return f,h

    def virial_trace(self,q):return self.evaluate(q)[2]
