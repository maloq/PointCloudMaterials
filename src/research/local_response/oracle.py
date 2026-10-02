"""Accelerated moving environments and local smooth path-response observables.

Noise is generated in original full-cell atom order before gathering environment
atoms. Nested environments therefore receive identical noise on shared atoms.
"""
import math
import torch
from ase import Atoms, units
from e3nn import o3

from src.project_runtime.paths import resolve_path
from src.research.response_performance.oracle import force_and_tangent, BatchedOracle
from .common import sha


class Teacher:
    def __init__(self,c,dtype=torch.float32):
        from mace.calculators import MACECalculator
        if sha(resolve_path(c['potential'])) != c['potential_sha256']:
            raise ValueError('MLIP teacher checkpoint changed')
        self.calculator=MACECalculator(model_paths=str(resolve_path(c['potential'])),device='cuda',
            default_dtype=str(dtype).split('.')[-1],enable_cueq=True,enable_oeq=False)
        self.model=self.calculator.models[0].eval().requires_grad_(False)
        template=self.calculator._atoms_to_batch(Atoms('Al2',positions=[[0,0,0],[2.8,0,0]],pbc=False)).to_dict()
        self.attrs=template['node_attrs'][:1].to(dtype)
        self.head=template['head'].detach()
        self.cutoff=float(self.model.r_max)


class OpenPotential:
    """No artificial patch periodicity; chunked GPU neighbor search bounds memory."""
    def __init__(self,teacher,atoms,skin=.6,chunk=256):
        self.teacher,self.atoms,self.skin,self.chunk=teacher,atoms,skin,chunk
        self.reference=None;self.candidates=None;self.static={};self.calls=0

    def graph(self,q):
        x=q.reshape(-1,self.atoms,3);batch,atoms,_=x.shape
        with torch.no_grad():
            rebuild=self.reference is None or self.reference.shape!=x.shape
            if not rebuild:rebuild=bool((x.detach()-self.reference).norm(dim=-1).max()*2>=self.skin)
            if rebuild:
                parts=[]
                for start in range(0,atoms,self.chunk):
                    stop=min(start+self.chunk,atoms)
                    delta=x.detach()[:,None,:,:]-x.detach()[:,start:stop,None,:]
                    mask=delta.square().sum(-1)<(self.teacher.cutoff+self.skin)**2
                    b,i,j=mask.nonzero(as_tuple=True);i=i+start
                    keep=i!=j;parts.append(torch.stack((b[keep],i[keep],j[keep])))
                self.candidates=torch.cat(parts,1)
                self.reference=x.detach().clone()
            b,i,j=self.candidates
            inside=(x.detach()[b,j]-x.detach()[b,i]).square().sum(-1)<self.teacher.cutoff**2
            b,i,j=b[inside],i[inside],j[inside]
            edge=torch.stack((b*atoms+i,b*atoms+j))
        if batch not in self.static:
            self.static[batch]=dict(node_attrs=self.teacher.attrs.repeat(batch*atoms,1),
                head=self.teacher.head.repeat(batch),cell=x.new_zeros(3*batch,3),
                batch=torch.arange(batch,device=x.device).repeat_interleave(atoms),
                ptr=torch.arange(batch+1,device=x.device)*atoms)
        zeros=x.new_zeros(edge.shape[1],3)
        return dict(self.static[batch],positions=x.flatten(0,1),edge_index=edge,shifts=zeros,unit_shifts=zeros)

    def __call__(self,q):
        self.calls+=1
        return self.teacher.model(self.graph(q),training=True,compute_force=False)['energy'].sum()


class LocalFeatures:
    """Eight radial densities + four smooth orientational powers at tracked center.

Neighbors are all simulator atoms within a smooth5.5A support, including atoms
that enter later. RFFs use changes relative to each path's own differentiable
initial anchor; prefixes20fs and20+100fs each have128 coordinates.
"""
    def __init__(self,c,dtype):
        self.cutoff=c['feature_cutoff_A'];self.scale=c['path_coordinate_scale']
        self.centers=torch.linspace(2,5,8,device='cuda',dtype=dtype)
        self.omega=[];self.phase=[]
        rng=torch.Generator(device='cuda').manual_seed(c['feature_seed'])
        for h in range(len(c['horizons_steps'])):
            self.omega.append((torch.randn(12*(h+1),128,generator=rng,device='cuda',dtype=torch.float64)
                /math.sqrt(12*(h+1))).to(dtype))
            self.phase.append((2*math.pi*torch.rand(128,generator=rng,device='cuda',dtype=torch.float64)).to(dtype))

    def observed(self,q):
        x=q.reshape(len(q),-1,3);v=x[:,1:]-x[:,:1]
        distance=v.norm(dim=-1);r=distance/self.cutoff
        w=(1-r).clamp_min(0).pow(4)*(1+4*r)
        radial=(torch.exp(-.5*((distance[...,None]-self.centers)/.35).square())*w[...,None]).sum(1)
        spherical=o3.spherical_harmonics([1,2,3,4],v,normalize=True,normalization='component')
        means=(spherical*w[...,None]).sum(1)/w.sum(1).clamp_min(1e-8)[:,None]
        orient=torch.stack([block.square().mean(-1) for block in means.split([3,5,7,9],dim=-1)],-1)
        return torch.cat((radial,orient),-1)

    def __call__(self,initial,path):
        anchor=self.observed(initial)
        change=torch.stack([self.observed(path[:,h])-anchor for h in range(path.shape[1])],1)/self.scale
        return torch.cat([math.sqrt(2/128)*torch.cos(change[:,:h+1].flatten(1)@omega+self.phase[h])
            for h,omega in enumerate(self.omega)],-1)


class LocalOracle(BatchedOracle):
    def __init__(self,potential,features,config,atom_rows,original_atoms):
        super().__init__(potential,features,config)
        self.atom_rows,self.original_atoms=atom_rows,original_atoms

    def simulate(self,initial,basis,seeds):
        c=self.config
        if len(initial)!=len(seeds) or basis.shape[:2]!=initial.shape:
            raise ValueError('Replica input shape changed')
        generators=[torch.Generator(device=initial.device).manual_seed(int(s)) for s in seeds]
        def noise():
            return torch.stack([torch.randn(self.original_atoms,3,generator=g,device=initial.device,
                dtype=torch.float64)[self.atom_rows].flatten() for g in generators])
        mass64=torch.full((initial.shape[1],),26.9815385,device=initial.device,dtype=torch.float64)
        mass=mass64.to(initial.dtype);kbt=units.kB*c['temperature_K'];dt=c['timestep_fs']*units.fs
        decay=math.exp(-dt/(c['friction_time_fs']*units.fs));scale64=(mass64*kbt*(1-decay*decay)).sqrt()
        q=initial.detach().clone();p=((mass64*kbt).sqrt()*noise()).to(initial.dtype)
        dq=basis.detach().clone();dp=torch.zeros_like(dq)
        force,tangent=force_and_tangent(self.potential,q,dq);points=[];tangents=[]
        for step in range(1,max(c['horizons_steps'])+1):
            p,dp=p+dt/2*force,dp+dt/2*tangent
            q,dq=q+dt/2*p/mass,dq+dt/2*dp/mass[None,:,None]
            p,dp=decay*p+(scale64*noise()).to(initial.dtype),decay*dp
            q,dq=q+dt/2*p/mass,dq+dt/2*dp/mass[None,:,None]
            force,tangent=force_and_tangent(self.potential,q,dq)
            p,dp=p+dt/2*force,dp+dt/2*tangent
            if step in c['horizons_steps']:points.append(q.clone());tangents.append(dq.clone())
        path,path_dq=torch.stack(points,1),torch.stack(tangents,1)
        value=self.features(initial,path);columns=[]
        for j in range(basis.shape[-1]):
            _,column=torch.autograd.functional.jvp(self.features,(initial,path),(basis[...,j],path_dq[...,j]))
            columns.append(column)
        response=torch.stack(columns,-1) if columns else value.new_empty(*value.shape,0)
        if not torch.isfinite(value).all() or not torch.isfinite(response).all():
            raise FloatingPointError('Nonfinite local simulator values/responses')
        return value.detach(),response.detach()


def make(c,state,radius,teacher):
    from .data import environment
    dtype=next(teacher.model.parameters()).dtype
    q,basis,rows=environment(state,radius,dtype)
    potential=OpenPotential(teacher,len(rows),c['oracle']['skin_A'])
    oracle=LocalOracle(potential,LocalFeatures(c['oracle'],dtype),c['oracle'],rows,state['original_atoms'])
    return oracle,q,basis
