"""Fixed initial velocities, stochastic BAOAB, and position/velocity tangents."""
import math
import torch
from ase import units
from src.project_runtime.paths import resolve_path
from src.research.response_meam.potential import Lee2003Al
from src.research.response_geometry.features import observed
from src.research.response_performance.oracle import force_and_tangent
from .periodic import PeriodicMEAM


class Features:
    def __init__(self,c,normalizer,device='cuda',box=None):
        dtype=getattr(torch,c.get('state_dtype','float64'))
        self.mean=normalizer['mean'].to(device=device,dtype=dtype)
        self.scale=normalizer['scale'].to(device=device,dtype=dtype)
        rng=torch.Generator(device='cpu').manual_seed(c['feature_seed'])
        # One map at every horizon: differences across horizons reflect dynamics.
        # Generate the same fixed map on CPU, then use the declared GPU precision.
        self.omega=(torch.randn(48,128,generator=rng,dtype=torch.float64)/math.sqrt(48)).to(device=device,dtype=dtype)
        self.phase=(2*math.pi*torch.rand(128,generator=rng,dtype=torch.float64)).to(device=device,dtype=dtype)
        self.box=box

    def observe(self,q):
        if self.box is None:return observed(q)
        x=q.reshape(len(q),-1,3);d=x-x[:,:1]
        return observed((d-self.box*torch.round(d/self.box)).flatten(1))

    def __call__(self,initial,path):
        anchor=self.observe(initial)
        future=torch.stack([self.observe(path[:,h]) for h in range(path.shape[1])],1)
        coordinates=torch.cat(((future-self.mean)/self.scale,(future-anchor[:,None])/self.scale),-1)
        return (math.sqrt(2/128)*torch.cos(coordinates@self.omega+self.phase)).flatten(1)


class Oracle:
    def __init__(self,c,state,rows,normalizer,*,periodic):
        self.c,self.state,self.rows=c,state,rows
        self.temperature_K=state['temperature_K'] if c.get('temperature_mode')=='source' else c['temperature_K']
        if not math.isfinite(self.temperature_K) or self.temperature_K<=0:raise ValueError('Invalid physical source temperature')
        files={k:resolve_path(v) for k,v in c['meam_files'].items()}
        dtype=getattr(torch,c.get('state_dtype','float64'))
        box=state['box'].to(device='cuda',dtype=dtype) if periodic else None
        if periodic and len(rows)!=state['original_atoms']:raise ValueError('Periodic oracle requires the complete native cell')
        self.fast=c.get('teacher_backend','reference')=='cuda_meam'
        if self.fast:
            from src.research.meam_gpu.potential import FastAl
            self.potential=FastAl(files,len(rows),box=box,dtype=getattr(torch,c['force_dtype']))
        else:
            self.potential=(PeriodicMEAM(files,len(rows),box,dtype=dtype) if periodic else
                            Lee2003Al(files,len(rows),dtype=dtype))
        self.features=Features(c,normalizer,box=box)

    def simulate(self,q,v,basis,seeds,*,force_free=False):
        c=self.c;count=len(seeds);n=q.numel();directions=basis.shape[-1]
        initial=q.flatten()[None].expand(count,n);x=initial.clone()
        p=(26.9815385/units.fs)*v.flatten()[None].expand(count,n).clone()
        dx=basis[0].reshape(1,n,directions).expand(count,n,directions).clone()
        dp=(26.9815385/units.fs)*basis[1].reshape(1,n,directions).expand(count,n,directions).clone()
        initial_dx=dx.clone();mass=26.9815385;dt=units.fs*c['timestep_fs']
        decay=math.exp(-c['timestep_fs']/c['friction_fs'])
        amplitude=math.sqrt(mass*units.kB*self.temperature_K*(1-decay**2))
        generators=[torch.Generator(device=x.device).manual_seed(int(s)) for s in seeds]
        def forces():
            if force_free:return torch.zeros_like(x),torch.zeros_like(dx)
            if self.fast:
                if directions:return self.potential.force_tangent(x,dx)
                _,force=self.potential.force(x)
                return force,torch.empty_like(dx)
            return force_and_tangent(self.potential,x,dx)
        force,tangent=forces();points=[];tangents=[]
        for step in range(1,max(c['horizons_fs'])//c['timestep_fs']+1):
            p,dp=p+dt/2*force,dp+dt/2*tangent
            x,dx=x+dt/2*p/mass,dx+dt/2*dp/mass
            noise=torch.stack([torch.randn(self.state['original_atoms'],3,device=x.device,dtype=x.dtype,
                generator=g)[self.rows].flatten() for g in generators])
            p,dp=decay*p+amplitude*noise,decay*dp
            x,dx=x+dt/2*p/mass,dx+dt/2*dp/mass
            force,tangent=forces();p,dp=p+dt/2*force,dp+dt/2*tangent
            if step*c['timestep_fs'] in c['horizons_fs']:points.append(x.clone());tangents.append(dx.clone())
        path=torch.stack(points,1);path_dx=torch.stack(tangents,1)
        values=self.features(initial,path)
        response=torch.stack([torch.autograd.functional.jvp(self.features,(initial,path),
            (initial_dx[...,j],path_dx[...,j]))[1] for j in range(directions)],-1) if directions else values.new_empty(count,values.shape[1],0)
        if not torch.isfinite(values).all() or not torch.isfinite(response).all():
            raise FloatingPointError(f'Nonfinite MEAM trajectory/response: parent{self.state["index"]},seeds{seeds}')
        return values.detach().cpu(),response.detach().cpu()
