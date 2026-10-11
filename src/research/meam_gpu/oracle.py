"""Complete response paths with compiled analytic MEAM force Jacobian products.

The BAOAB protocol, original-cell noise order, and feature JVP match LocalOracle.
Only the force/Jacobian implementation is replaced; states remain float64.
"""
import math
import torch
from ase import units
from src.research.local_response.oracle import LocalOracle


class FastOracle(LocalOracle):
    def simulate(self,initial,basis,seeds):
        c=self.config
        if len(initial)!=len(seeds) or basis.shape[:2]!=initial.shape:
            raise ValueError('Replica input shape changed')
        generators=[torch.Generator(device=initial.device).manual_seed(int(s)) for s in seeds]
        def noise():
            return torch.stack([torch.randn(self.original_atoms,3,generator=g,device=initial.device,
                dtype=torch.float64)[self.atom_rows].flatten() for g in generators])
        mass=torch.full((initial.shape[1],),26.9815385,device=initial.device,dtype=torch.float64)
        kbt=units.kB*c['temperature_K'];dt=c['timestep_fs']*units.fs
        decay=math.exp(-dt/(c['friction_time_fs']*units.fs));scale=(mass*kbt*(1-decay*decay)).sqrt()
        q=initial.detach().clone();p=(mass*kbt).sqrt()*noise()
        dq=basis.detach().clone();dp=torch.zeros_like(dq)
        force,tangent=self.potential.force_tangent(q,dq);points=[];tangents=[]
        for step in range(1,max(c['horizons_steps'])+1):
            p,dp=p+dt/2*force,dp+dt/2*tangent
            q,dq=q+dt/2*p/mass,dq+dt/2*dp/mass[None,:,None]
            p,dp=decay*p+scale*noise(),decay*dp
            q,dq=q+dt/2*p/mass,dq+dt/2*dp/mass[None,:,None]
            force,tangent=self.potential.force_tangent(q,dq)
            p,dp=p+dt/2*force,dp+dt/2*tangent
            if step in c['horizons_steps']:points.append(q.clone());tangents.append(dq.clone())
        path,path_dq=torch.stack(points,1),torch.stack(tangents,1)
        value=self.features(initial,path);columns=[]
        for j in range(basis.shape[-1]):
            _,column=torch.autograd.functional.jvp(self.features,(initial,path),(basis[...,j],path_dq[...,j]))
            columns.append(column)
        response=torch.stack(columns,-1) if columns else value.new_empty(*value.shape,0)
        if not torch.isfinite(value).all() or not torch.isfinite(response).all():
            raise FloatingPointError('Nonfinite fast MEAM values/responses')
        return value.detach(),response.detach()
