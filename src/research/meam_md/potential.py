"""Periodic specialization of the validated pure-Al PyTorch MEAM energy.

Derived from the same LAMMPS equations as response_meam.potential; GPL-2.0-or-later.
See src/research/response_meam/NOTICE.md and COPYING. Orthorhombic, single-cell
MD only. A spatial bin list replaces quadratic neighbor construction. Forces
may use torch.compile on the complete differentiable energy-gradient function;
the exact geometric support is refreshed outside that compiled function.
"""
import math
import torch

from src.research.response_meam.potential import (
    Lee2003Al, RC, DELR, RE, EC, EMBED, CMIN, CMAX, EBOUND, DR, fcut,
)


class PeriodicAl(Lee2003Al):
    def __init__(self, files, atoms, box, *, device='cuda', dtype=torch.float64,
                 skin=.6, compile_force=False):
        super().__init__(files, atoms, device=device, dtype=dtype, skin=skin)
        self.register_buffer('box', torch.as_tensor(box, device=device, dtype=dtype))
        cutoff = RC*math.sqrt(EBOUND)+skin
        if self.box.shape != (3,) or bool((self.box <= 2*cutoff).any()):
            raise ValueError('Periodic MEAM requires three box lengths exceeding twice the search cutoff')
        cells=(self.box/cutoff).floor().long()
        if bool((cells < 3).any()):
            raise ValueError('Spatial bins require at least three cells along each box axis')
        self.register_buffer('cells', cells)
        self.cell_count=int(cells.prod())
        offsets=torch.cartesian_prod(*[torch.arange(-1,2,device=device)]*3)
        self.register_buffer('offsets', offsets)
        self.gradient=torch.func.grad_and_value(self.energy_with_support)
        if compile_force:
            self.gradient=torch.compile(self.gradient, dynamic=True, fullgraph=True)

    def delta(self, vector):
        return vector-self.box*torch.round(vector/self.box)

    @torch.no_grad()
    def support(self, q):
        x=q.reshape(-1,3)
        if len(x)!=self.atoms:
            raise ValueError('Periodic potential accepts exactly one fixed-size cell')
        if (self.reference is None
                or bool(2*self.delta(x-self.reference).norm(dim=-1).max() >= self.skin)):
            cell=((x.remainder(self.box)/self.box)*self.cells).floor().long()
            cell=cell.remainder(self.cells)
            linear=(cell[:,0]*self.cells[1]+cell[:,1])*self.cells[2]+cell[:,2]
            order=torch.argsort(linear,stable=True)
            count=torch.bincount(linear,minlength=self.cell_count)
            width=int(count.max())
            start=count.cumsum(0)-count
            bins=torch.full((self.cell_count,width),-1,device=q.device,dtype=torch.long)
            bins[linear[order],torch.arange(self.atoms,device=q.device)-start[linear[order]]]=order
            parts=[]
            cutoff=RC*math.sqrt(EBOUND)+self.skin
            for first in range(0,self.atoms,1024):
                stop=min(first+1024,self.atoms)
                adjacent=(cell[first:stop,None]+self.offsets).remainder(self.cells)
                adjacent=(adjacent[...,0]*self.cells[1]+adjacent[...,1])*self.cells[2]+adjacent[...,2]
                neighbors=bins[adjacent].flatten(1)
                d=self.delta(x[neighbors.clamp_min(0)]-x[first:stop,None])
                distance2=d.square().sum(-1)
                valid=(neighbors>=0)&(distance2>0)&(distance2<cutoff**2)
                i,slot=valid.nonzero(as_tuple=True)
                parts.append(torch.stack([i+first,neighbors[i,slot]]))
            all_i,all_j=torch.cat(parts,1)
            counts=torch.bincount(all_i,minlength=self.atoms)
            starts=counts.cumsum(0)-counts
            slots=torch.arange(len(all_i),device=q.device)-starts[all_i]
            neighbors=torch.full((self.atoms,int(counts.max())),-1,device=q.device,dtype=torch.long)
            neighbors[all_i,slots]=all_j
            half=all_i<all_j
            self.pair_i,self.pair_j=all_i[half],all_j[half]
            self.neighbors=neighbors
            self.reference=x.clone()
            self.rebuilds+=1
        i,j=self.pair_i,self.pair_j
        distance2=self.delta(x[j]-x[i]).square().sum(-1)
        inside=distance2<RC**2
        i,j,distance2=i[inside],j[inside],distance2[inside]
        k=self.neighbors[i]
        ik2=self.delta(x[k.clamp_min(0)]-x[i,None]).square().sum(-1)
        jk2=self.delta(x[k.clamp_min(0)]-x[j,None]).square().sum(-1)
        a=1-((ik2-jk2)/distance2[:,None]).square()
        valid=((k>=0)&(k!=j[:,None])&(a>0)
               &(ik2<=EBOUND*distance2[:,None])&(jk2<=EBOUND*distance2[:,None]))
        c=(2*(ik2+jk2)/distance2[:,None]+a-2)/torch.where(valid,a,1.)
        alive=~(valid&(c<=CMIN)).any(-1)
        i,j,k,c,valid=i[alive],j[alive],k[alive],c[alive],valid[alive]
        pair,slot=(valid&(c>CMIN)&(c<CMAX)).nonzero(as_tuple=True)
        return i,j,pair,k[pair,slot]

    def energy_with_support(self,q,support):
        i,j,pair,k=support
        x=q.reshape(-1,3)
        vector=self.delta(x[j]-x[i])
        r2=vector.square().sum(-1);r=r2.sqrt()
        ik2=self.delta(x[k]-x[i[pair]]).square().sum(-1)
        jk2=self.delta(x[k]-x[j[pair]]).square().sum(-1)
        a=1-((ik2-jk2)/r2[pair]).square()
        c=(2*(ik2+jk2)/r2[pair]+a-2)/a
        partial=fcut((c-CMIN)/(CMAX-CMIN))
        log_screen=r.new_zeros(i.shape[0]).index_add(0,pair,partial.log())
        screen=log_screen.exp()*fcut((RC-r)/DELR)
        radial=torch.exp(-(r[:,None]/RE-1)*self.beta)*screen[:,None]
        unit=vector/r[:,None];u,v,w=unit.unbind(-1)
        order2=torch.stack([u*u,u*v,u*w,v*v,v*w,w*w],-1)
        order3=torch.stack([u*u*u,u*u*v,u*u*w,u*v*v,u*v*w,u*w*w,v*v*v,v*v*w,v*w*w,w*w*w],-1)
        contributions=torch.cat([radial[:,:1],radial[:,1:2]*unit,radial[:,2:3],
            radial[:,2:3]*order2,radial[:,3:4]*unit,radial[:,3:4]*order3],-1)
        density=x.new_zeros(self.atoms,24).index_add(0,i,contributions).index_add(0,j,contributions*self.parity)
        rho0=density[:,0]
        rho1=density[:,1:4].square().sum(-1)
        rho2=(density[:,5:11].square()*self.v2).sum(-1)-density[:,4].square()/3
        rho3=(density[:,14:24].square()*self.v3).sum(-1)-.6*density[:,11:14].square().sum(-1)
        gamma=(torch.stack([rho1,rho2,rho3],-1)*self.weights).sum(-1)
        gamma=gamma/torch.where(rho0>0,rho0.square(),1.)
        rhobar=rho0*(2*torch.sigmoid(gamma))/self.rho_ref
        embedding=EMBED*EC*rhobar*torch.where(rhobar>0,rhobar,1.).log()
        coordinate=r/DR
        index=coordinate.detach().long().clamp_max(998)
        fraction=(coordinate-index).clamp_max(1.)
        coeff=self.table[index]
        pair_energy=((coeff[:,3]*fraction+coeff[:,2])*fraction+coeff[:,1])*fraction+coeff[:,0]
        return embedding.sum()+(pair_energy*screen).sum()

    def forward(self,q):
        return self.energy_with_support(q,self.support(q))

    def force(self,q):
        support=self.support(q)
        gradient,energy=self.gradient(q,support)
        return energy.detach(),-gradient.detach()
