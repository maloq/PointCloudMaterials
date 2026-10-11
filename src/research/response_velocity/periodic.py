"""Periodic geometry for the exact Lee2003 MEAM energy and its derivatives.

The density/energy algebra is derived from response_meam/potential.py, whose
LAMMPS attribution and GPL-2.0-or-later terms also apply here. Only neighbor
construction and minimum-image displacement geometry differ. No potential fit.
"""
import itertools
import math
import torch
from src.research.response_meam.potential import Lee2003Al,RC,DELR,EBOUND,CMIN,CMAX,RE,EMBED,EC,DR,fcut


class PeriodicMEAM(Lee2003Al):
    def __init__(self,files,atoms,box,**kwargs):
        super().__init__(files,atoms,**kwargs)
        self.register_buffer('box',box.to(self.table))
        if self.box.shape!=(3,) or self.box.min()<=2*(RC*math.sqrt(EBOUND)+self.skin):
            raise ValueError('Require an orthorhombic cell larger than twice the neighbor support')
        self.register_buffer('offsets',torch.tensor(list(itertools.product([-1,0,1],repeat=3)),device=box.device))

    def wrap(self,d):
        return d-self.box*torch.round(d/self.box)

    @torch.no_grad()
    def support(self,q):
        x=q.reshape(-1,self.atoms,3);batch=len(x);flat=x.flatten(0,1)
        if self.reference is None or self.reference.shape!=x.shape or bool(2*(x-self.reference).norm(dim=-1).max()>=self.skin):
            cutoff=RC*math.sqrt(EBOUND)+self.skin
            shape=torch.floor(self.box/cutoff).long();ncell=int(shape.prod());cellsize=self.box/shape
            coord=torch.minimum(torch.floor(torch.remainder(flat,self.box)/cellsize).long(),shape-1)
            replica=torch.arange(batch,device=q.device).repeat_interleave(self.atoms)
            cell=(coord[:,0]*shape[1]+coord[:,1])*shape[2]+coord[:,2]+replica*ncell
            order=torch.argsort(cell,stable=True);counts=torch.bincount(cell,minlength=batch*ncell)
            starts=counts.cumsum(0)-counts;slots=torch.arange(len(flat),device=q.device)-starts[cell[order]]
            buckets=torch.full((batch*ncell,int(counts.max())),-1,dtype=torch.long,device=q.device)
            buckets[cell[order],slots]=order
            pieces=[]
            for start in range(0,len(flat),1024):
                stop=min(start+1024,len(flat));nodes=torch.arange(start,stop,device=q.device)
                neighboring=(coord[start:stop,None]+self.offsets[None])%shape
                keys=(neighboring[...,0]*shape[1]+neighboring[...,1])*shape[2]+neighboring[...,2]+replica[start:stop,None]*ncell
                candidates=buckets[keys].flatten(1)
                delta=self.wrap(flat[candidates.clamp_min(0)]-flat[nodes,None])
                keep=(candidates>=0)&(candidates!=nodes[:,None])&(delta.square().sum(-1)<cutoff**2)
                i,j=keep.nonzero(as_tuple=True);pieces.append(torch.stack((i+start,candidates[i,j])))
            all_i,all_j=torch.cat(pieces,1)
            counts=torch.bincount(all_i,minlength=len(flat));starts=counts.cumsum(0)-counts
            self.neighbors=torch.full((len(flat),int(counts.max())),-1,dtype=torch.long,device=q.device)
            self.neighbors[all_i,torch.arange(len(all_i),device=q.device)-starts[all_i]]=all_j
            keep=all_i<all_j;self.pair_i,self.pair_j=all_i[keep],all_j[keep]
            self.reference=x.clone();self.rebuilds+=1
        i,j=self.pair_i,self.pair_j
        distance2=self.wrap(flat[j]-flat[i]).square().sum(-1)
        keep=distance2<RC**2;i,j,distance2=i[keep],j[keep],distance2[keep]
        k=self.neighbors[i]
        ik2=self.wrap(flat[k.clamp_min(0)]-flat[i,None]).square().sum(-1)
        jk2=self.wrap(flat[k.clamp_min(0)]-flat[j,None]).square().sum(-1)
        a=1-((ik2-jk2)/distance2[:,None]).square()
        valid=(k>=0)&(k!=j[:,None])&(a>0)&(ik2<=EBOUND*distance2[:,None])&(jk2<=EBOUND*distance2[:,None])
        c=(2*(ik2+jk2)/distance2[:,None]+a-2)/torch.where(valid,a,1.)
        alive=~(valid&(c<=CMIN)).any(-1);i,j,k,c,valid=i[alive],j[alive],k[alive],c[alive],valid[alive]
        pair,slot=(valid&(c>CMIN)&(c<CMAX)).nonzero(as_tuple=True)
        return i,j,pair,k[pair,slot]

    def energy_with_support(self,q,support):
        i,j,pair,k=support;x=q.reshape(-1,3)
        vector=self.wrap(x[j]-x[i]);r2=vector.square().sum(-1);r=r2.sqrt()
        ik2=self.wrap(x[k]-x[i[pair]]).square().sum(-1)
        jk2=self.wrap(x[k]-x[j[pair]]).square().sum(-1)
        a=1-((ik2-jk2)/r2[pair]).square();c=(2*(ik2+jk2)/r2[pair]+a-2)/a
        partial=fcut((c-CMIN)/(CMAX-CMIN));log_screen=r.new_zeros(len(i)).index_add(0,pair,partial.log())
        screen=log_screen.exp()*fcut((RC-r)/DELR)
        radial=torch.exp(-(r[:,None]/RE-1)*self.beta)*screen[:,None]
        unit=vector/r[:,None];u,v,w=unit.unbind(-1)
        order2=torch.stack([u*u,u*v,u*w,v*v,v*w,w*w],-1)
        order3=torch.stack([u*u*u,u*u*v,u*u*w,u*v*v,u*v*w,u*w*w,v*v*v,v*v*w,v*w*w,w*w*w],-1)
        contributions=torch.cat([radial[:,:1],radial[:,1:2]*unit,radial[:,2:3],radial[:,2:3]*order2,
            radial[:,3:4]*unit,radial[:,3:4]*order3],-1)
        density=x.new_zeros(len(x),24).index_add(0,i,contributions).index_add(0,j,contributions*self.parity)
        rho0=density[:,0];rho1=density[:,1:4].square().sum(-1)
        rho2=(density[:,5:11].square()*self.v2).sum(-1)-density[:,4].square()/3
        rho3=(density[:,14:24].square()*self.v3).sum(-1)-.6*density[:,11:14].square().sum(-1)
        gamma=(torch.stack([rho1,rho2,rho3],-1)*self.weights).sum(-1)/torch.where(rho0>0,rho0.square(),1.)
        rhobar=rho0*(2*torch.sigmoid(gamma))/self.rho_ref
        embedding=EMBED*EC*rhobar*torch.where(rhobar>0,rhobar,1.).log()
        coordinate=r/DR;index=coordinate.detach().long().clamp_max(998);fraction=(coordinate-index).clamp_max(1.)
        coeff=self.table[index];pair_energy=((coeff[:,3]*fraction+coeff[:,2])*fraction+coeff[:,1])*fraction+coeff[:,0]
        return embedding.sum()+(pair_energy*screen).sum()
