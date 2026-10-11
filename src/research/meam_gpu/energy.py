"""Same Lee2003 energy with stable cutoff arithmetic and selectable precision.

GPL-2.0-or-later; derived from the validated response_meam energy and LAMMPS.
The polynomial x*(4-6*x+4*x*x-x*x*x) is exactly 1-(1-x)**4, with less
subtractive cancellation near zero. No cutoff, force-field parameter or term
is dropped. Scalar energy accumulation is float64 in both force precisions.
"""
import torch
from src.research.response_meam.potential import RC,DELR,RE,EC,EMBED,CMIN,CMAX,DR


def cutoff(x):
    x=x.clamp(0,1)
    return (x*(4+x*(-6+x*(4-x)))).square()


def energy(model,q,support,strain=None):
    i,j,pair,k=support
    x=q.reshape(-1,3).to(model.table.dtype)
    def delta(d):
        d=d if model.periodic_box is None else d-model.periodic_box*torch.round(d/model.periodic_box)
        return d if strain is None else d*strain
    vector=delta(x[j]-x[i]);r2=vector.square().sum(-1);r=r2.sqrt()
    ik2=delta(x[k]-x[i[pair]]).square().sum(-1)
    jk2=delta(x[k]-x[j[pair]]).square().sum(-1)
    a=1-((ik2-jk2)/r2[pair]).square()
    c=(2*(ik2+jk2)/r2[pair]+a-2)/a
    partial=cutoff((c-CMIN)/(CMAX-CMIN))
    positive=partial>0
    log_screen=r.new_zeros(i.shape[0]).index_add(0,pair,torch.where(positive,partial,1.).log())
    blocked=r.new_zeros(i.shape[0]).index_add(0,pair,(~positive).to(r.dtype))
    screen=log_screen.exp()*(blocked==0).to(r.dtype)*cutoff((RC-r)/DELR)
    radial=torch.exp(-(r[:,None]/RE-1)*model.beta)*screen[:,None]
    unit=vector/r[:,None];u,v,w=unit.unbind(-1)
    order2=torch.stack([u*u,u*v,u*w,v*v,v*w,w*w],-1)
    order3=torch.stack([u*u*u,u*u*v,u*u*w,u*v*v,u*v*w,u*w*w,v*v*v,v*v*w,v*w*w,w*w*w],-1)
    contributions=torch.cat([radial[:,:1],radial[:,1:2]*unit,radial[:,2:3],
        radial[:,2:3]*order2,radial[:,3:4]*unit,radial[:,3:4]*order3],-1)
    density=x.new_zeros(len(x),24).index_add(0,i,contributions).index_add(0,j,contributions*model.parity)
    rho0=density[:,0];rho1=density[:,1:4].square().sum(-1)
    rho2=(density[:,5:11].square()*model.v2).sum(-1)-density[:,4].square()/3
    rho3=(density[:,14:24].square()*model.v3).sum(-1)-.6*density[:,11:14].square().sum(-1)
    gamma=(torch.stack([rho1,rho2,rho3],-1)*model.weights).sum(-1)/torch.where(rho0>0,rho0.square(),1.)
    rhobar=rho0*(2*torch.sigmoid(gamma))/model.rho_ref
    embedding=EMBED*EC*rhobar*torch.where(rhobar>0,rhobar,1.).log()
    coordinate=r/DR;index=coordinate.detach().long().clamp_max(998)
    fraction=(coordinate-index).clamp_max(1.);coeff=model.table[index]
    pair_energy=((coeff[:,3]*fraction+coeff[:,2])*fraction+coeff[:,1])*fraction+coeff[:,0]
    return embedding.sum(dtype=torch.float64)+(pair_energy*screen).sum(dtype=torch.float64)
