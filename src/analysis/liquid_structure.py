"""Continuous order measurements and alpha-complex summaries for liquid patches.

PTM classes are deliberately absent. Alpha filtration values are squared radii;
we convert them to radii in Angstrom before constructing persistence images.
"""
from functools import lru_cache
import numpy as np
from scipy.special import sph_harm_y
import gudhi


ORDER_NAMES=['q4','q6','w4','w6','qbar6','mean_q6_coherence','density_r12','smooth_coordination']


@lru_cache(None)
def wigner_terms(ell):
    from sympy.physics.wigner import wigner_3j
    terms=[(a,b,-a-b) for a in range(-ell,ell+1) for b in range(-ell,ell+1) if abs(a+b)<=ell]
    indices=np.array(terms,dtype=np.int64)+ell
    weights=np.array([float(wigner_3j(ell,ell,ell,*t)) for t in terms])
    return indices,weights


def bond_order(vectors, coordination_radius):
    """vectors[B,13,12,3]: 12 bonds of center and its 12 nearest neighbors."""
    radius=np.linalg.norm(vectors,axis=-1)
    theta=np.arccos(np.clip(vectors[...,2]/radius,-1,1))
    phi=np.arctan2(vectors[...,1],vectors[...,0])
    q={}
    scalar=[]
    for ell in (4,6):
        q[ell]=np.stack([sph_harm_y(ell,m,theta,phi).mean(-1) for m in range(-ell,ell+1)],-1)
        scalar.append(np.sqrt(4*np.pi/(2*ell+1))*np.linalg.norm(q[ell][:,0],axis=-1))
    for ell in (4,6):
        indices,weights=wigner_terms(ell)
        values=q[ell][:,0]
        cubic=(values[:,indices].prod(-1)*weights).sum(-1).real
        norm=np.linalg.norm(values,axis=-1)
        scalar.append(np.divide(cubic,norm**3,out=np.zeros_like(cubic),where=norm>1e-14))
    norm=np.linalg.norm(q[6],axis=-1)
    unit=np.divide(q[6],norm[...,None],out=np.zeros_like(q[6]),where=norm[...,None]>1e-14)
    coherence=np.einsum('bm,bnm->bn',unit[:,0].conj(),unit[:,1:]).real
    qbar=np.sqrt(4*np.pi/13)*np.linalg.norm(q[6].mean(1),axis=-1)
    density=12/(4*np.pi*radius[:,0,-1]**3/3)
    coord=np.exp(-(radius[:,0]/coordination_radius)**8).sum(1)
    scalar.extend((qbar,coherence.mean(1),density,coord))
    counts=np.stack([(coherence>threshold).sum(1) for threshold in (.65,.70,.75)],-1)
    return np.stack(scalar,-1).astype(np.float32),counts.astype(np.int16)


def persistence_image(points):
    """144D H0/H1/H2 summary of the supplied 65- or 80-atom neighborhood.

    Finite deaths above 3.5 A are excluded to suppress large boundary voids.
    H1/H2 use lifetime-weighted Gaussian surfaces on fixed 8x8 grids. H0 uses
    a 16-bin Gaussian death-radius curve. This is a descriptor, not a phase label.
    """
    tree=gudhi.AlphaComplex(points=points,precision='safe').create_simplex_tree()
    tree.compute_persistence(homology_coeff_field=2,min_persistence=0.)
    death_grid=np.linspace(.7,2.5,16)
    birth_grid=np.linspace(.7,3.,8)
    life_grid=np.linspace(.025,1.1,8)
    result=[]
    for dim in (0,1,2):
        pairs=tree.persistence_intervals_in_dimension(dim)
        pairs=pairs[np.isfinite(pairs[:,1])]
        pairs=np.sqrt(np.maximum(pairs,0.))
        pairs=pairs[pairs[:,1]<=3.5]
        if dim==0:
            result.append(np.exp(-.5*((pairs[:,1,None]-death_grid)/.10)**2).sum(0)/(len(points)-1))
        else:
            lifetime=pairs[:,1]-pairs[:,0]
            birth=np.exp(-.5*((pairs[:,0,None]-birth_grid)/.15)**2)
            life=np.exp(-.5*((lifetime[:,None]-life_grid)/.09)**2)
            result.append(((birth*lifetime[:,None]).T@life).ravel()/(len(points)-1))
    return np.concatenate(result).astype(np.float32)


def nonaffine_displacement(initial,future):
    """Best affine residual for the same initial 24 neighbor identities, in A^2."""
    x,y=initial.astype(np.float64),future.astype(np.float64)
    affine=np.linalg.solve(x.transpose(0,2,1)@x,x.transpose(0,2,1)@y)
    return np.mean(np.sum((y-x@affine)**2,-1),1).astype(np.float32)
