"""Local orientational descriptors used by birth-history analysis."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import sph_harm_y

def local_order6(patch):
    """Exactly the three saved l=6 fields from liquid_predictability._orders."""
    distance=cdist(patch,patch);centers=np.arange(min(13,len(patch)))
    near=np.argsort(distance[centers],axis=1)[:,1:13]
    v=patch[near]-patch[centers,None];r=np.linalg.norm(v,axis=-1)
    theta=np.arccos(np.clip(v[...,2]/r,-1,1));phi=np.arctan2(v[...,1],v[...,0])
    q=sph_harm_y(6,np.arange(-6,7)[:,None,None],theta,phi).mean(-1).T
    norm=np.linalg.norm(q,axis=1);factor=np.sqrt(4*np.pi/13)
    unit=q/np.maximum(norm[:,None],1e-14)
    coherence=(unit[0].conj()*unit[1:]).sum(1).real
    return float(factor*norm[0]),float(factor*np.linalg.norm(q.mean(0))),float(coherence.mean())
