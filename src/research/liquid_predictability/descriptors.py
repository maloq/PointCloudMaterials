"""Fixed geometry descriptors of the exact radius-8 / nearest-80 observation.

No labels or cell-level neighbors enter these calculations. Alpha filtration is
converted from squared radii to Å. CNA is a bond fingerprint, not a PTM target.
"""
from functools import lru_cache
import numpy as np
import gudhi
from numba import njit
from scipy.spatial.distance import cdist
from scipy.special import sph_harm_y


@njit(cache=True)
def _longest_chain(graph):
    # Explicit DFS stack: recursive Numba cache entries are not safely portable
    # between the independently launched preparation processes in this environment.
    n=len(graph);best=0
    nodes=np.zeros(n+1,np.int64);next_neighbor=np.zeros(n+1,np.int64)
    for start in range(n):
        depth=0;nodes[0]=start;next_neighbor[0]=0;visited=1<<start
        while depth>=0:
            current=nodes[depth];nxt=next_neighbor[depth]
            if nxt==n:
                visited &= ~(1<<current);depth-=1
                continue
            next_neighbor[depth]+=1
            if not graph[current,nxt]:continue
            if nxt==start and depth>=2:
                best=max(best,depth+1)
            elif not (visited & (1<<nxt)):
                depth+=1;nodes[depth]=nxt;next_neighbor[depth]=0
                visited |= 1<<nxt;best=max(best,depth)
    return best


@njit(cache=True)
def cna_packet(distance, cutoff):
    """CNA (common atoms, bonds among them, longest simple chain/ring) histogram."""
    bonded = (distance < cutoff) & (distance > 1e-8)
    signatures = np.array([[4, 2, 1], [4, 2, 2], [4, 4, 4], [6, 6, 6],
                           [5, 5, 5], [5, 4, 4], [4, 3, 3]])
    counts = np.zeros(8)
    summaries = np.zeros(6)
    coordination = 0
    for j in range(1, len(distance)):
        if not bonded[0, j]:
            continue
        coordination += 1
        common = np.where(bonded[0] & bonded[j])[0]
        n = len(common)
        if n > 20:
            raise ValueError('CNA shared graph exceeds declared 20-atom bound')
        graph = np.zeros((n, n), np.bool_)
        for a in range(n):
            for b in range(n):
                graph[a, b] = bonded[common[a], common[b]]
        edges = graph.sum() // 2
        longest = _longest_chain(graph)
        index = 7
        for k in range(7):
            if n == signatures[k, 0] and edges == signatures[k, 1] and longest == signatures[k, 2]:
                index = k
        counts[index] += 1
        summaries += np.array([n, edges, longest, n*n, edges*edges, longest*longest])
    if coordination:
        counts /= coordination
        summaries /= coordination
    return np.concatenate((np.array([coordination]), counts, summaries))


@lru_cache(None)
def _wigner(ell):
    from sympy.physics.wigner import wigner_3j
    triples = [(a,b,-a-b) for a in range(-ell,ell+1) for b in range(-ell,ell+1) if abs(a+b)<=ell]
    return np.asarray(triples)+ell, np.array([float(wigner_3j(ell,ell,ell,*t)) for t in triples])


def _orders(x, distance):
    # Neighbor-averaged invariants use only atoms already present in this patch.
    centers = np.arange(min(13, len(x)))
    near = np.argsort(distance[centers], axis=1)[:, 1:13]
    v = x[near]-x[centers,None]
    r = np.linalg.norm(v,axis=-1)
    theta = np.arccos(np.clip(v[...,2]/r,-1,1)); phi = np.arctan2(v[...,1],v[...,0])
    values=[]; names=[]
    for ell in (2,4,6,8):
        harmonics=sph_harm_y(ell,np.arange(-ell,ell+1)[:,None,None],theta,phi)
        q=harmonics.mean(-1).T
        factor=np.sqrt(4*np.pi/(2*ell+1))
        norm=np.linalg.norm(q,axis=1)
        values.extend((factor*norm[0],factor*np.linalg.norm(q.mean(0)),factor*norm.mean(),factor*norm.std()))
        names.extend(f'l{ell}_{s}' for s in ('q','qbar','neighbor_q_mean','neighbor_q_std'))
        if ell in (4,6):
            ix,weight=_wigner(ell)
            for field,label in ((q[0],'w'),(q.mean(0),'wbar')):
                values.append(float((field[ix].prod(1)*weight).sum().real/max(np.linalg.norm(field)**3,1e-14)))
                names.append(f'l{ell}_{label}')
        unit=q/np.maximum(norm[:,None],1e-14)
        coherence=(unit[0].conj()*unit[1:]).sum(1).real
        values.extend((coherence.mean(),coherence.std(),*np.quantile(coherence,[.1,.5,.9])))
        names.extend(f'l{ell}_coherence_{s}' for s in ('mean','std','q10','q50','q90'))
    return np.asarray(values), names


def _topology(x):
    values=[]; names=[]
    for n in (32,80):
        cloud=x[:n]
        tree=gudhi.AlphaComplex(points=cloud,precision='safe').create_simplex_tree()
        tree.compute_persistence(homology_coeff_field=2,min_persistence=0.)
        norm=max(len(cloud)-1,1)
        for dim in (0,1,2):
            pairs=tree.persistence_intervals_in_dimension(dim)
            pairs=np.sqrt(np.maximum(pairs[np.isfinite(pairs[:,1])],0.))
            birth=pairs[:,0]; death=pairs[:,1]; life=death-birth
            prefix=f'n{n}_h{dim}'
            # All finite intervals retained; log moments remain finite for large
            # alpha radii of boundary tetrahedra. No hard death-radius exclusion.
            logs=np.log1p(life)
            p=life/max(life.sum(),1e-14)
            stats=[len(life)/norm,logs.sum()/norm,(logs**2).sum()/norm,
                   float(logs.max()) if len(logs) else 0.,
                   float(-(p[p>0]*np.log(p[p>0])).sum()),
                   np.log1p(birth).sum()/norm,np.log1p(death).sum()/norm]
            values.extend(stats);names.extend(f'{prefix}_{s}' for s in ('count','loglife_sum','loglife_sq','loglife_max','entropy','logbirth_sum','logdeath_sum'))
            if dim==0:
                grid=np.linspace(.6,3.,12)
                values.extend(np.exp(-.5*((death[:,None]-grid)/.15)**2).sum(0)/norm)
                names.extend(f'{prefix}_death{i}' for i in range(12))
            else:
                bg=np.linspace(.6,4.,6);lg=np.linspace(0,1.5,6)
                b=np.exp(-.5*((birth[:,None]-bg)/.25)**2)
                l=np.exp(-.5*((life[:,None]-lg)/.15)**2)
                values.extend(((b*np.tanh(life)[:,None]).T@l/norm).ravel())
                names.extend(f'{prefix}_image{i}' for i in range(36))
            for a in np.linspace(.75,4.25,8):
                # Smooth Betti curve; excludes the essential H0 component.
                enter=.5*(1+np.tanh((a-birth)/.1));leave=.5*(1+np.tanh((death-a)/.1))
                values.append(float((enter*leave).sum()/norm));names.append(f'{prefix}_betti{a:.2f}')
    return np.asarray(values),names


def patch_descriptors(points):
    x=np.asarray(points,dtype=np.float64)
    if x.shape!=(80,3) or not np.isfinite(x).all() or np.linalg.norm(x[0])>1e-5:
        raise ValueError(f'Expected finite centered (80,3) geometry, got {x.shape}')
    radius=np.linalg.norm(x,axis=1)
    x=x[np.argsort(radius,kind='stable')];x=x[np.linalg.norm(x,axis=1)<8]
    if len(x)<14:raise ValueError(f'Fewer than 14 consumed atoms: {len(x)}')
    distance=cdist(x,x)
    if np.any(distance[np.triu_indices(len(x),1)]<1e-8):raise ValueError('Duplicate input atoms')
    r=np.linalg.norm(x,axis=1)[1:]
    geometry=[];gn=[]
    def append(v,names):geometry.extend(v);gn.extend(names)
    append([len(x)-1,*np.quantile(r,[0,.1,.25,.5,.75,.9,1])],['count']+[f'r_q{q}' for q in (0,10,25,50,75,90,100)])
    append(r[:12],[f'neighbor_distance_{i+1}' for i in range(12)])
    grid=np.linspace(.5,7.5,24)
    append(np.exp(-.5*((r[:,None]-grid)/.3)**2).sum(0)/80,[f'rdf_{i}' for i in range(24)])
    pair=distance[np.triu_indices(len(x),1)]
    append(np.histogram(pair,bins=np.linspace(0,16,17))[0]/len(pair),[f'pair_hist_{i}' for i in range(16)])
    for n in (12,24):
        v=x[1:n+1];u=v/np.linalg.norm(v,axis=1)[:,None];cos=(u@u.T)[np.triu_indices(len(v),1)]
        append(np.histogram(cos,bins=np.linspace(-1.000001,1.000001,13))[0]/len(cos),[f'angle_n{n}_{i}' for i in range(12)])
    for n in (12,32,79):
        v=x[1:n+1];cov=v.T@v/len(v);eig=np.linalg.eigvalsh(cov)
        append([*eig,float(np.linalg.norm(v.mean(0))),np.linalg.det(cov)],[f'shape_n{n}_{s}' for s in ('e0','e1','e2','offset','det')])
    order,on=_orders(x,distance)
    cna=[];cn=[]
    for label,cutoff in (('fixed32',3.2),('fixed36',3.6),('adaptive12',r[:12].mean()*(1+np.sqrt(2))/2)):
        cna.extend(cna_packet(distance,cutoff))
        cn.extend(f'{label}_{s}' for s in ('coordination','421','422','444','666','555','544','433','other','common_mean','bonds_mean','chain_mean','common_second','bonds_second','chain_second'))
    topology,tn=_topology(x)
    blocks={'geometry':(np.asarray(geometry),gn),'bond_order':(order,on),'cna':(np.asarray(cna),cn),'tda':(topology,tn)}
    values=np.concatenate([v for v,_ in blocks.values()]).astype(np.float32)
    names=[f'{group}/{n}' for group,(_,ns) in blocks.items() for n in ns]
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite structural descriptor')
    return values,names


def summarize_context(x,actual):
    """Identical descriptors at every patch; scalarized spatial moments retain arrangement."""
    radius=np.linalg.norm(actual,axis=-1);parts=[];labels=[]
    for lo,hi in ((-1,4),(4,14),(14,24.0001)):
        w=((radius>lo)&(radius<=hi)).astype(float)
        if (w.sum(1)==0).any():raise ValueError(f'Empty context shell {lo}/{hi}')
        w/=w.sum(1,keepdims=True)
        mean=np.einsum('np,npf->nf',w,x)
        parts.extend([mean,np.sqrt(np.maximum(np.einsum('np,npf->nf',w,x*x)-mean*mean,0))])
        labels.extend([f'shell{hi:g}_mean',f'shell{hi:g}_std'])
    centered=x-x.mean(1,keepdims=True)
    direction=actual/24
    gradient=np.einsum('npf,npd->nfd',centered,direction)/25
    parts.append(np.linalg.norm(gradient,axis=-1));labels.append('gradient_norm')
    q=np.einsum('npi,npj->npij',direction,direction)
    q-=np.eye(3)*np.sum(direction**2,axis=-1)[...,None,None]/3
    quadrupole=np.einsum('npf,npij->nfij',centered,q)/25
    parts.append(np.sqrt(np.sum(quadrupole**2,axis=(-1,-2))));labels.append('quadrupole_norm')
    return np.concatenate(parts,axis=1).astype(np.float32),labels
