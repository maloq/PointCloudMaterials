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
from src.postprocessing.descriptors.rich import (harmonic_normalization, harmonics,
                                                order_invariants, persistence_packet)


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


@lru_cache(None)
def _order_names():
    names=[]
    for ell in (2,4,6,8):
        names.extend(f'l{ell}_{s}' for s in ('q','qbar','neighbor_q_mean','neighbor_q_std'))
        if ell in (4,6):names.extend(f'l{ell}_{s}' for s in ('w','wbar'))
        names.extend(f'l{ell}_coherence_{s}' for s in ('mean','std','q10','q50','q90'))
    return names


def _orders(x, distance):
    # Retain the original sort (including tie behavior) and patch-only neighbors.
    centers=np.arange(min(13,len(x)))
    near=np.argsort(distance[centers],axis=1)[:,1:13]
    vectors=x[near]-x[centers,None]
    q=harmonics(vectors,harmonic_normalization())
    # Coherence normalizes almost-zero order tensors in symmetric structures.
    # There, roundoff in Y_lm is amplified into a visible target difference.
    # Evaluate those orders with the original SciPy arithmetic, retaining the
    # existing target (including its numerical convention near zero).
    sensitive=np.any(np.sum(q.real**2+q.imag**2,axis=2)<1e-16,axis=1)
    if sensitive.any():
        radius=np.linalg.norm(vectors,axis=-1)
        theta=np.arccos(np.clip(vectors[...,2]/radius,-1,1))
        phi=np.arctan2(vectors[...,1],vectors[...,0])
        for k in np.flatnonzero(sensitive):
            ell=2*(k+1)
            q[k,:,:2*ell+1]=sph_harm_y(ell,np.arange(-ell,ell+1)[:,None,None],theta,phi).mean(-1).T
    return order_invariants(q,*_wigner(4),*_wigner(6)),list(_order_names())


@lru_cache(None)
def _topology_names():
    names=[]
    for n in (32,80):
        for dim in (0,1,2):
            prefix=f'n{n}_h{dim}'
            names.extend(f'{prefix}_{s}' for s in ('count','loglife_sum','loglife_sq','loglife_max','entropy','logbirth_sum','logdeath_sum'))
            names.extend(f'{prefix}_death{i}' for i in range(12)) if dim==0 else names.extend(f'{prefix}_image{i}' for i in range(36))
            names.extend(f'{prefix}_betti{a:.2f}' for a in np.linspace(.75,4.25,8))
    return names


def _topology(x, precision='safe', backend='gudhi', indices=None):
    names=list(_topology_names())
    if backend=='native':
        from src.postprocessing.descriptors.alpha3d import topology
        return topology(x,indices),names if indices is None else [names[i] for i in indices]
    packets=[]
    for n in (32,80):
        cloud=x[:n]
        tree=gudhi.AlphaComplex(points=cloud,precision=precision).create_simplex_tree()
        tree.compute_persistence(homology_coeff_field=2,min_persistence=0.)
        for dim in (0,1,2):
            packets.append(persistence_packet(tree.persistence_intervals_in_dimension(dim),dim,max(len(cloud)-1,1)))
    values=np.concatenate(packets)
    return (values,names) if indices is None else (values[indices],[names[i] for i in indices])


FAMILIES=('geometry','bond_order','cna','tda')
FAMILY_SETS=(FAMILIES,('bond_order','tda'),('bond_order','cna','tda'))


@lru_cache(None)
def _cna_names():
    return [f'{label}_{s}' for label in ('fixed32','fixed36','adaptive12') for s in
        ('coordination','421','422','444','666','555','544','433','other','common_mean',
         'bonds_mean','chain_mean','common_second','bonds_second','chain_second')]


def descriptor_options(config):
    settings=config.get('descriptors',{})
    families=tuple(settings.get('families',FAMILIES))
    precision=settings.get('alpha_precision','safe')
    backend=settings.get('alpha_backend','native')
    if families not in FAMILY_SETS:
        raise ValueError('Descriptor families must be full, bond_order/tda, or bond_order/cna/tda')
    if precision not in ('safe','fast'):
        raise ValueError('Alpha precision must be safe or explicitly requested fast')
    _validate_options(families,precision,backend)
    options=dict(families=families,alpha_precision=precision,alpha_backend=backend)
    if 'selection' in settings:
        from src.postprocessing.descriptors.selection import load
        options['selected_columns']=load(settings['selection'])
    return options


@lru_cache(None)
def _selected_indices(families,columns):
    local_names=[f'bond_order/{n}' for n in _order_names()]
    if 'cna' in families:local_names.extend(f'cna/{n}' for n in _cna_names())
    size=len(local_names)
    names=tuple(local_names+[f'tda/{n}' for n in _topology_names()])
    lookup={n:i for i,n in enumerate(names)}
    try:indices=[lookup[n] for n in columns]
    except KeyError as error:raise ValueError(f'Unknown spatial target: {error}') from error
    if not indices or indices!=sorted(set(indices)):
        raise ValueError('Selected spatial targets must be nonempty, unique and in producer order')
    local=[i for i in indices if i<size];tda=[i-size for i in indices if i>=size]
    if not local or not tda:raise ValueError('Spatial target selection must retain bond-order and TDA outputs')
    return local,tda


def _selection(families, columns):
    if columns is None:return None,None
    if tuple(families) not in FAMILY_SETS[1:]:
        raise ValueError('Column selection requires spatial families without geometry')
    return _selected_indices(tuple(families),tuple(columns))


def _validate_options(families, precision, backend):
    if tuple(families) not in FAMILY_SETS:
        raise ValueError(f'Undeclared descriptor families: {families}')
    if backend not in ('gudhi','native') or precision not in ('safe','fast'):
        raise ValueError(f'Undeclared alpha backend/precision: {backend}/{precision}')
    if backend=='native' and precision!='safe':
        raise ValueError('Native alpha uses safe precision; choose gudhi explicitly for the fast arithmetic ablation')


def _checked_patch(points):
    x=np.asarray(points,dtype=np.float64)
    if x.shape!=(80,3) or not np.isfinite(x).all() or np.linalg.norm(x[0])>1e-5:
        raise ValueError(f'Expected finite centered (80,3) geometry, got {x.shape}')
    radius=np.linalg.norm(x,axis=1)
    x=x[np.argsort(radius,kind='stable')];x=x[np.linalg.norm(x,axis=1)<8]
    if len(x)<14:raise ValueError(f'Fewer than 14 consumed atoms: {len(x)}')
    distance=cdist(x,x)
    if np.any(distance[np.triu_indices(len(x),1)]<1e-8):raise ValueError('Duplicate input atoms')
    return x,distance


def _local_blocks(x,distance,families):
    blocks={}
    if "geometry" in families:
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
        blocks["geometry"]=(np.asarray(geometry),gn)
    order,on=_orders(x,distance)
    blocks['bond_order']=(order,on)
    if 'cna' in families:
        r=np.linalg.norm(x,axis=1)[1:]
        cna=[];cn=[]
        for label,cutoff in (('fixed32',3.2),('fixed36',3.6),('adaptive12',r[:12].mean()*(1+np.sqrt(2))/2)):
            cna.extend(cna_packet(distance,cutoff))
            cn.extend(f'{label}_{s}' for s in ('coordination','421','422','444','666','555','544','433','other','common_mean','bonds_mean','chain_mean','common_second','bonds_second','chain_second'))
        blocks['cna']=(np.asarray(cna),cn)
    return blocks


def _assemble(blocks):
    values=np.concatenate([v for v,_ in blocks.values()]).astype(np.float32)
    names=[f'{group}/{n}' for group,(_,ns) in blocks.items() for n in ns]
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite structural descriptor')
    return values,names


def patch_descriptors(points, *, families=FAMILIES, alpha_precision='safe', alpha_backend='gudhi', selected_columns=None):
    _validate_options(families,alpha_precision,alpha_backend)
    local,tda=_selection(families,selected_columns)
    x,distance=_checked_patch(points)
    blocks=_local_blocks(x,distance,families)
    if local is not None:
        offset=0
        for group,(values,names) in list(blocks.items()):
            take=[i-offset for i in local if offset<=i<offset+len(values)]
            blocks[group]=(values[take],[names[i] for i in take]);offset+=len(values)
    blocks['tda']=_topology(x,alpha_precision,alpha_backend,tda)
    return _assemble(blocks)


def patch_descriptor_batch(points, *, families=FAMILIES, alpha_precision='safe', alpha_backend='native', threads=1, selected_columns=None):
    """Bounded native batches; threads are explicit to avoid nested oversubscription."""
    _validate_options(families,alpha_precision,alpha_backend)
    local,tda=_selection(families,selected_columns)
    points=np.asarray(points)
    if points.ndim!=3 or points.shape[1:]!=(80,3) or not len(points):
        raise ValueError(f'Expected nonempty (B,80,3) patches, got {points.shape}')
    if threads<1:raise ValueError('Descriptor thread count must be positive')
    if alpha_backend=='gudhi':
        rows=[patch_descriptors(p,families=families,alpha_precision=alpha_precision,alpha_backend='gudhi',selected_columns=selected_columns) for p in points]
        return np.stack([r[0] for r in rows]),rows[0][1]
    from src.postprocessing.descriptors.alpha3d import extension
    coordinates=np.zeros(points.shape,np.float64);counts=np.empty(len(points),np.int32)
    rows=[];names=None
    for i,patch in enumerate(points):
        try:
            x,distance=_checked_patch(patch)
            value,current=_assemble(_local_blocks(x,distance,families))
            if local is not None:value,current=value[local],[current[j] for j in local]
        except Exception as error:
            raise RuntimeError(f'Descriptor batch row {i}: {error}') from error
        if names is None:names=current
        elif current!=names:raise ValueError('Descriptor columns changed within batch')
        counts[i]=len(x);coordinates[i,:len(x)]=x;rows.append(value)
    topology=extension().packets(coordinates,counts,threads,tda)
    values=np.concatenate((np.stack(rows),topology.astype(np.float32)),axis=1)
    if not np.isfinite(values).all():raise FloatingPointError('Nonfinite batched structural descriptor')
    tnames=[f'tda/{n}' for n in _topology_names()]
    return values,names+(tnames if tda is None else [tnames[i] for i in tda])


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
