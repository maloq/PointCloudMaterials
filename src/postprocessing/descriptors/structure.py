"""Full-cell PTM and coherent-bond components with a shared neighbor calculation."""
import os
import numpy as np
from numba import njit
from .ptm import full_labels

@njit
def components(solid, edges, minimum):
    """Union-find without sparse graph construction, ordered by first atom.

    The final ascending atom scan reproduces scipy's undirected component
    numbering, including exclusion of components smaller than ``minimum``.
    """
    parent=np.arange(len(solid),dtype=np.int32)
    sizes=np.ones(len(solid),np.int32)
    for edge in range(len(edges)):
        a,b=edges[edge]
        if not solid[a] or not solid[b]:continue
        while parent[a]!=a:
            parent[a]=parent[parent[a]];a=parent[a]
        while parent[b]!=b:
            parent[b]=parent[parent[b]];b=parent[b]
        if a==b:continue
        if sizes[a]<sizes[b]:a,b=b,a
        parent[b]=a;sizes[a]+=sizes[b]
    names=np.zeros(len(solid),np.int32)
    labels=np.zeros(len(solid),np.int32);count=0
    for atom in range(len(solid)):
        if not solid[atom]:continue
        root=atom
        while parent[root]!=root:
            parent[root]=parent[parent[root]];root=parent[root]
        if sizes[root]<minimum:continue
        if names[root]==0:
            count+=1;names[root]=count
        labels[atom]=names[root]
    return labels


@njit(error_model='numpy')
def bond_correlations(harmonics,ql,pairs):
    """freud 3.4 SolidLiquid's float32 reduction, without its unused clustering.

    Preserve the ordered m-sum and normalization by Steinhardt's q_l. A generic
    vector norm/reduction changes rounding near the declared bond thresholds.
    Reference: github.com/glotzerlab/freud/blob/v3.4.0/freud/order/SolidLiquid.cc
    No fastmath, FMA reassociation, or on-disk JIT cache is used.
    """
    result=np.empty(len(pairs),np.float32)
    factor=np.float32(4*np.pi/harmonics.shape[1])
    for edge in range(len(pairs)):
        i,j=pairs[edge];value=np.float32(0)
        for m in range(harmonics.shape[1]):
            a=harmonics[i,m];b=harmonics[j,m]
            value+=a.real*b.real+a.imag*b.imag
        result[edge]=value*(factor/(ql[i]*ql[j]))
    return result


def structure(points,box,c,*,include_ptm=False):
    import freud
    if freud.__version__!='3.4.0':
        raise RuntimeError('Bond-correlation equivalence was verified for freud 3.4.0; audit a new version before use')
    freud.parallel.set_num_threads(int(os.environ.get('OVITO_THREAD_COUNT','1')))
    labels=full_labels(points,box,c['ptm_rmsd_cutoff'])
    system=(freud.box.Box(*box),np.asarray(points-box/2,np.float32))
    query=freud.locality.AABBQuery(*system)
    neighbors=query.query(system[1],dict(r_max=c['neighbor_cutoff_A'],exclude_ii=True)).toNeighborList()
    pair=np.column_stack((neighbors.query_point_indices,neighbors.point_indices))
    crystal=np.isin(labels,[1,2,3])
    results=[components(crystal,pair,c['minimum_tracked_atoms'])]
    counts=[int(crystal.sum())]
    if c['detectors']==['ptm']:
        # PTM-only histories share extraction without computing unused harmonics.
        result=(np.stack(results),np.asarray(counts,np.int64),None)
        return (*result,labels) if include_ptm else result
    order=freud.order.Steinhardt(l=6).compute(query,neighbors=neighbors)
    correlation=bond_correlations(order.particle_harmonics,order.ql,pair)
    if not np.isfinite(correlation).all():raise FloatingPointError('Nonfinite bond-order correlations')
    for q,n in c['bond_order']['thresholds']:
        accepted=correlation>q
        degree=np.bincount(pair[accepted,0],minlength=len(points))
        solid=degree>=n
        results.append(components(solid,pair[accepted],c['minimum_tracked_atoms']))
        counts.append(int(solid.sum()))
    result=(np.stack(results),np.asarray(counts,np.int64),np.histogram(correlation,bins=np.linspace(-1,1,41))[0])
    return (*result,labels) if include_ptm else result
