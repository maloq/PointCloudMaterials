"""Shared birth-history geometry and retrospective graph operations.

Cohort selection, frame windows, physical cutoffs and publication stay in the
protocol adapters. This module never creates birth labels.
"""
from functools import lru_cache
import hashlib
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from src.experiment_runner.artifacts import json_digest as digest
from src.project_runtime.paths import resolve_path
from src.postprocessing.descriptors.local import local_order6

@lru_cache(maxsize=16)
def _resolved_path(value):return resolve_path(value)


def root(c):return _resolved_path(c['output'])


def cache(c):return _resolved_path(c['cache'])


def slug(event_id):return 'birth-'+hashlib.sha256(event_id.encode()).hexdigest()[:20]


def clusters(q,box,labels,cutoff):
    ids=np.flatnonzero(np.isin(labels,[1,2,3]))
    if not len(ids):return []
    edges=cKDTree(q[ids],boxsize=box).query_pairs(cutoff,output_type='ndarray')
    graph=coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(ids),len(ids))).tocsr()
    _,component=connected_components(graph,directed=False)
    order=np.argsort(component,kind='stable');parts=np.split(order,np.flatnonzero(np.diff(component[order]))+1)
    return [ids[ix] for ix in parts]


def node(sid,frame,time_ps,members,atom_ids,known_family=None):
    aid=atom_ids[members].tolist();key='cluster-'+digest([sid,int(frame),aid])[:24]
    return dict(id=key,frame=int(frame),time_ps=float(time_ps),size=len(aid),atom_ids=aid,
        known_established_family=known_family,path_status='unlinked')


def precursor_links(first, start, frame_nodes, founders, maximum_frame_gap):
    """Shared backwards atom-overlap graph; slots are exact observed frames."""
    roots=[r for r in frame_nodes[first] if set(r['atom_ids'])==set(founders)]
    if len(roots)!=1:raise ValueError('Full-cell first component does not reproduce catalogue membership: ')
    anchor_node=roots[0];anchor_node['path_status']='direct';nodes={anchor_node['id']:anchor_node}
    frontier=[(first,anchor_node)];edges=[]
    for slot in range(first-1,start-1,-1):
        next_frontier=[];matched=set()
        for current in frame_nodes[slot]:
            atoms=set(current['atom_ids']);statuses=[];has_edge=False
            for next_slot,child in frontier:
                gap=next_slot-slot
                if gap>maximum_frame_gap:continue
                shared=len(atoms.intersection(child['atom_ids']))
                if not shared:continue
                has_edge=True
                fraction_parent=shared/current['size'];fraction_child=shared/child['size']
                strong=shared>=min(4,current['size'],child['size']) and max(fraction_parent,fraction_child)>=.5
                kind=('direct' if gap==1 else 'gap') if strong else 'weak_overlap'
                edges.append(dict(parent=current['id'],child=child['id'],shared_atoms=shared,
                    parent_fraction=fraction_parent,child_fraction=fraction_child,gap_frames=gap,kind=kind))
                if strong:
                    statuses.append('direct' if kind=='direct' and child['path_status']=='direct' else 'gap')
                    matched.add(child['id'])
            if has_edge:
                current['path_status']='direct' if 'direct' in statuses else 'gap' if statuses else 'weak_only'
                nodes[current['id']]=current
                if statuses:next_frontier.append((slot,current))
        next_frontier.extend((a,b) for a,b in frontier if b['id'] not in matched and a-slot<maximum_frame_gap)
        frontier=next_frontier
    return anchor_node, nodes, edges


def local_environment(q, box, anchor, radius, *, neighbors=None):
    """One periodic displacement field and nearest-neighbor query per frame."""
    displacement = q - q[anchor]
    displacement -= box * np.rint(displacement / box)
    local = np.flatnonzero(np.linalg.norm(displacement, axis=1) <= radius)
    if neighbors is None:
        _, neighbors = cKDTree(q, boxsize=box).query(q[anchor], k=80, workers=1)
    patch = displacement[neighbors]
    return displacement, local, patch, local_order6(patch)


def shape(cloud):
    if not len(cloud):
        return dict(radius_gyration_A=None, anisotropy=None)
    centered = cloud - cloud.mean(0)
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(cloud))
    return dict(radius_gyration_A=float(np.sqrt(eigenvalues.sum())),
                anisotropy=float((eigenvalues[-1]-eigenvalues[0])/max(eigenvalues.sum(),1e-12)))
