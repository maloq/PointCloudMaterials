"""A box-frame cuboctahedral stencil, with explicit actual-atom displacements."""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment


def stencil(radii=(10.,20.)):
    directions=[]
    for zero in range(3):
        for signs in itertools.product((-1.,1.),repeat=2):
            vector=np.zeros(3);vector[[i for i in range(3) if i!=zero]]=signs
            directions.append(vector/np.sqrt(2))
    directions=np.array(directions)
    return np.concatenate((np.zeros((1,3)),*[r*directions for r in radii])).astype(np.float32)


def assign(relative,atom_rows,queries,max_offset=4.):
    """Unique nearest atom assignment; center slot is separately fixed by caller.

    Queries, not real atom positions, are exactly symmetric. Under rigid rotations,
    rotate the query frame together with coordinates. A lab-fixed stencil has only
    cubic symmetry; we do not claim continuous SO(3) invariance for re-sampling it.
    """
    distance=np.sum((queries[1:,None]-relative[None])**2,axis=-1)
    q,indices=linear_sum_assignment(distance)
    if len(q)!=len(queries)-1:raise ValueError('Insufficient atoms for all structured slots')
    errors=np.sqrt(distance[q,indices])
    if errors.max()>max_offset:raise ValueError(f'Query assignment exceeds {max_offset} A: {errors.tolist()}')
    return atom_rows[indices],relative[indices]


def representatives(points,center,tree,box,queries,max_offset=4.):
    rows=np.asarray(tree.query_ball_point(points[center],float(np.linalg.norm(queries,axis=1).max()+max_offset)))
    rows=np.sort(rows[rows!=center]);relative=points[rows]-points[center];relative-=box*np.round(relative/box)
    atoms,geometry=assign(relative,rows,queries,max_offset)
    return np.r_[center,atoms],np.vstack((np.zeros(3),geometry)).astype(np.float32)
