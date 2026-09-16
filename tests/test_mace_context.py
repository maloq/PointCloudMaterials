"""Scientific graph support and boundary-crossing definitions, independent of GPU."""

import numpy as np

from src.models.encoders.mace_context import inner_weights, make_context_graph
from src.research.mace_context.evaluate import crossing_clouds


def test_two_hop_support_keeps_only_actual_ancestors():
    x=np.array([[0,0,0],[4,0,0],[8,0,0],[12,0,0],[0,20,0]],dtype=np.float32)
    g=make_context_graph([x],'halo_center',device='cpu')
    np.testing.assert_array_equal(g.positions.numpy(),x[:3])
    np.testing.assert_array_equal(g.first_keep.numpy(),[0,1])
    np.testing.assert_array_equal(g.second_keep.numpy(),[0])
    assert g.first_edges.shape[1]==3
    assert g.second_edges.shape[1]==1


def test_taper_endpoints_and_monotonicity():
    r=np.linspace(0,10,1001)
    w=inner_weights(r)
    assert np.all(w[r<=5]==1) and np.all(w[r>=7]==0)
    assert np.all(np.diff(w)<=1e-12)
    assert inner_weights([7-1e-3])[0] < 2e-9
    near_cutoff=np.linspace(6.99,7.01,10000,dtype=np.float32)
    assert np.all(inner_weights(near_cutoff)>=0)
    assert np.all(np.diff(inner_weights(near_cutoff))<=0)


def test_rank_crossing_moves_two_atoms_and_exchanges_membership():
    rng=np.random.default_rng(17)
    x=np.r_[np.zeros((1,3)),rng.normal(size=(120,3))].astype(np.float32)
    order=np.argsort(np.linalg.norm(x,axis=1));a,b=order[79:81]
    left,right=crossing_clouds(x,1e-4)
    original_direction=x[a]/np.linalg.norm(x[a])
    locate=lambda c:np.argmin(np.linalg.norm(c[1:]/np.linalg.norm(c[1:],axis=1)[:,None]-original_direction,axis=1))+1
    assert (locate(left)<80) != (locate(right)<80)
    # Tracked center survives both rank selections.
    np.testing.assert_array_equal(left[0],np.zeros(3))
    np.testing.assert_array_equal(right[0],np.zeros(3))
