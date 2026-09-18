"""Synthetic controls for the directional research measurements."""
import itertools

import numpy as np
from scipy.spatial.transform import Rotation
import torch

from src.research.gatr_equivariant.metrics import angular_metrics, phase_shuffle_expectation
from src.research.gatr_equivariant.model import vector_parts


def test_angles_distinguish_flip_axis_and_weak_vector():
    a = np.array([[1.,0,0],[0,1,0],[0,0,0]])
    b = np.array([[-1.,0,0],[1,0,0],[1,0,0]])
    result = angular_metrics(a,b,.1)
    assert result['valid_pairs'] == 2
    assert result['coverage'] == 2/3
    assert result['mean_angle_deg'] == 135
    assert result['mean_axis_angle_deg'] == 45
    assert result['flip90_fraction'] == .5
    assert angular_metrics(a*0,b,.1)['p1'] is None


def test_shuffle_matches_exhaustive_phase_permutations():
    u = np.random.default_rng(21).normal(size=(5,3))
    u /= np.linalg.norm(u,axis=-1,keepdims=True)
    labels = np.array([0,0,1,1,1])
    left,right = np.triu_indices(5,1)
    expected1,expected2 = phase_shuffle_expectation(u,labels,left,right)
    observed = []
    for a,b in itertools.product(itertools.permutations(range(2)),itertools.permutations(range(2,5))):
        v = u[list(a)+list(b)]
        observed.append(np.sum(v[left]*v[right],axis=-1))
    observed = np.array(observed)
    np.testing.assert_allclose(expected1,observed.mean(0),atol=1e-14)
    np.testing.assert_allclose(expected2,((3*observed**2-1)/2).mean(0),atol=1e-14)


def test_all_four_triplets_follow_pga_rotor():
    from gatr.interface import embed_rotation
    from gatr.primitives import geometric_product, reverse
    random = np.random.default_rng(8)
    mv = torch.tensor(random.normal(size=(6,16)),dtype=torch.float32)
    rotation = Rotation.random(random_state=8)
    rotor = embed_rotation(torch.tensor(rotation.as_quat(),dtype=torch.float32))
    actual = vector_parts(geometric_product(geometric_product(rotor,mv),reverse(rotor)))
    expected = vector_parts(mv).numpy()@rotation.as_matrix().T
    np.testing.assert_allclose(actual.numpy(),expected,atol=2e-6)


def test_spatial_phase_filter_preserves_periodic_pair_distance():
    from src.research.gatr_equivariant.metrics import spatial_metrics
    x = np.array([[.1,0,0],[9.9,0,0],[.4,0,0]])
    v = np.tile([1.,0,0],(3,1)); labels = np.array([1,1,0])
    rows = spatial_metrics(v,.1,labels,x,np.array([10.,10.,10.]),[0,1],np.eye(3),phase_code=1)
    assert len(rows)==1 and rows[0]['pairs']==1
    assert abs(rows[0]['distance_mean_A']-.2)<1e-12
    assert rows[0]['p1']==1 and rows[0]['excess_p1']==0


def test_cage_rotation_recovers_proper_rigid_motion():
    from src.research.gatr_equivariant.run import cage_rotations
    rng = np.random.default_rng(15)
    x = rng.normal(size=(80,3)); x[0] = 0
    r = Rotation.random(random_state=3).as_matrix().T
    a = dict(frames=np.arange(2),centers=np.array([0]),
        nearest_ids=np.tile(np.arange(80),(2,1)),offsets=np.array([0,80,160]))
    result = cage_rotations(a,np.concatenate((x,x@r)),np.tile(np.arange(80),2))
    np.testing.assert_allclose(result['cage_rotation'][0,0],r,atol=5e-8)
    assert result['cage_fit_rms_A'].max()<1e-12
