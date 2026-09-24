import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from ase.build import bulk

from src.research.geoframe_evolution.reference import bond_descriptors, contexts
from src.research.geoframe_evolution.metrics import boundary_coherence, participation


def lattice_bonds(symbol, structure, k):
    points = bulk(symbol, structure, a=4., cubic=True).repeat((7, 7, 7)).positions
    tree = cKDTree(points)
    _, central = tree.query(points.mean(0), k=1)
    _, core = tree.query(points[[central]], k=k+1)
    _, neighbors = tree.query(points[core], k=k+1)
    return points[neighbors[:, :, 1:]]-points[core][:, :, None]


def test_bond_order_ideal_lattices_and_rotation():
    for symbol, structure, k, q4, q6 in [('Al', 'fcc', 12, .190940654, .57452426),
                                       ('Ta', 'bcc', 14, .036369648, .510688231)]:
        vectors = lattice_bonds(symbol, structure, k)
        order, counts = bond_descriptors(vectors)
        np.testing.assert_allclose(order[0, :2], [q4, q6], atol=1e-7)
        np.testing.assert_allclose(order[0, 4], q6, atol=1e-7)
        assert (counts == k).all()
        rotated, rc = bond_descriptors(vectors@Rotation.from_rotvec([.7, -.3, 1.2]).as_matrix())
        np.testing.assert_allclose(order, rotated, atol=1e-7)
        np.testing.assert_array_equal(counts, rc)


def test_defect_context_is_material_specific():
    ptm = np.array([1, 2, 0, 0, 0, 4])
    fraction = np.array([1., 1., .9, .5, 0., 0.])
    fault = np.array([0, 3, 0, 0, 0, 0])
    order = np.zeros((6, 7)); order[4, 4] = .3
    al = contexts(ptm, fraction, fault, order, .25, 'Al', -.08)
    zr = contexts(ptm, fraction, fault, order, .25, 'Zr', -.08)
    np.testing.assert_array_equal(al, [1, 3, 4, 2, 6, 5])
    assert zr[1] == 1  # HCP Zr is not automatically an FCC planar defect.


def test_spatial_collapse_scores_chance_and_faithful_order_wins():
    rng = np.random.default_rng(8); n = 2000
    a = rng.integers(0, 3, n); b = rng.integers(0, 3, n)
    labels = np.r_[a, b]; z = np.eye(3)[labels]+rng.normal(scale=.01, size=(2*n, 3))
    distances = np.ones(n)
    kwargs = dict(labels=labels, pair_distance=distances, n=n, mask=np.ones(n, bool))
    assert boundary_coherence(z, **kwargs)['auc'] > .99
    assert boundary_coherence(np.zeros_like(z), **kwargs)['auc'] == .5
    assert participation(np.zeros_like(z))['effective_rank'] == 0
    shuffled = boundary_coherence(z[rng.permutation(2*n)], **kwargs)['auc']
    assert .45 < shuffled < .55
