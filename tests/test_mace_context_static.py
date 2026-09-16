"""Full-frame reuse preserves graph ancestry and tracked-center readouts."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.analysis.mace_context_adapter import _pool, encode_frame, node_graph


def test_arbitrary_targets_preserve_order_and_exact_two_hop_ancestors():
    points = np.array([[0,0,0], [4,0,0], [8,0,0], [12,0,0], [40,0,0]], dtype=float)
    graph = node_graph(points, cKDTree(points), np.array([3,0]), 'cpu')
    # Both requested atoms need exactly the four connected nodes, excluding 40 A.
    np.testing.assert_allclose(graph.positions.numpy(), points[:4]-points[:4].mean(0))
    np.testing.assert_array_equal(graph.second_keep.numpy(), [3,0])
    np.testing.assert_array_equal(graph.pool_index.numpy(), [0,1])
    assert graph.first_edges.shape[1] == 6
    assert graph.second_edges.shape[1] == 2


def test_center_readout_is_atom_matched_and_taper_excludes_outer_atom():
    points = np.array([[0,0,0], [4,0,0], [7,0,0], [20,0,0]], dtype=float)
    features = np.repeat(np.array([2.,4.,1000.,999.])[:,None], 256, axis=1)
    result = _pool(points, points[[0]], cKDTree(points), features, 5., 7.)
    np.testing.assert_allclose(result[0,:256], 3.)
    np.testing.assert_allclose(result[0,256:], 2.)


def test_static_edges_require_full_message_support():
    points = np.array([[0,0,0], [40,40,40], [16,20,20]], dtype=float)
    settings = dict(coordinate_scale=1., inner_radius_A=5., outer_radius_A=7.)
    with pytest.raises(ValueError, match='cannot support 17.0'):
        encode_frame(None, points, points[[2]], settings)
