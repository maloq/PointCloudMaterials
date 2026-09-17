"""Static structural inputs preserve tracked atoms, support and material scale."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.analysis.structural_adapter import snapshot_batch
from src.data.structural_pretraining.prepare import REFERENCE_RADIUS
from src.models.encoders.structural import ATOMIC_NUMBERS


def test_physical_support_center_species_and_fixed_scaling():
    points = np.array([[-40,-40,-40], [8,0,0], [0,0,0], [16,0,0],
                       [18,0,0], [40,40,40]], dtype=np.float32)
    batch = snapshot_batch(points, cKDTree(points), points[[2]],
                           scale=REFERENCE_RADIUS, material='Al')
    np.testing.assert_array_equal(batch['positions'][0,0], points[[1,2,3]])
    assert batch['centers'].item() == 1
    np.testing.assert_array_equal(batch['species'], ATOMIC_NUMBERS.index(13))
    np.testing.assert_allclose(batch['weights'][0,0], [1,1,.5])
    assert batch['log_scale'].item() == 0
    scaled = snapshot_batch(points, cKDTree(points), points[[2]],
                            scale=REFERENCE_RADIUS/2, material='Al')
    np.testing.assert_array_equal(scaled['positions'][0,0], points[[1,2]]*2)
    np.testing.assert_allclose(scaled['weights'][0,0], [.5,1])
    np.testing.assert_allclose(scaled['log_scale'], np.log(.5))


def test_truncated_boundaries_and_nonatom_centers_are_rejected():
    points = np.array([[-40,-40,-40], [0,0,0], [39,0,0], [40,40,40]], dtype=np.float32)
    tree = cKDTree(points)
    with pytest.raises(ValueError, match='cannot support'):
        snapshot_batch(points, tree, points[[2]], scale=REFERENCE_RADIUS, material='Al')
    with pytest.raises(ValueError, match='exact source atoms'):
        snapshot_batch(points, tree, np.array([[.1,0,0]]), scale=REFERENCE_RADIUS, material='Al')
