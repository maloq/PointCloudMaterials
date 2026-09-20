import numpy as np
import pytest

from src.data.structural_pretraining.support import local_crop
from src.research.gatr_conditional_information.comparison_data import pack_clouds
from src.research.trajectory_stability.encode import observation


@pytest.mark.parametrize('architecture', ['gatr', 'mace'])
def test_prepared_cropped_cloud_preserves_native_observation(architecture):
    rng = np.random.default_rng(73)
    directions = rng.normal(size=(120, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cloud = np.vstack((np.zeros((1, 3)), directions*np.linspace(2., 9., 120)[:, None])).astype(np.float32)
    order = rng.permutation(len(cloud))
    cloud = cloud[order]
    center = int(np.flatnonzero(order == 0).item())
    scale = 9.121389139452193
    packed, covariates, error = pack_clouds([cloud], [center], scale, 33)
    expected = observation(cloud, center, scale, architecture)
    actual = observation(packed['positions'], int(packed['center_indices'][0]), scale, architecture)
    for key in ('positions', 'weights', 'center', 'log_scale'):
        np.testing.assert_array_equal(expected[key], actual[key])
    if architecture == 'mace':
        np.testing.assert_array_equal(expected['edges'], actual['edges'])
    radius = np.sort(np.linalg.norm(cloud[local_crop(cloud, scale)[1]].astype(float), axis=1))
    reconstructed = np.sort(np.linalg.norm(packed['radial_positions'].astype(float), axis=1))
    np.testing.assert_allclose(radius, reconstructed, atol=1e-6, rtol=0)
    assert covariates.shape == (1, 38) and error < 1e-6
