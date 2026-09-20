"""Static structural inputs preserve tracked atoms, support and material scale."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.analysis.structural_adapter import snapshot_batch
from src.data.structural_pretraining.prepare import REFERENCE_RADIUS
from src.models.encoders.structural import ATOMIC_NUMBERS


def test_latest_checkpoint_extracts_exact_encoder_and_rejects_manifest_drift(tmp_path):
    import hashlib
    import json
    import torch
    from src.analysis.structural_adapter import _checkpoint_encoder
    release = tmp_path/'release'
    release.mkdir()
    manifest = release/'manifest.json'
    manifest.write_text(json.dumps({'scales': {'Al': 9.1}}))
    checkpoint = tmp_path/'last.pt'
    weight = torch.arange(6).reshape(2, 3)
    torch.save({'step': 400, 'identity': {'protocol': 'shared_pretraining_local_gatr_bond_v11',
        'config': {'release': str(release), 'architecture': 'gatr', 'history_frames': 1}},
        'model': {'encoder.test.weight': weight, 'physical.test.weight': weight+1}}, checkpoint)
    config = {'checkpoint': str(checkpoint), 'checkpoint_kind': 'latest',
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'release_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()}
    saved = _checkpoint_encoder(config)
    assert saved['step'] == 400 and saved['scales'] == {'Al': 9.1}
    assert set(saved['encoder']) == {'test.weight'}
    torch.testing.assert_close(saved['encoder']['test.weight'], weight, rtol=0, atol=0)
    manifest.write_text(json.dumps({'scales': {'Al': 10.}}))
    with pytest.raises(ValueError, match='release manifest changed'):
        _checkpoint_encoder(config)


def test_physical_support_center_species_and_fixed_scaling():
    points = np.array([[-40,-40,-40], [3.5,0,0], [0,0,0], [7,0,0],
                       [9,0,0], [40,40,40]], dtype=np.float32)
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


def test_mace_packing_preserves_cutoff_edges_without_padding_or_cross_graph_edges():
    points = np.array([[-60,-60,-60], [0,0,0], [4,0,0], [7,0,0],
                       [9,0,0], [13,0,0], [15,0,0], [16,0,0], [24,0,0], [60,60,60]], dtype=np.float32)
    batch = snapshot_batch(points, cKDTree(points), points[[1,7]],
                           scale=REFERENCE_RADIUS, material='Al', architecture='mace')
    np.testing.assert_array_equal(batch['packed_positions'][:,0], [0,4,7,-7,-3,-1,0])
    np.testing.assert_array_equal(batch['node_graph'], [0,0,0,1,1,1,1])
    np.testing.assert_array_equal(batch['packed_species'], ATOMIC_NUMBERS.index(13))
    np.testing.assert_allclose(batch['packed_weights'], [1,1,.5,.5,1,1,1])
    expected = {(0,1),(1,2),(3,4),(4,5),(4,6),(5,6)}
    expected |= {(j,i) for i,j in expected}
    assert set(map(tuple, batch['edges'].numpy().T)) == expected
    np.testing.assert_array_equal(batch['centers'], [0,3])
    np.testing.assert_array_equal(batch['packed_centers'], [0,6])
    assert batch['positions'].shape == (2,1,4,3)
