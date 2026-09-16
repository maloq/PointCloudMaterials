import numpy as np
import pytest
import torch
from pathlib import Path

from src.research.mace_tda_ridge_audit.math import balanced_errors, ridge_predict, ridge_path
from src.analysis.topology_metrics import ridge_predictions


def test_independent_ridge_matches_repository_with_constant_channel_and_offset():
    rng = np.random.default_rng(701)
    x = rng.normal(size=(101, 9))
    x[:, -1] = 3
    y = x @ rng.normal(size=(9, 144)) + 100
    train = np.arange(73)
    test = np.arange(73, 101)
    actual = ridge_predict(x[train], y[train], x[test])
    expected = ridge_predictions(x, y, train, test, 1.)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    y[test] = -1e9
    np.testing.assert_allclose(ridge_predictions(x, y, train, test, 1.), actual, rtol=1e-12, atol=1e-12)


def test_balancing_weights_each_homology_block_equally():
    target = np.zeros((2, 144))
    prediction = np.zeros_like(target)
    prediction[0, :16] = 2
    prediction[1, 16:80] = 3
    row_errors, blocks = balanced_errors(prediction, target, [2, 3, 4])
    np.testing.assert_array_equal(blocks, [[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(row_errors, [1/3, 1/3])


def test_svd_path_matches_independent_ridge_and_handles_nearly_duplicate_features():
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(719)
    x = rng.normal(size=(80, 8))
    x[:, -1] = x[:, 0] + 1e-7*rng.normal(size=80)
    y = rng.normal(size=(80, 144))
    path = ridge_path(x[:60], y[:60], x[60:], [1e-9, 1., 100.])
    scale = StandardScaler().fit(x[:60])
    for alpha, actual in path.items():
        expected = Ridge(alpha=alpha, solver='svd').fit(scale.transform(x[:60]),y[:60]).predict(scale.transform(x[60:]))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(path[1.],ridge_predict(x[:60],y[:60],x[60:]),rtol=1e-10,atol=1e-10)


def test_direct_geometry_uses_distances_and_agrees_for_different_coordinate_spaces():
    from src.research.mace_tda_ridge_audit.direct import compare_geometry, distances
    rng = np.random.default_rng(317)
    x = rng.normal(size=(29, 3))
    rotation = np.linalg.qr(rng.normal(size=(3,3)))[0]
    embedding = np.column_stack([7*x@rotation,np.zeros((len(x),4))])
    pairs = np.column_stack(np.triu_indices(len(x),k=1))
    target = {'H0':distances(x,'euclidean')}
    score,_ = compare_geometry(distances(embedding,'euclidean'),target,pairs,3,rng.permutation(len(x)))
    assert score[0]['spearman'] == pytest.approx(1.)
    assert score[0]['neighbor_overlap'] == 1.
    assert score[0]['chance_neighbor_overlap'] == 3/28


def test_direct_shuffle_remaps_neighbor_identities_exactly_and_excludes_self():
    from src.research.mace_tda_ridge_audit.direct import compare_geometry, distances, neighbors
    rng = np.random.default_rng(719)
    x = rng.normal(size=(41,5))
    matrix = distances(x,'cosine')
    target = {'H1':distances(rng.normal(size=(41,6)),'euclidean')}
    pairs = np.column_stack(np.triu_indices(len(x),k=1))
    permutation = rng.permutation(len(x))
    score,_ = compare_geometry(matrix,target,pairs,4,permutation)
    explicit,_ = compare_geometry(matrix[np.ix_(permutation,permutation)],target,pairs,4,np.arange(len(x)))
    assert score[0]['shuffled_spearman'] == explicit[0]['spearman']
    assert score[0]['shuffled_neighbor_overlap'] == explicit[0]['neighbor_overlap']
    indices,_ = neighbors(matrix,4)
    assert not np.any(indices==np.arange(len(x))[:,None])


def test_direct_neighbor_ties_are_explicit_and_zero_cosine_vectors_fail():
    from src.research.mace_tda_ridge_audit.direct import distances, neighbors
    x = np.array([[0.],[1.],[1.],[1.],[2.]])
    indices,ties = neighbors(distances(x,'euclidean'),2)
    np.testing.assert_array_equal(indices[0],[1,2])
    assert ties>0
    with pytest.raises(ValueError,match='zero embedding'):
        distances(x,'cosine')


@pytest.mark.skipif(not Path('output/pretrained_mace_spatiotemporal_20260906/mace_mp_0b2_small.model').exists(),
                    reason='Native initialization integration test requires the retained MLIP artifact')
def test_random_mace_rebuild_is_reproducible_and_copies_no_learned_weights():
    from src.research.mace_tda_ridge_audit.initialization import random_native, state_digest

    pretrained = torch.load('output/pretrained_mace_spatiotemporal_20260906/mace_mp_0b2_small.model',
                            map_location='cpu', weights_only=False).float()
    original = state_digest(pretrained)
    first, audit = random_native(pretrained, 701)
    repeated, _ = random_native(pretrained, 701)
    independent, _ = random_native(pretrained, 702)
    assert state_digest(first) == state_digest(repeated)
    assert state_digest(first) != state_digest(independent)
    assert state_digest(pretrained) == original
    assert all(not p['equal_to_mlip'] for p in audit['parameters'].values())
    assert torch.count_nonzero(first.atomic_energies_fn.atomic_energies) == 0
    torch.testing.assert_close(first.scale_shift.scale, torch.ones_like(first.scale_shift.scale))
    torch.testing.assert_close(first.scale_shift.shift, torch.zeros_like(first.scale_shift.shift))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Non-default GPU loading requires two GPUs')
def test_checkpoint_loader_constructs_on_requested_gpu_and_preserves_state(tmp_path):
    from src.utils.model_utils import load_model_from_checkpoint

    class CurrentDeviceModule(torch.nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.constructed_device = torch.cuda.current_device()
            self.weight = torch.nn.Parameter(torch.zeros(89, device='cuda'))
            self.register_buffer('numbers', torch.zeros(89, dtype=torch.long, device='cuda'))
            self.cpu_head = torch.nn.Linear(3, 2)

    saved = dict(weight=torch.arange(89, dtype=torch.float32), numbers=torch.arange(1, 90),
                 **{'cpu_head.weight': torch.ones(2, 3), 'cpu_head.bias': torch.arange(2, dtype=torch.float32)})
    checkpoint = tmp_path/'model.ckpt'
    torch.save({'state_dict': saved}, checkpoint)
    with torch.cuda.device(0):
        model = load_model_from_checkpoint(str(checkpoint), None, device='cuda:1', module=CurrentDeviceModule)
        assert torch.cuda.current_device() == 0
    assert model.constructed_device == 1
    for name, actual in model.state_dict().items():
        assert actual.device == torch.device('cuda:1')
        torch.testing.assert_close(actual.cpu(), saved[name], rtol=0, atol=0)
