import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.training_methods.bcr.model import BCR
from src.training_methods.bcr.data import pack, corrupt
from src.research.bcr_followup.common import root_split, fixed_batch
from src.research.bcr_followup.decoders import train_decoder
from src.research.bcr_followup.interventions import intervention_codes, fit_constant
from src.research.bcr_followup.readouts import fit_residual_probe, predict_probe, target_groups


torch.set_num_threads(1)


def small_model():
    torch.manual_seed(53)
    return BCR(dict(arm='bcr', encoder=dict(d0=2., n_ref=8., radius=4., channels=4, code_dim=8, cutoff=3.),
                    decoder=dict(irreps='4x0e + 3x1o + 2x2e')))


def patches_records():
    rng = np.random.default_rng(4)
    patches = [np.vstack((np.zeros((1, 3)), rng.normal(size=(8+i%3, 3)))).astype(np.float32) for i in range(8)]
    records = [dict(root=f'root{i//2}', source=i//2, block=0) for i in range(8)]
    return patches, records


def test_splits_keep_development_out_of_probe_tuning():
    records = [dict(root=f'{i:02d}', split='train' if i < 12 else 'development') for i in range(18)]
    result = root_split(records)
    assert result['tuning_roots'] == ['10', '11']
    assert not set(result['tune']) & set(result['development'])
    with pytest.raises(ValueError, match='independent'):
        root_split([*records[:-1], dict(root='00', split='development')])


def test_interventions_include_exact_code_mean_and_zero_variation():
    z = torch.arange(24.).reshape(3, 8); mean = z.mean(0); optimized = mean+1
    codes = intervention_codes(z, mean, optimized, [0, .5, 1, 2], {'donor': np.array([1, -1, 0])})
    torch.testing.assert_close(codes['alpha_0'], mean.expand_as(z))
    torch.testing.assert_close(codes['alpha_1'], z)
    torch.testing.assert_close(codes['alpha_2']-mean, 2*(z-mean))
    torch.testing.assert_close(codes['optimized_constant'][0], optimized)
    torch.testing.assert_close(codes['donor'][0], z[1])


def test_original_corruption_bank_replay():
    patches, _ = patches_records()
    clean, noisy, epsilon, sigma = fixed_batch(patches, 2, 6, 1, 1, [.04, .12], 2., 'cpu')
    for j, index in enumerate(range(2, 6)):
        one = {k: v[j:j+1] for k, v in clean.items()}
        y, e, s, _ = corrupt(one, [.04, .12], 2., torch.Generator().manual_seed(731+100003*index+1009+1), torch.tensor([1]))
        assert torch.equal(epsilon[j:j+1], e)
        assert torch.equal(noisy['positions'][j:j+1], y['positions'])
        assert torch.equal(sigma[j:j+1], s)


def test_residual_zero_step_keeps_ridge_and_uses_no_development_labels():
    rng = np.random.default_rng(7); x = rng.normal(size=(60, 4)); y = x@rng.normal(size=(4, 3))
    cfg = dict(width=12, learning_rate=.001, weight_decay=.0001, updates=0, batch_size=16, evaluate_every=2)
    a = fit_residual_probe(x, y, np.arange(30), np.arange(30, 45), cfg, 13)
    yy = y.copy(); yy[45:] += 1000
    b = fit_residual_probe(x, yy, np.arange(30), np.arange(30, 45), cfg, 13)
    pa, pr = predict_probe(a, x[45:], 'cpu')
    np.testing.assert_array_equal(pa, pr)
    np.testing.assert_array_equal(a['ridge'], b['ridge'])
    assert a['selected_step'] == 0 and a['selected_tuning_mse'] == a['ridge_tuning_mse']


def test_residual_selection_never_loses_ridge_on_tuning():
    rng = np.random.default_rng(7); x = rng.normal(size=(60, 4)); y = np.square(x[:, :2])
    cfg = dict(width=12, learning_rate=.03, weight_decay=.0001, updates=12, batch_size=16, evaluate_every=2)
    result = fit_residual_probe(x, y, np.arange(30), np.arange(30, 45), cfg, 13)
    assert result['selected_tuning_mse'] <= result['ridge_tuning_mse']


def test_target_breakdown_matches_descriptor_columns():
    assert sorted(sum(target_groups('radial').values(), [])) == list(range(17))
    assert list(target_groups('angular')) == ['q4', 'w4', 'q6', 'w6']
    assert list(target_groups('rich')) == ['l0', 'l2', 'l4', 'l6']
    assert sorted(sum(target_groups('rich').values(), [])) == list(range(144))


def test_fresh_decoder_freezes_encoder_matches_initialization_and_resumes(tmp_path):
    patches, records = patches_records(); a = small_model(); b = copy.deepcopy(a)
    with torch.no_grad(): codes = a.encode(pack(patches)).numpy()
    encoder = copy.deepcopy(a.encoder.state_dict()); initial = copy.deepcopy(a.decoder.state_dict())
    cfg = dict(updates=4, learning_rate=.001, minimum_learning_rate=.0001, batch_size=4, microbatch=2, save_every=2)
    common = (initial, codes, patches, records, list(range(8)), [.04, .12], cfg, 71)
    train_decoder(a, *common, tmp_path/'a', 'identity', 'cpu')
    train_decoder(b, *common, tmp_path/'b', 'identity', 'cpu', stop_after=2)
    train_decoder(b, *common, tmp_path/'b', 'identity', 'cpu')
    for key, value in a.decoder.state_dict().items():
        torch.testing.assert_close(value, b.decoder.state_dict()[key], rtol=0, atol=0)
    for key, value in a.encoder.state_dict().items():
        torch.testing.assert_close(value, encoder[key], rtol=0, atol=0)
    states = [torch.load(tmp_path/p/'last.pt', weights_only=False) for p in ('a', 'b')]
    assert states[0]['stream'] == states[1]['stream']
    assert torch.equal(states[0]['noise_rng'], states[1]['noise_rng'])


def test_optimized_constant_changes_only_one_global_code(tmp_path):
    patches, records = patches_records(); model = small_model().eval().requires_grad_(False)
    with torch.no_grad(): codes = model.encode(pack(patches)).numpy()
    before = copy.deepcopy(model.state_dict())
    study = SimpleNamespace(patches=patches, records=records, identity='constant-test',
        manifest=dict(d0=2., radius_A=4., noise_levels=[.12]),
        split=dict(train=list(range(8)), fit=list(range(4)), tune=list(range(4, 8)), tuning_roots=['root2','root3']),
        config=dict(seed=5, constant=dict(updates=2, batch_size=4, microbatch=2, learning_rate=.01, evaluate_every=1, tuning_anchors=4)))
    optimized, mean = fit_constant(study, model, codes, tmp_path, 'cpu', None)
    assert optimized.shape == mean.shape == (8,)
    for key, value in model.state_dict().items(): torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters())


def test_constant_resume_completes_interrupted_final_tuning(tmp_path, monkeypatch):
    from src.research.bcr_followup import interventions
    patches, records = patches_records(); model = small_model().eval().requires_grad_(False)
    with torch.no_grad(): codes = model.encode(pack(patches)).numpy()
    study = SimpleNamespace(patches=patches, records=records, identity='resume-constant',
        manifest=dict(d0=2., radius_A=4., noise_levels=[.12]),
        split=dict(train=list(range(8)), fit=list(range(4)), tune=list(range(4, 8)), tuning_roots=['root2','root3']),
        config=dict(seed=5, constant=dict(updates=1, batch_size=4, microbatch=2, learning_rate=.01, evaluate_every=1, tuning_anchors=4)))
    score = interventions.constant_score
    calls = 0
    def interrupt(*args):
        nonlocal calls
        calls += 1
        if calls == 2: raise TimeoutError('Interrupted selection')
        return score(*args)
    monkeypatch.setattr(interventions, 'constant_score', interrupt)
    with pytest.raises(TimeoutError): fit_constant(study, model, codes, tmp_path, 'cpu', None)
    monkeypatch.setattr(interventions, 'constant_score', score)
    fit_constant(study, model, codes, tmp_path, 'cpu', None)
    state = torch.load(tmp_path/'constant-last.pt', weights_only=False)
    assert [p['step'] for p in state['trace']] == [0, 1]
