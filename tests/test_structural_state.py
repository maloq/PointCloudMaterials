import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.training_methods.bcr.data import pack
from src.training_methods.bcr.model import Encoder
from src.research.structural_state.data import graph_arrays, splits, PairStream
from src.research.structural_state.model import StructuralModel, GraphBank, objective, teacher_distances

torch.set_num_threads(1)


def example():
    rng = np.random.default_rng(42)
    patches = [np.vstack((np.zeros((1, 3)), rng.normal(size=(8+i, 3)))).astype(np.float32) for i in range(4)]
    torch.manual_seed(51)
    model = StructuralModel(dict(d0=2., n_ref=10., radius=5., cutoff=3., channels=4, code_dim=12, backend='e3nn'))
    bank = GraphBank(graph_arrays(patches, 3.), model.encoder, 'cpu')
    return patches, model, bank


def test_cached_graph_matches_native_mace_outputs_and_gradients():
    patches, model, bank = example()
    ids = [3, 0, 3, 1]
    p = model.encoder.pooled_graph(bank.batch(ids))
    q = Encoder.pooled(model.encoder, pack([patches[i] for i in ids]))
    torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
    named = list(model.encoder.parameters())
    first = torch.autograd.grad(p.square().sum(), named, allow_unused=True)
    second = torch.autograd.grad(q.square().sum(), named, allow_unused=True)
    for a, b in zip(first, second, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


def test_encoder_rotation_and_permutation_invariance():
    patches, model, bank = example()
    rng = np.random.default_rng(77)
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0].astype(np.float32)
    changed = [np.vstack((p[:1], p[1:][::-1])) @ rotation for p in patches]
    other = GraphBank(graph_arrays(changed, 3.), model.encoder, 'cpu')
    torch.testing.assert_close(model(bank.batch([0, 1, 2, 3])), model(other.batch([0, 1, 2, 3])), atol=2e-6, rtol=2e-5)


def test_microbatch_keeps_full_pair_objective_and_gradient():
    _, model, bank = example()
    torch.manual_seed(9)
    targets = {d: torch.randn(4, 89) for d in ('observed', 'relaxed')}
    arm = dict(input='observed', relaxed_weight=.25, relation_weight=.1)
    z = model(bank.batch([0, 1, 2, 3]))
    loss, _ = objective(model, z, targets, arm, (.1, 1.))
    loss.backward()
    reference = {n: None if p.grad is None else p.grad.clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    losses = []
    for ix in ([0, 1], [2, 3]):
        part, _ = objective(model, model(bank.batch(ix)), {d: v[ix] for d, v in targets.items()}, arm, (.1, 1.))
        (part / 2).backward()
        losses.append(part.detach() / 2)
    torch.testing.assert_close(sum(losses), loss.detach(), atol=1e-6, rtol=2e-5)
    for n, p in model.named_parameters():
        if reference[n] is not None:
            torch.testing.assert_close(p.grad, reference[n], atol=2e-6, rtol=3e-5)


def test_teacher_loss_changes_main_encoder_and_uses_fixed_targets():
    _, model, bank = example()
    targets = {d: torch.randn(4, 89) for d in ('observed', 'relaxed')}
    loss, _ = objective(model, model(bank.batch([0, 1, 2, 3])), targets,
                        dict(input='observed', relaxed_weight=.25, relation_weight=0.), (1., 1.))
    loss.backward()
    assert model.encoder.center_embedding.weight.grad.norm() > 0
    assert model.encoder.readout[-1].weight.grad.norm() > 0
    assert model.heads['observed'].weight.grad.norm() > 0
    assert model.heads['relaxed'].weight.grad.norm() > 0
    assert all(v.grad is None for v in targets.values())


def test_relation_teacher_equal_block_weights():
    x = torch.zeros(2, 89)
    x[1, :17] = 1
    torch.testing.assert_close(teacher_distances(x), torch.tensor([(1/3)**.5]))


def test_export_preserves_pooled_signal_and_has_nonzero_residual_gradient():
    _, model, bank = example()
    p = model.encoder.pooled_graph(bank.batch([0, 1, 2, 3]))
    with torch.no_grad():
        model.encoder.pooled_mean.copy_(p.mean(0))
        model.encoder.pooled_scale.copy_(p.std(0, correction=0).clamp_min(1e-6))
    z = model.encoder.export_pooled(p)
    n = 2 * model.encoder.channels
    recovered = z[:, :n] * model.encoder.pooled_scale + model.encoder.pooled_mean
    torch.testing.assert_close(recovered, p)
    z[:, n:].square().sum().backward()
    assert model.encoder.readout[-1].weight.grad.norm() > 0


def test_fixed_distance_scale_resists_amplitude_shrinkage():
    from src.research.structural_state.model import pair_distances
    original = torch.tensor([[0., 0.], [1., 2.], [2., 1.], [3., 4.]])
    reference = pair_distances(original).mean()
    factor = torch.tensor(.1, requires_grad=True)
    loss = torch.nn.functional.huber_loss(pair_distances(factor * original) / reference,
                                         pair_distances(original) / reference)
    loss.backward()
    assert factor.grad < 0  # descent restores amplitude; denominator cannot chase it
    torch.testing.assert_close(reference, pair_distances(original).mean())


def test_fit_only_head_calibration_is_bounded_and_reconstructs():
    from src.research.structural_state.model import calibrate_heads
    _, model, _ = example()
    torch.manual_seed(19)
    x = torch.randn(40, 12)
    y = x @ torch.randn(12, 89) * .1 + 1.2
    receipt = calibrate_heads(model, x, {d: y for d in model.heads}, ridge=.01, maximum=10.)
    for domain, head in model.heads.items():
        assert head.weight.norm() <= 10.
        assert ((head(x)-y)**2).mean() < 1e-5
        assert receipt[domain]['ridge'] == .01


def test_untrained_hazard_is_exact_constant_control():
    from src.research.structural_state.evaluation import hazard_probe
    corpus = SimpleNamespace(split={'fit': np.arange(4), 'tune': np.arange(4,8),
                                   'development': np.arange(8,12)},
        records=[dict(source=i//2) for i in range(12)],
        targets={'at_risk': np.ones(12, bool), 'event_bin': np.array([2,5,5,5]*3),
                 'delay_ps': np.array([4.5,20,20,20]*3)})
    config = dict(seed=1, probes=dict(width=8, learning_rate=.001, weight_decay=.0001,
                                    updates=0, batch_size=4, evaluate_every=1))
    rng = np.random.default_rng(3)
    for kind in ('linear', 'mlp'):
        metrics, _, predictions = hazard_probe(rng.normal(size=(12,8)).astype(np.float32),
            np.ones((12,1), np.float32), corpus, config, kind, 'cpu', None)
        assert metrics['best_step'] == 0
        np.testing.assert_array_equal(predictions['logits'], np.broadcast_to(predictions['logits'][0], (4,5)))
        assert metrics['horizons']['12.0']['average_precision'] == pytest.approx(.25)


def test_source_splits_and_pair_stream_resume():
    records = [dict(root=f'r{i//4}', source=i//4, frame=i%2, center_atom_id=i,
                    split='fit' if i < 12 else 'tune' if i < 16 else 'development', temperature_K=400)
               for i in range(20)]
    ss = splits(records)
    phase = np.zeros(20, dtype=int)
    first = PairStream(records, ss['fit'], phase, 1)
    first.draw(16)
    state = copy.deepcopy(first.state_dict())
    expected = first.draw(16)
    resumed = PairStream(records, ss['fit'], phase, 999)
    resumed.load_state_dict(state)
    np.testing.assert_array_equal(resumed.draw(16), expected)
    assert set(expected) <= set(ss['fit'])
    for a, b in zip(expected[::2], expected[1::2], strict=True):
        assert records[a]['root'] != records[b]['root']
    records[-1]['root'] = records[0]['root']
    with pytest.raises(ValueError, match='ancestry'):
        splits(records)


@pytest.mark.parametrize('dynamics',[False,True])
def test_runtime_resume_matches_uninterrupted_training(tmp_path,dynamics):
    from src.research.structural_state.common import sha
    from src.research.structural_state.runtime import train
    from src.research.structural_state.data import Corpus
    patches, _, _ = example()
    cache = tmp_path / 'cache'
    cache.mkdir()
    records = [dict(root=f'r{i//4}', source=i//4, frame=i%2, center_atom_id=i,
                    split='fit' if i < 12 else 'tune' if i < 16 else 'development', temperature_K=400)
               for i in range(20)]
    (cache / 'records.json').write_text(json.dumps(records))
    rng = np.random.default_rng(8)
    targets = {'phase': np.zeros(20, dtype=int)}
    if dynamics:
        targets.update(current_order=rng.normal(size=(20,8)).astype(np.float32),
                       future_order_9=rng.normal(size=(20,8)).astype(np.float32))
    for domain in ('observed', 'relaxed'):
        targets[domain + '_radial'] = rng.normal(size=(20, 17)).astype(np.float32)
        targets[domain + '_rich'] = rng.normal(size=(20, 144)).astype(np.float32)
        np.savez(cache / f'{domain}-graphs.npz', **graph_arrays(patches*5, 3.))
    np.savez(cache / 'targets.npz', **targets)
    manifest = dict(d0=2., n_ref=10., files={p.name: sha(p) for p in cache.iterdir()})
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    arm = dict(name='test', input='observed', relaxed_weight=.25, relation_weight=.1)
    config = dict(seed=53, encoder=dict(channels=4, code_dim=12, radius=5., cutoff=3., backend='e3nn'),
        training=dict(updates=4, batch_size=8, microbatch=4, encoder_lr=.001, head_lr=.001,
            weight_decay=.0001, warmup=2, gradient_clip=5., head_norm_bound=10.,
            evaluate_every=2, save_every=2, log_every=2, head_ridge=1.,
            relation_warmup=2, minimum_spread_ratio=.01), retention_tolerance_relative_mse=.02)
    if dynamics:
        config.update(protocol='fixed_geometry_future_relation_v3',dynamics=dict(lag_ps=9.,baseline_ridge=1.))
        arm.update(current_weight=.25,future_weight=.25)
    study = SimpleNamespace(cache=cache, config=config, identity='test', arm=lambda name: arm)
    assert train(study, 'test', 'cpu', directory=tmp_path / 'continuous')
    assert not train(study, 'test', 'cpu', stop_after=2, directory=tmp_path / 'resumed')
    assert train(study, 'test', 'cpu', directory=tmp_path / 'resumed')
    a = torch.load(tmp_path / 'continuous/last.pt', weights_only=False)
    b = torch.load(tmp_path / 'resumed/last.pt', weights_only=False)
    initial_checkpoint = torch.load(tmp_path / 'continuous/initial.pt', weights_only=False)
    assert a['scale_z'] == initial_checkpoint['scale_z']
    assert a['stream'] == b['stream']
    assert a['scale_z'] == b['scale_z']
    for key, value in a['model'].items():
        torch.testing.assert_close(value, b['model'][key], atol=0, rtol=0)
    # Normalization must depend exclusively on fitting roots.
    initial = Corpus(study).scalers
    targets['observed_radial'][12:] += 10000
    np.savez(cache / 'targets.npz', **targets)
    manifest['files']['targets.npz'] = sha(cache / 'targets.npz')
    (cache / 'manifest.json').write_text(json.dumps(manifest))
    after = Corpus(study).scalers
    for domain in initial:
        for key in ('mean', 'scale'):
            np.testing.assert_array_equal(initial[domain][key], after[domain][key])


def test_native_head_report_does_not_refit_or_rescale_features(tmp_path):
    from src.research.structural_state.evaluation import native_head_scores
    from src.research.structural_state.model import block_error
    root = tmp_path/'fits'/'example'
    root.mkdir(parents=True)
    rng = np.random.default_rng(34)
    x = rng.normal(size=(8,128)).astype(np.float32)
    weight = rng.normal(size=(89,128)).astype(np.float32)
    bias = rng.normal(size=89).astype(np.float32)
    target = x@weight.T+bias+1
    saved = dict(target_scalers={'observed': {'mean': np.zeros(89), 'scale': np.ones(89)}},
                 model={'heads.observed.weight': torch.from_numpy(weight),
                        'heads.observed.bias': torch.from_numpy(bias)})
    for checkpoint in ('initial','last'):
        torch.save(saved, root/(checkpoint+'.pt'))
    study = SimpleNamespace(technical=tmp_path, arm=lambda _: {'input':'observed'})
    corpus = SimpleNamespace(split={'development': np.arange(8)}, geometry={'observed':target},
        targets={'phase':np.zeros(8,dtype=int)}, records=[{'source':i//4,'temperature_K':400} for i in range(8)])
    scores = native_head_scores(study,'example',corpus,{'exported':x,'initial_exported':x})
    expected = float(block_error(torch.from_numpy(x@weight.T+bias),torch.from_numpy(target)).mean())
    assert scores['scores']['last']['block_mean']['groups']['all']['mse'] == pytest.approx(expected)
    assert scores['scores']['last']['radial']['groups']['all']['mse'] == pytest.approx(1.)


def test_slurm_scripts_exclude_current_node_and_use_independent_gpus():
    from pathlib import Path
    from src.research.structural_state.queue import script
    config = json.loads(Path('configs/structural_state/screen_20260922.json').read_text())
    study = SimpleNamespace(config=config, config_path=Path('configs/structural_state/screen_20260922.json').resolve(),
                            technical=Path('/tmp/scientific-run/technical'))
    fit = script(study, Path('/tmp/frozen-code'), 'A-observed')
    assert '#SBATCH --exclude=node58' in fit
    assert '#SBATCH --gres=gpu:1' in fit
    assert '#SBATCH --partition=A100,H100,RTX6000PRO' in fit
    assert '--dependency' not in fit
    collect = script(study, Path('/tmp/frozen-code'), dependencies=['111', '222'])
    assert '#SBATCH --dependency=afterany:111:222' in collect
    assert '#SBATCH --gres' not in collect


def test_neighbor_distance_and_paired_source_uncertainty():
    from src.research.structural_state.evaluation import nearest_neighbors
    from src.research.structural_state.report import paired
    x = np.array([[0., 100.], [10., 0.], [0., 0.]])
    found = nearest_neighbors(x, np.array([0, 1]), np.array([2]), np.array([400]*3), np.zeros(3), k=1)
    assert found[0, 0] == 1  # raw Euclidean space, no hidden channel whitening
    result = paired({'1': 1., '2': 2.}, {'1': .9, '2': 1.8}, {1: 400, 2: 500}, 100, 3)
    assert result['change_percent'] == pytest.approx(-10)
    assert result['ci95_lower'] == pytest.approx(-10)
    with pytest.raises(ValueError, match='exact same'):
        paired({'1': 1.}, {'2': 1.}, {1: 400, 2: 500}, 100, 3)
