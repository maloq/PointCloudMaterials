"""Physical window boundaries, lineage leakage, forecast gradients and end-to-end fits."""

import json

import numpy as np
import pytest
import torch

from src.experiment_runner.registry import sha256, write_json
from src.training_methods.embedding_forecast.data import WindowDataset, fit_scaling, verify_cache, window_loader
from src.training_methods.embedding_forecast.metrics import evaluate, source_bootstrap
from src.training_methods.embedding_forecast.model import EmbeddingForecaster, bin_means, forecast_loss, joint_distribution
from src.training_methods.embedding_forecast.run import collect, evaluate_checkpoint, train
from src.training_methods.embedding_forecast.augmentation import augment_history


def test_augmentation_preserves_anchor_and_carries_only_past():
    history = torch.arange(60, dtype=torch.float32).reshape(4, 5, 3)
    original = history.clone()
    dropped = augment_history(history, dict(noise_std=0.0, frame_dropout=1.0))
    torch.testing.assert_close(dropped[:, :-1], history[:, :1].expand(-1, 4, -1))
    torch.testing.assert_close(dropped[:, -1], history[:, -1])
    jittered = augment_history(history, dict(noise_std=0.1, frame_dropout=0.0))
    torch.testing.assert_close(jittered[:, -1], history[:, -1])
    assert (jittered[:, :-1] - history[:, :-1]).abs().sum() > 0
    torch.testing.assert_close(history, original)


def test_chunk_resume_restores_augmented_optimizer_and_sampling(tmp_path):
    cache_fixture(tmp_path / 'cache')
    model_config = autoregressive_variant()
    config = dict(data=dict(cache=str(tmp_path / 'cache')), output=str(tmp_path / 'whole'),
        history_ps=1.5, anchor_history_ps=1.5, horizons_ps=[0.75, 1.5, 2.25], stride_ps=0.75, seeds=[3],
        training=dict(epochs=6, patience=6, cpu_threads=1, scale_floor_fraction=0.05,
            batch_size=13, workers=0, learning_rate=0.01, weight_decay=0.0001,
            minimum_lr_fraction=0.2, gradient_clip=5.0, warmup_epochs=2,
            augmentation=dict(noise_std=0.01, frame_dropout=0.15)), variants=[model_config],
        comparisons=[], bin_comparisons=[])
    train(config, model_config, 3, 'cpu')
    chunk_config = dict(config, output=str(tmp_path / 'chunked'))
    result = train(chunk_config, model_config, 3, 'cpu', epochs_per_invocation=3)
    directory = tmp_path / 'chunked' / f"{model_config['name']}-seed3"
    assert result['state'] == 'training_paused' and result['completed_epochs'] == 3
    assert not (directory / 'test_metrics.json').exists()
    train(chunk_config, model_config, 3, 'cpu', resume=True, epochs_per_invocation=3)
    whole = torch.load(tmp_path / 'whole' / directory.name / 'last.pt', weights_only=False)
    chunked = torch.load(directory / 'last.pt', weights_only=False)
    for name in whole['model']:
        torch.testing.assert_close(whole['model'][name], chunked['model'][name], rtol=0, atol=0)
    assert whole['scheduler'] == chunked['scheduler']
    assert whole['step'] == chunked['step']
    torch.testing.assert_close(whole['sampler_rng'], chunked['sampler_rng'], rtol=0, atol=0)
    assert len((directory / 'training.jsonl').read_text().splitlines()) == 6
    with pytest.raises(ValueError, match='exact scientific config'):
        train(dict(chunk_config, stride_ps=1.5), model_config, 3, 'cpu', resume=True)


def test_streaming_evaluation_matches_retained_rows(tmp_path):
    manifest = cache_fixture(tmp_path / 'cache')
    dataset = WindowDataset(tmp_path / 'cache', manifest, 'test', 1.5, 2.25, 0.75, 1.5)
    model = EmbeddingForecaster(3, 3, 0.75, [0.75, 1.5, 2.25], autoregressive_variant())
    loader = window_loader(dataset, 13, 0, False, 2)
    retained, rows, _ = evaluate(model, loader, torch.zeros(3), torch.ones(3), 'cpu')
    streamed, omitted, _ = evaluate(model, loader, torch.zeros(3), torch.ones(3), 'cpu', retain_rows=False)
    assert retained == streamed
    assert omitted == {}
    assert retained['sample_mean']['mse'] == pytest.approx(rows['mse'].mean(), rel=1e-6)
    np.testing.assert_allclose(retained['curves']['mse_by_step'], rows['mse_by_step'].mean(0), rtol=1e-6)


@pytest.mark.parametrize('existing_preparation', [False, True])
def test_slurm_queue_has_independent_chains_and_frozen_inputs(tmp_path, monkeypatch, existing_preparation):
    from types import SimpleNamespace
    from src.training_methods.embedding_forecast import queue

    repo = tmp_path / 'repo'
    package = repo / 'src/training_methods/embedding_forecast'
    package.mkdir(parents=True)
    (package / 'queue.py').write_text('# captured source\n')
    monkeypatch.setattr(queue, '__file__', str(package / 'queue.py'))
    sources = tmp_path / 'sources.json'
    write_json(sources, dict(protocol='embedding_forecast_sources', sources=[]))
    config_path, plan_path = tmp_path / 'config.json', tmp_path / 'plan.json'
    write_json(config_path, dict(data=dict(sources_config=str(sources)), training=dict(epochs=2),
                                variants=[dict(name='ar'), dict(name='direct')], seeds=[3]))
    resources = dict(partition='H100', cpus=16, gpus=1, memory='192G', time='24:00:00')
    root = tmp_path / 'queue'
    plan = dict(output=str(root), question='Compare future paths',
        gpu=resources, cpu=dict(resources, partition='CPU', gpus=0),
        epochs_per_invocation=2 if existing_preparation else 1)
    if existing_preparation:
        plan['preparation_job_id'] = '90'
    write_json(plan_path, plan)
    calls = []
    def submit(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout=f'{100 + len(calls)}\n', stderr='')
    monkeypatch.setattr(queue.subprocess, 'run', submit)
    result = queue.submit_queue(config_path, plan_path)
    assert result['state'] == 'submitted'
    if existing_preparation:
        assert [job['afterok'] for job in result['jobs']] == [['90'], ['90'], ['101', '102']]
        assert result['external_dependencies'] == {'prepare': '90'}
        assert not (root / 'prepare').exists()
        for name in ('ar', 'direct'):
            spec = json.loads((root / f'{name}-s3-e000/run_spec.json').read_text())
            assert '--resume' not in spec['command']
            assert spec['command'][-1] == '2'
        return
    assert [job['afterok'] for job in result['jobs']] == [[], ['101'], ['102'], ['101'], ['104'], ['103', '105']]
    assert calls[-1][calls[-1].index('--dependency') + 1] == 'afterok:103:105'
    prepared = json.loads((root / 'prepare/run_spec.json').read_text())
    assert str(root / 'config.json') in prepared['command']
    assert json.loads((root / 'config.json').read_text())['data']['sources_config'] == str(root / 'sources.json')
    assert (root / 'source/src/training_methods/embedding_forecast/queue.py').read_text() == '# captured source\n'
    continuation = json.loads((root / 'ar-s3-e001/run_spec.json').read_text())
    assert '--resume' in continuation['command']
    with pytest.raises(FileExistsError):
        queue.submit_queue(config_path, plan_path)


def variant(target='trajectory', architecture='gru', distribution='deterministic'):
    return dict(name=f'{target}_{architecture}', target=target, architecture=architecture,
        distribution=distribution, history_mode='real', width=16, layers=1, heads=2,
        dropout=0.0, covariance_rank=2, minimum_std=0.03,
        loss=dict(mse=1.0, bin_mse=0.25 if target == 'trajectory' else 0.0,
                  increment_mse=0.1 if target == 'trajectory' else 0.0,
                  nll=1.0 if distribution != 'deterministic' else 0.0))


def autoregressive_variant(training='rollout'):
    config = variant(architecture='autoregressive_gru')
    config['autoregressive'] = dict(initial_state='history_mean', training=training)
    if training == 'teacher_forcing':
        config['loss'] = dict(mse=1.0, bin_mse=0.0, increment_mse=0.0, nll=0.0)
    return config


def cache_fixture(root):
    root.mkdir()
    records = []
    for source, split in enumerate(('train', 'train', 'val', 'val', 'test', 'test')):
        directory = root / f'source_{source}'
        directory.mkdir()
        t = np.arange(28, dtype=np.float32)[None, :, None]
        z = np.array([1, 2], dtype=np.float32)[:, None, None] + t * np.array([0.05, 0.1, -0.05])[None, None, :]
        z = z.astype(np.float32) + source * 0.01
        np.save(directory / 'embeddings.npy', z)
        np.save(directory / 'frames.npy', np.arange(100, 128, dtype=np.int64))
        np.save(directory / 'atom_ids.npy', np.array([10, 20], dtype=np.int64))
        records.append(dict(directory=directory.name, source_index=source, preparation_seed=source,
            split=split, temperature_K=400.0, centers=2, frames=28,
            checksums={p.name: sha256(p) for p in directory.glob('*.npy')}))
    manifest = dict(state='complete', cadence_ps=0.75, embedding_dim=3, shards=records)
    write_json(root / 'manifest.json', manifest)
    return manifest


def test_causal_windows_disjoint_bins_and_batched_mmap(tmp_path):
    manifest = cache_fixture(tmp_path / 'cache')
    dataset = WindowDataset(tmp_path / 'cache', manifest, 'train', 6, 9, 0.75, 6)
    batch = dataset[[0, 8, len(dataset)-1]]
    assert batch['history'].shape == (3, 9, 3)
    assert batch['future'].shape == (3, 12, 3)
    assert batch['atom_id'].tolist() == [10, 20, 20]
    assert batch['anchor_frame'].tolist() == [108, 108, 115]
    shorter = WindowDataset(tmp_path / 'cache', manifest, 'train', 3, 9, 0.75, 6)
    torch.testing.assert_close(batch['future'], shorter[[0, 8, len(shorter)-1]]['future'])
    torch.testing.assert_close(batch['anchor_frame'], shorter[[0, 8, len(shorter)-1]]['anchor_frame'])
    # A monotonic clock makes inclusion of t=0 or swapping boundaries observable.
    path = torch.arange(1, 13, dtype=torch.float32).reshape(1, 12, 1)
    torch.testing.assert_close(bin_means(path, [0, 4, 8, 12]).flatten(), torch.tensor([2.5, 6.5, 10.5]))
    # Workers receive vectorized batches without reordering centers or time.
    loader = window_loader(dataset, 5, 2, False, 7)
    all_frames = torch.cat([b['anchor_frame'] for b in loader])
    torch.testing.assert_close(all_frames, dataset[list(range(len(dataset)))]['anchor_frame'])
    with pytest.raises(ValueError, match='multiple'):
        WindowDataset(tmp_path / 'cache', manifest, 'train', 1, 9, 0.75, 6)


def test_training_scaler_ignores_test_and_lineage_leaks_are_rejected(tmp_path):
    manifest = cache_fixture(tmp_path / 'cache')
    dataset = WindowDataset(tmp_path / 'cache', manifest, 'train', 1.5, 2.25, 0.75, 1.5)
    before = fit_scaling(dataset, 0.05)
    path = tmp_path / 'cache/source_4/embeddings.npy'
    np.save(path, np.load(path) * 1000)
    after = fit_scaling(dataset, 0.05)
    for a, b in zip(before, after):
        np.testing.assert_array_equal(a, b)
    with pytest.raises(ValueError, match='checksum'):
        verify_cache(tmp_path / 'cache')
    manifest['shards'][4]['checksums']['embeddings.npy'] = sha256(path)
    manifest['shards'][4]['preparation_seed'] = manifest['shards'][0]['preparation_seed']
    write_json(tmp_path / 'cache/manifest.json', manifest)
    with pytest.raises(ValueError, match='leakage'):
        verify_cache(tmp_path / 'cache')


@pytest.mark.parametrize('architecture', ['mlp', 'gru', 'mean_residual_gru', 'transformer'])
@pytest.mark.parametrize('target', ['bin_means', 'trajectory'])
def test_history_gradients_and_output_contract(architecture, target):
    torch.manual_seed(12)
    model = EmbeddingForecaster(3, 5, 0.75, [3, 6, 9], variant(target, architecture))
    history = torch.randn(2, 5, 3, requires_grad=True)
    output = model(history)
    assert output['mean'].shape == (2, 3 if target == 'bin_means' else 12, 3)
    output['mean'].square().sum().backward()
    assert torch.all(history.grad.abs().sum((0, 2)) > 0)
    changed = history.detach().clone()
    changed[:, 0] += 5
    assert (model(changed)['mean'] - output['mean']).abs().max() > 1e-6


def test_joint_gaussian_matches_dense_covariance_and_has_factor_gradient():
    torch.manual_seed(4)
    model = EmbeddingForecaster(3, 5, 0.75, [0.75, 1.5, 2.25], variant(distribution='low_rank_gaussian'))
    history, future = torch.randn(4, 5, 3), torch.randn(4, 3, 3)
    output = model(history)
    distribution = joint_distribution(output)
    dense = torch.distributions.MultivariateNormal(output['mean'].flatten(1),
        covariance_matrix=torch.diag_embed(output['std'].flatten(1).square()) +
        output['factor'] @ output['factor'].transpose(1, 2))
    torch.testing.assert_close(distribution.log_prob(future.flatten(1)), dense.log_prob(future.flatten(1)))
    loss, _ = forecast_loss(model, output, future, history[:, -1], model.config['loss'])
    loss.backward()
    assert torch.isfinite(model.factor_head.weight.grad).all()
    assert model.factor_head.weight.grad.abs().sum() > 0


def test_history_controls_and_undefined_source_interval():
    history = torch.randn(2, 5, 3)
    for control in ('anchor', 'mean'):
        cfg = dict(variant(), history_mode=control)
        model = EmbeddingForecaster(3, 5, 0.75, [3, 6, 9], cfg)
        reversed_history = torch.cat((history[:, :-1].flip(1), history[:, -1:]), 1)
        torch.testing.assert_close(model(history)['mean'], model(reversed_history)['mean'])
    result = source_bootstrap(np.array([0.1, 0.2]), np.ones(2), np.zeros(2), 1)
    assert result['ci95'] is None
    assert result['interval_status'].startswith('undefined')


def test_mean_residual_starts_at_history_mean_and_control_uses_only_anchor():
    history = torch.randn(2, 5, 3)
    model = EmbeddingForecaster(3, 5, 0.75, [3, 6, 9], variant(architecture='mean_residual_gru'))
    torch.nn.init.zeros_(model.mean_head.weight)
    torch.nn.init.zeros_(model.mean_head.bias)
    torch.testing.assert_close(model(history)['mean'], history.mean(1, keepdim=True).expand(-1, 12, -1))
    model.history_mode = 'anchor'
    torch.testing.assert_close(model(history)['mean'], history[:, -1:].expand(-1, 12, -1))


@pytest.mark.parametrize('model_config', [variant('bin_means'), variant('trajectory'),
    autoregressive_variant(), autoregressive_variant('teacher_forcing')],
    ids=['direct-bins', 'direct-path', 'autoregressive-rollout', 'autoregressive-teacher-forcing'])
def test_fit_checkpoint_roundtrip_and_collection(tmp_path, model_config):
    manifest = cache_fixture(tmp_path / 'cache')
    config = dict(data=dict(cache=str(tmp_path / 'cache')), output=str(tmp_path / 'runs'),
        history_ps=1.5, anchor_history_ps=1.5, horizons_ps=[0.75, 1.5, 2.25], stride_ps=0.75, seeds=[3],
        training=dict(epochs=15, patience=15, cpu_threads=1, scale_floor_fraction=0.05,
            batch_size=64, workers=0, learning_rate=0.01, weight_decay=0.0,
            minimum_lr_fraction=0.2, gradient_clip=5.0), variants=[model_config], comparisons=[], bin_comparisons=[])
    metrics = train(config, config['variants'][0], 3, 'cpu')
    assert metrics['source_mean']['mse'] < metrics['source_mean']['persistence_mse']
    directory = tmp_path / 'runs' / f"{model_config['name']}-seed3"
    restored = evaluate_checkpoint(directory, 'cpu')
    assert restored['source_mean']['mse'] == metrics['source_mean']['mse']
    assert (directory / 'last.pt').is_file()
    report = collect(config)
    assert report['variants'][model_config['name']]['persistence']['gain'] > 0
    with pytest.raises(FileExistsError, match='overwrite'):
        train(config, config['variants'][0], 3, 'cpu')


def test_probabilistic_evaluation_reports_path_scores(tmp_path):
    manifest = cache_fixture(tmp_path / 'cache')
    dataset = WindowDataset(tmp_path / 'cache', manifest, 'test', 1.5, 2.25, 0.75, 1.5)
    model = EmbeddingForecaster(3, 3, 0.75, [0.75, 1.5, 2.25], variant(distribution='low_rank_gaussian'))
    scores, _, examples = evaluate(model, window_loader(dataset, 64, 0, False, 2),
        torch.zeros(3), torch.ones(3), 'cpu', sample_paths=True)
    for key in ('nll', 'marginal_crps', 'energy_score', 'coverage90'):
        assert np.isfinite(scores['source_mean'][key])
    assert examples['sample_paths'].shape == (16, 16, 3, 3)


def test_autoregressive_feedback_and_all_history_gradients():
    torch.manual_seed(11)
    model = EmbeddingForecaster(3, 5, 0.75, [3, 6, 9], autoregressive_variant())
    model.eval()
    history = torch.randn(2, 5, 3, requires_grad=True)
    increments = []
    hook = model.mean_head.register_forward_hook(lambda module, inputs, output: increments.append(output))
    path = model(history)['mean']
    hook.remove()
    first_increment_grad = torch.autograd.grad(path[:, -1].sum(), increments[0], retain_graph=True)[0]
    assert first_increment_grad.abs().sum() > 0
    path[:, -1].square().sum().backward()
    assert torch.all(history.grad.abs().sum((0, 2)) > 0)
    assert path.shape == (2, 12, 3)
    # Change only the first predicted increment. Later frames must respond through feedback.
    steps = []
    def perturb_first(module, inputs, output):
        steps.append(1)
        return output + 1.0 if len(steps) == 1 else output
    hook = model.mean_head.register_forward_hook(perturb_first)
    changed = model(history.detach())['mean']
    hook.remove()
    assert (changed[:, -1] - path[:, -1]).abs().max() > 0.1


def test_teacher_forcing_is_shifted_and_never_used_at_evaluation():
    torch.manual_seed(5)
    model = EmbeddingForecaster(3, 5, 0.75, [3, 6, 9], autoregressive_variant('teacher_forcing'))
    history, future = torch.randn(2, 5, 3), torch.randn(2, 12, 3)
    original = model.teacher_forced(history, future)['mean']
    changed_future = future.clone()
    changed_future[:, 4] += 10
    changed = model.teacher_forced(history, changed_future)['mean']
    torch.testing.assert_close(original[:, :5], changed[:, :5])
    assert (original[:, 5:] - changed[:, 5:]).abs().max() > 0.1
    changed_future = future.clone()
    changed_future[:, -1] += 100
    torch.testing.assert_close(original, model.teacher_forced(history, changed_future)['mean'])
    model.eval()
    with pytest.raises(ValueError, match='validation/test'):
        model.teacher_forced(history, future)
    # Declaring teacher-forced training never changes the inference rollout.
    rollout = EmbeddingForecaster(3, 5, 0.75, [3, 6, 9], autoregressive_variant()).eval()
    rollout.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(history)['mean'], rollout(history)['mean'])


@pytest.mark.parametrize('initial_state', ['anchor', 'history_mean'])
def test_autoregressive_initial_baseline_and_mean_control(initial_state):
    cfg = autoregressive_variant()
    cfg['autoregressive']['initial_state'] = initial_state
    cfg['history_mode'] = 'mean'
    model = EmbeddingForecaster(3, 5, 0.75, [3, 6, 9], cfg).eval()
    history = torch.randn(2, 5, 3)
    reversed_past = torch.cat((history[:, :-1].flip(1), history[:, -1:]), 1)
    torch.testing.assert_close(model(history)['mean'], model(reversed_past)['mean'])
    model.history_mode = 'real'
    torch.nn.init.zeros_(model.mean_head.weight)
    torch.nn.init.zeros_(model.mean_head.bias)
    expected = history.mean(1) if initial_state == 'history_mean' else history[:, -1]
    torch.testing.assert_close(model(history)['mean'], expected[:, None].expand(-1, 12, -1))


def test_autoregressive_rejects_incompatible_protocols():
    cfg = autoregressive_variant()
    cfg['target'] = 'bin_means'
    with pytest.raises(ValueError, match='full trajectory'):
        EmbeddingForecaster(3, 5, 0.75, [3, 6, 9], cfg)
    cfg = autoregressive_variant('teacher_forcing')
    cfg['loss']['bin_mse'] = 0.25
    with pytest.raises(ValueError, match='one-step MSE'):
        EmbeddingForecaster(3, 5, 0.75, [3, 6, 9], cfg)
