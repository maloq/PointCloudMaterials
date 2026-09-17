"""Population joins, identical heads, effective batches and exact v2 resume."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from src.research.local_predictability.backbone_data import BackboneData, join_packets, population_splits
from src.research.local_predictability.backbone_v2 import update, restore, summarize, per_row_loss, fit, verify_gate, export
from src.research.local_predictability.model_factory import build_model
from src.research.local_predictability.native_data import SourceSampler, HORIZON_FRAMES
from src.research.local_predictability.native_preflight import save_checkpoint
from src.research.local_predictability.native_runtime import ObservationPrefetcher, peek_batch
from src.research.local_predictability.native_model import PhysicalMeans
from src.research.local_predictability.onset_model import OnsetModel
from test_axial_gatr import observation
from test_local_predictability_native_runtime import Windows


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def configuration():
    config = json.loads(Path('configs/local_predictability/backbone_v2/rtx6000_screen.json').read_text())
    config['mace_backend'] = 'e3nn'
    return config


def test_join_uses_ids_and_exact_lags_and_all_state_population():
    rows = [dict(source_id=7, center_id=99, anchor=64), dict(source_id=8, center_id=6, anchor=64),
            dict(source_id=7, center_id=12, anchor=104)]
    packet = np.arange(2*801*128, dtype=np.float32).reshape(2, 801, 128)
    indices, target = join_packets(rows, 7, dict(atom_ids=np.array([12, 99]),
        times_ps=np.arange(801)*.75, packet=packet))
    np.testing.assert_array_equal(indices, [0, 2])
    np.testing.assert_array_equal(target[0], packet[1, 64+np.r_[0, HORIZON_FRAMES]])
    np.testing.assert_array_equal(target[1], packet[0, 104+np.r_[0, HORIZON_FRAMES]])
    labels = dict(split=np.array(['train', 'train', 'test', 'test']), risk=np.array([True, False, True, False]))
    np.testing.assert_array_equal(population_splits(labels, 'physical_means')['train'], [0, 1])
    np.testing.assert_array_equal(population_splits(labels, 'physical_means')['test'], [2, 3])
    np.testing.assert_array_equal(population_splits(labels, 'onset')['train'], [0])


@pytest.mark.parametrize('objective,existing', [('physical_means', PhysicalMeans), ('onset', OnsetModel)])
def test_factory_preserves_existing_head_shapes_and_outputs(objective, existing):
    config = configuration()
    torch.manual_seed(4); old = existing(variant='snapshot', **config['mace'])
    new = build_model('mace', objective, 'snapshot', config)
    new.load_state_dict(old.state_dict())
    obs = [observation()]; condition = torch.zeros(1, 7)
    a, b = old(obs, condition), new(obs, condition)
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


@pytest.mark.parametrize('objective', ['physical_means', 'onset'])
def test_effective_batch_gradient_parity(objective):
    config = configuration()
    torch.manual_seed(9); a = build_model('axial_gatr', objective, 'snapshot', config)
    b = copy.deepcopy(a)
    data = SimpleNamespace(cond=torch.randn(8, 7), targets=torch.randn(8, 7, 128), events=torch.arange(8)%7)
    obs = [observation(k) for k in range(8)]
    # SGD isolates the accumulated gradients, without Adam epsilon amplification.
    oa = torch.optim.SGD(a.parameters(), lr=.001); ob = torch.optim.SGD(b.parameters(), lr=.001)
    la, _ = update(a, obs, list(range(8)), data, oa, 8)
    lb, _ = update(b, obs, list(range(8)), data, ob, 2)
    assert la == pytest.approx(lb, rel=1e-6)
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], atol=1e-7, rtol=1e-6)


def test_metric_source_weighting_and_objectives_are_explicit():
    data = SimpleNamespace(labels={'source_id': np.array([1, 1, 2])}, targets=torch.zeros(3, 7, 128),
        identity={'horizons_ps': (.75, 3, 9, 24, 48, 96)}, events=torch.tensor([0, 6, 1]))
    prediction = dict(present=torch.tensor([0., 0., 2.])[:, None].expand(3, 128), future=torch.ones(3, 6, 128))
    result = summarize(prediction, data, [0, 1, 2], 'physical_means')
    assert result['present_mse'] == 2. and result['future_mse'] == 1. and result['score'] == 3.
    torch.testing.assert_close(per_row_loss(prediction, data, [0, 1, 2], 'physical_means'), torch.tensor([1., 1., 5.]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Integration of GPU trainer and receipts')
def test_gate_fit_resume_export_and_receipt_rejection(tmp_path):
    config = configuration(); config.update(output=str(tmp_path), gate_updates=2, training_updates=2,
        gate_evaluate_every=1, evaluate_every=1, training_deadline_utc='2099-01-01T00:00:00+00:00')
    class ValidWindows(Windows):
        def observation(self, index, variant):
            return self.value.to(self.device)
    windows = ValidWindows()
    temperatures = [400, 450, 500, 510, 520, 400, 450, 500]
    windows.rows = [dict(source_id=s, split=split, row_id=f'{s}:{j}', temperature_K=temperatures[s%8])
        for k, split in enumerate(['train', 'selection', 'calibration', 'test']) for s in range(8*k, 8*k+8) for j in range(4)]
    n = len(windows.rows)
    labels = {key: np.array([row[key] for row in windows.rows]) for key in ['source_id', 'split']}
    labels.update(risk=np.ones(n, bool), event_bin=np.arange(n)%7, center_id=np.arange(n), anchor=np.full(n, 64))
    data = BackboneData(windows, labels, torch.zeros(n, 7, device='cuda'), torch.as_tensor(labels['event_bin'], device='cuda'),
        torch.zeros(n, 7, 128, device='cuda'), np.zeros((n, 7, 128), np.float32),
        dict(horizons_ps=[.75, 3, 9, 24, 48, 96], test_fixture=True))
    data.selection = lambda objective: data.splits(objective)['selection']
    kind = 'axial_gatr'
    assert fit(config, data, kind, 'physical_means', 'snapshot', diagnostic=True) == 'passed'
    verify_gate(config, data, kind)
    from src.research.local_predictability.backbone_profile import profile
    profile(config, data, kind)
    profile(config, data, kind, resume=True)
    recorded = json.loads((tmp_path/'technical/axial_gatr/profile/history12.json').read_text())
    assert recorded['actual_attention_kernels']
    assert fit(config, data, kind, 'physical_means', 'snapshot', stop_step=1) == 'paused'
    assert fit(config, data, kind, 'physical_means', 'snapshot', resume=True) == 'complete'
    export(config, data, kind, 'physical_means', 'snapshot')
    export(config, data, kind, 'physical_means', 'snapshot')  # Checksummed reuse.
    destination = tmp_path/'technical/axial_gatr/physical_means/snapshot'
    with np.load(destination/'test_predictions.npz') as values:
        assert values['future'].shape == (32, 6, 128)
        np.testing.assert_array_equal(values['indices'], data.splits('physical_means')['test'])
    # A continuation gets only this architecture/objective's exact parent.
    assert fit(config, data, kind, 'physical_means', 'history12', stop_step=1,
               parent=destination/'best.pt') == 'paused'
    assert fit(config, data, kind, 'onset', 'snapshot') == 'complete'
    export(config, data, kind, 'onset', 'snapshot')
    assert (tmp_path/'technical/axial_gatr/onset/snapshot/onset_assay.json').exists()
    changed = copy.deepcopy(config); changed['axial_gatr']['scalar_channels'] = 64
    with pytest.raises(ValueError, match='receipt'):
        verify_gate(changed, data, kind)


def test_uncommitted_log_tail_is_retained_separately(tmp_path):
    from src.research.local_predictability.backbone_v2 import truncate_uncommitted_logs
    path = tmp_path/'training.jsonl'
    path.write_text('{"step": 1}\n{"step": 2}\n{"step":')
    truncate_uncommitted_logs(tmp_path, 1)
    assert path.read_text() == '{"step": 1}\n'
    assert (tmp_path/'training-uncommitted.jsonl').read_text() == '{"step": 2}\n{"step":'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA prefetch and checkpoint state')
@pytest.mark.parametrize('kind', ['mace', 'axial_gatr'])
@pytest.mark.parametrize('objective', ['physical_means', 'onset'])
def test_prefetched_training_resume_is_exact(tmp_path, kind, objective):
    config = configuration(); config['mace_backend'] = 'cueq'
    torch.manual_seed(42); windows = Windows()
    model = build_model(kind, objective, 'snapshot', config).cuda()
    if kind == 'mace':
        assert model.encoder.mace_backend == 'cueq'
        assert type(model.encoder.products[0].symmetric_contractions).__module__.startswith('cuequivariance_torch')
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    sampler = SourceSampler(windows.rows)
    data = SimpleNamespace(cond=torch.randn(16, 7, device='cuda'), targets=torch.randn(16, 7, 128, device='cuda'),
        events=torch.arange(16, device='cuda')%7)
    with ObservationPrefetcher(windows) as inputs:
        indices = sampler.batch(); obs = inputs.take(inputs.submit(indices, 'snapshot'), indices)
        update(model, obs, indices, data, optimizer, 4)
        prefetched = inputs.submit(peek_batch(sampler), 'snapshot')
        saved_identity = dict(kind=kind, objective=objective)
        path = tmp_path/'latest.pt'
        save_checkpoint(path, model, optimizer, sampler, 1, saved_identity, dict(stage='test'))
        indices = sampler.batch(); obs = inputs.take(prefetched, indices)
        expected_loss, _ = update(model, obs, indices, data, optimizer, 4)
        expected = copy.deepcopy(model.state_dict())
        fresh = build_model(kind, objective, 'snapshot', config).cuda()
        fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=.0003)
        restore(path, fresh, fresh_optimizer, sampler, saved_identity, 'test')
        assert sampler.batch() == indices
        actual_loss, _ = update(fresh, obs, indices, data, fresh_optimizer, 4)
        assert actual_loss == pytest.approx(expected_loss, abs=1e-7)
        for key, value in fresh.state_dict().items():
            torch.testing.assert_close(value, expected[key], atol=1e-7, rtol=1e-6)
        with pytest.raises(ValueError, match='identity'):
            restore(path, fresh, fresh_optimizer, sampler, {'wrong': True}, 'test')
