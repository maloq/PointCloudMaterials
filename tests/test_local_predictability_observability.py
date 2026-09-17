import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from src.research.local_predictability.observability import source_arrays
from src.research.local_predictability.baselines import hazard_fit
from src.research.local_predictability.metrics import hazard_loss


def fixture_data():
    plan = json.loads(Path('configs/local_predictability/two_gpu_16h.json').read_text())
    packet = np.broadcast_to(np.arange(30, dtype=np.float32)[None, :, None], (3, 30, 128)).copy()
    labels = np.zeros((3, 30), np.uint8)
    labels[0, :] = 1  # Already crystalline: must remain in the all-state cohort.
    labels[1, 10:13] = 2  # Starts at the tested endpoint; two frames confirm it.
    data = dict(packet=packet, labels=labels, times_ps=np.arange(30) * .75, atom_ids=np.array([10, 20, 30]))
    source = dict(id=1, split='train', temperature_K=400)
    return data, source, plan


def test_current_and_future_state_keep_crystalline_rows_and_align_labels():
    data, source, plan = fixture_data()
    anchors = np.array([6])
    current = source_arrays(data, source, anchors, dict(task='state', horizon_ps=0), plan)
    future = source_arrays(data, source, anchors, dict(task='state', horizon_ps=3), plan)
    np.testing.assert_array_equal(current['center'], [10, 20, 30])
    np.testing.assert_array_equal(current['y'], [0, 1, 1])
    np.testing.assert_array_equal(future['y'], [0, 0, 1])
    np.testing.assert_array_equal(current['x'][:, :128], 6)
    np.testing.assert_array_equal(future['x'][:, :128], 10)
    changed = {**data, 'packet': data['packet'].copy()}
    changed['packet'][:, 7:] = -999
    np.testing.assert_array_equal(current['x'], source_arrays(changed, source, anchors, dict(task='state', horizon_ps=0), plan)['x'])


def test_sequence_risk_population_confirmation_and_future_boundaries():
    data, source, plan = fixture_data()
    case = dict(task='onset_sequence', horizon_ps=3)
    result = source_arrays(data, source, np.array([6]), case, plan)
    np.testing.assert_array_equal(result['center'], [20, 30])
    np.testing.assert_array_equal(result['y'], [0, 1])
    np.testing.assert_array_equal(result['x'][:, :6*128].reshape(2, 6, 128)[0, :, 0], np.arange(7, 13))
    changed = {**data, 'labels': data['labels'].copy()}
    changed['labels'][1, 12] = 0  # Removing final confirmation changes the event.
    np.testing.assert_array_equal(source_arrays(changed, source, np.array([6]), case, plan)['y'], [1, 1])
    changed = {**data, 'packet': data['packet'].copy()}
    changed['packet'][:, 13:] = -999
    np.testing.assert_array_equal(result['x'], source_arrays(changed, source, np.array([6]), case, plan)['x'])
    with pytest.raises(ValueError, match='incomplete observation/confirmation'):
        source_arrays(data, source, np.array([25]), case, plan)


def test_binary_adapter_matches_bce_and_saves_usable_model(tmp_path):
    _, _, plan = fixture_data()
    plan['sampling']['horizons_ps'] = [0]
    plan['descriptors']['mlp_hazard'].update(maximum_updates=10, validation_every=5, batch_size=16)
    plan['descriptors']['linear_hazard']['regularization_grid'] = [1]
    rng = np.random.default_rng(8)
    x = rng.normal(size=(80, 4)).astype(np.float32)
    y = (x[:, 0] <= 0).astype(np.int64)
    arrays = dict(x=x, y=y, source=np.repeat(np.arange(8), 10),
                  split=np.repeat(['train', 'selection', 'calibration', 'test'], 20),
                  temperature=np.full(80, 400), center=np.arange(80), anchor=np.full(80, 64))
    config = dict(seed=20260919, training_deadline_utc='2099-01-01T00:00:00+00:00')
    for kind in ['linear', 'mlp']:
        probability, logits, _ = hazard_fit(arrays, kind, plan, torch.device(os.environ.get('PCM_TEST_DEVICE', 'cpu')), config, tmp_path/kind)
        actual = torch.tensor(y == 0, dtype=torch.float32)
        torch.testing.assert_close(hazard_loss(torch.tensor(logits), torch.tensor(y)),
                                   binary_cross_entropy_with_logits(torch.tensor(logits[:, 0]), actual, reduction='none'))
        np.testing.assert_allclose(probability[:, 0], torch.sigmoid(torch.tensor(logits[:, 0])), atol=1e-7)
        assert (tmp_path/kind/'model.pt').exists()
        saved = np.load(tmp_path/kind/'predictions.npz')
        np.testing.assert_array_equal(saved['event_bin'], y)
