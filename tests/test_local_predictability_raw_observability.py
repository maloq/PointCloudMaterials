import json
import resource

import numpy as np
import pytest
import torch

from src.data.predictive_memory.prepare import file_hash
from src.research.local_predictability.raw_observability import CurrentStateModel, current_targets, fit
from src.research.local_predictability.native_queue import configure_file_limit
from src.research.local_predictability.native_data import BoundedCache
from test_local_predictability_native import observation


def test_current_labels_join_atom_ids_and_include_crystals(tmp_path):
    labels = np.zeros((2, 8), np.uint8)
    labels[0, 3] = 2
    labels[1, 4] = 1
    shard = tmp_path / 'source.npz'
    np.savez(shard, labels=labels, atom_ids=[99, 10])
    release = dict(sources=[dict(id=7, shard=shard.name, shard_sha256=file_hash(shard))])
    rows = [dict(source_id=7, center_id=10, anchor=3), dict(source_id=7, center_id=99, anchor=3),
            dict(source_id=7, center_id=10, anchor=4)]
    np.testing.assert_array_equal(current_targets(rows, release, tmp_path), [1, 0, 0])
    with shard.open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError, match='Changed label producer'):
        current_targets(rows, release, tmp_path)


def test_file_limit_raised_and_hard_limit_respected(monkeypatch):
    limits = [1024, 131072]
    monkeypatch.setattr(resource, 'getrlimit', lambda _: tuple(limits))
    monkeypatch.setattr(resource, 'setrlimit', lambda _, value: limits.__setitem__(slice(None), value))
    assert configure_file_limit()['soft'] == 8192
    limits[:] = [1024, 4096]
    with pytest.raises(RuntimeError, match='hard limit'):
        configure_file_limit()


def test_raw_state_gpu_checkpoint_resume(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip('Requires the allocated GPU')
    obs = observation().to('cuda')
    class Windows:
        device = torch.device('cuda')
        frames, observations = BoundedCache(0), BoundedCache(0)
        rows = [dict(source_id=i, center_id=1, anchor=64, split='train', row_id=str(i)) for i in range(8)]
        def observation(self, index, variant):
            assert variant == 'snapshot'
            return obs.to(self.device)
    config = dict(output=str(tmp_path), max_spatial_edges=1000000, updates_per_stage=2, mace_backend='cueq',
                  training_deadline_utc='2099-01-01T00:00:00+00:00')
    cond = torch.zeros(8, 7, device='cuda')
    target = torch.tensor([0, 1]*4, device='cuda')
    identity = {'test': True}
    path = fit(config, Windows(), cond, target, list(range(8)), identity)
    state = torch.load(path, weights_only=False, map_location='cpu')
    assert state['step'] == 2
    model = CurrentStateModel(activation_checkpoint=False).cuda()
    model.load_state_dict(state['model'])
    assert model([obs], cond[:1])['logits'].shape == (1, 1)
    fit(config, Windows(), cond, target, list(range(8)), identity, resume=True)
    assert json.loads((tmp_path/'technical/current-state/complete.json').read_text())['step'] == 2
