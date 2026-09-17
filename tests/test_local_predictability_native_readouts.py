import json

import numpy as np
import pytest
import torch

from src.research.local_predictability import native_readouts
from src.research.local_predictability.onset_model import OnsetModel
from src.research.local_predictability.native_data import BoundedCache
from test_local_predictability_native import observation


def test_frozen_state_readouts_preserve_native_logits_and_row_identity(tmp_path, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip('Allocated GPU required')
    torch.set_num_threads(2)
    obs = observation().to('cuda')
    class Windows:
        device = torch.device('cuda')
        frames, observations = BoundedCache(0), BoundedCache(0)
        rows = [dict(source_id=i, split=split, temperature_K=400, row_id=str(i))
                for i, split in enumerate(np.repeat(['train', 'selection', 'calibration', 'test'], 8))]
        def observation(self, index, variant):
            return obs.to(self.device)
    windows = Windows()
    labels = dict(risk=np.ones(32, bool), event_bin=np.tile([0, 6], 16), source_id=np.arange(32),
                  center_id=np.ones(32, int), anchor=np.full(32, 64),
                  split=np.repeat(['train', 'selection', 'calibration', 'test'], 8))
    cond = torch.zeros(32, 7, device='cuda')
    events = torch.tensor(labels['event_bin'], device='cuda')
    identity = {'test': True}
    monkeypatch.setattr(native_readouts, 'prepare_rows', lambda config:
                        (windows, labels, cond, events, {}, [], identity))
    native_root = tmp_path/'native/technical/snapshot'
    native_root.mkdir(parents=True)
    model = OnsetModel('snapshot', activation_checkpoint=False).cuda()
    torch.save(dict(identity=identity, model=model.state_dict()), native_root/'best.pt')
    (native_root/'complete.json').write_text(json.dumps(dict(identity=identity)))
    with torch.no_grad():
        expected = model([obs], cond[:1])['logits'].cpu().numpy()
    plan = json.loads(native_readouts.resolve_path('configs/local_predictability/two_gpu_16h.json').read_text())
    plan['descriptors']['linear_hazard']['regularization_grid'] = [1]
    plan['descriptors']['mlp_hazard'].update(maximum_updates=10, validation_every=5, batch_size=8)
    plan_path = tmp_path/'plan.json'
    plan_path.write_text(json.dumps(plan))
    config = dict(output=str(tmp_path/'output'), native_output=str(tmp_path/'native'),
                  mace_backend='cueq',
                  plan=str(plan_path), variants=['snapshot'], torch_threads=2, max_spatial_edges=1000000,
                  seed=20260919, training_deadline_utc='2099-01-01T00:00:00+00:00')
    native_readouts.run(config)
    result = tmp_path/'output/technical'
    assert json.loads((result/'readout_status.json').read_text())['state'] == 'complete'
    saved = np.load(result/'snapshot/frozen_states.npz')
    np.testing.assert_allclose(saved['native_logits'], np.repeat(expected, 32, axis=0), atol=2e-6, rtol=2e-5)
    for kind in ['linear', 'mlp']:
        predictions = np.load(result/'snapshot'/kind/'predictions.npz')
        np.testing.assert_array_equal(predictions['event_bin'], labels['event_bin'])
        np.testing.assert_array_equal(predictions['source'], labels['source_id'])
        assert predictions['probability'].shape == (32, 6)
