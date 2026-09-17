"""State interventions and readout selection must not manufacture predictive skill."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from src.training_methods.predictive_memory.diagnose import ridge_readout, state_interventions, verify_export


def test_ridge_predictive_signal_and_no_test_leakage():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(120, 5))
    targets = features@rng.normal(size=(5, 3))+.03*rng.normal(size=(120, 3))
    train, val, test = np.arange(70), np.arange(70, 95), np.arange(95, 120)
    prediction, state, selection = ridge_readout(features, targets, train, val)
    assert np.mean((prediction.numpy()[test]-targets[test])**2) < .01
    changed_features, changed_targets = features.copy(), targets.copy()
    changed_features[test] += 1000
    changed_targets[test] *= -1000
    _, changed_state, changed_selection = ridge_readout(changed_features, changed_targets, train, val)
    assert selection == changed_selection
    for key in state:
        torch.testing.assert_close(state[key], changed_state[key], rtol=0, atol=0)
    # Independent normal-equation calculation checks scaling, intercept and penalty.
    x = (features-state['center'].numpy())/state['scale'].numpy()
    expected = np.linalg.solve(x[train].T@x[train]+selection['alpha']*np.eye(5),
                               x[train].T@(targets[train]-targets[train].mean(0)))
    np.testing.assert_allclose(state['coefficient'], expected, atol=1e-12)


class KnownHead(torch.nn.Module):
    def __init__(self, use_state):
        super().__init__()
        self.use_state = use_state

    def forward(self, state, condition):
        value = state if self.use_state else condition
        mean = value.expand(-1, 128)
        return dict(present=mean, logits=torch.zeros(len(state), 1), mean=mean[:, None],
                    diagonal=torch.ones(len(state), 1, 128), factor=torch.zeros(len(state), 1, 128, 1))


def test_intervention_detects_state_use_and_condition_only_negative_control():
    z = torch.tensor([[-2.], [2.]])
    train_z = torch.tensor([[-1.], [1.]])
    condition = torch.tensor([[3.], [5.]])
    present = z.expand(-1, 128)
    future = present[:, None]
    signal = state_interventions(KnownHead(True), z, train_z, condition, present, future)
    torch.testing.assert_close(signal['mean_state_joint_nll_increase'], torch.full((2,), 2.))
    torch.testing.assert_close(signal['mean_state_future_mse_increase'], torch.full((2,), 4.))
    ignored = state_interventions(KnownHead(False), z, train_z, condition, present, future)
    for name in ('joint_nll', 'future_mse', 'present_mse'):
        torch.testing.assert_close(ignored[f'mean_state_{name}_increase'], torch.zeros(2), atol=0, rtol=0)


def test_exports_must_retain_source_center_anchor_pairing():
    rows = [dict(source_id=10, center_id=20, anchor=400)]
    dataset = SimpleNamespace(indices=dict(test=[0]), rows=rows)
    exported = dict(indices=[0], rows=[dict(rows[0])], embeddings=torch.zeros(1, 2))
    verify_export(dataset, exported, 'test')
    exported['rows'][0]['center_id'] = 21
    with pytest.raises(ValueError, match='source/center/anchor'):
        verify_export(dataset, exported, 'test')
    exported['indices'] = [1]
    with pytest.raises(ValueError, match='row indices'):
        verify_export(dataset, exported, 'test')
