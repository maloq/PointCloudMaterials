"""Scientific contracts for paired alignment and exact full-pass sampling."""
import copy
import numpy as np
import pytest
import torch
from torch import nn

from src.research.mace_epi.data import PassBatches
from src.research.mace_epi.objective import Objective
from src.training_methods.structural_pretraining.objective import vicreg
from src.training_methods.neighborhood_jepa.execution import training_step
from src.research.mace_epi.cached_queue import cached_copies


@pytest.mark.parametrize('treatment', ['vicreg', 'epi', 'epi-variance'])
def test_both_views_receive_gradients_and_ignore_physical_labels(treatment):
    torch.manual_seed(1)
    z = torch.randn(32, 248, requires_grad=True)
    target = dict(index=torch.arange(16), reservoir=torch.randn(16, 2, 64),
                  physical=torch.full((16, 2, 85), float('nan')))
    objective = Objective(treatment, .1)
    loss, terms = objective(None, z, target)
    swapped = z.reshape(16, 2, 248).flip(1).reshape(32, 248)
    other, _ = objective(None, swapped, dict(target, reservoir=target['reservoir'].flip(1)))
    torch.testing.assert_close(loss, other)
    loss.backward()
    assert torch.isfinite(z.grad).all()
    for view in (0, 1):
        assert z.grad.reshape(16, 2, 248)[:, view, :128].norm() > 0
    assert z.grad[:, 128:].count_nonzero() == 0
    if treatment == 'vicreg':
        expected, _ = vicreg(z[0::2, :128], z[1::2, :128])
        torch.testing.assert_close(loss, expected)
        assert set(terms) == {'alignment', 'variance', 'covariance'}
    elif treatment == 'epi':
        assert set(terms) == {'alignment', 'epi'}
    else:
        assert set(terms) == {'alignment', 'variance', 'epi'}


def test_complete_passes_and_resume_preserve_all_anchors():
    indices = np.arange(32)*3
    full = list(PassBatches(indices, 8, 123, 0, 12))
    for epoch in range(3):
        np.testing.assert_array_equal(np.sort(np.concatenate(full[epoch*4:(epoch+1)*4])), indices)
    assert full[0] != full[4]
    assert list(PassBatches(indices, 8, 123, 5, 12)) == full[5:]


def test_cached_screen_reads_cannot_change_the_template():
    reads = []

    def producer(path):
        reads.append(path)
        return {'tasks': [{'name': 'jepa-epi-direct-order'}]}

    load = cached_copies(producer)
    first = load('screen.json')
    first['tasks'][0]['name'] = 'mace-epi-s123-epoch012'
    first['tasks'].append({'name': 'extra'})
    assert load('screen.json') == {'tasks': [{'name': 'jepa-epi-direct-order'}]}
    assert reads == ['screen.json']


@pytest.mark.parametrize('treatment', ['vicreg', 'epi', 'epi-variance'])
def test_full_statistical_batch_replay_matches_direct_autograd(treatment):
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = GraphLinear()

    class GraphLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(5, 128)

        def forward(self, batch):
            return self.linear(batch['packed_positions'])

    torch.manual_seed(7)
    model = Tiny()
    direct = copy.deepcopy(model)
    x = torch.randn(24, 5)
    target = dict(index=torch.arange(12), reservoir=torch.randn(12, 2, 64))
    loss, _ = Objective(treatment, .1)(direct, direct.encoder(dict(packed_positions=x)), target)
    loss.backward()
    value, _, _ = training_step(model, Objective(treatment, .1),
        [dict(packed_positions=part) for part in x.split(8)], target, 'float32', gpu_cache=False)
    assert value == pytest.approx(float(loss.detach()), rel=2e-6)
    for a, b in zip(model.parameters(), direct.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=2e-6)
