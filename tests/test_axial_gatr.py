"""Physical symmetries, masking, nesting and causal isolation of the v2 encoder."""
from dataclasses import replace
import copy
import numpy as np
import pytest
import torch
from src.data.predictive_memory.observations import assemble, frame_observation
from src.models.encoders.axial_gatr import AxialGATrEncoder, support_bias


def observation(seed=17, rotation=None, translation=0., boost=0.):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4, 4, (9, 3)); x[4] = 0.; x[-1] = [16., 0., 0.]
    frames = []
    for k in range(17):
        p = x + rng.normal(0, .04, x.shape); p[4] = 0.; p[-1] = [16+k/8, 0, 0]
        v = rng.normal(size=x.shape)
        if rotation is not None:
            p = p @ rotation; v = v @ rotation
        frames.append(frame_observation(p+translation, v+boost, np.full(3, 100.),
                                       np.arange(9), 4, 17., 5.))
    return assemble(frames, np.arange(17)*.75, radius=17., cutoff=5.)


def model(variant='history12'):
    torch.manual_seed(42)
    return AxialGATrEncoder(variant=variant)


def grads(m, obs):
    m.zero_grad(set_to_none=True)
    y = m(obs)
    y.square().sum().backward()
    return y.detach(), {k: p.grad.clone() for k, p in m.named_parameters() if p.grad is not None}


@pytest.mark.parametrize('variant', ['history12', 'repeat12'])
def test_zero_gate_shared_outputs_and_gradients(variant):
    obs = observation()
    a, b = model('snapshot'), model(variant)
    b.load_state_dict(a.state_dict())
    ya, ga = grads(a, [obs]); yb, gb = grads(b, [obs])
    torch.testing.assert_close(ya, yb, atol=2e-6, rtol=2e-5)
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name], atol=3e-6, rtol=3e-4, msg=name)
    assert gb['history_alpha'].abs().min() > 1e-9
    assert all(g.abs().max() == 0 for k, g in gb.items() if k.startswith(('temporal.', 'time_embedding.')))
    with torch.no_grad(): b.history_alpha.fill_(.3)
    _, opened = grads(b, [obs])
    assert any(g.abs().max() > 1e-8 for k, g in opened.items() if k.startswith('temporal.'))


def test_symmetries_center_not_first_and_absent_values():
    m = model()
    with torch.no_grad(): m.history_alpha.fill_(.4)
    original = observation()
    rotation, _ = np.linalg.qr(np.random.default_rng(14).normal(size=(3, 3)))
    transformed = observation(rotation=rotation, translation=np.array([6, 9, 2]), boost=np.array([3, -1, 7]))
    torch.testing.assert_close(m([original]), m([transformed]), atol=3e-6, rtol=3e-5)
    # The scalar output is also reflection invariant.
    torch.testing.assert_close(m([original]), m([observation(rotation=-np.eye(3))]), atol=3e-6, rtol=3e-5)
    changed = copy.deepcopy(original)
    changed.positions[changed.weights == 0] = 12345
    changed.velocities[changed.weights == 0] = -4321
    torch.testing.assert_close(m([original]), m([changed]), atol=1e-7, rtol=1e-6)
    permutation = torch.tensor([4, 8, 0, 3, 1, 7, 2, 6, 5])
    permuted = replace(original, positions=original.positions[:, permutation],
        velocities=original.velocities[:, permutation], weights=original.weights[:, permutation],
        atom_ids=original.atom_ids[permutation] + 10000)
    torch.testing.assert_close(m([original]), m([permuted]), atol=3e-6, rtol=3e-5)


def test_batch_isolation_and_causal_prefix():
    m = model()
    with torch.no_grad(): m.history_alpha.fill_(.3)
    a, b = observation(), observation(22)
    torch.testing.assert_close(m([a, b]), torch.cat([m([a]), m([b])]), atol=3e-6, rtol=3e-5)
    prefix = replace(a, positions=a.positions[:9], velocities=a.velocities[:9],
        weights=a.weights[:9], offsets_ps=a.offsets_ps[:9], edges=a.edges[:9])
    full, _ = m.atom_features([a]); short, _ = m.atom_features([prefix])
    torch.testing.assert_close(full[:, :9], short, atol=3e-6, rtol=3e-5)
    changed = copy.deepcopy(a); changed.positions[9:] += 90; changed.velocities[9:] += 70
    changed.positions[-1, 4] = 0
    perturbed, _ = m.atom_features([changed])
    torch.testing.assert_close(full[:, :9], perturbed[:, :9], atol=3e-6, rtol=3e-5)


def test_repeat_uses_only_current_support_and_inputs():
    m = model('repeat12')
    with torch.no_grad(): m.history_alpha.fill_(.3)
    a = observation(); b = copy.deepcopy(a)
    b.positions[:-1] += 56; b.velocities[:-1] -= 30; b.weights[:-1] = 1
    torch.testing.assert_close(m([a]), m([b]), atol=0, rtol=0)


def test_support_is_inside_softmax_and_center_must_be_observed():
    weights = torch.tensor([1., .25, 0.])
    torch.testing.assert_close(support_bias(weights).softmax(-1), torch.tensor([.8, .2, 0.]))
    a = observation(); a.weights[-1, 4] = 0
    with pytest.raises(ValueError, match='exactly one observed'):
        model()([a])


def test_history_input_gradients_require_open_gate():
    a = observation(); a.positions.requires_grad_(); a.velocities.requires_grad_()
    m = model()
    m([a]).square().sum().backward()
    assert torch.count_nonzero(a.positions.grad[:-1]) == 0
    assert torch.count_nonzero(a.velocities.grad[:-1]) == 0
    assert a.velocities.grad[-1].abs().max() > 1e-8
    a.positions.grad = None; a.velocities.grad = None
    with torch.no_grad(): m.history_alpha.fill_(.3)
    m([a]).square().sum().backward()
    assert a.positions.grad[:-1].abs().max() > 1e-8
    assert a.velocities.grad[:-1].abs().max() > 1e-8


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Production SDPA CUDA path')
def test_cuda_masked_batch_symmetry_nesting_and_gradients():
    obs = observation().to('cuda')
    parent, child = model('snapshot').cuda(), model('history12').cuda()
    child.load_state_dict(parent.state_dict())
    a, ga = grads(parent, [obs]); b, gb = grads(child, [obs])
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name], atol=4e-6, rtol=4e-4, msg=name)
    with torch.no_grad(): child.history_alpha.fill_(.3)
    rotation = torch.linalg.qr(torch.randn(3, 3, device='cuda'))[0]
    rotated = replace(obs, positions=obs.positions@rotation, velocities=obs.velocities@rotation)
    # Add an entirely absent identity to force ragged batching and changed masks.
    padded = replace(obs, positions=torch.cat((obs.positions, obs.positions[:, :1]+44), 1),
        velocities=torch.cat((obs.velocities, obs.velocities[:, :1]-55), 1),
        weights=torch.cat((obs.weights, torch.zeros_like(obs.weights[:, :1])), 1),
        atom_ids=torch.cat((obs.atom_ids, obs.atom_ids[-1:]+1)))
    expected = child([obs]).expand(3, -1)
    torch.testing.assert_close(child([obs, rotated, padded]), expected, atol=3e-6, rtol=3e-5)
