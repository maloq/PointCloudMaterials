"""Scientific invariants for new packed/nested native computation."""
import copy

import numpy as np
import pytest
import torch
from src.data.predictive_memory.observations import frame_observation, assemble
from src.research.local_predictability.native_model import NativeEncoder, PhysicalMeans, anchor_only
from src.models.encoders.predictive_memory.model import PredictiveMemoryEncoder


def observation(seed=7):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-5, 5, (10, 3)); x[0] = 0
    # One history-only atom, including a disappearing node in the padded union.
    x[-1] = (16., 0., 0.)
    frames = []
    for k in range(17):
        position = x.copy(); position[-1, 0] += k / 8
        velocity = rng.normal(size=(10, 3))
        frames.append(frame_observation(position, velocity, np.full(3, 100.),
                                       np.arange(1, 11), 1, 17., 5.))
    return assemble(frames, np.arange(17) * .75, radius=17., cutoff=5.)


def model(variant):
    torch.manual_seed(20260919)
    return NativeEncoder(variant=variant, activation_checkpoint=False)


def gradients(encoder, obs):
    encoder.zero_grad(set_to_none=True)
    result = encoder(obs)
    result.square().sum().backward()
    return result.detach(), {name: p.grad.detach().clone() for name, p in encoder.named_parameters() if p.grad is not None}


def test_snapshot_matches_maintained_producer():
    obs = anchor_only(observation())
    native = model('snapshot')
    reference = PredictiveMemoryEncoder(use_history=False, activation_checkpoint=False)
    state = {k: v for k, v in native.state_dict().items() if k != 'history_alpha'}
    reference.load_state_dict(state)
    torch.testing.assert_close(native([obs]), reference([obs]), atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('variant', ['history12', 'repeat12'])
def test_zero_gate_is_nested_and_learns(variant):
    obs = observation()
    parent, child = model('snapshot'), model(variant)
    child.load_state_dict(parent.state_dict())
    a, ga = gradients(parent, [obs])
    b, gb = gradients(child, [obs])
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    for name in ga:
        torch.testing.assert_close(ga[name], gb[name], atol=3e-5, rtol=5e-4, msg=name)
    gate = gb['history_alpha']
    assert torch.isfinite(gate).all() and torch.all(gate.abs() > 1e-9)


@pytest.mark.parametrize('variant', ['snapshot', 'history12', 'repeat12'])
def test_packed_output_and_gradient_parity(variant):
    obs = [observation(7), observation(11)]
    packed, serial = model(variant), model(variant)
    with torch.no_grad():
        packed.history_alpha.fill_(.2)
    serial.load_state_dict(packed.state_dict())
    a, ga = gradients(packed, obs)
    serial.zero_grad(set_to_none=True)
    b = torch.cat([serial([item]) for item in obs])
    b.square().sum().backward()
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    for name, parameter in serial.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(ga[name], parameter.grad, atol=5e-5, rtol=5e-4, msg=name)


def test_full_cadence_required():
    obs = observation()
    obs.offsets_ps[0] = -13
    with pytest.raises(ValueError, match='entire declared causal cadence'):
        model('history12')([obs])


def test_heads_have_declared_shapes():
    torch.manual_seed(20260919)
    means = PhysicalMeans(activation_checkpoint=False)
    result = means([observation()], torch.zeros(1, 7))
    assert result['state'].shape == result['present'].shape == (1, 128)
    assert result['future'].shape == (1, 6, 128)
