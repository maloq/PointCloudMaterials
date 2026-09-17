"""CUDA backend equivalence, including the symmetric-product parameter basis."""
import copy
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from src.models.encoders.predictive_memory import PredictiveMemoryEncoder
from src.models.encoders.mace_backend import mace_backend_config, mace_backend_metadata, with_mace_backend
from src.training_methods.predictive_memory.objective import PathHeads, fit_scaler
from src.training_methods.predictive_memory.runtime import MemoryRuntime
from src.training_methods.predictive_memory.train import training_update, save_checkpoint, restore_checkpoint
from test_predictive_memory_batch import windows, Dataset


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def mapped_pair(**options):
    # Official MACE conversion is used only as an equivalence-test oracle.
    # Training starts from scratch; there is no checkpoint migration path.
    from mace.cli.convert_e3nn_cueq import transfer_weights
    args = dict(channels=4, output_dim=8, frame_chunk=8, **options)
    torch.manual_seed(23)
    reference = PredictiveMemoryEncoder(**args).cuda()
    accelerated = PredictiveMemoryEncoder(**args, mace_backend='cueq').cuda()
    transfer_weights(reference, accelerated, 2, 2, 2, True, True)
    a, b = dict(reference.named_parameters()), dict(accelerated.named_parameters())
    shared_a = {n for n in a if 'symmetric_contractions' not in n}
    shared_b = {n for n in b if 'symmetric_contractions' not in n}
    assert shared_a == shared_b  # conversion may not silently omit parameters
    for name in shared_a:
        torch.testing.assert_close(a[name], b[name].reshape_as(a[name]), atol=0, rtol=0)
    assert mace_backend_metadata('cueq')['layout'] == 'mul_ir'
    for block in accelerated.products:
        assert type(block.symmetric_contractions).__module__.startswith('cuequivariance_torch.')
        assert block.symmetric_contractions.method == 'uniform_1d'
    for block in accelerated.interactions:
        assert type(block.conv_tp).__module__.startswith('cuequivariance_torch.')
    return reference, accelerated


def check_parameter_gradients(reference, accelerated):
    import cuequivariance as cue
    from mace.tools.cg import O3_e3nn
    from mace.tools.cg_cueq_tools import symmetric_contraction_proj
    a, b = dict(reference.named_parameters()), dict(accelerated.named_parameters())
    maximum = 0.
    for name in a:
        if 'symmetric_contractions' in name:
            continue
        assert (a[name].grad is None) == (b[name].grad is None), name
        if a[name].grad is not None:
            actual = b[name].grad.reshape_as(a[name])
            torch.testing.assert_close(a[name].grad, actual, atol=3e-6, rtol=3e-4, msg=name)
            maximum = max(maximum, float((a[name].grad-actual).abs().max()))
    for layer in range(2):
        # W_cueq = W_e3nn @ P, hence grad_e3nn = grad_cueq @ P.T.
        _, projection = symmetric_contraction_proj(cue.Irreps(O3_e3nn, '0e + 1o + 2e'),
                                                   cue.Irreps(O3_e3nn, '0e + 1o + 2e'), [1, 2])
        weight = b[f'products.{layer}.symmetric_contractions.weight']
        projection = torch.as_tensor(projection, device=weight.device, dtype=weight.dtype)
        mapped = torch.einsum('zbu,ab->zau', weight.grad, projection)
        expected = torch.cat([a[f'products.{layer}.symmetric_contractions.contractions.{k}.weights{suffix}'].grad
                              for k in range(3) for suffix in ('_max', '.0')], dim=1)
        torch.testing.assert_close(mapped, expected, atol=3e-6, rtol=3e-4)
        maximum = max(maximum, float((mapped-expected).abs().max()))
    return maximum


def test_backend_selection_fails_loudly(monkeypatch):
    from mace.modules import wrapper_ops
    assert mace_backend_config('e3nn') is None
    with pytest.raises(ValueError, match='Unknown MACE backend'):
        mace_backend_config('auto')
    monkeypatch.setattr(wrapper_ops, 'CUET_AVAILABLE', False)
    with pytest.raises(RuntimeError, match='refusing an e3nn fallback'):
        mace_backend_config('cueq')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuEquivariance equivalence requires CUDA')
@pytest.mark.parametrize('history,velocity,repeat,checkpoint', [
    (False, False, False, False), (False, True, False, False), (True, False, False, False),
    (True, True, False, False), (True, True, True, False), (True, True, False, True)])
def test_cueq_matches_reference_outputs_input_and_all_parameter_gradients(history, velocity, repeat, checkpoint):
    reference, accelerated = mapped_pair(use_history=history, use_velocity=velocity,
                                        repeat_anchor=repeat, activation_checkpoint=checkpoint)
    first = [o.to('cuda') for o in windows(not history)]
    second = copy.deepcopy(first)
    for obs in first+second:
        obs.positions.requires_grad_(); obs.velocities.requires_grad_()
    a, b = reference(first), accelerated(second)
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    a.square().mean().backward(); b.square().mean().backward()
    check_parameter_gradients(reference, accelerated)
    for a,b in zip(first,second,strict=True):
        torch.testing.assert_close(a.positions.grad, b.positions.grad, atol=3e-6, rtol=3e-4)
        if velocity:
            torch.testing.assert_close(a.velocities.grad, b.velocities.grad, atol=3e-6, rtol=3e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuEquivariance test requires CUDA')
def test_cueq_full_history_rotation_and_independent_windows():
    torch.manual_seed(5)
    model = PredictiveMemoryEncoder(channels=4, output_dim=8, frame_chunk=65,
                                   activation_checkpoint=True, mace_backend='cueq').cuda()
    a,b = [o.to('cuda') for o in windows()[:2]]
    a = replace(a, positions=a.positions[-1:].repeat(65,1,1), velocities=a.velocities[-1:].repeat(65,1,1),
                weights=a.weights[-1:].repeat(65,1), edges=a.edges[-1:]*65,
                offsets_ps=torch.arange(-64,1,device='cuda',dtype=torch.float64)*.75)
    rotation = torch.linalg.qr(torch.randn(3,3,device='cuda'))[0]
    rotated = [replace(o,positions=o.positions@rotation,velocities=o.velocities@rotation) for o in (a,b)]
    values = model([a,b])
    torch.testing.assert_close(values, model(rotated), atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(values, torch.cat([model([a]),model([b])]), atol=3e-6, rtol=3e-5)
    values.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuEquivariance resume requires CUDA')
def test_cueq_optimizer_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(5)
    dataset = Dataset()
    runtime = MemoryRuntime(dataset, 'cuda', dict(observation_cache_gib=.01))
    normalizer = fit_scaler(runtime.present[:3], runtime.future[:3]); runtime.normalize(normalizer)
    model = nn.ModuleDict(dict(encoder=PredictiveMemoryEncoder(channels=4,output_dim=8,mace_backend='cueq'),
                              heads=PathHeads(8,2,components=2,rank=1,hidden=8))).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.0003)
    training = dict(present_weight=.2, gradient_clip=5.)
    config, variant = dict(encoder=dict(mace_backend='cueq'),training=training), dict(history_ps=12.)
    sampler = np.random.default_rng(8)
    training_update(model,runtime,[0,1,2],optimizer,training,2)
    save_checkpoint(tmp_path/'resume.pt',model,optimizer,1,2.,config,variant,normalizer,dataset,sampler)
    expected = training_update(model,runtime,[2,0,1],optimizer,training,2)
    state = copy.deepcopy(model.state_dict())
    restore_checkpoint(tmp_path/'resume.pt',model,optimizer,config,variant,dataset,sampler,'cuda')
    actual = training_update(model,runtime,[2,0,1],optimizer,training,2)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-6, rtol=1e-5)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=1e-7, rtol=1e-6, msg=name)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native backend mapping requires CUDA')
@pytest.mark.parametrize('variant', ['snapshot', 'history12', 'repeat12'])
def test_native_backend_mapping_preserves_state_rng_and_gradients(variant):
    from src.research.local_predictability.native_model import NativeEncoder
    from test_local_predictability_native import observation
    torch.manual_seed(11)
    reference = NativeEncoder(variant=variant, channels=4, output_dim=8, activation_checkpoint=False).cuda()
    with torch.no_grad():
        reference.history_alpha.fill_(.3)
    original = copy.deepcopy(reference.state_dict())
    rng = torch.get_rng_state().clone()
    accelerated = with_mace_backend(reference, 'cueq')
    assert torch.equal(rng, torch.get_rng_state())
    for name,value in reference.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)
    obs = observation().to('cuda')
    a, b = reference([obs]), accelerated([obs])
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    a.square().mean().backward(); b.square().mean().backward()
    check_parameter_gradients(reference, accelerated)
    reference.requires_grad_(False).eval()
    frozen = with_mace_backend(reference, 'cueq')
    assert not frozen.training and all(not p.requires_grad for p in frozen.parameters())
