"""Batched physics, statistical loss weighting, residency and exact new-run resume."""
import copy
from dataclasses import replace
import os

import numpy as np
import pytest
import torch
from torch import nn

from src.models.encoders.predictive_memory import PredictiveMemoryEncoder
from src.training_methods.predictive_memory.objective import PathHeads, fit_scaler
from src.training_methods.predictive_memory.runtime import MemoryRuntime, batch_settings, observation_bytes, sample_batch
from src.training_methods.predictive_memory.train import evaluate, training_update, save_checkpoint, restore_checkpoint
from test_predictive_memory import arrays, observe


@pytest.fixture(autouse=True, scope='module')
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def windows(snapshot=False):
    x, u, ids = arrays()
    samples = [observe(), observe(x[:, :12]*.9, u[:, :12]*1.2, ids[:12]), observe(x*1.1, -u, ids)]
    samples[1] = replace(samples[1], positions=samples[1].positions[-2:], velocities=samples[1].velocities[-2:],
                         weights=samples[1].weights[-2:], offsets_ps=samples[1].offsets_ps[-2:], edges=samples[1].edges[-2:])
    if snapshot:
        samples = [replace(o, positions=o.positions[-1:], velocities=o.velocities[-1:], weights=o.weights[-1:],
                           offsets_ps=o.offsets_ps[-1:], edges=o.edges[-1:]) for o in samples]
    return samples


def encoder(**options):
    return PredictiveMemoryEncoder(channels=2, output_dim=8, frame_chunk=4, num_layers=2, **options)


@pytest.mark.parametrize('history,velocity,repeat', [(False,False,False),(False,True,False),(True,False,False),(True,True,False),(True,True,True)])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_ragged_batch_matches_separate_outputs_parameter_and_input_gradients(history, velocity, repeat, checkpoint):
    device = os.environ.get('PCM_TEST_DEVICE', 'cpu')
    torch.manual_seed(8)
    packed = encoder(use_history=history, use_velocity=velocity, repeat_anchor=repeat, activation_checkpoint=checkpoint).to(device)
    separate = copy.deepcopy(packed)
    first = [o.to(device) for o in windows(not history)]
    second = copy.deepcopy(first)
    for o in first+second:
        o.positions.requires_grad_(); o.velocities.requires_grad_()
    a = packed(first)
    b = torch.cat([separate([o]) for o in second])
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)
    a.square().mean().backward(); b.square().mean().backward()
    for (name,p), (_,q) in zip(packed.named_parameters(), separate.named_parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None), name
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=2e-4, msg=name)
    for p,q in zip(first,second,strict=True):
        torch.testing.assert_close(p.positions.grad, q.positions.grad, atol=3e-6, rtol=2e-4)
        if velocity:
            torch.testing.assert_close(p.velocities.grad, q.velocities.grad, atol=3e-6, rtol=2e-4)


def test_windows_have_no_cross_attention_or_pooling_and_permutation_is_equivariant():
    torch.manual_seed(4)
    model = encoder(activation_checkpoint=False).eval()
    obs = windows()
    for o in obs:
        o.positions.requires_grad_()
    states = model(obs)
    states[0].square().sum().backward()
    assert obs[0].positions.grad.abs().sum() > 0
    for o in obs[1:]:
        assert torch.count_nonzero(o.positions.grad) == 0
    with torch.no_grad():
        torch.testing.assert_close(model([obs[2],obs[0],obs[1]]), states[[2,0,1]], atol=3e-6, rtol=3e-5)
    with pytest.raises(ValueError, match='nonempty list'):
        model(obs[0])


class Dataset:
    release_sha256 = 'batch-test-release'
    def __init__(self):
        self.observed = windows()*3
        self.rows = [dict(source_id=i//3, split=['train','val','test'][i//3], anchor=i%3, center_id=10) for i in range(9)]
        self.indices = dict(train=[0,1,2], val=[3,4,5], test=[6,7,8])
        self.present = torch.linspace(-1,1,9*128).reshape(9,128)
        self.future = torch.linspace(-2,2,9*2*128).reshape(9,2,128)
    def observation(self, index):
        return self.observed[index]
    def targets(self, indices, device):
        return self.present[indices].to(device), self.future[indices].to(device), torch.ones(len(indices),1,device=device)


def setup(device=None):
    dataset = Dataset()
    runtime = MemoryRuntime(dataset, device or os.environ.get('PCM_TEST_DEVICE', 'cpu'), dict(observation_cache_gib=.01))
    normalizer = fit_scaler(runtime.present[:3], runtime.future[:3])
    runtime.normalize(normalizer)
    model = nn.ModuleDict(dict(encoder=encoder(activation_checkpoint=False), heads=PathHeads(8,2,components=2,rank=1,hidden=8))).to(runtime.device)
    return dataset, runtime, normalizer, model


def test_microbatch_remainder_keeps_same_loss_gradients_and_optimizer_step():
    torch.manual_seed(3)
    _, runtime, _, a = setup()
    b = copy.deepcopy(a)
    training = dict(present_weight=.3, gradient_clip=5.)
    oa, ob = torch.optim.SGD(a.parameters(), lr=.001), torch.optim.SGD(b.parameters(), lr=.001)
    va = training_update(a,runtime,[0,1,2],oa,training,3)
    vb = training_update(b,runtime,[0,1,2],ob,training,2)
    for key in va:
        torch.testing.assert_close(va[key],vb[key],atol=3e-6,rtol=3e-5)
    for (name,p), (_,q) in zip(a.named_parameters(),b.named_parameters(),strict=True):
        torch.testing.assert_close(p,q,atol=3e-6,rtol=3e-5,msg=name)


def test_batched_evaluation_preserves_order_scores_and_last_short_batch():
    _, runtime, _, model = setup()
    indices = [4,3,5]
    a, za = evaluate(model,runtime,indices,1)
    b, zb = evaluate(model,runtime,indices,2)
    torch.testing.assert_close(za,zb,atol=3e-6,rtol=3e-5)
    for x,y in zip(a,b,strict=True):
        assert (x['source_id'],x['anchor'],x['center_id']) == (y['source_id'],y['anchor'],y['center_id'])
        for key in x:
            assert x[key] == pytest.approx(y[key],rel=3e-5,abs=3e-6)


def test_cache_is_byte_bounded_and_validation_does_not_evict_training():
    dataset = Dataset()
    maximum = max(observation_bytes(o) for o in dataset.observed)
    runtime = MemoryRuntime(dataset,'cpu',dict(observation_cache_gib=maximum/2**30))
    a = runtime.observation(0)
    assert runtime.observation(0) is a
    runtime.observation(3)
    assert list(runtime.cache) == [0]
    runtime.observation(1); runtime.observation(2)
    assert runtime.bytes <= maximum
    assert list(runtime.cache) == [2]
    assert runtime.observation(2).positions.requires_grad is False
    with pytest.raises(ValueError, match='positive integer'):
        batch_settings(dict(batch_size=0))
    assert batch_settings(dict(batch_size=8,micro_batch_size=3,evaluation_batch_size=4)) == (8,3,4)


@pytest.mark.parametrize('device', list(dict.fromkeys(['cpu', os.environ.get('PCM_TEST_DEVICE', 'cpu')])))
def test_resume_recovers_multiwindow_sampling_and_adam_state(tmp_path, device):
    torch.manual_seed(7)
    dataset,runtime,normalizer,model = setup(device)
    training = dict(batch_size=3,micro_batch_size=2,present_weight=.2,gradient_clip=5.)
    config, variant = dict(training=training), dict(history_ps=12.)
    optimizer = torch.optim.AdamW(model.parameters(),lr=.0003)
    sampler = np.random.default_rng(22)
    def update():
        ids = sample_batch(sampler,dataset.indices['train'],3)
        training_update(model,runtime,ids,optimizer,training,2)
        return ids
    update()
    save_checkpoint(tmp_path/'resume.pt',model,optimizer,1,2.,config,variant,normalizer,dataset,sampler)
    expected_ids = update()
    expected = copy.deepcopy(model.state_dict())
    step,_,_ = restore_checkpoint(tmp_path/'resume.pt',model,optimizer,config,variant,dataset,sampler,runtime.device)
    saved = torch.load(tmp_path/'resume.pt',weights_only=False,map_location='cpu')
    for name,p in model.state_dict().items():
        torch.testing.assert_close(p.cpu(), saved['model'][name],atol=0,rtol=0)
    for key,values in optimizer.state_dict()['state'].items():
        for name,value in values.items():
            torch.testing.assert_close(value.cpu(),saved['optimizer']['state'][key][name],atol=0,rtol=0)
    assert step == 1 and update() == expected_ids
    for name,p in model.state_dict().items():
        # CUDA scatter reductions can differ in final bits even without a restart.
        atol,rtol = (1e-7,1e-6) if device.startswith('cuda') else (0,0)
        torch.testing.assert_close(p,expected[name],atol=atol,rtol=rtol)
    old = torch.load(tmp_path/'resume.pt',weights_only=False)
    del old['format_version']
    torch.save(old,tmp_path/'old.pt')
    with pytest.raises(ValueError,match='original commit'):
        restore_checkpoint(tmp_path/'old.pt',model,optimizer,config,variant,dataset,sampler,runtime.device)


def test_full_48ps_and_short_history_can_share_spatial_batches():
    device = os.environ.get('PCM_TEST_DEVICE', 'cpu')
    base = windows()[0].to(device)
    def history(frames):
        return replace(base, positions=base.positions[-1:].repeat(frames,1,1),
                       velocities=base.velocities[-1:].repeat(frames,1,1),
                       weights=base.weights[-1:].repeat(frames,1), edges=base.edges[-1:]*frames,
                       offsets_ps=torch.arange(1-frames,1,device=device,dtype=torch.float64)*.75)
    a,b = history(17),history(65)
    model = encoder(activation_checkpoint=True).to(device)
    combined = model([a,b])
    torch.testing.assert_close(combined,torch.cat([model([a]),model([b])]),atol=3e-6,rtol=3e-5)
    combined.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
