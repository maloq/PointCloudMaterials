"""Execution changes must preserve joint teacher gradients and full-batch losses."""
import copy
import numpy as np
import pytest
import torch
from test_neighborhood_jepa import sample
from test_neighborhood_jepa_v2 import fcc
from test_neighborhood_multihorizon import setup
from src.data.structural_pretraining.batches import collate, move
from src.training_methods.neighborhood_jepa.execution import (
    MACE_FIELDS, collate_graphs, pack, prime_encoder, select_profile, stage_inputs, training_step)
from src.training_methods.neighborhood_jepa.multihorizon.data import pack as original_pack
from src.training_methods.neighborhood_jepa.multihorizon.model import Model
from src.training_methods.neighborhood_jepa.multihorizon.objective import Objective
from src.training_methods.neighborhood_jepa.v2.runtime import training_step as replay_step
from src.training_methods.shared_pretraining.compilation import compile_encoder


def observations(n=34):
    return [sample(fcc()*(1+.003*i)) for i in range(n)]


def test_explicit_memory_tiers_do_not_change_statistical_batch():
    profiles = [dict(min_vram_GiB=80, microbatch=512, retain_chunks=4, gpu_cache=True),
                dict(min_vram_GiB=40, microbatch=256, retain_chunks=2, gpu_cache=True)]
    assert select_profile(profiles, 94)['microbatch'] == 512
    assert select_profile(profiles, 45)['microbatch'] == 256
    with pytest.raises(ValueError, match='No execution profile'):
        select_profile(profiles, 24)


def test_packed_input_schema_exactly_matches_producer():
    views = observations(7)
    # Include unequal support sizes and zero edges, not just repeated shapes.
    views[1] = sample(fcc()[:4])
    views[2]['edges'] = np.empty((2, 0), np.int64)
    actual, expected = collate_graphs(views), collate(views, 'mace')
    assert set(actual) == set(MACE_FIELDS)
    for name in MACE_FIELDS:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_packing_keeps_anchor_view_order_masks_and_reservoir():
    _, _, _, target, _ = setup(n=2)
    target['future_valid'][0, 2] = False
    target['reservoir'] = torch.randn(2, 2, 64)
    target.update(query_atom_ids=torch.arange(14).reshape(2,7), frame=torch.tensor([10,20]),
                  index=torch.tensor([7,3]))
    views = observations()
    samples = [{**{k: v[i].numpy() for k, v in target.items()},
                'views': views[i*17:(i+1)*17]} for i in range(2)]
    for s in samples:
        for name in ('frame', 'index', 'group', 'temperature_K'):
            s[name] = s[name].item()
    a, ta = pack(samples, 7)
    b, tb = original_pack(samples, 7)
    assert ta.keys() == tb.keys()
    for name in ta:
        torch.testing.assert_close(ta[name], tb[name], rtol=0, atol=0)
    for x, y in zip(a, b, strict=True):
        for name in MACE_FIELDS:
            torch.testing.assert_close(x[name], y[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='cuEquivariance CUDA')
@pytest.mark.parametrize('regularizer,precision,retain,cache,compiled', [
    ('sigreg','float32',0,True,False),
    ('sigreg','float32',1,True,False),
    ('sigreg','float32',99,True,False),
    ('vicreg','bf16',1,True,False),
    ('epi','bf16',99,False,False),
    ('vicreg','bf16',1,True,True),
])
def test_real_joint_gradients_and_replay_savings(regularizer, precision, retain, cache, compiled):
    torch.manual_seed(9)
    spec, _, manifest, target, _ = setup(n=2)
    spec['regularizer'] = regularizer
    target['reservoir'] = torch.randn(2, 2, 64)
    target['future_valid'][0, 1:] = False
    target = move(target, 'cuda')
    a, b = Model(16, spec, 1).cuda(), Model(16, spec, 1).cuda()
    b.load_state_dict(a.state_dict())
    oa = Objective(manifest, dict(mean=[0.]*8, std=[1.]*8), spec).cuda()
    ob = copy.deepcopy(oa)
    views = observations()
    reference = [collate(views[i:i+5], 'mace') for i in range(0, 34, 5)]
    batches = [collate_graphs(views[i:i+7]) for i in range(0, 34, 7)]
    if compiled:
        compile_encoder(b.encoder, move(batches[0], 'cuda'), precision)
        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
        before = {k:v.clone() for k,v in b.state_dict().items()}
        prime_encoder(b.encoder, batches[0], precision)
        assert torch.equal(torch.get_rng_state(), cpu_rng)
        assert torch.equal(torch.cuda.get_rng_state(), gpu_rng)
        for key, value in b.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    counts = []
    hook = b.encoder.register_forward_hook(lambda *args: counts.append(1)) if not compiled else None
    old, _, _ = replay_step(a, oa, reference, target, precision)
    new, _, diagnostics = training_step(b, ob, batches, target, precision, True,
                                       retain_chunks=retain, gpu_cache=cache)
    if hook:
        hook.remove()
        assert len(counts) == 2*len(batches)-min(retain, len(batches))
    assert abs(old-new) < (2e-4 if precision == 'float32' else 2e-3)
    ga = {name:p.grad for name,p in a.named_parameters()}
    gb = {name:p.grad for name,p in b.named_parameters()}
    assert {k for k,v in ga.items() if v is None} == {k for k,v in gb.items() if v is None}
    va = torch.cat([v.flatten() for v in ga.values() if v is not None])
    vb = torch.cat([v.flatten() for v in gb.values() if v is not None])
    relative = (va-vb).norm()/va.norm()
    assert relative < (.004 if precision == 'float32' else .02), float(relative)
    assert diagnostics['encoded_observations'] == 34
    assert diagnostics['retained_chunks'] == min(retain, len(batches))
    assert oa.state_dict().keys() == ob.state_dict().keys()
    for key in oa.state_dict():
        torch.testing.assert_close(oa.state_dict()[key], ob.state_dict()[key])


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU input cache')
def test_gpu_cache_reuses_storage_and_omits_unused_arrays():
    inputs = [collate(observations(3), 'mace')]
    cached = stage_inputs(inputs, 'cuda')
    again = stage_inputs(cached, 'cuda')
    assert set(cached[0]) == set(MACE_FIELDS)
    for name in MACE_FIELDS:
        assert cached[0][name].data_ptr() == again[0][name].data_ptr()
        assert cached[0][name].device.type == 'cuda'
        torch.testing.assert_close(cached[0][name].cpu(), inputs[0][name])
