import copy
from types import SimpleNamespace

import numpy as np
import torch

from src.research.structural_state.dynamics import targets
from src.research.structural_state.model import StructuralModel,GraphBank,objective
from src.research.structural_state.data import graph_arrays


def test_future_baseline_and_normalizers_never_fit_heldout_labels():
    rng=np.random.default_rng(8)
    corpus=SimpleNamespace(split={'fit':np.arange(24)},
        records=[dict(temperature_K=400 if i%2 else 500,frame=64+4*i) for i in range(40)],
        geometry={'relaxed':rng.normal(size=(40,89))},
        targets={'current_order':rng.normal(size=(40,8)),'future_order_9':rng.normal(size=(40,8))})
    config={'dynamics':dict(lag_ps=9.,baseline_ridge=1.)}
    a,receipt,baseline=targets(corpus,config)
    corpus.targets['future_order_9'][24:]+=1e4
    corpus.targets['current_order'][24:]+=300
    corpus.geometry['relaxed'][24:]+=100
    b,changed,_=targets(corpus,config)
    assert receipt==changed
    for key in a:np.testing.assert_array_equal(a[key][:24],b[key][:24])
    assert np.isfinite(baseline).all()


def example():
    rng=np.random.default_rng(91);torch.manual_seed(3)
    patches=[np.vstack((np.zeros((1,3)),rng.normal(size=(9+i,3)))).astype(np.float32) for i in range(4)]
    model=StructuralModel(dict(d0=2.,n_ref=10.,radius=5.,cutoff=3.,channels=4,code_dim=12,backend='e3nn'),dynamics=True)
    bank=GraphBank(graph_arrays(patches,3.),model.encoder,'cpu')
    y={d:torch.randn(4,n) for d,n in [('observed',89),('relaxed',89),('current_order',8),('future_residual',8)]}
    arm=dict(input='relaxed',relaxed_weight=0.,current_weight=.25,relation_weight=.1,future_weight=.25)
    return model,bank,y,arm


def gradients(model,bank,y,arm):
    model.zero_grad(set_to_none=True)
    loss,_=objective(model,model(bank.batch([0,1,2,3])),y,arm,(1.,1.))
    loss.backward()
    return {n:None if p.grad is None else p.grad.clone() for n,p in model.named_parameters()}


def test_future_loss_updates_actual_encoder_but_disabled_arm_ignores_future():
    model,bank,y,arm=example();off=dict(arm,future_weight=0.)
    reference=gradients(model,bank,y,off)
    changed=copy.deepcopy(y);changed['future_residual']+=100
    repeat=gradients(model,bank,changed,off)
    for n in reference:
        if reference[n] is not None:torch.testing.assert_close(reference[n],repeat[n],rtol=0,atol=0)
    assert reference['heads.future_residual.weight'] is None
    active=gradients(model,bank,y,arm)
    assert active['heads.future_residual.weight'].norm()>0
    assert not torch.equal(active['encoder.center_embedding.weight'],reference['encoder.center_embedding.weight'])
    assert all(v.grad is None for v in y.values())


def test_future_factorial_microbatch_matches_full_gradients():
    model,bank,y,arm=example();reference=gradients(model,bank,y,arm)
    model.zero_grad(set_to_none=True)
    for ix in ([0,1],[2,3]):
        loss,_=objective(model,model(bank.batch(ix)),{k:v[ix] for k,v in y.items()},arm,(1.,1.))
        (loss/2).backward()
    for n,p in model.named_parameters():
        if reference[n] is not None:torch.testing.assert_close(p.grad,reference[n],atol=2e-6,rtol=3e-5)
