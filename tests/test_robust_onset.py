import copy
import numpy as np
import pytest
import torch
from sklearn.metrics import average_precision_score

from src.research.robust_onset.metrics import local_spacing,perturb_patch,smooth_ap
from src.research.robust_onset.model import Model,GraphBank
from src.research.robust_onset.train import ranking_backward
from src.research.structural_state.data import graph_arrays
from src.research.local_predictability.metrics import cumulative_risk

torch.set_num_threads(1)


def test_noise_fraction_is_three_dimensional_and_scale_invariant():
    rng=np.random.default_rng(7)
    x=np.vstack([np.zeros(3),rng.normal(size=(999,3))]).astype(np.float32)
    y,receipt=perturb_patch(x,.01,np.random.default_rng(10))
    assert np.array_equal(y[0],np.zeros(3))
    assert np.sqrt(receipt['input_relative_mse'])==pytest.approx(.01,rel=.04)
    yy,other=perturb_patch(x*3,.01,np.random.default_rng(10))
    np.testing.assert_allclose(yy,3*y,atol=1e-6)
    assert other['spacing_A']==pytest.approx(3*receipt['spacing_A'])
    assert local_spacing(np.vstack([np.zeros(3),np.eye(3).repeat(4,axis=0)*2]))==2.


def test_weighted_smooth_ap_converges_to_empirical_ap_and_is_not_auroc():
    score=torch.tensor([.1,.7,.4,.9,.3],dtype=torch.float64,requires_grad=True)
    y=torch.tensor([False,True,False,False,True]);w=torch.tensor([.1,.2,.3,.15,.25],dtype=torch.float64)
    expected=average_precision_score(y.numpy(),score.detach().numpy(),sample_weight=w.numpy())
    assert float(smooth_ap(score,y,w,1e-5))==pytest.approx(expected)
    loss=1-smooth_ap(score,y,w,.1);loss.backward()
    assert torch.isfinite(score.grad).all() and score.grad[y].sum()<0 and score.grad[~y].sum()>0
    torch.testing.assert_close(smooth_ap(score,y,w,.1),smooth_ap(score,y,5*w,.1))
    with pytest.raises(ValueError):smooth_ap(score,torch.zeros_like(y),w,.1)


def examples(tensor=True):
    rng=np.random.default_rng(71)
    patches=[np.vstack([np.zeros(3),rng.normal(size=(14+i,3))*1.5]).astype(np.float32) for i in range(4)]
    torch.manual_seed(8)
    model=Model(dict(d0=2.,n_ref=18.,radius=8.,cutoff=5.,channels=2,code_dim=12,backend='e3nn'),tensor,2)
    bank=GraphBank(graph_arrays(patches,5.),model.encoder,'cpu')
    return patches,model,bank


def test_tensor_readout_is_rotation_permutation_invariant_and_has_encoder_gradients():
    x,model,bank=examples()
    rot=np.linalg.qr(np.random.default_rng(5).normal(size=(3,3)))[0].astype(np.float32)
    changed=[np.vstack([p[:1],p[1:][::-1]])@rot for p in x]
    other=GraphBank(graph_arrays(changed,5.),model.encoder,'cpu')
    z=model(bank.batch([0,1,2,3]));zz=model(other.batch([0,1,2,3]))
    torch.testing.assert_close(z,zz,atol=3e-6,rtol=3e-5)
    z.square().sum().backward()
    assert model.encoder.center_embedding.weight.grad.norm()>0
    assert model.encoder.readout[-1].weight.grad.norm()>0


def test_full_ranking_gradient_cache_matches_full_graph_backpropagation():
    _,model,bank=examples(False);reference=copy.deepcopy(model)
    ids=np.arange(4);cond=torch.eye(2).repeat(2,1);labels=torch.tensor([False,True,False,True])
    weights=torch.tensor([.1,.2,.3,.4])
    value=ranking_backward(model,bank,ids,cond,labels,weights,2,.05,.2)
    z=reference(bank.batch(ids));score=cumulative_risk(reference.logits(z,cond))[:,-1]
    loss=.2*(1-smooth_ap(score,labels,weights,.05));loss.backward()
    assert value==pytest.approx(float(loss),rel=1e-5)
    for (name,p),(other,q) in zip(model.named_parameters(),reference.named_parameters(),strict=True):
        assert name==other and (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-4)
