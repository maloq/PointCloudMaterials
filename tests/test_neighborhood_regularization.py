"""Numerical objectives, causal export, replay, and geometric order anchors."""
import copy
import math
import numpy as np
import pytest
import torch
from e3nn import o3
from test_neighborhood_jepa import sample
from test_neighborhood_jepa_v2 import fixture, fcc
from src.data.structural_pretraining.batches import collate, move
from src.data.structural_pretraining.support import REFERENCE_RADIUS
from src.training_methods.neighborhood_jepa.regularization.data import order_targets
from src.training_methods.neighborhood_jepa.regularization.model import Model, Encoder
from src.training_methods.neighborhood_jepa.regularization.objective import Objective, epiplexity, vicreg_regularization
from src.training_methods.neighborhood_jepa.regularization.specs import variants
from src.training_methods.neighborhood_jepa.v2.runtime import training_step


def test_epiplexity_reference_gradient_and_scale_control():
    torch.manual_seed(81)
    h = torch.randn(40, 12, dtype=torch.double)
    z = (h @ torch.randn(12, 16, dtype=torch.double)).requires_grad_()
    normalized = (h-h.mean(0))/h.std(0, unbiased=False)/math.sqrt(12)
    w = torch.linalg.solve(normalized.T@normalized + 3*torch.eye(12), normalized.T@(z-z.mean(0)))
    expected = .5*torch.linalg.slogdet(torch.eye(16)+30*w.T@w).logabsdet/math.log(2)
    torch.testing.assert_close(epiplexity(z,h,False).double(), expected, rtol=1e-6, atol=1e-6)
    actual = epiplexity(z,h)
    torch.testing.assert_close(actual, epiplexity(10*z,h), rtol=1e-4, atol=1e-4)
    gradient = torch.autograd.grad(actual,z)[0]
    assert torch.isfinite(gradient).all() and gradient.norm()>0
    assert epiplexity(torch.ones_like(z),h)==0


def test_vicreg_formula_and_collapsed_penalty():
    torch.manual_seed(1)
    z = torch.randn(32,8,requires_grad=True)
    total,variance,covariance = vicreg_regularization(z)
    c = torch.cov(z.T)
    torch.testing.assert_close(variance, (1-torch.sqrt(c.diag()+1e-4)).relu().mean())
    torch.testing.assert_close(covariance, (c.square().sum()-c.diag().square().sum())/8)
    total.backward()
    assert torch.isfinite(z.grad).all()
    collapsed,_,_ = vicreg_regularization(torch.zeros_like(z))
    assert collapsed>total and abs(float(collapsed)-24.75)<1e-5


def test_temperature_vicreg_does_not_count_between_temperature_variance():
    spec=next(s for s in variants({'seed':1,'updates':8}) if s['name']=='vic-direct-raw-order')
    spec.update(regularizer_scope='temperature')
    _,plan,manifest,target,z=fixture(n=8);target['order']=torch.randn(8,2,8)
    target['temperature_K']=torch.tensor([400.,400.,400.,400.,500.,500.,500.,500.])
    model=Model(16,spec,1);objective=Objective(manifest,dict(mean=[0.]*8,std=[1.]*8),spec)
    with torch.no_grad():
        current=z.reshape(8,len(plan.views),-1)[:,plan.slot(1,0),:128]
        current[:4]=0;current[4:]=100
    loss,terms=objective(model,z,target)
    torch.testing.assert_close(terms['regularizer'],torch.tensor(spec['regularizer_weight']*24.75))
    loss.backward();assert torch.isfinite(z.grad).all()


def test_order_rotation_permutation_and_fcc():
    # Include surrounding FCC cells for the neighbors' own 12-bond environments.
    grid = np.array([[i,j,k] for i in range(-3,4) for j in range(-3,4) for k in range(-3,4)
                     if (i+j+k)%2==0 and (i,j,k)!=(0,0,0)],np.float64)*2
    x = np.vstack((np.zeros((1,3)),grid))
    a = order_targets(x,REFERENCE_RADIUS)
    r = o3.rand_matrix().double().numpy()
    b = order_targets(x@r.T,REFERENCE_RADIUS)
    c = order_targets(np.vstack((x[:1],x[1:][np.random.default_rng(2).permutation(len(x)-1)])),REFERENCE_RADIUS)
    np.testing.assert_allclose(a,b,rtol=3e-5,atol=1e-6)
    np.testing.assert_allclose(a,c,rtol=3e-5,atol=1e-6)
    assert abs(a[0]-.19094065)<1e-5 and abs(a[1]-.57452426)<1e-5


@pytest.mark.parametrize('spec',variants({'seed':1,'updates':8}),ids=lambda s:s['name'])
def test_objective_variants_and_target_gradients(spec):
    _,plan,manifest,target,z = fixture(n=8)
    target['order'] = torch.randn(8,2,8)
    target['reservoir'] = torch.randn(8,2,64)
    model = Model(16,spec,1)
    objective = Objective(manifest,dict(mean=[0.]*8,std=[1.]*8),spec)
    loss,terms = objective(model,z,target)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    future = z.grad.reshape(8,len(plan.views),-1)[:,plan.slot(2,1)]
    assert future.norm()>0
    if spec['order_weight']:
        assert model.order_decoder[-1].weight.grad.norm()>0
    assert set(terms)=={'physical','tda','geometry','future','prediction_invariant','prediction_equivariant','regularizer','order'}


@pytest.mark.skipif(not torch.cuda.is_available(),reason='cuEquivariance CUDA')
@pytest.mark.parametrize('norm',['layernorm','raw'])
def test_training_export_independent_of_other_observations(norm):
    torch.manual_seed(11)
    encoder=Encoder(16,norm).cuda().train()
    x=fcc(); other=fcc()*1.5
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        a=encoder(move(collate([sample(x)],'mace'),'cuda'))
        b=encoder(move(collate([sample(x),sample(other)],'mace'),'cuda'))
        r=o3.rand_matrix().numpy()
        rotated=encoder(move(collate([sample(x@r.T)],'mace'),'cuda'))
    torch.testing.assert_close(a,b[:1],atol=2e-3,rtol=2e-3)
    torch.testing.assert_close(a[:,:128],rotated[:,:128],atol=2e-3,rtol=2e-3)
    assert not any(isinstance(m,torch.nn.BatchNorm1d) for m in encoder.modules())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='cuEquivariance CUDA replay')
@pytest.mark.parametrize('regularizer',['vicreg','epi'])
def test_full_batch_vs_replay_gradient(regularizer):
    torch.manual_seed(7)
    spec=next(s for s in variants({'seed':1,'updates':8}) if s['regularizer']==regularizer)
    _,plan,manifest,target,_=fixture(n=3)
    target.update(order=torch.randn(3,2,8),reservoir=torch.randn(3,2,64))
    target=move(target,'cuda')
    a,b=Model(16,spec,1).cuda().train(),Model(16,spec,1).cuda().train()
    b.load_state_dict(a.state_dict())
    oa=Objective(manifest,dict(mean=[0.]*8,std=[1.]*8),spec).cuda();ob=copy.deepcopy(oa)
    samples=[sample(fcc()*(1+.01*i)) for i in range(3*len(plan.views))]
    encoded=a.encoder(move(collate(samples,'mace'),'cuda'))
    loss,_=oa(a,encoded,target);loss.backward()
    packed=[collate(samples[i:i+5],'mace') for i in range(0,len(samples),5)]
    replay,_,_=training_step(b,ob,packed,target,'float32')
    assert abs(float(loss.detach())-replay)<1e-4
    ga=torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None])
    gb=torch.cat([p.grad.flatten() for p in b.parameters() if p.grad is not None])
    assert (ga-gb).norm()/ga.norm()<.003
