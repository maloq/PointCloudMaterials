"""Physical-lag indexing, causal JEPA targets, censoring and full gradient replay."""
import copy
import numpy as np
import pytest
import torch
from test_neighborhood_jepa import sample
from test_neighborhood_jepa_v2 import fixture,fcc
from src.data.structural_pretraining.batches import collate,move
from src.training_methods.neighborhood_jepa.multihorizon.data import future_frames,view_plan
from src.training_methods.neighborhood_jepa.multihorizon.model import Model
from src.training_methods.neighborhood_jepa.multihorizon.objective import Objective,masked_horizon_means
from src.training_methods.neighborhood_jepa.multihorizon.specs import variants
from src.training_methods.neighborhood_jepa.v2.runtime import training_step


def setup(n=3):
    spec=variants(dict(seed=1,updates=8,horizons_ps=[3.,6.,9.]))[0]
    _,_,manifest,target,_=fixture(n=n)
    plan=view_plan(spec)
    target.update(order=torch.randn(n,2,8),moments=torch.randn(n,17,120),future_valid=torch.ones(n,3,dtype=torch.bool),
                  future_physical=torch.randn(n,3,85),future_tda=torch.randn(n,3,144))
    z=torch.randn(n*17,248,requires_grad=True)
    return spec,plan,manifest,target,z


def test_actual_physical_offsets_and_right_boundary():
    steps=np.arange(801)*250
    f,valid=future_frames(steps,100,3.,[3.,6.,9.])
    np.testing.assert_array_equal(f,[104,108,112]);assert valid.all()
    f,valid=future_frames(steps,796,3.,[3.,6.,9.])
    np.testing.assert_array_equal(f,[800,-1,-1]);np.testing.assert_array_equal(valid,[True,False,False])
    with pytest.raises(ValueError,match='absent'):future_frames(steps,100,3.,[3.1,6.,9.])


def test_masked_horizons_exclude_values_and_gradients():
    x=torch.tensor([[1.,100.,3.],[5.,900.,7.]],requires_grad=True)
    valid=torch.tensor([[True,False,True],[True,False,False]])
    loss=masked_horizon_means(x,valid)
    torch.testing.assert_close(loss,torch.tensor([3.,0.,3.]))
    loss.sum().backward();assert x.grad[~valid].count_nonzero()==0


def test_feature_blocks_keep_horizon_axis_and_persistence_broadcast():
    from src.training_methods.neighborhood_jepa.multihorizon.evaluate import feature_errors
    from src.training_methods.structural_pretraining.objective import PHYSICAL_BLOCKS,TDA_BLOCKS
    for width,blocks in ((85,PHYSICAL_BLOCKS),(144,TDA_BLOCKS)):
        target=torch.ones(2,3,width)*torch.tensor([1.,2.,3.])[None,:,None]
        error=feature_errors(torch.zeros(2,1,width),target,blocks)
        torch.testing.assert_close(error,torch.tensor([[1.,4.,9.],[1.,4.,9.]]))


def test_prediction_is_causal_and_targets_receive_gradients():
    spec,plan,manifest,target,z=setup()
    model=Model(16,spec,1)
    objective=Objective(manifest,dict(mean=[0.]*8,std=[1.]*8),spec)
    current=z.reshape(3,17,248)[:,plan.slot(1,0)]
    before=model.future_embeddings(current,target['temperature_K'])
    changed=z.detach().reshape(3,17,248).clone();changed[:,-3:]*=1000
    after=model.future_embeddings(changed[:,plan.slot(1,0)],target['temperature_K'])
    for a,b in zip(before,after):torch.testing.assert_close(a,b)
    target['future_valid'][0,2]=False
    total,terms=objective(model,z,target);total.backward()
    grad=z.grad.reshape(3,17,248)
    for slot in (14,15,16):assert grad[1,slot,:128].norm()>0 and grad[1,slot,128:].norm()>0
    assert grad[0,16].count_nonzero()==0
    assert grad[:,plan.slot(1,0)].norm()>0
    assert all(k in terms for k in ('future_embeddings_invariant','future_embeddings_equivariant','future_anchors'))
    assert abs(spec['future_weight']+.1875-.25)<1e-9
    assert abs(spec['prediction_weight']*spec['family_weights']['future_center']+.075-.1)<1e-9


@pytest.mark.skipif(not torch.cuda.is_available(),reason='Real cuEquivariance replay')
def test_seventeen_view_replay_preserves_joint_gradients():
    torch.manual_seed(9)
    spec,plan,manifest,target,_=setup(n=2)
    target['future_valid'][0,1:]=False
    a,b=Model(16,spec,1).cuda(),Model(16,spec,1).cuda();b.load_state_dict(a.state_dict())
    oa=Objective(manifest,dict(mean=[0.]*8,std=[1.]*8),spec).cuda();ob=copy.deepcopy(oa)
    target=move(target,'cuda');samples=[sample(fcc()*(1+.003*i)) for i in range(34)]
    x=a.encoder(move(collate(samples,'mace'),'cuda'));loss,_=oa(a,x,target);loss.backward()
    batches=[collate(samples[i:i+5],'mace') for i in range(0,34,5)]
    replay,_,_=training_step(b,ob,batches,target,'float32')
    assert abs(float(loss.detach())-replay)<2e-4
    ga=torch.cat([p.grad.flatten() for p in a.parameters() if p.grad is not None]);gb=torch.cat([p.grad.flatten() for p in b.parameters() if p.grad is not None])
    assert (ga-gb).norm()/ga.norm()<.004
