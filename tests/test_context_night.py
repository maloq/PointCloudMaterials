import numpy as np
import pytest
import torch
from src.research.context_night.context import information_values,context_mask,INFO_DIM
from src.research.context_night.metrics import short_logits
from src.research.local_predictability.metrics import cumulative_risk


def test_context_strictly_causal_and_tracked_center():
    x=torch.arange(2*24*3*97,dtype=torch.float32).reshape(2,24,3,97)
    rows=torch.tensor([[0,32,1,450],[1,48,2,500]])
    y=information_values(x,rows)
    assert y.shape==(2,376)
    torch.testing.assert_close(y[0,:93],x[0,8,1,:93]);torch.testing.assert_close(y[0,93:186],x[0,8,1,:93]-x[0,7,1,:93])
    changed=x.clone();changed[0,9:]=1e8;changed[1,13:]=1e8
    torch.testing.assert_close(y,information_values(changed,rows))
    with pytest.raises(ValueError):information_values(x,torch.tensor([[0,12,1,450]]))


def test_matched_context_masks():
    assert INFO_DIM==504
    assert context_mask('control').sum()==0
    assert context_mask('shells').sum()==4
    assert context_mask('history').sum()==372
    assert context_mask('both').sum()==376
    assert context_mask('both_new').sum()==504
    with pytest.raises(ValueError):context_mask('typo')


def test_short_hazards_reconstruct_selected_dense_risk():
    rng=np.random.default_rng(2);cdf=1-np.cumprod(1-rng.uniform(.0001,.1,(7,128)),axis=1)
    actual=cumulative_risk(torch.from_numpy(short_logits(cdf))).numpy()
    np.testing.assert_allclose(actual,cdf[:,[0,3,7,11,15]],rtol=1e-10,atol=1e-10)


def test_descriptors_only_cannot_observe_encoder_features():
    from test_crystallization_path_refinement import example
    from src.research.context_night.context import ContextForecaster
    parent,observed,_=example('direct')
    model=ContextForecaster(dict(parent.spec,information_context='both',remove_encoder_context=True)).eval()
    model.load_state_dict(parent.state_dict(),strict=False)
    observed['information']=torch.randn(3,504)
    expected=model.encode(observed)
    observed['features']=torch.randn_like(observed['features'])*100
    torch.testing.assert_close(expected,model.encode(observed))


@pytest.mark.parametrize('arm',[0,1,2,3])
def test_information_objective_gradient_and_teacher_targets(arm):
    from src.research.context_night.encoder import Model,Objective
    from src.research.context_night.specs import encoder_specs
    from test_neighborhood_jepa_v2 import fixture
    spec=encoder_specs({'encoder':{'seed':7,'updates':8,'warm_checkpoint':'unused'}})[arm]
    _,plan,manifest,target,z=fixture(n=8);target['order']=torch.randn(8,2,8)
    model=Model(16,spec,7);objective=Objective(manifest,dict(mean=[0.]*8,std=[1.]*8),spec)
    loss,terms=objective(model,z,target);loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert z.grad.reshape(8,len(plan.views),-1)[:,plan.slot(2,1)].norm()>0
    assert (model.linear_order.weight.grad.norm()>0)==(spec['linear_order_weight']>0)
    assert (model.linear_angular.weight.grad.norm()>0)==(spec['linear_angular_weight']>0)


@pytest.mark.parametrize('method',['direct','ar_mse','mixture','diffusion'])
def test_context_forecasts_start_at_parent_and_train_additional_signal(method):
    from test_crystallization_path_refinement import example
    from src.research.context_night.context import ContextForecaster
    parent,observed,target=example(method)
    model=ContextForecaster(dict(parent.spec,information_context='both'))
    result=model.load_state_dict(parent.state_dict(),strict=False)
    assert all(k.startswith('information_') for k in result.missing_keys) and not result.unexpected_keys
    observed['information']=torch.randn(3,504)
    parent.eval();model.eval()
    torch.testing.assert_close(model.encode(observed),parent.encode({k:v for k,v in observed.items() if k!='information'}))
    model.train();model.loss(observed,target,.5).mean().backward()
    assert model.information_head[-1].weight.grad.norm()>0
    model.eval();paths,cdf=model.forecast(observed,2,4)
    assert torch.isfinite(paths).all() and (cdf.diff(dim=-1)>=-1e-6).all()
