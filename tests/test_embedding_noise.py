"""Scientific controls for noise size, row weighting and reference normalization."""
import numpy as np
import pytest

from src.research.trajectory_stability.noise_metrics import noise_response
from src.research.trajectory_stability.noise import perturb


def test_linear_map_response_scales_with_noise_and_uses_matched_motion():
    x=np.zeros((6,2)); temporal=np.tile([2.,0.],(6,1)); source=np.array([1,1,1,2,2,2])
    a=noise_response(x,x+[.3,.4],x,temporal,2.,source,np.full(6,.25),.1)
    b=noise_response(x,x+[.6,.8],x,temporal,2.,source,np.full(6,1.),.2)
    assert a['response_rms']==pytest.approx(.25)
    assert a['noise_to_temporal_ratio']==pytest.approx(.25)
    assert a['sensitivity_per_A']==pytest.approx(.5)
    assert b['response_rms']==pytest.approx(2*a['response_rms'])
    assert b['sensitivity_per_A']==pytest.approx(a['sensitivity_per_A'])
    assert a['repeat_rms']==0 and a['response_to_repeat_ratio'] is None


def test_equal_source_weight_not_equal_row_weight_and_no_hidden_zero_denominator():
    x=np.zeros((4,1)); delta=np.array([[1.],[3.],[3.],[3.]])
    row=noise_response(x,delta,x,np.zeros_like(x),.5,np.array(['a','b','b','b']),np.ones(4),1.)
    assert row['response_rms']==pytest.approx(np.sqrt(5))
    assert row['noise_to_temporal_ratio'] is None
    assert row['response_p95']==3.
    with pytest.raises(ValueError,match='Positive fixed reference'):
        noise_response(x,delta,x,x,0,np.arange(4),np.ones(4),1.)


def test_embedding_rescaling_and_rotation_preserve_normalized_response():
    rng=np.random.default_rng(7);x=rng.normal(size=(20,3));eps=rng.normal(size=x.shape);motion=3*eps
    q,_=np.linalg.qr(rng.normal(size=(3,3)))
    a=noise_response(x,x+eps,x,motion,4.,np.repeat([1,2],10),np.ones(20),.1)
    b=noise_response((x@q)*5+8,((x+eps)@q)*5+8,(x@q)*5+8,(motion@q)*5,100.,np.repeat([1,2],10),np.ones(20),.1)
    for key in ['response_rms','noise_to_temporal_ratio','response_p95']:
        assert a[key]==pytest.approx(b[key])


def test_perturbation_holds_center_fixed_and_measures_realized_input_energy():
    rng=np.random.default_rng(15);x=rng.normal(size=(100,3)).astype(np.float32);x[17]=0
    eps=rng.normal(size=x.shape);eps[17]=0;ids=np.arange(100)
    a=perturb([x],[ids],[17],[eps],.01)
    np.testing.assert_array_equal(a['positions'][17],[0,0,0])
    expected=np.square(a['positions'].astype(float)-x).sum(1)[np.arange(100)!=17].mean()
    assert a['input_mse_A2'][0]==pytest.approx(expected)
    assert a['nearest80'].shape==(1,80,3)
    clean=perturb([x],[ids],[17],[eps],0.)
    np.testing.assert_array_equal(clean['positions'],x)
    assert clean['input_mse_A2'][0]==0
    eps[17]=1
    with pytest.raises(ValueError,match='fixed tracked center'):
        perturb([x],[ids],[17],[eps],.1)
