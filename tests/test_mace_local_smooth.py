import numpy as np
import torch

from src.research.mace_local_state.smooth import within_covariance, jump_metrics


def test_context_variance_does_not_reward_source_offsets():
    z=torch.tensor([[0.,1.],[2.,3.],[1.,4.],[3.,2.]],dtype=torch.float64)
    w=torch.tensor([1.,1.,3.,3.],dtype=torch.float64)
    context=torch.tensor([0,0,1,1])
    expected=torch.tensor([[1.,-.5],[-.5,1.]],dtype=torch.float64)
    torch.testing.assert_close(within_covariance(z,w,context),expected)
    shifted=z+torch.tensor([[1000.,-7.],[1000.,-7.],[-600.,90.],[-600.,90.]])
    torch.testing.assert_close(within_covariance(shifted,w,context),expected)


def test_jump_is_scale_invariant_and_training_reference_only():
    z=np.array([[0.,0.],[0.,0.],[2.,0.],[2.,0.],[1.,0.],[1.5,0.],[1.,0.],[2.,0.]])
    data={'raw_target':np.zeros((8,169))}
    actual=jump_metrics(z,data,np.array([2,3]),np.array([0,1]),.3)
    assert np.isclose(actual['rms_jump'],np.sqrt(.625/2))
    scaled=jump_metrics(z*100,data,np.array([2,3]),np.array([0,1]),.3)
    assert np.isclose(scaled['rms_jump'],actual['rms_jump'])
    moved=z.copy();moved[4:]+=100
    other=jump_metrics(moved,data,np.array([2,3]),np.array([0,1]),.3)
    assert np.isclose(other['reference_squared_distance'],actual['reference_squared_distance'])
    assert np.isclose(other['rms_jump'],actual['rms_jump'])
