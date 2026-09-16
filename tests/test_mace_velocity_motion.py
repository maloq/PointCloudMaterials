"""Temporal geometry checks retained for end-to-end encoder training."""
import pytest
import torch

from src.research.mace_velocity.motion import orthogonal_basis, projection_residual, time_differences


def test_uneven_times_have_zero_bending_for_constant_velocity():
    t = torch.tensor([[0.,.03,.13,.43,1.18]],dtype=torch.float64)
    speed = torch.tensor([2.,-3.],dtype=torch.float64)
    z = 5+t[...,None]*speed
    _,v,a,b = time_differences(z,t)
    torch.testing.assert_close(v,speed.expand_as(v))
    torch.testing.assert_close(a,torch.zeros_like(a),atol=1e-10,rtol=0)
    torch.testing.assert_close(b,torch.zeros_like(b),atol=1e-12,rtol=0)
    with pytest.raises(ValueError,match='increase'): time_differences(z,t*0)


def test_uneven_times_recover_constant_acceleration():
    t = torch.tensor([[0.,.03,.13,.43,1.18]],dtype=torch.float64)
    acceleration = torch.tensor([2.,-4.],dtype=torch.float64)
    z = .5*t[...,None]**2*acceleration
    _,_,a,_ = time_differences(z,t)
    torch.testing.assert_close(a,acceleration.expand_as(a))


def test_projection_depends_on_subspace_and_not_basis_orientation():
    torch.manual_seed(3)
    basis = orthogonal_basis(torch.randn(7,12,4,dtype=torch.float64))
    rotation = orthogonal_basis(torch.randn(7,4,4,dtype=torch.float64))
    changed = basis@rotation
    delta = torch.randn(7,12,dtype=torch.float64)
    torch.testing.assert_close(projection_residual(delta,basis),projection_residual(delta,changed))
    scale = torch.tensor(3.,dtype=torch.float64,requires_grad=True)
    fraction = projection_residual(scale*delta,basis).sum()/(scale*delta).square().sum()
    fraction.backward()
    assert abs(scale.grad.item())<1e-12
