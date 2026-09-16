"""Scientific failure modes of the sequence/motion experiment."""
import numpy as np
import pytest
import torch

from src.research.mace_local_state.motion import (MotionState, objective, orthogonal_basis,
    projector_distance, projection_residual, time_differences)
from src.research.mace_local_state.motion_data import contiguous_window
from src.research.mace_local_state.smooth import FAMILIES


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
    torch.testing.assert_close(projector_distance(basis,changed),torch.zeros(7,dtype=torch.float64),atol=1e-12,rtol=0)
    scale = torch.tensor(3.,dtype=torch.float64,requires_grad=True)
    fraction = projection_residual(scale*delta,basis).sum()/(scale*delta).square().sum()
    fraction.backward()
    assert abs(scale.grad.item())<1e-12


def test_windows_never_bridge_unobserved_frames():
    eligible = np.r_[np.arange(1,5),np.arange(30,45)]
    selected = contiguous_window(eligible,9)
    assert len(selected)==9 and np.all(np.diff(selected)==1) and selected.min()>=30
    with pytest.raises(ValueError,match='consecutive'): contiguous_window([1,3,5,7,9],3)


def test_snapshot_state_and_basis_never_read_future_frames():
    torch.manual_seed(5)
    model = MotionState(32,4).eval()
    x = torch.randn(2,5,256);changed=x.clone();changed[:,3:]+=100
    z,_ = model(x);other,_ = model(changed)
    torch.testing.assert_close(z[:,:3],other[:,:3])
    torch.testing.assert_close(model.basis(z[:,2]),model.basis(other[:,2]))


def test_basis_fit_does_not_smooth_the_control_state():
    torch.manual_seed(6);torch.set_num_threads(1)
    n,nt,d = 8,5,32
    model = MotionState(d,4)
    x = torch.randn(n,nt,256);y = torch.randn(n,nt,160)
    edge_ids = torch.arange(n*(nt-1));anchors = (edge_ids//(nt-1))*nt+edge_ids%(nt-1)
    batch = dict(x=x,y=y,times=torch.arange(nt,dtype=torch.float64)[None].repeat(n,1)*.75,
        weights=torch.ones(n*nt),low=torch.ones(n*nt,dtype=torch.bool),
        context=(torch.arange(n)//4)[:,None]*nt+torch.arange(nt)[None],
        edge_weights=torch.ones(2,n,nt-1)/(n*(nt-1)),curve_weights=torch.ones(2,n,nt-2)/(n*(nt-2)),
        direction_weights=torch.ones(2,n*(nt-1))/(n*(nt-1)),direction_ids=edge_ids,
        anchor_ids=anchors,neighbor_ids=anchors.roll(7))
    batch['context']=batch['context'].ravel()
    loss,_,_=objective(model,batch,dict(temporal=0.,direction=0.,curvature=0.),
        dict(covariance_weight=0.,basis_neighbor_weight=.01))
    loss.backward();actual=[p.grad.clone() for p in model.mapping.parameters()]
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.directions.parameters())
    model.zero_grad(set_to_none=True)
    _,pred = model(x);error=(pred-y).square()
    plain=torch.stack([error[...,s].mean() for s in FAMILIES.values()]).mean();plain.backward()
    for got,p in zip(actual,model.mapping.parameters(),strict=True):
        torch.testing.assert_close(got,p.grad,atol=1e-7,rtol=1e-5)
