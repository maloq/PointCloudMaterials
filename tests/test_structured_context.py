import copy
import numpy as np
import pytest
import torch
from src.research.structured_context.geometry import stencil,assign
from src.research.structured_context.model import StructuredHead,StructuredBlock


def test_symmetric_stencil_and_unique_equivariant_assignment():
    q=stencil();assert q.shape==(25,3)
    for shell in (q[1:13],q[13:]):
        np.testing.assert_allclose(shell.mean(0),0,atol=1e-7)
        moment=shell.T@shell/12
        np.testing.assert_allclose(moment,np.eye(3)*np.trace(moment)/3,atol=2e-5)
        assert np.max(np.min(np.linalg.norm(shell[:,None]+shell[None],axis=-1),axis=1))<1e-6
    rng=np.random.default_rng(7);x=q[1:]+rng.normal(0,.1,(24,3));ids=np.arange(24)
    chosen,actual=assign(x,ids,q)
    rot,_=np.linalg.qr(rng.normal(size=(3,3)))
    moved,positions=assign(x@rot,ids,q@rot)
    np.testing.assert_array_equal(chosen,moved);np.testing.assert_allclose(positions,actual@rot)
    with pytest.raises(ValueError,match='exceeds'):assign(x+100,ids,q)


def test_structured_head_rotation_permutation_and_gradients():
    torch.manual_seed(9);torch.set_num_threads(1)
    spec=dict(head_width=32,heads=4,depth=2,norm_eps=1e-8,shell_radii_A=[10.,20.])
    model=StructuredHead(spec).eval();b,t,n=2,4,25
    f=torch.randn(b,t*n,128,requires_grad=True);xyz=model.queries.repeat(t,1)[None].repeat(b,1,1)+torch.randn(b,t*n,3)*.2
    times=torch.tensor([-48.,-12.,-3.,0.]).repeat_interleave(n).expand(b,-1)
    g=torch.cat((xyz,times[...,None]),-1);condition=torch.randn(b,7)
    _,w=model.inputs(f,g);model.normalization.calibrate(f.detach(),w)
    base=model(f,g,condition);base.square().mean().backward();assert torch.isfinite(f.grad).all() and f.grad.abs().sum()>0
    rotated=copy.deepcopy(model);rotation=torch.linalg.qr(torch.randn(3,3)).Q
    rotated.queries.copy_(model.queries@rotation);rg=torch.cat((xyz@rotation,times[...,None]),-1)
    torch.testing.assert_close(rotated(f,rg,condition),base,atol=2e-6,rtol=2e-5)
    p=torch.cat((torch.tensor([0]),1+torch.randperm(12),13+torch.randperm(12)))
    permuted=copy.deepcopy(model);permuted.queries.copy_(model.queries[p]);permuted.slot_weights.copy_(model.slot_weights[p])
    pf=f.reshape(b,t,n,128)[:,:,p].flatten(1,2);pg=g.reshape(b,t,n,4)[:,:,p].flatten(1,2)
    torch.testing.assert_close(permuted(pf,pg,condition),base,atol=2e-6,rtol=2e-5)


def test_temporal_attention_cannot_read_later_observation():
    torch.manual_seed(3);block=StructuredBlock(16,4).eval();x=torch.randn(2,4,16)
    actual=torch.randn(2,4,4);actual[:,:,3]=torch.tensor([-48.,-12.,-3.,0.]);nominal=torch.randn(2,1,3).expand(-1,4,-1)
    before=block(x,actual,nominal,torch.ones(2,4),True)
    changed=x.clone();changed[:,-1]+=100
    after=block(changed,actual,nominal,torch.ones(2,4),True)
    torch.testing.assert_close(before[:,:3],after[:,:3],atol=1e-6,rtol=1e-6)
