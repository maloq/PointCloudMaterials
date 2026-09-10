"""Identity-preserving thermal pairs and exact six-view gradient replay."""
from types import SimpleNamespace
import numpy as np
import torch
from src.data_utils.mace_relaxed import paired_clouds
from src.training_methods.mace_objective import objective,cached_step,training_views
from src.training_methods.mace_performance import encode_views


def test_relaxation_keeps_hot_membership_even_when_neighbors_exchange():
    rng=np.random.default_rng(12);hot=rng.uniform(0,12,(100,3));relaxed=hot.copy()
    a,b,ids,error=paired_clouds(hot,relaxed,np.full(3,12.),np.array([0,1]))
    np.testing.assert_array_equal(a,b)
    moved=ids[0,-1];relaxed[moved]=hot[0]+.05
    a,b,after,error=paired_clouds(hot,relaxed,np.full(3,12.),np.array([0,1]))
    np.testing.assert_array_equal(ids,after)
    np.testing.assert_allclose(b[0,-1],.05,atol=2e-5)
    assert max(error)<.003
    np.testing.assert_array_equal(b[:,0],np.zeros((2,3)))


def test_six_view_replay_matches_direct_with_delayed_shared_tda_targets():
    class Encoder(torch.nn.Module):
        def __init__(self):super().__init__();self.linear=torch.nn.Linear(3,256)
        def build_geometry(self,x,m):return SimpleNamespace(batch_size=len(x),x=x.mean(1))
        def forward_from_geometry(self,g):return self.linear(g.x)
        def forward(self,x,m):return self.forward_from_geometry(self.build_geometry(x,m))
    torch.manual_seed(8)
    model=torch.nn.Module();model.encoder=Encoder();model.tda=torch.nn.Linear(256,32)
    loss_cfg=dict(invariance=25.,variance=25.,covariance=1.,thermal=25.,tda=1.,tda_start_epoch=6)
    cfg=dict(loss=loss_cfg,microbatch_size=7);views=training_views(loss_cfg)
    target=torch.randn(12,4,32).repeat(1,2,1)
    batch=(torch.randn(12,8,80,3),target,torch.randn(12,5),torch.arange(12)%3)
    mask=torch.ones(12,dtype=torch.bool)
    for epoch in (5,6):
        model.tda.requires_grad_(epoch>=6);model.zero_grad(set_to_none=True)
        value,parts=cached_step(model,batch,mask,cfg,epoch)
        assert 'loss_thermal' in parts and ('tda_mse' in parts)==(epoch==6)
        expected={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
        model.zero_grad(set_to_none=True)
        z=encode_views(model,batch[0][:,views],batch[3],7)
        direct,_=objective(model,z,target[:,views],mask,loss_cfg,epoch);direct.backward()
        for name,p in model.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p.grad,expected[name],rtol=3e-4,atol=3e-5)
