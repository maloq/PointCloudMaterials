"""Scientific contracts for the simplified 80-atom training protocol."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.data_utils.pretrained_mace import Quadruplets
from src.training_methods.mace_objective import objective,cached_step,make_scheduler
from src.training_methods.mace_performance import encode_views
from src.analysis.liquid_structure import persistence_image


LOSS=dict(invariance=25.,variance=25.,covariance=1.,tda=1.,tda_start_epoch=6)


class ToyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(3,256)

    def build_geometry(self,x,m):return SimpleNamespace(batch_size=len(x),x=x.mean(1))
    def forward_from_geometry(self,g):return self.linear(g.x)
    def forward(self,x,m):return self.forward_from_geometry(self.build_geometry(x,m))


def test_uniform_epoch_uses_every_row_once_including_tail():
    data=Quadruplets.__new__(Quadruplets)
    data.pools={'train':[np.array([(m,i) for i in range(n)]) for m,n in enumerate((17,4,2))]}
    batches=list(data.epoch('train',8,np.random.default_rng(4)))
    assert [len(b) for b in batches]==[8,8,7]
    rows=np.concatenate(batches)
    assert len(set(map(tuple,rows)))==23
    assert np.bincount(rows[:,0]).tolist()==[17,4,2]
    assert data.epoch_steps(8)==3


@pytest.mark.parametrize('tda_start,total_epochs',[(6,12),(1,24)])
def test_tda_head_optimizer_and_schedule_follow_configured_activation(tda_start,total_epochs):
    torch.manual_seed(9)
    model=torch.nn.Module();model.encoder=ToyEncoder();model.tda=torch.nn.Linear(256,32)
    settings=dict(LOSS,tda_start_epoch=tda_start)
    cfg=dict(epochs=total_epochs,learning_rate=1e-4,head_learning_rate=1e-3,loss=settings,scheduler=dict(warmup_epochs=1,head_warmup_epochs=.5,min_lr=1e-6))
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=1e-4),dict(params=model.tda.parameters(),lr=1e-3)],weight_decay=.1)
    scheduler=make_scheduler(optimizer,cfg,2)
    original={k:v.clone() for k,v in model.tda.state_dict().items()}
    x=torch.randn(12,3,80,3);m=torch.zeros(12,dtype=torch.long);target=torch.randn(12,3,32);mask=torch.ones(12,dtype=torch.bool)
    for epoch in range(1,total_epochs+1):
        active=epoch>=tda_start;model.tda.requires_grad_(active)
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss,parts=objective(model,encode_views(model,x,m,8),target,mask,settings,epoch)
            loss.backward()
            assert ('tda_mse' in parts)==active
            if not active:
                assert optimizer.param_groups[1]['lr']==0
                assert all(p.grad is None and p not in optimizer.state for p in model.tda.parameters())
            else:assert optimizer.param_groups[1]['lr']>0
            optimizer.step();scheduler.step()
        if not active:
            for k,v in model.tda.state_dict().items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    assert any(not torch.equal(v,original[k]) for k,v in model.tda.state_dict().items())
    assert scheduler.last_epoch==total_epochs*2
    for group in optimizer.param_groups:assert group['lr']==pytest.approx(cfg['scheduler']['min_lr'])


def test_cached_gradient_matches_direct_before_and_after_tda_activation():
    torch.manual_seed(3)
    model=torch.nn.Module();model.encoder=ToyEncoder();model.tda=torch.nn.Linear(256,32)
    cfg=dict(loss=LOSS,microbatch_size=7)
    batch=(torch.randn(12,4,80,3),torch.randn(12,4,32),torch.randn(12,5),torch.arange(12)%3)
    mask=torch.ones(12,dtype=torch.bool);mask[:2]=False
    for epoch in (5,6):
        model.tda.requires_grad_(epoch>=6);model.zero_grad(set_to_none=True)
        value,parts=cached_step(model,batch,mask,cfg,epoch)
        gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
        assert abs(value-sum(v for k,v in parts.items() if k.startswith('loss_')))<1e-4
        model.zero_grad(set_to_none=True)
        z=encode_views(model,batch[0][:,:3],batch[3],7)
        loss,_=objective(model,z,batch[1][:,:3],mask,LOSS,epoch);loss.backward()
        for n,p in model.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p.grad,gradients[n],rtol=2e-4,atol=2e-5)


def test_tda_uses_outer_atoms_and_is_rigid_motion_invariant():
    rng=np.random.default_rng(17);x=rng.normal(size=(80,3))*2
    changed=x.copy();changed[-15:]*=.6
    assert np.linalg.norm(persistence_image(x)-persistence_image(changed))>1e-3
    rotation=np.linalg.qr(rng.normal(size=(3,3)))[0]
    np.testing.assert_allclose(persistence_image(x),persistence_image(x[rng.permutation(80)]@rotation+3),rtol=1e-5,atol=1e-6)
