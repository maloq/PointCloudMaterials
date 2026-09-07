"""Scientific invariants of the descriptor-free temporal campaign."""
from types import SimpleNamespace
import torch
from torch import nn
from src.training_methods.temporal_campaign import local_graph,objective,spread_loss


def test_graph_excludes_identity_even_when_cdist_diagonal_rounds_positive():
    torch.manual_seed(14);x=torch.randn(2,193,3)*4;x[:,0]=0
    assert (torch.cdist(x,x).diagonal(dim1=-2,dim2=-1)>0).any()
    edges,counts=local_graph(x,4.)
    for i,count in enumerate(counts):assert (edges[i,:count,0]!=edges[i,:count,1]).all()


def test_species_separation_does_not_satisfy_anti_collapse():
    material=torch.arange(96)%3
    z=torch.eye(3)[material].repeat(1,8)*10
    torch.testing.assert_close(spread_loss(z,material),torch.tensor(24.75))


class ToyRepresentation(nn.Module):
    def __init__(self):
        super().__init__();self.map=nn.Linear(3,8)

    def forward(self,x,material):
        return self.map(x.mean(1))


def test_motion_supervision_reaches_encoder_only_in_explicit_ablation():
    torch.manual_seed(14)
    model=SimpleNamespace(representation=ToyRepresentation(),forecast=nn.Linear(13,8),motion=nn.Linear(13,2))
    teacher=ToyRepresentation().requires_grad_(False)
    batch=[torch.randn(96,193,3),torch.randn(96,193,3),torch.arange(96)%3,torch.randn(96,5),torch.randn(96,2)]
    cfg=dict(jitter_std_A=.01,jitter_clip_A=.03)
    _,parts=objective(model,teacher,batch,dict(mode='predictive'),cfg)
    gradient=torch.autograd.grad(parts['motion'],model.representation.map.weight,allow_unused=True)[0]
    assert gradient is None
    _,parts=objective(model,teacher,batch,dict(mode='motion'),cfg)
    gradient=torch.autograd.grad(parts['motion'],model.representation.map.weight)[0]
    assert gradient.abs().sum()>0
