"""Gradient replay across separate view groups with partial final chunks."""
from types import SimpleNamespace
import pytest
import torch
from src.training_methods.mace_performance import encode_clouds,replay_chunks
from src.models.encoders.pretrained_mace import dense_matmul_precision


class SmallEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.linear=torch.nn.Linear(3,4)

    def build_geometry(self,x,m):
        return SimpleNamespace(batch_size=len(x),features=x.mean(1)+m[:,None])

    def forward_from_geometry(self,g):
        return self.linear(g.features)

    def forward(self,x,m):
        return self.forward_from_geometry(self.build_geometry(x,m))


def test_replay_preserves_partial_view_group_boundaries_and_gradients():
    torch.manual_seed(7);model=SimpleNamespace(encoder=SmallEncoder())
    x=torch.randn(39,80,3);m=torch.arange(39)%3;derivative=torch.randn(39,4)
    expected=model.encoder(x,m);expected.backward(derivative)
    gradients=[p.grad.clone() for p in model.encoder.parameters()];model.encoder.zero_grad()
    cache=[]
    with torch.no_grad():
        a=encode_clouds(model,x[:32],m[:32],12,cache)
        b=encode_clouds(model,x[32:],m[32:],12,cache)
    torch.testing.assert_close(torch.cat((a,b)),expected)
    assert [g.batch_size for g in cache]==[12,12,8,7]
    for actual,sl in replay_chunks(model,x,m,12,cache):actual.backward(derivative[sl])
    for parameter,gradient in zip(model.encoder.parameters(),gradients):torch.testing.assert_close(parameter.grad,gradient)


def test_dense_precision_restores_setting_after_error():
    previous=torch.get_float32_matmul_precision()
    with pytest.raises(RuntimeError,match='deliberate'):
        with dense_matmul_precision('high'):
            assert torch.get_float32_matmul_precision()=='high'
            raise RuntimeError('deliberate')
    assert torch.get_float32_matmul_precision()==previous

