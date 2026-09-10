"""Numerical qualification of compensated forward and backward matrices."""
import pytest
import torch
from src.models.encoders.mace_bf16 import CompensatedMM,BF16MM


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA BF16 matrix outputs require a GPU')
@pytest.mark.parametrize('compiled',[False,True])
def test_compensation_preserves_forward_and_parameter_gradient_precision(compiled):
    torch.manual_seed(31)
    a=torch.randn(513,64,device='cuda',requires_grad=True)
    b=torch.randn(64,128,device='cuda',requires_grad=True)
    derivative=torch.randn(513,128,device='cuda')
    reference=a@b;expected=torch.autograd.grad(reference,(a,b),derivative)
    fn=torch.compile(CompensatedMM.apply,fullgraph=True,options={'emulate_precision_casts':True}) if compiled else CompensatedMM.apply
    naive=BF16MM.apply(a,b);actual=fn(a,b)
    assert float(((actual-reference).norm()/reference.norm()).detach())<2e-5
    assert float((actual-reference).norm().detach())<.02*float((naive-reference).norm().detach())
    for gradient,target in zip(torch.autograd.grad(actual,(a,b),derivative),expected):
        assert float((gradient-target).norm()/target.norm())<2e-5
    assert actual.dtype==torch.float32 and a.dtype==b.dtype==torch.float32
