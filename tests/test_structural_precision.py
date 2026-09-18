"""The enlarged encoders protect geometry rather than merely casting exports."""
import copy

import numpy as np
import pytest
import torch

from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.gatr_block import GATrBlock
from gatr.layers.mlp.config import MLPConfig
from gatr.interface import embed_point

from src.models.encoders.structural import StructuralModel
from src.models.encoders.structural_precision import (FullPrecision, FloatOutput, SplitEquiLinear,
    CompensatedScalarLinear, TensorGeometricAttention, protect_gatr)
from src.models.encoders.compensated_bf16 import compensated_mm
from src.data.structural_pretraining.batches import collate, move
from test_structural_pretraining_model import sample


def test_gatr_precision_adapter_matches_upstream_in_fp32():
    torch.manual_seed(51)
    reference = GATrBlock(8, 32, SelfAttentionConfig(num_heads=4, pos_encoding=False), MLPConfig())
    adapted = protect_gatr(copy.deepcopy(reference))
    mv = embed_point(torch.randn(2, 9, 3)).unsqueeze(-2).expand(-1, -1, 8, -1).clone()
    scalar = torch.randn(2, 9, 32)
    kwargs = dict(scalars=scalar, reference_mv=embed_point(torch.zeros(3)))
    a, b = reference(mv, **kwargs), adapted(mv, **kwargs)
    for expected, actual in zip(a, b, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    sum(x.square().mean() for x in a).backward()
    sum(x.square().mean() for x in b).backward()
    for p, q in zip(reference.parameters(), adapted.parameters(), strict=True):
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-5, atol=2e-6)


def test_tensor_attention_matches_upstream_masked_multiquery_and_gradients():
    from gatr.layers.attention.attention import GeometricAttention
    torch.manual_seed(891)
    upstream=GeometricAttention(SelfAttentionConfig(in_mv_channels=4,num_heads=4))
    adapted=TensorGeometricAttention(copy.deepcopy(upstream))
    shapes=[(2,4,7,2,16),(2,1,9,2,16),(2,1,9,3,16),
            (2,4,7,12),(2,1,9,12),(2,1,9,11)]
    inputs=[torch.randn(s,requires_grad=True) for s in shapes]
    copies=[x.detach().clone().requires_grad_() for x in inputs]
    mask=torch.randn(2,1,7,9);mask[...,-2:]=-torch.inf
    expected=upstream(*inputs,attention_mask=mask)
    actual=adapted(*copies,attention_mask=mask)
    for x,y in zip(actual,expected,strict=True):torch.testing.assert_close(x,y,atol=2e-6,rtol=2e-6)
    sum(x.square().mean() for x in expected).backward()
    sum(x.square().mean() for x in actual).backward()
    for x,y in zip(inputs,copies,strict=True):torch.testing.assert_close(x.grad,y.grad,atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(upstream.log_weights.grad,adapted.module.log_weights.grad,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('architecture,old,new', [('mace',73491,147139), ('gatr',399504,803216)])
def test_snapshot_parameter_budget(architecture, old, new):
    model = StructuralModel(architecture)
    assert sum(p.numel() for p in model.encoder.parameters()) == new
    assert abs(new / old - 2) < .02
    assert not hasattr(model.encoder, 'temporal')


def test_snapshot_rejects_history_instead_of_silently_discarding_it():
    model = StructuralModel('gatr')
    with pytest.raises(ValueError, match='history=True'):
        model.encoder(collate([sample(3, t=3)], 'gatr'))
    with pytest.raises(ValueError, match='snapshot-only'):
        StructuralModel('mace', history=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FP32-output BF16 GEMM requires CUDA')
def test_compensated_scalar_accuracy_and_gradients(monkeypatch):
    torch.manual_seed(121)
    reference = torch.nn.Linear(192,256,bias=False).cuda()
    compensated = CompensatedScalarLinear(copy.deepcopy(reference))
    a = torch.randn(3,16,384,device='cuda')[...,::2].requires_grad_()
    b = a.detach().clone().requires_grad_()
    target = torch.randn(3,16,256,device='cuda')
    expected = reference(a)
    (expected*target).sum().backward()
    observed = []
    def tracked_mm(x, y):
        observed.append((x.dtype,y.dtype))
        return compensated_mm(x,y)
    monkeypatch.setattr('src.models.encoders.structural_precision.compensated_mm',tracked_mm)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual = compensated(b)
    (actual*target).sum().backward()
    assert len(observed)==3
    assert set(observed)=={(torch.float32,torch.float32)}
    for x,y in [(actual,expected),(b.grad,a.grad),(compensated.linear.weight.grad,reference.weight.grad)]:
        assert (x-y).norm()/y.norm()<5e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA Triton kernel')
@pytest.mark.parametrize('m,k,n', [(31,67,93),(192,4097,192)])
def test_fused_compensated_noncontiguous_and_split_k(m,k,n):
    torch.manual_seed(88)
    torch.backends.cuda.matmul.allow_tf32=False
    a=torch.randn(k,m,device='cuda').T
    b=torch.randn(n,k,device='cuda').T
    expected=a@b
    actual=compensated_mm(a,b)
    assert (actual-expected).norm()/expected.norm()<5e-5
    torch.testing.assert_close(compensated_mm(a,b),actual,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA compiler validation')
@pytest.mark.parametrize('architecture', ['mace','gatr'])
def test_dynamic_compilation_preserves_checkpoint_gradients_and_rotation(architecture):
    from src.training_methods.shared_pretraining.compilation import compile_encoder
    torch.manual_seed(112)
    original=StructuralModel(architecture).cuda().eval()
    # Fresh construction matches training; deepcopy is unsupported by cuEq's
    # generated FX contraction constants and their bound dtype-conversion hooks.
    compiled=StructuralModel(architecture).cuda().eval()
    compiled.load_state_dict(original.state_dict())
    example=move(collate([sample(1)],architecture),'cuda')
    compile_encoder(compiled.encoder,example,'bf16')
    assert list(original.state_dict())==list(compiled.state_dict())
    rotation=torch.linalg.qr(torch.randn(3,3,device='cuda',dtype=torch.float64)).Q
    for n in (12,19):
        batch=move(collate([sample(i,n=n+i) for i in range(4)],architecture),'cuda')
        if architecture=='gatr':
            # True dynamic tracing must not turn atom counts into constants
            # inside opt_einsum, even if two sizes fit the recompile cache.
            for key,dim in [('positions',2),('weights',2),('species',1)]:
                torch._dynamo.mark_dynamic(batch[key],dim)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            a=original.encoder(batch)
            b=compiled.encoder(batch)
        torch.testing.assert_close(b,a,atol=1e-5,rtol=1e-4)
        target=torch.randn_like(a)
        (a*target).mean().backward();(b*target).mean().backward()
        expected_grad=[];actual_grad=[]
        for (name,p),(_,q) in zip(original.encoder.named_parameters(),compiled.encoder.named_parameters(),strict=True):
            if p.grad is None:
                assert q.grad is None or torch.count_nonzero(q.grad)==0,name
            else:
                assert q.grad is not None and torch.isfinite(q.grad).all(),name
                expected_grad.append(p.grad.flatten());actual_grad.append(q.grad.flatten())
        reference_gradient=torch.cat(expected_grad);actual_gradient=torch.cat(actual_grad)
        assert (actual_gradient-reference_gradient).norm()/reference_gradient.norm()<.002
        rotated=dict(batch)
        for key in ('positions','packed_positions'):
            if key in rotated:rotated[key]=(rotated[key].double()@rotation).float()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            c=compiled.encoder(rotated)
        signal=(b.double()-b.double().mean(0)).square().mean().sqrt()
        assert (b.double()-c.double()).square().mean().sqrt()/signal<.01
        original.zero_grad(set_to_none=True);compiled.zero_grad(set_to_none=True)
    restored=StructuralModel(architecture).cuda()
    restored.load_state_dict(compiled.state_dict(),strict=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Check actual CUDA BF16 kernels')
@pytest.mark.parametrize('architecture,history', [('mace',False), ('gatr',False), ('gatr',True)])
def test_amp_boundaries_rotation_and_gradients(architecture, history):
    torch.manual_seed(67)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = StructuralModel(architecture, history=history).cuda().eval()
    samples = [sample(i, t=3 if history else 1, n=12+i) for i in range(8)]
    batch = move(collate(samples, architecture), 'cuda')
    scalar_dtypes, geometry_dtypes, handles = [], [], []

    def scalar_hook(module, args, output):
        scalar_dtypes.append(output.dtype)

    def geometry_hook(module, args):
        geometry_dtypes.append((args[0].dtype, torch.is_autocast_enabled('cuda')))

    for module in model.encoder.modules():
        if isinstance(module, SplitEquiLinear) and module.linear.s2s is not None:
            handles.append(module.linear.s2s.register_forward_hook(scalar_hook))
        if isinstance(module, FloatOutput):
            handles.append(module.module.register_forward_hook(scalar_hook))
        if isinstance(module, FullPrecision):
            handles.append(module.module.register_forward_pre_hook(geometry_hook))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        z = model.encoder(batch)
        heads = model.heads(z)
    for handle in handles:
        handle.remove()
    assert z.dtype == torch.float32
    assert all(v.dtype == torch.float32 for v in heads.values())
    assert scalar_dtypes and set(scalar_dtypes) == {torch.bfloat16 if architecture=='mace' else torch.float32}
    assert geometry_dtypes and set(geometry_dtypes) == {(torch.float32, False)}
    sum(v.square().mean() for v in heads.values()).backward()
    gradients = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.square().sum()) for g in gradients) > 0

    reference = dict(z=z.detach(), **{k:v.detach() for k,v in heads.items()})
    rng = np.random.default_rng(932)
    for _ in range(3):
        rotation = np.linalg.qr(rng.normal(size=(3,3)))[0]
        if np.linalg.det(rotation) < 0:
            rotation[:,0] *= -1
        current = dict(batch)
        for key in ('positions', 'packed_positions'):
            if key in current:
                current[key] = (current[key].double() @ torch.tensor(rotation, device='cuda')).float()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            rotated_z = model.encoder(current)
            rotated = dict(z=rotated_z, **model.heads(rotated_z))
        for name, original in reference.items():
            error = (rotated[name].double()-original.double()).square().mean().sqrt()
            signal = (original.double()-original.double().mean(0)).square().mean().sqrt()
            assert signal > 0
            assert error / signal < .01, (name, float(error), float(signal))
