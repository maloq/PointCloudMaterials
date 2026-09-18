"""Precision boundaries for the pinned MACE and GATr structural backbones.

Autocast may accelerate invariant scalar matrix products. Geometric operations
and residual streams remain FP32, including GATr's joint scalar/MV attention.
The adapters use GATr's original parameters, initialization and primitives.
"""
import torch
from torch import nn
from torch.nn import functional as F

from gatr.interface import embed_scalar
from gatr.layers.attention.attention import GeometricAttention
from gatr.layers.layer_norm import EquiLayerNorm
from gatr.layers.linear import EquiLinear
from gatr.layers.mlp.geometric_bilinears import GeometricBilinear
from gatr.layers.mlp.nonlinearities import ScalarGatedNonlinearity
from gatr.primitives.bilinear import geometric_product
from gatr.primitives.dual import equivariant_join
from gatr.primitives.linear import equi_linear
from gatr.primitives.attention import (
    _TRIVECTOR_IDX, _INNER_PRODUCT_WO_TRI_IDX, _build_dist_basis, _build_dist_vec,
)
from .compensated_bf16 import compensated_mm


def bf16_three_product(a, b):
    """Approximate an FP32 matrix product with three BF16 tensor-core products.

    a ~= ah + al and b ~= bh + bl; omit only the small al @ bl term.
    Crucially, each GEMM returns FP32, rather than rounding its output to BF16.
    See Henry, Tang & Heinecke (2019), https://arxiv.org/abs/1904.06376.
    """
    ah, bh = a.bfloat16(), b.bfloat16()
    al, bl = (a-ah.float()).bfloat16(), (b-bh.float()).bfloat16()
    correction = torch.mm(ah, bl, out_dtype=torch.float32) + torch.mm(al, bh, out_dtype=torch.float32)
    return torch.mm(ah, bh, out_dtype=torch.float32) + correction


class _CompensatedMatmul(torch.autograd.Function):
    """First-order AMP matmul gradients using the same compensated arithmetic.

    torch.mm(out_dtype=FP32) has no autograd rule in the pinned PyTorch 2.14.
    Like ordinary AMP, these are approximate gradients of the FP32 linear map,
    not derivatives of discontinuous floating-point rounding operations.
    """
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return compensated_mm(x.reshape(-1, x.shape[-1]), weight.T).reshape(*x.shape[:-1], weight.shape[0])

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, gradient):
        x, weight = ctx.saved_tensors
        with torch.autocast(x.device.type, enabled=False):
            g = gradient.float().reshape(-1, weight.shape[0])
            dx = compensated_mm(g, weight).reshape_as(x) if ctx.needs_input_grad[0] else None
            dw = compensated_mm(g.T, x.reshape(-1, x.shape[-1])) if ctx.needs_input_grad[1] else None
        return dx, dw


class CompensatedScalarLinear(nn.Module):
    """BF16 tensor-core scalar map with FP32 accumulation and residual correction."""
    def __init__(self, linear):
        super().__init__()
        if linear.bias is not None:
            raise ValueError('GATr scalar-to-scalar maps must be bias-free')
        self.linear = linear

    def forward(self, x):
        if not torch.is_autocast_enabled(x.device.type):
            return self.linear(x)
        if x.device.type != 'cuda' or torch.get_autocast_dtype('cuda') != torch.bfloat16:
            raise ValueError('Compensated GATr autocast requires CUDA BF16; disable autocast for FP32')
        with torch.autocast('cuda', enabled=False):
            return _CompensatedMatmul.apply(x.float(), self.linear.weight)


class FullPrecision(nn.Module):
    """Protect a module whose first input is a floating-point feature tensor.

    Other floating inputs at these boundaries are already FP32: protected
    geometric projections and FloatOutput explicitly establish that contract.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        with torch.autocast(x.device.type, enabled=False):
            return self.module(x.float(), *args, **kwargs)


class FloatOutput(nn.Module):
    """Allow scalar autocast, then restore FP32 before geometric operations."""
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x).float()


class TensorGeometricAttention(FullPrecision):
    """Pinned GATr attention for tensor masks, with no disabled SDPA wrapper.

    The upstream wrapper deliberately breaks Dynamo for its xFormers path.
    Our structural models use tensor masks and [batch, heads, tokens, ...]
    tensors exclusively. Keep its distance features and padding correction,
    and broadcast the single KV head directly for native PyTorch SDPA.
    Parameter paths remain identical to FullPrecision(GeometricAttention).
    """
    def forward(self, q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
        if q_mv.ndim != 5 or k_mv.ndim != 5 or v_mv.ndim != 5:
            raise ValueError('Structural attention requires [batch, heads, tokens, channels, 16]')
        if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
            raise TypeError('Structural attention supports tensor masks only')
        with torch.autocast(q_mv.device.type, enabled=False):
            q_mv, k_mv, v_mv = q_mv.float(), k_mv.float(), v_mv.float()
            q_s, k_s, v_s = q_s.float(), k_s.float(), v_s.float()
            bq, bk = _build_dist_basis(q_mv.device, q_mv.dtype)
            qd = _build_dist_vec(q_mv[..., _TRIVECTOR_IDX], bq, self.module.normalizer)
            kd = _build_dist_vec(k_mv[..., _TRIVECTOR_IDX], bk, self.module.normalizer)
            qd = qd * self.module.log_weights.exp()[..., None]
            q = torch.cat((q_mv[..., _INNER_PRODUCT_WO_TRI_IDX].flatten(-2), qd.flatten(-2), q_s), -1)
            k = torch.cat((k_mv[..., _INNER_PRODUCT_WO_TRI_IDX].flatten(-2), kd.flatten(-2), k_s), -1)
            v = torch.cat((v_mv.flatten(-2), v_s), -1)
            qk_width = q.shape[-1]
            width = ((max(qk_width, v.shape[-1]) + 7) // 8) * 8
            q = F.pad(q, (0, width - qk_width))
            k = F.pad(k, (0, width - qk_width)) * (width / qk_width) ** .5
            v = F.pad(v, (0, width - v.shape[-1]))
            # Explicit expansion preserves the pinned implementation's SDPA
            # dispatch as well as its multi-query attention equations.
            k = k.expand(q.shape[0], q.shape[1], k.shape[-2], width)
            v = v.expand(q.shape[0], q.shape[1], v.shape[-2], width)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
            mv_width = v_mv.shape[-2] * 16
            return (out[..., :mv_width].reshape(*out.shape[:-1], v_mv.shape[-2], 16),
                    out[..., mv_width:mv_width + v_s.shape[-1]])


class FullPrecisionReadout(nn.Sequential):
    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):
            return super().forward(x.float())


class SplitEquiLinear(nn.Module):
    """GATr EquiLinear with autocast only on the scalar-to-scalar product.

    Mirrors the pinned Qualcomm GATr EquiLinear.forward using its own primitives;
    scalar/MV mixing and MV contractions keep their original FP32 resolution.
    """
    def __init__(self, linear):
        super().__init__()
        if linear.s2s is not None:
            linear.s2s = CompensatedScalarLinear(linear.s2s)
        self.linear = linear

    def forward(self, multivectors, scalars=None):
        linear = self.linear
        with torch.autocast(multivectors.device.type, enabled=False):
            mv = multivectors.float()
            scalar = None if scalars is None else scalars.float()
            outputs_mv = equi_linear(mv, linear.weight)
            if linear.bias is not None:
                outputs_mv = outputs_mv + embed_scalar(linear.bias)
            if linear.s2mvs is not None and scalar is not None:
                outputs_mv = outputs_mv + embed_scalar(linear.s2mvs(scalar).unsqueeze(-1))
            # Slicing grade zero from Inductor's MV layout can give a column
            # stride proportional to B*N. Materialize this small scalar view
            # before Linear so AOT saved-tensor layouts remain shape-polymorphic.
            outputs_s = None if linear.mvs2s is None else linear.mvs2s(mv[..., 0].contiguous())
        if outputs_s is not None and linear.s2s is not None and scalar is not None:
            # The surrounding training autocast determines FP32 versus BF16.
            # Add the result to the invariant stream in FP32, never in BF16.
            outputs_s = outputs_s + linear.s2s(scalar).float()
        return outputs_mv, outputs_s


class SplitGeometricBilinear(nn.Module):
    """Keep products/joins FP32 while its output's scalar linear map uses AMP."""
    def __init__(self, bilinear):
        super().__init__()
        self.bilinear = bilinear

    def forward(self, multivectors, reference_mv, scalars):
        layer = self.bilinear
        left, _ = layer.linear_left(multivectors, scalars=scalars)
        right, _ = layer.linear_right(multivectors, scalars=scalars)
        with torch.autocast(multivectors.device.type, enabled=False):
            product = geometric_product(left, right)
        left, _ = layer.linear_join_left(multivectors, scalars=scalars)
        right, _ = layer.linear_join_right(multivectors, scalars=scalars)
        with torch.autocast(multivectors.device.type, enabled=False):
            joined = equivariant_join(left, right, reference_mv)
        return layer.linear_out(torch.cat((product, joined), dim=-2), scalars=scalars)


def protect_gatr(module):
    """Install precision boundaries in newly constructed upstream GATr modules.

    Does not recreate parameters, consume RNG, change equations, or patch the
    installed library. This is specific to our pinned GATr implementation.
    """
    for name, child in list(module.named_children()):
        setattr(module, name, protect_gatr(child))
    if isinstance(module, EquiLinear):
        return SplitEquiLinear(module)
    if isinstance(module, GeometricBilinear):
        return SplitGeometricBilinear(module)
    if isinstance(module, GeometricAttention):
        return TensorGeometricAttention(module)
    if isinstance(module, (EquiLayerNorm, ScalarGatedNonlinearity)):
        return FullPrecision(module)
    return module
