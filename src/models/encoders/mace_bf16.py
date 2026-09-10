"""Selective BF16 radial matrices with FP32 activations and master weights."""
import torch
from torch import nn


def bf16_mm(a,b):
    return torch.mm(a.to(torch.bfloat16),b.to(torch.bfloat16),out_dtype=torch.float32)


class BF16MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b):
        ctx.save_for_backward(a,b)
        return bf16_mm(a,b)

    @staticmethod
    def backward(ctx,gradient):
        a,b=ctx.saved_tensors
        return bf16_mm(gradient,b.T),bf16_mm(a.T,gradient)


def compensated_mm(a,b):
    """Three BF16 products, each accumulated/output in FP32; omit low*low."""
    ah=a.to(torch.bfloat16);bh=b.to(torch.bfloat16)
    al=(a-ah.float()).to(torch.bfloat16);bl=(b-bh.float()).to(torch.bfloat16)
    return (torch.mm(ah,bh,out_dtype=torch.float32)
            +torch.mm(ah,bl,out_dtype=torch.float32)
            +torch.mm(al,bh,out_dtype=torch.float32))


class CompensatedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx,a,b):
        ctx.save_for_backward(a,b)
        return compensated_mm(a,b)

    @staticmethod
    def backward(ctx,gradient):
        a,b=ctx.saved_tensors
        return compensated_mm(gradient,b.T),compensated_mm(a.T,gradient)


class BF16RadialLayer(nn.Module):
    """Retain the repository checkpoint's e3nn layer normalization and keys."""
    def __init__(self,source,compensated):
        super().__init__()
        self.weight=source.weight;self.act=source.act
        self.h_in=source.h_in;self.var_in=source.var_in;self.var_out=source.var_out
        self.compensated=compensated

    def forward(self,x):
        denominator=(self.h_in*self.var_in/(1 if self.act is not None else self.var_out))**.5
        weight=self.weight/denominator
        if self.compensated:
            x=CompensatedMM.apply(x,weight)
        else:
            x=BF16MM.apply(x,weight)
        if self.act is not None:x=self.act(x)*self.var_out**.5
        return x
