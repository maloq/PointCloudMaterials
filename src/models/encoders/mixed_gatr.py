"""Shared snapshot GATr with explicitly grouped auxiliary-head normalization."""
import torch
from torch import nn

from .structural import StructuralGATr
from .structural_precision import FullPrecisionReadout
from .equivariant_bond import GATrBondOrder

MIXED_ARCHITECTURE_REVISION = 'gatr_v10_local_grouped_heads_snapshot'
GATR_BOND_REVISION = 'gatr_v11_local_grouped_heads_equivariant_bond_order'


class GroupBatchNorm(nn.Module):
    """Training moments within each domain; frozen training-reference moments at eval.

    No pooled-domain moving averages. Calibration fits the same population
    variance used by the forward pass. Domain affine parameters allow physical
    heads to express different material/potential means after centering.
    """
    def __init__(self,groups,width,affine=False,eps=1e-6):
        super().__init__();self.eps=eps;self.groups=groups
        self.register_buffer('running_mean',torch.zeros(groups,width))
        self.register_buffer('running_var',torch.ones(groups,width))
        self.register_buffer('calibrated',torch.zeros(groups,dtype=torch.bool))
        self.weight=nn.Parameter(torch.ones(groups,width)) if affine else None
        self.bias=nn.Parameter(torch.zeros(groups,width)) if affine else None

    def forward(self,x,groups):
        if groups.shape!=(len(x),) or bool(((groups<0)|(groups>=self.groups)).any()):
            raise ValueError('Invalid per-observation normalization group')
        with torch.autocast(x.device.type,enabled=False):
            x=x.float();result=torch.zeros_like(x)
            for group in range(self.groups):
                mask=groups==group;n=int(mask.sum())
                if not n:continue
                value=x[mask]
                if self.training:
                    if n<2:raise ValueError(f'Group {group} has fewer than two training observations')
                    mean=value.mean(0);var=value.var(0,unbiased=False)
                else:
                    if not bool(self.calibrated[group]):raise ValueError(f'Group {group} has no training-fitted head moments')
                    mean=self.running_mean[group];var=self.running_var[group]
                value=(value-mean)*torch.rsqrt(var+self.eps)
                if self.weight is not None:value=value*self.weight[group]+self.bias[group]
                result[mask]=value
            return result

    @torch.no_grad()
    def fit(self,x,groups):
        if self.training:raise ValueError('Fit head moments in evaluation mode')
        for group in range(self.groups):
            value=x[groups==group].double()
            if len(value)<2 or not torch.isfinite(value).all():
                raise ValueError(f'Invalid training calibration observations for group {group}')
            self.running_mean[group].copy_(value.mean(0).float())
            self.running_var[group].copy_(value.var(0,unbiased=False).float())
            self.calibrated[group]=True


class GroupReadout(nn.Module):
    def __init__(self,groups,outputs):
        super().__init__();self.norm=GroupBatchNorm(groups,128,affine=True)
        self.network=FullPrecisionReadout(nn.Linear(128,128),
            nn.LayerNorm(128,elementwise_affine=False),nn.SiLU(),nn.Linear(128,outputs))

    def forward(self,x,groups):
        return self.network(self.norm(x,groups))


class GroupProjector(nn.Module):
    def __init__(self,groups):
        super().__init__();self.input_norm=GroupBatchNorm(groups,128)
        self.input=FullPrecisionReadout(nn.Linear(128,256,bias=False))
        self.hidden_norm=GroupBatchNorm(groups,256,affine=True,eps=1e-5)
        self.output=FullPrecisionReadout(nn.ReLU(),nn.Linear(256,64))

    def forward(self,x,groups):
        return self.output(self.hidden_norm(self.input(self.input_norm(x,groups)),groups))


class MixedSnapshotGATr(nn.Module):
    def __init__(self,group_keys):
        super().__init__();self.group_keys=tuple(tuple(k) for k in group_keys)
        if not self.group_keys or any(k[2] for k in self.group_keys):
            raise ValueError('Mixed triplet GATr requires dynamic material/potential groups')
        self.encoder=StructuralGATr(history=False)
        self.projector=GroupProjector(len(group_keys))
        self.physical=GroupReadout(len(group_keys),85)
        self.tda=GroupReadout(len(group_keys),144)

    def heads(self,z,groups):
        return dict(q=self.projector(z,groups),physical=self.physical(z,groups),tda=self.tda(z,groups))

    @torch.no_grad()
    def calibrate(self,z,groups):
        if self.training:raise ValueError('Calibrate in evaluation mode')
        self.physical.norm.fit(z,groups);self.tda.norm.fit(z,groups)
        self.projector.input_norm.fit(z,groups)
        hidden=self.projector.input(self.projector.input_norm(z,groups))
        self.projector.hidden_norm.fit(hidden,groups)


class MixedBondGATr(MixedSnapshotGATr):
    def __init__(self,group_keys):
        super().__init__(group_keys)
        self.bond_order=GATrBondOrder(self.encoder.mv_channels)
