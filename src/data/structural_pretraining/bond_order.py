"""Central 12-neighbor q4m/q6m in the real e3nn component-normalized basis.

The neighbor definition matches analysis.liquid_structure.bond_order. Real
components are sqrt(4*pi) times an orthogonal conversion of its complex,
integral-normalized harmonics. Thus Q_l = sqrt(mean_m(q_lm**2)). No material,
orientation or validation-fitted standardization is applied to these tensors.
"""
import torch
from e3nn import o3

BOND_IRREPS=o3.Irreps('1x4e + 1x6e')


@torch.no_grad()
def bond_order_targets(vectors):
    if vectors.ndim!=3 or vectors.shape[1:]!=(12,3):
        raise ValueError(f'Expected central bond vectors [B,12,3], got {vectors.shape}')
    if not torch.isfinite(vectors).all() or bool((vectors.square().sum(-1)<=0).any()):
        raise ValueError('Bond order requires finite, nonzero bond vectors')
    with torch.autocast(vectors.device.type,enabled=False):
        return o3.spherical_harmonics([4,6],vectors.float(),normalize=True,normalization='component').mean(1)


def bond_order_errors(prediction,target):
    """[B,2] rotation-invariant component MSE, scaled by 12 neighbors.

    Isotropic independent random bonds give the zero predictor expected loss 1.
    This fixed scale avoids breaking equivariance by whitening individual m's.
    """
    if prediction.shape!=target.shape or prediction.shape[-1]!=22:
        raise ValueError('Bond-order predictions and targets must have shape [B,22]')
    with torch.autocast(prediction.device.type,enabled=False):
        error=(prediction.float()-target.float()).square()
        return 12*torch.stack((error[:,:9].mean(-1),error[:,9:].mean(-1)),-1)


def bond_order_magnitudes(q):
    return torch.stack((q[...,:9].square().mean(-1).sqrt(),q[...,9:].square().mean(-1).sqrt()),-1)
