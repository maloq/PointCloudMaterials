"""Mixed-domain MACE and a tensor-only, training-time q4m/q6m readout."""
import torch
from torch import nn
from e3nn import o3

from .structural import StructuralMACE
from .mixed_gatr import GroupProjector,GroupReadout,MixedSnapshotGATr

MACE_BOND_REVISION='mace_v9_grouped_heads_equivariant_bond_order'


class EquivariantBondOrder(nn.Module):
    """Learn l=4 and l=6 via tensor products of MACE's learned l=2 channels.

    This head has no coordinates, spherical-harmonic labels or invariant z as
    inputs. It therefore cannot copy bond targets through a geometric shortcut.
    FP32 contractions preserve O(3) covariance under the BF16 training context.
    """
    def __init__(self,channels=32,width=4):
        super().__init__();self.channels=channels
        self.project=o3.Linear(f'{channels}x2e',f'{width}x2e')
        self.order4=o3.FullyConnectedTensorProduct(f'{width}x2e',f'{width}x2e',f'{width}x4e')
        self.order6=o3.FullyConnectedTensorProduct(f'{width}x4e',f'{width}x2e',f'{width}x6e')
        self.out4=o3.Linear(f'{width}x4e','1x4e')
        self.out6=o3.Linear(f'{width}x6e','1x6e')

    def forward(self,atom_features):
        with torch.autocast(atom_features.device.type,enabled=False):
            tensor=atom_features.float()[:,4*self.channels:]
            # One invariant scale for the entire irrep, never per-m whitening.
            tensor=tensor/torch.sqrt(tensor.square().mean(-1,keepdim=True)+1e-4)
            q2=self.project(tensor);q4=self.order4(q2,q2);q6=self.order6(q4,q2)
            return torch.cat((self.out4(q4),self.out6(q6)),-1)


class MixedSnapshotMACE(nn.Module):
    def __init__(self,group_keys,backend='cueq'):
        super().__init__();self.group_keys=tuple(tuple(k) for k in group_keys)
        if not self.group_keys or any(k[2] for k in self.group_keys):
            raise ValueError('Mixed MACE requires dynamic material/potential groups')
        self.encoder=StructuralMACE(backend=backend)
        self.projector=GroupProjector(len(group_keys))
        self.physical=GroupReadout(len(group_keys),85)
        self.tda=GroupReadout(len(group_keys),144)
        self.bond_order=EquivariantBondOrder(self.encoder.channels)

    heads=MixedSnapshotGATr.heads
    calibrate=MixedSnapshotGATr.calibrate


def training_encode(model,batch):
    """Cache tensor features only in the explicitly selected MACE protocol."""
    if isinstance(model,MixedSnapshotMACE):
        return model.encoder(batch,return_equivariant=True)
    return model.encoder(batch)
