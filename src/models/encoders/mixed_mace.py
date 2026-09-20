"""Mixed-domain MACE and a tensor-only, training-time q4m/q6m readout."""
import torch
from torch import nn

from .structural import StructuralMACE
from .equivariant_bond import EquivariantBondOrder
from .mixed_gatr import GroupProjector,GroupReadout,MixedSnapshotGATr,MixedBondGATr

MACE_BOND_REVISION='mace_v10_local_grouped_heads_equivariant_bond_order'


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
    """Cache tensor features for either explicit bond-supervised protocol."""
    if isinstance(model,(MixedSnapshotMACE,MixedBondGATr)):
        return model.encoder(batch,return_equivariant=True)
    return model.encoder(batch)
