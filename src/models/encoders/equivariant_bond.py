"""Even-rank bond readouts from learned equivariant atom features only."""
import torch
from torch import nn
from e3nn import o3


class QuadrupoleBondOrder(nn.Module):
    def __init__(self,channels,width=4):
        super().__init__();self.channels=channels
        self.project=o3.Linear(f'{channels}x2e',f'{width}x2e')
        self.order4=o3.FullyConnectedTensorProduct(f'{width}x2e',f'{width}x2e',f'{width}x4e')
        self.order6=o3.FullyConnectedTensorProduct(f'{width}x4e',f'{width}x2e',f'{width}x6e')
        self.out4=o3.Linear(f'{width}x4e','1x4e')
        self.out6=o3.Linear(f'{width}x6e','1x6e')

    def forward(self,tensor):
        with torch.autocast(tensor.device.type,enabled=False):
            tensor=tensor.float()
            tensor=tensor/torch.sqrt(tensor.square().mean(-1,keepdim=True)+1e-4)
            q2=self.project(tensor);q4=self.order4(q2,q2);q6=self.order6(q4,q2)
            return torch.cat((self.out4(q4),self.out6(q6)),-1)


class EquivariantBondOrder(QuadrupoleBondOrder):
    """MACE: retain the learned l=2 block of its 0e+1o+2e center features."""
    def __init__(self,channels=32,width=4):
        super().__init__(channels,width)

    def forward(self,atom_features):
        return super().forward(atom_features[:,4*self.channels:])


def gatr_bond_features(multivectors,weights,harmonics):
    """Form even-rank atom tensors before pooling, retaining aligned local order.

    PGA plane normals, ideal/axial bivectors and point numerators each transform
    as polar or axial 3D vectors. Even harmonics are formed separately before
    any mixing; polar and axial vectors are never linearly mixed. No
    division by homogeneous point coordinates, raw coordinates, invariant z or
    target descriptors is used. Unlike taking powers of a pooled vector, this
    retains nonzero q4/q6 information in inversion-symmetric neighborhoods.
    """
    with torch.autocast(multivectors.device.type,enabled=False):
        mv=multivectors.float()
        vectors=torch.stack((mv[...,2:5],mv[...,5:8],
            torch.stack((-mv[...,10],mv[...,9],-mv[...,8]),-1),
            torch.stack((-mv[...,13],mv[...,12],-mv[...,11]),-1)),-2)
        vectors=vectors*torch.rsqrt(vectors.square().sum(-1,keepdim=True)+1e-4)
        tensors=harmonics(vectors)
        w=weights.float()
        pooled=(tensors*w[...,None,None,None]).sum(1)/w.sum(1)[:,None,None,None]
        return torch.cat((pooled[...,:9].flatten(1),pooled[...,9:].flatten(1)),-1)


class GATrBondOrder(nn.Module):
    """Linear equivariant readout of pooled l=4/l=6 learned atom tensors."""
    def __init__(self,mv_channels=8):
        super().__init__()
        self.output=o3.Linear(f'{4*mv_channels}x4e + {4*mv_channels}x6e','1x4e + 1x6e')

    def forward(self,atom_features):
        with torch.autocast(atom_features.device.type,enabled=False):
            return self.output(atom_features.float())
