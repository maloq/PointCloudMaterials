"""MLIP-initialized MACE: full 80-atom graph and mean-pooled scalar features."""
import torch
from torch import nn
from mace.cli.convert_e3nn_cueq import run as convert_cueq
from contextlib import contextmanager
from dataclasses import dataclass
from .base import Encoder
from .registry import register_encoder


@contextmanager
def dense_matmul_precision(precision):
    """Limit faster matrix arithmetic to learned features and their backward pass."""
    previous=torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(precision)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@dataclass
class MACEGeometry:
    batch_size: int
    points: int
    attrs: torch.Tensor
    edges: torch.Tensor
    angular: torch.Tensor
    radial: torch.Tensor
    cutoff: torch.Tensor | None


class PretrainedMACEEncoder(nn.Module):
    """Pool both pretrained scalar blocks over all atoms; retain native 5 Å edges."""
    def __init__(self,checkpoint,accelerated=True,performance=None):
        super().__init__()
        backbone=torch.load(checkpoint,map_location='cpu',weights_only=False).float()
        assert float(backbone.r_max)==5. and int(backbone.num_interactions)==2
        assert str(backbone.products[0].linear.irreps_out)=='128x0e'
        self.backbone=convert_cueq(backbone,device='cuda',layout='ir_mul') if accelerated else backbone
        self.backbone.readouts.requires_grad_(False)
        self.register_buffer('element_indices',torch.tensor([(self.backbone.atomic_numbers==a).nonzero().item() for a in (13,12,73)],dtype=torch.long))
        self.register_buffer('feature_mean',torch.zeros(256))
        self.register_buffer('feature_std',torch.ones(256))
        self.dense_precision='highest' if performance is None else performance['dense_precision']
        self.bf16_mode='none' if performance is None else performance.get('bf16_mode','none')
        if self.bf16_mode not in ('none','radial','radial_compensated'):
            raise ValueError(f'Unsupported BF16 scope: {self.bf16_mode}')
        if self.dense_precision not in ('highest','high'):
            raise ValueError(f'Unsupported learned-matrix precision: {self.dense_precision}')
        if performance is not None:
            if self.bf16_mode in ('radial','radial_compensated'):
                from src.models.encoders.mace_bf16 import BF16RadialLayer
                for interaction in self.backbone.interactions:
                    radial=interaction.conv_tp_weights
                    for name,layer in list(radial.named_children()):
                        setattr(radial,name,BF16RadialLayer(layer,self.bf16_mode=='radial_compensated'))
            if performance['geometry_cache'] and any(p.requires_grad for p in self.backbone.radial_embedding.parameters()):
                raise ValueError('Geometry reuse requires the repository checkpoint with a fixed radial embedding')
            if performance['compile_radial_mlp']:
                for interaction in self.backbone.interactions:
                    interaction.conv_tp_weights.compile(fullgraph=True,dynamic=True,backend='inductor',options={'triton.cudagraphs':False,'emulate_precision_casts':self.bf16_mode=='radial_compensated'})

    def build_geometry(self,x,material):
        with dense_matmul_precision('highest'):
            return self._build_geometry(x,material)

    def _build_geometry(self, x, material):
        b,n,_=x.shape
        if n!=80:raise ValueError(f'Plain MACE requires 80 atoms, got {n}')
        valid=(torch.cdist(x,x)<self.backbone.r_max)&~torch.eye(n,device=x.device,dtype=torch.bool)[None]
        ids=valid.nonzero();edges=(ids[:,1:]+ids[:,0,None]*n).T.contiguous()
        attrs=torch.nn.functional.one_hot(self.element_indices[material],len(self.backbone.atomic_numbers)).to(x.dtype).repeat_interleave(n,0)
        positions=x.reshape(-1,3);vectors=positions[edges[1]]-positions[edges[0]]
        angular=self.backbone.spherical_harmonics(vectors)
        radial,cutoff=self.backbone.radial_embedding(vectors.norm(dim=-1,keepdim=True),attrs,edges,self.backbone.atomic_numbers)
        return MACEGeometry(b,n,attrs,edges,angular,radial,cutoff)

    def raw_features_from_geometry(self,g):
        with dense_matmul_precision(self.dense_precision):
            return self._learned_features(g)

    def _learned_features(self, g):
        return self._learned_node_features(g).mean(1)

    def raw_node_features(self, x, material):
        """Return both scalar blocks as (B, 80, 256), preserving atom order."""
        geometry = self.build_geometry(x, material)
        with dense_matmul_precision(self.dense_precision):
            return self._learned_node_features(geometry)

    def _learned_node_features(self, g):
        model=self.backbone;h=model.node_embedding(g.attrs);features=[]
        for i,(interaction,product) in enumerate(zip(model.interactions,model.products)):
            h,sc=interaction(node_attrs=g.attrs,node_feats=h,edge_attrs=g.angular,
                             edge_feats=g.radial,edge_index=g.edges,cutoff=g.cutoff,first_layer=i==0)
            h=product(node_feats=h,sc=sc,node_attrs=g.attrs)
            features.append(h.reshape(g.batch_size,g.points,128))
        return torch.cat(features,dim=2)

    def raw_features(self,x,material):
        return self.raw_features_from_geometry(self.build_geometry(x,material))

    def forward_from_geometry(self,g):
        return (self.raw_features_from_geometry(g)-self.feature_mean)/self.feature_std

    def forward(self,x,material):
        return (self.raw_features(x,material)-self.feature_mean)/self.feature_std


@register_encoder('PretrainedMACEGeometry')
class PretrainedMACEGeometryEncoder(Encoder):
    """Original normalized point-cloud API, with one fixed pretrained Al channel.

    The data loader divides coordinates by its source cutoff. One shared scale
    converts those dimensionless inputs to the Al distance range of the MLIP;
    actual material identity and its physical radius never enter this encoder.
    VICRegModule owns the projector and every training loss.
    """
    input_layout='bn3'
    output_contract='invariant'
    invariant_dim=256
    equivariant_dim=None

    def __init__(self,pretrained_checkpoint,reference_radius_A,performance):
        super().__init__()
        self.mace=PretrainedMACEEncoder(pretrained_checkpoint,performance=performance)
        self.reference_radius_A=float(reference_radius_A)

    def forward(self,points):
        channel=torch.zeros(len(points),dtype=torch.long,device=points.device)
        return self.mace.raw_features(points*self.reference_radius_A,channel)
