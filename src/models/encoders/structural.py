"""Species-aware snapshot MACE and causal multi-frame GATr structural encoders."""
import torch
from torch import nn
from torch.nn import functional as F
from src.data.structural_pretraining.support import POOL_SCALES

from .mace_causal import CausalMACEEncoder, normalize_atom_features
from e3nn import o3
from .axial_gatr import support_bias
from .structural_precision import FullPrecision, FloatOutput, FullPrecisionReadout, protect_gatr
from .equivariant_bond import gatr_bond_features
from gatr.interface import embed_point
from gatr.layers.linear import EquiLinear
from gatr.layers.gatr_block import GATrBlock
from gatr.layers.attention.config import SelfAttentionConfig
from gatr.layers.mlp.config import MLPConfig

ATOMIC_NUMBERS = (12, 13, 22, 40, 73)
ARCHITECTURE_REVISION = 'structural_v10_local'


def normalized_readout(inputs,hidden,outputs):
    """Normalize scalar interfaces and preactivations before the SiLU tail."""
    return FullPrecisionReadout(nn.LayerNorm(inputs,elementwise_affine=False),nn.Linear(inputs,hidden),
        nn.LayerNorm(hidden,elementwise_affine=False),nn.SiLU(),nn.Linear(hidden,outputs))


def conditioned_readout(inputs,hidden,outputs):
    """Condition across observations; retain per-row hidden stabilization.

    Heads see the full statistical batch once, outside encoder gradient replay.
    Inference uses training-only moments refreshed for the current encoder.
    The exported encoder itself contains no batch-dependent normalization.
    """
    return FullPrecisionReadout(nn.BatchNorm1d(inputs,affine=False,eps=1e-6),
        nn.Linear(inputs,hidden),nn.LayerNorm(hidden,elementwise_affine=False),
        nn.SiLU(),nn.Linear(hidden,outputs))


class StructuralMACE(nn.Module):
    def __init__(self, channels=32, readout_hidden=104, backend='cueq'):
        super().__init__()
        base = CausalMACEEncoder(channels=channels, output_dim=128, num_layers=2,
            atomic_numbers=ATOMIC_NUMBERS, use_velocity=False, use_history=False,
            scales_A=POOL_SCALES, mace_backend=backend)
        for name in ('node_embedding','radial_embedding','spherical_harmonics','interactions','products','pool'):
            setattr(self,name,getattr(base,name))
        self.pool.compress=normalized_readout(len(self.pool.scales)*(5*channels+1),readout_hidden,128)
        self.output_norm=nn.LayerNorm(128,elementwise_affine=False)
        self.register_buffer('atomic_numbers',base.atomic_numbers)
        self.scale_input=nn.Linear(1,channels)
        self.backend=backend
        self.channels=channels
        for name in ('node_embedding','radial_embedding','spherical_harmonics','pool','scale_input'):
            setattr(self,name,FullPrecision(getattr(self,name)))
        for interaction in self.interactions:
            # The radial network only receives invariant distances. Its weights
            # return to FP32 before they multiply tensor-valued messages.
            interaction.conv_tp_weights=FloatOutput(interaction.conv_tp_weights)
            for name in ('linear_up','conv_tp','linear','skip_tp'):
                setattr(interaction,name,FullPrecision(getattr(interaction,name)))
        self.products=nn.ModuleList([FullPrecision(product) for product in self.products])

    def atom_features(self,batch):
        """Shared MACE atom pathway; coordinates/support retain the local-patch contract."""
        x=batch['packed_positions'].float(); graph=batch['node_graph']; edges=batch['edges']
        attrs=F.one_hot(batch['packed_species'],len(ATOMIC_NUMBERS)).to(x.dtype)
        h=self.node_embedding(attrs)+self.scale_input(batch['log_scale'][graph,None])
        sender,receiver=edges
        vectors=x[receiver]-x[sender]; distances=vectors.norm(dim=-1,keepdim=True)
        angular=self.spherical_harmonics(vectors)
        radial,cutoff=self.radial_embedding(distances,attrs,edges,self.atomic_numbers)
        w=batch['packed_weights']; radial=radial*(w[sender]*w[receiver])[:,None]
        for i,(interaction,product) in enumerate(zip(self.interactions,self.products,strict=True)):
            h,sc=interaction(node_attrs=attrs,node_feats=h,edge_attrs=angular,
                edge_feats=radial,edge_index=edges,cutoff=cutoff,first_layer=i==0)
            h=normalize_atom_features(product(h,sc=sc,node_attrs=attrs))*w[:,None]
        return dict(features=h, scalars=h[:,:self.channels], positions=x, graph=graph,
            weights=w, count=len(batch['log_scale']))

    def forward(self,batch,return_equivariant=False):
        atoms=self.atom_features(batch)
        h,x,graph=atoms['features'],atoms['positions'],atoms['graph']
        state=self.output_norm(self.pool(h,x,graph,atoms['count']))
        if return_equivariant:
            # Training-only cache interface. The default exported snapshot stays
            # 128-dimensional and contains no auxiliary prediction head.
            return torch.cat((state,h[batch['packed_centers']]),-1)
        return state


def causal_bias(weights,times):
    """Causal per-atom temporal attention; padded queries get finite dummy rows."""
    b,t,n=weights.shape
    lag=times[:,:,None]-times[:,None,:]
    mask=support_bias(weights.transpose(1,2))[:,:,None,:].expand(b,n,t,t).clone()
    mask=mask.masked_fill(lag[:,None]<0,-torch.inf)
    absent=weights.transpose(1,2)<=0
    eye=torch.eye(t,dtype=torch.bool,device=weights.device)
    return torch.where(absent[...,None]&eye,0.,mask).reshape(b*n,1,t,t)


class StructuralGATr(nn.Module):
    def __init__(self,mv_channels=8,scalar_channels=192,layers=2,history=False):
        super().__init__()
        self.mv_channels=mv_channels; self.scalar_channels=scalar_channels
        self.species=nn.Embedding(len(ATOMIC_NUMBERS),16)
        self.history=history
        self.input=protect_gatr(EquiLinear(1,mv_channels,in_s_channels=20,out_s_channels=scalar_channels))
        attention=SelfAttentionConfig(num_heads=4,pos_encoding=False)
        def block():return protect_gatr(GATrBlock(mv_channels,scalar_channels,attention,MLPConfig()))
        self.spatial=nn.ModuleList([block() for _ in range(layers)])
        if history:
            self.temporal=nn.ModuleList([block() for _ in range(layers)])
            self.time_embedding=nn.ModuleList([FullPrecision(nn.Linear(2,scalar_channels)) for _ in range(layers)])
            self.history_alpha=nn.Parameter(torch.full((layers,),.1))
        self.register_buffer('join_reference',embed_point(torch.zeros(3)))
        self.bond_harmonics=FullPrecision(o3.SphericalHarmonics([4,6],normalize=False,normalization='component'))
        self.readout=nn.Sequential(normalized_readout(scalar_channels,128,128),
            nn.LayerNorm(128,elementwise_affine=False))

    def atom_features(self,batch,return_multivectors=False):
        w=batch['weights']; mask=w>0; b,t,n=w.shape
        if t>1 and not self.history:
            raise ValueError('Snapshot GATr received multiple frames; construct with history=True')
        x=torch.where(mask[...,None],batch['positions']/5.,0.)
        species=self.species(batch['species'])[:,None].expand(b,t,n,16)
        count=w.sum(-1,keepdim=True).expand_as(w)/100.
        scale=batch['log_scale'][:,None,None].expand_as(w)
        scalar=torch.cat((torch.stack((w,x.square().sum(-1),count,scale),-1),species),-1)
        mv,scalar=self.input(embed_point(x).unsqueeze(-2),scalars=scalar)
        spatial_mask=support_bias(w).reshape(b*t,1,1,n)
        time_mask=causal_bias(w,batch['times']) if t>1 else None
        offsets=batch['times'] # physical ps, identical convention across materials
        time_features=torch.stack((offsets,offsets.square()),-1)
        for layer,block in enumerate(self.spatial):
            mv,scalar=block(mv.reshape(b*t,n,self.mv_channels,16),
                scalar.reshape(b*t,n,self.scalar_channels),reference_mv=self.join_reference,
                attention_mask=spatial_mask)
            mv=mv.reshape(b,t,n,self.mv_channels,16)*mask[...,None,None]
            scalar=scalar.reshape(b,t,n,self.scalar_channels)*mask[...,None]
            if t>1:
                ts=scalar+self.time_embedding[layer](time_features)[:,:,None]
                next_mv,next_s=self.temporal[layer](mv.transpose(1,2).reshape(b*n,t,self.mv_channels,16),
                    ts.transpose(1,2).reshape(b*n,t,self.scalar_channels),reference_mv=self.join_reference,
                    attention_mask=time_mask)
                next_mv=next_mv.reshape(b,n,t,self.mv_channels,16).transpose(1,2)*mask[...,None,None]
                next_s=next_s.reshape(b,n,t,self.scalar_channels).transpose(1,2)*mask[...,None]
                gate=self.history_alpha[layer].tanh()
                mv=mv+gate*(next_mv-mv); scalar=scalar+gate*(next_s-scalar)
        return (mv,scalar) if return_multivectors else scalar

    def forward(self,batch,return_equivariant=False):
        mv,features=self.atom_features(batch,return_multivectors=True)
        rows=torch.arange(len(features),device=features.device)
        state=self.readout(features[rows,-1,batch['centers']])
        if return_equivariant:
            return torch.cat((state,gatr_bond_features(mv[:,-1],batch['weights'][:,-1],self.bond_harmonics)),-1)
        return state


class StructuralModel(nn.Module):
    def __init__(self,architecture,backend='cueq',history=False):
        super().__init__()
        if architecture not in ('mace','gatr'):
            raise ValueError(f'Unknown architecture: {architecture}')
        if architecture=='mace' and history:
            raise ValueError('Structural MACE is snapshot-only')
        self.encoder=StructuralMACE(backend=backend) if architecture=='mace' else StructuralGATr(history=history)
        self.projector=FullPrecisionReadout(nn.BatchNorm1d(128,affine=False,eps=1e-6),
            nn.Linear(128,256,bias=False),nn.BatchNorm1d(256),nn.ReLU(),nn.Linear(256,64))
        self.physical=conditioned_readout(128,128,85)
        self.tda=conditioned_readout(128,128,144)
        self.predictor=FullPrecisionReadout(nn.Linear(65,128),nn.SiLU(),nn.Linear(128,64))

    def heads(self,z):
        return dict(q=self.projector(z),physical=self.physical(z),tda=self.tda(z))
