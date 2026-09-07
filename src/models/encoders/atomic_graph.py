"""Central-atom representations on complete, buffered two-hop atomic graphs."""
import numpy as np
import torch
from torch import nn
from mace.modules.models import MACE
from mace.modules.blocks import RealAgnosticInteractionBlock,RealAgnosticResidualInteractionBlock
from mace.modules.wrapper_ops import CuEquivarianceConfig
from e3nn import o3
from .smooth_density import SmoothDensity


def graph_data(clouds,material,local_edges,edge_counts):
    """Batch repository-owned fixed-size patches and padded directed edge lists."""
    b,n,_=clouds.shape
    valid=torch.arange(local_edges.shape[1],device=clouds.device)[None]<edge_counts[:,None]
    edge=(local_edges.long()+torch.arange(b,device=clouds.device)[:,None,None]*n)[valid].T.contiguous()
    return dict(positions=clouds.reshape(-1,3),edge_index=edge,
        node_attrs=torch.nn.functional.one_hot(material,3).to(clouds.dtype).repeat_interleave(n,0),
        batch=torch.arange(b,device=clouds.device).repeat_interleave(n),
        ptr=torch.arange(b+1,device=clouds.device)*n,head=torch.zeros(b,device=clouds.device,dtype=torch.long),
        shifts=clouds.new_zeros(edge.shape[1],3),unit_shifts=clouds.new_zeros(edge.shape[1],3),cell=clouds.new_zeros(b,3,3))


class ReferenceMACEEncoder(nn.Module):
    """Reference MACE, two interactions, central scalar features from both layers.

    This instantiates mace.modules.models.MACE itself. The energy readouts are
    unused: self-supervised training acts through a separate external projector.
    Physical Angstrom coordinates and true Al/Mg/Ta element attributes are inputs.
    """
    def __init__(self,channels=64,max_ell=2,correlation=3,cutoff_A=4.,accelerated=True,avg_num_neighbors=20.):
        super().__init__()
        self.channels=channels
        hidden=o3.Irreps([(channels,(ell,(-1)**ell)) for ell in range(max_ell+1)])
        self.first_dim=hidden.dim
        # The fused cuEquivariance convolution descriptor uses ir_mul layout.
        # Using mul_ir here scrambles angular channels while leaving shapes valid.
        cueq=CuEquivarianceConfig(enabled=True,layout='ir_mul',group='O3_e3nn',optimize_all=True,conv_fusion=True) if accelerated else None
        if accelerated and not cueq.enabled:
            raise RuntimeError('Requested cuEquivariance is unavailable; select an explicit backend')
        self.backbone=MACE(r_max=cutoff_A,num_bessel=8,num_polynomial_cutoff=6,max_ell=max_ell,
            interaction_cls=RealAgnosticResidualInteractionBlock,interaction_cls_first=RealAgnosticInteractionBlock,
            num_interactions=2,num_elements=3,hidden_irreps=hidden,MLP_irreps=o3.Irreps('16x0e'),
            atomic_energies=np.zeros(3),avg_num_neighbors=avg_num_neighbors,atomic_numbers=[13,12,73],correlation=correlation,
            gate=torch.nn.functional.silu,radial_MLP=[64,64,64],radial_type='bessel',
            apply_cutoff=False,cueq_config=cueq)
        self.backbone.readouts.requires_grad_(False)

    def forward(self,clouds,material,local_edges,edge_counts):
        data=graph_data(clouds,material,local_edges,edge_counts)
        features=self.backbone(data,training=self.training,compute_force=False)['node_feats'][data['ptr'][:-1]]
        return torch.cat((features[:,:self.channels],features[:,self.first_dim:self.first_dim+self.channels]),1)


class ShiftedSoftplus(nn.Module):
    def forward(self,x):
        return torch.nn.functional.softplus(x)-np.log(2.)


class SchNetEncoder(nn.Module):
    """Two SchNet continuous-filter interaction blocks with a central readout."""
    def __init__(self,channels=64,cutoff_A=4.,radial_basis=32):
        super().__init__()
        self.cutoff_A=cutoff_A
        self.register_buffer('radial_centers',torch.linspace(0,cutoff_A,radial_basis))
        self.width=cutoff_A/(radial_basis-1)
        self.embedding=nn.Embedding(3,channels)
        self.filters=nn.ModuleList([nn.Sequential(nn.Linear(radial_basis,channels),ShiftedSoftplus(),nn.Linear(channels,channels)) for _ in range(2)])
        self.node_maps=nn.ModuleList([nn.Linear(channels,channels,bias=False) for _ in range(2)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(channels,channels),ShiftedSoftplus(),nn.Linear(channels,channels)) for _ in range(2)])

    def forward(self,clouds,material,local_edges,edge_counts):
        data=graph_data(clouds,material,local_edges,edge_counts)
        src,dst=data['edge_index']
        r=(data['positions'][dst]-data['positions'][src]).norm(dim=-1)
        u=(r/self.cutoff_A).clamp(0,1)
        taper=1-u**3*(10-u*(15-6*u))
        radial=torch.exp(-.5*((r[:,None]-self.radial_centers)/self.width)**2)
        h=self.embedding(material).repeat_interleave(clouds.shape[1],0)
        center=data['ptr'][:-1]
        outputs=[]
        for filters,node_map,update in zip(self.filters,self.node_maps,self.updates):
            messages=node_map(h)[src]*filters(radial)*taper[:,None]
            summed=torch.zeros_like(h).index_add(0,dst,messages)
            h=h+update(summed/20.)
            outputs.append(h[center])
        return torch.cat(outputs,1)


class DensityMLPEncoder(nn.Module):
    """Smooth invariant density powers with chemical identity and a learned MLP."""
    def __init__(self):
        super().__init__()
        self.density=SmoothDensity()
        self.register_buffer('mean',torch.zeros(260))
        self.register_buffer('std',torch.ones(260))
        self.head=nn.Sequential(nn.Linear(263,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,128))

    def forward(self,clouds,material,local_edges,edge_counts):
        power=self.density.power(self.density(clouds[:,1:]/8.))
        types=torch.nn.functional.one_hot(material,3).to(clouds.dtype)
        return self.head(torch.cat(((power-self.mean)/self.std,types),1))
