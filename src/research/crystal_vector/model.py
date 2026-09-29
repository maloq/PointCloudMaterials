"""End-to-end typed MACE and shared vector-message localization mixtures."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from src.research.supervised_onset.model import CapacityEncoder
from src.research.equivariant_context.features import TypedPatchExport
from src.research.equivariant_context.model import VectorMessages, geometry, radial_basis
from src.research.encoder_context.geometry import graph
from src.research.spatial_distance.model import parameters, component_cdf
from src.research.distance_encoder.model import loss_terms


class JointCrystalVector(nn.Module):
    def __init__(self, encoder_config, config):
        super().__init__(); self.config=config
        self.patch=TypedPatchExport(CapacityEncoder(**encoder_config))
        c=encoder_config['channels'];f=config['predictor']['field_channels'];w=config['predictor']['width']
        self.latent_dim=self.encoder.projection.out_features
        self.vector_export=nn.Linear(2*c,f,bias=False)
        self.register_buffer('scalar_mean',torch.zeros(self.latent_dim));self.register_buffer('scalar_scale',torch.ones(self.latent_dim))
        self.register_buffer('vector_scale',torch.ones(f))
        self.stem=nn.Sequential(nn.Linear(self.latent_dim,w),nn.LayerNorm(w),nn.SiLU())
        self.geometry=nn.Linear(4,w)
        self.blocks=nn.ModuleList([VectorMessages(w,f) for _ in range(config['predictor']['depth'])])
        self.distance_head=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w),nn.SiLU(),nn.Linear(w,3))
        self.direction_channels=nn.Linear(f,1,bias=False)
        self.direction_offset=nn.Linear(w,1)
        self.combine=nn.Sequential(nn.Linear(14,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.combine[-1].weight);nn.init.zeros_(self.combine[-1].bias)

    @property
    def encoder(self):return self.patch.encoder

    def encode(self, positions):
        chunk=self.config['patch_chunk'];outputs=[]
        for start in range(0,len(positions),chunk):
            xyz=positions[start:start+chunk];count=len(xyz)
            if count<chunk:
                pad=xyz.new_full((chunk-count,80,3),100.);pad[:,0]=0
                xyz=torch.cat((xyz,pad))
            g=graph(xyz,self.encoder)
            if self.training and torch.is_grad_enabled() and self.config['activation_checkpointing']:
                value=checkpoint(self.patch,g,use_reentrant=False)
            else:value=self.patch(g)
            outputs.append(value[:count])
        raw=torch.cat(outputs).float();c=self.encoder.channels
        z=raw[:,:self.latent_dim]
        vector=raw[:,self.latent_dim:self.latent_dim+6*c].reshape(-1,2*c,3)
        # Bias-free mixing acts on channels only, preserving Cartesian components.
        vector=F.linear(vector.transpose(-1,-2),self.vector_export.weight.float()).transpose(-1,-2)
        return (z-self.scalar_mean)/self.scalar_scale, vector/self.vector_scale[None,:,None]

    def forward(self,batch):
        z,v=self.encode(batch['positions']);inverse=batch['inverse']
        z=z[inverse];v=v[inverse]
        # Actual query-relative offsets are the complete geometry. No fixed lab stencil.
        g=geometry(batch['actual'],batch['actual'])
        s=self.stem(z)+self.geometry(g['node']);fields={1:v}
        for block in self.blocks:s,fields=block(s,fields,g)
        raw=self.distance_head(s).float()
        provisional=parameters(raw,torch.zeros_like(raw[...,0]))
        risk=component_cdf(provisional,raw.new_tensor([4,8,12,20,32,64]))
        scores=self.combine(torch.cat((risk,radial_basis(batch['actual'].norm(dim=-1),28.)),-1)).squeeze(-1).float()
        parts=parameters(raw,scores)
        vector=self.direction_channels(fields[1].transpose(-1,-2)).squeeze(-1)
        vector=vector-self.direction_offset(s)*batch['actual']/24.
        # Norm below one permits a uniform spherical density at zero, without an arbitrary axis.
        direction=vector.float()/torch.sqrt(vector.float().square().sum(-1,keepdim=True)+1e-8)
        return dict(parts=parts,direction=direction,z=z.float(),v=v.float(),state=s.float())


def component_log_distance(parts,distance,cap):
    _,zero,mu,sigma=parts
    d=distance.clamp(1e-6,cap)[:,None]
    positive=F.logsigmoid(-zero)
    density=positive-d.log()-sigma.log()-.5*math.log(2*math.pi)-.5*((d.log()-mu)/sigma).square()
    survival=positive+torch.special.log_ndtr((mu-math.log(cap))/sigma)
    return torch.where(distance[:,None]==0,F.logsigmoid(zero),torch.where(distance[:,None]>=cap,survival,density))


def directional_log_density(direction,target,distance,config):
    k=config['direction_kappa']/(1+(distance.clamp(max=config['distance_cap'])/config['direction_scale_A']).square())
    eta=direction*k[:,None,None]
    k2=eta.square().sum(-1);norm=k2.clamp_min(1e-12).sqrt()
    # log(sinh(k)/k), with its analytic small-k limit and finite unused branches.
    safe=norm.clamp_min(.01)
    log_sinhc=torch.where(k2<1e-4,k2/6-k2.square()/180,torch.sinh(safe).log()-safe.log())
    return (eta*target[:,None]).sum(-1)-math.log(4*math.pi)-log_sinhc


def objective(output,batch,config,directional):
    parts=output['parts'];d=batch['distance']
    total,nll,early=loss_terms(parts,d,config)
    log_d=component_log_distance(parts,d,config['distance_cap'])
    log_joint=torch.logsumexp(parts[0]+log_d+directional_log_density(output['direction'],batch['direction'],d,config),1)
    angular=torch.where(batch['valid'],-log_joint-nll,torch.zeros_like(nll))
    return dict(objective=total+config['direction_weight']*angular if directional else total,
        distance_nll=nll,direction_nll=angular,proximity_log_loss=early)


def vcreg(output,config):
    from .parallel import global_covariance
    # Independent target-population draws: every query has equal mass. Each has
    # 25 patches; average their sufficient statistics without claiming independence.
    terms=[];stats={}
    for name,field in [('scalar',output['z']),('vector',output['v'])]:
        x=field.flatten(0,1)
        if name=='scalar':x=x[...,None]
        covariance=global_covariance(x)
        std=(covariance.diagonal()+config['epsilon']).sqrt()
        variance=F.relu(config['std_floor']-std).mean()
        off=covariance-torch.diag_embed(covariance.diagonal())
        redundancy=off.square().sum()/(len(off)*(len(off)-1))
        terms.append(config['variance_weight']*variance+config['covariance_weight']*redundancy)
        stats[f'{name}_variance_penalty']=variance.detach();stats[f'{name}_covariance_penalty']=redundancy.detach()
        stats[f'{name}_minimum_std']=std.min().detach()
    return sum(terms),stats
