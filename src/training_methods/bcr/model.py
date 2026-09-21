"""MACE scalar bottleneck and a separate O(3) conditional noise decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from e3nn import o3
from src.models.encoders.mace_causal import CausalMACEEncoder,normalize_atom_features
from .data import edges,taper


class Encoder(nn.Module):
    def __init__(self,d0,n_ref,radius,channels=32,code_dim=128,cutoff=5.,backend='e3nn'):
        super().__init__();self.channels=channels;self.radius=radius;self.cutoff=cutoff
        self.register_buffer('d0',torch.tensor(float(d0)));self.register_buffer('n_ref',torch.tensor(float(n_ref)))
        base=CausalMACEEncoder(channels=channels,output_dim=code_dim,num_layers=2,cutoff_A=cutoff,
                              use_velocity=False,use_history=False,mace_backend=backend)
        for name in ('node_embedding','radial_embedding','spherical_harmonics','interactions','products'):
            setattr(self,name,getattr(base,name))
        self.register_buffer('atomic_numbers',base.atomic_numbers)
        self.center_embedding=nn.Linear(1,channels,bias=False)
        self.readout=nn.Sequential(nn.Linear(2*channels,128),nn.SiLU(),nn.Linear(128,code_dim))

    def node_features(self,batch):
        if torch.any(batch['species'][batch['mask']]!=0):raise ValueError('BCR-v1 encoder accepts Al only')
        x=batch['positions'][batch['mask']];center=batch['center'][batch['mask']]
        edge=edges(batch['positions'],batch['mask'],self.cutoff)
        attrs=torch.ones(len(x),1,device=x.device,dtype=x.dtype)
        h=self.node_embedding(attrs)+self.center_embedding(center.to(x.dtype)[:,None])
        s,r=edge;v=x[r]-x[s];distance=v.norm(dim=-1,keepdim=True)
        angular=self.spherical_harmonics(v);radial,cut=self.radial_embedding(distance,attrs,edge,self.atomic_numbers)
        w=taper(x.norm(dim=-1),self.radius)
        radial=radial*(w[s]*w[r])[:,None]
        for k,(interaction,prod) in enumerate(zip(self.interactions,self.products,strict=True)):
            message,skip=interaction(node_attrs=attrs,node_feats=h,edge_attrs=angular,edge_feats=radial,
                edge_index=edge,cutoff=cut,first_layer=k==0)
            h=normalize_atom_features(prod(message,sc=skip,node_attrs=attrs))*w[:,None]
        return h

    def pooled(self,batch):
        h=self.node_features(batch)[:,:self.channels];m=batch['mask'];c=batch['center'][m]
        group=torch.arange(len(m),device=m.device)[:,None].expand_as(m)[m]
        w=taper(batch['positions'][m].norm(dim=-1),self.radius)
        pool=h.new_zeros(len(m),self.channels).index_add(0,group,h*w[:,None])/self.n_ref
        return torch.cat((h[c],pool),-1)

    def forward(self,batch):return self.readout(self.pooled(batch))
    def export(self,batch):return self(batch)


class Block(nn.Module):
    def __init__(self,irreps,condition_dim):
        super().__init__();self.irreps=o3.Irreps(irreps);sh=o3.Irreps('0e + 1o + 2e')
        # Channel-wise tensor products followed by learned same-degree mixing.
        # Record every allowed O(3) triangle/parity path for reproducibility.
        mid=[];instructions=[];self.paths=[]
        for i,(mul,ir) in enumerate(self.irreps):
            for j,(_,angular) in enumerate(sh):
                for out in ir*angular:
                    if out in [v.ir for v in self.irreps]:
                        instructions.append((i,j,len(mid),'uvu',True));mid.append((mul,out));self.paths.append((str(ir),str(angular),str(out)))
        self.tp=o3.TensorProduct(self.irreps,sh,o3.Irreps(mid),instructions,internal_weights=True,shared_weights=True)
        self.mix=o3.Linear(o3.Irreps(mid),self.irreps)
        self.skip=o3.Linear(self.irreps,self.irreps)
        self.radial=nn.Sequential(nn.Linear(16,32),nn.SiLU(),nn.Linear(32,1))
        self.film=nn.Linear(condition_dim,sum(m for m,ir in self.irreps)+self.irreps[0].mul)
        nn.init.normal_(self.film.weight,std=.01);nn.init.zeros_(self.film.bias)
        self.gates=nn.Linear(self.irreps[0].mul,sum(m for m,ir in self.irreps[1:]))

    def forward(self,h,edge,sh,radial,cutoff,condition):
        s,r=edge
        messages=self.tp(h[s],sh)*self.radial(radial)*cutoff[:,None]
        h=self.skip(h)+self.mix(h.new_zeros(len(h),messages.shape[-1]).index_add(0,r,messages))/12**.5
        scale,bias=self.film(condition).split([sum(m for m,ir in self.irreps),self.irreps[0].mul],-1);parts=[];offset=0
        scalars=h[:,:self.irreps[0].mul]
        gates=self.gates(scalars);gi=0
        for (mul,ir),sl in zip(self.irreps,self.irreps.slices(),strict=True):
            v=h[:,sl].reshape(len(h),mul,ir.dim)*(1+scale[:,offset:offset+mul,None])
            if ir.l==0:v=F.silu(v+bias[:,offset:offset+mul,None])
            else:
                v=v*torch.sigmoid(gates[:,gi:gi+mul,None]);gi+=mul
            parts.append(v.flatten(1));offset+=mul
        return torch.cat(parts,-1)


class Decoder(nn.Module):
    """Only noisy geometry and exported z cross this API; no clean graph input."""
    def __init__(self,d0,code_dim=128,irreps='64x0e + 32x1o + 16x2e',depth=2):
        super().__init__();self.irreps=o3.Irreps(irreps);self.register_buffer('d0',torch.tensor(float(d0)))
        self.embed=nn.Linear(2,self.irreps[0].mul)
        self.noise=nn.Sequential(nn.Linear(1,16),nn.SiLU(),nn.Linear(16,16))
        self.blocks=nn.ModuleList([Block(self.irreps,code_dim+16) for _ in range(depth)])
        self.out=o3.Linear(self.irreps,'1o')
        self.register_buffer('centers',torch.linspace(0,1,16))

    def forward(self,positions,species,center,mask,z,log_sigma):
        # Species has a single allowed value in v1, explicitly reject other data.
        if torch.any(species[mask]!=0):raise ValueError('BCR-v1 decoder accepts Al only')
        x=positions[mask];group=torch.arange(len(mask),device=x.device)[:,None].expand_as(mask)[mask]
        edge=edges(positions,mask,2*self.d0);s,r=edge;v=(x[r]-x[s])/(2*self.d0)
        distance=v.norm(dim=-1);cut=taper(distance,1.)
        # Solid harmonics are nonsingular at collisions, with radial powers
        # explicit. Unlike normalized directions, derivatives stay finite at 0.
        sh=o3.spherical_harmonics([0,1,2],v,normalize=False,normalization='component')
        radial=torch.exp(-((distance[:,None]-self.centers)*16)**2)
        scalar=self.embed(torch.stack((torch.ones(len(x),device=x.device,dtype=x.dtype),center[mask].to(x.dtype)),-1))
        h=torch.cat((scalar,x.new_zeros(len(x),self.irreps.dim-scalar.shape[-1])),-1)
        cond=torch.cat((z,self.noise(log_sigma[:,None])),-1)[group]
        for block in self.blocks:h=block(h,edge,sh,radial,cut,cond)
        result=torch.zeros_like(positions);result[mask]=self.out(h)
        return result*(mask&~center)[...,None]


class BCR(nn.Module):
    def __init__(self,config):
        super().__init__();self.config=config;self.arm=config.get('arm','bcr')
        if self.arm not in ('bcr','unconditional','frozen_random','frozen_vicreg','denoising','vicreg'):raise ValueError(self.arm)
        self.encoder=Encoder(**config['encoder']);dim=config['encoder'].get('code_dim',128)
        self.decoder=Decoder(config['encoder']['d0'],code_dim=dim,**config.get('decoder',{}))
        if self.arm=='unconditional':self.constant=nn.Parameter(torch.zeros(dim))
        if self.arm in ('unconditional','frozen_random','frozen_vicreg'):
            self.encoder.requires_grad_(False)
        if self.arm=='frozen_vicreg':
            checkpoint=torch.load(config['vicreg_checkpoint'],weights_only=False,map_location='cpu')
            if checkpoint['config']['arm']!='vicreg' or checkpoint['config']['encoder']!=config['encoder']:
                raise ValueError('Frozen VICReg must use identical BCR support/readout and declared lineage exposure')
            if checkpoint['data_identity']!=config['data_identity']:raise ValueError('VICReg data exposure differs')
            self.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in checkpoint['model'].items() if k.startswith('encoder.')})
        if self.arm=='denoising':
            ch=config['encoder'].get('channels',32)
            self.node_head=o3.Linear(f'{ch}x0e + {ch}x1o + {ch}x2e','1o')
            self.noise_gain=nn.Sequential(nn.Linear(1,16),nn.SiLU(),nn.Linear(16,1))
            self.encoder.readout.requires_grad_(False);self.decoder.requires_grad_(False)
        if self.arm=='vicreg':self.decoder.requires_grad_(False)

    def encode(self,batch):
        # Denoising exports trained pooled scalar features, never a random MLP.
        return self.encoder.pooled(batch) if self.arm=='denoising' else self.encoder.export(batch)

    def forward(self,clean,noisy,sigma):
        if self.arm=='denoising':
            out=torch.zeros_like(noisy['positions']);out[noisy['mask']]=self.node_head(self.encoder.node_features(noisy))
            out=out*(1+self.noise_gain(torch.log(sigma/self.encoder.d0)[:,None]))[:,None,:]
            return out,self.encode(clean)
        z=self.constant[None].expand(len(sigma),-1) if self.arm=='unconditional' else self.encoder(clean)
        return self.decoder(noisy['positions'],noisy['species'],noisy['center'],noisy['mask'],z,
                            torch.log(sigma/self.encoder.d0)),z
