"""Scalar and tensor-aware, rotation-invariant spatial/temporal hazard readouts."""
import torch
from torch import nn
from torch.nn import functional as F
from src.models.encoders.structural import StructuralMACE
from e3nn import o3


def tensor_invariants(features,geometry):
    """Retain l=1,2 orientation until contractions with neighbor directions/context."""
    h=features[...,128:];r=geometry[...,:3]
    pieces=[h[...,:32]];start=32
    for l in (1,2):
        width=2*l+1;q=h[...,start:start+32*width].reshape(*h.shape[:-1],32,width);start+=32*width
        sh=o3.spherical_harmonics(l,r,normalize=True,normalization='component')
        # Last token is not assumed to be center: all node directions are explicit.
        pieces.extend([(q.square().sum(-1)+1e-8).sqrt(),(q*sh[...,None,:]).sum(-1)/(width**.5)])
    return torch.cat(pieces,-1)


class ContextHead(nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=spec
        self.register_buffer('mean',torch.zeros(128));self.register_buffer('scale',torch.ones(128))
        self.register_buffer('descriptor_mean',torch.zeros(136));self.register_buffer('descriptor_scale',torch.ones(136))
        inp=128+(160 if spec.get('equivariant') else 0)
        self.token=nn.Sequential(nn.Linear(inp,128),nn.LayerNorm(128),nn.SiLU())
        self.time=nn.Linear(2,128)
        self.qkv=nn.Linear(128,384);self.pair=nn.Sequential(nn.Linear(5 if spec.get('equivariant') else 3,32),nn.SiLU(),nn.Linear(32,4))
        baseline=spec.get('baseline')
        size=7 if baseline=='condition' else (143 if baseline=='descriptor' else 263)
        self.output=(nn.Linear(size,6) if spec['aggregation']=='linear' else nn.Sequential(nn.Linear(size,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,6)))

    def forward(self,features,geometry,condition,descriptor):
        baseline=self.spec.get('baseline')
        if baseline=='condition':return self.output(condition)
        if baseline=='descriptor':return self.output(torch.cat(((descriptor-self.descriptor_mean)/self.descriptor_scale,condition),-1))
        z=(features[...,:128]-self.mean)/self.scale
        inp=torch.cat((z,tensor_invariants(features,geometry)),dim=-1) if self.spec.get('equivariant') else z
        tokens=self.token(inp);dt=geometry[...,3]/48
        tokens=tokens+self.time(torch.stack((dt,dt.square()),-1))
        radius=self.spec['radius_A'];d=geometry[...,:3].norm(dim=-1)
        # All representatives are included with a smooth support envelope.
        u=(d/max(radius,1)).clamp(0,1);w=(1-10*u**3+15*u**4-6*u**5).clamp_min(0)
        center_index=geometry.shape[1]-(1 if radius==0 else 7)
        if self.spec['aggregation'] in ('linear','mean'):tokens=z
        current=tokens[:,center_index]
        if self.spec['aggregation']=='attention':
            b,n,_=tokens.shape;q,k,v=self.qkv(tokens).reshape(b,n,3,4,32).permute(2,0,3,1,4)
            r=geometry[...,:3]/25;distance=(r[:,:,None]-r[:,None,:]).square().sum(-1)
            lag=dt[:,:,None]-dt[:,None,:]
            pair=torch.stack((distance,lag,lag.square()),-1)
            if self.spec.get('equivariant'):
                align=[]
                for tensor in (features[...,160:256],features[...,256:416]):
                    magnitude=tensor.square().sum(-1)
                    align.append((tensor@tensor.transpose(-1,-2))/torch.sqrt(magnitude[:,:,None]*magnitude[:,None,:]+1e-8))
                pair=torch.cat((pair,torch.stack(align,-1)),-1)
            log_weight=w.clamp_min(1e-12).log().masked_fill(w==0,float('-inf'))
            bias=self.pair(pair).permute(0,3,1,2)+log_weight[:,None,None,:]
            # All tokens belong to the observed past; no future tokens exist.
            tokens=tokens+F.scaled_dot_product_attention(q,k,v,attn_mask=bias).transpose(1,2).reshape(b,n,128)
        pooled=(tokens*w[...,None]).sum(1)/w.sum(1,keepdim=True).clamp_min(1e-8)
        return self.output(torch.cat((current,pooled,condition),-1))


class Predictor(nn.Module):
    def __init__(self,spec,parent):
        super().__init__();self.spec=spec
        if spec.get('protocol')=='adaptive_v1':
            from .attention import AdaptiveContextHead
            self.head=AdaptiveContextHead(spec)
        else:self.head=ContextHead(spec)
        self.encoder=None
        if spec['mode']!='frozen':
            self.encoder=StructuralMACE()
            if spec['mode']=='finetune':self.encoder.load_state_dict(parent,strict=True)
            elif spec['mode']!='scratch':raise ValueError(spec['mode'])

    def encode(self,batch):
        if 'features' in batch:return batch['features']
        else:
            with torch.autocast('cuda',dtype=torch.bfloat16):
                features=self.encoder(batch['graph_batch'],return_equivariant=True).float()
            features=features.reshape(len(batch['event']),-1,416)
        return features

    def forward(self,batch):
        features=self.encode(batch)
        if self.spec.get('protocol')=='adaptive_v1':
            return self.head(features,batch['geometry'],batch['condition'],sample_weight=batch.get('loss_weight'))
        return self.head(features,batch['geometry'],batch['condition'],batch['descriptor'])
