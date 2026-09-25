"""Shared node predictors and a learned mixture of focal-event distributions.

The origin identifies the prediction location geometrically. No node index,
central-token residual, special encoder, or hand-weighted central vote is used.
These are compact adaptations inspired by PaiNN, Equiformer and harmonic
multiscale message passing, not reproductions of their published models.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from e3nn import o3

FIELD_ORDERS = {'symmetric_invariant': (), 'vector_messages': (1,),
                'tensor_attention': (1, 2), 'harmonic_hierarchy': (1, 2, 4, 6)}
VARIANTS = tuple(FIELD_ORDERS)


def context_fields(variant):
    """Exactly the cached observations consumed by a predictor variant."""
    if variant not in FIELD_ORDERS:
        raise ValueError(f'Unknown context variant: {variant}')
    return ('z', 'actual', *(f'f{l}' for l in FIELD_ORDERS[variant]))


def envelope(distance, radius):
    x=(distance/radius).clamp(0,1)
    return (1-x).pow(3)*(1+3*x+6*x.square())


def event_log_probabilities(logits):
    survival=F.logsigmoid(-logits).cumsum(-1)
    prefix=F.pad(survival[...,:-1],(1,0))
    return torch.cat((prefix+F.logsigmoid(logits),survival[...,-1:]),-1)


def mix_predictions(node_logits, scores):
    return torch.logsumexp(event_log_probabilities(node_logits)+F.log_softmax(scores,dim=1)[...,None],dim=1)


def radial_basis(distance, radius=48.):
    return torch.exp(-((distance[...,None]/radius-torch.linspace(0,1,8,device=distance.device))/.18).square())


def geometry(actual, nominal):
    displacement=actual[:,None]-actual[:,:,None]  # sender minus receiver
    distance=displacement.norm(dim=-1)
    unit=displacement/distance[...,None].clamp_min(1e-8)
    offset=(actual-nominal)/4
    dq=(nominal[:,None]-nominal[:,:,None])/20
    pair=torch.stack(((distance/25).square(),dq.square().sum(-1),
        torch.einsum('bic,bjc->bij',offset,offset),torch.zeros_like(distance),torch.zeros_like(distance)),-1)
    node=torch.stack((nominal.norm(dim=-1)/20,actual.norm(dim=-1)/25,
        offset.square().sum(-1),(offset*nominal/20).sum(-1)),-1)
    return dict(unit=unit,distance=distance,pair=pair,node=node,radial=radial_basis(distance),
                support=envelope(distance,48.))


class ScalarAttention(nn.Module):
    def __init__(self,width,heads=4):
        super().__init__();self.heads=heads;self.width=width
        self.norm=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
        self.pair=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,heads))
        self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.SiLU(),nn.Linear(2*width,width))

    def attend(self,s,g):
        b,n,d=s.shape
        q,k,v=self.qkv(self.norm(s)).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        score=q@k.transpose(-1,-2)/math.sqrt(d//self.heads)+self.pair(g['pair']).permute(0,3,1,2)
        score=score+g['support'].log()[:,None]
        attention=score.softmax(-1)
        out=(attention@v).transpose(1,2).reshape(b,n,d)
        return s+self.out(out),attention.mean(1)

    def forward(self,s,fields,g):
        s,_=self.attend(s,g)
        return s+self.ff(s),fields


class VectorMessages(nn.Module):
    def __init__(self,width,channels):
        super().__init__();self.channels=channels
        self.sender=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width+2*channels))
        self.radial=nn.Linear(8,width+2*channels)
        self.value=nn.Linear(channels,channels,bias=False)
        self.update=nn.Sequential(nn.Linear(width+channels,width),nn.SiLU(),nn.Linear(width,width+channels))

    def forward(self,s,fields,g):
        v=fields[1];c=self.channels;d=s.shape[-1]
        edge=self.sender(s)[:,None]*self.radial(g['radial'])
        weight=g['support']/g['support'].sum(-1,keepdim=True)
        sm=(edge[...,:d]*weight[...,None]).sum(2)
        values=self.value(v.transpose(-1,-2)).transpose(-1,-2)
        vm=edge[...,d:d+c,None]*values[:,None]+edge[...,d+c:,None]*g['unit'][...,None,:]
        v=v+(vm*weight[...,None,None]).sum(2)
        update=self.update(torch.cat((s+sm,v.square().sum(-1)),dim=-1))
        s=s+sm+update[...,:d]
        v=v*torch.sigmoid(update[...,d:,None])
        return s,{1:v}


class TensorAttention(ScalarAttention):
    def __init__(self,width,channels):
        super().__init__(width);self.channels=channels
        self.values=nn.ModuleDict({str(l):nn.Linear(channels,channels,bias=False) for l in (1,2)})
        self.inject=nn.Linear(width,2*channels)
        self.feedback=nn.Linear(2*channels,width)
        self.register_buffer('cg112',o3.wigner_3j(1,1,2))
        self.register_buffer('cg211',o3.wigner_3j(2,1,1))

    def forward(self,s,fields,g):
        s,attention=self.attend(s,g)
        f={l:self.values[str(l)](v.transpose(-1,-2)).transpose(-1,-2) for l,v in fields.items()}
        coefficients=self.inject(s).reshape(*s.shape[:2],2,self.channels)
        updated={}
        for l,coupling in ((1,self.cg211),(2,self.cg112)):
            other=f[3-l]
            coupled=torch.einsum('bjca,biju,aum->bijcm',other,g['unit'],coupling)
            y=o3.spherical_harmonics(l,g['unit'],normalize=False,normalization='component')
            message=f[l][:,None]+coupled+coefficients[:,:,l-1][:,None,...,None]*y[...,None,:]
            value=fields[l]+torch.einsum('bij,bijcm->bicm',attention,message)
            updated[l]=value/torch.sqrt(1+value.square().mean((-1,-2),keepdim=True))
        s=s+self.feedback(torch.cat([updated[l].square().sum(-1) for l in (1,2)],-1))
        return s+self.ff(s),updated


class HarmonicHierarchy(nn.Module):
    """Nested neighborhoods around EVERY patch, with equivariant regional fields."""
    def __init__(self,width,channels):
        super().__init__();self.orders=(1,2,4,6);self.radii=(10.,20.,48.)
        self.values=nn.ModuleDict({str(l):nn.Linear(channels,channels,bias=False) for l in self.orders})
        self.inject=nn.Linear(width,4*channels)
        self.scales=nn.Linear(width,3)
        self.update=nn.Sequential(nn.Linear(2*width+4*channels,width),nn.SiLU(),nn.Linear(width,width))

    def forward(self,s,fields,g):
        weights=torch.stack([envelope(g['distance'],r) for r in self.radii],1)
        weights=weights/weights.sum(-1,keepdim=True)
        scale=self.scales(s).softmax(-1)
        # Mix the linear regional sums first. Do not materialize
        # [batch, receiver, sender, channel, irrep_component] messages.
        effective=torch.einsum('bir,brij->bij',scale,weights)
        parent_s=effective@s
        coefficients=self.inject(s).reshape(*s.shape[:2],4,-1)
        updated={};alignment=[]
        for k,l in enumerate(self.orders):
            f=self.values[str(l)](fields[l].transpose(-1,-2)).transpose(-1,-2)
            y=o3.spherical_harmonics(l,g['unit'],normalize=False,normalization='component')
            neighbor=torch.einsum('bij,bjcm->bicm',effective,f)
            weighted_coeff=effective[...,None]*coefficients[:,None,:,k,:]
            combined=neighbor+weighted_coeff.transpose(-1,-2)@y
            alignment.append((fields[l]*combined).sum(-1))
            value=fields[l]+combined
            updated[l]=value/torch.sqrt(1+value.square().mean((-1,-2),keepdim=True))
        update=self.update(torch.cat((s,parent_s,*alignment),-1))
        return s+update,updated


class ContextPredictor(nn.Module):
    def __init__(self,variant,width=128,field_channels=16,depth=2,encoder_channels=128):
        super().__init__()
        if variant not in VARIANTS:raise ValueError(variant)
        self.variant=variant
        self.stem=nn.Sequential(nn.Linear(128,width),nn.LayerNorm(width),nn.SiLU())
        self.geometry=nn.Linear(4,width)
        orders=FIELD_ORDERS[variant]
        self.fields=nn.ModuleDict({str(l):nn.Linear(2*encoder_channels if l<3 else 2,field_channels,bias=False) for l in orders})
        block={'symmetric_invariant':lambda:ScalarAttention(width),
               'vector_messages':lambda:VectorMessages(width,field_channels),
               'tensor_attention':lambda:TensorAttention(width,field_channels),
               'harmonic_hierarchy':lambda:HarmonicHierarchy(width,field_channels)}[variant]
        self.blocks=nn.ModuleList([block() for _ in range(depth)])
        self.node_head=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,5))
        # Shared combiner sees patch predictions and geometry, not a privileged z.
        self.combine=nn.Sequential(nn.Linear(6+8,64),nn.SiLU(),nn.Linear(64,1))
        nn.init.zeros_(self.combine[-1].weight);nn.init.zeros_(self.combine[-1].bias)

    def forward(self,batch,return_parts=False):
        g=geometry(batch['actual'],batch['nominal'])
        s=self.stem(batch['z'])+self.geometry(g['node'])
        fields={int(l):layer(batch[f'f{l}'].transpose(-1,-2)).transpose(-1,-2) for l,layer in self.fields.items()}
        for block in self.blocks:s,fields=block(s,fields,g)
        node_logits=self.node_head(s)
        node_logp=event_log_probabilities(node_logits)
        radius=radial_basis(batch['actual'].norm(dim=-1),28.)
        scores=self.combine(torch.cat((node_logp.exp(),radius),-1)).squeeze(-1)
        prediction=mix_predictions(node_logits,scores)
        if return_parts:return prediction,node_logp,scores.softmax(1),fields
        return prediction
