"""Adaptive invariant interfaces and controlled spatial/temporal attention heads."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import tensor_invariants


def radial_weights(geometry,radius):
    distance=geometry[...,:3].norm(dim=-1)
    u=(distance/max(radius,1)).clamp(0,1)
    return (1-10*u**3+15*u**4-6*u**5).clamp_min(0).masked_fill(distance>=max(radius,1),0)


class TrainingMoments(nn.Module):
    """Full statistical-batch moments in training; explicit train-only calibration at inference.

    Mean/variance remain differentiable in training: a common encoder offset must
    not acquire the huge gradient produced by subtracting a stale constant mean.
    Each observation has unit mass before applying its source-balancing weight.
    """
    def __init__(self,width,eps):
        super().__init__();self.eps=eps
        self.register_buffer('mean',torch.zeros(width));self.register_buffer('variance',torch.ones(width))
        self.register_buffer('calibrated',torch.tensor(False))

    def forward(self,x,weights,sample_weight=None):
        if self.training:
            if sample_weight is None:raise ValueError('Training normalization requires full-batch source weights')
            w=weights/weights.sum(1,keepdim=True)*sample_weight[:,None]
            w=w/w.sum();mean=(x*w[...,None]).sum((0,1))
            variance=((x-mean).square()*w[...,None]).sum((0,1))
        else:
            if not self.calibrated:raise RuntimeError('Inference moments have not been calibrated on training inputs')
            mean=self.mean;variance=self.variance
        return (x-mean)*torch.rsqrt(variance+self.eps)

    @torch.no_grad()
    def calibrate(self,x,weights):
        # Float64 prevents cancellation at the ~1e-6 native embedding variance.
        x=x.double();w=weights.double()/weights.double().sum(1,keepdim=True);w=w/w.sum()
        mean=(x*w[...,None]).sum((0,1));variance=((x-mean).square()*w[...,None]).sum((0,1))
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():raise FloatingPointError('Nonfinite train-only calibration moments')
        self.mean.copy_(mean);self.variance.copy_(variance);self.calibrated.fill_(True)


class AttentionBlock(nn.Module):
    def __init__(self,width,heads,geometry_bias,tensor=False):
        super().__init__()
        if width%heads:raise ValueError('Attention width must divide by heads')
        self.heads=heads;self.width=width;self.geometry_bias=geometry_bias;self.tensor=tensor
        self.norm=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
        self.pair=nn.Sequential(nn.Linear(5 if tensor else 3,32),nn.SiLU(),nn.Linear(32,heads))
        self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))

    def forward(self,x,geometry,weights,*,causal,tensors=None):
        b,n,_=x.shape;h=self.heads;q,k,v=self.qkv(self.norm(x)).reshape(b,n,3,h,self.width//h).permute(2,0,3,1,4)
        lag=(geometry[:,:,None,3]-geometry[:,None,:,3])/48
        distance=((geometry[:,:,None,:3]-geometry[:,None,:,:3])/25).square().sum(-1)
        pair=torch.stack((distance,lag,lag.square()),-1)
        if self.tensor:
            alignment=[]
            for t in (tensors[...,32:128],tensors[...,128:288]):
                magnitude=t.square().sum(-1)
                alignment.append((t@t.transpose(-1,-2))/torch.sqrt(magnitude[:,:,None]*magnitude[:,None,:]+1e-8))
            pair=torch.cat((pair,torch.stack(alignment,-1)),-1)
        bias=self.pair(pair).permute(0,3,1,2) if self.geometry_bias else torch.zeros((b,h,n,n),device=x.device,dtype=x.dtype)
        bias=bias+weights.clamp_min(1e-12).log().masked_fill(weights==0,float('-inf'))[:,None,None,:]
        if causal:bias=bias.masked_fill(lag[:,None]<0,float('-inf'))
        value=F.scaled_dot_product_attention(q,k,v,attn_mask=bias).transpose(1,2).reshape(b,n,self.width)
        x=x+self.out(value);return x+self.ff(x)


class AdaptiveContextHead(nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=spec;self.topology=spec['attention'];self.width=spec['head_width']
        tensor=spec['equivariant'];width=self.width;depth=spec['depth'];heads=spec['heads']
        self.normalization=TrainingMoments(288 if tensor else 128,spec['norm_eps'])
        self.token=nn.Sequential(nn.Linear(288 if tensor else 128,width),nn.LayerNorm(width),nn.SiLU())
        self.time=nn.Linear(2,width)
        self.spatial=nn.ModuleList([AttentionBlock(width,heads,spec['geometry_bias'],tensor) for _ in range(depth)]) if self.topology in ('spatial','factorized','joint') else nn.ModuleList()
        self.temporal=nn.ModuleList([AttentionBlock(width,heads,True) for _ in range(depth)]) if self.topology in ('temporal','factorized') else nn.ModuleList()
        self.output=nn.Sequential(nn.Linear(2*width+7,width),nn.LayerNorm(width),nn.SiLU(),nn.Linear(width,6))

    def inputs(self,features,geometry):
        x=features[...,:128]
        if self.spec['equivariant']:x=torch.cat((x,tensor_invariants(features,geometry)),-1)
        return x,radial_weights(geometry,self.spec['radius_A'])

    def forward(self,features,geometry,condition,descriptor=None,sample_weight=None):
        x,w=self.inputs(features,geometry);x=self.normalization(x,w,sample_weight)
        x=self.token(x);dt=geometry[...,3]/48;x=x+self.time(torch.stack((dt,dt.square()),-1))
        b,n,d=x.shape;nodes=1 if self.spec['radius_A']==0 else 7;t=n//nodes
        current=x[:,-nodes];frame_w=w.reshape(b,t,nodes)
        if self.topology=='joint':
            for block in self.spatial:x=block(x,geometry,w,causal=True,tensors=features[...,128:])
            frames=x.reshape(b,t,nodes,d)
        else:
            frames=x.reshape(b*t,nodes,d);g=geometry.reshape(b*t,nodes,4);fw=w.reshape(b*t,nodes)
            tensors=features[...,128:].reshape(b*t,nodes,288) if self.spec['equivariant'] else None
            for block in self.spatial:frames=block(frames,g,fw,causal=False,tensors=tensors)
            frames=frames.reshape(b,t,nodes,d)
        summaries=(frames*frame_w[...,None]).sum(2)/frame_w.sum(2,keepdim=True)
        if self.topology in ('temporal','factorized'):
            g=geometry.reshape(b,t,nodes,4)[:,:,0];tw=torch.ones((b,t),device=x.device,dtype=x.dtype)
            for block in self.temporal:summaries=block(summaries,g,tw,causal=True)
            context=summaries[:,-1]
        elif self.topology=='joint':context=summaries[:,-1]
        elif self.topology in ('spatial','mean'):context=summaries.mean(1)
        else:raise ValueError(f'Unknown context topology: {self.topology}')
        return self.output(torch.cat((current,context,condition),-1))
