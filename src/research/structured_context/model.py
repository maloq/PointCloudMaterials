"""Geometry-aware spatial blocks interleaved with causal attention per query slot."""
import torch
from torch import nn
from torch.nn import functional as F
from src.research.crystallization_transfer.attention import TrainingMoments
from src.research.crystallization_paths.refined_model import RefinedForecaster
from src.research.context_night.context import INFO_DIM,context_mask
from .geometry import stencil


class StructuredBlock(nn.Module):
    def __init__(self,width,heads):
        super().__init__();self.heads=heads;self.width=width
        self.norm=nn.LayerNorm(width);self.qkv=nn.Linear(width,3*width);self.out=nn.Linear(width,width)
        self.pair=nn.Sequential(nn.Linear(5,32),nn.SiLU(),nn.Linear(32,heads))
        self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,2*width),nn.GELU(),nn.Linear(2*width,width))

    def forward(self,x,actual,nominal,weights,causal):
        b,n,_=x.shape;h=self.heads
        q,k,v=self.qkv(self.norm(x)).reshape(b,n,3,h,self.width//h).permute(2,0,3,1,4)
        lag=(actual[:,:,None,3]-actual[:,None,:,3])/48
        dr=(actual[:,:,None,:3]-actual[:,None,:,:3])/25
        dq=(nominal[:,:,None]-nominal[:,None,:])/20
        offset=(actual[...,:3]-nominal)/4
        features=torch.stack((dr.square().sum(-1),dq.square().sum(-1),
            (offset[:,:,None]*offset[:,None,:]).sum(-1),lag,lag.square()),-1)
        bias=self.pair(features).permute(0,3,1,2)+weights.log()[:,None,None,:]
        if causal:bias=bias.masked_fill(lag[:,None]<0,-torch.inf)
        values=F.scaled_dot_product_attention(q,k,v,attn_mask=bias).transpose(1,2).reshape(b,n,self.width)
        x=x+self.out(values);return x+self.ff(x)


class StructuredHead(nn.Module):
    def __init__(self,spec):
        super().__init__();w=spec['head_width'];self.width=w;self.nodes=25
        self.register_buffer('queries',torch.from_numpy(stencil(spec['shell_radii_A'])))
        # Equal mass for center / inner shell / outer shell. No boundary suppression.
        self.register_buffer('slot_weights',torch.tensor([1.]+[1/12]*24))
        self.normalization=TrainingMoments(128,spec['norm_eps'])
        self.token=nn.Sequential(nn.Linear(128,w),nn.LayerNorm(w),nn.SiLU())
        self.geometry=nn.Linear(4,w);self.time=nn.Linear(2,w)
        self.spatial=nn.ModuleList([StructuredBlock(w,spec['heads']) for _ in range(spec['depth'])])
        self.temporal=nn.ModuleList([StructuredBlock(w,spec['heads']) for _ in range(spec['depth'])])
        self.output=nn.Sequential(nn.Linear(2*w+7,w),nn.LayerNorm(w),nn.SiLU())

    def inputs(self,features,geometry):
        b,n,_=features.shape
        if n%self.nodes:raise ValueError(f'Incomplete structured context: {n} tokens for 25 slots')
        return features,self.slot_weights.repeat(n//self.nodes).expand(b,-1)

    def forward(self,features,geometry,condition):
        b,n,_=features.shape;t=n//self.nodes;d=self.width
        f,weights=self.inputs(features,geometry);f=self.normalization(f,weights)
        nominal=self.queries.repeat(t,1).expand(b,-1,-1)
        offset=(geometry[...,:3]-nominal)/4
        scalar=torch.stack((nominal.norm(dim=-1)/20,geometry[...,:3].norm(dim=-1)/25,
            offset.square().sum(-1),(offset*nominal/20).sum(-1)),-1)
        dt=geometry[...,3]/48
        x=self.token(f)+self.geometry(scalar)+self.time(torch.stack((dt,dt.square()),-1))
        current=x[:,-self.nodes]
        actual=geometry.reshape(b,t,self.nodes,4);query=nominal.reshape(b,t,self.nodes,3)
        for space,time in zip(self.spatial,self.temporal,strict=True):
            x=space(x.reshape(b*t,self.nodes,d),actual.reshape(b*t,self.nodes,4),query.reshape(b*t,self.nodes,3),
                self.slot_weights.expand(b*t,-1),False).reshape(b,t,self.nodes,d)
            # Follow a fixed spatial query slot; its assigned real atom may change.
            x=time(x.transpose(1,2).reshape(b*self.nodes,t,d),actual.transpose(1,2).reshape(b*self.nodes,t,4),
                query.transpose(1,2).reshape(b*self.nodes,t,3),x.new_ones(b*self.nodes,t),True)
            x=x.reshape(b,self.nodes,t,d).transpose(1,2)
        context=(x[:,-1]*self.slot_weights[None,:,None]).sum(1)/self.slot_weights.sum()
        return self.output(torch.cat((current,context,condition),-1))


class StructuredForecaster(RefinedForecaster):
    def __init__(self,spec):
        super().__init__(spec);self.context=StructuredHead(spec);w=spec['head_width']
        self.information_head=nn.Sequential(nn.Linear(INFO_DIM,w),nn.LayerNorm(w),nn.SiLU(),nn.Linear(w,w))
        nn.init.zeros_(self.information_head[-1].weight);nn.init.zeros_(self.information_head[-1].bias)
        self.register_buffer('information_mean',torch.zeros(INFO_DIM));self.register_buffer('information_scale',torch.ones(INFO_DIM))
        self.register_buffer('information_mask',context_mask(spec['information_context']))

    def encode(self,observed):
        self.context.normalization.eval()
        value=self.context(**{k:v for k,v in observed.items() if k!='information'})
        info=(observed['information']-self.information_mean)/self.information_scale
        return value+self.information_head(info*self.information_mask)

    def initialize_information(self,data):
        self.information_mean.copy_(data.information_mean);self.information_scale.copy_(data.information_scale)

    def anchor(self,observed,context):
        value=self.initial(context)
        z=(observed['features'][:,-25]-self.target_mean[:128])/self.target_scale[:128]
        liquid=(-self.target_mean[264]/self.target_scale[264]).expand(len(z),1)
        return torch.cat((z,value[:,128:264],liquid),-1)
