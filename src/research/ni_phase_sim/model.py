"""Rotation/permutation-equivariant vector states and learned latent evolution."""
import torch
from torch import nn


class VectorField(nn.Module):
    """Invariant coefficients mix atom vectors and smooth radial neighbor sums."""
    def __init__(self, channels, width):
        super().__init__(); self.channels=channels
        self.register_buffer('centers',torch.linspace(.25,1.1,4))
        self.net=nn.Sequential(nn.Linear(channels*(channels+1)//2+12,width),nn.SiLU(),
            nn.Linear(width,width),nn.SiLU(),nn.Linear(width,channels*(channels+8)))
        self.register_buffer('ia',torch.triu_indices(channels,channels)[0])
        self.register_buffer('ib',torch.triu_indices(channels,channels)[1])
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self,z,state):
        q,v=state.unbind(-2)
        dq=q[:,:,None]-q[:,None,:]; dv=v[:,:,None]-v[:,None,:]
        r2=dq.square().sum(-1); r=(r2+1e-8).sqrt()
        mask=1-torch.eye(q.shape[1],device=q.device,dtype=q.dtype)
        envelope=(1-(r/1.2).square()).clamp_min(0).pow(3)*mask
        w=torch.exp(-.5*((r[...,None]-self.centers)/.18).square())*envelope[...,None]/12
        aq=torch.einsum('bijr,bijc->birc',w,dq)
        av=torch.einsum('bijr,bijc->birc',w,dv)
        gram=z@z.transpose(-1,-2)
        features=torch.cat((gram[...,self.ia,self.ib],w.sum(2),
                            (aq*q[:,:,None]).sum(-1),(av*v[:,:,None]).sum(-1)),-1)
        weights=self.net(torch.asinh(features)).reshape(*z.shape[:2],self.channels,self.channels+8)
        vectors=torch.cat((z,aq,av),-2)
        return .5*torch.tanh(weights)@vectors


class PhaseProcess(nn.Module):
    def __init__(self,c,arm):
        super().__init__(); self.latent=arm['model']=='latent'
        channels=c['model']['channels'] if self.latent else 2
        self.field=VectorField(channels,c['model']['width'])
        if self.latent:
            self.lift=nn.Linear(2,channels,bias=False); self.readout=nn.Linear(channels,2,bias=False)
            self.encoder=VectorField(channels,c['model']['width'])
            with torch.no_grad():
                self.lift.weight.zero_(); self.lift.weight[:2]=torch.eye(2)
                self.lift.weight[2:]=torch.randn(channels-2,2)*.1
                self.readout.weight.zero_(); self.readout.weight[:,:2]=torch.eye(2)

    def linear_encode(self,h):
        return self.lift(h.transpose(-1,-2)).transpose(-1,-2) if self.latent else h

    def encode(self,h):
        h=h.reshape(*h.shape[:-1],2,3)
        z=self.linear_encode(h)
        return z+self.encoder(z,h) if self.latent else z

    def decode(self,z):
        h=self.readout(z.transpose(-1,-2)).transpose(-1,-2) if self.latent else z
        # Both position and velocity are relative to the fixed central atom.
        return h-h[:,:1]

    def step(self,z):
        h=self.decode(z)
        ballistic=torch.stack((.01*h[...,1,:],torch.zeros_like(h[...,1,:])),-2)
        return z+self.linear_encode(ballistic)+.05*self.field(z,h)

    def forward(self,h,steps,*,latents=False):
        z=self.encode(h); decoded0=self.decode(z); anchor=h.reshape(-1,h.shape[-2],2,3)
        path=[]; codes=[]
        for _ in range(steps):
            z=self.step(z)
            # Decode changes from the actual initial observation exactly.
            path.append(anchor+self.decode(z)-decoded0); codes.append(z)
        value=torch.stack(path,1).flatten(-2)
        return (value,torch.stack(codes,1)) if latents else value


def ballistic(h,steps):
    time=torch.arange(1,steps+1,device=h.device,dtype=h.dtype)*.01
    q=h[:,None,:,:3]+time[None,:,None,None]*h[:,None,:,3:]
    return torch.cat((q,h[:,None,:,3:].expand(-1,steps,-1,-1)),-1)
