"""Compact stochastic recurrence: physical time per transition is 0.1 ps."""
import math
import torch
from torch import nn


def mlp(a,b,width=256):
    return nn.Sequential(nn.Linear(a,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,b))


def make_model(arm):
    if 'neural_step_fs' in arm:
        from .fine_model import FineLatent
        return FineLatent(arm)
    return Latent(arm)


class Latent(nn.Module):
    def __init__(self,arm):
        super().__init__();self.d=arm['dimension'];self.memory=arm['history']
        self.encoder=mlp(280,self.d);self.decoder=mlp(self.d,560)
        self.transition=mlp(self.d*self.memory,2*self.d)
        nn.init.normal_(self.transition[-1].weight,std=.001);nn.init.zeros_(self.transition[-1].bias)
        with torch.no_grad():self.transition[-1].bias[self.d:]=-3

    def encode(self,x):return self.encoder(x)

    def emit(self,z):
        mean,raw=self.decoder(z).chunk(2,-1)
        return mean,-3+5*torch.tanh(raw/5)

    def step(self,history,noise):
        delta,raw=self.transition(history.flatten(-2)).chunk(2,-1)
        z=history[...,-1,:]+.1*delta+torch.nn.functional.softplus(raw)*noise
        return torch.cat((history[...,1:,:],z.unsqueeze(-2)),-2)

    def rollout(self,x,steps,particles,generator):
        z=self.encode(x[:,-self.memory:]);hist=z[:,None].expand(-1,particles,-1,-1)
        outputs=[]
        for _ in range(steps):
            noise=torch.randn(*hist.shape[:2],self.d,device=x.device,generator=generator)
            hist=self.step(hist,noise);outputs.append(self.emit(hist[:,:,-1]))
        return torch.stack([v[0] for v in outputs],2),torch.stack([v[1] for v in outputs],2)


def nll(mean,logvar,target):
    logp=-.5*((target[:,None]-mean).square()*(-logvar).exp()+logvar+math.log(2*math.pi)).sum(-1)
    return -(torch.logsumexp(logp,dim=1)-math.log(mean.shape[1]))/mean.shape[-1]
