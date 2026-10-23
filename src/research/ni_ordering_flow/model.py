"""Joint measured-path flow conditioned by an observed-history attention encoder."""
import math
import torch
from torch import nn
from src.research.ni_latent_flow.model import BlockFlow, _ParallelAttention, _noise_like


class History(nn.Module):
    def __init__(self,width,heads,bottleneck):
        super().__init__();self.bottleneck=bottleneck
        self.projection=nn.Linear(331,width)
        self.norms=nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.layers=nn.ModuleList([_ParallelAttention(width,heads) for _ in range(2)])
        self.final=nn.LayerNorm(width)
    def forward(self,x):
        z=self.projection(x)
        for norm,layer in zip(self.norms,self.layers):z=z+layer(norm(z))
        z=self.final(z)
        if self.bottleneck:z=z.mean(1,keepdim=True).expand_as(z)
        return z


class Model(BlockFlow):
    def __init__(self,c,arm):
        # Reuse the maintained flow interpolant and prediction network. Only its
        # observed-context producer changes; no physical-time covariates enter.
        super().__init__(244,dict(c['flow'],history=4,history_noise=0.))
        self.history=arm['history_frames'];self.arm=arm
        self.conditioning_mask=torch.cat((torch.ones(self.history),torch.zeros(self.horizon))).long()
        width=c['flow']['width']
        self.encoder=History(width,c['flow']['heads'],arm['bottleneck'])
        self.condition_projection=nn.Linear(width,width)
        self.dense_head=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,20*91)) if arm['teacher']!='none' else None

    def context(self,history):
        z=self.encoder(history)
        return torch.cat((z,z.mean(1,keepdim=True).expand(-1,self.horizon,-1)),1)

    def features(self,history):
        return self.encoder(history).mean(1)

    def objective(self,history,target,weights):
        clean=(torch.cat((history[...,:244],target),1)-self.location)/self.scale
        noise=torch.randn_like(clean);t=.001+.998*torch.rand(len(history),device=history.device)
        angle=.5*math.pi*t[:,None,None]
        predicted=self._prediction(angle.sin()*clean+angle.cos()*noise,t,self.context(history))
        main=(predicted-clean).square().mean()
        future=self.location+self.scale*predicted[:,self.history:]
        auxiliary=((future-target).square()*weights).sum(-1).mean()
        return main+.25*auxiliary,dict(flow=main,decoded=auxiliary)

    def dense(self,current):
        # Physical teachers expose only the91 smooth observables. BO/TDA and
        # regional/history features are absent, not held at inconsistent values
        # under a physical perturbation. The observation encoder is shared.
        z=torch.cat((torch.zeros_like(current[...,:153]),current[...,153:],
                     current.new_zeros(len(current),87)),-1)[:,None]
        delta=self.dense_head(self.encoder(z)[:,0]).reshape(-1,20,91)
        return current[:,None,153:]+delta

    def teacher_objective(self,t,indices,response):
        current=t['value'][indices,0];target=t['value'][indices,1:,153:]
        prediction=self.dense(current)
        value=(prediction-target).square().mean();loss=value
        scores=dict(teacher_value=value)
        if response:
            direction=int(torch.randint(2,(),device=current.device))
            tangent=torch.zeros_like(current);tangent[:,153:]=t['response'][indices,0,:,direction]
            # The math attention backend supports second derivatives needed to
            # optimize the exact network directional derivative.
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel(SDPBackend.MATH):
                _,jvp=torch.autograd.functional.jvp(self.dense,current,tangent,create_graph=True)
            derivative=((jvp-t['response'][indices,1:,:,direction]).square()/t['power']).mean()
            loss=loss+.25*derivative;scores['teacher_response']=derivative
        return .1*loss,scores

    @torch.no_grad()
    def rollout(self,history,steps,particles,seed,deterministic=False):
        expanded=history[:,None].expand(-1,particles,-1,-1).flatten(0,1)
        context=self.context(expanded)
        generator=torch.Generator(device=history.device).manual_seed(seed)
        generated=history.new_zeros(len(expanded),self.history+self.horizon,244)
        if not deterministic:generated=_noise_like(generated,generator)
        generated=self._integrate(generated,context)
        result=self.location+self.scale*generated[:,self.history:self.history+steps]
        return result.reshape(len(history),particles,steps,244)
