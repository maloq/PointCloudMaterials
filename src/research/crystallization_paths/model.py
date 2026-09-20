"""Direct, recurrent, mixture and conditional denoising structural-path predictors."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from src.research.crystallization_transfer.attention import AdaptiveContextHead
from .data import STATE_DIM, STEPS, SUBSTEPS


def block_loss(error):
    """Retain physical anchors without allowing 128 latent channels to dominate."""
    return (.25*error[...,:128].mean((-1,-2))+error[...,128:256].mean((-1,-2))
            +error[...,256:264].mean((-1,-2))+error[...,264].mean(-1))/3.25


def event_nll(logits,event):
    logits=logits.flatten(-2).float();bins=torch.arange(128,device=logits.device)
    # Event==128 is right censoring at 96 ps, not a 129th observed event.
    return (F.softplus(logits)*(bins<event[...,None])).sum(-1)+(F.softplus(-logits)*(bins==event[...,None])).sum(-1)


def cdf_from_logits(logits):
    return -torch.expm1(F.logsigmoid(-logits.float().flatten(-2)).cumsum(-1))


def sampled_onset_cdf(event_paths):
    """Project each generated absorbing indicator path by its first threshold crossing."""
    return (event_paths.flatten(-2)>0).cummax(-1).values.float().mean(1)


class Forecaster(nn.Module):
    def __init__(self,spec):
        super().__init__();self.spec=spec;self.method=spec['method'];width=spec['head_width']
        self.context=AdaptiveContextHead(spec)
        self.context.output=nn.Sequential(nn.Linear(2*width+7,width),nn.LayerNorm(width),nn.SiLU())
        self.position=nn.Parameter(torch.randn(STEPS,width)*.02)
        self.initial=nn.Linear(width,STATE_DIM)
        if self.method in ('ar_mse','ar_gaussian'):
            self.cell=nn.GRUCell(STATE_DIM+width,width)
            self.decode=nn.Linear(width,STATE_DIM*(2 if self.method=='ar_gaussian' else 1)+SUBSTEPS)
        else:
            layer=nn.TransformerEncoderLayer(width,spec['heads'],2*width,dropout=0.,batch_first=True,norm_first=True)
            self.decoder=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
            self.decode=nn.Linear(width,(STATE_DIM+SUBSTEPS) if self.method in ('direct','diffusion') else 2*STATE_DIM+SUBSTEPS)
        if self.method=='mixture':
            self.components=nn.Parameter(torch.randn(4,width)*.02);self.mixing=nn.Linear(width,4)
        if self.method=='diffusion':
            self.noisy=nn.Linear(STATE_DIM+SUBSTEPS,width)
            self.noise_time=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
            time=torch.linspace(0,1,65,dtype=torch.float64)
            alpha=torch.cos((time+.008)/1.008*math.pi/2).square();alpha=alpha/alpha[0]
            beta=(1-alpha[1:]/alpha[:-1]).clamp(max=.999)
            self.register_buffer('alpha_bar',(1-beta).cumprod(0).float())
        elif self.method not in ('direct','ar_mse','ar_gaussian','mixture'):raise ValueError(self.method)

    def encode(self,observed):
        # Frozen MACE: one fixed set of training-only moments for training and inference.
        self.context.normalization.eval()
        return self.context(**observed)

    def parallel(self,context):
        x=context[:,None]+self.position
        if self.method=='mixture':
            x=(x[:,None]+self.components[None,:,None]).flatten(0,1)
        out=self.decode(self.decoder(x))
        if self.method=='mixture':
            out=out.reshape(len(context),4,STEPS,-1)
            return out[...,:STATE_DIM],out[...,STATE_DIM:2*STATE_DIM].clamp(-5,2),out[...,-SUBSTEPS:]
        return out[...,:STATE_DIM],out[...,-SUBSTEPS:]

    def recurrent(self,context,target=None,teacher_probability=0.,stochastic=False):
        if target is not None and not self.training:raise ValueError('Future targets forbidden in evaluation rollout')
        previous=self.initial(context);hidden=context;means=[];scales=[];hazards=[];path=[]
        for t in range(STEPS):
            hidden=self.cell(torch.cat((previous,context+self.position[t]),-1),hidden)
            out=self.decode(hidden);mean=out[:,:STATE_DIM];hazards.append(out[:,-SUBSTEPS:])
            logstd=out[:,STATE_DIM:2*STATE_DIM].clamp(-5,2) if self.method=='ar_gaussian' else torch.zeros_like(mean)
            value=mean+torch.exp(logstd)*torch.randn_like(mean) if stochastic else mean
            means.append(mean);scales.append(logstd);path.append(value)
            previous=value
            if target is not None and teacher_probability>0:
                use=torch.rand(len(context),1,device=context.device)<teacher_probability
                previous=torch.where(use,target[:,t],previous)
        return torch.stack(means,1),torch.stack(scales,1),torch.stack(hazards,1),torch.stack(path,1)

    def denoise(self,x,level,context):
        half=self.spec['head_width']//2
        freq=torch.exp(torch.arange(half,device=x.device)*(-math.log(10000)/max(half-1,1)))
        angle=level[:,None].float()*freq[None]
        time=self.noise_time(torch.cat((angle.sin(),angle.cos()),-1))
        return self.decode(self.decoder(self.noisy(x)+self.position+context[:,None]+time[:,None]))

    def loss(self,observed,target,teacher_probability):
        context=self.encode(observed);actual=target['state'];event=target['event']
        if self.method=='diffusion':
            clean=torch.cat((actual,2*target['occurred']-1),-1)
            noise=torch.randn_like(clean);level=torch.randint(len(self.alpha_bar),(len(clean),),device=clean.device)
            a=self.alpha_bar[level,None,None];noisy=a.sqrt()*clean+(1-a).sqrt()*noise
            error=(self.denoise(noisy,level,context)-noise).square()
            return block_loss(error[...,:STATE_DIM])+error[...,-SUBSTEPS:].mean((-1,-2))
        if self.method=='direct':
            mean,hazard=self.parallel(context)
            return block_loss((mean-actual).square())+event_nll(hazard,event)
        if self.method=='mixture':
            mean,logstd,hazard=self.parallel(context)
            energy=block_loss(.5*((mean-actual[:,None])*torch.exp(-logstd)).square()+logstd)
            energy=energy+event_nll(hazard,event[:,None])
            return -torch.logsumexp(F.log_softmax(self.mixing(context),-1)-energy,dim=1)
        mean,logstd,hazard,_=self.recurrent(context,actual,teacher_probability)
        state_error=(mean-actual).square() if self.method=='ar_mse' else .5*((mean-actual)*torch.exp(-logstd)).square()+logstd
        return block_loss(state_error)+event_nll(hazard,event)

    @torch.no_grad()
    def forecast(self,observed,samples=32,diffusion_steps=16):
        if self.training:raise ValueError('Forecast must run in evaluation mode')
        context=self.encode(observed);batch=len(context)
        if self.method=='direct':
            mean,hazard=self.parallel(context);return mean[:,None],cdf_from_logits(hazard)
        if self.method=='mixture':
            mean,logstd,hazard=self.parallel(context);mix=F.softmax(self.mixing(context),-1)
            choice=torch.multinomial(mix,samples,replacement=True);row=torch.arange(batch,device=context.device)[:,None]
            path=mean[row,choice]+torch.exp(logstd[row,choice])*torch.randn((batch,samples,STEPS,STATE_DIM),device=context.device)
            return path,(cdf_from_logits(hazard)*mix[...,None]).sum(1)
        if self.method in ('ar_mse','ar_gaussian'):
            count=1 if self.method=='ar_mse' else samples
            _,_,hazard,path=self.recurrent(context.repeat_interleave(count,0),stochastic=self.method=='ar_gaussian')
            return path.reshape(batch,count,STEPS,STATE_DIM),cdf_from_logits(hazard).reshape(batch,count,128).mean(1)
        # DDIM eta=0: randomness comes from independent initial noise, not future observations.
        context=context.repeat_interleave(samples,0)
        x=torch.randn((len(context),STEPS,STATE_DIM+SUBSTEPS),device=context.device)
        schedule=torch.linspace(len(self.alpha_bar)-1,0,diffusion_steps,device=x.device).round().long()
        for i,t in enumerate(schedule):
            noise=self.denoise(x,t.expand(len(x)),context);a=self.alpha_bar[t]
            clean=(x-(1-a).sqrt()*noise)/a.sqrt()
            before=self.alpha_bar[schedule[i+1]] if i+1<len(schedule) else torch.ones((),device=x.device)
            x=before.sqrt()*clean+(1-before).sqrt()*noise
        paths=x.reshape(batch,samples,STEPS,-1)
        # Preserve finite support for likelihood comparisons with finite Monte Carlo samples.
        raw=sampled_onset_cdf(paths[...,-SUBSTEPS:])
        prior=observed['features'].new_tensor(self.spec['training_event_cdf'])
        return paths[...,:STATE_DIM],(samples*raw+prior[None])/(samples+1)
