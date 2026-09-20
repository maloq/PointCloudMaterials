"""Forecast refinements with explicit probabilistic training and stable diffusion."""
import torch
from torch import nn
from torch.nn import functional as F
from .data import STATE_DIM,STEPS,SUBSTEPS
from .model import Forecaster,block_loss,event_nll,cdf_from_logits,sampled_onset_cdf


def project_onset(paths):
    """Least-squares projection onto 129 valid absorbing step curves, including censoring."""
    x=paths.flatten(-2)
    suffix=x.flip(-1).cumsum(-1).flip(-1)
    score=torch.cat((suffix,torch.zeros_like(suffix[...,:1])),-1)
    onset=128-score.flip(-1).argmax(-1)  # Equal-energy ties prefer the latest onset/censoring.
    return (torch.arange(128,device=x.device)>=onset[...,None]).float().mean(1)


def clean_and_noise(x,prediction,alpha,kind):
    if kind=='v':
        return alpha.sqrt()*x-(1-alpha).sqrt()*prediction,(1-alpha).sqrt()*x+alpha.sqrt()*prediction
    if kind=='x0':return prediction,(x-alpha.sqrt()*prediction)/(1-alpha).sqrt()
    raise ValueError(kind)


class RefinedForecaster(Forecaster):
    def __init__(self,spec):
        super().__init__(spec);width=spec['head_width'];p=spec['dropout']
        self.register_buffer('target_mean',torch.zeros(STATE_DIM));self.register_buffer('target_scale',torch.ones(STATE_DIM))
        self.context.token.add_module('dropout',nn.Dropout(p));self.context.output.add_module('dropout',nn.Dropout(p))
        for block in [*self.context.spatial,*self.context.temporal]:block.ff.add_module('dropout',nn.Dropout(p))
        self.output_dropout=nn.Dropout(p)
        if spec['motion_input']:self.motion=nn.Sequential(nn.Linear(43,width),nn.LayerNorm(width),nn.SiLU())
        if hasattr(self,'decoder'):
            for layer in self.decoder.layers:
                layer.dropout.p=p;layer.dropout1.p=p;layer.dropout2.p=p;layer.self_attn.dropout=p
        self.rank=spec['gaussian_rank'] if self.method=='ar_gaussian' else 0
        if self.rank:
            self.decode=nn.Linear(width,2*STATE_DIM+STATE_DIM*self.rank+SUBSTEPS)
            nn.init.zeros_(self.decode.weight[2*STATE_DIM:-SUBSTEPS]);nn.init.normal_(self.decode.bias[2*STATE_DIM:-SUBSTEPS],std=.01)
        if self.method=='mixture' and spec['mixture_style']=='stratified':
            boundaries=[0,*spec['mixture_boundaries'],129];count=len(boundaries)-1
            self.components=nn.Parameter(torch.randn(count,width)*.02);self.mixing=nn.Linear(width,count)
            self.event_decoder=nn.Linear(width,129)
            self.register_buffer('boundaries',torch.tensor(spec['mixture_boundaries']))
            t=torch.arange(129)
            self.register_buffer('event_support',torch.stack([(t>=a)&(t<b) for a,b in zip(boundaries[:-1],boundaries[1:])]))
        if self.method=='diffusion':
            # Exact zero terminal SNR, keeping the original first noise level.
            a=self.alpha_bar.sqrt();a=(a-a[-1])*a[0]/(a[0]-a[-1]);self.alpha_bar.copy_(a.square())
            self.noise_skip=nn.Linear(STATE_DIM+SUBSTEPS,STATE_DIM+SUBSTEPS,bias=False)
            nn.init.zeros_(self.noise_skip.weight)
            if spec['diffusion_prediction']=='x0':nn.init.eye_(self.noise_skip.weight)

    def encode(self,observed):
        context=super().encode({k:v for k,v in observed.items() if k!='motion'})
        if self.spec['motion_input']:context=context+self.motion(observed['motion'])
        return context

    def anchor(self,observed,context):
        value=self.initial(context)
        nodes=1 if self.spec['radius_A']==0 else 7
        z=(observed['features'][:,-nodes,:128]-self.target_mean[:128])/self.target_scale[:128]
        # Eligible origins are liquid. The motion/geometry packet is decoded, never supplied.
        liquid=(-self.target_mean[264]/self.target_scale[264]).expand(len(z),1)
        return torch.cat((z,value[:,128:264],liquid),-1)

    def recurrent_refined(self,context,anchor,target=None,teacher_probability=0.,stochastic=False):
        if target is not None and not self.training:raise ValueError('Future targets forbidden in evaluation rollout')
        previous=anchor if self.spec['residual_anchor'] else self.initial(context)
        hidden=context;means=[];scales=[];factors=[];hazards=[];paths=[]
        for t in range(STEPS):
            hidden=self.cell(torch.cat((previous,context+self.position[t]),-1),hidden)
            out=self.decode(self.output_dropout(hidden));mean=out[:,:STATE_DIM]
            if self.spec['residual_anchor']:mean=mean+anchor
            logstd=out[:,STATE_DIM:2*STATE_DIM].clamp(-5,2) if self.method=='ar_gaussian' else torch.zeros_like(mean)
            factor=out[:,2*STATE_DIM:-SUBSTEPS].reshape(len(mean),STATE_DIM,self.rank) if self.rank else mean.new_zeros(len(mean),STATE_DIM,0)
            value=mean
            if stochastic:
                value=mean+logstd.exp()*torch.randn_like(mean)
                if self.rank:value=value+(factor@torch.randn(len(mean),self.rank,1,device=mean.device)).squeeze(-1)
            means.append(mean);scales.append(logstd);factors.append(factor);hazards.append(out[:,-SUBSTEPS:]);paths.append(value)
            previous=value
            if target is not None and teacher_probability>0:
                previous=torch.where(torch.rand(len(mean),1,device=mean.device)<teacher_probability,target[:,t],previous)
        return tuple(torch.stack(x,1) for x in (means,scales,factors,hazards,paths))

    def stratified(self,context,anchor):
        count=len(self.components)
        x=(context[:,None,None]+self.position[None,None]+self.components[None,:,None]).flatten(0,1)
        hidden=self.decoder(x).reshape(len(context),count,STEPS,-1);out=self.decode(hidden)
        mean=out[...,:STATE_DIM]
        if self.spec['residual_anchor']:mean=mean+anchor[:,None,None]
        mass=self.event_decoder(hidden.mean(2)).masked_fill(~self.event_support[None],float('-inf')).log_softmax(-1)
        return mean,out[...,STATE_DIM:2*STATE_DIM].clamp(-5,2),mass

    def denoise(self,x,level,context):
        return super().denoise(x,level,context)+self.alpha_bar[level,None,None].sqrt()*self.noise_skip(x)

    def loss(self,observed,target,teacher_probability):
        context=self.encode(observed);anchor=self.anchor(observed,context);actual=target['state'];event=target['event']
        present=self.spec['present_weight']*block_loss((anchor[:,None]-target['present'][:,None]).square())
        sw=self.spec['state_weight']
        if self.method=='diffusion':
            clean=torch.cat((actual,2*target['occurred']-1),-1);noise=torch.randn_like(clean)
            level=torch.randint(64,(len(clean),),device=clean.device)
            level=torch.where(torch.rand(len(clean),device=clean.device)<self.spec['terminal_fraction'],63,level)
            a=self.alpha_bar[level,None,None];x=a.sqrt()*clean+(1-a).sqrt()*noise
            expected=a.sqrt()*noise-(1-a).sqrt()*clean if self.spec['diffusion_prediction']=='v' else clean
            error=(self.denoise(x,level,context)-expected).square()
            return sw*block_loss(error[...,:STATE_DIM])+self.spec['event_weight']*error[...,-SUBSTEPS:].mean((-1,-2))
        if self.method=='direct':
            mean,hazard=self.parallel(context)
            if self.spec['residual_anchor']:mean=mean+anchor[:,None]
            return sw*block_loss((mean-actual).square())+event_nll(hazard,event)+present
        if self.method=='mixture':
            if self.spec['mixture_style']=='stratified':
                mean,logstd,logmass=self.stratified(context,anchor);category=torch.bucketize(event,self.boundaries,right=True)
                row=torch.arange(len(context),device=context.device);mean=mean[row,category];logstd=logstd[row,category]
                state=block_loss(.5*((mean-actual)*(-logstd).exp()).square()+logstd)
                return sw*state-self.mixing(context).log_softmax(-1)[row,category]-logmass[row,category,event]+present
            mean,logstd,hazard=self.parallel(context)
            if self.spec['residual_anchor']:mean=mean+anchor[:,None,None]
            energy=sw*block_loss(.5*((mean-actual[:,None])*(-logstd).exp()).square()+logstd)+event_nll(hazard,event[:,None])
            return -torch.logsumexp(self.mixing(context).log_softmax(-1)-energy,1)+present
        mean,logstd,factor,hazard,_=self.recurrent_refined(context,anchor,actual,teacher_probability)
        if self.method=='ar_mse':state=block_loss((mean-actual).square())
        elif self.rank:
            distribution=torch.distributions.LowRankMultivariateNormal(mean,factor,(2*logstd).exp())
            state=-distribution.log_prob(actual).mean(-1)/STATE_DIM
        else:state=(.5*((mean-actual)*(-logstd).exp()).square()+logstd).mean((-1,-2))
        return sw*state+event_nll(hazard,event)+present

    @torch.no_grad()
    def forecast(self,observed,samples=32,diffusion_steps=16):
        if self.training:raise ValueError('Forecast must run in evaluation mode')
        context=self.encode(observed);anchor=self.anchor(observed,context);batch=len(context)
        if self.method=='direct':
            mean,hazard=self.parallel(context)
            if self.spec['residual_anchor']:mean=mean+anchor[:,None]
            return mean[:,None],cdf_from_logits(hazard)
        if self.method=='mixture':
            if self.spec['mixture_style']=='stratified':
                mean,logstd,logmass=self.stratified(context,anchor)
                mix=self.mixing(context).softmax(-1);cdf=(logmass.exp()*mix[...,None]).sum(1).cumsum(-1)[:,:128].clamp(max=1)
            else:
                mean,logstd,hazard=self.parallel(context)
                if self.spec['residual_anchor']:mean=mean+anchor[:,None,None]
                mix=self.mixing(context).softmax(-1);cdf=(cdf_from_logits(hazard)*mix[...,None]).sum(1).clamp(max=1)
            choice=torch.multinomial(mix,samples,replacement=True);row=torch.arange(batch,device=context.device)[:,None]
            return mean[row,choice]+logstd[row,choice].exp()*torch.randn(batch,samples,STEPS,STATE_DIM,device=context.device),cdf
        if self.method in ('ar_mse','ar_gaussian'):
            count=1 if self.method=='ar_mse' else samples
            _,_,_,hazard,path=self.recurrent_refined(context.repeat_interleave(count,0),anchor.repeat_interleave(count,0),stochastic=self.method=='ar_gaussian')
            return path.reshape(batch,count,STEPS,STATE_DIM),cdf_from_logits(hazard).reshape(batch,count,128).mean(1)
        context=context.repeat_interleave(samples,0)
        x=torch.randn(len(context),STEPS,STATE_DIM+SUBSTEPS,device=context.device)
        schedule=torch.linspace(63,0,diffusion_steps,device=x.device).round().long()
        for i,t in enumerate(schedule):
            prediction=self.denoise(x,t.expand(len(x)),context)
            clean,noise=clean_and_noise(x,prediction,self.alpha_bar[t],self.spec['diffusion_prediction'])
            a=self.alpha_bar[schedule[i+1]] if i+1<len(schedule) else x.new_ones(())
            x=a.sqrt()*clean+(1-a).sqrt()*noise
        paths=x.reshape(batch,samples,STEPS,-1)
        cdf=project_onset(paths[...,-SUBSTEPS:]) if self.spec['onset_projection']=='step' else sampled_onset_cdf(paths[...,-SUBSTEPS:])
        prior=x.new_tensor(self.spec['training_event_cdf'])
        return paths[...,:STATE_DIM],(samples*cdf+prior)/(samples+1)
