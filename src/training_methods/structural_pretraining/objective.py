"""Explicit physical blocks and full-within-material-batch representation losses."""
import torch
from torch import nn
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley

PHYSICAL_BLOCKS={'radial':(0,32),'pair':(32,64),'angular':(64,80),'moments':(80,85)}
TDA_BLOCKS={'h0':(0,16),'h1':(16,80),'h2':(80,144)}


def block_errors(prediction,target,blocks):
    return torch.stack([(prediction[:,a:b]-target[:,a:b]).square().mean(-1) for a,b in blocks.values()],-1)


def physical_correlation_loss(prediction,target):
    """Train-batch contrast anchor; the original MSE still fixes physical scale.

    Predictions use standardized decoder units, targets the release's physical
    units. Pearson correlation is invariant to positive affine target scaling.
    Exclude target channels with raw within-batch variance <= 1e-6; add 1e-8
    to each variance to make a finite informative gradient at constant outputs.
    No batch statistics enter the exported encoder or inference-time decoder.
    """
    x=prediction.float()-prediction.float().mean(0)
    y=target.float()-target.float().mean(0)
    vx=x.square().mean(0);vy=y.square().mean(0)
    active=vy>1e-6
    corr=(x*y).mean(0)/torch.sqrt((vx+1e-8)*(vy+1e-8))
    return ((1-corr)*active).sum()/active.sum().clamp_min(1)


def vicreg(q0,q1):
    if q0.shape!=q1.shape or len(q0)<2:raise ValueError('VICReg requires two equal nontrivial batches')
    with torch.autocast(q0.device.type,enabled=False):
        q0,q1=q0.float(),q1.float()
        invariance=(q0-q1).square().mean()
        variance=[]; covariance=[];stds=[];ranks=[];violations=[]
        for q in (q0,q1):
            centered=q-q.mean(0)
            std=torch.sqrt(centered.square().sum(0)/(len(q)-1)+1e-4)
            variance.append(torch.relu(1-std).mean())
            cov=centered.T@centered/(len(q)-1)
            off=cov-torch.diag_embed(cov.diagonal())
            covariance.append(off.square().sum()/q.shape[1])
            # Diagnostics use the actual covariance, without the VICReg epsilon.
            with torch.no_grad():
                stds.append(cov.diagonal().clamp_min(0).sqrt().mean())
                ranks.append(cov.trace().square()/cov.square().sum().clamp_min(1e-30))
                violations.append((std<1).float().mean())
        var=torch.stack(variance).mean();cov=torch.stack(covariance).mean()
        total=25*invariance+25*var+cov
    return total/51,dict(invariance=invariance,variance=var,covariance=cov,vicreg=total,
        projector_std=torch.stack(stds).mean(),projector_participation_ratio=torch.stack(ranks).mean(),
        variance_active_fraction=torch.stack(violations).mean())


class Objective(nn.Module):
    def __init__(self,normalization,method,physical_correlation_weight=0.):
        super().__init__()
        if method not in ('vicreg','lejepa'):raise ValueError(f'Unknown structural method: {method}')
        self.method=method
        if physical_correlation_weight<0:raise ValueError('Correlation anchor weight must be nonnegative')
        self.physical_correlation_weight=physical_correlation_weight
        for name in ('physical','tda'):
            for field in ('mean','std'):
                self.register_buffer(f'{name}_{field}',torch.tensor(normalization[name][field],dtype=torch.float32))
        self.sigreg=SlicingUnivariateTest(EppsPulley(t_max=3.,n_points=17,integration='trapezoid'),num_slices=256)

    def physical_errors(self,heads,targets):
        physical=(targets['physical']-self.physical_mean)/self.physical_std
        tda=(targets['tda']-self.tda_mean)/self.tda_std
        return block_errors(heads['physical'],physical,PHYSICAL_BLOCKS),block_errors(heads['tda'],tda,TDA_BLOCKS)

    def forward(self,model,z,targets,temporal,delta):
        heads={k:v.float() for k,v in model.heads(z).items()};p,h=self.physical_errors(heads,targets)
        mask=targets['tda_valid']; present=p.mean()
        hot=h[mask].mean() if bool(mask.any()) else h.sum()*0
        q0,q1=heads['q'].chunk(2);terms={}
        if self.method=='vicreg':
            reg,terms=vicreg(q0,q1)
            terms['vicreg_weighted']=.1*reg
        else:
            sig=(self.sigreg(q0)+self.sigreg(q1))/2
            pred=model.predictor(torch.cat((q0,delta[:,None]),-1))
            forecast=(pred-q1).square().mean() if temporal else pred.sum()*0
            reg=.95*forecast+.05*sig
            terms=dict(next_latent_mse=forecast,sigreg=sig,latent_persistence=(q0-q1).square().mean())
        loss=present+.25*hot+.1*reg
        if self.physical_correlation_weight:
            correlation=physical_correlation_loss(heads['physical'],targets['physical'])
            weighted=self.physical_correlation_weight*correlation
            loss=loss+weighted
            terms.update(physical_correlation_loss=correlation,physical_correlation_weighted=weighted)
        with torch.no_grad():
            terms['state_std']=z.float().std(0).mean()
            terms['physical_prediction_std']=heads['physical'].std(0).mean()
            terms['tda_prediction_std']=heads['tda'].std(0).mean()
        return loss,dict(loss=loss,physical=present,instantaneous_tda=hot,representation=reg,
                         labelled_views=mask.sum(),**terms)
