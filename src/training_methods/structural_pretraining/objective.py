"""Explicit physical blocks and full-within-material-batch representation losses."""
import torch
from torch import nn
from lejepa.multivariate import SlicingUnivariateTest
from lejepa.univariate import EppsPulley

PHYSICAL_BLOCKS={'radial':(0,32),'pair':(32,64),'angular':(64,80),'moments':(80,85)}
TDA_BLOCKS={'h0':(0,16),'h1':(16,80),'h2':(80,144)}


def block_errors(prediction,target,blocks):
    return torch.stack([(prediction[:,a:b]-target[:,a:b]).square().mean(-1) for a,b in blocks.values()],-1)


def vicreg(q0,q1):
    if q0.shape!=q1.shape or len(q0)<2:raise ValueError('VICReg requires two equal nontrivial batches')
    invariance=(q0-q1).square().mean()
    variance=[]; covariance=[]
    for q in (q0,q1):
        centered=q-q.mean(0)
        variance.append(torch.relu(1-torch.sqrt(centered.square().sum(0)/(len(q)-1)+1e-4)).mean())
        cov=centered.T@centered/(len(q)-1)
        off=cov-torch.diag_embed(cov.diagonal())
        covariance.append(off.square().sum()/q.shape[1])
    var=torch.stack(variance).mean();cov=torch.stack(covariance).mean()
    return (25*invariance+25*var+cov)/51,dict(invariance=invariance,variance=var,covariance=cov)


class Objective(nn.Module):
    def __init__(self,normalization,method):
        super().__init__()
        if method not in ('vicreg','lejepa'):raise ValueError(f'Unknown structural method: {method}')
        self.method=method
        for name in ('physical','tda'):
            for field in ('mean','std'):
                self.register_buffer(f'{name}_{field}',torch.tensor(normalization[name][field],dtype=torch.float32))
        self.sigreg=SlicingUnivariateTest(EppsPulley(t_max=3.,n_points=17,integration='trapezoid'),num_slices=256)

    def physical_errors(self,heads,targets):
        physical=(targets['physical']-self.physical_mean)/self.physical_std
        tda=(targets['tda']-self.tda_mean)/self.tda_std
        return block_errors(heads['physical'],physical,PHYSICAL_BLOCKS),block_errors(heads['tda'],tda,TDA_BLOCKS)

    def forward(self,model,z,targets,temporal,delta):
        heads=model.heads(z);p,h=self.physical_errors(heads,targets)
        mask=targets['tda_valid']; present=p.mean()
        hot=h[mask].mean() if bool(mask.any()) else h.sum()*0
        q0,q1=heads['q'].chunk(2);terms={}
        if self.method=='vicreg':reg,terms=vicreg(q0,q1)
        else:
            sig=(self.sigreg(q0)+self.sigreg(q1))/2
            pred=model.predictor(torch.cat((q0,delta[:,None]),-1))
            forecast=(pred-q1).square().mean() if temporal else pred.sum()*0
            reg=.95*forecast+.05*sig
            terms=dict(next_latent_mse=forecast,sigreg=sig,latent_persistence=(q0-q1).square().mean())
        loss=present+.25*hot+.1*reg
        return loss,dict(loss=loss,physical=present,instantaneous_tda=hot,representation=reg,
                         labelled_views=mask.sum(),**terms)
