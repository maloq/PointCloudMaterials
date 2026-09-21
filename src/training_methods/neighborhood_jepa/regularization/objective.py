"""SIGReg, VICReg variance/covariance, and scale-controlled EpiJEPA adaptation."""
import math
import torch
from torch import nn
from ..v2.objective import Objective as BaseObjective,SIGReg


def vicreg_regularization(z):
    centered=z-z.mean(0)
    covariance=centered.T@centered/(len(z)-1)
    variance=torch.relu(1-torch.sqrt(torch.diagonal(covariance)+1e-4)).mean()
    off=covariance-torch.diag_embed(torch.diagonal(covariance))
    decorrelation=off.square().sum()/z.shape[1]
    return 25*variance+decorrelation,variance,decorrelation


def ridge_map(h,rho=3.):
    with torch.no_grad():
        h=h.double();h=(h-h.mean(0))/h.std(0,unbiased=False).clamp_min(1e-6)/math.sqrt(h.shape[1])
        eye=torch.eye(h.shape[1],device=h.device,dtype=h.dtype)
        q,r=torch.linalg.qr(torch.cat((h,math.sqrt(rho)*eye)),mode='reduced')
        return torch.linalg.solve_triangular(r,q[:len(h)].T,upper=True)


def epiplexity(z,h,normalize=True):
    """Blog's ridge/logdet score; common RMS normalization removes scale inflation.

    Reservoir is frozen random MACE, not an image CNN. This is an adaptation,
    not a reproduction of the blog's image-view alignment experiment.
    """
    centered=z.double()-z.double().mean(0)
    if normalize:centered=centered/torch.sqrt(centered.square().mean()+1e-4)
    w=ridge_map(h)@centered
    # Sylvester determinant identity evaluates the smaller reservoir-side matrix.
    matrix=torch.eye(w.shape[0],device=w.device,dtype=w.dtype)+30*(w@w.T)
    factor=torch.linalg.cholesky(matrix)
    return (torch.log(torch.diagonal(factor)).sum()/math.log(2)).float()


class ZeroRegularizer(nn.Module):
    def forward(self,z):
        zero=z.sum()*0
        return zero,zero,zero


class Objective(BaseObjective):
    def __init__(self,manifest,order_manifest,spec):
        super().__init__(manifest,spec)
        if spec.get('regularizer_scope','global') not in ('global','temperature'):raise ValueError('Unknown regularizer scope')
        self.sigreg=ZeroRegularizer()
        self.gaussian=SIGReg('per_sample_discrepancy')
        self.register_buffer('order_mean',torch.tensor(order_manifest['mean'],dtype=torch.float32))
        self.register_buffer('order_std',torch.tensor(order_manifest['std'],dtype=torch.float32))
        self.register_buffer('epi_initial_scale',torch.tensor(1.))

    def forward(self,model,encoded,target):
        _,terms=super().forward(model,encoded,target)
        z=encoded.reshape(len(target['group']),len(self.plan.views),-1)
        current=z[:,self.plan.slot(1,0),:128]
        slots=[self.plan.slot(t,0) for t in (1,2)]
        prediction=model.order_decoder(z[:,slots,:128])
        order=((prediction-(target['order']-self.order_mean)/self.order_std).square()).mean()
        projected=model.projector(current)
        mode=self.spec['regularizer']
        extra={}
        if self.spec.get('regularizer_scope','global')=='temperature':
            penalties=[];variables=[];covariances=[]
            for temperature in torch.unique(target['temperature_K']):
                values=projected[target['temperature_K']==temperature]
                if len(values)<2:raise ValueError('Conditional regularization requires >=2 anchors per sampled temperature')
                if mode=='sigreg':p,_,_=self.gaussian(values)
                elif mode=='vicreg':
                    p,var,cov=vicreg_regularization(values);variables.append(var);covariances.append(cov)
                else:raise ValueError(f'Conditional regularizer unsupported: {mode}')
                penalties.append(p)
            penalty=torch.stack(penalties).mean()
            if mode=='vicreg':extra=dict(vicreg_variance=float(torch.stack(variables).mean().detach()),vicreg_covariance=float(torch.stack(covariances).mean().detach()),vicreg_total=float(penalty.detach()))
            else:extra=dict(sigreg_discrepancy=float(penalty.detach()))
        elif mode=='sigreg':
            penalty,raw,discrepancy=self.gaussian(projected)
            extra=dict(sigreg_raw=float(raw.detach()),sigreg_discrepancy=float(discrepancy.detach()))
        elif mode=='vicreg':
            penalty,var,cov=vicreg_regularization(projected)
            extra=dict(vicreg_variance=float(var.detach()),vicreg_covariance=float(cov.detach()),vicreg_total=float(penalty.detach()))
        elif mode=='epi':
            score=epiplexity(projected,target['reservoir'][:,0])
            penalty=-score/self.epi_initial_scale
            extra=dict(epi_score=float(score.detach()),epi_initial_scale=float(self.epi_initial_scale))
        elif mode=='none':penalty=current.sum()*0
        else:raise ValueError(mode)
        terms['order']=self.spec['order_weight']*order
        terms['regularizer']=self.spec['regularizer_weight']*penalty
        self.diagnostics.update(extra,order_unweighted=float(order.detach()),
                                projected_rms=float(projected.detach().square().mean().sqrt()))
        return sum(terms.values()),terms
