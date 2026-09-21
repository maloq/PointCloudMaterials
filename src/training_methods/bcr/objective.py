"""Primary objective: equal-environment, boundary-weighted noise MSE only."""
import torch
from .data import taper


def per_environment(prediction,epsilon,clean,radius,region='weighted'):
    r=clean['positions'].norm(dim=-1);valid=clean['mask']&~clean['center']
    if region=='weighted':w=taper(r,radius)*valid
    elif region=='interior':w=valid*(r<=.65*radius)
    elif region=='middle':w=valid*(r>.65*radius)*(r<=.8*radius)
    elif region=='outer':w=valid*(r>.8*radius)
    elif region=='unweighted':w=valid.to(r.dtype)
    else:raise ValueError(region)
    count=w.sum(-1)
    if region in ('weighted','unweighted') and (count<=0).any():raise ValueError('Noise loss requires noncentral support')
    error=((prediction-epsilon).square().sum(-1)*w).sum(-1)/(3*count.clamp_min(1e-12))
    return torch.where(count>0,error,torch.nan)


def vicreg(a,b):
    """Separate matched reference, not a hidden BCR auxiliary objective."""
    if len(a)<2:raise ValueError('VICReg requires a complete statistical batch >=2')
    variance=sum(torch.relu(1-torch.sqrt(z.var(0)+1e-4)).mean() for z in (a,b))
    cov=0.
    for z in (a,b):
        c=(z-z.mean(0)).T@(z-z.mean(0))/(len(z)-1)
        cov=cov+(c.square().sum()-c.diag().square().sum())/z.shape[1]
    return 25*(a-b).square().mean()+25*variance+cov
