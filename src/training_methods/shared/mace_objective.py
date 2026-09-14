"""Fixed spatial/temporal VICReg, followed by 80-atom TDA supervision."""
import math
import numpy as np
import torch
from src.training_methods.mace_performance import encode_views, replay_chunks


def variance_covariance(z):
    centered=z-z.mean(0)
    cov=centered.T@centered/(len(z)-1)
    variance=torch.relu(1-torch.sqrt(cov.diagonal()+1e-4)).mean()
    covariance=(cov.square().sum()-cov.diagonal().square().sum())/z.shape[1]
    return variance,covariance


def training_views(settings):
    return [0,1,2,4,5,6] if 'thermal' in settings else [0,1,2]


def objective(model,z,target,temporal_mask,settings,epoch):
    groups=z.reshape(len(z),2 if 'thermal' in settings else 1,3,z.shape[-1])
    anchor,spatial,temporal=groups.unbind(2)
    spatial_mse=(anchor-spatial).square().mean()
    temporal_mse=(anchor[temporal_mask]-temporal[temporal_mask]).square().mean()
    regularizers=[variance_covariance(view) for view in z.unbind(1)]
    variance=torch.stack([r[0] for r in regularizers]).mean()
    covariance=torch.stack([r[1] for r in regularizers]).mean()
    terms=dict(invariance=settings['invariance']*.5*(spatial_mse+temporal_mse),
               variance=settings['variance']*variance,covariance=settings['covariance']*covariance)
    metrics=dict(spatial_mse=spatial_mse,temporal_mse=temporal_mse,variance_penalty=variance,covariance_penalty=covariance)
    if 'thermal' in settings:
        thermal=(z[:,:3]-z[:,3:]).square().mean()
        terms['thermal']=settings['thermal']*thermal;metrics['hot_relaxed_mse']=thermal
    if epoch>=settings['tda_start_epoch']:
        tda=(model.tda(z)-target).square().mean()
        terms['tda']=settings['tda']*tda;metrics['tda_mse']=tda
    loss=sum(terms.values())
    metrics.update({'loss_'+k:v for k,v in terms.items()})
    return loss,metrics


def cached_step(model,batch,mask,cfg,epoch):
    x,target,_,material=batch;views=training_views(cfg['loss']);x=x[:,views];target=target[:,views]
    geometries=[]
    with torch.no_grad():z=encode_views(model,x,material,cfg['microbatch_size'],geometries)
    z.requires_grad_(True)
    loss,parts=objective(model,z,target,mask,cfg['loss'],epoch)
    if not bool(torch.isfinite(loss)):raise FloatingPointError(f'Nonfinite plain MACE loss: {parts}')
    loss.backward()
    for actual,sl in replay_chunks(model,x.flatten(0,1),material.repeat_interleave(len(views)),cfg['microbatch_size'],geometries):
        actual.backward(z.grad.flatten(0,1)[sl])
    return float(loss.detach()),{k:float(v.detach()) for k,v in parts.items()}


def lr_factor(step,*,start,total,warmup,peak,minimum):
    if step<start:return 0.
    elapsed=step-start
    if elapsed<warmup:return .05+.95*elapsed/warmup
    progress=min((elapsed-warmup)/(total-start-warmup),1.)
    return minimum/peak+(1-minimum/peak)*.5*(1+math.cos(math.pi*progress))


def make_scheduler(optimizer,cfg,steps_per_epoch):
    from functools import partial
    total=cfg['epochs']*steps_per_epoch
    common=dict(total=total,minimum=cfg['scheduler']['min_lr'])
    encoder=partial(lr_factor,start=0,warmup=cfg['scheduler']['warmup_epochs']*steps_per_epoch,peak=cfg['learning_rate'],**common)
    head=partial(lr_factor,start=(cfg['loss']['tda_start_epoch']-1)*steps_per_epoch,
                 warmup=cfg['scheduler']['head_warmup_epochs']*steps_per_epoch,peak=cfg['head_learning_rate'],**common)
    return torch.optim.lr_scheduler.LambdaLR(optimizer,[encoder,head])


def representation_ratios(z,material,eligible):
    result={}
    for i,name in enumerate(('Al','Mg','Ta')):
        mask=material==i;a=z[mask,0];s=z[mask,1]
        shuffle=torch.as_tensor(np.random.default_rng(123+i).permutation(len(a)),device=z.device)
        spatial=(a-s).square().mean()/(a-s[shuffle]).square().mean()
        a=z[mask&eligible,0];t=z[mask&eligible,2]
        shuffle=torch.as_tensor(np.random.default_rng(123+i).permutation(len(a)),device=z.device)
        temporal=(a-t).square().mean()/(a-t[shuffle]).square().mean()
        result[name]=dict(spatial_ratio=float(spatial),temporal_ratio=float(temporal))
    return result
