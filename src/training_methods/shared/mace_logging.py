"""Checkpoint and online tracking utilities for MACE training."""
import torch
from src.data_utils.temporal_campaign import write_json


def save_checkpoint(path,model,optimizer,cfg,epoch,step,seen,validation):
    temp=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,epoch=epoch,step=step,anchor_exposures=seen,validation=validation),temp);temp.replace(path)


def flatten_metrics(prefix,values):
    result={}
    for key,value in values.items():
        if isinstance(value,dict):result.update(flatten_metrics(prefix+'/'+key,value))
        else:result[prefix+'/'+key]=value
    return result


def start_wandb(cfg,out):
    import wandb
    settings=cfg['wandb']
    run=wandb.init(project=settings['project'],name=settings['name'],id=settings['id'],
        mode='online',resume='never',config=cfg,dir=str(out),save_code=False)
    if run.offline:raise RuntimeError('Online W&B was requested but the run is offline')
    run.define_metric('training_step')
    run.define_metric('train/*',step_metric='training_step')
    run.define_metric('validation/*',step_metric='training_step')
    write_json(out/'wandb_run.json',dict(id=run.id,url=run.url,entity=run.entity,project=run.project,mode='online'))
    return run
