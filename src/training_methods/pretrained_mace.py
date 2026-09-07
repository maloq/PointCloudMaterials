"""Fine-tune an MLIP MACE using spatial/temporal VICReg, TDA and forecasting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn
from src.data_utils.pretrained_mace import Quadruplets,prepare
from src.data_utils.temporal_campaign import write_json
from src.models.encoders.pretrained_mace import PretrainedMACEEncoder


class Learner(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.encoder=PretrainedMACEEncoder(cfg['pretrained_checkpoint'],outer_radius_A=cfg.get('outer_radius_A'))
        self.tda=nn.Sequential(nn.Linear(256,256),nn.SiLU(),nn.Linear(256,cfg['tda_components']))
        self.forecast=nn.Sequential(nn.Linear(261,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256))
        nn.init.zeros_(self.forecast[-1].weight);nn.init.zeros_(self.forecast[-1].bias)


def spread(z,m):
    # Subtract material means: chemical identity cannot satisfy the variance floor.
    residual=z-torch.stack([z[m==i].mean(0) for i in range(3)])[m]
    cov=residual.T@residual/(len(z)-3)
    var=torch.relu(1-torch.sqrt(cov.diagonal()+1e-4)).mean()
    decor=(cov.square().sum()-cov.diagonal().square().sum())/z.shape[1]
    return 25*var+decor


def objective(model,z,target,condition,material,cfg,temporal_mask=None,density=None):
    a,s,t,f=z.unbind(1)
    spatial=(a-s).square().mean()
    selected=torch.ones(len(a),device=a.device,dtype=torch.bool) if temporal_mask is None else temporal_mask
    temporal=(a[selected]-t[selected]).square().mean()
    regularizers=[spread(v,material) for v in (a,s,t,f)]
    spatial_vicreg=25*spatial+.5*(regularizers[0]+regularizers[1])
    temporal_spread=.5*(spread(a[selected],material[selected])+spread(t[selected],material[selected]))
    temporal_vicreg=25*temporal+temporal_spread
    tda_error=(model.tda(z)-target).square().mean()
    tda_used=tda_error;topology_loss=z.new_zeros(());topology_parts={}
    if 'topology' in cfg:
        from src.training_methods.topology_objective import topology_terms
        sw,tw,topology_loss,pair_count,weights=topology_terms(z,target,material,density,cfg['topology'])
        spatial_used=(sw*(a-s).square().mean(-1)).mean()
        temporal_used=(tw[selected]*(a[selected]-t[selected]).square().mean(-1)).mean()
        spatial_vicreg=25*spatial_used+.5*(regularizers[0]+regularizers[1])
        temporal_vicreg=25*temporal_used+temporal_spread
        tda_used=(((model.tda(z)-target).square()*weights).sum(-1)/weights.sum()).mean()
        topology_parts=dict(topology_distance_loss=topology_loss,topology_pair_count=z.new_tensor(pair_count),spatial_attraction_mean=sw.mean(),temporal_attraction_mean=tw[selected].mean(),tda_reliability_mse=tda_used)
    prediction=a+model.forecast(torch.cat((a,condition),1))*condition[:,1:2]
    # The target is the current encoder's future output, detached only in this loss.
    # It still receives VICReg/TDA gradients. There is no target/teacher network.
    forecast_error=(prediction-f.detach()).square().mean()
    persistence_error=(a-f).square().mean()
    loss=cfg['spatial_weight']*spatial_vicreg+cfg['temporal_weight']*temporal_vicreg+cfg['tda_weight']*tda_used+cfg['prediction_weight']*forecast_error+.25*regularizers[3]
    if 'topology' in cfg:loss=loss+cfg['topology']['distance_weight']*topology_loss
    return loss,dict(spatial_mse=spatial,temporal_mse=temporal,tda_mse=tda_error,forecast_mse=forecast_error,persistence_mse=persistence_error,spread=torch.stack(regularizers).mean(),**topology_parts)


def gpu_batch(batch,scaling):
    x,t,c,m=batch
    t=((t-scaling['tda_mean'])@scaling['tda_components'].T)/scaling['tda_std']
    return [torch.from_numpy(v).pin_memory().to('cuda',non_blocking=True) for v in (x,t.astype(np.float32),c,m)]


def encode(model,x,m,microbatch):
    cloud=x.flatten(0,1);material=m.repeat_interleave(4)
    return torch.cat([model.encoder(cloud[i:i+microbatch],material[i:i+microbatch]) for i in range(0,len(cloud),microbatch)]).reshape(len(x),4,256)


def gradient_cached_step(model,batch,cfg,temporal_mask=None):
    """Exact full-batch VICReg gradients with bounded graph activation memory.

    No stochastic layers or coordinate augmentations occur between the two passes.
    The small heads are differentiated once, then cached dL/dz is replayed through
    each encoder chunk. This differs from gradient accumulation of small VICRegs.
    """
    x,target,condition,material=batch
    with torch.no_grad():z=encode(model,x,material,cfg['microbatch_size'])
    z.requires_grad_(True)
    density=None
    if 'topology' in cfg:
        from src.training_methods.topology_objective import neighborhood_density
        density=neighborhood_density(x)
    loss,parts=objective(model,z,target,condition,material,cfg,temporal_mask=temporal_mask,density=density)
    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite MACE objective: {parts}')
    loss.backward()
    derivative=z.grad.flatten(0,1);clouds=x.flatten(0,1);materials=material.repeat_interleave(4)
    size=cfg['microbatch_size']
    for start in range(0,len(clouds),size):
        actual=model.encoder(clouds[start:start+size],materials[start:start+size])
        actual.backward(derivative[start:start+size])
    return float(loss.detach()),{k:float(v.detach()) for k,v in parts.items()}


def fit_scaling(model,data,cfg,out):
    rng=np.random.default_rng(cfg['seed']+100)
    indices=np.concatenate([p[rng.choice(len(p),min(4096,len(p)),replace=False)] for p in data.pools['train']])
    t=np.concatenate([data.tda[i][j] for i,j in indices])
    pca=PCA(n_components=cfg['tda_components'],svd_solver='full').fit(t)
    scaling=dict(tda_mean=pca.mean_.astype(np.float32),tda_components=pca.components_.astype(np.float32),tda_std=np.maximum(np.sqrt(pca.explained_variance_),1e-5).astype(np.float32))
    features=[]
    with torch.no_grad():
        for start in range(0,len(indices),cfg['microbatch_size']):
            selected=indices[start:start+cfg['microbatch_size']]
            x=np.stack([data.clouds[i][j,0] for i,j in selected]).astype(np.float32);m=np.array([data.records[i]['material'] for i,j in selected])
            features.append(model.encoder.raw_features(torch.from_numpy(x).cuda(),torch.from_numpy(m).cuda()).cpu().numpy())
    features=np.concatenate(features);mean=features.mean(0);std=np.maximum(features.std(0),.01)
    model.encoder.feature_mean.copy_(torch.from_numpy(mean));model.encoder.feature_std.copy_(torch.from_numpy(std))
    scaling.update(feature_mean=mean,feature_std=std)
    np.savez(out/'scaling.npz',**scaling)
    write_json(out/'scaling.json',dict(fit_split='train',fit_anchors=len(indices),tda_fit_views=len(t),tda_components=cfg['tda_components'],tda_explained_variance=float(pca.explained_variance_ratio_.sum()),feature_std_min=float(std.min())))
    return scaling


@torch.no_grad()
def validate(model,data,indices,scaling,cfg):
    model.eval();all_z=[];all_t=[];all_c=[];all_m=[];all_masks=[];all_density=[]
    for start in range(0,len(indices),cfg['batch_size']):
        x,t,c,m=gpu_batch(data.get(indices[start:start+cfg['batch_size']]),scaling)
        all_z.append(encode(model,x,m,cfg['microbatch_size']));all_t.append(t);all_c.append(c);all_m.append(m)
        all_masks.append(torch.from_numpy(data.temporal_mask(indices[start:start+cfg['batch_size']])).cuda())
        if 'topology' in cfg:
            from src.training_methods.topology_objective import neighborhood_density
            all_density.append(neighborhood_density(x))
    z=torch.cat(all_z);t=torch.cat(all_t);c=torch.cat(all_c);m=torch.cat(all_m);eligible=torch.cat(all_masks)
    density=torch.cat(all_density) if 'topology' in cfg else None
    # Use identical balanced batch sizes for validation covariance and training.
    losses=[];parts=[]
    for start in range(0,len(z),cfg['batch_size']):
        loss,p=objective(model,z[start:start+cfg['batch_size']],t[start:start+cfg['batch_size']],c[start:start+cfg['batch_size']],m[start:start+cfg['batch_size']],cfg,temporal_mask=eligible[start:start+cfg['batch_size']],density=density[start:start+cfg['batch_size']] if density is not None else None)
        losses.append(float(loss));parts.append({k:float(v) for k,v in p.items()})
    result={k:float(np.mean([p[k] for p in parts])) for k in parts[0]};result['loss']=float(np.mean(losses))
    result['forecast_gain_vs_persistence']=1-result['forecast_mse']/result['persistence_mse']
    result['by_material']={}
    for i,name in enumerate(('Al','Mg','Ta')):
        v=z[m==i,0];cov=torch.cov(v.T).double();eig=torch.linalg.eigvalsh(cov).clamp_min(0);prob=eig/eig.sum();rank=torch.exp(-(prob*prob.clamp_min(1e-20).log()).sum())
        prediction=z[m==i,0]+model.forecast(torch.cat((z[m==i,0],c[m==i]),1))*c[m==i,1:2]
        err=(prediction-z[m==i,3]).square().mean();base=(z[m==i,0]-z[m==i,3]).square().mean()
        result['by_material'][name]=dict(effective_rank=float(rank),tda_mse=float((model.tda(z[m==i])-t[m==i]).square().mean()),forecast_gain_vs_persistence=float(1-err/base))
    return result


def save_checkpoint(path,model,optimizer,cfg,epoch,step,seen,validation):
    temp=path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),config=cfg,epoch=epoch,step=step,anchor_exposures=seen,validation=validation),temp);temp.replace(path)


def load_warm_start(model,source):
    saved=torch.load(source['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model'],strict=True)
    return saved


def train(cfg,wandb_run=None):
    out=Path(cfg['output'])
    if (out/'initial_encoder.pt').exists():
        raise FileExistsError(f'Fresh fine-tuning would overwrite {out}; use a new output directory. Analysis alone uses --stage analysis.')
    data=Quadruplets(cfg);torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    model=Learner(cfg).cuda()
    if 'warm_start' in cfg:
        source=cfg['warm_start']
        saved=load_warm_start(model,source)
        scaling=dict(np.load(Path(source['scaling_dir'])/'scaling.npz'))
        np.savez(out/'scaling.npz',**scaling)
        scaling_record=json.loads((Path(source['scaling_dir'])/'scaling.json').read_text())
        scaling_record['reused_from']=source['scaling_dir']
        write_json(out/'scaling.json',scaling_record)
        write_json(out/'initialization.json',dict(protocol='Retain encoder, prediction heads and fixed train-only scalers; restart optimizer and schedule for the changed data/objective.',source_checkpoint=source['checkpoint'],source_step=saved['step'],source_epoch=saved['epoch'],source_anchor_exposures=saved['anchor_exposures']))
    else:scaling=fit_scaling(model,data,cfg,out)
    backbone=[p for p in model.encoder.parameters() if p.requires_grad];heads=list(model.tda.parameters())+list(model.forecast.parameters())
    optimizer=torch.optim.AdamW([dict(params=backbone,lr=cfg['learning_rate']),dict(params=heads,lr=cfg['head_learning_rate'])],weight_decay=cfg['weight_decay'],fused=True)
    per_step='scheduler' in cfg
    if per_step:
        from src.utils.training_utils import build_step_cosine_scheduler
        schedule=cfg['scheduler']
        if schedule['name']!='cosine_per_step':raise ValueError(f'Unsupported MACE scheduler: {schedule}')
        epoch_steps=data.epoch_steps(cfg['batch_size']);total_steps=epoch_steps*cfg['epochs'];warmup_steps=epoch_steps*schedule['warmup_epochs']
        scheduler=build_step_cosine_scheduler(optimizer,total_steps=total_steps,warmup_steps=warmup_steps,start_factor=schedule['warmup_start_factor'],min_lr=schedule['min_lr'])
        write_json(out/'scheduler.json',dict(name='linear_warmup_then_cosine',interval='optimizer_step',steps_per_epoch=epoch_steps,total_steps=total_steps,warmup_steps=warmup_steps,peak_lrs=[cfg['learning_rate'],cfg['head_learning_rate']],minimum_lr=schedule['min_lr'],start_factor=schedule['warmup_start_factor']))
    else:scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=1,threshold=.002)
    per=cfg['validation_anchors_per_material'];val=data.validation_indices(per,rng)
    initial=validate(model,data,val,scaling,cfg);write_json(out/'initial_validation.json',initial)
    if wandb_run is not None:
        wandb_run.log({'training_step':0,**flatten_metrics('validation_initial',initial)})
    # Save the exact pretrained encoder with its fixed feature scaler for comparisons.
    torch.save(dict(encoder=model.encoder.state_dict(),config=cfg),out/'initial_encoder.pt')
    ckpt=Path(cfg.get('checkpoint_directory',str(Path(cfg['cache'])/'checkpoints')));ckpt.mkdir(parents=True,exist_ok=True)
    best=float('inf');bad=0;step=0;seen=0;start=time.monotonic();stop='maximum_epochs'
    log=open(out/'training.jsonl','a',buffering=1)
    for epoch in range(1,cfg['epochs']+1):
        model.train();epoch_start=time.monotonic();total=0.;steps=0
        iterator=iter(data.epoch('train',cfg['batch_size'],rng))
        with ThreadPoolExecutor(max_workers=1) as pool:
            indices=next(iterator);pending=pool.submit(data.get,indices)
            while True:
                batch=gpu_batch(pending.result(),scaling)
                upcoming=next(iterator,None)
                if upcoming is not None:pending=pool.submit(data.get,upcoming)
                optimizer.zero_grad(set_to_none=True)
                used_lrs=[g['lr'] for g in optimizer.param_groups]
                temporal_mask=torch.from_numpy(data.temporal_mask(indices)).cuda()
                loss,parts=gradient_cached_step(model,batch,cfg,temporal_mask=temporal_mask)
                grad=torch.nn.utils.clip_grad_norm_(backbone+heads,cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
                if per_step:scheduler.step()
                step+=1;steps+=1;seen+=len(indices);total+=loss
                if step%100==0:
                    save_checkpoint(ckpt/'last.pt',model,optimizer,cfg,epoch,step,seen,initial if epoch==1 else validation)
                if step%10==0:
                    status=dict(state='training',epoch=epoch,step=step,anchor_exposures=seen,view_exposures=4*seen,neighborhood_points=cfg.get('model_points',cfg['points']),elapsed_seconds=time.monotonic()-start,loss=loss,learning_rates_used=used_lrs,temporal_pairs=int(temporal_mask.sum()),temporal_lag_ps=cfg.get('temporal_lag_ps'),gradient_norm=float(grad),parts=parts,utc=datetime.now(timezone.utc).isoformat())
                    if wandb_run is not None:
                        status['wandb_url']=wandb_run.url
                        wandb_run.log({'training_step':step,'train/epoch':epoch,'train/loss':loss,'train/gradient_norm':float(grad),'train/anchor_exposures':seen,**flatten_metrics('train',parts),'train/backbone_lr':used_lrs[0],'train/head_lr':used_lrs[1],'train/temporal_pairs':int(temporal_mask.sum())})
                    write_json(out/'status.json',status);print('TRAIN',json.dumps(status),flush=True)
                if time.monotonic()-start>=cfg['max_training_seconds']:
                    stop='time_limit';break
                if upcoming is None:break
                indices=upcoming
        validation=validate(model,data,val,scaling,cfg)
        if not per_step:scheduler.step(validation['loss'])
        record=dict(epoch=epoch,steps=step,anchor_exposures=seen,train_loss=total/steps,validation=validation,epoch_seconds=time.monotonic()-epoch_start,elapsed_seconds=time.monotonic()-start,learning_rates=[g['lr'] for g in optimizer.param_groups])
        log.write(json.dumps(record,allow_nan=False)+'\n');print('VALIDATION',json.dumps(record),flush=True)
        if wandb_run is not None:
            wandb_run.log({'training_step':step,**flatten_metrics('validation',validation)})
        save_checkpoint(ckpt/'last.pt',model,optimizer,cfg,epoch,step,seen,validation)
        if validation['loss']<best*(1-.001):
            best=validation['loss'];bad=0;save_checkpoint(ckpt/'best.pt',model,optimizer,cfg,epoch,step,seen,validation)
        else:bad+=1
        if stop=='time_limit':break
        if bad>=cfg['patience'] and epoch>=4:stop='early_stopping';break
    log.close()
    best_saved=torch.load(ckpt/'best.pt',map_location='cpu',weights_only=False)
    selection=cfg.get('checkpoint_selection','best')
    if selection not in ('best','last'):raise ValueError(f'Unknown checkpoint_selection: {selection}')
    selected=best_saved if selection=='best' else torch.load(ckpt/'last.pt',map_location='cpu',weights_only=False)
    # Keep selected model weights, without optimizer state, inside the repository.
    selected.pop('optimizer');torch.save(selected,out/'best.pt')
    write_json(out/'training_summary.json',dict(stop_reason=stop,epochs_completed=epoch-(stop=='time_limit'),partial_epoch=(stop=='time_limit'),steps=step,anchor_exposures=seen,view_exposures=4*seen,seconds=time.monotonic()-start,best_epoch=best_saved['epoch'],best_validation=best_saved['validation'],checkpoint_selection=selection,selected_epoch=selected['epoch'],selected_validation=selected['validation'],initial_validation=initial,converged=stop=='early_stopping'))


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


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['prepare','train','analysis','all'],default='all');args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    def interrupted(signum,frame):raise InterruptedError(f'MACE workflow interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    wandb_run=None
    try:
        from src.experiment_runner.tracking import tracked_run
        with tracked_run(out, kind='training' if args.stage in ('train', 'all') else 'analysis',
                         configs=[Path(args.config)], command=[sys.executable, *sys.argv]):
            if 'wandb' in cfg and args.stage in ('train','all'):
                wandb_run=start_wandb(cfg,out)
            if args.stage in ('prepare','all'):prepare(cfg)
            if args.stage in ('train','all'):
                train(cfg,wandb_run)
                if wandb_run is not None:
                    summary=json.loads((out/'training_summary.json').read_text())
                    wandb_run.summary.update({'best_epoch':summary['best_epoch'],'stop_reason':summary['stop_reason'],'anchor_exposures':summary['anchor_exposures'],**flatten_metrics('best_validation',summary['best_validation'])})
            if args.stage in ('analysis','all'):
                from src.analysis.pretrained_mace_adapter import export_encoder
                export_encoder(cfg)
                write_json(out/'status.json',dict(state='static_analysis'))
                subprocess.run([sys.executable,'-m','src.analysis.pipeline',cfg['analysis_config']],check=True)
                from src.analysis.pretrained_mace_adapter import write_report
                write_report(cfg)
                write_json(out/'status.json',dict(state='complete',utc=datetime.now(timezone.utc).isoformat(),wandb_url=wandb_run.url if wandb_run is not None else None))
            if wandb_run is not None:
                wandb_run.summary['workflow_state']='complete'
                wandb_run.finish()
    except BaseException:
        if wandb_run is not None:
            wandb_run.summary['workflow_state']='failed'
            wandb_run.finish(exit_code=1)
        write_json(out/'status.json',dict(state='failed',traceback=traceback.format_exc(),utc=datetime.now(timezone.utc).isoformat()))
        raise

if __name__=='__main__':main()
