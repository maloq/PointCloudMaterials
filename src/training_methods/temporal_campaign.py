"""Time-budgeted descriptor-free temporal representation hypothesis campaign."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import traceback

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from src.data_utils.temporal_campaign import ROOT,TemporalPairs,prepare,write_json
from src.models.encoders.atomic_graph import ReferenceMACEEncoder,SchNetEncoder,DensityMLPEncoder
from src.models.encoders.geo_frame_transformer_v2 import GeoFrameTransformerV2Encoder


def local_graph(x,cutoff):
    """Recompute compact directed edges in a complete two-interaction halo."""
    b,n,_=x.shape
    r=x.norm(dim=-1);halo=r<=2*cutoff+.12
    distance=torch.cdist(x,x)
    # cdist's matrix implementation can round diagonal distances above zero.
    # Exclude self edges by identity, not a floating-point distance comparison.
    valid=(distance<cutoff)&~torch.eye(n,device=x.device,dtype=torch.bool)[None]&halo[:,:,None]&halo[:,None,:]
    ids=valid.nonzero();counts=torch.bincount(ids[:,0],minlength=b)
    offsets=counts.cumsum(0)-counts
    edges=torch.zeros((b,int(counts.max()),2),dtype=torch.int16,device=x.device)
    edges[ids[:,0],torch.arange(len(ids),device=x.device)-offsets[ids[:,0]]]=ids[:,1:].to(torch.int16)
    return edges,counts


class Representation(nn.Module):
    def __init__(self,hyp,cfg):
        super().__init__();self.hyp=hyp;self.points=769 if hyp['cutoff_A']==6 else 193
        name=hyp['model'];dim=128
        if name=='MACE':
            counts=json.loads((ROOT/cfg['output']/'scaling/geometry_scaling.json').read_text())
            self.encoder=ReferenceMACEEncoder(channels=hyp['channels'],cutoff_A=hyp['cutoff_A'],avg_num_neighbors=counts[str(hyp['cutoff_A'])]);dim=2*hyp['channels']
        elif name=='SchNet':self.encoder=SchNetEncoder()
        elif name=='DensityMLP':
            self.encoder=DensityMLPEncoder()
            scaling=torch.load(ROOT/cfg['output']/'scaling/density_scaling.pt',map_location='cpu',weights_only=True)
            self.encoder.mean.copy_(scaling['mean']);self.encoder.std.copy_(scaling['std'])
        elif name=='GeoFrame':
            saved=torch.load(ROOT/cfg['geoframe_config_checkpoint'],map_location='cpu',weights_only=False)
            kwargs=OmegaConf.to_container(OmegaConf.create(saved['hyper_parameters']).encoder.kwargs,resolve=True)
            self.encoder=GeoFrameTransformerV2Encoder(**kwargs)
            self.register_buffer('radii',torch.tensor(cfg['geoframe_radii_A']))
        else:raise ValueError(name)
        self.output=nn.Sequential(nn.Linear(dim,cfg['latent_dim']),nn.LayerNorm(cfg['latent_dim']))

    def forward(self,x,material):
        x=x[:,:self.points]
        if self.hyp['model'] in ('MACE','SchNet'):
            edges,counts=local_graph(x,self.hyp['cutoff_A']);z=self.encoder(x,material,edges,counts)
        elif self.hyp['model']=='DensityMLP':z=self.encoder(x,material,None,None)
        else:
            order=x.square().sum(-1).argsort(1);x=x.gather(1,order[:,:,None].expand(-1,-1,3))
            z=self.encoder.forward_features(x[:,:80]/self.radii[material,None,None])
        return self.output(z)


class Learner(nn.Module):
    def __init__(self,hyp,cfg):
        super().__init__();self.representation=Representation(hyp,cfg);d=cfg['latent_dim']
        self.forecast=nn.Sequential(nn.Linear(d+5,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,d))
        self.motion=nn.Sequential(nn.Linear(d+5,256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,2))


def spread_loss(z,material):
    # Species identity alone must not satisfy the anti-collapse objective.
    means=torch.stack([z[material==m].mean(0) for m in range(3)])
    z=z-means[material];cov=z.T@z/(len(z)-3)
    variance=torch.relu(1-torch.sqrt(cov.diagonal()+1e-4)).mean()
    offdiag=(cov.square().sum()-cov.diagonal().square().sum())/z.shape[1]
    return 25*variance+offdiag


def to_gpu(batch):
    return [torch.from_numpy(x).pin_memory().to('cuda',non_blocking=True) for x in batch]


def noisy(x,cfg):
    noise=(torch.randn_like(x)*cfg['jitter_std_A']).clamp(-cfg['jitter_clip_A'],cfg['jitter_clip_A']);noise[:,0]=0
    return x+noise


def objective(model,teacher,batch,hyp,cfg):
    x,y,material,condition,motion=batch
    z=model.representation(noisy(x,cfg),material)
    mode=hyp['mode']
    if mode in ('static','temporal'):
        future=model.representation(noisy(x if mode=='static' else y,cfg),material)
        temporal=(z-future).square().mean();regularizer=.5*(spread_loss(z,material)+spread_loss(future,material))
    else:
        with torch.no_grad():future=teacher(y,material)
        prediction=z+model.forecast(torch.cat((z,condition),1))*condition[:,1:2]
        temporal=(prediction-future).square().mean();regularizer=spread_loss(z,material)
    # Every trial has the same physical-motion probe. Only the explicit motion
    # hypothesis lets this supervision update the representation.
    motion_z=z if mode=='motion' else z.detach()
    movement=(model.motion(torch.cat((motion_z,condition),1))-motion).square().mean()
    smooth=z.new_zeros(())
    if mode=='smooth':
        epsilon=torch.randn_like(x)*.001;epsilon[:,0]=0
        clean=model.representation(x,material);perturbed=model.representation(x+epsilon,material)
        smooth=.05*(clean-perturbed).square().mean()/.01
    loss=25*temporal+regularizer+(5 if mode=='motion' else 1)*movement+smooth
    return loss,dict(temporal=temporal,spread=regularizer,motion=movement,smooth=smooth)


@torch.no_grad()
def validate(model,data,cfg,hyp):
    model.eval();rng=np.random.default_rng(2026090601);errors=[];materials=[];features=[]
    # The exact validation draws and all lags are common to every hypothesis.
    bs=min(cfg['validation_batch_size'],batch_size(hyp,cfg))
    count=cfg['validation_batches']*cfg['validation_batch_size']
    for _ in range(cfg['validation_batches']):
        whole=data.batch('val',cfg['validation_batch_size'],rng,points=model.representation.points)
        for start in range(0,cfg['validation_batch_size'],bs):
            x,_,material,condition,target=to_gpu([v[start:start+bs] for v in whole])
            z=model.representation(x,material);prediction=model.motion(torch.cat((z,condition),1))
            errors.append((prediction-target).square().mean(1).cpu().numpy());materials.append(material.cpu().numpy());features.append(z.cpu().numpy())
    e=np.concatenate(errors);m=np.concatenate(materials);z=np.concatenate(features)
    material_losses=[float(e[m==i].mean()) for i in range(3)]
    eig=np.linalg.eigvalsh(np.cov(z.astype(np.float64),rowvar=False));eig=np.maximum(eig,0);p=eig/eig.sum();p=p[p>0]
    rank=float(np.exp(-(p*np.log(p)).sum()))
    if not np.isfinite(e).all() or not np.isfinite(rank):raise FloatingPointError(f'Invalid validation for {hyp["name"]}')
    return dict(selection_score=float(np.mean(material_losses)),motion_by_material=material_losses,effective_rank=rank)


def batch_size(hyp,cfg):
    key='MACE_wide_halo' if hyp['cutoff_A']==6 else 'MACE_capacity' if hyp['model']=='MACE' and hyp['channels']==128 else hyp['model']
    return cfg['batch_sizes'][key]


def train(hyp,seed,rate,seconds,cfg,out,data,stage):
    key=f'{stage}_{hyp["name"]}_seed{seed}_lr{rate:g}';directory=Path(cfg['checkpoint_cache'])/key;directory.mkdir(parents=True,exist_ok=False)
    logs=out/'trials'/key;logs.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed);rng=np.random.default_rng(seed+1000)
    model=Learner(hyp,cfg).cuda();teacher=Representation(hyp,cfg).cuda().eval().requires_grad_(False)
    teacher.load_state_dict(model.representation.state_dict())
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=rate,weight_decay=cfg['weight_decay'],fused=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=3,threshold=1e-4,min_lr=1e-6)
    start=time.monotonic();end=start+seconds;next_validation=start;step=0;best=float('inf');best_step=0;seen=0;records=[]
    while time.monotonic()<end:
        model.train();batch=to_gpu(data.batch('train',batch_size(hyp,cfg),rng,fraction=hyp['fraction'],points=model.representation.points))
        optimizer.zero_grad(set_to_none=True);loss,parts=objective(model,teacher,batch,hyp,cfg)
        if not torch.isfinite(loss):raise FloatingPointError(f'{key}: nonfinite training loss at step {step}')
        loss.backward();gradient=torch.nn.utils.clip_grad_norm_(parameters,cfg['gradient_clip'],error_if_nonfinite=True);optimizer.step()
        with torch.no_grad():
            for target,current in zip(teacher.parameters(),model.representation.parameters()):target.lerp_(current,1-cfg['ema_decay'])
            for target,current in zip(teacher.buffers(),model.representation.buffers()):target.copy_(current)
        step+=1;seen+=len(batch[0])
        if time.monotonic()>=next_validation or time.monotonic()>=end:
            metrics=validate(model,data,cfg,hyp)
            row=dict(step=step,examples_seen=seen,seconds=time.monotonic()-start,learning_rate=optimizer.param_groups[0]['lr'],
                gradient_norm=float(gradient),**{k:float(v.detach()) for k,v in parts.items()},**metrics)
            records.append(row)
            with (logs/'history.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
            scheduler.step(metrics['selection_score'])
            payload=dict(model=model.state_dict(),teacher=teacher.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=rng.bit_generator.state,hypothesis=hyp,config=cfg,seed=seed,initial_rate=rate,**row)
            if metrics['selection_score']<best:
                best=metrics['selection_score'];best_step=step;torch.save(payload,directory/'best.pt')
            torch.save(payload,directory/'last.pt')
            write_json(logs/'status.json',dict(state='running',best_score=best,**row));print('VALIDATION',key,row,flush=True)
            next_validation=time.monotonic()+cfg['validation_every_seconds']
    result=dict(name=hyp['name'],hypothesis=hyp,seed=seed,learning_rate=rate,stage=stage,steps=step,examples_seen=seen,
        seconds=time.monotonic()-start,selection_score=best,best_step=best_step,checkpoint=str(directory/'best.pt'),
        state='budget_complete',converged=False,stop_reason='Allocated wall-clock budget; no convergence claim',history=str(logs/'history.jsonl'))
    write_json(logs/'status.json',result)
    del model,teacher,optimizer,scheduler,parameters,payload,batch,loss,parts;torch.cuda.empty_cache()
    return result


def fit_density_scaling(data,cfg):
    from src.models.encoders.smooth_density import SmoothDensity
    density=SmoothDensity().cuda();powers=[];neighbors={4.:[],6.:[]};rng=np.random.default_rng(cfg['data_seed']+400)
    with torch.no_grad():
        for _ in range(16):
            x=torch.tensor(data.batch('train',1024,rng)[0],device='cuda')
            powers.append(density.power(density(x[:,1:]/8.)).cpu())
            for cutoff in neighbors:neighbors[cutoff].append(float((x[:,1:].norm(dim=-1)<cutoff).sum(1).float().mean()))
    p=torch.cat(powers);std=p.std(0);std=std.clamp_min(.001*std.median())
    directory=ROOT/cfg['output']/'scaling';directory.mkdir(exist_ok=True)
    torch.save(dict(mean=p.mean(0),std=std),directory/'density_scaling.pt')
    write_json(directory/'geometry_scaling.json',{str(k):float(np.mean(v)) for k,v in neighbors.items()})
    del density;torch.cuda.empty_cache()


def run(cfg,out):
    from src.analysis.temporal_campaign import evaluate
    launch=time.time();deadline=launch+cfg['duration_seconds'];train_deadline=deadline-cfg['evaluation_reserve_seconds']
    write_json(out/'schedule_runtime.json',dict(start_utc=datetime.fromtimestamp(launch,timezone.utc).isoformat(),
        deadline_utc=datetime.fromtimestamp(deadline,timezone.utc).isoformat(),training_deadline_utc=datetime.fromtimestamp(train_deadline,timezone.utc).isoformat()))
    prepare(cfg,out);data=TemporalPairs(cfg);fit_density_scaling(data,cfg)
    np.savez(out/'scaling/motion_scaling.npz',mean=data.mean,std=data.std)
    results=[]
    for hyp in cfg['hypotheses']:
        for rate in cfg['learning_rates']:
            if time.time()+cfg['screen_seconds']>train_deadline:raise RuntimeError('Preparation exceeded the budget needed for the hypothesis sweep')
            results.append(train(hyp,123,rate,cfg['screen_seconds'],cfg,out,data,'screen'))
            write_json(out/'training_results.json',results)
    selected=[min((r for r in results if r['name']==h['name']),key=lambda r:r['selection_score']) for h in cfg['hypotheses']]
    # Confirm the strongest validation candidate from three different encoder
    # families, so faster density variants cannot occupy every repeat slot.
    families={r['hypothesis']['model'] for r in selected}
    family_best=sorted([min((r for r in selected if r['hypothesis']['model']==m),key=lambda r:r['selection_score']) for m in families],key=lambda r:r['selection_score'])
    finalists=family_best[:cfg['confirmation_finalists']];write_json(out/'validation_finalists.json',finalists)
    write_json(out/'screen_selected.json',selected)
    selected=[r for r in selected if r['name'] not in {f['name'] for f in finalists}]
    jobs=[(r,s) for r in finalists for s in [123]+cfg['confirmation_seeds']]
    seconds=(train_deadline-time.time()-30*len(jobs))/len(jobs)
    if seconds<cfg['screen_seconds']:raise RuntimeError('Insufficient budget for confirmation longer than screening')
    write_json(out/'confirmation_budget.json',dict(seconds_per_seed=seconds,jobs=len(jobs)))
    for reference,seed in jobs:
        result=train(reference['hypothesis'],seed,reference['learning_rate'],seconds,cfg,out,data,'confirm')
        results.append(result);selected.append(result);write_json(out/'training_results.json',results)
    write_json(out/'selected_runs.json',selected)
    evaluate(cfg,out,selected,deadline)


def preflight(cfg,out):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    from src.data_utils.temporal_campaign import task_list,prepare_task
    cache=Path(cfg['cache']);cache.mkdir(parents=True,exist_ok=True);tasks=task_list(cfg)
    sample=[next(t for t in tasks if t['split']==split and t['material']==m) for split in ('train','val') for m in range(3)]
    records=[];pending=[]
    for task in sample:
        path=cache/task['name']/'manifest.json'
        if path.exists():
            np.testing.assert_array_equal(np.load(path.parent/'center_rows.npy'),task['rows'])
            records.append(json.loads(path.read_text()))
        else:pending.append(task)
    with ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn')) as pool:
        records.extend(pool.map(prepare_task,pending,[cfg]*len(pending)))
    write_json(cache/'manifest.json',dict(shards=records,config=cfg))
    data=TemporalPairs(cfg);fit_density_scaling(data,cfg);results=[]
    for hyp in cfg['hypotheses']:
        torch.manual_seed(20260906);model=Learner(hyp,cfg).cuda();teacher=Representation(hyp,cfg).cuda().eval().requires_grad_(False)
        teacher.load_state_dict(model.representation.state_dict())
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,fused=True)
        rng=np.random.default_rng(20260906);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        for _ in range(3):
            batch=to_gpu(data.batch('train',batch_size(hyp,cfg),rng,points=model.representation.points))
            loss,_=objective(model,teacher,batch,hyp,cfg);optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step()
        torch.cuda.synchronize();seconds=time.monotonic()-start;model.eval()
        with torch.no_grad():
            x=batch[0][:8];m=batch[2][:8];q,_=torch.linalg.qr(torch.tensor([[.2,.3,.7],[.5,-.2,.1],[.4,.8,.3]],device='cuda'))
            a=model.representation(x,m);b=model.representation(x@q,m);error=float((a-b).abs().max())
            if hyp['model']!='GeoFrame':torch.testing.assert_close(a,b,atol=5e-4,rtol=5e-4)
        record=dict(name=hyp['name'],batch_size=batch_size(hyp,cfg),seconds_per_step=seconds/3,
            peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,rotation_max_absolute_error=error,loss=float(loss))
        results.append(record);write_json(out/'preflight_results.json',results);print('PREFLIGHT',record,flush=True)
        del model,teacher,optimizer,batch,loss,a,b;torch.cuda.empty_cache()
    # Exercise checkpointing, validation, and the complete budget-stop path.
    train(cfg['hypotheses'][0],999,.0003,15,cfg,out,data,'preflight_final')
    write_json(out/'preflight_status.json',dict(state='complete',models=len(results)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--preflight',action='store_true')
    mode.add_argument('--analysis-only',action='store_true',help='Evaluate the fixed selected_runs.json; archive previous evaluation directories before replay. No training or reselection.')
    mode.add_argument('--screen-analysis',action='store_true',help='Evaluate all twelve fixed screening winners under screen_analysis/ for equal-budget ablations. No training or reselection.')
    mode.add_argument('--review-only',action='store_true',help='Summarize completed evaluation artifacts and training histories; no model inference.')
    mode.add_argument('--static-config',type=Path,help='Run full saved-center static analysis using this explicit analysis configuration; no training.')
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text());out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.static_config:
        from src.analysis.temporal_static import main as static_main
        static_main(cfg,args.static_config);return
    if args.review_only:
        from src.analysis.temporal_campaign import review
        review(cfg,out);return
    if args.analysis_only or args.screen_analysis:
        from src.analysis.temporal_campaign import evaluate
        saved_cfg=json.loads((out/'config.json').read_text())
        if cfg!=saved_cfg:raise ValueError('Analysis replay must use the original saved campaign configuration')
        selected=json.loads((out/('screen_selected.json' if args.screen_analysis else 'selected_runs.json')).read_text())
        if args.screen_analysis:
            out=out/'screen_analysis';out.mkdir(exist_ok=False)
        status=dict(state='analyzing',pid=os.getpid(),training_complete=True,analysis_started_at=datetime.now(timezone.utc).isoformat())
        write_json(out/'status.json',status)
        try:
            evaluate(cfg,out,selected,float('inf'))
            status.update(state='complete',finished_at=datetime.now(timezone.utc).isoformat())
        except BaseException as error:
            status.update(state='failed',error=repr(error),traceback=traceback.format_exc());raise
        finally:write_json(out/'status.json',status)
        return
    if (out/'status.json').exists():raise FileExistsError(f'Campaign already initialized: {out}')
    if args.preflight:
        preflight(cfg,out);return
    status=dict(state='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat());write_json(out/'status.json',status);write_json(out/'config.json',cfg)
    def terminate(signum,frame):raise InterruptedError(f'Campaign received signal {signum}; durable checkpoints and logs are retained')
    signal.signal(signal.SIGTERM,terminate)
    try:
        run(cfg,out);status.update(state='complete',finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:write_json(out/'status.json',status)
