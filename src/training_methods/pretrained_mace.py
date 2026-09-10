"""80-atom MACE: shuffled data, fixed VICReg, configurable TDA activation."""
import json
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone

import numpy as np
from sklearn.decomposition import PCA
import torch
from torch import nn

from src.data_utils.pretrained_mace import Quadruplets
from src.data_utils.pretrained_mace_gpu import GPUQuadruplets
from src.data_utils.temporal_campaign import write_json
from src.models.encoders.pretrained_mace import PretrainedMACEEncoder
from src.training_methods.mace_logging import flatten_metrics,start_wandb,save_checkpoint
from src.training_methods.mace_objective import objective,cached_step,make_scheduler,training_views
from src.training_methods.mace_performance import encode_views


class Learner(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.encoder=PretrainedMACEEncoder(cfg['pretrained_checkpoint'],performance=cfg['performance'])
        self.tda=nn.Sequential(nn.Linear(256,256),nn.SiLU(),nn.Linear(256,cfg['tda_components']))


def fit_scaling(model,data,cfg,out):
    rng=np.random.default_rng(cfg['seed']+100);pool=data.all_indices('train')
    rows=pool[rng.choice(len(pool),min(12288,len(pool)),replace=False)]
    targets=np.stack([data.tda[i][j,:3] for i,j in rows]).reshape(-1,144)
    pca=PCA(n_components=cfg['tda_components'],svd_solver='full').fit(targets)
    scaling=dict(tda_mean=pca.mean_.astype(np.float32),tda_components=pca.components_.astype(np.float32),
                 tda_std=np.maximum(np.sqrt(pca.explained_variance_),1e-5).astype(np.float32))
    features=[]
    scaling_views=[0,4] if cfg["protocol"]=="thermal80" else [0]
    with torch.no_grad():
        for start in range(0,len(rows),cfg['microbatch_size']):
            selected=rows[start:start+cfg['microbatch_size']]
            x=torch.tensor(np.stack([data.clouds[i][j,v] for i,j in selected for v in scaling_views]).astype(np.float32),device='cuda')
            m=torch.tensor([data.records[i]['material'] for i,j in selected for v in scaling_views],device='cuda')
            features.append(model.encoder.raw_features(x,m).cpu().numpy())
    features=np.concatenate(features)
    scaling.update(feature_mean=features.mean(0),feature_std=np.maximum(features.std(0),.01))
    for name in ('feature_mean','feature_std'):getattr(model.encoder,name).copy_(torch.from_numpy(scaling[name]))
    np.savez(out/'scaling.npz',**scaling)
    write_json(out/'scaling.json',dict(fit_split='train',sampling='uniform stored rows, no element quotas',fit_anchors=len(rows),
        tda_fit_views=len(targets),tda_points=80,tda_components=cfg['tda_components'],tda_explained_variance=float(pca.explained_variance_ratio_.sum())))
    return scaling


@torch.no_grad()
def validate(model,data,gpu,cfg,epoch):
    model.eval();indices=data.all_indices('val');zs=[];targets=[];materials=[];views=training_views(cfg['loss'])
    for start in range(0,len(indices),cfg['batch_size']):
        x,t,_,m=gpu.get(indices[start:start+cfg['batch_size']])
        zs.append(encode_views(model,x[:,views],m,cfg['microbatch_size']));targets.append(t[:,views]);materials.append(m)
    z=torch.cat(zs);target=torch.cat(targets);m=torch.cat(materials)
    eligible=torch.tensor(data.temporal_mask(indices),device='cuda')
    loss,parts=objective(model,z,target,eligible,cfg['loss'],epoch)
    result=dict(loss=float(loss),**{k:float(v) for k,v in parts.items()},by_material={})
    from src.training_methods.mace_objective import representation_ratios
    ratios=representation_ratios(z,m,eligible)
    for i,name in enumerate(('Al','Mg','Ta')):
        covariance=torch.cov(z[m==i,0].T).double()
        eigen=torch.linalg.eigvalsh(covariance).clamp_min(0);prob=eigen/eigen.sum()
        values=dict(anchors=int((m==i).sum()),effective_rank=float(torch.exp(-(prob*prob.clamp_min(1e-20).log()).sum())),**ratios[name])
        if epoch>=cfg['loss']['tda_start_epoch']:values['tda_mse']=float((model.tda(z[m==i])-target[m==i]).square().mean())
        result['by_material'][name]=values
    return result


def summarize_data(cfg):
    import hashlib
    data=Quadruplets(cfg);counts={}
    for split in ('train','val'):
        counts[split]={}
        for material,name in enumerate(('Al','Mg','Ta')):
            unique=set();anchors=0
            for record in data.records:
                if record['split']!=split or record['material']!=material:continue
                directory=Path(record['directory']);ids=np.load(directory/'ids.npy')[:,:3]
                frames=np.load(directory/'frames.npy')[:,:3];anchors+=len(ids)
                for frame,atom in zip(frames.ravel(),ids.ravel()):
                    source=f"initial_Al_parent_{record['parent']}" if record['kind']=='shooting' and frame==0 else record['path']
                    unique.add((source,int(frame),int(atom)))
            counts[split][name]=dict(anchor_quadruplets=anchors,view_slots=3*anchors,distinct_neighborhood_states=len(unique))
    summary=dict(counts=counts,limits=cfg['data_limitations'],physical_cutoff_A=5.,
                 pretrained_checkpoint_sha256=hashlib.sha256(Path(cfg['pretrained_checkpoint']).read_bytes()).hexdigest(),
                 temporal='Same atom at 0.1 ps for Al/Mg/Ta; Al shooting 0.3 ps pairs excluded from temporal VICReg.')
    summary.update(neighborhood_points=80,required_halo_A=None,training_views=3,
        sampling='One uniform shuffled pass through stored rows per epoch; no element quotas or oversampling.',
        encoder='Mean-pool both scalar feature blocks of the full 80-atom MACE graph; native 5 A edge cutoff; no added taper or context projection.',
        tda='Fresh 144D H0/H1/H2 persistence images from all 80 input atoms, normalized by 79; train-only PCA.',
        training_objectives=['fixed spatial/temporal VICReg',f"TDA reconstruction from epoch {cfg['loss']['tda_start_epoch']}"],
        spatial='One of the six nearest actual atoms, independently centered.',
        forecast='No forecasting loss or head. The stored fourth view is only available to frozen post-training probes.')
    if cfg['protocol']=='thermal80':
        summary.update(training_views=6,stored_views=8,tda='TDA of relaxed versions of the same 80 hot-selected atom IDs; hot and relaxed views share targets.',
            relaxation=cfg['relaxation']['controls'],source_coverage=cfg['relaxation']['shards'],
            training_objectives=['spatial/temporal VICReg on hot and relaxed views','matched hot-relaxed consistency',f"relaxed TDA from epoch {cfg['loss']['tda_start_epoch']}"])
    write_json(Path(cfg['output'])/'data_summary.json',summary)


def train(cfg,wandb_run):
    out=Path(cfg['output']);ckpt=Path(cfg['checkpoint_directory']);ckpt.mkdir(parents=True,exist_ok=True)
    if (out/'initial_encoder.pt').exists():raise FileExistsError(f'Training would overwrite {out}; choose a new run directory')
    torch.manual_seed(cfg['seed']);rng=np.random.default_rng(cfg['seed'])
    data=Quadruplets(cfg);model=Learner(cfg).cuda()
    scaling=fit_scaling(model,data,cfg,out);gpu=GPUQuadruplets(data,scaling)
    steps_per_epoch=data.epoch_steps(cfg['batch_size']);total_steps=steps_per_epoch*cfg['epochs']
    backbone=[p for p in model.encoder.parameters() if p.requires_grad];heads=list(model.tda.parameters())
    optimizer=torch.optim.AdamW([dict(params=backbone,lr=cfg['learning_rate']),dict(params=heads,lr=cfg['head_learning_rate'])],weight_decay=cfg['weight_decay'],fused=True)
    scheduler=make_scheduler(optimizer,cfg,steps_per_epoch)
    counts=dict(encoder_trainable=sum(p.numel() for p in backbone),tda_head=sum(p.numel() for p in heads),forecast_head=0)
    write_json(out/'parameter_counts.json',counts);wandb_run.summary.update(counts)
    write_json(out/'scheduler.json',dict(interval='optimizer_step',steps_per_epoch=steps_per_epoch,total_steps=total_steps,tda_start_epoch=cfg['loss']['tda_start_epoch'],settings=cfg['scheduler']))
    initial=validate(model,data,gpu,cfg,0);write_json(out/'initial_validation.json',initial)
    torch.save(dict(encoder=model.encoder.state_dict(),config=cfg),out/'initial_encoder.pt')
    wandb_run.log({'training_step':0,**flatten_metrics('validation_initial',initial)})
    best=float('inf');best_epoch=None;step=0;seen=0;started=time.monotonic();view_count=len(training_views(cfg['loss']))
    with (out/'training.jsonl').open('w',buffering=1) as log:
        for epoch in range(1,cfg['epochs']+1):
            model.train();active=epoch>=cfg['loss']['tda_start_epoch'];model.tda.requires_grad_(active)
            epoch_start=time.monotonic();totals={};epoch_seen=0
            for indices in data.epoch('train',cfg['batch_size'],rng):
                batch=gpu.get(indices);mask=torch.tensor(data.temporal_mask(indices),device='cuda')
                optimizer.zero_grad(set_to_none=True);lrs=[g['lr'] for g in optimizer.param_groups]
                loss,parts=cached_step(model,batch,mask,cfg,epoch)
                grad=torch.nn.utils.clip_grad_norm_(backbone+heads,cfg['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();scheduler.step();step+=1;seen+=len(indices);epoch_seen+=len(indices)
                for k,v in dict(loss=loss,**parts).items():totals[k]=totals.get(k,0.)+v*len(indices)
                if step%10==0:
                    status=dict(state='training',epoch=epoch,step=step,total_steps=total_steps,anchor_exposures=seen,view_exposures=view_count*seen,
                        tda_enabled=active,learning_rates_used=lrs,gradient_norm=float(grad),loss=loss,parts=parts,
                        elapsed_seconds=time.monotonic()-started,utc=datetime.now(timezone.utc).isoformat(),wandb_url=wandb_run.url)
                    write_json(out/'status.json',status);print('TRAIN',json.dumps(status),flush=True)
                    wandb_run.log(dict(training_step=step,**flatten_metrics('train',dict(loss=loss,epoch=epoch,tda_enabled=active,gradient_norm=float(grad),backbone_lr=lrs[0],head_lr=lrs[1],**parts))))
                if time.monotonic()-started>cfg['max_training_seconds']:
                    save_checkpoint(ckpt/'last.pt',model,optimizer,cfg,epoch,step,seen,initial)
                    raise TimeoutError('Plain MACE training exceeded the configured time budget; incomplete checkpoint preserved')
            validation=validate(model,data,gpu,cfg,epoch)
            record=dict(epoch=epoch,steps=step,anchor_exposures=seen,train_loss=totals['loss']/epoch_seen,
                train_metrics={k:v/epoch_seen for k,v in totals.items()},validation=validation,tda_enabled=active,epoch_seconds=time.monotonic()-epoch_start)
            log.write(json.dumps(record,allow_nan=False)+'\n');print('VALIDATION',json.dumps(record),flush=True)
            wandb_run.log({'training_step':step,**flatten_metrics('validation',validation)})
            save_checkpoint(ckpt/'last.pt',model,optimizer,cfg,epoch,step,seen,validation)
            if active and validation['loss']<best:
                best=validation['loss'];best_epoch=epoch
                save_checkpoint(ckpt/'best.pt',model,optimizer,cfg,epoch,step,seen,validation)
    selected=torch.load(ckpt/'best.pt',map_location='cpu',weights_only=False);selected.pop('optimizer');torch.save(selected,out/'best.pt')
    write_json(out/'training_summary.json',dict(stop_reason='maximum_epochs',epochs_completed=epoch,partial_epoch=False,steps=step,
        anchor_exposures=seen,view_exposures=view_count*seen,seconds=time.monotonic()-started,best_epoch=best_epoch,selected_epoch=best_epoch,
        checkpoint_selection=f"lowest fixed validation loss among epochs {cfg['loss']['tda_start_epoch']}–{cfg['epochs']}",best_validation=selected['validation'],selected_validation=selected['validation'],initial_validation=initial))


def run(cfg,stage):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    def interrupted(signum,frame):raise InterruptedError(f'Plain MACE interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    wandb_run=None
    from src.experiment_runner.tracking import tracked_run
    try:
        with tracked_run(out,kind='training' if stage in ('train','all') else 'analysis',configs=[Path(sys.argv[sys.argv.index('--config')+1])],command=[sys.executable,*sys.argv]):
            if stage in ('prepare','all'):
                if cfg['protocol']=='thermal80':
                    from src.data_utils.mace_relaxed import prepare
                else:
                    from src.data_utils.pretrained_mace import prepare
                prepare(cfg);summarize_data(cfg)
            if stage=='preflight':
                from src.training_methods.mace_preflight import preflight
                preflight(cfg)
            if stage in ('train','all'):
                write_json(out/'status.json',dict(state='initializing_training',utc=datetime.now(timezone.utc).isoformat()))
                summarize_data(cfg);wandb_run=start_wandb(cfg,out);train(cfg,wandb_run)
                wandb_run.summary.update(json.loads((out/'training_summary.json').read_text()));wandb_run.finish();wandb_run=None
            if stage in ('analysis','all'):
                from src.analysis.pretrained_mace_adapter import export_encoder,write_report
                export_encoder(cfg)
                subprocess.run([sys.executable,'-m','src.analysis.pipeline',cfg['analysis_config']],check=True)
                write_report(cfg)
            write_json(out/'status.json',dict(state='complete',stage=stage,utc=datetime.now(timezone.utc).isoformat()))
    except BaseException:
        if wandb_run is not None:wandb_run.finish(exit_code=1)
        write_json(out/'status.json',dict(state='failed',traceback=traceback.format_exc()));raise


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--stage",choices=["prepare","preflight","train","analysis","potential-audit","all"],default="all")
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text())
    if cfg.get("protocol") in ("denoising80", "denoising80_reuse"):
        from src.training_methods.mace_denoising import run as run_denoising
        run_denoising(cfg, args.stage)
        return
    if cfg.get("protocol") == "temporal80":
        from src.training_methods.mace_temporal import run as run_temporal
        run_temporal(cfg, args.stage)
        return
    if cfg.get("protocol") not in ("plain80","thermal80"):
        raise ValueError("Supported protocols: plain80, thermal80, temporal80, denoising80, denoising80_reuse. Retired experiments require their recorded source snapshot.")
    run(cfg,args.stage)


if __name__=="__main__":main()
