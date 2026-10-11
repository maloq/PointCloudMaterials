"""Matched likelihood training and short simulator tangent supervision."""
from collections import defaultdict
from copy import deepcopy
import math
import os
import time
import numpy as np
import torch
from src.experiment_runner.checkpoints import TrainingState
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from src.project_runtime.paths import resolve_path
from .data import FAMILY, read, root, load, atomic_save, digest, sha, write_json
from .model import MotionModel, mixture_nll, noncollapse
from . import teacher


class Sampler:
    """Ancestor -> descendant source -> local observation, all uniformly."""
    def __init__(self, rows, role):
        groups=defaultdict(lambda:defaultdict(list))
        for i,r in enumerate(rows):
            if r['role']==role:
                groups[r['ancestor']][r['source']].append(i)
        self.groups=[list(x.values()) for x in groups.values()]
        if not self.groups:
            raise ValueError(f'No observations for {role}')

    def draw(self, rng, count):
        result=[]
        for _ in range(count):
            source=self.groups[int(rng.integers(len(self.groups)))]
            rows=source[int(rng.integers(len(source)))]
            result.append(rows[int(rng.integers(len(rows)))])
        return np.asarray(result)


@torch.no_grad()
def initialize(model, states, micro):
    encoder=model.geometry.encoder
    pooled=torch.cat([encoder.pooled_graph(model.geometry.graph(x[...,:3])) for x in states.split(micro)])
    encoder.pooled_mean.copy_(pooled.mean(0))
    encoder.pooled_scale.copy_(pooled.std(0,unbiased=False).clamp_min(1e-5))


def short_loss(model, target_geometry, records, c, seed):
    """Compare conditional mean latent endpoints and their input JVPs.

    The teacher uses independent BAOAB noise shots. Its encoder is an EMA
    target with stop-gradient parameters. Only 100 fs targets enter fitting;
    10/20/50 fs observations remain withheld diagnostics.
    """
    settings=c['teacher'];selected=[settings['save_fs'].index(h) for h in settings['training_fs']]
    x=torch.stack([r['state'] for r in records]).float().cuda()
    basis=torch.stack([r['basis'] for r in records]).float().cuda()
    future=torch.stack([r['future'][:,selected] for r in records]).float().cuda()
    tangent=torch.stack([r['tangent'][:,selected] for r in records]).float().cuda()
    b,shots,nh=future.shape[:3]
    q=future[...,:3].reshape(-1,80,3)
    with torch.no_grad():
        target=target_geometry(q).reshape(b,shots,nh,-1).mean(1)
    horizons=[h/1000 for h in settings['training_fs']]
    def predicted(state):
        z=model.encode(state,create_graph=True)
        return model.path(z,horizons,particles=c['model']['particles'],seed=seed)[...,:model.d].mean(1)
    prediction=predicted(x)
    value=.5*(prediction-target).square().mean()
    response=value.new_zeros(())
    if model.arm=='derivative':
        for k in range(2):
            direction=tangent[...,:3,k].reshape(-1,80,3)
            with torch.enable_grad():
                _,dt=torch.autograd.functional.jvp(target_geometry,q,direction,create_graph=False,strict=True)
            dt=dt.reshape(b,shots,nh,-1).mean(1).detach()
            _,dp=torch.autograd.functional.jvp(predicted,x,basis[...,k],create_graph=True,strict=True)
            # Relative JVP error prevents the arbitrary scale of the learned
            # latent and perturbation units from dominating the likelihood.
            response=response+.25*(dp-dt).square().mean()/dt.square().mean().clamp_min(1e-6)
    return value,response


def future_backward(model, states, targets, c, *, microbatch, seed):
    """Accumulate one batch's exact objective, including its global covariance.

    For split execution, a detached structural pass caches dL_noncollapse/ds
    over all256 examples. Recomputed halves inject those cached cotangents.
    Each half consumes its slice of the reference full-batch noise stream.
    """
    count=len(states);parts=list(range(0,count,microbatch));weight=c['training']['noncollapse_weight']
    if len(parts)>1:
        with torch.no_grad():
            structure=torch.cat([model.geometry(states[i:i+microbatch,...,:3]) for i in parts])
        structure.requires_grad_(True)
        nc=noncollapse(structure)
        cotangent=torch.autograd.grad(nc,structure)[0].detach()
        nc_value=float(nc.detach())
    nll_value=0.
    for offset in parts:
        x=states[offset:offset+microbatch]
        z=model.encode(x,create_graph=True)
        paths=model.path(z,c['horizons_ps'],particles=c['model']['particles'],seed=seed,
                         random_batch_size=count,sample_offset=offset)
        mu,lv=model.prediction(paths)
        nll=mixture_nll(mu,lv,targets[offset:offset+len(x)]).mean()
        if len(parts)>1:
            # This linear surrogate supplies the exact global-loss gradient;
            # report nc_value, not the surrogate's scalar value.
            loss=nll*len(x)/count+weight*(z[:,:model.d]*cotangent[offset:offset+len(x)]).sum()
        else:
            nc=noncollapse(z[:,:model.d]);nc_value=float(nc.detach())
            loss=nll+weight*nc
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite future or global noncollapse objective')
        loss.backward();nll_value+=float(nll.detach())*len(x)/count
    return nll_value,nc_value


def fit(c, plan, arm, seed, deadline, *, runtime_microbatch=None, execution_profile=None):
    torch.set_num_threads(2);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    folder=root(c)/'analyses'/f'{arm}-{seed}';tech=folder/'technical';tech.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(plan=plan['identity'],arm=arm,seed=seed,
                         config=c,
                         code=read(root(c)/'technical/code-identity.json')['identity']))
    metric_folder=folder if execution_profile is None else root(c)/'analyses'/(
        f"{arm}-{seed}-profile-{execution_profile['implementation_identity'][:8]}")
    (metric_folder/'technical').mkdir(parents=True,exist_ok=True)
    if (tech/'complete.json').exists():
        if read(tech/'complete.json')['identity']!=identity:
            raise ValueError('Changed completed fit')
        return
    data=load(c);rows=data['rows'];bank=teacher.load(c,plan)
    sampler=Sampler(rows,'train');short_sampler=Sampler(bank,'train')
    rng=np.random.default_rng(seed);cfg=c['training'];micro=cfg['microbatch'] if runtime_microbatch is None else runtime_microbatch
    if micro not in (128,256) or cfg['batch_size']!=256:
        raise ValueError('Supported execution profiles are256 or128x2, effective batch256')
    selection=np.unique(Sampler(rows,'selection').draw(np.random.default_rng(c['seed']+91),cfg['selection_rows']))
    model=MotionModel(c,arm).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    model.inference_microbatch=micro
    initialize(model,data['state'][sampler.draw(np.random.default_rng(c['seed']),256)].cuda(),micro)
    target_geometry=deepcopy(model.geometry).eval().requires_grad_(False)
    center=data['mean'].float().cuda();scale=data['scale'].float().cuda()
    history=[];best=math.inf;selected_update=0;start=1
    if not (tech/'last.pt').exists() and c.get('resume_campaign'):
        previous=resolve_path(c['resume_campaign'])/'analyses'/f'{arm}-{seed}'/'technical'
        if (previous/'last.pt').exists():
            prior_identity=read(previous/'identity.json')['identity']
            state=TrainingState.read(previous/'last.pt',identity=prior_identity,device='cpu')
            selected=torch.load(previous/'best.pt',map_location='cpu',weights_only=False)
            if selected['identity']!=prior_identity:
                raise ValueError('Continuation selected checkpoint has wrong identity')
            for key in ('encoder','model','teacher','horizons_ps','seed'):
                if selected['config'][key]!=c[key]:
                    raise ValueError(f'Checkpoint continuation changes {key}')
            provenance=dict(previous_folder=str(previous),previous_identity=prior_identity,
                last_sha256=sha(previous/'last.pt'),best_sha256=sha(previous/'best.pt'),
                continued_from_update=state.payload['update'],new_total_updates=cfg['updates'],
                optimizer_and_rng_preserved=True,
                previous_tracking=read(previous/'wandb.json') if (previous/'wandb.json').exists() else None)
            state.payload['identity']=identity
            selected.update(identity=identity,config=c)
            atomic_save(tech/'best.pt',selected)
            state.save(tech/'last.pt')
            write_json(tech/'continuation.json',provenance)
    if (tech/'last.pt').exists():
        state=TrainingState.read(tech/'last.pt',identity=identity,device='cuda')
        state.restore(model,optimizer,rng,restore_torch_rng=True)
        payload=state.payload;target_geometry.load_state_dict(payload['target_geometry'])
        history=payload['history'];best=payload['best'];selected_update=payload['selected'];start=payload['update']+1
    execution=dict(profile=execution_profile,microbatch=micro,effective_batch=cfg['batch_size'],
        accumulation_steps=math.ceil(cfg['batch_size']/micro),start_update=start,pid=os.getpid(),
        host=os.uname().nodename,allocation=os.environ.get('SLURM_JOB_ID'),
        device=torch.cuda.get_device_name(),started_unix=time.time(),
        reference_identity=identity,reference_code_identity=read(root(c)/'technical/code-identity.json')['identity'],
        state='running',global_noncollapse=True,full_batch_noise_slices=True)
    execution_path=tech/'executions'/f'{os.uname().nodename}-{os.getpid()}.json'
    write_json(execution_path,execution)
    def predictive(indices):
        result=[]
        model.eval()
        # Encode may split internally, but scoring/noise use the reference
        #256-sized blocks independent of the runtime memory profile.
        for chunk in np.array_split(indices,max(1,math.ceil(len(indices)/cfg['microbatch']))):
            with torch.no_grad():
                z=model.encode(data['state'][chunk].cuda(),create_graph=False)
                paths=model.path(z,c['horizons_ps'],particles=8,seed=c['seed']+19)
                mu,lv=model.prediction(paths)
                y=(data['target'][chunk].float().cuda()-center)/scale
                result.append(mixture_nll(mu,lv,y).cpu())
        return torch.cat(result).mean().item()
    def checkpoint(update):
        return dict(identity=identity,model=model.state_dict(),target_geometry=target_geometry.state_dict(),
                    mean=data['mean'],scale=data['scale'],update=update,selection_nll=best,arm=arm,config=c,
                    execution=execution)
    context=dict(identity=identity,arm=arm,seed=seed,training=cfg,
                 inputs=read(root(c)/'technical/prediction-context.json'),
                 selector='inner-original-training-ancestor validation NLL of fixed future24 statistics',
                 parameters=sum(p.numel() for p in model.parameters()),
                 objective='future likelihood + short conditional mean latent endpoint + noncollapse; derivative arm adds local JVP loss',
                 latent_noise='Monte Carlo latent SDE particles; not paired to physical BAOAB Brownian paths',
                 targets='12 radial + 12 angular powers; future only; no RFF, PTM, species or reconstruction')
    context['runtime_execution']='training.microbatch is the canonical reference; actual microbatch/accumulation and producer hashes are recorded per segment in technical/executions and checkpoints'
    if (tech/'continuation.json').exists():
        context['continuation']=read(tech/'continuation.json')
    write_json(tech/'identity.json',context)
    with online_training(c['wandb'],run_id='cmotion-'+identity[:16],name=f'{arm}-{seed}',config=context,
                         folder=tech,receipt_path=tech/'wandb.json',job_type='encoder',group=c['protocol'],
                         receipt_fields=dict(identity=identity)) as run:
        run.summary.update({'execution/latest_microbatch':micro,'execution/effective_batch':cfg['batch_size'],
                            'execution/profile_receipt':str(execution_path)})
        if start==1:
            best=predictive(selection);atomic_save(tech/'best.pt',checkpoint(0))
        for update in range(start,cfg['updates']+1):
            if time.time()>=deadline-300:
                raise TimeoutError('Training budget reached; last checkpoint retained')
            begin=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
            indices=sampler.draw(rng,cfg['batch_size']);totals=np.zeros(4)
            totals[:2]=future_backward(model,data['state'][indices].cuda(),
                (data['target'][indices].float().cuda()-center)/scale,c,microbatch=micro,
                seed=int(rng.integers(2**31)))
            # All arms receive the identical short value bank. Tangents are the
            # only additional teacher signal in the derivative treatment.
            short_ids=short_sampler.draw(rng,cfg['short_batch'])
            for i in short_ids:
                lv,ld=short_loss(model,target_geometry,[bank[i]],c,int(rng.integers(2**31)))
                loss=(cfg['short_weight']*lv+cfg['derivative_weight']*ld)/len(short_ids)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'{arm}/{update}: nonfinite tangent objective')
                loss.backward();totals[2:]+=np.array([float(lv.detach()),float(ld.detach())])/len(short_ids)
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True))
            optimizer.step()
            with torch.no_grad():
                for target,current in zip(target_geometry.parameters(),model.geometry.parameters(),strict=True):
                    target.lerp_(current,.01)
            row=dict(update=update,future_nll=float(totals[0]),noncollapse=float(totals[1]),
                     short_latent_half_mse=float(totals[2]),tangent_half_mse=float(totals[3]),gradient_norm=norm)
            if update%cfg['selection_every']==0:
                row['selection_nll']=predictive(selection)
                if row['selection_nll']<best:
                    best=row['selection_nll'];selected_update=update;atomic_save(tech/'best.pt',checkpoint(update))
            history.append(row)
            TrainingState.capture(model,optimizer,rng,identity=identity,capture_torch_rng=True,
                target_geometry=target_geometry.state_dict(),history=history,best=best,selected=selected_update,
                update=update,execution=execution).save(tech/'last.pt')
            run.log(row,step=update)
            write_json(tech/'progress.json',dict(state='training',update=update,total=cfg['updates'],
                       last_update_seconds=time.monotonic()-begin,selection_nll=best))
            if update==start:
                execution['first_update_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
                write_json(execution_path,execution)
            if update==1 or update%10==0:
                print(arm,seed,row,'seconds',round(time.monotonic()-begin,2),flush=True)
        selected=torch.load(tech/'best.pt',map_location='cuda',weights_only=False)
        model.load_state_dict(selected['model']);model.eval()
        from .evaluate import evaluate
        metrics=evaluate(c,plan,model,data,bank,metric_folder,deadline)
        run.summary.update(dict(selected_update=selected_update,selection_nll=best,
            **{f"evaluation/{r['track']}/{r['horizon_ps']}ps/{k}":v for r in metrics for k,v in r.items()
               if k in ('nll','mean_mse','ballistic_mean_mse')}))
    write_metric_rows(history,metric_folder,family=FAMILY,name='learning',columns=[
        'update','future_nll','noncollapse','short_latent_half_mse','tangent_half_mse','gradient_norm','selection_nll'])
    write_json(tech/'complete.json',dict(identity=identity,state='complete',checkpoint_sha256=sha(tech/'best.pt'),
               metrics=metrics,metrics_root=str(metric_folder),execution=execution))
    execution.update(state='complete',end_update=cfg['updates'],finished_unix=time.time())
    write_json(execution_path,execution)
