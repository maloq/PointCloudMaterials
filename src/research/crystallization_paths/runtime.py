"""Exact epoch resumes with resident timelines and strictly open-loop validation."""
import json
import copy
import math
import signal
import time
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json
from src.research.crystallization_transfer.training import configure_training,epoch_batch
from src.research.local_predictability.metrics import source_weights
from src.experiment_runner.metric_docs import write_metric_table
from .model import Forecaster
from .metrics import dense_brier,path_scores,summarize


def make_model(spec):
    if spec.get('context_layout')=='cuboctahedral_v1':
        from src.research.structured_context.model import StructuredForecaster
        return StructuredForecaster(spec)
    if 'information_context' in spec:
        from src.research.context_night.context import ContextForecaster
        return ContextForecaster(spec)
    if spec.get('protocol')=='path_refinement_v2':
        from .refined_model import RefinedForecaster
        return RefinedForecaster(spec)
    return Forecaster(spec)


def teacher_probability(spec,step,per_epoch,updates):
    if spec.get('protocol')!='path_refinement_v2':return max(0.,1-step/max(1,updates//2))
    if spec['teacher_mode']=='likelihood':return 1.
    if spec['teacher_mode']=='free':return 0.
    if spec['teacher_mode']=='scheduled':return max(0.,1-step/(spec['teacher_epochs']*per_epoch))
    raise ValueError(spec['teacher_mode'])


def selected_indices(corpus,role,count,seed):
    ids=np.asarray(corpus.splits[role]);chosen=[]
    for source in np.unique(corpus.source_ids[ids]):
        group=ids[corpus.source_ids[ids]==source]
        rng=np.random.default_rng(np.random.SeedSequence([seed,int(source),718]))
        chosen.extend(rng.permutation(group)[:count])
    return sorted(chosen)


@torch.no_grad()
def initialize(model,data):
    if model.spec.get('protocol')=='path_refinement_v2':
        model.target_mean.copy_(data.mean);model.target_scale.copy_(data.scale)
    ids=selected_indices(data.corpus,'train',8,data.plan['config']['seed'])
    observed=data.observed(ids);x,w=model.context.inputs(observed['features'],observed['geometry'])
    model.context.normalization.calibrate(x,w)
    # Train-only dense hazards provide a non-collapsed starting event distribution.
    counts=torch.zeros(129,device=data.device,dtype=torch.float64)
    for group in data.corpus.groups.values():
        event=data.event_bins(group)
        counts+=torch.bincount(event,minlength=129).double()/len(group)/len(data.corpus.groups)
    counts=(counts+1e-6)/(1+129e-6)
    cdf=counts[:128].cumsum(0);model.spec['training_event_cdf']=cdf.cpu().tolist()
    # Position-dependent base hazard enters the event bias via its average; the position tokens can refine it.
    if model.method=='mixture' and model.spec.get('mixture_style')=='stratified':
        mass=(counts[None]*model.event_support).sum(-1)
        model.mixing.bias.copy_(mass.float().log());model.event_decoder.bias.copy_(counts.float().log())
    elif model.method!='diffusion':
        hazard=counts[:128]/counts.flip(0).cumsum(0).flip(0)[:128]
        initial=torch.logit(hazard.float()).reshape(32,4).mean(0)
        model.decode.bias[-4:].copy_(initial)


@torch.no_grad()
def predict(model,data,indices,samples,*,retain=False):
    model.eval();records={'cdf':[],'event':[],'path_scores':[],'persistence_scores':[]};examples=[]
    # Fixed batch size and seed make checkpoint selection reproducible, without changing training RNG.
    with torch.random.fork_rng(devices=[data.device.index or 0] if data.device.type=='cuda' else []):
        torch.manual_seed(data.plan['config']['seed']+918)
        for start in range(0,len(indices),64):
            ids=indices[start:start+64];target=data.targets(ids)
            paths,cdf=model.forecast(data.observed(ids),samples=samples,diffusion_steps=model.spec.get('diffusion_steps',16))
            if not torch.isfinite(paths).all() or not torch.isfinite(cdf).all():raise FloatingPointError(f'Nonfinite {model.method} free rollout')
            records['cdf'].append(cdf.cpu().numpy());records['event'].append(target['event'].cpu().numpy())
            records['path_scores'].append(path_scores(paths,target['state']).cpu().numpy())
            records['persistence_scores'].append(path_scores(data.baseline(ids)[:,None],target['state']).cpu().numpy())
            if retain and start<64:
                examples.append(dict(indices=np.asarray(ids),paths=paths.cpu().numpy().astype(np.float16),
                    target=target['state'].cpu().numpy(),mean=data.mean.cpu().numpy(),scale=data.scale.cpu().numpy()))
    result={k:np.concatenate(v) for k,v in records.items()}
    if retain:result['examples']=examples[0]
    return result


def fit(plan,spec,data,deadline):
    config=plan['config'];root=resolve_path(config['output'])/'technical/runs'/spec['name'];root.mkdir(parents=True,exist_ok=True)
    data.set_context(spec)
    # Recreate the full training population for every fit and resume.
    data.corpus.groups={s:ids.copy() for s,ids in data.full_training_groups.items()}
    data.corpus.splits['train']=list(data.full_training_indices)
    summary=configure_training(data.corpus,spec);save_json(root/'training-population.json',summary)
    torch.manual_seed(config['seed']);model=make_model(dict(spec)).to(data.device);initialize(model,data)
    if 'information_context' in spec:model.initialize_information(data)
    ema=copy.deepcopy(model) if spec.get('ema_decay',0)>0 else None
    optimizer=torch.optim.AdamW(model.parameters(),lr=spec['head_lr'],weight_decay=spec['weight_decay'])
    updates=summary['updates'];per_epoch=summary['updates_per_epoch'];step=0;best=float('inf');stale=0;early_stopped=False
    stop=False
    def request_stop(*_):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGUSR1,request_stop)
    last=root/'last.pt';best_path=root/'best.pt';started=time.monotonic()
    if last.exists():
        saved=torch.load(last,map_location=data.device,weights_only=False)
        if saved['plan_identity']!=plan['identity'] or saved['spec']!=spec:raise ValueError('Path resume identity changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);step=saved['step'];best=saved['best']
        stale=saved['stale'];early_stopped=saved['early_stopped']
        if ema is not None:ema.load_state_dict(saved['ema'])
        torch.set_rng_state(saved['rng'].cpu())
        if data.device.type=='cuda':torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    def checkpoint():
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,best=best,
            spec=spec,plan_identity=plan['identity'],mean=data.mean,scale=data.scale,rng=torch.get_rng_state(),
            stale=stale,early_stopped=early_stopped,ema=ema.state_dict() if ema is not None else None,
            cuda_rng=torch.cuda.get_rng_state() if data.device.type=='cuda' else None),root/'last.building.pt')
        (root/'last.building.pt').replace(last)
    selection=selected_indices(data.corpus,'selection',config['selection_per_source'],config['seed'])
    selection_weights=source_weights(data.corpus.source_ids[selection])
    save_json(root/'model.json',dict(spec=spec,parameters=sum(p.numel() for p in model.parameters()),
        normalization='Fixed training-only source-balanced feature and target moments',
        selection='Dense integrated event Brier score on selection sources; free rollout'))
    while step<updates and not early_stopped:
        if stop or time.time()>deadline-600:
            checkpoint();save_json(root/'status.json',dict(state='checkpointed',step=step,updates=updates,
                best_selection_brier=best if math.isfinite(best) else None));return False
        ids=epoch_batch(data.corpus,step);model.train()
        warmup=per_epoch;factor=min((step+1)/warmup,1.) if step<warmup else .05+.95*.5*(1+math.cos(math.pi*(step-warmup)/max(1,updates-warmup)))
        for group in optimizer.param_groups:group['lr']=spec['head_lr']*factor
        teacher=teacher_probability(spec,step,per_epoch,updates)
        target=data.targets(ids);target['present']=data.baseline(ids)[:,0]
        loss=model.loss(data.observed(ids),target,teacher)
        weights=torch.as_tensor(data.corpus.training_weights[ids],device=data.device)
        loss=(loss*weights).mean()
        if not torch.isfinite(loss):raise FloatingPointError(f'{spec["name"]}: nonfinite training loss at {step}')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);optimizer.step();step+=1
        if ema is not None:
            with torch.no_grad():
                for averaged,current in zip(ema.parameters(),model.parameters()):averaged.lerp_(current,1-spec['ema_decay'])
                for averaged,current in zip(ema.buffers(),model.buffers()):averaged.copy_(current)
        if step%128==0:
            record=dict(state='running',step=step,updates=updates,epochs=step/per_epoch,loss=float(loss.detach()),
                lr=spec['head_lr']*factor,teacher_probability=teacher,gradient_norm=float(grad),seconds=time.monotonic()-started)
            save_json(root/'status.json',record)
            with (root/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        if step%per_epoch==0 or step==updates:
            evaluated=ema if ema is not None else model
            prediction=predict(evaluated,data,selection,config['selection_samples'])
            score=float(selection_weights@dense_brier(prediction['cdf'],prediction['event']))
            physical=float(selection_weights@prediction['path_scores'][:,:,1:4].mean((1,2)))
            if score<best:
                best=score;stale=0;torch.save(dict(model=evaluated.state_dict(),spec=evaluated.spec,step=step,selection_brier=score,selection_physical_mse=physical,
                    plan_identity=plan['identity'],mean=data.mean,scale=data.scale),best_path)
            else:stale+=1
            early_stopped=spec.get('patience',0)>0 and stale>=spec['patience'] and step>=spec['minimum_epochs']*per_epoch
            with (root/'validation.jsonl').open('a') as stream:stream.write(json.dumps(dict(step=step,selection_brier=score,selection_physical_mse=physical,best=best,stale=stale))+'\n')
            checkpoint()
    if stop or time.time()>deadline-1800:
        save_json(root/'status.json',dict(state='checkpointed',stage='evaluation',step=step,updates=updates));return False
    selected=torch.load(best_path,map_location=data.device,weights_only=False);model.load_state_dict(selected['model'])
    save_json(root/'status.json',dict(state='evaluating',step=step,selected_step=selected['step'],best_selection_brier=best))
    cal=data.corpus.splits['calibration'];test=data.corpus.splits['test']
    calibration=predict(model,data,cal,config['evaluation_samples']);prediction=predict(model,data,test,config['evaluation_samples'],retain=True)
    examples=prediction.pop('examples');np.savez_compressed(root/'sample-trajectories.npz',**examples)
    np.savez_compressed(root/'predictions.npz',test_indices=test,calibration_indices=cal,
        **{f'test_{k}':v for k,v in prediction.items()},**{f'calibration_{k}':v for k,v in calibration.items()})
    metrics=summarize(data.corpus,test,cal,prediction,calibration)
    if 'information_context' in spec:
        from src.research.context_night.metrics import short_scores
        metrics['short_horizon']=short_scores(data,test,cal,prediction,calibration)
    metrics['training']=dict(summary,selected_step=selected['step'],best_selection_brier=best)
    metrics['training'].update(updates=step,complete_epochs=step//per_epoch,partial_epoch_updates=step%per_epoch,
        samples=(step//per_epoch)*summary['eligible_windows'],early_stopped=early_stopped)
    save_json(root/'metrics.json',metrics)
    flat={key:{str(row['horizon_ps']):row for row in metrics[key]} for key in ('classification','timing','spatial')}
    flat.update({key:metrics[key] for key in ('fine_timing','path','test_event_nll','dense_integrated_brier',
        'restricted_mean_time_mae_ps','physical_persistence_standardized_mse','embedding_persistence_standardized_mse','training')})
    family='crystallization_paths_refinement' if spec.get('protocol')=='path_refinement_v2' else 'crystallization_paths'
    if 'information_context' in spec:
        flat['short_horizon']=metrics['short_horizon'];family='context_night'
    if spec.get('context_layout')=='cuboctahedral_v1':family='structured_context'
    write_metric_table(flat,resolve_path(config['output']),family=family,name=spec['name'])
    save_json(root/'status.json',dict(state='complete',step=step,selected_step=selected['step'],
        best_selection_brier=best,best_selection_physical_mse=selected['selection_physical_mse'],early_stopped=early_stopped,
        test_event_nll=metrics['test_event_nll'],dense_integrated_brier=metrics['dense_integrated_brier']))
    return True
