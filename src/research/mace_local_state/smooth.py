"""Nonlinear physical state maps with direct, source-balanced temporal penalties.

This is the frozen-backbone stage. No teacher-feature retention, forecast target,
temporal history, or manifold-motion constraint is applied in this protocol.
"""
import csv
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from .smooth_data import load


FAMILIES = {'bond_order': slice(0,16), 'instantaneous_TDA_H0': slice(16,32),
            'instantaneous_TDA_H1': slice(32,96), 'instantaneous_TDA_H2': slice(96,160)}


def within_covariance(z, weights, context):
    """Population covariance of residuals about source/frame context means."""
    _, inverse = torch.unique(context, sorted=True, return_inverse=True)
    n = int(inverse.max())+1
    total = z.new_zeros(n).index_add(0, inverse, weights)
    sums = z.new_zeros(n,z.shape[1]).index_add(0,inverse,z*weights[:,None])
    residual = z-sums[inverse]/total[inverse,None]
    return (residual*weights[:,None]).T@residual/weights.sum()


class StateMap(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.mapping = (nn.Sequential(nn.Linear(256,128), nn.SiLU(), nn.Linear(128,64),
                                     nn.SiLU(), nn.Linear(64,dimension)) if dimension else nn.Identity())
        self.register_buffer('initial_scale', torch.ones(dimension or 256))
        self.readout = nn.Sequential(nn.Linear(dimension or 256,128), nn.SiLU(), nn.Linear(128,160))

    def forward(self, x):
        z = self.mapping(x)/self.initial_scale
        return z, self.readout(z)


def tensors(data, split, config):
    pairs = np.flatnonzero(data['split']==split)
    rows = np.ravel(np.c_[2*pairs,2*pairs+1])
    device = config['device']
    low = data['raw_target'][rows,4] < config['low_order_threshold']
    lag = np.round(data['lag'][pairs],9)
    pair_weights = data['weights'][pairs].astype(np.float64)
    if split != 0: pair_weights[:] = 1.
    temporal = []; counts = []
    for subset in ('all','low_order'):
        eligible = lag <= config['maximum_training_lag_ps']+1e-9
        if subset == 'low_order': eligible &= low[::2]&low[1::2]
        if not eligible.any(): raise ValueError(f'No eligible temporal pairs: split={split}, subset={subset}')
        w = np.zeros(len(pairs))
        bins = np.unique(lag[eligible])
        for value in bins:
            ids = eligible & (lag==value)
            w[ids] = pair_weights[ids]/pair_weights[ids].sum()/len(bins)
            counts.append(dict(subset=subset,lag_ps=float(value),pairs=int(ids.sum())))
        temporal.append(w)
    return dict(x=torch.as_tensor(data['embedding'][rows,:256],device=device),
        y=torch.as_tensor(data['target'][rows,:160],device=device),
        weights=torch.as_tensor(np.repeat(pair_weights,2),dtype=torch.float32,device=device),
        context=torch.as_tensor(data['context'][rows],device=device),
        low=torch.as_tensor(low,device=device),
        temporal=torch.as_tensor(np.stack(temporal),dtype=torch.float32,device=device),
        pairs=pairs,rows=rows,counts=counts)


def objective(model, batch, temporal_weight, covariance_weight):
    z, prediction = model(batch['x'])
    w = batch['weights']; error = (prediction-batch['y']).square()
    physical = torch.stack([(error[:,s].mean(1)*w).sum()/w.sum() for s in FAMILIES.values()])
    low = batch['low']
    cov = within_covariance(z,w,batch['context'])
    cov_low = within_covariance(z[low],w[low],batch['context'][low])
    energies = (z[::2]-z[1::2]).square().sum(1)
    traces = torch.stack([cov.trace(),cov_low.trace()])
    if torch.any(traces <= 1e-12) or not torch.isfinite(traces).all():
        raise FloatingPointError(f'Collapsed/nonfinite within-context variance: {traces.detach().cpu().tolist()}')
    temporal = ((batch['temporal']*energies[None,:]).sum(1)/(2*traces)).mean()
    eye = torch.eye(z.shape[1],device=z.device)
    calibration = ((cov-eye).square().sum()+(cov_low-eye).square().sum())/(2*z.shape[1])
    value = physical.mean()+temporal_weight*temporal
    if model.dimension: value = value+covariance_weight*calibration
    if not torch.isfinite(value): raise FloatingPointError('Nonfinite smooth-state objective')
    return value, dict(physical=physical,temporal=temporal,calibration=calibration,
                      within_variance=traces), z


def specifications(config):
    for seed in config['seeds']:
        yield dict(name=f'reference-seed{seed}',dimension=0,temporal_weight=0.,seed=seed)
        for dimension in config['dimensions']:
            for weight in config['temporal_weights']:
                yield dict(name=f'd{dimension}-temporal{weight:g}-seed{seed}',dimension=dimension,
                           temporal_weight=weight,seed=seed)


def fingerprint(config):
    return dict(config=config, implementation={str(p):sha256(p) for p in
        [Path(__file__),Path(__file__).with_name('smooth_data.py')]},
        cache_sha256=sha256(Path(config['cache'])/'features.npz'))


def fit(config, root):
    torch.set_num_threads(config['cpu_threads'])
    data = load(config); train = tensors(data,0,config); val = tensors(data,1,config)
    identity = fingerprint(config)
    write_json(root/'technical/training-population.json', dict(train=train['counts'],validation=val['counts']))
    initial = torch.load(config['checkpoint'],map_location='cpu',weights_only=False)
    for spec in specifications(config):
        directory = root/'technical'/spec['name']; directory.mkdir(exist_ok=True)
        best_path = directory/'best.pt'; last_path = directory/'last.pt'
        torch.manual_seed(spec['seed'])
        model = StateMap(spec['dimension']).to(config['device'])
        with torch.no_grad():
            if spec['dimension']:
                z = model.mapping(train['x'])
                scale = within_covariance(z,train['weights'],train['context']).diag().sqrt()
                if torch.any(scale<=1e-8): raise ValueError('Degenerate initial nonlinear map')
                model.initial_scale.copy_(scale)
            else:
                state = {k.removeprefix('structure.'):v for k,v in initial['head_state'].items() if k.startswith('structure.')}
                model.readout.load_state_dict(state,strict=True)
        optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
        start_epoch=0; best=float('inf'); history=[]
        if last_path.exists():
            last=torch.load(last_path,map_location=config['device'],weights_only=False)
            if last['identity']!=identity or last['spec']!=spec: raise ValueError(f'Changed resume identity: {last_path}')
            model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer'])
            start_epoch=last['epoch']+1;best=last['best'];history=last['history']
        if start_epoch>config['epochs']: continue
        started=time.monotonic()
        for epoch in range(start_epoch,config['epochs']+1):
            if epoch:
                model.train();optimizer.zero_grad(set_to_none=True)
                warmup=min(1.,max(0.,(epoch-config['warmup_epochs'])/config['temporal_ramp_epochs']))
                value,metrics,_=objective(model,train,spec['temporal_weight']*warmup,config['covariance_weight'])
                value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
            if epoch%config['validation_every']==0 or epoch==config['epochs']:
                model.eval()
                with torch.no_grad():
                    _,metrics,_=objective(model,val,spec['temporal_weight'],config['covariance_weight'])
                physical=metrics['physical'].cpu().numpy()
                # Selection includes temporal quality. Variance calibration is a
                # training regularizer, not a change to the evaluation distance.
                score=float(physical.mean())+spec['temporal_weight']*float(metrics['temporal'])
                record=dict(epoch=epoch,physical={k:float(v) for k,v in zip(FAMILIES,physical,strict=True)},
                    temporal=float(metrics['temporal']),calibration=float(metrics['calibration']),selection_score=score)
                history.append(record)
                payload=dict(protocol=config['protocol'],identity=identity,spec=spec,epoch=epoch,
                    model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=min(best,score))
                if score<best:
                    best=score;torch.save(payload,best_path)
                temporary=directory/'last.pt.building';torch.save(payload,temporary);temporary.replace(last_path)
                write_json(directory/'history.json',history)
                status=dict(state='training',variant=spec['name'],epoch=epoch,epochs=config['epochs'],
                    best_validation=best,elapsed_seconds=time.monotonic()-started)
                write_json(root/'technical/fit-status.json',status)
        write_json(directory/'status.json',dict(state='complete',completed_epochs=config['epochs'],
            selected_epoch=torch.load(best_path,map_location='cpu',weights_only=False)['epoch'],
            elapsed_seconds=time.monotonic()-started))
        print('SMOOTH FIT',spec['name'],'seconds',time.monotonic()-started,'best',best,flush=True)
    write_json(root/'technical/fit-status.json',dict(state='complete',variants=len(list(specifications(config)))))


def jump_metrics(z, data, pairs, references, threshold, low_order=False):
    raw=data['raw_target']
    if low_order:
        pairs=pairs[(raw[2*pairs,4]<threshold)&(raw[2*pairs+1,4]<threshold)]
        references=references[raw[2*references,4]<threshold]
    if len(pairs)<2 or len(references)<2: raise ValueError('Insufficient jump/reference population')
    scale=float(2*np.var(z[2*references].astype(np.float64),axis=0).sum())
    if scale<=1e-12: raise FloatingPointError('Collapsed evaluation representation')
    energy=np.sum((z[2*pairs].astype(float)-z[2*pairs+1])**2,axis=1)
    d=np.sqrt(energy/scale)
    return dict(pairs=len(pairs),reference_rows=len(references),reference_squared_distance=scale,
        absolute_rms_increment=float(np.sqrt(energy.mean())),rms_jump=float(np.sqrt(np.mean(d*d))),
        median_jump=float(np.median(d)),p95_jump=float(np.quantile(d,.95)),max_jump=float(d.max()))


def evaluate(config, root):
    torch.set_num_threads(config['cpu_threads']);data=load(config)
    x=torch.as_tensor(data['embedding'][:,:256],device=config['device'])
    reference=data['reference_pair_ids']; rows=[]; reference_errors={}
    for spec in specifications(config):
        directory=root/'technical'/spec['name']
        checkpoint=torch.load(directory/'best.pt',map_location=config['device'],weights_only=False)
        if checkpoint['identity']!=fingerprint(config): raise ValueError(f'Changed evaluation identity: {directory}')
        model=StateMap(spec['dimension']).to(config['device']);model.load_state_dict(checkpoint['model']);model.eval()
        with torch.no_grad():z,prediction=model(x)
        z=z.cpu().numpy();prediction=prediction.cpu().numpy()
        np.savez(directory/'predictions.npz',embedding=z,prediction=prediction,selected_epoch=checkpoint['epoch'])
        report=dict(spec=spec,selected_epoch=checkpoint['epoch'],splits={})
        for split_name,split in [('validation',1),('development_test',2)]:
            pairs=np.flatnonzero(data['split']==split)
            selected=np.ravel(np.c_[2*pairs,2*pairs+1])
            errors=(prediction[selected]-data['target'][selected,:160])**2
            sources=np.repeat(data['source_id'][pairs],2)
            low_pairs=(data['raw_target'][2*pairs,4]<config['low_order_threshold'])&(data['raw_target'][2*pairs+1,4]<config['low_order_threshold'])
            low_rows=np.repeat(low_pairs,2)
            per_source=[];per_source_low=[]
            for source in np.unique(sources):
                mask=sources==source
                per_source.append(dict(source_id=int(source),**{k:float(errors[mask,s].mean()) for k,s in FAMILIES.items()}))
                if np.any(mask&low_rows):
                    per_source_low.append(dict(source_id=int(source),**{k:float(errors[mask&low_rows,s].mean()) for k,s in FAMILIES.items()}))
            physical={k:float(np.mean([v[k] for v in per_source])) for k in FAMILIES}
            physical_low={k:float(np.mean([v[k] for v in per_source_low])) for k in FAMILIES}
            if spec['dimension']==0:
                reference_errors[spec['seed'],split_name]=(physical,physical_low)
            mixed=jump_metrics(z,data,pairs,reference,config['low_order_threshold'])
            low=jump_metrics(z,data,pairs,reference,config['low_order_threshold'],True)
            report['splits'][split_name]=dict(physical=physical,low_order_physical=physical_low,
                jumps=mixed,low_order_jumps=low,per_source=per_source,per_source_low=per_source_low)
            base,base_low=reference_errors[spec['seed'],split_name]
            ratios={k:physical[k]/base[k] for k in FAMILIES}
            ratios.update({f'low_order_{k}':physical_low[k]/base_low[k] for k in FAMILIES})
            row=dict(variant=spec['name'],dimension=spec['dimension'] or 256,temporal_weight=spec['temporal_weight'],
                seed=spec['seed'],selected_epoch=checkpoint['epoch'],split=split_name,**physical,
                **{f'low_order_{k}':v for k,v in physical_low.items()},
                rms_jump=mixed['rms_jump'],p95_jump=mixed['p95_jump'],max_jump=mixed['max_jump'],
                low_order_rms_jump=low['rms_jump'],low_order_p95_jump=low['p95_jump'],
                worst_physical_error_ratio=max(ratios.values()),
                information_pass=int(max(ratios.values())<=1+config['retention_allowance']),
                jump_pass=int(mixed['rms_jump']<=config['target_jump']),
                low_order_jump_pass=int(low['rms_jump']<=config['target_jump']))
            rows.append(row)
        write_json(directory/'evaluation.json',report)
    snapshot_metric_docs(root,'mace_local_smooth')
    with (root/'tables/comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    validation=[r for r in rows if r['split']=='validation' and r['dimension']!=256]
    eligible=[r for r in validation if r['information_pass']]
    selected=min(eligible,key=lambda r:r['low_order_rms_jump']) if eligible else None
    summary=dict(state='complete',rows=len(rows),validation_selected=selected,
        caveat='Existing development test sources; no new blind generalization claim. Frozen snapshot maps only; no curvature/history training.')
    write_json(root/'technical/evaluation-status.json',summary)
    plot(config,root,rows)
    print('SMOOTH EVALUATED',summary,flush=True)


def plot(config,root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,split in zip(axes,['validation','development_test'],strict=True):
        for dimension in [256,*config['dimensions']]:
            selected=[r for r in rows if r['split']==split and r['dimension']==dimension]
            ax.scatter([r['low_order_rms_jump'] for r in selected],
                       [r['worst_physical_error_ratio'] for r in selected],label=f'{dimension}D')
        ax.axvline(config['target_jump'],color='k',linestyle='--',alpha=.5)
        ax.axhline(1+config['retention_allowance'],color='k',linestyle=':',alpha=.5)
        ax.set(xlabel='Within-low-order RMS jump at 0.75 ps',ylabel='Worst physical-error ratio to reference',title=split)
        ax.legend()
    fig.tight_layout();fig.savefig(root/'plots/smoothness_information.png',dpi=180);plt.close(fig)
