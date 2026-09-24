"""Frozen encoder metric interventions on the completed expanded Al cohort."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import average_precision_score

from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.crystallization_information.data import event_bins
from src.research.forecast_crystallization.local_metrics import first_sustained_onset
from .metrics import participation, fit_transform, fit_physical_metric, neighbor_metrics, lag_pairs

ORDER_NAMES = ['q4', 'q6', 'w4', 'w6', 'qbar6', 'coherence', 'density_r12', 'coordination']
DETAIL = [0, 2, 3, 4, 5]
MODES = ['raw', 'standardized', 'whitened', 'physical_metric']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temp.replace(path)


def table(path, rows):
    if not rows: raise ValueError(f'No rows for {path}')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def standardize(train, values):
    mean = train.mean(0); scale = train.std(0); constant = scale == 0
    scale[constant] = 1.
    return (values-mean)/scale


def prepare(config, root):
    """Freeze outcome-blind rows, original physical labels and exact input hashes."""
    root = Path(root); tech = root/'technical'; tech.mkdir(parents=True, exist_ok=True)
    assay = resolve_path(config['expanded_root'])/'technical/assay'
    pop_path = assay/'population.npz'; pop = dict(np.load(pop_path))
    if pop['rows'].shape != (18771, 3): raise ValueError('Unexpected completed expanded population')
    plan_path = resolve_path(config['physical_plan']); plan = json.loads(plan_path.read_text())
    anchors = np.asarray(plan['anchors']); frames = anchors[pop['rows'][:, 1]]
    sources = {s['id']:s for s in plan['sources']}
    current = np.empty((len(frames), 8), np.float32); future = current.copy()
    labels = np.empty(len(frames), np.int64); atoms = labels.copy()
    shard_hashes = {}
    for sid in np.unique(pop['source']):
        record = sources[int(sid)]; p = resolve_path(plan['config']['assay_cache'])/record['shard']
        if sha(p) != record['shard_sha256']: raise ValueError(f'Physical shard changed: {sid}')
        shard_hashes[str(sid)] = record['shard_sha256']; ids = np.flatnonzero(pop['source']==sid)
        centers = pop['rows'][ids,2]; f = frames[ids]
        with np.load(p) as a:
            np.testing.assert_allclose(a['times_ps'], np.arange(801)*.75, rtol=0, atol=1e-8)
            current[ids] = a['order'][centers,f]
            future[ids] = a['order'][centers,f+config['future_lag_frames']]
            labels[ids] = a['labels'][centers,f]
            atoms[ids] = a['atom_ids'][centers]
            onset=first_sustained_onset(np.isin(a['labels'],[1,2,3]),3)[centers]
            np.testing.assert_array_equal(pop['event'][ids],event_bins(onset,f))
        np.testing.assert_array_equal(pop['rows'][ids,0],sid)
        np.testing.assert_array_equal(pop['temperature'][ids],record['temperature_K'])
        if not np.array_equal(np.asarray(record['atom_ids'])[centers],atoms[ids]):raise ValueError('Atom identity differs')
        np.testing.assert_array_equal(pop['role'][ids], record.get('validation_role',record['split']))
    np.testing.assert_array_equal(current,pop['original_geometry'][:,128:136])
    if len({sources[int(s)]['lineage'] for s in np.unique(pop['source'])}) != len(np.unique(pop['source'])):
        raise ValueError('Sources share simulation ancestry')
    # No spatial coordinates are inferred from embedding or array row numbers.
    descriptors = np.load(assay/'hot-descriptors.npy')
    if descriptors.shape != (len(frames),237):raise ValueError('Hot descriptor producer shape differs')
    # Labels are explicitly independent PTM classifications, not forecast risk membership.
    liquid = labels == 0
    chosen=[];rng=np.random.default_rng(config['seed'])
    for role in ('train','selection','calibration','test'):
        for sid in sorted(np.unique(pop['source'][pop['role']==role])):
            ids=np.flatnonzero(liquid & (pop['source']==sid) & (pop['role']==role))
            cap=config['max_rows_per_source']
            chosen.extend(sorted(rng.choice(ids,min(cap,len(ids)),replace=False).tolist()))
    chosen=np.array(chosen,np.int64)
    if any((pop['role'][chosen]==r).sum()==0 for r in ('train','selection','test')):raise ValueError('PTM mask empties a role')
    values={k:pop[k][chosen] for k in ('source','role','temperature','condition','event','rows')}
    values.update(indices=chosen,frame=frames[chosen],atom=atoms[chosen],current=current[chosen],future=future[chosen],
                  descriptors=descriptors[chosen],labels=labels[chosen],strict_liquid=current[chosen,1]<config['strict_q6'])
    np.savez(tech/'population.npz',**values)
    inputs={str(pop_path):sha(pop_path),str(plan_path):sha(plan_path),str(assay/'hot-descriptors.npy'):sha(assay/'hot-descriptors.npy')}
    for model in config['models']:
        folder=assay/model['name'];feature=folder/'features.npy';receipt=json.loads((folder/'complete.json').read_text())
        record=json.loads((folder/'record.json').read_text())
        if record['population_sha256']!=sha(pop_path) or record['protected_overlap']:
            raise ValueError(f'Encoder population or ancestry differs: {model["name"]}')
        if receipt['checkpoint_sha256']!=record['checkpoint_sha256'] or record['input']!=model['domain']:
            raise ValueError(f'Encoder checkpoint or domain differs: {model["name"]}')
        inputs[str(folder/'record.json')]=sha(folder/'record.json')
        h=sha(feature)
        if receipt['feature_sha256']!=h:raise ValueError(f'Feature receipt mismatch: {model["name"]}')
        if np.load(feature,mmap_mode='r').shape!=(len(frames),128):raise ValueError('Feature row population mismatch')
        inputs[str(feature)]=h
        for kind in ('linear','mlp'):
            directory=resolve_path(config['expanded_root'])/'readouts/technical/fits'/model['name']/'snapshot'/kind
            for filename in ('metrics.json','predictions.npz'):inputs[str(directory/filename)]=sha(directory/filename)
            with np.load(directory/'predictions.npz') as pred:
                np.testing.assert_array_equal(pred['test_indices'],np.flatnonzero(pop['role']=='test'))
    write_json(tech/'identity.json',dict(config=config,inputs=inputs,physical_shards=shard_hashes,
               population_sha256=sha(tech/'population.npz'),n=len(chosen),roles={r:int((values['role']==r).sum()) for r in np.unique(values['role'])},
               mask='Current PTM Other==0; outcome-blind capped source samples; future labels never select rows',
               support='Retained checkpoint-native cached features; hot/cold input domains explicit',
               spatial='No physical spatial coordinates used in expanded arm; latest contextual arm has its own geometry.'))
    print(f'Prepared {len(chosen)} PTM-Other rows',flush=True)


def run_model(config, root, name):
    root=Path(root);tech=root/'technical';out=tech/'models'/name;out.mkdir(parents=True,exist_ok=True)
    identity=json.loads((tech/'identity.json').read_text());pop=dict(np.load(tech/'population.npz'))
    if sha(tech/'population.npz')!=identity['population_sha256'] or config!=identity['config']:
        raise ValueError('Frozen population or study configuration changed')
    entry=next(m for m in config['models'] if m['name']==name)
    path=resolve_path(config['expanded_root'])/'technical/assay'/name/'features.npy'
    if sha(path)!=identity['inputs'][str(path)]:raise ValueError(f'Features changed: {name}')
    x=np.load(path)[pop['indices']].astype(np.float64);train=np.flatnonzero(pop['role']=='train');test=np.flatnonzero(pop['role']=='test')
    if set(pop['source'][train]) & set(pop['source'][test]):raise ValueError('Source leakage')
    y=standardize(pop['current'][train],pop['current']);future=standardize(pop['future'][train],pop['future'])
    topology=standardize(pop['descriptors'][train,85:229],pop['descriptors'][:,85:229])
    nuisance=np.c_[pop['condition'],pop['current'][:,[1,6,7]]]
    nuisance=standardize(nuisance[train],nuisance)
    physical_base=Ridge(alpha=config['ridge_alpha'],solver='svd').fit(nuisance[train],y[train][:,DETAIL])
    residual=y[:,DETAIL]-physical_base.predict(nuisance)
    future_nuisance=np.c_[nuisance,y]
    future_base=Ridge(alpha=config['ridge_alpha'],solver='svd').fit(future_nuisance[train],future[train][:,DETAIL])
    residual_future=future[:,DETAIL]-future_base.predict(future_nuisance)
    zt,za,_=fit_transform(x[train],x,'standardized')
    physical_probe=Ridge(alpha=config['ridge_alpha'],solver='svd').fit(zt,residual[train])
    future_probe=Ridge(alpha=config['ridge_alpha'],solver='svd').fit(zt,residual_future[train])
    probe_error=((residual-physical_probe.predict(za))**2).mean(1)
    probe_base=(residual**2).mean(1)
    future_error=((residual_future-future_probe.predict(za))**2).mean(1);future_base_error=(residual_future**2).mean(1)
    source_rows=[];summary=[];arrays={}
    for mode in MODES:
        if mode=='physical_metric':ref,allz,meta=fit_physical_metric(x[train],residual[train],x,config['ridge_alpha'])
        else:ref,allz,meta=fit_transform(x[train],x,mode,config['whitening_ridge'])
        write_json(out/f'{mode}-transform.json',meta)
        target_error=np.full(len(x),np.nan);future_neighbor_error=target_error.copy();topology_error=target_error.copy();knn_risk=target_error.copy()
        rng=np.random.default_rng(config['seed']);random_error=target_error.copy()
        for temp in np.unique(pop['temperature'][test]):
            q=test[pop['temperature'][test]==temp];r=train[pop['temperature'][train]==temp]
            result=neighbor_metrics(allz[r],allz[q],y[r],y[q],pop['source'][r],pop['source'][q],config['neighbors'])
            ni=r[result['neighbor_indices']]
            target_error[q]=result['neighbor_target_mse']
            future_neighbor_error[q]=((future[ni].mean(1)-future[q])**2).mean(1)
            topology_error[q]=((topology[ni].mean(1)-topology[q])**2).mean(1)
            knn_risk[q]=((pop['event'][ni]<5).sum(1)+.5)/(config['neighbors']+1.)
            random=rng.choice(r,(len(q),config['neighbors']),replace=True)
            random_error[q]=((y[random]-y[q,None])**2).mean((1,2))
        if not np.isfinite(target_error[test]).all():raise FloatingPointError('Incomplete retrieval rows')
        current_rank=participation(allz[test]);a,b=lag_pairs(pop['source'][test],pop['atom'][test],pop['frame'][test],config['temporal_lag_frames'])
        if len(a):
            jumps=((allz[test[a]]-allz[test[b]])**2).sum(1)
            scaled=[]
            for sid in np.unique(pop['source'][test[a]]):
                mask=pop['source'][test[a]]==sid;temp=pop['temperature'][test[a]][mask][0]
                candidates=train[pop['temperature'][train]==temp]
                denom=2*np.mean([participation(allz[candidates[pop['source'][candidates]==s]])['trace']
                                for s in np.unique(pop['source'][candidates])])
                if denom==0:raise ValueError(f'No within-source training variation at {temp} K')
                scaled.append(jumps[mask].mean()/denom)
            normalized_jump=float(np.mean(scaled))
        else:normalized_jump=None
        for sid in np.unique(pop['source'][test]):
            ix=test[pop['source'][test]==sid];rank=participation(allz[ix])
            actual=(pop['event'][ix]<5).astype(float);p=knn_risk[ix]
            strict=ix[pop['strict_liquid'][ix]]
            source_rows.append(dict(model=name,domain=entry['domain'],mode=mode,source=int(sid),temperature=float(pop['temperature'][ix[0]]),n=len(ix),
                rank=rank['rank'],trace=rank['trace'],neighbor_order_mse=float(target_error[ix].mean()),random_order_mse=float(random_error[ix].mean()),
                neighbor_order_gain=float(1-target_error[ix].mean()/random_error[ix].mean()),future_neighbor_mse=float(future_neighbor_error[ix].mean()),
                metric_untrained_topology_mse=float(topology_error[ix].mean()),physical_probe_mse=float(probe_error[ix].mean()),physical_baseline_mse=float(probe_base[ix].mean()),
                future_probe_mse=float(future_error[ix].mean()),future_baseline_mse=float(future_base_error[ix].mean()),
                knn_brier=float(((p-actual)**2).mean()),knn_logloss=float(-(actual*np.log(p)+(1-actual)*np.log1p(-p)).mean()),
                strict_n=len(strict),strict_neighbor_mse=float(target_error[strict].mean()) if len(strict) else None))
        selected=[r for r in source_rows if r['mode']==mode]
        summary.append(dict(model=name,domain=entry['domain'],mode=mode,global_liquid_rank=current_rank['rank'],
            conditional_rank=float(np.mean([r['rank'] for r in selected])),
            neighbor_order_gain=float(np.mean([r['neighbor_order_gain'] for r in selected])),
            physical_probe_gain=float(1-np.mean([r['physical_probe_mse'] for r in selected])/np.mean([r['physical_baseline_mse'] for r in selected])),
            future_probe_gain=float(1-np.mean([r['future_probe_mse'] for r in selected])/np.mean([r['future_baseline_mse'] for r in selected])),
            future_neighbor_mse=float(np.mean([r['future_neighbor_mse'] for r in selected])),
            metric_untrained_topology_mse=float(np.mean([r['metric_untrained_topology_mse'] for r in selected])),
            knn_brier=float(np.mean([r['knn_brier'] for r in selected])),temporal_pairs=len(a),normalized_jump=normalized_jump))
        arrays[mode+'_knn_risk']=knn_risk[test]
    # Negative control keeps phase/temperature-conditioned feature distribution while breaking sample identity.
    shuffled=za.copy();rng=np.random.default_rng(config['seed'])
    for role in np.unique(pop['role']):
        for temp in np.unique(pop['temperature']):
            ids=np.flatnonzero((pop['role']==role)&(pop['temperature']==temp));shuffled[ids]=za[rng.permutation(ids)]
    control=[]
    for temp in np.unique(pop['temperature'][test]):
        q=test[pop['temperature'][test]==temp];r=train[pop['temperature'][train]==temp]
        v=neighbor_metrics(shuffled[r],shuffled[q],y[r],y[q],pop['source'][r],pop['source'][q],config['neighbors'])
        control.extend(dict(source=int(s),error=float(e)) for s,e in zip(pop['source'][q],v['neighbor_target_mse']))
    write_json(out/'shuffle-control.json',dict(source_mse={str(s):float(np.mean([r['error'] for r in control if r['source']==s])) for s in sorted({r['source'] for r in control})},
               definition='Within-role/temperature permutation of feature rows; targets and identities held fixed'))
    np.savez(out/'predictions.npz',indices=pop['indices'][test],**arrays)
    table(root/'tables'/f'{name}-sources.csv',source_rows);table(root/'tables'/f'{name}-summary.csv',summary)
    write_json(out/'status.json',dict(state='complete',model=name,feature_sha256=sha(path)))
    snapshot_metric_docs(root,'liquid_geometry');print(f'Completed {name}',flush=True)


def rho(x,y):
    if len(x)<4 or np.std(x)==0 or np.std(y)==0:return None
    return float(spearmanr(x,y).statistic)


def report(config,root):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);summary=[];sources=[];missing=[]
    identity=json.loads((root/'technical/identity.json').read_text())
    if config!=identity['config'] or sha(root/'technical/population.npz')!=identity['population_sha256']:
        raise ValueError('Frozen report population or config changed')
    for entry in config['models']:
        name=entry['name'];status=root/'technical/models'/name/'status.json'
        if not status.exists() or json.loads(status.read_text())['state']!='complete':missing.append(name);continue
        summary.extend(list(csv.DictReader((root/'tables'/f'{name}-summary.csv').open())))
        sources.extend(list(csv.DictReader((root/'tables'/f'{name}-sources.csv').open())))
    if not summary:raise ValueError('No diagnostic models completed')
    table(root/'tables/representation-summary.csv',summary)
    # Forecast associations use the same PTM-Other sampled rows, not unaligned published aggregate scores.
    pop=dict(np.load(root/'technical/population.npz'));test=np.flatnonzero(pop['role']=='test')
    full=dict(np.load(resolve_path(config['expanded_root'])/'technical/assay/population.npz'))
    full_test=np.flatnonzero(full['role']=='test');lookup={int(v):i for i,v in enumerate(full_test)};take=np.array([lookup[int(v)] for v in pop['indices'][test]])
    forecast={};forecast_rows=[]
    source_ids=np.unique(pop['source'][test]);temps=np.array([pop['temperature'][test][pop['source'][test]==s][0] for s in source_ids])
    for entry in config['models']:
        name=entry['name']
        if name in missing:continue
        for head in ('linear','mlp'):
            d=resolve_path(config['expanded_root'])/'readouts/technical/fits'/name/'snapshot'/head
            if sha(d/'predictions.npz')!=identity['inputs'][str(d/'predictions.npz')]:
                raise ValueError(f'Frozen forecast changed: {name}/{head}')
            p=np.load(d/'predictions.npz');np.testing.assert_array_equal(p['test_indices'],full_test)
            risk=1-np.prod(1-expit(p['test'][take].astype(np.float64)),axis=1);y=(pop['event'][test]<5).astype(float)
            risk=np.clip(risk,1e-12,1-1e-12)
            per=np.array([np.mean(-(y[pop['source'][test]==s]*np.log(risk[pop['source'][test]==s])+(1-y[pop['source'][test]==s])*np.log1p(-risk[pop['source'][test]==s]))) for s in source_ids])
            per_brier=np.array([np.mean((risk[pop['source'][test]==s]-y[pop['source'][test]==s])**2) for s in source_ids])
            forecast[name,head]=dict(log_loss=per,brier=per_brier,risk=risk)
            weights=np.array([1/np.sum(pop['source'][test]==s) for s in pop['source'][test]])
            brier=np.average((risk-y)**2,weights=weights)
            forecast_rows.append(dict(model=name,domain=entry['domain'],head=head,log_loss=float(per.mean()),brier=float(brier),ap=float(average_precision_score(y,risk,sample_weight=weights))))
    table(root/'tables/matched-forecast.csv',forecast_rows)
    associations=[];rng=np.random.default_rng(config['seed'])
    groups=[np.flatnonzero(temps==t) for t in np.unique(temps)]
    draws=[np.concatenate([rng.choice(g,len(g),replace=True) for g in groups]) for _ in range(config['bootstrap_draws'])]
    # AP is an aggregate ranking score, never an average of source AP values.
    # A source drawn m times contributes m/n_source weight to each of its rows.
    # Sort each frozen risk vector once; tied scores use their group's endpoint,
    # exactly matching sklearn's non-interpolated average-precision convention.
    source_inverse=np.searchsorted(source_ids,pop['source'][test])
    source_counts=np.bincount(source_inverse,minlength=len(source_ids))
    draw_counts=np.array([np.bincount(draw,minlength=len(source_ids)) for draw in draws])
    bootstrap_weights=draw_counts[:,source_inverse]/source_counts[source_inverse][None]
    base_weights=1/source_counts[source_inverse]
    actual=(pop['event'][test]<5).astype(float)
    for key,values in forecast.items():
        scores=values['risk'];order=np.argsort(-scores,kind='stable')
        ends=np.r_[np.flatnonzero(np.diff(scores[order])!=0),len(order)-1]
        weighted=np.vstack((base_weights,bootstrap_weights))[:,order]
        tp=np.cumsum(weighted*actual[order][None],axis=1)[:,ends]
        total=np.cumsum(weighted,axis=1)[:,ends]
        precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
        numerator=(precision*np.diff(np.c_[np.zeros(len(tp)),tp],axis=1)).sum(1)
        aps=np.divide(numerator,tp[:,-1],out=np.full(len(tp),np.nan),where=tp[:,-1]>0)
        if actual.any():
            np.testing.assert_allclose(aps[0],average_precision_score(actual,scores,sample_weight=base_weights),rtol=1e-12,atol=1e-12)
        if len(draws) and np.sum(bootstrap_weights[0]*actual)>0:
            np.testing.assert_allclose(aps[1],average_precision_score(actual,scores,sample_weight=bootstrap_weights[0]),rtol=1e-12,atol=1e-12)
        values['ap']=aps[0];values['ap_bootstrap']=aps[1:]
    effects=[]
    for entry in config['models']:
      name=entry['name']
      if name in missing:continue
      for mode in MODES[1:]:
       for metric in ('neighbor_order_mse','future_neighbor_mse','metric_untrained_topology_mse','knn_brier','knn_logloss'):
        def values(which):
            return np.array([float(next(r for r in sources if r['model']==name and r['mode']==which and int(r['source'])==sid)[metric]) for sid in source_ids])
        delta=values(mode)-values('raw');boots=[delta[d].mean() for d in draws]
        effects.append(dict(model=name,mode=mode,metric=metric,delta_from_raw=float(delta.mean()),
                            ci_low=float(np.quantile(boots,.025)),ci_high=float(np.quantile(boots,.975)),sources=len(delta)))
    table(root/'tables/distance-interventions.csv',effects)
    for domain in ('all','hot','cold'):
      for head in ('linear','mlp'):
       names=[e['name'] for e in config['models'] if e['name'] not in missing and (domain=='all' or e['domain']==domain)]
       if not names:continue
       for metric in ('rank','neighbor_order_gain','physical_probe_mse','future_probe_mse','future_neighbor_mse','metric_untrained_topology_mse'):
        x=np.array([[float(next(r for r in sources if r['model']==name and r['mode']=='standardized' and int(r['source'])==sid)[metric]) for sid in source_ids] for name in names])
        for score_name in ('log_loss','brier','ap'):
         if score_name=='ap':
          ys=np.array([forecast[n,head]['ap'] for n in names]);yb=np.array([forecast[n,head]['ap_bootstrap'] for n in names])
         else:
          per_source=np.array([forecast[n,head][score_name] for n in names]);ys=per_source.mean(1)
          yb=np.stack([per_source[:,draw].mean(1) for draw in draws],axis=1)
         point=rho(x.mean(1),ys) if np.isfinite(ys).all() else None;boots=[]
         if point is not None:
          for draw_index,draw in enumerate(draws):
           if not np.isfinite(yb[:,draw_index]).all():continue
           v=rho(x[:,draw].mean(1),yb[:,draw_index])
           if v is not None:boots.append(v)
         associations.append(dict(domain=domain,head=head,metric=metric,forecast_metric='matched_12ps_'+score_name,models=len(names),spearman=point,
             ci_low=float(np.quantile(boots,.025)) if boots else None,ci_high=float(np.quantile(boots,.975)) if boots else None,valid_draws=len(boots)))
    table(root/'tables/forecast-associations.csv',associations)
    (root/'plots').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,metric in zip(axes,('conditional_rank','neighbor_order_gain')):
        for row in summary:
            if row['mode']!='standardized':continue
            score=next(r for r in forecast_rows if r['model']==row['model'] and r['head']=='mlp')
            ax.scatter(float(row[metric]),score['log_loss'],c='tab:blue' if row['domain']=='hot' else 'tab:orange')
            ax.annotate(row['model'],(float(row[metric]),score['log_loss']),fontsize=6)
        ax.set_xlabel(metric);ax.set_ylabel('Matched liquid 12ps MLP log loss (lower better)')
    fig.savefig(root/'plots/metrics-vs-forecast.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    names=[e['name'] for e in config['models'] if e['name'] not in missing]
    for ax,metric in zip(axes.flat,('neighbor_order_gain','future_neighbor_mse','metric_untrained_topology_mse','knn_brier')):
        matrix=np.array([[float(next(r for r in summary if r['model']==n and r['mode']==m)[metric]) for m in MODES] for n in names])
        im=ax.imshow(matrix,aspect='auto',cmap='viridis');fig.colorbar(im,ax=ax)
        ax.set_xticks(range(4),MODES,rotation=20);ax.set_yticks(range(len(names)),names)
        ax.set_title(metric + (' (higher better)' if metric.endswith('gain') else ' (lower better)'))
    fig.savefig(root/'plots/distance-interventions.png',dpi=180);plt.close(fig)
    components=dict(expanded='partial' if missing else 'complete',
                    latest='complete' if (root/'technical/latest/provenance.json').exists() else 'pending',
                    checkpoints='complete' if (root/'technical/checkpoints/results.json').exists() else 'pending')
    for stage in ('latest','checkpoints'):
        if components[stage]!='complete' and (root/'technical'/f'failure-{stage}-all.json').exists():components[stage]='failed'
    lines=['# Liquid geometry diagnostic study','',f'Completed encoders: {len(config["models"])-len(missing)}/{len(config["models"])}; missing: {missing}.',
      '',f'Component status: {components}.',
      '','The expanded comparison uses independent PTM-Other masks and matched held-out rows for both geometric and forecasting metrics. Existing forecasts are frozen; new distance interventions never fit to onset labels.','',
      'Four interventions: raw centered features, train-standardized features, shrinkage whitening, and a train-fitted physical residual metric. The physical metric fits q4/w4/w6/qbar6/coherence residuals beyond temperature/time/current q6/density/coordination; topology remains an untrained metric target.',
      '', 'Associations are exploratory across a fixed related model family; source bootstrap conditions on trained weights and seed. Four forecast heads or multiple transforms are not independent encoder replicates. Pooled hot/cold associations are confounded by input domain; see separate domains. Fewer than four models yield no correlation estimate.',
      '', 'Forecast associations cover source-equal 12 ps log loss and Brier error (lower is better), and source-weighted aggregate AP (higher is better). Every paired source-bootstrap draw recomputes aggregate AP with source multiplicity divided by that source\'s row count; source AP values are never averaged. Draws without a positive outcome yield undefined AP and are excluded from AP intervals.',
      '', '[Representation metrics](tables/representation-summary.csv) · [Matched forecasts](tables/matched-forecast.csv) · [Associations](tables/forecast-associations.csv) · [Figure](plots/metrics-vs-forecast.png)',
      '', '[Distance interventions with paired source intervals](tables/distance-interventions.csv) · [Intervention figure](plots/distance-interventions.png)',
      '', 'Physical/future ridge probes always use the standardized original features, so their values are deliberately identical across distance modes. The future probe baseline includes all eight current order descriptors plus temperature/time. Topology is absent from the new metric fit, but was already used in encoder training; it is not an unseen pretraining target.',
      '', 'The latest completed contextual forecast comparison and genuine same-run checkpoint audit are separate stages; their populations must not be pooled with this one.']
    lines += ['', '| Encoder | Within-source rank | Physical neighbor gain | Physical probe gain | Future probe gain | MLP AP |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in summary:
        if row['mode']!='standardized':continue
        ap=next(f['ap'] for f in forecast_rows if f['model']==row['model'] and f['head']=='mlp')
        lines.append('| '+row['model']+' | '+' | '.join(f'{float(row[k]):.3f}' for k in ('conditional_rank','neighbor_order_gain','physical_probe_gain','future_probe_gain'))+f' | {ap:.3f} |')
    if components['latest']=='complete':
        lines += ['', 'Latest contextual comparison: [paired source changes](tables/latest_paired_deltas.csv), [associations](tables/latest_associations.csv), [figure](plots/latest_geometry_forecast.png).']
    if components['checkpoints']=='complete':
        lines += ['', 'Historical same-run endpoints: [figure](plots/checkpoint-endpoints.png), [full numeric results](technical/checkpoints/results.json). These are all-phase development diagnostics.']
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n');snapshot_metric_docs(root,'liquid_geometry')
    write_json(root/'technical/report-status.json',dict(state='complete' if all(v=='complete' for v in components.values()) else 'partial',missing=missing,components=components))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','model','report','latest','checkpoints']);p.add_argument('--config',required=True);p.add_argument('--model');args=p.parse_args()
    config=json.loads(Path(args.config).read_text());root=resolve_path(config['output']);root.mkdir(parents=True,exist_ok=True)
    try:
        if args.stage=='prepare':prepare(config,root)
        elif args.stage=='model':run_model(config,root,args.model)
        elif args.stage=='report':report(config,root)
        elif args.stage=='latest':
            from .latest import run
            run(config,root)
        elif args.stage=='checkpoints':
            from .checkpoints import run
            run(config,root)
    except Exception:
        write_json(root/'technical'/f'failure-{args.stage}-{args.model or "all"}.json',dict(state='failed',traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
