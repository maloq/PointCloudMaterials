"""Weighted distance-distribution boosting/MLP controls; validation-only selection."""
import json
import time
from pathlib import Path
import numpy as np
from src.data.fixed_cohort.protocol import sha,digest,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import write_metric_rows
from .data import config
from .descriptor_data import load,parent


def targets(distance,c):
    boundaries=np.asarray(c['distance_edges_A'][1:])
    y=np.searchsorted(boundaries,distance,side='left')
    y[distance>=boundaries[-1]]=len(boundaries)
    return y.astype(np.int64)


def quantities(prob,distance,c):
    edges=np.asarray(c['distance_edges_A']);width=np.r_[np.diff(edges),1.]
    midpoint=np.r_[(edges[:-1]+edges[1:])/2,edges[-1]]
    p=np.maximum(np.asarray(prob,dtype=float),1e-12);p/=p.sum(1,keepdims=True)
    y=targets(distance,c)
    result=dict(nll=-np.log(p[np.arange(len(y)),y])+np.log(width[y]),mean_A=p@midpoint)
    result['cdf']=np.stack([p[:,:int(np.flatnonzero(edges==r)[0])].sum(1) for r in (20,32,48)],1)
    return result


def metrics(pred,truth,w):
    w=w/w.sum()
    result=dict(distance_nll=float(w@pred['nll']),distance_rmse_A=float(np.sqrt(w@(pred['mean_A']-np.minimum(truth,64))**2)))
    for j,r in enumerate((20,32,48)):
        result[f'brier{r}A']=float(w@(pred['cdf'][:,j]-(truth<=r))**2)
    return result


def table(root,name,rows):
    return write_metric_rows(rows, root, family='liquid_descriptors', name=name)


def fit(c,name):
    arm=next(a for a in c['arms'] if a['name']==name);root=resolve_path(c['output'])/name;tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    if (tech/'complete.json').exists():return
    x,rows,columns,manifest=load(c)
    selected=np.array([i for i,v in enumerate(columns) if v['family'] in arm['families']],int)
    roles={r:np.flatnonzero(rows['role']==r) for r in ('train','selection','calibration','test')}
    train=roles['train'];valid=roles['selection'];w=rows['weights'];y=targets(rows['target'],c)
    k=len(c['distance_edges_A']);prior=np.bincount(y[train],weights=w[train],minlength=k)+1e-8;prior/=prior.sum()
    implementation={p.name:sha(p) for p in Path(__file__).parent.glob('descriptor*.py')}
    binding=dict(config=c,arm=arm,dataset=manifest['identity'],implementation=implementation)
    identity=digest(binding)
    if (tech/'identity.json').exists() and config(tech/'identity.json')!=binding:raise ValueError('Descriptor fit identity changed')
    write_json(tech/'identity.json',binding)
    write_json(tech/'prediction-context.json',dict(encoder=None,predictor=dict(model=arm['model'],families=arm['families'],
        feature_count=len(selected),patches=25,max_support_A=32,nearest_candidates=80,patch_radius_A=8,
        spatial_aggregation='three shells: mean/std, gradient norm, traceless quadrupole norm',
        geometry_only=True,history=False,motion=False,conditions=[],relaxation=c.get('observation',{}).get('relaxed',False),teacher=None),
        labels_as_inputs=False,clearance_as_input=False,cohort=parent(c)['fixed_dataset'],
        observation=c.get('observation',{'domain':'original_MD'}),target_protocol=c.get('target_protocol','original MD crystal distance'),
        selection='minimum full source-held-out validation distance negative log likelihood',tracking='local descriptor control'))
    write_json(tech/'state.json',dict(state='training',model=name))
    started=time.time()
    if arm['model']=='prior':
        prob=np.broadcast_to(prior,(len(y),k)).copy();selection_iteration=0
    elif arm['model']=='catboost':
        from catboost import CatBoostClassifier,Pool
        # Weights rescaled to mean one consistently in fitting and validation;
        # no class balancing or label-dependent proximity oversampling.
        def pool(ids):return Pool(np.array(x[np.ix_(ids,selected)]),y[ids],weight=w[ids]/w[ids].mean(),feature_names=[columns[i]['name'] for i in selected])
        tr=pool(train);va=pool(valid)
        settings=dict(c['catboost'],**arm.get('parameters',{}))
        model=CatBoostClassifier(**settings,loss_function='MultiClass',eval_metric='MultiClass',classes_count=k,
             random_seed=c['seed'],thread_count=c['fit_threads'],train_dir=str(tech/'catboost'),
             allow_writing_files=True)
        model.fit(tr,eval_set=va,use_best_model=True,early_stopping_rounds=c['patience'],verbose=25,
                  save_snapshot=True,snapshot_file=str(tech/'catboost.snapshot'),snapshot_interval=60)
        model.save_model(str(tech/'best.cbm'));selection_iteration=int(model.get_best_iteration())+1
        write_json(tech/'resolved-catboost-parameters.json',model.get_all_params())
        prob=np.empty((len(y),k),float)
        if not np.array_equal(model.classes_,np.arange(k)):raise ValueError(f'Missing output classes: {model.classes_}')
        for start in range(0,len(y),4096):prob[start:start+4096]=model.predict_proba(np.array(x[start:start+4096,selected]),thread_count=c['fit_threads'])
        importance=model.get_feature_importance(tr,thread_count=c['fit_threads'])
        table(root/'analyses/diagnostics-v1','feature-importance',[
            dict(feature=columns[i]['name'],family=columns[i]['family'],importance=float(v),kind='training PredictionValuesChange; not held-out evidence')
            for i,v in zip(selected,importance)])
        write_json(tech/'learning-curve.json',model.get_evals_result())
    elif arm['model']=='mlp':
        prob,selection_iteration=fit_mlp(c,arm,x,selected,rows,roles,y,prior,tech,identity)
    else:raise ValueError(arm['model'])
    if not np.isfinite(prob).all() or (prob<0).any() or not np.allclose(prob.sum(1),1,atol=1e-5):raise ValueError('Invalid predicted distribution')
    predictions=quantities(prob,rows['target'],c);analysis=root/'analyses/predictability-v1';(analysis/'technical').mkdir(parents=True,exist_ok=True)
    np.savez(analysis/'technical/predictions.npz',ids=rows['ids'],probability=prob.astype(np.float32),**predictions)
    scores=[];reliability=[]
    for role,ids in roles.items():
        # Distance subgroups remain diagnostics, not model selectors.
        groups=[('all',np.ones(len(ids),bool)),('beyond_32A_observation_envelope',rows['target'][ids]>32)]
        for lo,hi in zip((0,20,32,48,64),(20,32,48,64,np.inf)):
            groups.append((f'distance_{lo}_{hi}A',(rows['target'][ids]>=lo)&(rows['target'][ids]<hi)))
        for group,mask in groups:
            chosen=ids[mask]
            if not len(chosen):continue
            scores.append(dict(role=role,subset=group,rows=len(chosen),sources=len(np.unique(rows['source'][chosen])),
                **metrics({k:v[chosen] for k,v in predictions.items()},rows['target'][chosen],w[chosen])))
        for j,radius in enumerate((20,32,48)):
            p=predictions['cdf'][ids,j]
            for lo in np.arange(10)/10:
                take=(p>=lo)&(p<lo+.1 if lo<.9 else p<=1);chosen=ids[take];mass=float(w[chosen].sum())
                reliability.append(dict(role=role,radius_A=radius,bin_low=lo,rows=len(chosen),mass=mass,
                    mean_probability=float(w[chosen]@p[take]/mass) if mass else None,
                    frequency=float(w[chosen]@(rows['target'][chosen]<=radius)/mass) if mass else None))
    table(analysis,'scores',scores);table(analysis,'reliability',reliability)
    receipt=dict(identity=identity,selected_iteration=selection_iteration,seconds=time.time()-started,
        predictions_sha256=sha(analysis/'technical/predictions.npz'),validation_nll=next(r['distance_nll'] for r in scores if r['role']=='selection' and r['subset']=='all'),
        test_used_for_selection=False,feature_count=len(selected),training_rows=len(train),training_sources=len(np.unique(rows['source'][train])))
    write_json(tech/'complete.json',receipt);write_json(tech/'state.json',dict(state='complete',**receipt))
    print(json.dumps(dict(model=name,**receipt)),flush=True)


def fit_mlp(c,arm,x,selected,rows,roles,y,prior,tech,identity):
    import torch
    from torch import nn
    torch.set_num_threads(c['fit_threads']);torch.manual_seed(c['seed'])
    ids=roles['train'];w=rows['weights'][ids]
    mean=np.zeros(len(selected));second=mean.copy()
    for start in range(0,len(ids),1024):
        ix=ids[start:start+1024];v=np.asarray(x[np.ix_(ix,selected)],dtype=float);weight=w[start:start+len(ix)]
        mean+=weight@v;second+=weight@(v*v)
    scale=np.sqrt(np.maximum(second-mean*mean,0)).clip(1e-4)
    values=torch.from_numpy(np.array(x[:,selected]));values.sub_(torch.tensor(mean,dtype=torch.float32)).div_(torch.tensor(scale,dtype=torch.float32))
    cfg=c['mlp'];model=nn.Sequential(nn.Linear(len(selected),cfg['width']),nn.SiLU(),nn.Dropout(cfg['dropout']),
        nn.Linear(cfg['width'],cfg['width']),nn.SiLU(),nn.Dropout(cfg['dropout']),nn.Linear(cfg['width'],len(prior)))
    with torch.no_grad():model[-1].weight.mul_(.01);model[-1].bias.copy_(torch.tensor(np.log(prior)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(c['seed']);labels=torch.tensor(y);best=float('inf');epoch=step=0;selected_epoch=0
    last=tech/'last.pt'
    if last.exists():
        saved=torch.load(last,weights_only=False)
        if saved['identity']!=identity:raise ValueError('MLP checkpoint identity changed')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng']
        torch.set_rng_state(saved['torch_rng']);epoch=saved['epoch'];step=saved['step'];best=saved['best'];selected_epoch=saved['selected_epoch']
    def save(path):
        tmp=path.with_suffix('.building.pt');torch.save(dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng.bit_generator.state,
            torch_rng=torch.get_rng_state(),epoch=epoch,step=step,best=best,selected_epoch=selected_epoch,mean=mean,scale=scale,columns=selected),tmp);tmp.replace(path)
    while epoch<cfg['blocks']:
        model.train();loss_sum=0.;started=time.time()
        while step<cfg['updates_per_block']:
            ix=rng.choice(ids,cfg['batch_size'],p=w,replace=True);optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.cross_entropy(model(values[ix]),labels[ix])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite descriptor MLP loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True);optimizer.step();step+=1;loss_sum+=float(loss.detach())
            if step%64==0:save(last)
        model.eval();val=0.
        with torch.no_grad():
            for begin in range(0,len(roles['selection']),2048):
                ix=roles['selection'][begin:begin+2048]
                loss=nn.functional.cross_entropy(model(values[ix]),labels[ix],reduction='none')
                val+=float(rows['weights'][ix]@loss.numpy())
        epoch+=1;step=0
        if val<best:best=val;selected_epoch=epoch;save(tech/'best.pt')
        save(last)
        record=dict(block=epoch,validation_categorical_nll=val,best_validation_categorical_nll=best,seconds=time.time()-started)
        with (tech/'learning-curve.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        write_json(tech/'state.json',dict(state='training',**record));print(json.dumps(record),flush=True)
    model.load_state_dict(torch.load(tech/'best.pt',weights_only=False)['model']);model.eval()
    prob=np.empty((len(y),len(prior)))
    with torch.no_grad():
        for start in range(0,len(y),2048):prob[start:start+2048]=model(values[start:start+2048]).softmax(1).numpy()
    return prob,selected_epoch


def compare(c):
    _,rows,_,_=load(c);root=resolve_path(c['output']);analysis=root/'analyses/comparison-v1'
    names=[a['name'] for a in c['arms']];complete={n:config(root/n/'technical/complete.json') for n in names}
    candidates=[n for n in names if n!='prior']
    chosen=min(candidates,key=lambda n:complete[n]['validation_nll'])
    winner=min(names,key=lambda n:complete[n]['validation_nll'])
    records=[];tables=[];ref=None;ncomp=len(candidates)
    for name in names:
        p=root/name/'analyses/predictability-v1/technical/predictions.npz'
        if sha(p)!=complete[name]['predictions_sha256']:raise ValueError('Changed exported predictions')
        with np.load(p) as a:value={k:a[k] for k in ('ids','nll','mean_A','cdf')}
        if not np.array_equal(value['ids'],rows['ids']):raise ValueError('Unmatched rows')
        if name=='prior':ref=value
        for role in ('selection','calibration','test'):
            ids=np.flatnonzero(rows['role']==role);w=rows['weights'][ids];truth=rows['target'][ids]
            scores=metrics({k:v[ids] for k,v in value.items() if k!='ids'},truth,w)
            records.append(dict(model=name,role=role,selected_by_validation=name==chosen,**scores))
            if name=='prior':continue
            src,inv=np.unique(rows['source'][ids],return_inverse=True)
            gain=ref['nll'][ids]-value['nll'][ids];mse=(value['mean_A'][ids]-np.minimum(truth,64))**2;refmse=(ref['mean_A'][ids]-np.minimum(truth,64))**2
            sums=np.stack([np.bincount(inv,weights=w*z) for z in (np.ones(len(ids)),gain,mse,refmse)],1)
            boot=np.random.default_rng(c['seed']).integers(0,len(src),(c['bootstrap_draws'],len(src)));s=sums[boot].sum(1)
            ng=s[:,1]/s[:,0];rg=1-np.sqrt(s[:,2]/s[:,3])
            tables.append(dict(model=name,role=role,rows=len(ids),sources=len(src),nll_gain=float(w@gain),
                nll_gain_ci95_low=float(np.quantile(ng,.025)),nll_gain_ci95_high=float(np.quantile(ng,.975)),
                nll_gain_familywise_low=float(np.quantile(ng,.025/ncomp)),nll_gain_familywise_high=float(np.quantile(ng,1-.025/ncomp)),
                rmse_reduction_fraction=float(1-np.sqrt((w@mse)/(w@refmse))),rmse_reduction_ci95_low=float(np.quantile(rg,.025)),rmse_reduction_ci95_high=float(np.quantile(rg,.975)),
                rmse_reduction_familywise_upper=float(np.quantile(rg,1-.05/ncomp))))
    table(analysis,'scores',records);table(analysis,'paired-comparisons',tables)
    write_json(analysis/'technical/selection.json',dict(best_descriptor_model=chosen,best_including_prior=winner,criterion='full-validation distance NLL only',
        validation_nll={n:complete[n]['validation_nll'] for n in names},one_seed=True,test_used_for_selection=False))
    lines=['# Descriptor benchmark','',f'Validation-selected descriptor: **{chosen}**. Best including no-input prior: **{winner}**.','',
        '| Model | Test distance NLL | Test RMSE (Å) |','|---|---:|---:|']
    for r in records:
        if r['role']=='test':lines.append(f"| {r['model']} | {r['distance_nll']:.5f} | {r['distance_rmse_A']:.5f} |")
    lines+=['','Distance likelihood uses a piecewise-uniform density below 64 Å and a right-censored tail mass. It is not the previous lognormal mixture family.',
        'Same source roles and clear-population rows as the MACE assay. Paired source intervals do not include training-seed uncertainty. The reused test set is not a fresh confirmation cohort.']
    (analysis/'README.md').write_text('\n'.join(lines)+'\n');write_json(analysis/'technical/complete.json',dict(selected=chosen))
