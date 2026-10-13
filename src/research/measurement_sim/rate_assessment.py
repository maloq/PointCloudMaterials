"""Train/test-only physical forecasts, source curves and sensitivity diagnostics."""
from collections import defaultdict
import time
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from .data import root, atomic_save, write_json, sha
from .rate_data import load, FAMILY, GROUPS
from .rate_learning import RateProcess, response_prediction
from .model import feature_weights
from .assessment import scores, prior, scramble, plot
from .evaluate import aggregate
from .assays import response_rows


def run(c, arm, deadline):
    out = root(c)/'analyses'/arm['name']; folder = out/'technical'
    if (folder/'evaluation-complete.json').exists(): return True
    _,records,data,n,norm = load(c,arm['bank'],test=True,device='cuda')
    saved = torch.load(folder/'final.pt',map_location='cpu',weights_only=False)
    model = RateProcess(c,data['decay_rate'],arm).cuda().eval(); model.load_state_dict(saved['model'])
    weights = feature_weights(norm['scale'].cuda()); fingerprint = sha(folder/'final.pt')
    ids = np.arange(n,len(records)); keys = c['evaluation']['outer_keys']; particles = c['evaluation']['particles']
    metrics = defaultdict(list); responses = []; halves = []; cases = []; start = 0
    partial = folder/'evaluation-partial.pt'
    if partial.exists():
        old = torch.load(partial,map_location='cpu',weights_only=False)
        if old['checkpoint_sha256'] != fingerprint: raise ValueError('Changed evaluated checkpoint')
        metrics.update(old['metrics']); responses=old['responses']; halves=old['halves']; cases=old['cases']; start=old['offset']
    def save(offset):
        atomic_save(partial,dict(checkpoint_sha256=fingerprint,metrics=dict(metrics),responses=responses,
                                halves=halves,cases=cases,offset=offset))
    batch = c['evaluation']['batch_size']
    with torch.no_grad():
        for offset in range(start,len(ids),batch):
            if time.time()>deadline-300: save(offset); return False
            chosen=ids[offset:offset+batch]; initial=data['initial'][chosen]; targets=data['target'][chosen]
            rr=[]; hh=[]; shown=[]
            for outer,key in enumerate(keys):
                seed=key+offset
                prediction=model(initial,data['rates'][chosen],particles,seed); shown.append(prediction.cpu())
                baseline=prior(initial,data['decay_rate'],20,particles,seed,.005)
                for method,paths in [('model',prediction),('ou',baseline),('shuffled_particles',scramble(prediction,seed+1))]:
                    for steps in [10,20]:
                        values=scores(paths[:,:,:steps],targets[:,:,:steps],initial,weights,data['increment_scale'],c,True)
                        for metric,value in values.items(): metrics[(outer,method,steps,metric)].append(value.cpu().numpy())
                for steps in [10,20]:
                    for method,value in [('mean',torch.zeros_like(initial)),('persistence',initial)]:
                        error=((targets[:,:,steps-1]-value[:,None]).square()*weights).sum(-1).mean(1)
                        metrics[(outer,method,steps,'endpoint_mse')].append(error.cpu().numpy())
                rr.append(response_prediction(model,data,chosen,c['training']['response_epsilon'],seed,particles).cpu())
                hh.append(response_prediction(model,data,chosen,c['training']['response_epsilon']/2,seed,particles,half=True).cpu())
            responses.append(torch.stack(rr,1)); halves.append(torch.stack(hh,1))
            if offset==0:
                for k,index in enumerate(chosen[:4]):
                    cases.append(dict(index=int(index),row={k:v for k,v in records[index].items() if not torch.is_tensor(v)},
                        initial=initial[k].cpu(),truth=targets[k].cpu(),samples=torch.stack([s[k] for s in shown])))
            save(offset+len(chosen)); print('evaluation',arm['name'],offset+len(chosen),len(ids),flush=True)
    held=[records[i] for i in ids]; ancestors=[r['ancestor'] for r in held]
    arrays={k:np.concatenate(v) for k,v in metrics.items()}; rows=[]
    cohorts=np.array([r['cohort'] for r in held])
    for (outer,method,steps,metric),values in arrays.items():
        for cohort in ['all','original60','new30']:
            mask=np.ones(len(ids),dtype=bool) if cohort=='all' else cohorts==cohort
            names=np.array(ancestors)[mask].tolist(); mean,lo,hi=aggregate(values[mask],names)
            rows.append(dict(outer=outer,method=method,cohort=cohort,horizon_ps=steps*.005,metric=metric,
                             value=mean,lo=lo,hi=hi,rows=int(mask.sum()),ancestors=len(set(names))))
    response=torch.cat(responses); half=torch.cat(halves); actual=data['response'][ids].cpu(); initial=data['initial_response'][ids].cpu()
    rr=[]
    for direction,label in enumerate(['q','v']):
        for cohort in ['all','original60','new30']:
            selected=np.flatnonzero(np.ones(len(ids),dtype=bool) if cohort=='all' else cohorts==cohort)
            record_subset=[held[i] for i in selected]
            for outer in range(len(keys)):
                for method,prediction in [('model',response[:,outer]),('half_epsilon',half[:,outer])]:
                    rr.extend(response_rows(prediction[selected,...,direction],actual[selected,...,direction],record_subset,
                                            outer=outer,method=method,direction=label,cohort=cohort))
            for method,prediction in [('zero',torch.zeros_like(actual)),('persistence',initial[:,None].expand_as(actual))]:
                rr.extend(response_rows(prediction[selected,...,direction],actual[selected,...,direction],record_subset,
                                        outer=-1,method=method,direction=label,cohort=cohort))
    atomic_save(folder/'evaluation-arrays.pt',dict(jobs=[r['job'] for r in held],ancestors=ancestors,values=arrays))
    atomic_save(folder/'sensitivity-arrays.pt',dict(prediction=response,half_prediction=half,target=actual,initial=initial,
                                                  jobs=[r['job'] for r in held],ancestors=ancestors))
    atomic_save(folder/'examples.pt',dict(cases=cases,columns=norm['columns'],mean=norm['mean'],scale=norm['scale'],dt_ps=.005,physical=True))
    write_metric_rows(rows,out,family=FAMILY,name='predictions'); write_metric_rows(rr,out,family=FAMILY,name='sensitivity')
    plot(out)
    update_recorded_summary(folder/'wandb.json',{'test/'+r['metric']+'/'+str(r['horizon_ps']):r['value'] for r in rows
        if r['method']=='model' and r['outer']==0 and r['cohort']=='all'},evaluation='density-rates-all-nontrain',expected={'mode':'online'})
    write_json(folder/'evaluation-complete.json',dict(state='complete',rows=len(ids),ancestors=len(set(ancestors)),
        checkpoint_sha256=fingerprint,update=saved['update'],split='train/test only'))
    return True


def readouts(c, bank):
    """Fixed hyperparameters: no hidden validation split and no test selection."""
    out=root(c)/'analyses'/('readouts-'+bank); marker=out/'technical/complete.json'
    if marker.exists(): return
    _,records,data,n,_=load(c,bank,test=True)
    h=data['initial'].double(); rates=data['rates'].double(); held=records[n:]; rows=[]
    for direction,label in enumerate(['q','v']):
        initial=data['initial_response'][...,direction].double(); target=data['response'][...,direction].double()
        query_scale=initial[:n].square().mean(0).sqrt().clamp_min(1e-5); u=initial/query_scale
        norm=torch.empty(target.shape[1:],dtype=torch.float64)
        for a,b in GROUPS: norm[:,a:b]=target[:n,:,a:b].square().mean((0,2)).clamp_min(1e-5).sqrt()[:,None]
        for kind,with_rates in [('transport',False),('residual_state',False),('residual_state',True)]:
            features=torch.cat((h,rates),-1) if with_rates else h
            z=(features-features[:n].mean(0))/features[:n].std(0).clamp_min(.01)
            matrix=(u@u[:n].T)/u.shape[1]
            skip=initial[:,None].expand_as(target) if kind=='residual_state' else torch.zeros_like(target)
            if kind=='residual_state':
                distances=torch.cdist(z,z[:n]).square()/z.shape[1]
                bandwidth=distances[:n].flatten().median().clamp_min(1e-8)
                matrix=matrix*(1+torch.exp(-distances/(2*bandwidth*c['assays']['state_length']**2)))/2
            y=((target-skip)/norm).flatten(1)
            coefficients=torch.linalg.solve(matrix[:n]+n*c['assays']['penalty']*torch.eye(n,dtype=torch.float64),y[:n])
            prediction=(matrix[n:]@coefficients).reshape_as(target[n:])*norm+skip[n:]
            rows.extend(response_rows(prediction,target[n:],held,bank=bank,readout=kind,rates=with_rates,direction=label,
                                      fit_anchors=n,fit_ancestors=len({r['ancestor'] for r in records[:n]})))
            atomic_save(out/'technical'/f'{label}-{kind}-rates{int(with_rates)}.pt',dict(prediction=prediction,coefficients=coefficients,
                query_scale=query_scale,target_scale=norm,feature_mean=features[:n].mean(0),feature_scale=features[:n].std(0).clamp_min(.01)))
    write_metric_rows(rows,out,family=FAMILY,name='response')
    # Aggregate two-shot target-mean noise estimate; not a per-state noise floor.
    shots=data['response_shots'][n:]; noise=(shots[:,0]-shots[:,1]).square()/4
    noise_rows=[]
    for direction,label in enumerate(['q','v']):
        for g,(a,b) in enumerate(GROUPS):
            values=noise[:,:,a:b,direction].mean(-1).numpy()
            for t in [0,3,9,19]:
                value,lo,hi=aggregate(values[:,t],[r['ancestor'] for r in held])
                noise_rows.append(dict(direction=label,group=g,horizon_fs=5*(t+1),mean_noise_power=value,lo=lo,hi=hi))
    write_metric_rows(noise_rows,out,family=FAMILY,name='teacher_noise')
    write_json(marker,dict(state='complete',fits=6,train=n,test=len(held),selection='none; fixed penalty/state length'))


def compare(c):
    rows=[]
    for arm in c['arms']:
        out=root(c)/'analyses'/arm['name']
        if not (out/'technical/evaluation-complete.json').exists(): continue
        import csv
        for table in ['predictions','sensitivity']:
            with (out/'tables'/f'{table}.csv').open() as stream:
                rows.extend(dict(arm=arm['name'],bank=arm['bank'],rates=arm['rates'],response=arm['response'],
                                 step_fs=arm['step_fs'],seed=arm['seed'],table=table,**r) for r in csv.DictReader(stream))
    if rows:
        write_metric_rows(rows,root(c)/'analyses/comparison',family=FAMILY,name='all_scores',
                          columns=list(dict.fromkeys(k for row in rows for k in row)))
