"""Independent full-cohort evaluation for the overnight memory/response study."""
from collections import defaultdict
import math
import time
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from .data import root, atomic_save, write_json, sha
from .memory import load, settings, MemoryProcess, native_history, FAMILY
from .learning import value_scores
from .train import response_prediction
from .evaluate import aggregate
from .assays import response_rows


def prior(initial, rate, steps, particles, seed, dt):
    generator = torch.Generator(device=initial.device).manual_seed(seed)
    value = initial[:, None].expand(-1, particles, -1)
    decay = torch.exp(-dt*rate); scale = (1-decay.square()).sqrt(); result = []
    for _ in range(steps):
        value = decay*value+scale*torch.randn(value.shape,device=value.device,generator=generator)
        result.append(value)
    return torch.stack(result,2)


def scramble(paths, seed):
    generator = torch.Generator(device=paths.device).manual_seed(seed)
    order = torch.rand(paths.shape[:3],device=paths.device,generator=generator).argsort(1)
    return paths.gather(1,order[...,None].expand_as(paths))


def scores(paths, targets, initial, weights, increments, c, physical):
    result = value_scores(paths,targets,initial,weights,increments,c['training'],physical=physical)
    actual = targets if physical else targets[:,None]
    result['endpoint_mse'] = ((paths.mean(1)[:,None,-1]-actual[:,:,-1]).square()*weights).sum(-1).mean(1)
    result['temporal_power'] = (paths.var(2,unbiased=False).mean(1)*weights).sum(-1)
    result['MD_temporal_power'] = (actual.var(2,unbiased=False).mean(1)*weights).sum(-1)
    result['mean_temporal_power'] = (paths.mean(1).var(1,unbiased=False)*weights).sum(-1)
    lo,hi = paths.quantile(.05,dim=1),paths.quantile(.95,dim=1)
    covered = (actual>=lo[:,None]) & (actual<=hi[:,None])
    result['measurement_90coverage'] = (covered.float()*weights).sum(-1).mean((1,2))
    for name,(a,b) in zip(['BO','TDA','density','velocity'],[(0,40),(40,153),(153,227),(227,244)]):
        w=weights[a:b]*4
        result[name+'_endpoint_mse']=((paths.mean(1)[:,None,-1,a:b]-actual[:,:,-1,a:b]).square()*w).sum(-1).mean(1)
        logp=-.5*((actual[:,:,None,:,a:b]-paths[:,None,:,:,a:b]).square()/c['training']['likelihood_bandwidth']**2
            +2*math.log(c['training']['likelihood_bandwidth'])+math.log(2*math.pi))
        result[name+'_nll']=(-(torch.logsumexp(logp,2)-math.log(paths.shape[1]))*w).sum(-1).mean((1,2))
        result[name+'_90coverage']=(covered[...,a:b].float()*w).sum(-1).mean((1,2))
    return result


def run(c, arm, deadline=float('inf')):
    out=root(c)/'analyses'/arm['name']; folder=out/'technical'
    if (folder/'evaluation-complete.json').exists(): return True
    _,data,x,bank,physical=load(c,device='cuda'); is_physical=arm['task']=='physical'
    rate,inc,weights=settings(data,physical,arm,'cuda')
    saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    model=MemoryProcess(c,rate,arm).cuda().eval(); model.load_state_dict(saved['model'])
    records=bank if is_physical else data['rows']; ids=np.flatnonzero([r['role']=='test' for r in records])
    horizons=[10,20] if is_physical else [10,30,50]; dt=.005 if is_physical else .1
    # Outer neural noise replications diagnose sensitivity MC error. Native
    # full-cohort scores use one common key, plus three keys for shown paths.
    keys=c['evaluation']['outer_keys'] if is_physical else c['evaluation']['outer_keys'][:1]
    batch=c['evaluation']['batch_size']; particles=c['evaluation']['particles']
    metrics=defaultdict(list); cases=[]; response=[]; half_response=[]; offset_start=0
    fingerprint=sha(folder/'best.pt'); partial=folder/'evaluation-partial.pt'
    if partial.exists():
        old=torch.load(partial,map_location='cpu',weights_only=False)
        if old['checkpoint_sha256']!=fingerprint: raise ValueError('Changed evaluated checkpoint')
        metrics.update(old['metrics']);cases=old['cases'];offset_start=old['next_offset']
        response=old['response'];half_response=old['half_response']
    def save(offset):
        atomic_save(partial,dict(checkpoint_sha256=fingerprint,next_offset=offset,metrics=dict(metrics),
            cases=cases,response=response,half_response=half_response))
    example_ids=set()
    for regime in sorted({r.get('regime','physical') for r in records}):
        available=[int(i) for i in ids if records[i].get('regime','physical')==regime]
        example_ids.update(np.random.default_rng(c['seed']).choice(available,min(4,len(available)),replace=False).tolist())
    with torch.no_grad():
        for offset in range(offset_start,len(ids),batch):
            if time.time()>deadline-300: save(offset);return False
            chosen=ids[offset:offset+batch]
            h=physical['history'][chosen,-1:] if is_physical else native_history(x[chosen],arm)
            targets=physical['target'][chosen] if is_physical else x[chosen,14:]
            predictions=[]
            for outer,key in enumerate(keys):
                seed=key+offset
                predicted=model(h,1 if is_physical else 50,particles,seed,dense=is_physical)
                predictions.append(predicted)
                baseline=prior(h[:,-1],rate,max(horizons),particles,seed,dt)
                for method,paths in [('model',predicted),('ou',baseline),('shuffled_particles',scramble(predicted,seed+1))]:
                    for steps in horizons:
                        target=targets[:,:,:steps] if is_physical else targets[:,:steps]
                        values=scores(paths[:,:,:steps],target,h[:,-1],weights,inc,c,is_physical)
                        for metric,value in values.items(): metrics[(outer,method,steps,metric)].append(value.cpu().numpy())
                for steps in horizons:
                    target=targets[:,:,steps-1] if is_physical else targets[:,None,steps-1]
                    for method,value in [('mean',torch.zeros_like(h[:,-1])),('persistence',h[:,-1])]:
                        error=((target-value[:,None]).square()*weights).sum(-1).mean(1)
                        metrics[(outer,method,steps,'endpoint_mse')].append(error.cpu().numpy())
            if is_physical:
                rr=[];hh=[]
                for key in keys:
                    rr.append(response_prediction(model,h,physical['pairs'][chosen],c['training']['response_epsilon'],key+offset,particles).cpu())
                    hh.append(response_prediction(model,h,physical['half_pairs'][chosen],c['training']['response_epsilon']/2,key+offset,particles).cpu())
                response.append(torch.stack(rr,1));half_response.append(torch.stack(hh,1))
            for k,index in enumerate(chosen):
                if int(index) not in example_ids: continue
                shown=[p[k].cpu() for p in predictions]
                if not is_physical:
                    for key in c['evaluation']['outer_keys'][1:]: shown.append(model(h[k:k+1],50,particles,key+offset).squeeze(0).cpu())
                cases.append(dict(index=int(index),row={k:v for k,v in records[index].items() if not torch.is_tensor(v)},
                    initial=h[k,-1].cpu(),truth=targets[k].cpu(),samples=torch.stack(shown)))
            if offset%320==0:
                save(offset+len(chosen));print('evaluate',arm['name'],offset+len(chosen),len(ids),flush=True)
    save(len(ids))
    ancestors=[records[i]['ancestor'] for i in ids]; arrays={k:np.concatenate(v) for k,v in metrics.items()}; rows=[]
    regimes=np.array([records[i].get('regime','physical') for i in ids])
    for (outer,method,steps,metric),values in arrays.items():
        for regime in ['all',*sorted(set(regimes))]:
            mask=np.ones(len(ids),dtype=bool) if regime=='all' else regimes==regime
            sources=np.array(ancestors)[mask].tolist();mean,lo,hi=aggregate(values[mask],sources)
            rows.append(dict(outer=outer,method=method,regime=regime,horizon_ps=steps*dt,metric=metric,value=mean,lo=lo,hi=hi,rows=int(mask.sum()),ancestors=len(set(sources))))
    atomic_save(folder/'evaluation-arrays.pt',dict(indices=ids,ancestors=ancestors,values=arrays,dt_ps=dt))
    atomic_save(folder/'examples.pt',dict(cases=cases,columns=data['columns'],mean=data['mean'],scale=data['scale'],dt_ps=dt,physical=is_physical))
    write_metric_rows(rows,out,family=FAMILY,name='predictions')
    if is_physical:
        response=torch.cat(response);half_response=torch.cat(half_response)
        actual=physical['response'][ids].cpu();initial=physical['initial_response'][ids].cpu(); rr=[]
        for direction,label in enumerate(['q','v']):
            for outer in range(len(keys)):
                for method,prediction in [('model',response[:,outer]),('half_epsilon',half_response[:,outer])]:
                    rr.extend(response_rows(prediction[...,direction],actual[...,direction],[bank[i] for i in ids],outer=outer,method=method,direction=label))
            for method,prediction in [('zero',torch.zeros_like(actual)),('persistence',initial[:,None].expand_as(actual))]:
                rr.extend(response_rows(prediction[...,direction],actual[...,direction],[bank[i] for i in ids],outer=-1,method=method,direction=label))
        atomic_save(folder/'sensitivity-arrays.pt',dict(prediction=response,half_prediction=half_response,target=actual,initial=initial,indices=ids,ancestors=ancestors))
        write_metric_rows(rr,out,family=FAMILY,name='sensitivity')
    plot(out)
    update_recorded_summary(folder/'wandb.json',{'followup/'+r['metric']+'/'+str(r['horizon_ps'])+'ps':r['value'] for r in rows if r['method']=='model' and r['outer']==0 and r['regime']=='all'},evaluation='overnight-full-heldout',expected={'mode':'online'})
    write_json(folder/'evaluation-complete.json',dict(state='complete',rows=len(ids),ancestors=len(set(ancestors)),checkpoint_sha256=fingerprint,selected_update=saved['update']))
    return True


def plot(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    saved=torch.load(out/'technical/examples.pt',map_location='cpu',weights_only=False)
    folder=out/'plots/trajectories';folder.mkdir(parents=True,exist_ok=True)
    # Fixed channels across runs, including signed bond/flow measurements.
    channels=[0,16,40,153,227,237]
    for case in saved['cases']:
        fig,axes=plt.subplots(3,2,figsize=(12,9),constrained_layout=True)
        initial=case['initial']; truth=case['truth'] if saved['physical'] else case['truth'][None]
        samples=case['samples'][0]; times=np.arange(samples.shape[1]+1)*saved['dt_ps']
        for ax,k in zip(axes.flat,channels):
            paths=torch.cat((initial[k].expand(len(samples),1),samples[...,k]),1)*saved['scale'][k]+saved['mean'][k]
            observed=torch.cat((initial[k].expand(len(truth),1),truth[...,k]),1)*saved['scale'][k]+saved['mean'][k]
            for j in range(min(8,len(paths))): ax.plot(times,paths[j],color='tab:blue',alpha=.16,lw=.7)
            ax.fill_between(times,paths.quantile(.05,dim=0),paths.quantile(.95,dim=0),alpha=.15,color='tab:blue')
            ax.plot(times,paths.mean(0),color='tab:blue',label='Neural mean and 90% band')
            for j,path in enumerate(observed): ax.plot(times,path,color='black',lw=1.2,alpha=.8,label='Saved physical path' if j==0 else None)
            ax.set_title(str(saved['columns'][k]));ax.set_xlabel('Time after forecast origin (ps)');ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8);fig.suptitle(f'{out.name} · fixed center · row {case["index"]}')
        fig.savefig(folder/f'row-{case["index"]}.png',dpi=130);plt.close(fig)


def compare(c):
    rows=[];sensitivity=[]
    for seed in sorted({a['seed'] for a in c['arms']}):
        arms={a['treatment']:root(c)/'analyses'/a['name']/'technical' for a in c['arms'] if a['seed']==seed}
        for treatment,control in [('history14','history4'),('history_summary','history4'),('history14','history_summary'),('physical_response','physical_values')]:
            left,right=[arms[key]/'evaluation-arrays.pt' for key in [treatment,control]]
            if not (left.exists() and right.exists()): continue
            a,b=[torch.load(path,map_location='cpu',weights_only=False) for path in [left,right]]
            if a['ancestors']!=b['ancestors'] or not np.array_equal(a['indices'],b['indices']): raise ValueError('Unpaired evaluation rows')
            for key in a['values']:
                outer,method,steps,metric=key
                if method!='model':continue
                value,lo,hi=aggregate(a['values'][key]-b['values'][key],a['ancestors'],seed)
                rows.append(dict(seed=seed,treatment=treatment,control=control,outer=outer,horizon_ps=steps*a['dt_ps'],metric=metric,value=value,lo=lo,hi=hi))
        left,right=[arms[key]/'sensitivity-arrays.pt' for key in ['physical_response','physical_values']]
        if left.exists() and right.exists():
            a,b=[torch.load(path,map_location='cpu',weights_only=False) for path in [left,right]]
            if a['ancestors']!=b['ancestors'] or not torch.equal(a['target'],b['target']):raise ValueError('Unpaired physical response targets')
            ancestors=np.array(a['ancestors']);unique=sorted(set(ancestors));draws=np.random.default_rng(seed).integers(len(unique),size=(1000,len(unique)))
            for group,(lo,hi) in zip(['density','velocity_shells','bond_rates','flow'],[(0,74),(74,83),(83,87),(87,91)]):
                target=a['target'][:,:,lo:hi]; power=target.square().mean(2).numpy()
                for outer in range(a['prediction'].shape[1]):
                    delta=((a['prediction'][:,outer,:,lo:hi]-target).square()-(b['prediction'][:,outer,:,lo:hi]-target).square()).mean(2).numpy()
                    dd=np.stack([delta[ancestors==s].mean(0) for s in unique]);pp=np.stack([power[ancestors==s].mean(0) for s in unique])
                    value=dd.mean(0)/pp.mean(0).clip(1e-5);boot=dd[draws].mean(1)/pp[draws].mean(1).clip(1e-5);lower,upper=np.quantile(boot,[.025,.975],axis=0)
                    for t in range(20):
                        for d,direction in enumerate(['q','v']):sensitivity.append(dict(seed=seed,outer=outer,group=group,direction=direction,horizon_fs=5*(t+1),value=float(value[t,d]),lo=float(lower[t,d]),hi=float(upper[t,d])))
    if rows: write_metric_rows(rows,root(c)/'analyses/comparison',family=FAMILY,name='paired_predictions')
    if sensitivity:write_metric_rows(sensitivity,root(c)/'analyses/comparison',family=FAMILY,name='paired_sensitivity')
