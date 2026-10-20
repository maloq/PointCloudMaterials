"""Independent path, rare-entry, response and actual-nucleus evaluations."""
from collections import defaultdict
import csv,time
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.research.ni_kinetic_latent import assessment as metrics
from src.research.ni_phase_sim.births import fit_spaces
from src.research.ni_kinetic_latent.births import _compare_native, _compare_births
from .data import load,load_births,teacher_bank,root,cache,read,sha,atomic_save,write_json,inverse,FAMILY
from .learning import fitted


def diagnostics(c):
    """Reuse the original fixed measurement-phase readout and increment units."""
    parent=read(root(c)/'technical/plan.json')
    source=__import__('pathlib').Path(c['diagnostic_path'])
    if sha(source)!=c['diagnostic_sha256']:raise ValueError('Changed fixed diagnostic readout')
    return torch.load(source,map_location='cpu',weights_only=False)


def evaluate(c,arm,deadline,rare=False):
    out=root(c)/'analyses'/arm['name'];tech=out/'technical'
    name='rare' if rare else 'evaluation';marker=tech/(name+'-complete.json')
    if marker.exists():return True
    p,bank=load(c,arm);model,saved=fitted(c,arm);d=diagnostics(c)
    labels=torch.load(p['labels_path'],map_location='cpu',weights_only=False)
    ids=metrics.selected_rows(bank,c['evaluation']['rare_per_source_regime'] if rare else 100000,c['seed'])
    particles=c['evaluation']['rare_particles'] if rare else c['evaluation']['particles']
    batch=2 if rare else c['evaluation']['batch_size']
    values=defaultdict(list);examples=[];start=0;partial=tech/(name+'-partial.pt')
    if partial.exists():
        packet=torch.load(partial,map_location='cpu',weights_only=False)
        if packet['identity']!=saved['identity'] or not np.array_equal(packet['ids'],ids):raise ValueError('Changed evaluation identity')
        values.update(packet['values']);start=packet['next'];examples=packet['examples']
    def save(n):atomic_save(partial,dict(identity=saved['identity'],ids=ids,values=dict(values),next=n,examples=examples))
    weights=bank['weights'].cuda();increments=d['increment_scale'].cuda()
    cols=d['order_columns'][:2];threshold=d['high_order_threshold']
    with torch.no_grad():
        for offset in range(start,len(ids),batch):
            if time.time()>deadline-180:save(offset);return False
            ix=ids[offset:offset+batch];truth=bank['x'][ix,4:].cuda();initial=bank['x'][ix,3].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                generated=model.rollout(bank['history'][ix].cuda(),60,particles,c['seed']+offset)
            generated=generated.float()
            if not torch.isfinite(generated).all():raise FloatingPointError('Nonfinite autonomous generation at '+str(offset))
            actual_phase=np.asarray(labels['crystal'])[ix,4:];truth_entry=metrics.first_entry(actual_phase)
            predicted_phase=metrics.probability(generated.cpu().numpy(),d['probe'])>=.5
            oracle_phase=metrics.probability(truth.cpu().numpy(),d['probe'])>=.5
            entries=metrics.first_entry(predicted_phase);oracle_entry=metrics.first_entry(oracle_phase)
            for steps in (10,30,60):
                event=truth_entry<=steps;k=(entries<=steps).sum(1);prob=k/particles
                scores=dict(local_entry_brier=(prob-event)**2,
                    local_entry_brier_mc_corrected=k*(k-1)/(particles*(particles-1))-2*event*prob+event,
                    local_entry_observed=event.astype(float),local_entry_predicted=prob,
                    local_entry_oracle_brier=((oracle_entry<=steps)-event.astype(float))**2,
                    local_entry_negative_brier=event.astype(float))
                for key,v in scores.items():values[('model',steps,key)].append(v)
                if not rare:
                    for method,samples in [('model',generated[:,:,:steps]),('persistence',initial[:,None,None].expand(-1,1,steps,-1))]:
                        for key,v in metrics.scores(samples,truth[:,:steps],initial,weights,increments).items():
                            values[method,steps,key].append(v.cpu().numpy())
            if not rare:
                raw=inverse(generated.cpu(),bank).numpy();actual=inverse(truth.cpu(),bank).numpy()
                valid=np.isfinite(raw).all((1,2,3))
                high=(raw[...,cols[0]]>threshold[0])&(raw[...,cols[1]]>threshold[1])&~predicted_phase
                actual_high=(actual[...,cols[0]]>threshold[0])&(actual[...,cols[1]]>threshold[1])&~actual_phase
                for steps in (10,30,60):
                    occupancy=high[:,:,:steps].mean((1,2));observed=actual_high[:,:steps].mean(1)
                    scores=dict(high_order_occupancy=occupancy,high_order_observed_occupancy=observed,
                        high_order_occupancy_error=(occupancy-observed)**2,
                        high_order_longest_dwell_error=(metrics.longest_run(high[:,:,:steps]).mean(1)-metrics.longest_run(actual_high[:,:steps]))**2)
                    for key,v in scores.items():
                        v[~valid]=np.nan;values['model',steps,key].append(v)
                if len(examples)<8:
                    examples.append(dict(row=bank['rows'][int(ix[0])],truth=actual[0,:,cols].T,
                        samples=raw[0][:,:,cols],history=bank['history'][ix[0],:,:244]))
            save(offset+len(ix));write_json(tech/(name+'-progress.json'),dict(rows=offset+len(ix),total=len(ids),particles=particles,updated=time.time()))
    arrays={key:np.concatenate(v) for key,v in values.items()};anc=np.array([bank['rows'][i]['ancestor'] for i in ids])
    liquid=(~np.asarray(labels['crystal'])[ids,3])&(np.asarray(labels['fraction'])[ids,3]<=.05)
    reports=[];sources=[]
    for (method,steps,metric),v in arrays.items():
        for subset,mask in [('all',np.ones(len(ids),bool)),('present_liquid',liquid)]:
            scope=dict(model=arm['name'],method=method,horizon_ps=steps*.1,metric=metric,subset=subset,regime='all')
            reports.append(dict(scope,rows=int(mask.sum()),**metrics.estimate(v[mask],anc[mask],c['seed'],c['evaluation']['bootstrap'])))
            keys,means=metrics.source_values(v[mask],anc[mask])
            sources.extend(dict(scope,ancestor=str(k),value=float(x) if np.isfinite(x) else None) for k,x in zip(keys,means))
    table='rare_entry' if rare else 'trajectory'
    write_metric_rows(reports,out,family=FAMILY,name=table);write_metric_rows(sources,out,family=FAMILY,name=table+'_sources')
    if not rare:
        atomic_save(tech/'examples.pt',examples)
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,4,figsize=(16,7),squeeze=False)
        for ax,example in zip(axes.flat,examples):
            x=np.arange(1,61)*.1;ys=example['samples'][...,0]
            ax.plot(x,ys[:8].T,alpha=.25,color='tab:blue');ax.plot(x,example['truth'][:,0],color='black',label='MD')
            ax.plot(x,ys.mean(0),color='tab:blue',label='sample mean');ax.set_title(example['row']['regime']);ax.set_xlabel('Future offset (ps)')
        axes[0,0].legend();fig.tight_layout();(out/'plots').mkdir(exist_ok=True);fig.savefig(out/'plots/order-trajectories.png',dpi=150);plt.close(fig)
    update_recorded_summary(tech/'wandb.json',{f'{table}/{r["subset"]}/{r["method"]}/{r["horizon_ps"]}/{r["metric"]}':r['value'] for r in reports if r['value'] is not None},evaluation=name,expected=dict(identity=saved['identity']))
    write_json(marker,dict(state='complete',rows=len(ids),sources=len(set(anc)),particles=particles,checkpoint_sha256=sha(tech/'final.pt')))
    return True


def response(c,arm,deadline):
    out=root(c)/'analyses'/arm['name'];tech=out/'technical';marker=tech/'response-complete.json'
    if marker.exists():return True
    _,bank=load(c,arm);t=teacher_bank(c,bank);model,_=fitted(c,arm)
    take=~t['train'];current=t['value'][take,0].cuda();truth=t['response'][take,1:].cuda();power=t['power'].cuda()
    results=[]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    for start in range(0,len(current),16):
        q=current[start:start+16];v=model.dense(q)
        for direction in range(2):
            tangent=torch.zeros_like(q);tangent[:,153:]=t['response'][take,0,:,direction][start:start+16].cuda()
            with sdpa_kernel(SDPBackend.MATH):_,jvp=torch.autograd.functional.jvp(model.dense,q,tangent)
            for frame in (0,3,9,19):
                target=truth[start:start+16,frame,:,direction]
                results.append(dict(frame_fs=(frame+1)*5,direction=direction,rows=len(q),
                    response_error=float(((jvp[:,frame]-target).square()/power).mean()),
                    zero_response_error=float((target.square()/power).mean()),
                    value_error=float((v[:,frame]-t['value'][take,frame+1,153:][start:start+16].cuda()).square().mean())))
    grouped=[]
    for frame in (5,20,50,100):
        for direction in range(2):
            rows=[r for r in results if r['frame_fs']==frame and r['direction']==direction]
            grouped.append(dict(frame_fs=frame,direction=direction,rows=sum(r['rows'] for r in rows),**{k:np.average([r[k] for r in rows],weights=[r['rows'] for r in rows]) for k in ('response_error','zero_response_error','value_error')}))
    write_metric_rows(grouped,out,family=FAMILY,name='response');write_json(marker,dict(state='complete',teacher_states=int(take.sum())))
    return True


def births(c,arm,deadline):
    out=root(c)/'analyses'/arm['name'];tech=out/'technical';marker=tech/'births-complete.json'
    if marker.exists():return True
    p,_=load(c,arm);cohort,h=load_births(c,arm);model,saved=fitted(c,arm)
    featurefile=tech/'birth-features.pt'
    feature=dict(next=0,z=[],future=[])
    if featurefile.exists():feature=torch.load(featurefile,map_location='cpu',weights_only=False)
    with torch.no_grad():
        for offset in range(feature['next'],len(h),c['evaluation']['batch_size']):
            if time.time()>deadline-180:atomic_save(featurefile,feature);return False
            history=h[offset:offset+c['evaluation']['batch_size']].cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=model.features(history);future=model.rollout(history,60,32,c['seed']+offset)
            # Fixed summaries: endpoint mean and standard deviation at1/3/6ps.
            summary=torch.cat([part for frame in (9,29,59) for part in
                (future[:,:,frame].mean(1),future[:,:,frame].std(1,unbiased=False))],-1)
            feature['z'].append(z.float().cpu());feature['future'].append(summary.float().cpu());feature['next']=offset+len(history)
            if offset%256==0:
                atomic_save(featurefile,feature);write_json(tech/'births-progress.json',dict(rows=feature['next'],total=len(h),stage='features'))
    atomic_save(featurefile,feature)
    spaces=dict(history=h.numpy().reshape(len(h),-1),history_representation=torch.cat(feature['z']).numpy(),
                generated_future=torch.cat(feature['future']).numpy())
    settings=dict(c['birth_probe'],output=str(root(c)/'births'),cache=str(cache(c)/'birth-features'))
    failures=fit_spaces(settings,p,cohort,arm['name'],spaces,family=FAMILY)
    if failures:raise RuntimeError('Failed fixed birth readouts; see receipts')
    # A fixed linear likelihood probe is a diagnostic, with train-only scaling.
    from sklearn.linear_model import LogisticRegression
    from src.research.birth_prediction.fit import scores
    from threadpoolctl import threadpool_limits
    reports=[]
    for space,x in spaces.items():
        for task in settings['tasks']:
            for lead in settings['leads_ps']:
                indices=[i for i,r in enumerate(cohort['rows']) if r['task']==task and r['lead']==lead]
                rows=[cohort['rows'][i] for i in indices];f=x[np.asarray(cohort['indices'])[indices]]
                tr=np.array([r['role']=='train' for r in rows]);y=np.array([r['label'] for r in rows]);w=np.array([r['weight'] for r in rows])
                mean=f[tr].mean(0);scale=f[tr].std(0).clip(1e-5)
                with threadpool_limits(limits=2):
                    classifier=LogisticRegression(C=1.,max_iter=1000).fit((f[tr]-mean)/scale,y[tr],sample_weight=w[tr]/w[tr].mean())
                    prob=classifier.predict_proba((f-mean)/scale)[:,1]
                for metric,value in scores(y[~tr],prob[~tr],w[~tr]).items():
                    reports.append(dict(space=space,task=task,lead_ps=lead,metric=metric,value=value,test_rows=int((~tr).sum())))
    write_metric_rows(reports,out,family=FAMILY,name='birth_linear')
    write_json(marker,dict(state='complete',rows=len(cohort['rows']),unique=len(h),support='nearest80 reselected each observation; same central atom; no neighborhood filter',checkpoint_sha256=sha(tech/'final.pt')))
    return True


def compare(c):
    out=root(c)/'analyses/comparison'
    _compare_native(c,out,family=FAMILY,treatment_key='treatment')
    _compare_births(c,out,family=FAMILY,treatment_key='treatment',spaces=['history','history_representation','generated_future'])
