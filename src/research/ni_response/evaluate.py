"""Local frozen probes, training-only NLL selection, pooled held-out Ni sources."""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from src.research.response_dense_birth.pooled import fit
from src.research.birth_prediction.fit import scores
from src.research.birth_prediction.analysis import source_draws,bootstrap_scores
from src.research.structural_response.train import predict_batches
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from .data import load,read,root,cache,sha,write_json,digest,FAMILY
from .model import Motion
from src.research.equivariant_context.cache import RetainedCache
from .data import atomic_save


def infer(c,p,name,micro):
    metadata=dict(plan=p['identity'],model=name,producer=sha(Path(__file__)))
    pool=RetainedCache(cache(c)/'features',6)
    with pool.lease(digest(metadata),deadline=time.time()+86400,metadata=metadata) as folder:
        receipt=folder/'complete.json'
        if receipt.exists():
            if sha(folder/'features.pt')!=read(receipt)['sha256']:raise ValueError('Changed feature cache')
        else:
            values=infer_values(c,p,name,micro)
            atomic_save(folder/'features.pt',values)
            write_json(receipt,dict(sha256=sha(folder/'features.pt')))
        write_json(cache(c)/('features-'+name+'.json'),dict(path=str(folder),metadata=metadata))


def infer_values(c,p,name,micro):
    model=Motion(c).cuda().eval()
    if name!='spatial':
        arm=next(a for a in c['arms'] if a['name']==name);model.set_arm(arm)
        f=root(c)/'analyses'/name/'technical';done=read(f/'complete.json')
        if sha(f/'best.pt')!=done['checkpoint_sha256']:raise ValueError('Changed inference checkpoint')
        saved=torch.load(f/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model'])
    values={}
    for task in ['data','births']:
        data=load(c,task);z=[]
        with torch.no_grad():
            for x in data['state'].split(micro):z.append(model.encode(x.cuda(),create_graph=False).cpu())
        values[task]=torch.cat(z).numpy()
    if name!='spatial':
        data=load(c,'data');target=load(c,'targets');ids=np.flatnonzero([r['role']=='test' for r in data['rows']])
        values['future']=predict_batches(model,data,target['future'],target['codes'] if arm['frozen'] else None,ids,c,micro)
        values['future_ids']=ids
    return values


def cohorts(c):
    data=load(c,'data');birth=load(c,'births')
    clear=np.flatnonzero(data['clear']);result=[]
    for horizon in c['appearance_horizons_ps']:
        rows=[]
        for i in clear:
            r=data['rows'][i];rows.append(dict(r,role=r['original_role'],label=int(data['first'][i]<=horizon+1e-8),event=r['id'],pair=r['id'],weight=1.))
        result.append(('appearance',horizon,'data',clear,rows))
    for task in ['nucleus32','nucleus64']:
        for lead in c['birth_leads_ps']:
            indices=[i for i,r in enumerate(birth['rows']) if r['task']==task and r['lead']==lead]
            rows=[dict(birth['rows'][i]) for i in indices]
            counts=Counter(r['event'] for r in rows)
            for r in rows:r['weight']=1/counts[r['event']]
            result.append((task,lead,'births',indices,rows))
    return result


def evaluate(c,p,name,*,family=FAMILY):
    pointer=read(cache(c)/('features-'+name+'.json'));pool=RetainedCache(cache(c)/'features',6)
    with pool.lease(digest(pointer['metadata']),deadline=time.time()+86400,metadata=pointer['metadata'],shared=True) as folder:
        if sha(folder/'features.pt')!=read(folder/'complete.json')['sha256']:raise ValueError('Changed evaluated feature cache')
        features=torch.load(folder/'features.pt',weights_only=False,map_location='cpu')
    report=[];coverage=[];summary={}
    for task,horizon,key,indices,records in cohorts(c):
        rows={k:np.asarray([r[k] for r in records]) for k in ['id','ancestor','role','event','pair','label','weight']}
        partition=p['partition'];ix={k:np.flatnonzero(np.isin(rows['ancestor'],v)) for k,v in partition.items()}
        coverage.append(dict(task=task,offset_ps=horizon,rows=len(records),train_rows=len(ix['final_train']),test_rows=len(ix['test']),test_positive_rows=int(rows['label'][ix['test']].sum()),test_events=len(set(rows['event'][ix['test']])),positive_test_events=len({r['event'] for r in records if r['role']!='train' and r['label']})))
        absent=[k for k in ['inner_fit','inner_selection','test'] if len(np.unique(rows['label'][ix[k]]))<2]
        if absent:
            coverage[-1]['unavailable_reason']='both classes unavailable in '+','.join(absent);continue
        ec=dict(c,output=str(root(c)/'analyses'/task));plan=dict(identity=digest([p['identity'],task,horizon]),partition=partition)
        test=ix['test'];y=rows['label'][test];w=rows['weight'][test]
        ancestors,slot,counts=source_draws(rows['ancestor'][test],c['evaluation']['bootstrap_draws'],c['seed'])
        allz=features[key][indices]
        for space in c['evaluation']['spaces']:
            x={'structural':allz[:,:128],'motion':allz[:,128:],'structural_motion':allz}[space]
            for kind in ['linear','catboost']:
                fit(ec,dict(config=c['evaluation']),rows,x,dict(name=name),space,horizon,kind,plan=plan)
                out=root(ec)/'analyses/fits-v1'/name/space/f'lead-{horizon:g}ps'/kind/'technical'
                probability=np.load(out/'predictions.npz')['probability'][test]
                vals=scores(y,probability,w);boots,_=bootstrap_scores(y,probability,w,slot,counts,None)
                for metric in ['nll','brier','ap','auroc']:
                    lo,hi=np.quantile(boots[metric],[.025,.975]) if len(boots[metric]) else (None,None)
                    report.append(dict(model=name,task=task,offset_ps=horizon,offset_kind='future_horizon' if task=='appearance' else 'lead_before_persistent_threshold_start',space=space,readout=kind,metric=metric,value=vals[metric],lo=lo,hi=hi,test_ancestors=len(ancestors),test_rows=len(test),prevalence=float(np.average(y,weights=w))))
                    summary[f'{task}/{horizon:g}ps/{space}/{kind}/{metric}']=vals[metric]
        prior=float(np.average(rows['label'][ix['final_train']],weights=rows['weight'][ix['final_train']]))
        vals=scores(y,np.full(len(y),prior),w)
        for metric in ['nll','brier','ap','auroc']:report.append(dict(model='training_prior',task=task,offset_ps=horizon,offset_kind='future_horizon' if task=='appearance' else 'lead_before_persistent_threshold_start',space='prior',readout='prior',metric=metric,value=vals[metric],lo=None,hi=None,test_ancestors=len(ancestors),test_rows=len(test),prevalence=float(np.average(y,weights=w))))
    out=root(c)/'analyses/evaluation-v1'/name
    if report:write_metric_rows(report,out,family=family,name='scores',columns=list(report[0]))
    write_json(out/'technical/coverage.json',coverage)
    if name!='spatial':
        future=features['future'];fr=[]
        for metric in ['nll','descriptor_mse','structural_mse','coverage90']:
            for j,h in enumerate(c['horizons_ps']):
                value=float(future[metric][:,j].mean());fr.append(dict(model=name,horizon_ps=h,metric=metric,value=value));summary[f'future/{h:g}ps/{metric}']=value
        write_metric_rows(fr,out,family=family,name='future',columns=list(fr[0]))
        receipt=root(c)/'analyses'/name/'technical/wandb.json'
        update_recorded_summary(receipt,summary,evaluation='ni-three-tasks-v1',expected=dict(mode='online'))
    write_json(out/'technical/complete.json',dict(state='complete',model=name,rows=len(report)))
