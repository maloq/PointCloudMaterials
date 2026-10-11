"""Consolidated birth prediction with no additional neighborhood screening."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor,as_completed
import fcntl
import hashlib
import multiprocessing
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree
from .data import read,write_json,digest,sha,resolve_path,root,FAMILY
from .birth_data import prepare
from src.postprocessing.io import geometry
from src.postprocessing.io import arrays
from src.experiment_runner.metric_docs import write_metric_rows


def extract_source(c,p,item):
    """Actual Q,V patches around catalogue founders and matched nonbirth atoms.

    Control centers exclude founder IDs of every established birth catalogued
    in this source. No current crystal fraction, component size or distance to
    crystals determines eligibility for positives or controls.
    """
    spec=c['birth_evaluation'];folder=resolve_path(spec['cache'])/'sources'/item['source_id'];folder.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(cohort=p['identity'],source=item['source_id'],implementation=sha(Path(__file__))))
    if (folder/'complete.json').exists():
        done=read(folder/'complete.json')
        if done['identity']!=identity:raise ValueError('Changed no-filter source extraction')
        return done
    selected=[e for e in p['events'] if e['source_id']==item['source_id']]
    events_path=Path(item['label_folder'])/'events.json'
    if sha(events_path)!=item['label_files']['events.json']:raise ValueError('Changed catalogue label shard')
    upstream=read(events_path);events={e['event_id']:upstream[e['event_row']] for e in selected}
    s=item['source'];native=arrays(s['parts'][0]['path'],s['parts'][0]['manifest_sha256']);atom_ids=native['atom_ids']
    velocities=[]
    for part in s['parts']:
        m=read(Path(part['path'])/'manifest.json');v=m['arrays']['velocities']
        velocities.append(np.load(Path(part['path'])/v['file'],mmap_mode='r'))
        if not np.array_equal(arrays(part['path'],part['manifest_sha256'])['atom_ids'],atom_ids):raise ValueError('Atom ordering changed across source parts')
    forbidden=set(a for e in events.values() for a in e['first_members'])
    available=np.flatnonzero(~np.isin(atom_ids,list(forbidden)))
    rng=np.random.default_rng(int(digest([c['seed'],item['source_id']])[:16],16))
    leads=np.asarray(spec['leads_ps']);view=np.asarray(s['views']['010']);times=np.asarray(s['times_ps'])
    steps=np.rint(leads/.1).astype(int);states=[];rows=[];coverage=[]
    for summary in selected:
        e=events[summary['event_id']];first=e['first_frame'];slot=int(np.searchsorted(view,first))
        if slot==len(view) or view[slot]!=first:raise ValueError('Birth is outside010 grid')
        if slot<steps.max():
            coverage.append(dict(event_id=summary['event_id'],source=s['id'],reason='missing_exact_lead',accepted=False));continue
        frames=view[slot-steps]
        if not np.allclose(times[first]-times[frames],leads,atol=1e-8,rtol=0):raise ValueError('Birth lead mismatch')
        members=np.searchsorted(atom_ids,e['first_members'])
        if not np.array_equal(atom_ids[members],e['first_members']):raise ValueError('Missing founder atom IDs')
        positives=rng.permutation(members)[:spec['centers_per_event']]
        if len(available)<4*len(positives):raise ValueError('Insufficient nonbirth control atoms')
        negatives=rng.choice(available,4*len(positives),replace=False)
        centers=np.column_stack((positives,negatives.reshape(-1,4))).ravel()
        q0,box0=geometry(s,first);pi,rf=s['frames'][first];v0=np.asarray(velocities[pi][rf],np.float32)
        physical='birth-'+digest(dict(ancestor=s['ancestry_group'],time_ps=round(float(times[first]),8),
            positions=hashlib.sha256(np.ascontiguousarray(q0).tobytes()).hexdigest(),
            velocities=hashlib.sha256(np.ascontiguousarray(v0).tobytes()).hexdigest(),
            box=box0.tolist(),first_members=e['first_members']))[:32]
        patches=[]
        for frame in frames:
            q,box=geometry(s,int(frame));_,neighbors=cKDTree(q,boxsize=box).query(q[centers],k=80,workers=1)
            if not np.array_equal(neighbors[:,0],centers):raise ValueError('Patch center identity mismatch')
            delta=q[neighbors]-q[centers,None];delta-=box*np.rint(delta/box)
            pi,rf=s['frames'][int(frame)];v=np.asarray(velocities[pi][rf,neighbors],np.float32)
            patches.append(np.concatenate((delta.astype(np.float32),v),-1))
        block=np.stack(patches,axis=1);offset=len(states);states.extend(block)
        for j,center in enumerate(centers):
            case=j//5;pair=summary['event_id']+':'+str(int(atom_ids[positives[case]]))
            rows.append(dict(id=pair+':'+str(j%5),pair=pair,event=physical,event_view=summary['event_id'],
                atom=int(atom_ids[center]),label=int(j%5==0),ancestor=s['ancestry_group'],role=s['role'],
                source=s['id'],source_id=item['source_id'],native_cadence_ps=s['native_cadence_ps'],
                index=offset+j,input_hash=hashlib.sha256(block[j].tobytes()).hexdigest()))
        coverage.append(dict(event_id=summary['event_id'],source=s['id'],physical_event=physical,reason='accepted',accepted=True))
    values=np.asarray(states,dtype=np.float32).reshape(-1,len(leads),80,6)
    np.save(folder/'states.npy',values);write_json(folder/'rows.json',rows);write_json(folder/'coverage.json',coverage)
    done=dict(identity=identity,source=s['id'],source_id=item['source_id'],rows=len(rows),candidate_events=len(selected),
        accepted_event_records=sum(e['accepted'] for e in coverage),neighborhood_filter=False,
        files={n:sha(folder/n) for n in ('states.npy','rows.json','coverage.json')})
    write_json(folder/'complete.json',done);return done


def prepare_observations(c):
    p=prepare(c);spec=c['birth_evaluation'];folder=resolve_path(spec['cache']);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'prepare.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (folder/'complete.json').exists():
            done=read(folder/'complete.json')
            if done['cohort']!=p['identity']:raise ValueError('Changed prepared birth inputs')
            return done
        active={e['source_id'] for e in p['events']};sources=[s for s in p['sources'] if s['source_id'] in active]
        with ProcessPoolExecutor(max_workers=spec['prepare_workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
            tasks=[executor.submit(extract_source,c,p,item) for item in sources]
            for i,future in enumerate(as_completed(tasks)):
                done=future.result();write_json(folder/'progress.json',dict(state='extracting',sources=i+1,total=len(tasks),last=done))
                print('birth inputs',i+1,len(tasks),done['source'],done['accepted_event_records'],flush=True)
        states=[];rows=[];coverage=[];seen=set();aliases=[]
        for item in sources:
            part=folder/'sources'/item['source_id'];done=read(part/'complete.json')
            if any(sha(part/n)!=h for n,h in done['files'].items()):raise ValueError('Changed birth observation packet')
            r=read(part/'rows.json');x=np.load(part/'states.npy',mmap_mode='r');coverage.extend(read(part/'coverage.json'))
            for start in range(0,len(r),5):
                block=r[start:start+5];case=block[0]
                if len(block)!=5 or case['label']!=1 or sum(a['label']for a in block)!=1:raise ValueError('Malformed matched controls')
                key=(case['event'],case['atom'],case['input_hash'])
                if key in seen:
                    aliases.append(dict(pair=case['pair'],event=case['event'],reason='same physical birth, center and exact Q,V input'));continue
                seen.add(key)
                for row in block:
                    states.append(x[row['index']]);rows.append(dict(row,index=len(states)-1))
        if not rows:raise ValueError('No catalogue births have the required leads')
        count=Counter(r['event'] for r in rows)
        for r in rows:r['weight']=1/count[r['event']]
        np.save(folder/'states.npy',np.asarray(states,np.float32));write_json(folder/'rows.json',rows)
        write_json(folder/'coverage.json',coverage);write_json(folder/'aliases.json',aliases)
        counts=[]
        for role in ('train','pooled_test'):
            rr=[r for r in rows if (r['role']=='train')==(role=='train')]
            counts.append(dict(role=role,rows=len(rr),positive_rows=sum(r['label'] for r in rr),events=len({r['event']for r in rr}),ancestors=len({r['ancestor']for r in rr})))
        write_metric_rows(counts,root(c)/'analyses/birth-cohort',family=FAMILY,name='coverage')
        done=dict(cohort=p['identity'],rows=len(rows),counts=counts,neighborhood_filter=False,
            velocity_units='angstrom_per_ps',files={n:sha(folder/n) for n in ('states.npy','rows.json','coverage.json','aliases.json')})
        write_json(folder/'complete.json',done);return done


def evaluate(c,model,method,arm,checkpoint,path,deadline):
    """Frozen NLL-selected readouts; all held-out roles are pooled, never fitted."""
    import torch
    from src.research.response_dense_birth.pooled import fit
    from src.research.birth_prediction.fit import scores
    from src.research.birth_prediction.analysis import source_draws,bootstrap_scores
    spec=c['birth_evaluation'];folder=resolve_path(spec['cache']);done=read(folder/'complete.json')
    if any(sha(folder/n)!=h for n,h in done['files'].items()):raise ValueError('Changed birth cohort inputs')
    raw=read(folder/'rows.json');rows={k:np.asarray([r[k]for r in raw]) for k in raw[0]}
    states=np.load(folder/'states.npy',mmap_mode='r');out=path.parent/'analyses/birth-prediction';out.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(cohort=done['cohort'],checkpoint=sha(checkpoint),arm=arm,implementation=sha(Path(__file__))))
    if (out/'technical/complete.json').exists():
        saved=read(out/'technical/complete.json')
        if saved['identity']!=identity:raise ValueError('Changed birth evaluation')
        return saved['metrics']
    ancestors=sorted(set(rows['ancestor'][rows['role']=='train']),key=lambda a:digest([c['seed'],str(a),'birth-inner-selection']))
    n=spec['inner_selection_ancestors'];held=sorted(set(rows['ancestor'][rows['role']!='train']))
    if set(ancestors)&set(held) or len(ancestors)<=n:raise ValueError('Invalid birth ancestor partition')
    partition=dict(inner_fit=ancestors[n:],inner_selection=ancestors[:n],final_train=ancestors,test=held)
    plan=dict(partition=partition,cohort=done['cohort'],identity=digest([identity,partition]))
    write_json(out/'technical/readout-plan.json',plan)
    fit_config=dict(spec,seed=c['seed'],output=str(out));producer=dict(config=fit_config)
    name=method+'-'+arm['name'];model_spec=dict(name=name,arm=arm['name'],seed=c['seed'],checkpoint_sha256=sha(checkpoint))
    metrics=[];mask=rows['role']!='train';y=rows['label'][mask];w=rows['weight'][mask]
    _,slot,draws=source_draws(rows['ancestor'][mask],spec['bootstrap_draws'],c['seed'])
    model.eval();micro=c['runtime']['small_motion_microbatch']
    for j,lead in enumerate(spec['leads_ps']):
        if time.time()>deadline-300:raise TimeoutError('Birth evaluation continuation required')
        codes=[];laws=[]
        for start in range(0,len(states),micro):
            if time.time()>deadline-300:raise TimeoutError('Birth inference continuation required')
            x=torch.from_numpy(np.array(states[start:start+micro,j],copy=True)).cuda()
            with torch.no_grad():
                z=model.encode(x,create_graph=False)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    mu,lv=model.prediction(model.path(z,[lead],particles=8,seed=c['seed']+111))
                mu,lv=mu.float(),lv.float();mean=mu.mean(1)[:,0]
                variance=(lv.exp()+mu.square()).mean(1)[:,0]-mean.square()
                law=torch.cat((mean,variance.clamp_min(1e-8).log()),-1)
            codes.append(z.cpu().numpy());laws.append(law.cpu().numpy())
        codes=np.concatenate(codes);law=np.concatenate(laws)
        for space,x in [('structural',codes[:,:128]),('structural_motion',codes),('future_law',law)]:
            if not np.isfinite(x).all():raise FloatingPointError('Nonfinite birth features')
            for kind in spec['readouts']:
                if time.time()>deadline-180:raise TimeoutError('Birth readout continuation required')
                fit(fit_config,producer,rows,x,model_spec,space,lead,kind,plan=plan)
                fp=out/'analyses/fits-v1'/name/space/f'lead-{lead:g}ps'/kind/'technical/predictions.npz'
                with np.load(fp) as saved:prob=saved['probability'][mask]
                values=scores(y,prob,w);boot,_=bootstrap_scores(y,prob,w,slot,draws,None)
                for metric in ('ap','auroc','nll','brier'):
                    lo,hi=np.quantile(boot[metric],[.025,.975])
                    metrics.append(dict(space=space,lead_ps=lead,readout=kind,metric=metric,value=values[metric],lower=float(lo),upper=float(hi),test_events=len(set(rows['event'][mask])),test_ancestors=len(held),test_rows=len(y)))
    write_metric_rows(metrics,out,family=FAMILY,name='birth-scores')
    write_json(out/'technical/complete.json',dict(identity=identity,metrics=metrics,neighborhood_filter=False,cohort=done))
    return metrics
