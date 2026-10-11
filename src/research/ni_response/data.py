"""Pinned native Ni Q,V; outcome-blind motion and distinct appearance/birth assays."""
from collections import Counter
from pathlib import Path
import hashlib
import numpy as np
import torch
from src.research.continuous_motion.data import Source, atomic_save, read, root, cache, digest, sha, write_json, resolve_path
from src.research.response_geometry.features import observed
from src.postprocessing.extraction import Frames
from src.postprocessing.io import arrays
from src.postprocessing.io import geometry
from src.postprocessing.descriptors.ptm import full_labels, producer_identity

FAMILY='ni_esen_response'

def load(c,name):
    path=cache(c)/(name+'.pt');done=read(cache(c)/(name+'.json'))
    if sha(path)!=done['sha256']:raise ValueError('Changed Ni cache '+name)
    return torch.load(path,map_location='cpu',weights_only=False)

def save(c,name,value):
    path=cache(c)/(name+'.pt');atomic_save(path,value)
    write_json(cache(c)/(name+'.json'),dict(sha256=sha(path)))

def prepare(c):
    if sha(resolve_path(c['checkpoint']))!=c['checkpoint_sha256']:raise ValueError('Wrong unrelaxed eSEN checkpoint')
    manifests=read(resolve_path(c['spatial_manifest']))
    if sha(resolve_path(c['spatial_manifest']))!=c['spatial_manifest_sha256']:raise ValueError('Changed spatial ancestry')
    spatial_train={r['task']['lineage'] for r in manifests['shards'] if r['task']['role']=='train'}
    sources=[];upstream=[];events={}
    for spec in c['catalogues']:
        folder=resolve_path(spec['path']);p=read(folder/'technical/plans.json')
        if sha(folder/'technical/plans.json')!=spec['sha256']:raise ValueError('Changed Ni catalogue')
        upstream.append(dict(path=str(folder),sha256=spec['sha256']))
        for s in p['parent']['sources']:
            if s['role']!='train' and s['ancestry_group'] in spatial_train:raise ValueError('Spatial training ancestry leaks into evaluation: '+s['id'])
            reader=Source(s);times=np.asarray(s['times_ps'])
            if not np.allclose(np.diff(times),.1,atol=1e-8,rtol=0):raise ValueError('Mixed/non-native cadence: '+s['id'])
            for part in s['parts']:
                a=arrays(part['path'],part['manifest_sha256'])
                if not np.allclose(np.diff(a['timesteps'])*part['timestep_fs']/1000,.1,atol=1e-8,rtol=0):raise ValueError('Actual Ni timestep cadence changed')
            if s['potential']!=['ni-mendelev2012-eam']:raise ValueError('Unexpected Ni generating potential')
            event_path=folder/'technical/sources'/s['id']/'010/ptm/events.json'
            ev=read(event_path);events[s['id']]=ev
            rng=np.random.default_rng(int(digest([c['seed'],s['id']])[:12],16))
            centers=rng.choice(reader.ids,c['centers_per_source'],replace=False).tolist()
            sources.append(dict(key=s['id'],source=s,centers=centers,search=str(folder),events_sha256=sha(event_path)))
    if len({s['key'] for s in sources})!=len(sources):raise ValueError('Duplicate sources')
    ancestors=sorted({s['source']['ancestry_group'] for s in sources if s['source']['role']=='train'},key=lambda x:digest([c['seed'],x]))
    selection=ancestors[:c['inner_selection_ancestors']];jobs=[]
    for item in sources:
        s=item['source'];item['original_role']=s['role'];item['role']=('selection' if s['ancestry_group'] in selection else 'train') if s['role']=='train' else 'test'
        times=np.asarray(s['times_ps']);eligible=np.flatnonzero((times>=c['hold_start_ps'])&(times<=times[-1]-max(c['horizons_ps'])))
        if len(eligible)<len(c['anchor_fractions']):raise ValueError('Insufficient complete Ni observation windows')
        for f in c['anchor_fractions']:
            frame=int(eligible[round(f*(len(eligible)-1))]);future=[frame+round(h/.1) for h in c['horizons_ps']]
            if not np.allclose(times[future]-times[frame],c['horizons_ps'],atol=1e-8,rtol=0):raise ValueError('Wrong saved future offsets')
            key=digest([s['id'],frame])[:20]
            jobs.append(dict(key=key,source_key=s['id'],frame=frame,future_frames=future,teacher_center=int(key[:8],16)%len(item['centers'])))
    partition=dict(inner_fit=ancestors[len(selection):],inner_selection=selection,final_train=ancestors,test=sorted({s['source']['ancestry_group'] for s in sources if s['source']['role']!='train'}))
    plan=dict(config=c,sources=sources,jobs=jobs,partition=partition,upstream=upstream,
        catalogue_events={str(n):sum(e['is_birth_record'] and e['origin']=='emergence_from_liquid' and f'{n}@1.5ps' in e['milestones'] for ev in events.values() for e in ev) for n in [32,64]})
    plan['identity']=digest(plan);path=root(c)/'technical/plan.json'
    if path.exists() and read(path)!=plan:raise ValueError('Changed Ni response plan; new revision required')
    write_json(path,plan);return plan

def observation_job(c,p,job):
    out=cache(c)/'observations'/job['key'];identity=digest([p['identity'],job])
    if (out/'complete.json').exists():
        done=read(out/'complete.json')
        if done['identity']!=identity or sha(out/'data.pt')!=done['sha256']:raise ValueError('Changed observations')
        return
    item=next(s for s in p['sources'] if s['key']==job['source_key']);s=item['source'];reader=Source(s)
    state,neighbors=reader.patches(job['frame'],item['centers']);state*=c['length_factor']
    future=np.stack([reader.patches(f,item['centers'])[0][...,:3]*c['length_factor'] for f in job['future_frames']],1)
    target=observed(torch.from_numpy(future.reshape(-1,80,3))).reshape(len(state),len(c['horizons_ps']),24).double()
    parent=read(Path(item['search'])/'technical/plans.json')['parent'];labels=Frames(parent,s,'ptm')
    provenance=Counter()
    def atomic_ptm(frame):
        labels.labels(frame)
        if labels.has_ptm:
            provenance['retained_atom_ptm']+=1
            return labels.raw_ptm(frame)
        # Older chunks retained components but discarded individual PTM atoms.
        # Explicitly compute the missing atomic target with the same cutoff.
        path=out/f'ptm-{frame:06d}.npy';receipt=path.with_suffix('.json')
        identity=digest([s['input_identity'],frame,parent['config']['ptm_rmsd_cutoff'],producer_identity()])
        if receipt.exists():
            record=read(receipt)
            if record['identity']!=identity or sha(path)!=record['sha256']:raise ValueError('Changed atomic PTM completion')
            value=np.load(path)
        else:
            out.mkdir(parents=True,exist_ok=True)
            value=full_labels(*geometry(s,frame),parent['config']['ptm_rmsd_cutoff'])
            np.save(path,value);write_json(receipt,dict(identity=identity,sha256=sha(path),producer='OVITO full-cell PTM',cutoff=parent['config']['ptm_rmsd_cutoff']))
        provenance['computed_missing_atom_ptm']+=1
        return value
    support=np.linalg.norm(state[...,:3],axis=-1)<c['encoder']['radius']
    clear=~np.any(np.isin(atomic_ptm(job['frame'])[neighbors],[1,2,3])&support,axis=1)
    first=np.full(len(state),np.inf)
    for step in range(1,round(max(c['appearance_horizons_ps'])/.1)+1):
        present=np.any(np.isin(atomic_ptm(job['frame']+step)[neighbors],[1,2,3])&support,axis=1)
        first[np.isinf(first)&present]=step*.1
    rows=[dict(id=digest([s['id'],job['frame'],center]),source=s['id'],ancestor=s['ancestry_group'],role=item['role'],original_role=s['role'],frame=job['frame'],time_ps=s['times_ps'][job['frame']],center=center) for center in item['centers']]
    atomic_save(out/'data.pt',dict(state=torch.from_numpy(state).float(),future=torch.from_numpy(future).float(),target=target,rows=rows,clear=clear,first=first))
    write_json(out/'complete.json',dict(identity=identity,sha256=sha(out/'data.pt'),atomic_ptm=dict(provenance)))

def seal(c,p):
    packets=[]
    for j in p['jobs']:
        f=cache(c)/'observations'/j['key'];done=read(f/'complete.json')
        if done['identity']!=digest([p['identity'],j]) or sha(f/'data.pt')!=done['sha256']:raise ValueError('Unsealed observations')
        packets.append(torch.load(f/'data.pt',map_location='cpu',weights_only=False))
    data={k:torch.cat([x[k] for x in packets]) for k in ['state','future','target']}
    data['rows']=[r for x in packets for r in x['rows']]
    if len({r['id'] for r in data['rows']})!=len(data['rows']):raise ValueError('Duplicate physical observations')
    geometries={};states=set()
    for state,r in zip(data['state'].numpy(),data['rows'],strict=True):
        qhash=hashlib.sha256(np.ascontiguousarray(state[:,:3]).tobytes()).hexdigest()
        xhash=hashlib.sha256(state.tobytes()).hexdigest()
        if qhash in geometries and geometries[qhash]!=r['ancestor']:raise ValueError('Identical present geometry across independent ancestors')
        if xhash in states:raise ValueError('Duplicate exact Q,V observation')
        geometries[qhash]=r['ancestor'];states.add(xhash)
    training=np.array([r['role']=='train' for r in data['rows']]);v=data['target'][training].flatten(0,1)
    data.update(mean=v.mean(0),scale=v.std(0).clamp_min(1e-4),clear=np.concatenate([x['clear'] for x in packets]),first=np.concatenate([x['first'] for x in packets]))
    save(c,'data',data);write_json(root(c)/'technical/data-complete.json',dict(rows=len(data['rows']),roles=dict(Counter(r['role'] for r in data['rows'])),clear_present=int(data['clear'].sum()),sha256=sha(cache(c)/'data.pt')))

def birth_source(c,p,item):
    out=cache(c)/'births'/item['key'];identity=digest([p['identity'],item['key'],'threshold-start'])
    if (out/'complete.json').exists():
        done=read(out/'complete.json')
        if done['identity']!=identity or sha(out/'data.pt')!=done['sha256']:raise ValueError('Changed birth rows')
        return
    s=item['source'];reader=Source(s);times=np.asarray(s['times_ps'])
    ep=Path(item['search'])/'technical/sources'/s['id']/'010/ptm/events.json'
    if sha(ep)!=item['events_sha256']:raise ValueError('Changed event ledger')
    ev=[e for e in read(ep) if e['is_birth_record'] and e['origin']=='emergence_from_liquid']
    state=[];rows=[];excluded=[]
    for threshold in [32,64]:
        chosen=[e for e in ev if f'{threshold}@1.5ps' in e['milestones']]
        reserved={int(a) for e in chosen for a in e['first_members']}
        controls=np.array([a for a in reader.ids if int(a) not in reserved])
        for e in chosen:
            event=f"{s['id']}:{e['track']}:{e['episode']}";rng=np.random.default_rng(int(digest([c['seed'],event,threshold])[:12],16))
            founders=rng.choice(e['first_members'],min(c['centers_per_event'],len(e['first_members'])),replace=False)
            negative=rng.choice(controls,len(founders)*4,replace=False)
            centers=np.r_[founders,negative].astype(int);onset=e['milestones'][f'{threshold}@1.5ps']['start_ps']
            for lead in c['birth_leads_ps']:
                f=int(np.argmin(abs(times-(onset-lead))))
                if abs(times[f]-(onset-lead))>1e-8:
                    excluded.append(dict(event=event,threshold=threshold,lead=lead,reason='exact lead unavailable'));continue
                qv,_=reader.patches(f,centers);qv*=c['length_factor']
                for i,center in enumerate(centers):
                    rows.append(dict(id=digest([s['id'],f,int(center)]),source=s['id'],ancestor=s['ancestry_group'],role=s['role'],event=event,pair=event,label=int(i<len(founders)),weight=1/len(centers),task=f'nucleus{threshold}',lead=float(lead),frame=f,center=int(center),catalogue_embryo_onset_ps=e['first_time_ps'],threshold_start_ps=onset,confirmation_ps=e['milestones'][f'{threshold}@1.5ps']['confirmation_ps']))
                    state.append(qv[i])
    atomic_save(out/'data.pt',dict(state=torch.from_numpy(np.stack(state)).float() if state else torch.empty(0,80,6),rows=rows,excluded=excluded))
    write_json(out/'complete.json',dict(identity=identity,sha256=sha(out/'data.pt')))

def seal_births(c,p):
    packets=[]
    for item in p['sources']:
        out=cache(c)/'births'/item['key'];done=read(out/'complete.json')
        if sha(out/'data.pt')!=done['sha256']:raise ValueError('Changed birth packet')
        packets.append(torch.load(out/'data.pt',weights_only=False,map_location='cpu'))
    rows=[r for x in packets for r in x['rows']];states=torch.cat([x['state'] for x in packets]);keep=[];seen={}
    for i,r in enumerate(rows):
        key=(r['task'],r['lead'],r['id'])
        if key in seen:
            old=rows[seen[key]]
            if old['label']!=r['label'] or old['ancestor']!=r['ancestor']:raise ValueError('Conflicting duplicate birth identity')
            continue
        seen[key]=i;keep.append(i)
    save(c,'births',dict(state=states[keep],rows=[rows[i] for i in keep],excluded=[e for x in packets for e in x['excluded']],duplicates_removed=len(rows)-len(keep)))
