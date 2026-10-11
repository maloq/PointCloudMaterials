"""Versioned dense-production cohort, exact leads and ancestry-preserving roles."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import json
import time
import numpy as np
from scipy.spatial import cKDTree
from src.project_runtime.paths import resolve_path
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.postprocessing.io import geometry
from src.postprocessing.io import arrays
from src.postprocessing.extraction import Frames
from src.research.liquid_predictability.descriptors import patch_descriptors
from src.experiment_runner.metric_docs import write_metric_rows

FAMILY = 'response_dense_birth'
ROLES = ['train', 'selection', 'calibration', 'test']

def read(path): return json.loads(Path(path).read_text())
def output(c): return resolve_path(c['output'])
def cache(c): return resolve_path(c['cache'])


def prepare(c):
    from .scope import eligibility, bind_reuse
    split = read(resolve_path(c['fixed_release'])/'splits.json')
    if split['identity'] != c['fixed_identity']: raise ValueError('Changed Al64 release')
    roles = {s['lineage']:s['role'] for s in split['sources']}
    sources, releases, excluded, seen, ancestors = [], [], [], set(), {}
    for release in c['releases']:
        root=resolve_path(release); p=read(root/'technical/plan.json')
        parent_path=resolve_path(p['config']['parent_output'])/'technical/plan.json'
        if sha(parent_path)!=p['parent_plan_sha256']: raise ValueError('Changed search population')
        parent=read(parent_path)
        releases.append(dict(root=str(root),identity=p['identity'],plan_sha256=sha(root/'technical/plan.json'),parent_sha256=sha(parent_path)))
        for s in parent['sources']:
            reason=eligibility(c,s)
            if reason:
                excluded.append(dict(release=str(root),source=s['id'],dataset=s['dataset'],reason=reason));continue
            if s['input_identity'] in seen: raise ValueError('Repeated dense source inputs')
            seen.add(s['input_identity'])
            if s['ancestry_group'] not in roles or roles[s['ancestry_group']] != s['role']:
                raise ValueError(f'Fixed ancestor role mismatch: {s["dataset"]}')
            old=ancestors.setdefault(s['ancestry_group'],s['role'])
            if old!=s['role']: raise ValueError('Ancestor crosses roles')
            times=np.asarray(s['times_ps'])[s['views']['010']]
            np.testing.assert_allclose(np.diff(times),.1,atol=1e-8,rtol=0)
            if c.get('population')!='all_010_with_saved_velocities' and s['views']['010']!=list(range(len(s['times_ps']))):raise ValueError('Non-native view')
            folder=root/'technical/sources'/s['id']/'010/ptm'
            task=next(t for t in p['jobs'] if t['source']==s['id'] and t['cadence']=='010' and t['detector']=='ptm')
            sources.append(dict(source=s,release=str(root),release_identity=p['identity'],label_folder=str(folder),
                label_identity=digest(dict(plan=p['identity'],task=task)),parent_path=str(parent_path),
                key=digest(dict(release=p['identity'],source=s['id']))[:20]))
    # Prefer complete native0.1 views when deduplicating matching observations.
    if c.get('population')=='all_010_with_saved_velocities':
        sources.sort(key=lambda x:(x['source']['origin']=='historical_partial',abs(x['source']['native_cadence_ps']-.1)>1e-8,-len(x['source']['views']['010']),x['key']))
    reuse_reference=bind_reuse(c,sources)
    campaign=resolve_path(c['campaign']); cfg=read(campaign/'technical/code/config.json')
    if read(campaign/'technical/binding.json')['identity']!=c['campaign_identity'] or cfg['fixed_identity']!=c['fixed_identity']:
        raise ValueError('Changed response campaign/ancestry')
    import torch
    parent_states=torch.load(campaign/'technical/parents.pt',map_location='cpu',weights_only=False)
    for s in parent_states:
        if roles[s['lineage']]!=s['role']:raise ValueError('Encoder source ancestry leak')
    models=[]
    for seed in cfg['fit_seeds']:
        for arm in cfg['arms']:
            name=arm['name']+'-'+str(seed);f=campaign/'analyses'/name/'technical'
            done=read(f/'complete.json')
            if sha(f/'best.pt')!=done['checkpoint_sha256']:raise ValueError('Checkpoint changed')
            models.append(dict(name=name,arm=arm['name'],seed=seed,velocities=arm['velocities'],checkpoint=str(f/'best.pt'),sha256=done['checkpoint_sha256']))
    names=['src/research/response_velocity/model.py','src/research/local_response/model.py',
        'src/research/supervised_onset/model.py','src/models/encoders/spatial_mace.py','src/models/encoders/mace_backend.py','config.json']
    from src.experiment_runner.metric_docs import check_metric_docs
    p=dict(config=c,sources=sources,releases=releases,excluded=excluded,models=models,reuse_reference=reuse_reference,
        dependencies={f:sha(campaign/'technical/code'/f) for f in names},
        implementations=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    p['identity']=digest(p);target=output(c)/'technical/plan.json'
    if target.exists() and read(target)!=p:raise ValueError('Changed dense-birth plan; use new revision')
    write_json(target,p);cache(c).mkdir(parents=True,exist_ok=True)
    receipt=read(campaign/'technical/data-complete.json')
    if sha(Path(receipt['cache']))!=receipt['sha256']:raise ValueError('Changed teacher bank')
    records=torch.load(receipt['cache'],map_location='cpu',weights_only=False)['records']
    records=[r for r in records if r['role']=='test'][:8]
    np.savez_compressed(cache(c)/'replay.npz',state=np.stack([torch.cat((r['position'],r['velocity']),-1).float().numpy() for r in records]))
    write_json(output(c)/'technical/prediction-context.json',dict(
        encoder=dict(positions='unrelaxed centered nearest80, 8A support, 5A edges; no halo',velocity='actual same-frame same-atom A/fs for QV; ignored by Q',history=1,conditions=[],precision='float32 inference from float16 archive',training_only_teacher='frozen FastAl native-V 100/500fs responses'),
        predictor=dict(inputs='embedding128 or existing nonlinear head256; separate rich442+current-component controls',history=1,conditions=[],leads_ps=c['leads_ps'],lead_as_input=False,selection='selection-source NLL',calibration='calibration-source affine logits'),
        labels='root emergence at8 atoms that later sustains64 for1.5ps; retrospective matched1:4 assay, not population hazard',
        exclusions='No existing8+ component within8A at any input lead; subcritical/single-atom crystal allowed and controlled diagnostically',
        scope=c.get('population','Three v3 releases, native010 complete main production only'),
        cadence='Exactly0.10ps labels/observations; native source cadence retained separately',
        missing_velocity='Explicitly excluded from the matched Q/QV comparison; no estimated/replacement velocities'))
    return p


class Access:
    def __init__(self, parent, source):
        self.s=source; self.labels=Frames(parent,source,'ptm')
        self.frame=lru_cache(maxsize=12)(self._frame)
        self.arrays=arrays(source['parts'][0]['path'],source['parts'][0]['manifest_sha256'])
        self.manifests=[];self.velocities=[]
        for part in source['parts']:
            m=read(Path(part['path'])/'manifest.json')
            if m['velocity_units']!='angstrom_per_ps':raise ValueError('Unexpected velocity units')
            spec=m['arrays']['velocities'];v=np.load(Path(part['path'])/spec['file'],mmap_mode='r')
            if list(v.shape)!=spec['shape'] or str(v.dtype)!=spec['dtype']:raise ValueError('Velocity schema mismatch')
            np.testing.assert_array_equal(arrays(part['path'],part['manifest_sha256'])['atom_ids'],self.arrays['atom_ids'])
            self.manifests.append(m);self.velocities.append(v)
    def _frame(self, f):
        q,b=geometry(self.s,f); lab=self.labels.labels(f).copy(); counts=np.bincount(lab)
        solid=(lab>0)&(counts[lab]>=8);ids=np.flatnonzero(solid)
        return q,b,lab,cKDTree(q[ids],boxsize=b) if len(ids) else None
    def clear(self,f,atoms):
        q,_,_,tree=self.frame(int(f))
        return np.ones(len(atoms),bool) if tree is None else tree.query(q[atoms],workers=1)[0]>8.
    def patch(self,f,atoms):
        q,b,lab,_=self.frame(f); _,neighbors=cKDTree(q,boxsize=b).query(q[atoms],k=80,workers=1)
        if not np.array_equal(neighbors[:,0],atoms):raise ValueError('Nearest-neighbor identity mismatch')
        delta=q[neighbors]-q[atoms,None];delta-=b*np.rint(delta/b)
        pi,raw_frame=self.s['frames'][f]
        velocity=np.asarray(self.velocities[pi][raw_frame,neighbors],np.float32)/np.float32(1000.)
        counts=np.bincount(lab)
        visible=np.stack(((lab[neighbors]>0).mean(1),counts[lab[neighbors]].max(1)),1)
        # Background label zero is liquid, not a giant connected crystal.
        counts[0]=0;visible[:,1]=counts[lab[neighbors]].max(1)
        return delta.astype(np.float32),velocity,visible.astype(np.float32)


def source(c,p,item):
    folder=cache(c)/'sources'/item['key'];folder.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(plan=p['identity'],source=item['key']))
    if (folder/'complete.json').exists():
        done=read(folder/'complete.json')
        if done['identity']!=identity or any(sha(folder/f)!=h for f,h in done['files'].items()):raise ValueError('Prepared source changed')
        return done
    labels=Path(item['label_folder']);receipt=read(labels/'complete.json')
    if receipt['identity']!=item['label_identity'] or sha(labels/'events.json')!=receipt['files']['events.json']:raise ValueError('Unsealed/changed lineage labels')
    events=[e for e in read(labels/'events.json') if e['record_kind']=='root' and e['origin']=='emergence_from_liquid' and e['establishment_confirmation_ps'] is not None]
    parent=read(item['parent_path']);s=item['source'];access=Access(parent,s)
    if item.get('reuse'):
        from .scope import reuse
        return reuse(c,p,item,folder,access,events)
    rng=np.random.default_rng(int(digest([c['seed'],item['key']])[:16],16))
    leads=np.asarray(c['leads_ps']);steps=np.rint(leads/.1).astype(int)
    np.testing.assert_allclose(steps*.1,leads,atol=1e-8,rtol=0)
    rows=[];keys=[];coverage=[];used=set();atom_ids=access.arrays['atom_ids']
    launch=output(c)/'technical/launch.json'
    deadline=read(launch)['deadline_unix'] if launch.exists() else float('inf')
    for event in events:
        if time.time()>deadline-300:raise TimeoutError('Source preparation deadline; completed sources retained')
        first=event['first_frame'];view=np.asarray(s['views']['010']);slot=int(np.searchsorted(view,first))
        if slot>=len(view) or view[slot]!=first:raise ValueError('Event is outside its exact0.1ps view')
        follow_slot=slot+int(round(c['control_followup_ps']/.1))
        event_id=item['key']+':'+str(event['track']);reasons=Counter();accepted=0
        if slot-steps.max()<0 or follow_slot>=len(view):
            coverage.append(dict(source=s['id'],event=event_id,accepted_pairs=0,reason='insufficient_input_or_control_followup'));continue
        frames=view[slot-steps]
        np.testing.assert_allclose(np.asarray(s['times_ps'])[first]-np.asarray(s['times_ps'])[frames],leads,atol=1e-8,rtol=0)
        members=np.searchsorted(atom_ids,event['first_members']);np.testing.assert_array_equal(atom_ids[members],event['first_members'])
        candidates=rng.permutation(members)[:c['centers_per_event']]
        candidates=candidates[np.logical_and.reduce([access.clear(f,candidates) for f in frames])]
        negatives=rng.choice(s['atom_count'],min(c['negative_candidates'],s['atom_count']),replace=False)
        q,b,_,_=access.frame(first);d=q[negatives]-np.asarray(event['centroid_A']);d-=b*np.rint(d/b)
        negatives=negatives[np.linalg.norm(d,axis=1)>2*c['radius_A']]
        for f in view[slot-steps.max():follow_slot+1]:
            negatives=negatives[access.clear(f,negatives)]
            if len(negatives)<4:break
        cursor=0
        for center in candidates:
            if (first,int(center)) in used:reasons['duplicate_input']+=1;continue
            available=[int(a) for a in negatives[cursor:] if (first,int(a)) not in used]
            if len(available)<4:reasons['insufficient_matched_controls']+=1;break
            chosen=available[:4];cursor=int(np.flatnonzero(negatives==chosen[-1])[0])+1
            pair=event_id+':'+str(int(atom_ids[center]));centers=[int(center),*chosen]
            for slot,atom in enumerate(centers):
                idx=[]
                for f in frames:idx.append(len(keys));keys.append((int(f),atom))
                rows.append(dict(id=pair+':'+str(slot),pair=pair,event=event_id,source=s['id'],ancestor=s['ancestry_group'],
                    role=s['role'],label=int(slot==0),indices=idx,atom=int(atom_ids[atom]),first_frame=first,
                    first_time_ps=event['first_time_ps'],establishment_start_ps=event['establishment_start_ps'],
                    establishment_confirmation_ps=event['establishment_confirmation_ps'],release=item['release']))
                used.add((first,atom))
            accepted+=1
        coverage.append(dict(source=s['id'],event=event_id,accepted_pairs=accepted,reason=json.dumps(dict(reasons)),eligible_centers=len(candidates)))
        write_json(folder/'progress.json',dict(stage='select',events_done=len(coverage),events=len(events),rows=len(rows)))
    unique=sorted(set(keys));lookup={key:i for i,key in enumerate(unique)}
    for row in rows:row['indices']=[lookup[keys[i]] for i in row['indices']]
    positions=np.empty((len(unique),80,3),np.float32);velocities=np.empty_like(positions);desc=np.empty((len(unique),444),np.float32)
    columns=None
    for f in sorted({k[0] for k in unique}):
        if time.time()>deadline-300:raise TimeoutError('Descriptor preparation deadline')
        inds=[i for i,k in enumerate(unique) if k[0]==f];atoms=np.array([unique[i][1] for i in inds])
        q,v,visible=access.patch(f,atoms);positions[inds]=q;velocities[inds]=v
        for i,cloud,extra in zip(inds,q,visible):
            values,names=patch_descriptors(cloud)
            if len(values)!=442:raise ValueError('Descriptor schema changed')
            if columns is not None and names!=columns:raise ValueError('Descriptor columns changed')
            columns=names;desc[i]=np.r_[values,extra]
        write_json(folder/'progress.json',dict(stage='patches',observations_done=max(inds)+1,total=len(unique)))
    for name,data in [('positions',positions),('velocities',velocities),('descriptors',desc)]:
        if not np.isfinite(data).all():raise FloatingPointError('Nonfinite native inputs')
        np.save(folder/(name+'.npy'),data)
    write_json(folder/'rows.json',rows);write_json(folder/'coverage.json',coverage)
    done=dict(identity=identity,source=s['id'],role=s['role'],ancestor=s['ancestry_group'],candidate_events=len(events),
        accepted_events=sum(e['accepted_pairs']>0 for e in coverage),rows=len(rows),observations=len(unique),
        labels_sha256=receipt['files']['events.json'],velocity_manifest=[m['arrays']['velocities'] for m in access.manifests],
        velocity_provenance='native same-frame same-atom array; producer manifest SHA bound; extracted bytes hashed below',
        files={n:sha(folder/n) for n in ['positions.npy','velocities.npy','descriptors.npy','rows.json','coverage.json']})
    from .scope import canonicalize
    done['event_members']={item['key']+':'+str(e['track']):e['first_members'] for e in events}
    return canonicalize(c,p,item,folder,access,rows,coverage,done)


def seal(c,p,*,metric_family=FAMILY):
    for release in p['releases']:
        done=read(Path(release['root'])/'technical/complete.json')
        if done['identity']!=release['identity']:raise ValueError('Newest lineage release not complete')
    rows=[];clouds=[];velocities=[];descriptors=[];coverage=[];counts=[];offset=0
    for item in p['sources']:
        folder=cache(c)/'sources'/item['key'];done=read(folder/'complete.json')
        if done['identity']!=digest(dict(plan=p['identity'],source=item['key'])) or any(sha(folder/f)!=h for f,h in done['files'].items()):raise ValueError('Unsealed dense source')
        for r in read(folder/'rows.json'):rows.append(dict(r,indices=[i+offset for i in r['indices']]))
        q=np.load(folder/'positions.npy');clouds.append(q);offset+=len(q)
        velocities.append(np.load(folder/'velocities.npy'));descriptors.append(np.load(folder/'descriptors.npy'))
        coverage.extend(read(folder/'coverage.json'));counts.append({k:done[k] for k in ['source','role','ancestor','candidate_events','accepted_events','rows','observations']})
    if not rows:raise ValueError('No eligible dense births')
    from .scope import deduplicate
    rows,aliases=deduplicate(rows)
    out=output(c)/'analyses/coverage-v1'
    write_metric_rows(counts,out,family=metric_family,name='sources')
    write_metric_rows(coverage,out,family=metric_family,name='eligibility',columns=tuple(dict.fromkeys(k for r in coverage for k in r)))
    write_metric_rows(aliases,out,family=metric_family,name='duplicate-pairs',columns=['event','atom','dropped_pair','retained_pair','dropped_view','reason'])
    write_metric_rows(p['excluded'],out,family=metric_family,name='exclusions',columns=['release','source','dataset','reason'])
    data={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    event_counts=Counter(data['event']);data['weight']=np.array([1/event_counts[e] for e in data['event']])
    if len(set(data['id']))!=len(rows):raise ValueError('Duplicate dense row identity')
    for role in ROLES:
        ids=data['role']==role
        if len(set(data['event'][ids]))<c['minimum_events'][role]:raise ValueError(f'Insufficient {role} events; do not resplit or relax labels')
    for name,values in [('positions',clouds),('velocities',velocities),('descriptors',descriptors)]:np.save(cache(c)/(name+'.npy'),np.concatenate(values))
    np.savez_compressed(cache(c)/'rows.npz',**data)
    files={f:sha(cache(c)/f) for f in ['positions.npy','velocities.npy','descriptors.npy','rows.npz']}
    write_json(cache(c)/'complete.json',dict(identity=p['identity'],rows=len(rows),observations=offset,files=files,
        physical_events=len(set(data['event'])),event_views=len(set(data['event_view'])),dropped_duplicate_pairs=len(aliases),
        trajectory_views=len(p['sources']),melt_ancestors=len(set(data['ancestor']))))
    return data


def dataset(c):
    done=read(cache(c)/'complete.json');p=read(output(c)/'technical/plan.json')
    if done['identity']!=p['identity']:raise ValueError('Dense cohort identity changed')
    for f,h in done['files'].items():
        if sha(cache(c)/f)!=h:raise ValueError('Changed sealed cohort '+f)
    with np.load(cache(c)/'rows.npz') as z:return {k:z[k] for k in z.files}
