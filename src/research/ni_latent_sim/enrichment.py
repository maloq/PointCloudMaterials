"""Fixed-atom liquid persistence and advancing-front sequence enrichment."""
from collections import Counter, defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
from src.postprocessing.io import geometry, arrays
from src.postprocessing.extraction import Frames
from src.postprocessing.descriptors.ptm import full_labels, producer_identity
from .data import (read, root, cache, sha, digest, write_json, atomic_save,
                   resolve_path, Source, Motion, observation, parent, load)


def base(c):
    spec=c['enrichment'];folder=resolve_path(spec['base_output'])
    path=folder/'technical/plan.json';receipt=resolve_path(spec['base_cache'])/'data.json'
    if sha(path)!=spec['plan_sha256'] or sha(receipt)!=spec['receipt_sha256']:
        raise ValueError('Changed base sequence plan or data receipt')
    p=read(path);sealed=read(receipt)
    if p['identity']!=sealed['identity']:raise ValueError('Base sequence identity mismatch')
    return p


def spaced(values,count):
    return np.asarray(values)[np.unique(np.linspace(0,len(values)-1,min(count,len(values)),dtype=int))].astype(int).tolist()


def prepare(c):
    old=base(c);spec=c['enrichment'];sources=[]
    for item in old['sources']:
        s=item['source'];times=np.asarray(s['times_ps'])
        for part in s['parts']:
            steps=arrays(part['path'],part['manifest_sha256'])['timesteps']
            if not np.allclose(np.diff(steps)*part['timestep_fs']/1000,.1,rtol=0,atol=1e-8):raise ValueError('Changed native cadence: '+s['id'])
        path=Path(item['search'])/'technical/sources'/s['id']/'010/ptm/trajectory.json'
        history=read(path)
        if [r['frame'] for r in history]!=list(range(len(times))) or not np.allclose([r['time_ps'] for r in history],times,atol=1e-8,rtol=0):
            raise ValueError('PTM history and actual source timeline disagree: '+s['id'])
        count=np.array([r['tracked_atoms'] for r in history]);largest=np.array([r['largest_cluster'] for r in history])
        eligible=np.arange(200,len(times)-60,10)
        liquid=[f for f in eligible if count[f:f+61].max()<=spec['screen_liquid_max_atoms']]
        front=eligible[(count[eligible]>=spec['screen_front_min_atoms'])&(count[eligible]<.85*s['atom_count'])
            &(count[eligible+60]-count[eligible]>=spec['screen_growth_min_atoms'])&(largest[eligible]>=128)]
        if not liquid:raise ValueError('No liquid candidate windows: '+s['id'])
        frames=dict(liquid_liquid=spaced(liquid,spec['liquid_anchors']),growing_front=spaced(front,spec['front_anchors']))
        for f in sum(frames.values(),[]):
            if not np.allclose(times[np.arange(f-3,f+61)]-times[f],np.arange(-3,61)*.1,rtol=0,atol=1e-8):raise ValueError('Incomplete exact window')
        sources.append(dict(item,candidate_frames=frames,trajectory_table_sha256=sha(path)))
    p=dict(config=c,partition=old['partition'],sources=sources,jobs=old['jobs'],parent_identity=old['parent_identity'],
           base_identity=old['identity'],target_rows=14400,added_per_regime=4800)
    p['identity']=digest(p);path=root(c)/'technical/plan.json'
    if path.exists() and read(path)!=p:raise ValueError('Changed enrichment; use a new revision')
    write_json(path,p)
    context=read(resolve_path(spec['base_output'])/'technical/prediction-context.json')
    context.update(sequence_identity='one fixed central atom ID across all64 observations; nearest80 membership can change; no front-following recentering',
        sampling='4800 original +4800 liquid-persistent +4800 advancing-front local sequences; phase/future-selected additions, not natural occupancy',
        labels='PTM only selects and audits added motion rows; no PTM input/loss and unchanged birth probe rows',
        teachers='unchanged300 short physical-teacher groups; no extra MD; new long sequences from existing NPT histories',
        normalization='frozen original training-only mean/scale; common score units across releases')
    write_json(root(c)/'technical/prediction-context.json',context)
    return p


class Labels:
    def __init__(self,c,p,item):
        self.c,self.item=c,item;s=item['source']
        self.parent=read(Path(item['search'])/'technical/plans.json')['parent'];self.reader=Frames(self.parent,s,'ptm')
        self.cutoff=self.parent['config']['ptm_rmsd_cutoff'];self.used={};self.prior={}
        if self.cutoff!=.1:raise ValueError('Enrichment requires the declared PTM cutoff0.1')
        pc=parent(c)['config']
        for j in p['jobs']:
            if j['source_key']==item['key']:
                for f in (cache(pc)/'observations'/j['key']).glob('ptm-*.npy'):
                    self.prior[int(f.stem.split('-')[1])]=f

    def get(self,frame):
        self.reader.labels(frame)
        if self.reader.has_ptm:
            value=self.reader.raw_ptm(frame)
            self.used[str(self.reader.path)]=self.reader.checksum
            return value
        identity=digest([self.item['source']['input_identity'],frame,self.cutoff,producer_identity()])
        path=self.prior.get(frame,cache(self.c)/'atomic-ptm'/self.item['key']/f'ptm-{frame:06d}.npy')
        receipt=path.with_suffix('.json')
        if frame in self.prior and not receipt.exists():raise ValueError('Historical atomic PTM is missing its receipt: '+str(path))
        if receipt.exists():
            record=read(receipt)
            if record['identity']!=identity or record['sha256']!=sha(path):raise ValueError('Changed atomic PTM cache')
        else:
            path.parent.mkdir(parents=True,exist_ok=True)
            value=full_labels(*geometry(self.item['source'],frame),self.cutoff);np.save(path,value)
            write_json(receipt,dict(identity=identity,sha256=sha(path),producer='OVITO full-cell PTM',cutoff=self.cutoff))
        self.used[str(path)]=read(receipt)['sha256']
        return np.load(path,allow_pickle=False)


def phase(item,frame,labels,factor,centers=None):
    q,box=geometry(item['source'],frame);tree=cKDTree(q,boxsize=box)
    if centers is None:centers=np.arange(len(q))
    distance,near=tree.query(q[centers],k=80,workers=1);support=distance*factor<8.
    crystal=np.isin(labels,[1,2,3]);fraction=(crystal[near]&support).sum(1)/support.sum(1)
    return q,box,crystal,fraction


def scan_source(c,p,item,deadline):
    folder=cache(c)/'candidates'/item['key'];done=folder/'complete.json'
    identity=digest([p['identity'],item['key']])
    if done.exists():
        receipt=read(done)
        if receipt['identity']!=identity or sha(folder/'rows.pt')!=receipt['sha256']:raise ValueError('Changed enrichment candidates')
        return receipt
    labels=Labels(c,p,item);s=item['source'];reader=Source(s);factor=parent(c)['config']['length_factor']
    old=base(c);used={(j['frame'],int(a)) for j in old['jobs'] if j['source_key']==item['key'] for a in j['centers']}
    selected=[];audits=[]
    for regime,frames in item['candidate_frames'].items():
        for frame in frames:
            if time.time()>deadline-600:raise TimeoutError('Enrichment classification reached deadline')
            rng=np.random.default_rng(int(digest([c['seed'],item['key'],frame,regime])[:12],16))
            initial=labels.get(frame);q,box,crystal,frac=phase(item,frame,initial,factor)
            if regime=='liquid_liquid':mask=(frac<=.05)&~crystal
            else:
                core=crystal&(frac>=.8);liquid=frac<=.05
                dc=np.full(len(q),np.inf);dl=dc.copy()
                if core.any():dc=cKDTree(q[core],boxsize=box).query(q,k=1,workers=1)[0]*factor
                if liquid.any():dl=cKDTree(q[liquid],boxsize=box).query(q,k=1,workers=1)[0]*factor
                mask=(frac>=.2)&(frac<.8)&~crystal&(dc<=8)&(dl<=8)
            candidates=np.flatnonzero(mask)
            candidates=np.array([a for a in candidates if (frame,int(reader.ids[a])) not in used],dtype=np.int64)
            limit=c['enrichment']['liquid_candidates_per_anchor' if regime=='liquid_liquid' else 'front_candidates_per_anchor']
            indices=rng.choice(candidates,min(len(candidates),limit),replace=False)
            initial_fraction=frac[indices];keep=np.ones(len(indices),bool);max_fraction=initial_fraction.copy()
            final_fraction=initial_fraction.copy();persistent=np.zeros(len(indices))
            if len(indices):
                if regime=='liquid_liquid':
                    for step in range(1,61):
                        types=labels.get(frame+step);_,_,cr,fr=phase(item,frame+step,types,factor,indices)
                        max_fraction=np.maximum(max_fraction,fr);keep&=(fr<=.05)&~cr[indices]
                        if step==60:final_fraction=fr
                else:
                    final=labels.get(frame+60);_,_,_,final_fraction=phase(item,frame+60,final,factor,indices)
                    for step in range(50,61):persistent+=np.isin(labels.get(frame+step)[indices],[1,2,3])/11
                    keep=(final_fraction-initial_fraction>=.2)&(persistent>=.8)&np.isin(final[indices],[1,2,3])
            for i in np.flatnonzero(keep):
                center=int(reader.ids[indices[i]])
                selected.append(dict(id=digest([item['key'],frame,center]),source=item['key'],ancestor=s['ancestry_group'],
                    role=item['role'],original_role=item['original_role'],frame=int(frame),center=center,regime=regime,
                    initial_fraction=float(initial_fraction[i]),final_fraction=float(final_fraction[i]),
                    maximum_liquid_fraction=float(max_fraction[i]) if regime=='liquid_liquid' else None,
                    final_center_persistence=float(persistent[i]) if regime=='growing_front' else None))
            audits.append(dict(regime=regime,frame=frame,proposed=len(indices),qualified=int(keep.sum())))
            write_json(folder/'progress.json',dict(last=audits[-1],windows_done=len(audits),qualified=len(selected)))
    atomic_save(folder/'rows.pt',selected)
    receipt=dict(identity=identity,sha256=sha(folder/'rows.pt'),counts=dict(Counter(r['regime'] for r in selected)),windows=audits,ptm_inputs=labels.used)
    write_json(done,receipt);return receipt


def scan(c,p,lane,deadline):
    for i,item in enumerate(p['sources'][lane::2]):
        result=scan_source(c,p,item,deadline)
        write_json(root(c)/'technical'/f'worker-{lane}.json',dict(stage='classifying_fixed_atom_sequences',source=item['key'],sources_done=i+1,counts=result['counts']))
        print('enrichment scan',item['key'],result['counts'],flush=True)
    write_json(root(c)/'technical'/f'scan-complete-{lane}.json',dict(state='complete',identity=p['identity']))


def interleave(groups):
    result=[];keys=sorted(groups)
    for i in range(max(len(v) for v in groups.values())):
        result.extend(groups[k][i] for k in keys if i<len(groups[k]))
    return result


def select(c,p):
    path=root(c)/'technical/selected-sequences.json'
    if path.exists():
        selected=read(path)
        if selected['plan_identity']!=p['identity'] or selected['identity']!=digest({k:v for k,v in selected.items() if k!='identity'}):raise ValueError('Changed enrichment selection')
        return selected
    candidates=[];receipts={}
    for item in p['sources']:
        folder=cache(c)/'candidates'/item['key'];done=read(folder/'complete.json')
        if done['identity']!=digest([p['identity'],item['key']]) or sha(folder/'rows.pt')!=done['sha256']:raise ValueError('Unsealed candidate pool')
        candidates.extend(torch.load(folder/'rows.pt',weights_only=False,map_location='cpu'));receipts[item['key']]=sha(folder/'complete.json')
    rows=[];population=[]
    for regime in ['liquid_liquid','growing_front']:
        for role,number in [('train',1280),('selection',320),('test',3200)]:
            groups=defaultdict(lambda:defaultdict(list))
            for r in candidates:
                if r['regime']==regime and r['role']==role:groups[r['source']][r['frame']].append(r)
            queues={}
            for source,frames in groups.items():
                for frame,values in frames.items():values.sort(key=lambda r:digest([c['seed'],r['id']]))
                queues[source]=interleave(frames)
            if not queues:raise ValueError(f'No eligible {role}/{regime} sources; splits will not be changed')
            available=interleave(queues)
            if len(available)<number:raise ValueError(f'Need {number} {role}/{regime} rows, found {len(available)}; revise sampling explicitly')
            chosen=available[:number];rows.extend(chosen)
            population.append(dict(role=role,regime=regime,rows=len(chosen),sources=len({r['source'] for r in chosen}),candidates=len(available)))
    if len(rows)!=9600 or len({r['id'] for r in rows})!=9600:raise ValueError('Enrichment has duplicate or missing observations')
    result=dict(plan_identity=p['identity'],rows=rows,population=population,candidate_receipts=receipts)
    result['identity']=digest(result);write_json(path,result);return result


def encode(c,p,lane,deadline):
    chosen=read(root(c)/'technical/selected-sequences.json');model=Motion(parent(c)['config']).cuda().eval().requires_grad_(False)
    factor=parent(c)['config']['length_factor']
    for item in p['sources'][lane::2]:
        groups=defaultdict(list)
        for r in chosen['rows']:
            if r['source']==item['key']:groups[r['frame']].append(r)
        reader=Source(item['source'])
        for frame,rows in sorted(groups.items()):
            key=digest([item['key'],frame])[:20];folder=cache(c)/'additions'/key
            identity=digest([chosen['identity'],rows])
            if (folder/'complete.json').exists():
                done=read(folder/'complete.json')
                if done['identity']!=identity or sha(folder/'data.pt')!=done['sha256']:raise ValueError('Changed encoded sequence packet')
                continue
            if time.time()>deadline-600:raise TimeoutError('Enriched encoding reached deadline')
            centers=[r['center'] for r in rows]
            # This same immutable list of central atom IDs is used at EVERY frame.
            states=np.stack([reader.patches(f,centers)[0] for f in range(frame-3,frame+61)],1)
            x=observation(model,torch.from_numpy(states*factor).float().flatten(0,1),c['encoder_microbatch']).reshape(len(rows),64,280)
            if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite enriched eSEN features')
            atomic_save(folder/'data.pt',dict(x=x,rows=rows));write_json(folder/'complete.json',dict(identity=identity,sha256=sha(folder/'data.pt')))
            write_json(root(c)/'technical'/f'worker-{lane}.json',dict(stage='encoding_fixed_atom_sequences',source=item['key'],frame=frame,rows=len(rows)))
            print('enrichment encoded',item['key'],frame,len(rows),flush=True)
    write_json(root(c)/'technical'/f'data-complete-{lane}.json',dict(state='complete',identity=p['identity']))


def seal(c,p):
    if (cache(c)/'data.json').exists():
        if read(cache(c)/'data.json')['identity']!=p['identity']:raise ValueError('Changed enriched sealed identity')
        return
    old=base(c);data=load(old['config'],'data');chosen=read(root(c)/'technical/selected-sequences.json')
    rows=[dict(r,regime='original') for r in data['rows']];parts=[data['x']]
    groups=defaultdict(list)
    for r in chosen['rows']:groups[(r['source'],r['frame'])].append(r)
    for (source,frame),rr in sorted(groups.items()):
        folder=cache(c)/'additions'/digest([source,frame])[:20];done=read(folder/'complete.json')
        if done['identity']!=digest([chosen['identity'],rr]) or sha(folder/'data.pt')!=done['sha256']:raise ValueError('Unsealed added sequence')
        packet=torch.load(folder/'data.pt',map_location='cpu',weights_only=False);parts.append(packet['x']);rows.extend(packet['rows'])
    if len(rows)!=14400 or len({r['id'] for r in rows})!=14400:raise ValueError('Expanded row identity/size mismatch')
    expected={'train':3840,'selection':960,'test':9600}
    if dict(Counter(r['role'] for r in rows))!=expected:raise ValueError('Expanded roles differ from declared quotas')
    roles=defaultdict(set)
    for r in rows:roles[r['role']].add(r['ancestor'])
    for a in roles:
        for b in roles:
            if a!=b and roles[a]&roles[b]:raise ValueError('Expanded ancestry leakage')
    atomic_save(cache(c)/'data.pt',dict(x=torch.cat(parts),rows=rows,mean=data['mean'],scale=data['scale']))
    for name in ['births','short']:
        # Reuse immutable banks through explicit links; no duplicate physical rows.
        load(old['config'],name)
        path=cache(c)/(name+'.pt');target=cache(old['config'])/(name+'.pt')
        if not path.exists():path.symlink_to(target)
        if sha(path)!=sha(target):raise ValueError('Changed shared '+name+' bank')
    write_json(cache(c)/'data.json',dict(identity=p['identity'],files={n:sha(cache(c)/(n+'.pt')) for n in ['data','births','short']},
        rows=len(rows),roles=expected,frames=64,spacing_ps=.1,normalization='frozen original train',population=chosen['population']))
    write_json(root(c)/'technical/data-complete.json',dict(state='complete',rows=14400,roles=expected,population=chosen['population'],
        original_rows_preserved=4800,added_rows=9600,source_ancestors=60,sequence_identity='fixed central atom ID; changing nearest80 neighbors',
        data_receipt_sha256=sha(cache(c)/'data.json')))
