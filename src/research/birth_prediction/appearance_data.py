"""Crystal-free histories for verified failed embryos, with original source roles."""
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import sys
import time

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin import ancestry
from .data import SourceFrames, read, plan, load
from .relaxed_data import cell, checked, INPUT


def original(c):
    return read(resolve_path(c['birth_config']))


def cache_config(c, domain):
    b = original(c)
    b.update(cache=c[domain+'_cache'], output=str(resolve_path(c['output'])/'technical'/domain),
             prepare_tasks=c['prepare_workers'], encoders=[])
    if domain == 'relaxed':
        b['input_domain'] = dict(INPUT, labels='Original-MD crystal appearance, established or verified transient; matched liquid controls')
    return b


def candidate_records(c):
    root = resolve_path(c['fate_output'])/'technical'
    done = read(root/'complete.json')
    if done['identity'] != c['fate_identity'] or done['sources'] != 150:
        raise ValueError('Require the complete frozen fate audit')
    records = []
    for item in read(root/'prepared.json')['sources']:
        folder = root/'sources'/str(item['id'])
        receipt = read(folder/'complete.json')
        if sha(folder/'transient-episodes.json') != receipt['files']['transient-episodes.json']:
            raise ValueError('Changed transient catalogue')
        records.extend(r for r in read(folder/'transient-episodes.json') if
            r['terminal_status']=='dissolved' and r['origin']=='isolated' and r['root_count']==1
            and r['strong_lineage'] and r['peak_atoms'] >= 8)
    if len(records) != 184 or sum(r['primary_failed_candidate'] for r in records) != 28:
        raise ValueError('Requested nested 184/28 candidate pools changed')
    return records


def prepare(c):
    b = original(c); p = plan(b)
    records = candidate_records(c)
    root = resolve_path(c['output'])/'technical'
    value = dict(config=c, original_plan=p, candidates=records,
                 implementation=sha(Path(__file__)))
    value['identity'] = digest(value)
    path = root/'data-plan.json'
    if path.exists() and read(path) != value:
        raise ValueError('Appearance data contract changed; use a new output')
    write_json(path,value)
    return value


def prepare_source(c, p, item):
    b = original(c); source = item['id']
    folder = resolve_path(c['original_cache'])/'sources'/str(source)
    folder.mkdir(parents=True,exist_ok=True)
    identity = digest(dict(plan=p['identity'],source=source))
    if (folder/'complete.json').exists():
        saved = read(folder/'complete.json')
        if saved['identity'] != identity or any(sha(folder/k)!=v for k,v in saved['files'].items()):
            raise ValueError('Changed prepared transient source')
        return saved
    begun=time.monotonic(); access=SourceFrames(p['original_plan'],item)
    old=resolve_path(b['cache'])/'sources'/str(source)
    with np.load(old/'rows.npz') as a:
        used=set(zip(a['atom'].tolist(),a['end_frame'].tolist()))
    with np.load(access.root/'graph.npz') as g:
        starts=g['start'].copy(); sizes=g['size'].copy()
    rng=np.random.default_rng(np.random.SeedSequence([c['seed'],source,2]))
    sequences=[]; metadata=[]; coverage=[]
    history=b['history_frames']; radius=b['radius_A']
    for event in [e for e in p['candidates'] if e['source']==source]:
        first=event['first_frame']; xyz,box,labels,_,_=access.frame(first)
        local,_=ancestry.components(xyz,box,labels,p['original_plan']['audit']['lineage'])
        members=np.flatnonzero(np.where(local>0,local+starts[first],0)==event['first_node'])
        if len(members)!=sizes[event['first_node']]:raise ValueError('Changed transient first-frame membership')
        delta=xyz-np.asarray(event['centroid_A']);delta-=box*np.rint(delta/box)
        centers=rng.permutation(np.flatnonzero(np.linalg.norm(delta,axis=1)<=radius))
        accepted=0; reasons=Counter()
        for center in centers[:b['maximum_centers_scanned_per_event']]:
            if accepted==b['centers_per_event']:break
            begin=max(0,first-b['appearance_search_frames']);appearance=None
            for frame in range(begin,first+1):
                if not access.clear(frame,np.array([center]),radius)[0]:appearance=frame;break
            if appearance is None:reasons['no_appearance']+=1;continue
            if appearance==begin or appearance<history:reasons['left_censored_history']+=1;continue
            px,_,_,solid,tree=access.frame(appearance)
            near=solid[np.asarray(tree.query_ball_point(px[center],radius),int)]
            if not np.intersect1d(near,members).size:reasons['appearance_not_episode_core']+=1;continue
            end=appearance-1
            atom=int(access.raw.atom_ids[center])
            if (atom,end) in used:reasons['duplicate_existing_history']+=1;continue
            frames=np.arange(end-history+1,end+1,dtype=int)
            if not all(access.clear(f,np.array([center]),radius)[0] for f in frames):
                reasons['crystal_in_input']+=1;continue
            follow=max(event['dissolution_frame'],end+b['minimum_followup_frames'])
            if follow>=item['frame_count']:reasons['incomplete_followup']+=1;continue
            controls=[]
            draw=rng.choice(item['atom_count'],min(item['atom_count'],b['negative_candidates']),replace=False)
            for block in np.array_split(draw,max(1,len(draw)//128)):
                keep=np.ones(len(block),bool)
                for f in range(int(frames[0]),follow+1):
                    active=np.flatnonzero(keep)
                    if not len(active):break
                    keep[active]&=access.clear(f,block[active],radius)
                controls.extend(int(a) for a in block[keep] if
                    (int(access.raw.atom_ids[a]),end) not in used and a!=center)
                if len(controls)>=b['negatives_per_positive']:break
            if len(controls)<b['negatives_per_positive']:reasons['insufficient_liquid_controls']+=1;continue
            pair=f'{event["episode"]}:{atom}:{end}'
            for a,label in [(int(center),1)]+[(a,0) for a in controls[:b['negatives_per_positive']]]:
                aid=int(access.raw.atom_ids[a]);used.add((aid,end))
                sequences.append([(int(f),a) for f in frames])
                metadata.append(dict(source=source,role=item['role'],event=-event['first_node'],pair=pair,
                    atom=aid,label=label,end_frame=end,appearance_frame=appearance,birth_frame=first,
                    confirmation_frame=follow,start_frame=int(frames[0]),episode=event['episode'],
                    stratum='failed_strong' if event['primary_failed_candidate'] else 'failed_other'))
            accepted+=1
        coverage.append(dict(source=source,role=item['role'],episode=event['episode'],
            strong=event['primary_failed_candidate'],peak_atoms=event['peak_atoms'],
            histories=accepted,exclusions=dict(reasons)))
    keys=sorted({key for seq in sequences for key in seq});lookup={key:i for i,key in enumerate(keys)}
    positions=np.empty((len(keys),80,3),np.float32)
    grouped=defaultdict(list)
    for f,a in keys:grouped[f].append(a)
    for frame,atoms in grouped.items():
        clouds,_=access.patch(frame,np.asarray(atoms))
        for atom,xyz in zip(atoms,clouds):positions[lookup[(frame,atom)]]=xyz
    fields=('source','role','event','pair','atom','label','end_frame','appearance_frame','birth_frame',
            'confirmation_frame','start_frame','episode','stratum')
    arrays={key:np.asarray([m[key] for m in metadata],dtype='U160' if key in
                ('role','pair','episode','stratum') else np.int64) for key in fields}
    arrays['indices']=np.asarray([[lookup[key] for key in seq] for seq in sequences],np.int64).reshape(-1,history)
    np.save(folder/'positions.npy',positions);np.savez_compressed(folder/'rows.npz',**arrays)
    write_json(folder/'coverage.json',coverage)
    receipt=dict(identity=identity,source=source,role=item['role'],rows=len(metadata),patches=len(positions),
        candidate_events=len(coverage),retained_events=sum(v['histories']>0 for v in coverage),
        seconds=time.monotonic()-begun,files={f:sha(folder/f) for f in ('positions.npy','rows.npz','coverage.json')})
    write_json(folder/'complete.json',receipt)
    # Bound shared method caches between independent source tasks.
    access.frame.cache_clear();access.labels.cache_clear()
    return receipt


def seal(c):
    p=read(resolve_path(c['output'])/'technical/data-plan.json')
    root=resolve_path(c['original_cache']);root.mkdir(parents=True,exist_ok=True)
    banks=[];parts=[];records=[];coverage=[];offset=0
    for item in p['original_plan']['sources']:
        f=root/'sources'/str(item['id']);done=read(f/'complete.json')
        if done['identity']!=digest(dict(plan=p['identity'],source=item['id'])) or any(sha(f/k)!=h for k,h in done['files'].items()):
            raise ValueError('Changed prepared transient inputs')
        with np.load(f/'rows.npz') as data:part={k:data[k] for k in data.files}
        part['indices']+=offset;parts.append(part);banks.append(np.load(f/'positions.npy'))
        offset+=done['patches'];records.append(done);coverage.extend(read(f/'coverage.json'))
    rows={key:np.concatenate([r[key] for r in parts]) for key in parts[0]}
    rows['id']=np.array([f'{s}:{a}:{e}:{y}' for s,a,e,y in zip(rows['source'],rows['atom'],rows['end_frame'],rows['label'])])
    if len(set(rows['id']))!=len(rows['id']):raise ValueError('Duplicate transient rows')
    groups=[f'{s}:{e}' for s,e in zip(rows['source'],rows['event'])];totals=Counter(groups)
    rows['weight']=np.array([1/totals[g] for g in groups])
    if set(rows['label'][rows['role']=='train'])!={0,1} or set(rows['label'][rows['role']!='train'])!={0,1}:
        raise ValueError('Transient cohort lacks source-held-out class support')
    write_json(root/'plan.json',p);np.save(root/'positions.npy',np.concatenate(banks));np.savez_compressed(root/'rows.npz',**rows)
    write_json(root/'manifest.json',dict(identity=p['identity'],files={f:sha(root/f) for f in
        ('plan.json','positions.npy','rows.npz')},sources=records,supported=True,summary=[],
        cohort='Failed-embryo appearance versus matched liquid; labels are appearance, not successful establishment'))
    write_json(resolve_path(c['output'])/'technical/eligibility.json',coverage)
    return rows


def relaxation_plan(c):
    target=resolve_path(c['output'])/'technical/relaxation-plan.json'
    if target.exists():return read(target)
    recipe=read(resolve_path(c['relaxation_recipe']))
    recipe.update(cache=c['relaxed_cache'],scratch=c['scratch'],archive=c['archive'],ranks=c['relaxation_ranks'])
    positions,rows,manifest=load(cache_config(c,'original'))
    p=read(resolve_path(c['output'])/'technical/data-plan.json');bindings={}
    for i in range(len(rows['id'])):
        for k,patch in enumerate(rows['indices'][i]):
            key=(int(rows['source'][i]),int(rows['start_frame'][i])+k,int(rows['atom'][i]))
            if int(patch) in bindings and bindings[int(patch)]!=key:raise ValueError('Inconsistent physical patch identity')
            bindings[int(patch)]=key
    if sorted(bindings)!=list(range(len(positions))):raise ValueError('Unbound patches')
    groups=defaultdict(list)
    for patch,(source,frame,atom) in sorted(bindings.items()):groups[(source,frame)].append(dict(patch=patch,atom=atom))
    tasks=[dict(id=f'{s}-{f}',source=s,frame=f,patches=v) for (s,f),v in sorted(groups.items())]
    np.random.default_rng(c['seed']).shuffle(tasks)
    binary=resolve_path(recipe['lammps_binary'])
    value=dict(config=recipe,ancestor_manifest=manifest,ancestor_cache=str(resolve_path(c['original_cache'])),
        sources=p['original_plan']['sources'],tasks=tasks,patch_count=len(positions),rows_sha256=manifest['files']['rows.npz'],
        potential_files=[str(resolve_path(f)) for f in recipe['potential_files']],lammps=str(binary),lammps_sha256=sha(binary),
        mpi_launcher=[str(Path(sys.executable).parent/'mpiexec'),'-n','{ranks}'],input_domain=cache_config(c,'relaxed')['input_domain'],
        cell_producer=sha(Path(cell.__code__.co_filename)))
    value['identity']=digest(value);write_json(target,value)
    return value


def seal_relaxed(c):
    p=relaxation_plan(c);root=resolve_path(c['relaxed_cache']);root.mkdir(parents=True,exist_ok=True)
    positions=np.empty((p['patch_count'],80,3),np.float32);coverage=np.zeros(len(positions),int)
    for task in p['tasks']:
        checked(p,task)
        with np.load(root/'cells'/task['id']/'patches.npz') as data:
            positions[data['indices']]=data['positions'];coverage[data['indices']]+=1
    if not np.all(coverage==1):raise ValueError('Missing or duplicated relaxed inputs')
    np.save(root/'positions.npy',positions);shutil.copy2(Path(p['ancestor_cache'])/'rows.npz',root/'rows.npz')
    write_json(root/'plan.json',p);records=[];offset=0
    for item in p['ancestor_manifest']['sources']:
        f=root/'sources'/str(item['source']);f.mkdir(parents=True,exist_ok=True)
        np.save(f/'positions.npy',positions[offset:offset+item['patches']]);offset+=item['patches']
        record=dict(item,identity=digest(dict(original=item['identity'],relaxation=p['identity'])),files={'positions.npy':sha(f/'positions.npy')})
        write_json(f/'complete.json',record);records.append(record)
    write_json(root/'manifest.json',dict(identity=p['identity'],sources=records,supported=True,summary=[],
        files={f:sha(root/f) for f in ('plan.json','positions.npy','rows.npz')},
        cohort=p['ancestor_manifest']['cohort'],input_domain=p['input_domain']))
