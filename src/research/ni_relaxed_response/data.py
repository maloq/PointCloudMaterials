"""Exact matched rows from the unrelaxed Ni protocol; only geometry is quenched."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from src.research.ni_response.data import read,root,cache,digest,sha,write_json,resolve_path,load,save,Source,atomic_save
from src.research.response_geometry.features import observed
from src.data import relaxation_index as index

FAMILY='ni_relaxed_response'

def parent(c):
    path=resolve_path(c['parent'])/'technical/plan.json'
    if sha(path)!=c['parent_plan_sha256']:raise ValueError('Changed matched unrelaxed plan')
    return read(path)

def prepare(c):
    path=root(c)/'technical/plan.json'
    if path.exists():
        old=read(path)
        if old['config']!=c:raise ValueError('Changed relaxed response configuration')
        if sha(resolve_path(c['checkpoint']))!=c['checkpoint_sha256']:raise ValueError('Changed relaxed checkpoint')
        return old
    old=parent(c);data=load(old['config'],'data');birth=load(old['config'],'births')
    if sha(resolve_path(c['checkpoint']))!=c['checkpoint_sha256']:raise ValueError('Relaxed checkpoint not complete or changed')
    manifest=read(resolve_path(c['spatial_manifest']))
    if sha(resolve_path(c['spatial_manifest']))!=c['spatial_manifest_sha256']:raise ValueError('Changed relaxed spatial ancestry')
    training={r['task']['lineage'] for r in manifest['shards'] if r['task']['role']=='train'}
    if training.intersection(old['partition']['test']):raise ValueError('Relaxed spatial checkpoint trained on held-out Ni ancestors')
    sources={s['key']:s for s in old['sources']}
    requests=defaultdict(set)
    for r in data['rows']+birth['rows']:requests[(r['source'],r['frame'])].add(r['center'])
    for j in old['jobs']:
        item=sources[j['source_key']]
        for frame in j['future_frames']:requests[(j['source_key'],frame)].update(item['centers'])
    spec=index.recipe(c['quench_settings']);tasks=[]
    for (source,frame),centers in sorted(requests.items()):
        s=sources[source]['source'];part,rawframe=s['frames'][frame]
        meta=dict(source_manifest_sha256=s['parts'][part]['manifest_sha256'],source_frame=rawframe,source_timestep=round(s['times_ps'][frame]*1000/s['parts'][part]['timestep_fs']))
        hits=[e['id'] for e in index.candidates(meta,spec=spec) if e['precision']=='native_restart' and index.available(e)]
        tasks.append(dict(key=digest([source,frame])[:20],source=source,frame=frame,centers=sorted(centers),reused_candidates=hits))
    response_jobs=[j for j in old['jobs'] if sources[j['source_key']]['original_role']=='train']
    p=dict(config=c,sources=old['sources'],jobs=old['jobs'],partition=old['partition'],native_tasks=tasks,response_jobs=response_jobs,
           parent_identity=old['identity'],parent_data_sha256=sha(cache(old['config'])/'data.pt'),parent_births_sha256=sha(cache(old['config'])/'births.pt'),
           summary=dict(native_cells=len(tasks),indexed_cells=sum(bool(t['reused_candidates']) for t in tasks),motion_rows=len(data['rows']),birth_rows=len(birth['rows']),response_initial_states=len(response_jobs),perturbed_endpoint_quenches=10*len(response_jobs)))
    p['identity']=digest(p);path=root(c)/'technical/plan.json'
    if path.exists() and read(path)!=p:raise ValueError('Changed relaxed plan; use another revision')
    write_json(path,p);return p

def native_path(c,task):return resolve_path(c['relaxation_archive'])/'native'/task['key']

def native_packet(c,task):
    folder=native_path(c,task);done=read(folder/'complete.json')
    if done['task']!=task or sha(folder/'patches.npz')!=done['patches_sha256']:raise ValueError('Changed relaxed native packet')
    with np.load(folder/'patches.npz') as a:return {k:a[k] for k in a.files}

def assemble(c,p):
    if (cache(c)/'data.json').exists() and (cache(c)/'births.json').exists():load(c,'data');load(c,'births');return
    old=parent(c);data=load(old['config'],'data');birth=load(old['config'],'births')
    packets={(t['source'],t['frame']):native_packet(c,t) for t in p['native_tasks']}
    lookup={key:{int(center):i for i,center in enumerate(a['center_atom_ids'])} for key,a in packets.items()}
    def state(source,frame,center):
        key=(source,frame);a=packets[key];i=lookup[key][center]
        return np.concatenate((a['positions'][i],a['velocities'][i]),-1)*c['length_factor']
    data['state']=torch.from_numpy(np.stack([state(r['source'],r['frame'],r['center']) for r in data['rows']])).float()
    sources={s['key']:s for s in p['sources']};future=[]
    for job in p['jobs']:
        item=sources[job['source_key']]
        future.append(np.stack([np.stack([state(item['key'],f,center)[:,:3] for f in job['future_frames']]) for center in item['centers']]))
    data['future']=torch.from_numpy(np.concatenate(future)).float()
    targets=[observed(block.double()) for block in data['future'].flatten(0,1).split(256)]
    data['target']=torch.cat(targets).reshape(len(data['rows']),len(c['horizons_ps']),24)
    train=np.array([r['role']=='train' for r in data['rows']]);v=data['target'][train].flatten(0,1)
    data['mean']=v.mean(0);data['scale']=v.std(0).clamp_min(1e-4)
    birth['state']=torch.from_numpy(np.stack([state(r['source'],r['frame'],r['center']) for r in birth['rows']])).float()
    save(c,'data',data);save(c,'births',birth)
    write_json(root(c)/'technical/data-complete.json',dict(identity=p['identity'],rows=len(data['rows']),birth_rows=len(birth['rows']),original_labels_and_row_order_unchanged=True,geometry='full-cell quenches, original selected atom identities',velocities='paired original MD velocities, not velocities generated by minimization'))
