"""Explicit phase enrichment of an existing spatial fitting population."""
import argparse
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree

from src.postprocessing import spatial_dataset as base
from src.postprocessing.descriptors import ptm
from src.data.fixed_cohort.protocol import centered, digest, sha, write_json
from src.data.structural_pretraining.prepare import source_arrays, chart
from src.data.trajectories.shooting import _array_sha256
from src.experiment_runner.preparation import PreparationShard
from src.project_runtime.paths import REPO, resolve_path

read, root = base.read, base.root


def implementation():
    return dict(base.implementation(), enrichment=sha(Path(__file__)), ptm=sha(Path(ptm.__file__)))


def scan_plan(c):
    dest=root(c);dest.mkdir(parents=True,exist_ok=True);path=dest/'scan-plan.json'
    if path.exists():
        p=read(path)
        if p['config']!=c or p['implementation']!=implementation():raise ValueError('Changed enrichment execution; use a new release')
        return p
    parent=resolve_path(c['parent_cache']);old=read(parent/'plan.json');done=read(parent/'complete.json')
    if old['identity']!=c['parent_prepared_identity'] or done['state']!='complete' or done['identity']!=old['identity']:
        raise ValueError('Parent dataset identity or completion changed')
    sources=[];excluded=[]
    for s in old['sources']:
        raw=source_arrays(s);path=resolve_path(s['path']).parent/'structure.csv'
        observations={int(r['step']):r for r in csv.DictReader(path.open())}
        frames=[f for f in s['frames'] if float(observations[int(raw['timesteps'][f])]['crystal_fraction'])>=c['phase']['screen_global_fraction']]
        if not frames:
            excluded.append(s['id']);continue
        chosen=np.unique(np.linspace(0,len(frames)-1,min(len(frames),c['frames_per_source']),dtype=int))
        sources.append(dict(s,frames=[frames[i] for i in chosen],monitor_sha256=sha(path),eligible_frames=len(frames)))
    counts=done['counts']
    p=dict(config=c,implementation=implementation(),parent_plan_sha256=sha(parent/'plan.json'),
        parent_manifest_sha256=sha(parent/'manifest.json'),sources=sources,excluded_no_grown_frames=excluded,
        normalization=old['normalization'],additional_rows={
            'train':round(counts['train/Ni']*c['additional_ni_fraction']),
            'evaluation':round((counts['selection/Ni']+counts['test/Ni'])*c['additional_ni_fraction'])})
    p['identity']=digest(p);write_json(dest/'scan-plan.json',p)
    return p


def classify(raw,frame,p):
    """PTM center identity plus a local phase fraction and two-sided interface support."""
    rule=p['config']['phase'];x,tree,box=chart(raw,frame,False)
    labels=ptm.full_labels(x,box,rule['ptm_rmsd_cutoff']);crystal=np.isin(labels,[1,2,3])
    factor=p['normalization']['reference_scale_A']/p['normalization']['scales_A']['Ni']
    distance,near=tree.query(x,k=80,workers=1);support=distance*factor<rule['support_radius']
    fraction=(crystal[near]&support).sum(1)/support.sum(1)
    interior=crystal&(fraction>=rule['crystal_min_fraction']);liquid=fraction<=rule['liquid_max_fraction']
    core_distance=np.full(len(x),np.inf);liquid_distance=core_distance.copy()
    if interior.any() and liquid.any():
        core_distance=cKDTree(x[interior],boxsize=box).query(x,k=1,workers=1)[0]*factor
        liquid_distance=cKDTree(x[liquid],boxsize=box).query(x,k=1,workers=1)[0]*factor
    interface=((fraction>=rule['interface_min_fraction'])&(fraction<rule['crystal_min_fraction'])
        &(core_distance<=rule['interface_distance'])&(liquid_distance<=rule['interface_distance']))
    category=np.zeros(len(x),np.uint8);category[interior]=1;category[interface]=2
    return dict(ptm=labels,fraction=fraction.astype(np.float32),category=category,
        core_distance=core_distance.astype(np.float32),liquid_distance=liquid_distance.astype(np.float32))


def phase_reference(c,p):
    """Reuse the interrupted scan only when its classifier and physical inputs agree."""
    folder=resolve_path(c['phase_reference']);old=read(folder/'scan-plan.json')
    code=resolve_path(c['phase_reference_code'])
    def classifier(path):
        return ast.dump(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='classify'))
    if (old['identity']!=c['phase_reference_identity'] or sha(code)!=old['implementation']['enrichment']
        or old['config']['phase']!=c['phase'] or old['normalization']!=p['normalization']
        or classifier(code)!=classifier(Path(__file__))
        or any(old['implementation'][k]!=p['implementation'][k] for k in ('ptm','extraction'))):
        raise ValueError('Phase reference is not the same classifier and normalization')
    return folder,old


def scan_source(args):
    c,p,s=args;dest=root(c)/'phase'/s['id'];dest.mkdir(parents=True,exist_ok=True)
    shard=PreparationShard(dest,p['identity'])
    if shard.verified(source=s) is not None:return dict(source=s['id'],reused=True)
    raw=source_arrays(s);manifest=read(resolve_path(s['path'])/'manifest.json')
    for name in ('positions','atom_ids','timesteps','box_low','box_high'):
        if _array_sha256(raw[name])!=manifest['arrays'][name]['sha256']:raise ValueError(f'Changed Ni input {s["id"]}/{name}')
    old=resolve_path(c['parent_cache']);m=read(old/'manifest.json')
    prior=[t for t in m['shards'] if t['task']['source']==s['id'] and t['task']['role']==s['role']]
    if len(prior)!=1:raise ValueError(f'Expected one original Ni shard: {s["id"]}')
    previous=old/'shards'/prior[0]['task']['id']/'row_ids.npy'
    if sha(previous)!=prior[0]['files']['row_ids.npy']:raise ValueError('Changed parent row identities')
    used=np.load(previous);candidates=[];frames=[]
    reference,old_scan=phase_reference(c,p)
    old_source=next((v for v in old_scan['sources'] if v['id']==s['id']),None)
    for frame in s['frames']:
        folder=dest/f'frame-{frame:06d}';folder.mkdir(exist_ok=True);saved=PreparationShard(folder,p['identity'])
        receipt=saved.verified(frame=frame)
        if receipt is None and old_source is not None and frame in old_source['frames']:
            if old_source['manifest_sha256']!=s['manifest_sha256']:raise ValueError('Phase reference trajectory changed')
            previous_frame=reference/'phase'/s['id']/folder.name
            prior_receipt=PreparationShard(previous_frame,old_scan['identity']).verified(frame=frame)
            if prior_receipt is not None:
                shutil.copy2(previous_frame/'labels.npz',folder/'labels.npz')
                receipt=saved.complete(('labels.npz',),frame=frame,reused_scan=old_scan['identity'])
        if receipt is None:
            value=classify(raw,frame,p);np.savez_compressed(folder/'labels.npz',**value)
            receipt=saved.complete(('labels.npz',),frame=frame)
        with np.load(folder/'labels.npz') as labels:category=labels['category'].copy()
        already=used[used[:,0]==frame,1];indices=np.searchsorted(raw['atom_ids'],already)
        if not np.array_equal(raw['atom_ids'][indices],already):raise ValueError('Invalid original center identities')
        category[indices]=0;rows=np.flatnonzero(category)
        candidates.append(np.column_stack((np.full(len(rows),frame,np.int64),rows,category[rows])).astype(np.int64))
        frames.append(dict(frame=frame,sha256=receipt['files']['labels.npz']))
        shard.progress(frames_done=len(frames),frames_total=len(s['frames']))
    values=np.concatenate(candidates);np.save(dest/'candidates.npy',values)
    write_json(dest/'frames.json',frames)
    counts={str(i):int((values[:,2]==i).sum()) for i in (1,2)}
    shard.complete(('candidates.npy','frames.json'),source=s,counts=counts)
    return dict(source=s['id'],counts=counts)


def scan(c):
    p=scan_plan(c)
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        work=[pool.submit(scan_source,(c,p,s)) for s in p['sources']]
        for i,f in enumerate(as_completed(work),1):
            status=dict(state='classifying',sources_done=i,sources_total=len(work),last=f.result())
            write_json(root(c)/'status.json',status);print(status,flush=True)


def plan(c):
    scanp=scan_plan(c);dest=root(c);path=dest/'plan.json'
    if path.exists():return base.plan(c)
    parent=resolve_path(c['parent_cache']);old=read(parent/'plan.json');m=read(resolve_path(c['parent_packed'])/'manifest.json')
    if m['identity']!=c['parent_identity'] or m['binding']['dataset']!=old['identity']:raise ValueError('Changed packed parent')
    records=read(parent/'manifest.json');tasks=[];begin=0;role_begin=Counter()
    for record in m['shards']:
        t=record['task'];n=record['rows']
        tasks.append(dict(id='base-'+t['id'],source=t['source'],lineage=t['lineage'],role='train',material=t['material'],
            potential=t['potential'],kind='parent_packed',begin=begin,rows=n));begin+=n
    for record in records['shards']:
        t=record['task'];role=t['role']
        if role=='train':continue
        merged='evaluation' if role in ('selection','test') else role
        tasks.append(dict(id='base-'+t['id'],source=t['source'],lineage=t['lineage'],role=merged,original_role=role,material=t['material'],
            potential=t['potential'],kind='parent_shards',rows=t['rows'],inputs=[dict(shard=record,role_row_begin=role_begin[role])]))
        role_begin[role]+=t['rows']
    candidates=[];receipts=[]
    for i,s in enumerate(scanp['sources']):
        d=dest/'phase'/s['id'];receipt=PreparationShard(d,scanp['identity']).verified(source=s)
        if receipt is None:raise ValueError(f'Incomplete phase scan {s["id"]}')
        a=np.load(d/'candidates.npy');candidates.append(np.column_stack((np.full(len(a),i,np.int64),a)))
        receipts.append(dict(source=s['id'],complete_sha256=sha(d/'complete.json')))
    pool=np.concatenate(candidates);draws=[];requested=scanp['additional_rows'];counts={};rng=np.random.default_rng(c['seed'])
    pool_roles=np.array(['train' if s['role']=='train' else 'evaluation' for s in scanp['sources']])[pool[:,0]]
    for role,number in requested.items():
        for category,n in ((1,number//2),(2,number-number//2)):
            ids=np.flatnonzero((pool[:,3]==category)&(pool_roles==role))
            if len(ids)<n:raise ValueError(f'Need {n} {role} category {category} observations, found {len(ids)}')
            draws.append(pool[rng.choice(ids,n,replace=False)]);counts[role+'/'+str(category)]=dict(available=len(ids),selected=n)
    selected=np.concatenate(draws);sources=[];inputs=dest/'inputs';inputs.mkdir(exist_ok=True)
    for i,s in enumerate(scanp['sources']):
        a=selected[selected[:,0]==i]
        if not len(a):continue
        a=a[np.lexsort((a[:,2],a[:,1]))];samples=a[:,1:3];name='enrich-'+s['id'];np.save(inputs/(name+'.npy'),samples)
        np.save(inputs/(name+'-category.npy'),a[:,3].astype(np.uint8));raw=source_arrays(s)
        sources.append(dict(s,center_ids=raw['atom_ids'].astype(np.int64).tolist()))
        role='train' if s['role']=='train' else 'evaluation'
        tasks.append(dict(id=name,source=s['id'],lineage=s['lineage'],role=role,original_role=s['role'],material='Ni',potential=s['potential'],kind='ni',
            rows=len(a),samples='inputs/'+name+'.npy',samples_sha256=sha(inputs/(name+'.npy')),
            categories='inputs/'+name+'-category.npy',categories_sha256=sha(inputs/(name+'-category.npy'))))
    population=Counter()
    roles={}
    for t in tasks:population[t['role']+'/'+t['material']]+=t['rows'];roles.setdefault(t['role'],set()).add(t['lineage'])
    for a in roles:
        for b in roles:
            if a!=b and roles[a]&roles[b]:raise ValueError(f'Ancestry overlap: {a}/{b}')
    p=dict(protocol=c['protocol'],config=c,implementation=base.implementation(),enrichment_implementation=implementation(),
        columns=m['columns'],tasks=tasks,sources=sources,normalization=scanp['normalization'],counts=dict(population),
        parent_columns=list(range(len(m['columns']))),parent_training_rows=begin,ni_training_rows=population['train/Ni'],
        ni_fraction=population['train/Ni']/(begin+requested['train']),scan_identity=scanp['identity'],candidate_receipts=receipts,
        additional_rows=requested,categories=counts,parent_manifest_sha256=sha(resolve_path(c['parent_packed'])/'manifest.json'),
        parent_cache_manifest_sha256=sha(parent/'manifest.json'),
        sampling='all parent rows + equal crystal/interfacial additions drawn uniformly without replacement from declared candidates',
        inputs='unchanged unrelaxed normalized nearest80 geometry; PTM and phase categories are selection/audit metadata only')
    p['identity']=digest(p);write_json(path,p);write_json(resolve_path(c['output'])/'technical/plan-summary.json',
        dict(identity=p['identity'],counts=p['counts'],additional_rows=requested,categories=counts,ni_fraction=p['ni_fraction']))
    return p


def audit(c):
    p=scan_plan(c);s=p['sources'][0];scan_source((c,p,s));raw=source_arrays(s)
    a=np.load(root(c)/'phase'/s['id']/'candidates.npy');selected=np.concatenate([a[a[:,2]==i][:3] for i in (1,2)])
    if len(selected)!=6:raise ValueError('Preflight source needs both phase categories')
    from src.research.liquid_predictability.descriptors import patch_descriptor_batch,patch_descriptors,descriptor_options
    patches=[];factor=np.float32(p['normalization']['reference_scale_A']/p['normalization']['scales_A']['Ni'])
    for frame,row,category in selected:
        x,tree,box=chart(raw,int(frame),False);near=tree.query(x[[row]],k=80,workers=1)[1]
        patches.append((centered(x,box,np.array([row]),near)*factor)[0])
    values,names=patch_descriptor_batch(np.stack(patches),**descriptor_options(c));worst=0.
    for x,y in zip(patches,values):
        full,columns=patch_descriptors(x,alpha_backend='gudhi');expected=full[[columns.index(n) for n in names]]
        error=float(np.abs(expected-y).max());worst=max(worst,error)
        if not np.allclose(expected,y,atol=1e-10,rtol=0):raise ValueError(f'Native/reference target mismatch: {error}')
    if not scan_source((c,p,s)).get('reused'):raise ValueError('Phase scan reuse failed')
    write_json(resolve_path(c['output'])/'technical/preflight.json',dict(identity=p['identity'],source=s['id'],
        reference_rows=len(selected),max_absolute_error=worst,reused=True))
    print(dict(reference_rows=len(selected),max_absolute_error=worst),flush=True)


def submit(c):
    from src.experiment_runner.execution import ExecutionBundle,SlurmQueue
    from src.experiment_runner.metric_docs import check_metric_docs
    p=scan_plan(c);tech=resolve_path(c['output'])/'technical';check=read(tech/'preflight.json')
    if check['identity']!=p['identity'] or not check['reused']:raise ValueError('Missing matching phase/target audit')
    if (tech/'launch.json').exists():raise ValueError('Already submitted; use recorded frozen execution')
    check_metric_docs(family=c['metric_family'])
    bundle=ExecutionBundle.freeze(REPO,tech/'code',c,directories=('src','docs/metrics','configs/data'),files=())
    env=dict(PCM_PROJECT_ROOT=str(REPO),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OVITO_THREAD_COUNT='1',
        OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        NUMBA_CACHE_DIR=str(root(c)/'compiled'))
    receipt=dict(owner='Current Ni spatial dataset chat',scan_identity=p['identity'],code=str(bundle.root),jobs={},
        submitted_at=datetime.now(timezone.utc).isoformat())
    queue=SlurmQueue(tech,bundle,'src.postprocessing.spatial_enrichment',env,tech/'launch.json',receipt,'SPATIAL-NI-PHASE')
    with queue.submission():queue.submit('run',[f'--cpus-per-task={c["workers"]}','--hint=nomultithread','--mem=20G',
        '--time=02:00:00','--nodelist='+c['cpu_node']])
    print(receipt,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('scan-plan','audit','scan','plan','run','submit'))
    parser.add_argument('--config',required=True);args=parser.parse_args();c=read(resolve_path(args.config))
    if args.stage=='scan-plan':
        p=scan_plan(c);print(dict(identity=p['identity'],sources=len(p['sources']),additional_rows=p['additional_rows']))
    elif args.stage=='audit':audit(c)
    elif args.stage=='submit':submit(c)
    elif args.stage=='scan':scan(c)
    elif args.stage=='plan':plan(c)
    else:scan(c);plan(c);base.prepare(c);base.publish(c)


if __name__=='__main__':
    from src.postprocessing.spatial_enrichment import main as entry
    entry()
