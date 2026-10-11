"""Prepare paired relaxed targets and queue the existing Small eSEN trainer."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np

from src.data import relaxation_index as index
from src.data.fixed_cohort.protocol import centered, digest, sha, write_json
from src.data.structural_pretraining.prepare import source_arrays, chart
from src.experiment_runner.preparation import PreparationShard
from src.project_runtime.paths import REPO, resolve_path
from src.research.liquid_predictability.descriptors import descriptor_options, patch_descriptor_batch
from src.research.liquid_predictability.rich_multimaterial_data import seal
from .relaxed_spatial_inventory import plan, pair_arrays, checked_frame, read

MODULE = 'src.postprocessing.relaxed_spatial'
OWNER = 'Ni relaxation / paired relaxed Small eSEN chat; authorized 2026-10-08'


def root(c):
    return resolve_path(c['cache'])


@lru_cache(maxsize=8192)
def verify_file(path, expected):
    if sha(path) != expected:
        raise ValueError(f'Changed paired input: {path}')


def verify_pair(t):
    if 'entry' in t:
        verify_file(t['entry'],t['entry_sha256'])
        e=read(t['entry'])
        index.validate_metadata(e['metadata'],e['path'])
        for path,h in e['files'].items():
            verify_file(path,h)
    if 'input_files' in t:
        for name,h in t['input_files'].items():
            verify_file(str(Path(t['pair_path'])/name),h)


def original_checksum(source):
    """Once per source and preparation lane, verify the manifest's actual arrays."""
    from src.data.trajectories.shooting import _array_sha256
    arrays=source_arrays(source);m=read(Path(source['path'])/'manifest.json')
    for name,value in arrays.items():
        if _array_sha256(value)!=m['arrays'][name]['sha256']:
            raise ValueError(f'Original array changed: {source["path"]}/{name}')
    return source['manifest_sha256']


def native_coordinates(entry):
    """Read a saved double-precision restart on CPU; no force calls or dynamics."""
    from lammps import lammps
    model=lammps(cmdargs=['-log','none','-screen','none'])
    try:
        model.commands_string('units metal\natom_style atomic\nread_restart '+entry['payload'])
        n=int(model.get_natoms());ids=model.numpy.extract_atom('id')[:n].copy()
        xyz=model.numpy.extract_atom('x')[:n].copy();lo,hi,xy,yz,xz,*_=model.extract_box()
        if any(v!=0 for v in (xy,yz,xz)) or xyz.dtype!=np.float64:
            raise ValueError('Expected an orthogonal native float64 relaxed cell')
        order=np.argsort(ids)
        return ids[order],xyz[order],np.asarray(lo),np.asarray(hi)
    finally:
        model.close()


def cell_pair(t,e):
    arrays=checked_frame(t['original'],e['metadata']);frame=t['frame']
    x,tree,box=chart(arrays,frame,False)
    rows=np.searchsorted(arrays['atom_ids'],t['center_ids'])
    np.testing.assert_array_equal(arrays['atom_ids'][rows],t['center_ids'])
    neighbors=tree.query(x[rows],k=80,workers=1)[1]
    if not np.array_equal(neighbors[:,0],rows):
        raise ValueError('Original center is not the nearest atom')
    if e['precision']=='native_restart':
        ids,relaxed,lo,hi=native_coordinates(e)
    elif e['precision']=='archived_float16_cell':
        cold=source_arrays(dict(kind='dynamic',path=e['payload'],
            manifest_sha256=sha(Path(e['payload'])/'manifest.json')))
        ids=cold['atom_ids'];relaxed=cold['positions'][0].astype(np.float64)
        lo=cold['box_low'][0];hi=cold['box_high'][0]
        if int(cold['timesteps'][0])!=t['source_timestep']:
            raise ValueError('Relaxed timeline changed')
    else:
        raise ValueError('Unsupported full-cell precision')
    np.testing.assert_array_equal(ids,arrays['atom_ids'])
    np.testing.assert_allclose(lo,arrays['box_low'][frame],rtol=0,atol=1e-5)
    np.testing.assert_allclose(hi,arrays['box_high'][frame],rtol=0,atol=1e-5)
    if t['kind']=='ni':
        h=hashlib.sha256()
        for value in (arrays['positions'][frame],arrays['box_low'][frame],arrays['box_high'][frame],
                      arrays['timesteps'][frame:frame+1],arrays['atom_ids'],arrays['atom_types']):
            h.update(np.ascontiguousarray(value).tobytes())
        if h.hexdigest()!=t['ni_task']['frame_sha256']:
            raise ValueError('Ni original frame differs from the quenched input')
    return np.asarray(t['center_ids']),arrays['atom_ids'][neighbors],centered(x,box,rows,neighbors),centered(relaxed,box,rows,neighbors)


def load_pair(c,t):
    provenance={}
    if t['kind'].startswith('static_'):
        source=t['static_source'];verify_file(str(resolve_path(source['path'])),source['file_sha256'])
        if t['kind']=='static_cached':
            verify_pair(t);folder=Path(t['pair_path'])
            ids=np.load(folder/'center_ids.npy');near=np.load(folder/'neighbor_ids.npy');cold=np.load(folder/'hot.npy')
        else:
            arrays=source_arrays(source);x,tree,_=chart(arrays,0,True)
            ids=np.asarray(t['center_ids']);near=tree.query(x[ids],k=80,workers=1)[1]
            cold=(x[near]-x[ids,None]).astype(np.float32)
        if cold.shape!=(t['rows'],80,3) or not np.isfinite(cold).all() or np.any(cold[:,0]) or not np.array_equal(ids,near[:,0]):
            raise ValueError('Invalid static relaxed observation: '+t['id'])
        return (ids,near,None,cold),dict(relaxation_evidence=t['relaxation_evidence'],
            generating_potential='unknown',paired_original_available=False,static_source_sha256=source['file_sha256'])
    if t['kind']=='ni':
        folder=resolve_path(c['ni_relaxation'])/'technical'
        receipt_path=folder/'cells'/(t['ni_task']['id']+'.json');done=read(receipt_path)
        if done['identity']!=c['ni_identity'] or done['task']!=t['ni_task']:
            raise ValueError('Ni relaxation receipt does not match the planned frame')
        hit=index.full_cell(done['native_index_key'])
        if hit is None:
            raise FileNotFoundError('Missing native Ni relaxation '+done['native_index_key'])
        _,e=hit
        if (e['metadata']['source_manifest_sha256']!=t['source_manifest_sha256']
            or e['metadata']['source_frame']!=t['frame']):
            raise ValueError('Ni restart has a different original')
        values=cell_pair(t,e)
        provenance=dict(relaxation_entry=e['id'],relaxation_recipe=e['recipe'],
            ni_receipt=str(receipt_path),ni_receipt_sha256=sha(receipt_path),relaxation_files=e['files'])
    elif t['kind']=='cell':
        verify_pair(t);e=read(t['entry']);values=cell_pair(t,e)
        provenance=dict(relaxation_entry=e['id'],relaxation_recipe=e['recipe'])
    else:
        verify_pair(t);ids,near,hot,cold=pair_arrays(t);selected=np.asarray(t['selected_rows'])
        if hot is None:
            hot=np.zeros_like(cold)
            if t['original'] is not None:
                e=read(t['entry']);a=checked_frame(t['original'],e['metadata'])
                x,_,box=chart(a,t['frame'],False)
                center_rows=np.searchsorted(a['atom_ids'],ids[selected])
                rows=np.searchsorted(a['atom_ids'],near[selected])
                np.testing.assert_array_equal(a['atom_ids'][rows],near[selected])
                hot[selected]=centered(x,box,center_rows,rows)
            else:
                for r in t['original_references']:
                    old=r['original_task'];verify_pair(old)
                    oi,on,oh,_=pair_arrays(old);i=r['row'];j=r['original_row']
                    np.testing.assert_array_equal(on[j],near[i])
                    if oi[j]!=ids[i]:raise ValueError('Original center mismatch')
                    hot[i]=oh[j]
        values=ids[selected],near[selected],hot[selected],cold[selected]
        e=read(t['entry']);provenance=dict(relaxation_entry=e['id'],relaxation_recipe=e['recipe'])
    ids,near,hot,cold=values
    if hot.shape!=cold.shape or cold.shape!=(t['rows'],80,3) or not np.isfinite(hot).all() or not np.isfinite(cold).all():
        raise ValueError(f'Invalid paired geometry: {t["id"]}')
    if np.any(hot[:,0]) or np.any(cold[:,0]) or not np.array_equal(ids,near[:,0]):
        raise ValueError(f'Paired center/membership mismatch: {t["id"]}')
    return values,provenance


def initialize(c,p):
    global _config,_plan
    _config=c;_plan=p


def prepare_task(t):
    c,p=_config,_plan;dest=root(c)/'shards'/t['id'];dest.mkdir(parents=True,exist_ok=True)
    shard=PreparationShard(dest,p['identity'])
    if shard.verified(task=t) is not None:
        return dict(task=t['id'],rows=t['rows'],reused=True)
    started=time.monotonic();(ids,near,hot,cold),provenance=load_pair(c,t)
    factor=np.float32(p['normalization']['reference_scale_A']/p['normalization']['scales_A'][t['material']])
    x=np.asarray(cold,np.float32)*factor
    start=shard.offset('rows_done',total=t['rows']);mode='r+' if shard.resuming else 'w+'
    values=np.lib.format.open_memmap(dest/'features.npy',mode=mode,dtype=np.float32,shape=(t['rows'],len(p['columns'])))
    names=[v['name'] for v in p['columns']];options=descriptor_options(c)
    np.save(dest/'positions.npy',x)
    if hot is not None:np.save(dest/'original-positions.npy',np.asarray(hot,np.float32)*factor)
    np.savez(dest/'pair-identity.npz',center_atom_ids=ids,neighbor_atom_ids=near)
    write_json(dest/'pair-provenance.json',dict(provenance,source_manifest_sha256=t['source_manifest_sha256'],
        frame=t['frame'],source_timestep=t['source_timestep'],precision=t['precision'],factor=float(factor),
        encoder_inputs='relaxed positions only; paired originals excluded from training tensors'))
    shard.progress(rows_done=start)
    for begin in range(start,t['rows'],64):
        end=min(begin+64,t['rows'])
        y,current=patch_descriptor_batch(x[begin:end],**options)
        if current!=names:raise ValueError('Target schema changed')
        values[begin:end]=y;values.flush();shard.progress(rows_done=end)
    a=np.asarray(values,np.float64);np.savez(dest/'moments.npz',total=a.sum(0),second=(a*a).sum(0))
    values._mmap.close()
    files=['positions.npy','features.npy','moments.npz','pair-identity.npz','pair-provenance.json']
    if hot is not None:files.append('original-positions.npy')
    record=shard.complete(files,task=t,seconds=time.monotonic()-started)
    return dict(task=t['id'],rows=t['rows'],seconds=record['seconds'])


def prepare(c,lane):
    p=plan(c);tasks=[t for t in p['tasks'] if (t['kind']=='ni')==(lane=='ni')]
    if lane=='ni':
        complete=read(resolve_path(c['ni_relaxation'])/'technical/complete.json')
        if complete['identity']!=c['ni_identity'] or complete['state']!='complete' or complete['remaining_cells']!=0:
            raise ValueError('Ni campaign must be fully complete before target preparation')
    originals={t['original']['manifest_sha256']:t['original'] for t in tasks if t.get('original') is not None}
    context=multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=min(c['workers'],4),mp_context=context) as pool:
        verified=list(pool.map(original_checksum,originals.values()))
    write_json(root(c)/f'originals-{lane}.json',dict(identity=p['identity'],verified=verified))
    started=time.monotonic();done=rows=0
    with ProcessPoolExecutor(max_workers=c['workers'],initializer=initialize,initargs=(c,p),mp_context=context) as pool:
        futures={pool.submit(prepare_task,t):t['id'] for t in tasks}
        for future in as_completed(futures):
            try:value=future.result()
            except Exception as error:raise RuntimeError('Failed paired relaxed shard: '+futures[future]) from error
            done+=1;rows+=value['rows']
            if done%16==0 or done==len(tasks):
                status=dict(identity=p['identity'],lane=lane,shards_done=done,shards_total=len(tasks),rows_done=rows,
                    elapsed_seconds=time.monotonic()-started,last=value)
                write_json(root(c)/f'prepare-{lane}.json',status);print(json.dumps(status),flush=True)
    write_json(root(c)/f'complete-{lane}.json',dict(identity=p['identity'],tasks=len(tasks),rows=rows))


def publish(c):
    p=plan(c)
    for lane in ('al','ni'):
        if read(root(c)/f'complete-{lane}.json')['identity']!=p['identity']:
            raise ValueError('Preparation lane identity changed')
    seal(c);manifest=read(root(c)/'manifest.json');folder=root(c)/'packed';folder.mkdir(exist_ok=True)
    training=[s for s in manifest['shards'] if s['task']['role']=='train'];n=sum(s['task']['rows'] for s in training)
    positions=np.lib.format.open_memmap(folder/'positions.npy',mode='w+',dtype=np.float32,shape=(n,80,3))
    features=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float32,shape=(n,len(p['columns'])))
    records=[];begin=0
    for s in training:
        size=s['task']['rows'];d=root(c)/'shards'/s['task']['id']
        positions[begin:begin+size]=np.load(d/'positions.npy',mmap_mode='r')
        features[begin:begin+size]=np.load(d/'features.npy',mmap_mode='r')
        records.append(dict(task=s['task'],rows=size));begin+=size
    for value in (positions,features):value.flush();value._mmap.close()
    np.save(folder/'training-pool-row-ids.npy',np.arange(n,dtype=np.int64))
    with np.load(root(c)/'standardization.npz') as a:
        mean=a['mean'].astype(np.float32);scale=a['scale'].astype(np.float32);active=a['active']
    families=np.array([v['family'] for v in p['columns']]);weights=np.zeros(len(families),np.float32)
    for family in np.unique(families):
        take=(families==family)&active
        if not take.any():raise ValueError('No active targets in '+family)
        weights[take]=1/(3*take.sum())
    np.savez(folder/'target-standardization.npz',mean=mean,scale=scale,active=active,loss_weight=weights)
    files={name:sha(folder/name) for name in ('positions.npy','features.npy','training-pool-row-ids.npy','target-standardization.npz')}
    binding=dict(dataset=p['identity'],rows=n,subset_sha256=files['training-pool-row-ids.npy'],
        transform_sha256=files['target-standardization.npz'],descriptor_manifest_sha256=sha(root(c)/'manifest.json'))
    packed=dict(identity=digest(binding),binding=binding,files=files,shards=records,
        material_rows={k.split('/')[1]:v for k,v in p['counts'].items() if k.startswith('train/')},
        pool_rows=n,source_count=len({s['task']['source'] for s in training}),columns=p['columns'])
    write_json(folder/'manifest.json',packed)
    training_config=json.loads(json.dumps(c['reference_training']))
    training_config.update(protocol='spatial_esen_small_paired_and_static_relaxed_12epoch_v2',cache=c['cache'],
        output=c['output'],derived_cache=c['derived_cache'],prepared_identity=p['identity'],
        loader=dict(packed_cache=c['cache']+'/packed',max_open_shards=128),
        structural_dataset=dict(identity=p['identity'],normalization=p['normalization']),
        data=dict(training_rows=n,evaluation_role='evaluation',source_kind='dynamic_and_static',domain='relaxed',
            target='same 198 BO/CNA/TDA outputs; train-only relaxed moments and active mask',phase_filter=None,
            sampling='All eligible retained paired dynamic relaxations plus user-confirmed relaxed static Mg/Al/Zr/Ta. Existing cached neighborhoods retained; 256 fixed centers per source for uncached full cells.'),
        paired_relaxation=dict(identity=p['identity'],plan_sha256=sha(root(c)/'plan.json'),
            original_required_for_dynamic=True,original_is_model_input=False,precision_rows=p['precision_rows'],
            static_exception=dict(materials=['Mg','Al','Zr','Ta'],evidence='explicit user confirmation, 2026-10-08',rows=p['static_rows'])),
        scope='Fresh Small eSEN on paired dynamic relaxations and already-relaxed static Mg/Al/Zr/Ta; same architecture, VICReg and structural targets as October 8 Ni-enriched reference. Fixed epoch12. No motion or birth-label loss.')
    steps=math.ceil(n/training_config['spatial_training']['batch_size'])
    training_config['spatial_training'].update(warmup=steps,updates=12*steps)
    config_path=resolve_path(c['output'])/'technical/training-config.json';write_json(config_path,training_config)
    from src.research.spatial_architecture.workflow import prepare as prepare_training
    prepared=prepare_training(training_config)
    from src.experiment_runner.metric_docs import write_metric_rows
    write_metric_rows([dict(role=k.split('/')[0],material=k.split('/')[1],rows=v) for k,v in p['counts'].items()],
        resolve_path(c['output'])/'analyses/dataset',family='spatial_paired_relaxed',name='population')
    write_json(root(c)/'complete.json',dict(state='complete',identity=p['identity'],counts=p['counts'],
        training_config=str(config_path),training_plan_identity=prepared['identity'],packed_identity=packed['identity']))


def audit(c):
    p=plan(c);initialize(c,p);rows=[]
    kinds={(t['kind'],t['material']) for t in p['tasks']};chosen=[]
    for kind,material in sorted(kinds):
        choices=[t for t in p['tasks'] if t['kind']==kind and t['material']==material]
        if kind=='ni':
            choices=[t for t in choices if (resolve_path(c['ni_relaxation'])/'technical/cells'/(t['ni_task']['id']+'.json')).is_file()]
        if not choices:raise ValueError('No available audit input for '+kind+'/'+material)
        chosen.append(choices[0])
    options=descriptor_options(c)
    for t in chosen:
        started=time.monotonic();(ids,near,hot,cold),provenance=load_pair(c,t)
        factor=np.float32(p['normalization']['reference_scale_A']/p['normalization']['scales_A'][t['material']])
        x=cold[:2]*factor;native,names=patch_descriptor_batch(x,**options)
        reference,rnames=patch_descriptor_batch(x,**dict(options,alpha_backend='gudhi'))
        np.testing.assert_allclose(native,reference,rtol=2e-5,atol=2e-5)
        if names!=rnames or names!=[v['name'] for v in p['columns']]:raise ValueError('Audit target schema differs')
        rows.append(dict(kind=t['kind'],task=t['id'],material=t['material'],rows=t['rows'],columns=len(names),
            maximum_target_error=float(np.abs(native-reference).max()),seconds=time.monotonic()-started,
            precision=t['precision'],paired=hot is not None))
    result=dict(identity=p['identity'],passed=True,checks=rows,device='CPU; no simulator force or GPU call',
        implementations={n:sha(Path(__file__).with_name(n)) for n in ('relaxed_spatial.py','relaxed_spatial_inventory.py')})
    write_json(resolve_path(c['output'])/'technical/paired-preflight.json',result);print(json.dumps(result),flush=True)


def submit(c):
    from src.experiment_runner.execution import ExecutionBundle,SlurmQueue
    from src.experiment_runner.metric_docs import check_metric_docs
    from src.research.spatial_architecture.workflow import environment
    p=plan(c);tech=resolve_path(c['output'])/'technical'
    if (tech/'launch.json').exists():raise FileExistsError('Existing launch; do not duplicate this queue')
    preflight=read(tech/'paired-preflight.json')
    if not preflight['passed'] or preflight['identity']!=p['identity']:
        raise ValueError('Require completed matching paired input and descriptor audits')
    for n,h in preflight['implementations'].items():
        if sha(Path(__file__).with_name(n))!=h:raise ValueError('Preparation changed after audit')
    check_metric_docs(family='spatial_architecture');check_metric_docs(family='spatial_paired_relaxed')
    bundle=ExecutionBundle.freeze(REPO,tech/'code',c,directories=('src','docs/metrics','configs/data','configs/spatial_architecture'))
    env=dict(environment(),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='1',NUMBA_CACHE_DIR=str(root(c)/'compiled'))
    receipt=dict(owner=OWNER,jobs={},identity=p['identity'],bundle=str(bundle.root),launch_unix=time.time(),
        ni_relaxation_dependency=c['ni_status_job'],architectures=['esen'],
        execution='Al paired and static targets now; Ni targets after verified full relaxation completion; seal then one GPU Small eSEN')
    queue=SlurmQueue(tech,bundle,MODULE,env,tech/'launch.json',receipt,'RELAXED-ESEN')
    with queue.submission():
        options=[f'--cpus-per-task={c["workers"]}','--hint=nomultithread','--mem=32G',f'--time={c["preparation_hours"]:02d}:00:00']
        al=queue.submit('al',options,command_stage='prepare',arguments=('--lane','al'))
        ni=queue.submit('ni',options,dependency='afterok:'+c['ni_status_job'],command_stage='prepare',arguments=('--lane','ni'))
        sealed=queue.submit('publish',['--cpus-per-task=4','--mem=32G','--time=02:00:00'],dependency='afterok:'+al+':'+ni)
        queue.submit('queued',['--gpus=1',f'--cpus-per-task={c["runtime"]["cpu_threads"]}',
            f'--mem={c["runtime"]["memory_GB"]}G',f'--time={c["runtime"]["queue_hours"]:02d}:00:00','--requeue'],
            dependency='afterok:'+sealed,partition=c['runtime']['partition'],command_stage='train')
    print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('plan','audit','prepare','publish','submit','train'))
    parser.add_argument('--config',required=True);parser.add_argument('--lane',choices=('al','ni'))
    args=parser.parse_args();c=read(resolve_path(args.config))
    if args.stage=='plan':print(json.dumps({k:v for k,v in plan(c).items() if k in ('identity','counts','excluded','precision_rows')}))
    elif args.stage=='audit':audit(c)
    elif args.stage=='prepare':
        if args.lane is None:parser.error('prepare requires --lane')
        prepare(c,args.lane)
    elif args.stage=='publish':publish(c)
    elif args.stage=='submit':submit(c)
    else:
        from src.research.spatial_architecture.workflow import queued_worker
        path=resolve_path(c['output'])/'technical/training-config.json'
        training=read(path)
        if training['prepared_identity']!=plan(c)['identity']:raise ValueError('Training release changed')
        queued_worker(training,path)


if __name__=='__main__':
    # Canonical module names also keep spawned workers pickleable through the
    # shared runpy command dispatcher.
    from src.postprocessing.relaxed_spatial import main as entry
    entry()
