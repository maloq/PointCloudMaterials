"""Append outcome-blind Ni observations to the fixed spatial fitting population."""
import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from src.data.fixed_cohort.protocol import centered, digest, sha, write_json
from src.data.structural_pretraining.native import frames_at_spacing
from src.data.structural_pretraining.prepare import source_arrays, chart
from src.data.trajectories.shooting import _array_sha256
from src.experiment_runner.preparation import PreparationShard
from src.project_runtime.paths import REPO, resolve_path
from src.research.liquid_predictability.descriptors import descriptor_options, patch_descriptor_batch
from src.research.liquid_predictability.rich_multimaterial_data import descriptor_implementation, seal


def read(path):return json.loads(Path(path).read_text())
def root(c):return resolve_path(c['cache'])


def implementation():
    from src.data.structural_pretraining import prepare, native
    from src.data.trajectories import shooting
    return dict(descriptor_implementation(), spatial_dataset=sha(Path(__file__)),
                extraction=sha(Path(prepare.__file__)), spacing=sha(Path(native.__file__)),
                array_checksum=sha(Path(shooting.__file__)))


def discover(c):
    sources=[];excluded=[];plans={}
    for name in c['ni_campaigns']:
        p=resolve_path(name);campaign=read(p);plans[name]=sha(p);cfg=campaign['config']
        for record in campaign['records']:
            directory=resolve_path(cfg['archive'])/record['id'];done=directory/'complete.json'
            path=directory/'trajectory_binary_float16';manifest=path/'manifest.json'
            if not done.exists() or not manifest.exists():
                excluded.append(dict(source=record['id'],dataset=cfg['campaign_id'],role=record['role'],reason='not published complete at inventory freeze'))
                continue
            complete=read(done);meta=read(directory/'metadata.json');m=read(manifest)
            if (complete['state']!='complete' or complete['record']!=record
                or complete['trajectory_manifest_sha256']!=sha(manifest)
                or meta['material']!='Ni' or meta['potential']!='ni-mendelev2012-eam'
                or meta['potential_sha256']!=cfg['potential_sha256']
                or meta['root_lineage']!=record['id'] or meta['split']!=record['role']):
                raise ValueError(f'Ni production provenance changed: {directory}')
            s=dict(id=record['id'],lineage=meta['root_lineage'],role=record['role'],split=record['role'],
                material='Ni',potential=meta['potential'],potential_sha256=meta['potential_sha256'],
                kind='dynamic',path=f"{cfg['archive']}/{record['id']}/trajectory_binary_float16",
                manifest_sha256=sha(manifest),metadata_sha256=sha(directory/'metadata.json'),
                complete_sha256=sha(done),dataset=cfg['campaign_id'],timestep_fs=1000*cfg['timestep_ps'],
                frame_count=m['frame_count'],preparation_seed=record['preparation_seed'],temperature_K=record['temperature_K'])
            arrays=source_arrays(s);steps=np.asarray(arrays['timesteps'])
            if not np.allclose(np.diff(steps)*s['timestep_fs']/1000,.1,atol=1e-10,rtol=0):
                raise ValueError(f'Ni source is not native 0.10 ps: {s["id"]}')
            s['frames']=frames_at_spacing(steps,s['timestep_fs'],c['frame_spacing_ps'])
            s['atom_count']=len(arrays['atom_ids'])
            count=int(round(s['atom_count']*c['center_fraction']))
            rng=np.random.default_rng(int(digest([c['seed'],s['id'],'centers'])[:16],16))
            rows=np.sort(rng.choice(s['atom_count'],count,replace=False))
            s['center_ids']=arrays['atom_ids'][rows].astype(np.int64).tolist()
            sources.append(s)
    for key in ('id','lineage','preparation_seed','manifest_sha256'):
        if len({s[key] for s in sources})!=len(sources):raise ValueError(f'Duplicate Ni {key}')
    return sources,excluded,plans


def calibrate_ni(sources,seed):
    rng=np.random.default_rng(seed);eligible=[s for s in sources if s['role']=='train']
    chosen=[eligible[i] for i in rng.choice(len(eligible),min(8,len(eligible)),replace=False)]
    distances=[];evidence=[]
    for i,s in enumerate(chosen):
        arrays=source_arrays(s);frame=min(2,s['frame_count']-1);x,tree,_=chart(arrays,frame,False)
        count=4000//len(chosen)+(i<4000%len(chosen));ids=rng.choice(len(x),count,replace=False)
        distances.extend(tree.query(x[ids],k=160,workers=1)[0][:,-1].tolist())
        evidence.append(dict(source=s['id'],lineage=s['lineage'],frame=frame,center_ids=arrays['atom_ids'][ids].tolist()))
    return float(np.quantile(distances,.995)*1.02),evidence


def plan(c):
    dest=root(c);dest.mkdir(parents=True,exist_ok=True)
    if (dest/'plan.json').exists():
        p=read(dest/'plan.json')
        if p['config']!=c or p['implementation']!=implementation():raise ValueError('Changed spatial extension plan; use a new release')
        return p
    parent=resolve_path(c['parent_packed']);m=read(parent/'manifest.json')
    old=resolve_path(c['parent_cache']);raw=read(old/'manifest.json')
    if m['identity']!=c['parent_identity'] or raw['identity']!=m['binding']['dataset']:
        raise ValueError('Changed parent spatial dataset')
    options=descriptor_options(c);names=options['selected_columns'];lookup={v['name']:i for i,v in enumerate(m['columns'])}
    columns=[m['columns'][lookup[n]] for n in names]
    groups=OrderedDict();offset=0
    for shard in m['shards']:
        t=shard['task'];key=('train',t['source'])
        if t['role']!='train':raise ValueError('Nontraining ancestor in parent fitting set')
        if key not in groups:
            groups[key]=dict(id='base-train-'+t['source'],source=t['source'],lineage=t['lineage'],role='train',
                material=t['material'],potential=t['potential'],kind='parent_packed',begin=offset,rows=0)
        g=groups[key]
        if g['begin']+g['rows']!=offset:raise ValueError('Parent source blocks must be contiguous')
        g['rows']+=shard['rows'];offset+=shard['rows']
    if offset!=m['binding']['rows']:raise ValueError('Parent packed row count changed')
    role_offsets=Counter()
    for shard in raw['shards']:
        t=shard['task'];role=t['role']
        if role=='train':continue
        key=(role,t['source'])
        if key not in groups:
            groups[key]=dict(id=f'base-{role}-'+t['source'],source=t['source'],lineage=t['lineage'],role=role,
                material=t['material'],potential=t['potential'],kind='parent_shards',rows=0,inputs=[])
        g=groups[key];g['inputs'].append(dict(shard=shard,role_row_begin=role_offsets[role]))
        g['rows']+=t['rows'];role_offsets[role]+=t['rows']
    tasks=list(groups.values());sources,excluded,campaigns=discover(c)
    ni_scale,calibration=calibrate_ni(sources,c['seed']+11)
    ni_rows=round(offset*c['ni_training_fraction']/(1-c['ni_training_fraction']))
    for role,requested in [('train',ni_rows),('selection',c['ni_heldout_rows']),('test',c['ni_heldout_rows'])]:
        eligible=[s for s in sources if s['role']==role]
        counts=[len(s['frames'])*len(s['center_ids']) for s in eligible];ends=np.cumsum(counts)
        if not len(ends) or requested>ends[-1]:raise ValueError(f'Insufficient Ni {role} candidate rows')
        rng=np.random.default_rng(int(digest([c['seed'],role,'rows'])[:16],16))
        selected=np.sort(rng.choice(int(ends[-1]),requested,replace=False));begin=0
        for s,end in zip(eligible,ends):
            a,b=np.searchsorted(selected,[begin,end]);local=selected[a:b]-begin;begin=end
            if not len(local):continue
            frames=np.array(s['frames'])[local//len(s['center_ids'])]
            centers=local%len(s['center_ids']);samples=np.stack((frames,centers),axis=1).astype(np.int64)
            target=dest/'inputs'/(s['id']+'.npy');target.parent.mkdir(exist_ok=True);np.save(target,samples)
            tasks.append(dict(id=s['id'],source=s['id'],lineage=s['lineage'],role=role,material='Ni',
                potential=s['potential'],kind='ni',rows=len(samples),samples='inputs/'+target.name,samples_sha256=sha(target)))
    train={t['lineage'] for t in tasks if t['role']=='train'};held={t['lineage'] for t in tasks if t['role']!='train'}
    if train&held:raise ValueError('Training and held-out ancestry overlap')
    counts=Counter()
    for t in tasks:counts[t['role']+'/'+t['material']]+=t['rows']
    normalization=dict(c['normalization'],scales_A=dict(c['normalization']['scales_A'],Ni=ni_scale),
        ni_calibration=calibration,ni_rule='0.995 quantile of nearest-160 distances * 1.02; 4000 training points across up to 8 training sources')
    p=dict(protocol=c['protocol'],config=c,implementation=implementation(),columns=columns,tasks=tasks,
        sources=sources,excluded_ni=excluded,ni_campaigns=campaigns,normalization=normalization,
        counts=dict(counts),parent_manifest_sha256=sha(parent/'manifest.json'),parent_cache_manifest_sha256=sha(old/'manifest.json'),
        parent_columns=[lookup[n] for n in names],parent_training_rows=offset,ni_training_rows=ni_rows,
        ni_fraction=ni_rows/(offset+ni_rows),sampling='all existing fitting rows + uniform Ni frame/center candidates; no phase/event labels',
        inputs='unrelaxed normalized nearest80 positions only; no species, time, temperature, velocities or labels')
    p['identity']=digest(p);write_json(dest/'plan.json',p)
    write_json(resolve_path(c['output'])/'technical/plan-summary.json',dict(identity=p['identity'],counts=p['counts'],
        ni_fraction=p['ni_fraction'],ni_scale_A=ni_scale,included_ni_sources=len(sources),excluded_ni=excluded))
    return p


def initialize(c,p):
    global _config,_plan
    _config=c;_plan=p


def prepare_task(task):
    c,p=_config,_plan;dest=root(c)/'shards'/task['id'];dest.mkdir(parents=True,exist_ok=True)
    shard=PreparationShard(dest,p['identity'])
    if shard.verified(task=task) is not None:return dict(task=task['id'],rows=task['rows'],reused=True)
    start=shard.offset('rows_done',total=task['rows']);mode='r+' if shard.resuming else 'w+'
    arrays={name:np.lib.format.open_memmap(dest/(name+'.npy'),mode=mode,dtype=dtype,shape=shape) for name,dtype,shape in
        [('positions',np.float32,(task['rows'],80,3)),('features',np.float32,(task['rows'],len(p['columns']))),
         ('row_ids',np.int64,(task['rows'],2))]}
    started=time.monotonic();shard.progress(rows_done=start)
    def save(begin,x,y,ids):
        end=int(begin+len(x));arrays['positions'][begin:end]=x;arrays['features'][begin:end]=y;arrays['row_ids'][begin:end]=ids
        for a in arrays.values():a.flush()
        shard.progress(rows_done=end)
    if task['kind']=='parent_packed':
        parent=resolve_path(c['parent_packed']);x=np.load(parent/'positions.npy',mmap_mode='r');y=np.load(parent/'features.npy',mmap_mode='r')
        for begin in range(start,task['rows'],8192):
            end=min(begin+8192,task['rows']);a=task['begin']+begin;b=task['begin']+end
            save(begin,x[a:b],y[a:b][:,p['parent_columns']],np.stack((np.zeros(b-a,np.int64),np.arange(a,b)),1))
    elif task['kind']=='parent_shards':
        begin=0
        for item in task['inputs']:
            record=item['shard'];size=record['task']['rows'];source=resolve_path(c['parent_cache'])/'shards'/record['task']['id']
            if begin+size>start:
                for name in ('positions.npy','features.npy'):
                    if sha(source/name)!=record['files'][name]:raise ValueError(f'Changed parent shard {source/name}')
                x=np.load(source/'positions.npy',mmap_mode='r');y=np.load(source/'features.npy',mmap_mode='r')
                a=max(0,start-begin);oldrows=np.arange(item['role_row_begin']+a,item['role_row_begin']+size)
                save(begin+a,x[a:],y[a:][:,p['parent_columns']],np.stack((np.ones(len(oldrows),np.int64),oldrows),1))
            begin+=size
    else:
        source=next(s for s in p['sources'] if s['id']==task['source']);raw=source_arrays(source)
        manifest=read(resolve_path(source['path'])/'manifest.json')
        for name in ('positions','atom_ids','timesteps','box_low','box_high'):
            spec=manifest['arrays'][name]
            if _array_sha256(raw[name])!=spec['sha256']:raise ValueError(f'Changed Ni input: {source["id"]}/{name}')
        samples=root(c)/task['samples']
        if sha(samples)!=task['samples_sha256']:raise ValueError('Changed Ni sampling')
        selected=np.load(samples);center_rows=np.searchsorted(raw['atom_ids'],source['center_ids'])
        if not np.array_equal(raw['atom_ids'][center_rows],source['center_ids']):raise ValueError('Ni center identity changed')
        factor=np.float32(p['normalization']['reference_scale_A']/p['normalization']['scales_A']['Ni'])
        for frame in np.unique(selected[start:,0]):
            a,b=np.searchsorted(selected[:,0],[frame,frame+1]);a=max(a,start)
            x,tree,box=chart(raw,int(frame),False);rows=center_rows[selected[a:b,1]]
            near=tree.query(x[rows],k=80,workers=1)[1]
            if not np.array_equal(near[:,0],rows):raise ValueError(f'Ni center tie: {source["id"]}/{frame}')
            patches=centered(x,box,rows,near)*factor
            values,names=patch_descriptor_batch(patches,**descriptor_options(c))
            if names!=[v['name'] for v in p['columns']]:raise ValueError('Changed target names')
            save(a,patches,values,np.stack((np.full(b-a,frame,np.int64),np.asarray(source['center_ids'])[selected[a:b,1]]),1))
    features=arrays['features'];total=np.zeros(len(p['columns']));second=total.copy()
    for begin in range(0,len(features),8192):
        a=np.asarray(features[begin:begin+8192],np.float64)
        if not np.isfinite(a).all():raise FloatingPointError(f'Nonfinite targets {task["id"]}')
        total+=a.sum(0);second+=(a*a).sum(0)
    np.savez(dest/'moments.npz',total=total,second=second)
    for a in arrays.values():a._mmap.close()
    receipt=shard.complete(('positions.npy','features.npy','row_ids.npy','moments.npz'),task=task,seconds=time.monotonic()-started)
    return dict(task=task['id'],rows=task['rows'],seconds=receipt['seconds'])


def prepare(c):
    p=plan(c);parent=resolve_path(c['parent_packed']);m=read(parent/'manifest.json')
    for name,h in m['files'].items():
        if sha(parent/name)!=h:raise ValueError(f'Changed parent packed array {name}')
    started=time.monotonic();done=rows=0
    with ProcessPoolExecutor(max_workers=c['workers'],initializer=initialize,initargs=(c,p)) as pool:
        futures={pool.submit(prepare_task,t):t for t in sorted(p['tasks'],key=lambda t:(t['kind']!='ni',-t['rows']))}
        for f in as_completed(futures):
            value=f.result();done+=1;rows+=value['rows']
            status=dict(state='preparing',shards_done=done,shards_total=len(p['tasks']),rows_done=rows,
                elapsed_seconds=time.monotonic()-started,last=value)
            write_json(root(c)/'status.json',status);print(json.dumps(status),flush=True)


def publish(c):
    p=plan(c);seal(c);m=read(root(c)/'manifest.json');folder=root(c)/'packed';folder.mkdir(exist_ok=True)
    training=[s for s in m['shards'] if s['task']['role']=='train'];n=sum(s['task']['rows'] for s in training)
    positions=np.lib.format.open_memmap(folder/'positions.npy',mode='w+',dtype=np.float32,shape=(n,80,3))
    features=np.lib.format.open_memmap(folder/'features.npy',mode='w+',dtype=np.float32,shape=(n,len(p['columns'])))
    records=[];begin=0
    for s in training:
        size=s['task']['rows'];d=root(c)/'shards'/s['task']['id']
        positions[begin:begin+size]=np.load(d/'positions.npy',mmap_mode='r');features[begin:begin+size]=np.load(d/'features.npy',mmap_mode='r')
        records.append(dict(task=s['task'],rows=size));begin+=size
    positions.flush();features.flush();positions._mmap.close();features._mmap.close()
    np.save(folder/'training-pool-row-ids.npy',np.arange(n,dtype=np.int64))
    with np.load(root(c)/'standardization.npz') as a:
        mean=a['mean'].astype(np.float32);scale=a['scale'].astype(np.float32);active=a['active']
    families=np.array([v['family'] for v in p['columns']]);weights=np.zeros(len(families),np.float32)
    for family in np.unique(families):
        take=(families==family)&active
        if not take.any():raise ValueError(f'No active targets in {family}')
        weights[take]=1/(3*take.sum())
    np.savez(folder/'target-standardization.npz',mean=mean,scale=scale,active=active,loss_weight=weights)
    files={name:sha(folder/name) for name in ('positions.npy','features.npy','training-pool-row-ids.npy','target-standardization.npz')}
    binding=dict(dataset=p['identity'],rows=n,subset_sha256=files['training-pool-row-ids.npy'],
        transform_sha256=files['target-standardization.npz'],descriptor_manifest_sha256=sha(root(c)/'manifest.json'))
    material_rows={k.split('/')[1]:v for k,v in p['counts'].items() if k.startswith('train/')}
    packed=dict(identity=digest(binding),binding=binding,files=files,shards=records,material_rows=material_rows,
        pool_rows=n,source_count=len({s['task']['source'] for s in training}),columns=p['columns'])
    write_json(folder/'manifest.json',packed)
    consumer=dict(cache=c['cache'],prepared_identity=p['identity'],data=dict(training_rows=n,
        evaluation_role=c.get('evaluation_role','selection')),
        loader=dict(packed_cache=c['cache']+'/packed',max_open_shards=128),descriptors=c['descriptors'],
        structural_dataset=dict(identity=p['identity'],normalization=p['normalization']),fixed_dataset=c['fixed_dataset'])
    write_json(root(c)/'consumer.json',consumer)
    from src.research.structural_response.data import spatial_data
    from src.research.liquid_predictability.rich_multimaterial_data import CachedPatches
    data,ids,w=spatial_data(consumer);batch=data.batch([0,p['parent_training_rows']-1,p['parent_training_rows'],n-1])
    if not np.isfinite(batch['target']).all() or not np.isclose(w.sum(),1):raise ValueError('Published training consumer failed')
    data.close()
    for role in sorted({s['task']['role'] for s in m['shards']} - {'train'}):
        held=CachedPatches(consumer,role);held.batch([0,len(held)-1]);held.close()
    complete=dict(state='complete',identity=p['identity'],counts=p['counts'],ni_training_fraction=p['ni_fraction'],
        packed_identity=packed['identity'],columns=len(ids),active_columns=int(active.sum()),
        completed_at=datetime.now(timezone.utc).isoformat(),consumer=str(root(c)/'consumer.json'))
    from src.experiment_runner.metric_docs import write_metric_rows
    write_metric_rows([dict(role=k.split('/')[0],material=k.split('/')[1],rows=v) for k,v in p['counts'].items()],
        resolve_path(c['output']),family=c.get('metric_family','spatial_ni_dataset'),name='population')
    write_json(root(c)/'complete.json',complete);write_json(root(c)/'status.json',complete)
    write_json(resolve_path(c['output'])/'technical/complete.json',complete);print(json.dumps(complete),flush=True)


def submit(c):
    from src.experiment_runner.execution import ExecutionBundle,SlurmQueue
    tech=resolve_path(c['output'])/'technical';p=plan(c)
    if (tech/'launch.json').exists():raise ValueError('Already launched; use the recorded frozen execution to resume')
    audit=read(tech/'preflight.json')
    if audit['identity']!=p['identity'] or not audit['reused']:
        raise ValueError('Require a matching completed Ni shard and reference-descriptor audit before submission')
    from src.experiment_runner.metric_docs import check_metric_docs
    check_metric_docs(family='spatial_ni_dataset')
    bundle=ExecutionBundle.freeze(REPO,tech/'code',c,directories=('src','docs/metrics','configs/data'),files=())
    env=dict(PCM_PROJECT_ROOT=str(REPO),PYTHONNOUSERSITE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
        NUMBA_CACHE_DIR=str(root(c)/'compiled'))
    receipt=dict(owner='Current nuclei/target preparation chat',identity=p['identity'],code=str(bundle.root),jobs={},
        submitted_at=datetime.now(timezone.utc).isoformat())
    queue=SlurmQueue(tech,bundle,'src.postprocessing.spatial_dataset',env,tech/'launch.json',receipt,'SPATIAL-NI20')
    with queue.submission():
        queue.submit('run',[f'--cpus-per-task={c["workers"]}','--hint=nomultithread','--mem=20G',
            '--time=02:00:00','--nodelist='+c['cpu_node']])
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('plan','prepare','publish','run','submit'))
    parser.add_argument('--config',required=True);args=parser.parse_args();c=read(resolve_path(args.config))
    if args.stage=='plan':
        p=plan(c);print(json.dumps(dict(identity=p['identity'],counts=p['counts'],ni_fraction=p['ni_fraction'])))
    elif args.stage=='submit':print(json.dumps(submit(c)))
    else:
        if args.stage in ('prepare','run'):prepare(c)
        if args.stage in ('publish','run'):publish(c)


if __name__=='__main__':
    from src.postprocessing.spatial_dataset import main as entry
    entry()
