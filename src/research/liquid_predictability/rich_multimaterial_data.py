"""Raw multi-material patch descriptors, immutable ancestry and bounded streaming."""
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.structural_pretraining.native_dataset import NativeStructuralDataset
from src.project_runtime.paths import resolve_path
from src.experiment_runner.artifacts import implementation_hashes
from .data import config
from .descriptors import patch_descriptors


def plan(c):
    spec=c['structural_dataset'];source_root=resolve_path(spec['root'])
    original=NativeStructuralDataset(spec['root'],normalization=spec['normalization'])
    if original.identity!=spec['identity']:raise ValueError('Structural release identity changed')
    manifest=config(source_root/'manifest.json');sources={s['id']:s for s in original.plan['sources']}
    tasks=[]
    for s in manifest['shards']:
        t=s['task'];source=sources[t['source']]
        if not t['observed'] or source['kind']!='dynamic':continue
        tasks.append(dict(id=t['id'],source=t['source'],lineage=s['lineage'],material=s['material'],
            potential=s['potential'],role=t['split'],rows=s['rows'],factor=float(original.factors[s['material']]),
            input=f"{spec['root']}/shards/{t['id']}/hot.npy",input_sha256=s['files']['hot.npy'],
            original_task=t,original_files=s['files']))
    fixed_root,fixed=read_release(c['fixed_dataset']['root'])
    if fixed['identity']!=c['fixed_dataset']['identity']:raise ValueError('Fixed Al release changed')
    fm=config(fixed_root/'benchmark/manifest.json')
    if sha(fixed_root/'benchmark/population.npz')!=fm['population_sha256']:raise ValueError('Fixed rows changed')
    with np.load(fixed_root/'benchmark/population.npz') as a:pop={k:a[k] for k in ('source','role','patch_index','sample_id')}
    fixed_sources={s['id']:s for s in fixed['sources']}
    for s in fm['sources']:
        if s['role'] not in c['evaluation']['fixed_roles']:continue
        ids=np.flatnonzero(pop['source']==s['id'])
        tasks.append(dict(id=f"fixed-{s['id']}",source=str(s['id']),lineage=fixed_sources[s['id']]['lineage'],
            material='Al',potential='al-lee2003-meam',role=s['role'],rows=len(ids),factor=1.,
            input=f"{c['fixed_dataset']['root']}/benchmark/sources/{s['id']}/hot.npy",input_sha256=s['files']['hot.npy'],
            row_indices=pop['patch_index'][ids].tolist(),fixed_global_rows=ids.tolist(),sample_ids=pop['sample_id'][ids].tolist()))
    fit_lineages={t['lineage'] for t in tasks if t['role']=='train'}
    heldout={t['lineage'] for t in tasks if t['role']!='train'}
    if fit_lineages&heldout:raise ValueError(f'Train/held-out ancestry overlap: {fit_lineages&heldout}')
    counts=Counter()
    for t in tasks:counts[t['role']+'/'+t['material']]+=t['rows']
    sample=np.load(resolve_path(tasks[0]['input']),mmap_mode='r')[0]*tasks[0]['factor']
    values,names=patch_descriptors(sample)
    if len(values)!=442:raise ValueError('Expected original 442 local rich descriptors')
    columns=[dict(name=n,family=n.split('/')[0]) for n in names]
    binding=dict(protocol=c['protocol'],source_manifest_sha256=sha(source_root/'manifest.json'),
        structural_identity=spec['identity'],normalization=spec['normalization'],fixed_identity=fixed['identity'],
        fixed_population_sha256=fm['population_sha256'],
        implementation={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('descriptors.py'))},
        execution=implementation_hashes('src/experiment_runner/preparation.py',
                                        'src/experiment_runner/artifacts.py'),
        tasks=tasks,columns=columns,counts=dict(counts),sampling='all raw dynamic observed rows; no phase labels')
    binding['identity']=digest(binding)
    root=resolve_path(c['cache']);root.mkdir(parents=True,exist_ok=True)
    target=root/'plan.json'
    if target.exists() and config(target)!=binding:raise ValueError('Multimaterial descriptor plan changed; use a new release')
    write_json(target,binding);original.close()
    return binding


def prepare_task(args):
    from src.experiment_runner.preparation import PreparationShard
    c,identity,columns,task=args;root=resolve_path(c['cache'])/'shards'/task['id'];root.mkdir(parents=True,exist_ok=True)
    shard = PreparationShard(root, identity)
    if shard.verified(task=task) is not None:
        return dict(task=task['id'],rows=task['rows'],reused=True)
    path=resolve_path(task['input'])
    if sha(path)!=task['input_sha256']:raise ValueError(f'Changed raw input {path}')
    raw=np.load(path,mmap_mode='r').reshape(-1,80,3)
    index=np.asarray(task['row_indices'],np.int64) if 'row_indices' in task else np.arange(task['rows'])
    if len(index)!=task['rows'] or (len(index) and index.max()>=len(raw)):raise ValueError(f'Invalid input rows: {task["id"]}')
    start = shard.offset('rows_done', total=len(index))
    mode='r+' if shard.resuming else 'w+'
    positions=np.lib.format.open_memmap(root/'positions.npy',mode=mode,dtype=np.float32,shape=(len(index),80,3))
    features=np.lib.format.open_memmap(root/'features.npy',mode=mode,dtype=np.float32,shape=(len(index),len(columns)))
    names=[v['name'] for v in columns];started=time.monotonic()
    shard.progress(rows_done=start)
    for begin in range(start,len(index),c['preparation']['checkpoint_rows']):
        stop=min(begin+c['preparation']['checkpoint_rows'],len(index))
        x=np.asarray(raw[index[begin:stop]],np.float32)*np.float32(task['factor'])
        for j,points in enumerate(x):
            try:value,current=patch_descriptors(points)
            except Exception as e:raise RuntimeError(f'Descriptor failure: {task["id"]}, input row {index[begin+j]}, {task["material"]}') from e
            if current!=names:raise ValueError('Descriptor columns changed')
            features[begin+j]=value
        positions[begin:stop]=x;positions.flush();features.flush()
        shard.progress(rows_done=stop)
    a=np.asarray(features,dtype=np.float64)
    np.savez(root/'moments.npz',total=a.sum(0),second=(a*a).sum(0))
    del a,features,positions,raw
    saved = shard.complete(('positions.npy', 'features.npy', 'moments.npz'),
                           task=task, seconds=time.monotonic()-started)
    return dict(task=task['id'],rows=len(index),seconds=saved['seconds'])


def prepare(c,lane):
    root=resolve_path(c['cache']);p=config(root/'plan.json')
    tasks=sorted(p['tasks'],key=lambda t:(-t['rows'],t['id']))[lane::c['preparation']['lanes']]
    # One shard per process task; each process caches the CNA JIT and Wigner coefficients.
    with ProcessPoolExecutor(max_workers=c['preparation']['workers']) as pool:
        done=rows=0
        for value in pool.map(prepare_task,((c,p['identity'],p['columns'],t) for t in tasks)):
            done+=1;rows+=value['rows']
            if done%32==0 or done==len(tasks):
                status=dict(lane=lane,shards_done=done,shards_total=len(tasks),rows_done=rows)
                write_json(root/f'prepare-{lane}.json',status);print(json.dumps(status),flush=True)


def seal(c):
    root=resolve_path(c['cache']);p=config(root/'plan.json');records=[]
    total=np.zeros(len(p['columns']));second=total.copy();n=0
    for t in p['tasks']:
        folder=root/'shards'/t['id'];r=config(folder/'complete.json')
        if r['identity']!=p['identity'] or r['task']!=t:raise ValueError(f'Incomplete or changed task {t["id"]}')
        for name,h in r['files'].items():
            if sha(folder/name)!=h:raise ValueError(f'Changed derived file {folder/name}')
        if t['role']=='train':
            with np.load(folder/'moments.npz') as a:total+=a['total'];second+=a['second']
            n+=t['rows']
        records.append(dict(task=t,files=r['files']))
    mean=total/n;std=np.sqrt(np.maximum(second/n-mean**2,0));active=std>=1e-4
    np.savez(root/'standardization.npz',mean=mean,scale=std.clip(1e-4),active=active,rows=n)
    result=dict(state='complete',identity=p['identity'],plan_sha256=sha(root/'plan.json'),
        standardization_sha256=sha(root/'standardization.npz'),counts=p['counts'],shards=records,columns=p['columns'])
    write_json(root/'manifest.json',result)
    print(json.dumps(dict(state='complete',identity=p['identity'],counts=p['counts'])),flush=True)


class CachedPatches:
    """Each process keeps a bounded mmap cache; no material/time fields reach the model."""
    epoch=NativeStructuralDataset.epoch

    def __init__(self,c,role):
        self.root=resolve_path(c['cache']);m=config(self.root/'manifest.json')
        if m['state']!='complete' or sha(self.root/'plan.json')!=m['plan_sha256']:raise ValueError('Descriptor release incomplete or changed')
        if sha(self.root/'standardization.npz')!=m['standardization_sha256']:raise ValueError('Target transform changed')
        self.identity=m['identity'];self.columns=m['columns'];self.role=role
        self.shards=[dict(r,rows=r['task']['rows']) for r in m['shards'] if r['task']['role']==role]
        if not self.shards:raise ValueError(f'No rows for {role}')
        self.ends=np.cumsum([s['rows'] for s in self.shards]);self.starts=np.r_[0,self.ends[:-1]]
        with np.load(self.root/'standardization.npz') as a:
            self.mean=a['mean'].astype(np.float32);self.scale=a['scale'].astype(np.float32);self.active=a['active']
        self.max_open=c['loader']['max_open_shards'];self._open=OrderedDict();self._verified=set()
        self.loss_weight=np.zeros(len(self.mean),np.float32)
        families=np.asarray([v['family'] for v in self.columns])
        for f in ('geometry','bond_order','cna','tda'):
            mask=(families==f)&self.active
            if not mask.any():raise ValueError(f'No varying {f} targets')
            self.loss_weight[mask]=1/(4*mask.sum())

    def __len__(self):return int(self.ends[-1])

    def select(self,ids):
        """Freeze one raw training subset; validation/calibration/test are untouched."""
        if self.role!='train' or any('selected_indices' in s for s in self.shards):raise ValueError('Subset must be applied once to the raw training pool')
        ids=np.asarray(ids,np.int64)
        if not len(ids) or (np.diff(ids)<=0).any() or ids[0]<0 or ids[-1]>=len(self):raise ValueError('Require sorted, unique, valid training row IDs')
        self.close();selected=[]
        for i,s in enumerate(self.shards):
            lo,hi=np.searchsorted(ids,[self.starts[i],self.ends[i]])
            if hi>lo:selected.append(dict(s,rows=hi-lo,selected_indices=ids[lo:hi]-self.starts[i]))
        self.shards=selected;self.ends=np.cumsum([s['rows'] for s in selected]);self.starts=np.r_[0,self.ends[:-1]]
        self._verified.clear()
        # The transforms must describe actual fitting rows, not unused train-pool rows.
        total=np.zeros(len(self.mean));second=total.copy()
        for i,s in enumerate(self.shards):
            a=np.asarray(self._arrays(i)['features'][s['selected_indices']],np.float64)
            total+=a.sum(0);second+=(a*a).sum(0)
        self.mean=(total/len(self)).astype(np.float32)
        std=np.sqrt(np.maximum(second/len(self)-(total/len(self))**2,0))
        self.scale=std.clip(1e-4).astype(np.float32);self.active=std>=1e-4
        self.loss_weight.fill(0);families=np.array([v['family'] for v in self.columns])
        for f in ('geometry','bond_order','cna','tda'):
            mask=(families==f)&self.active
            if not mask.any():raise ValueError(f'No varying subset targets in {f}')
            self.loss_weight[mask]=1/(4*mask.sum())

    def use_transform(self,training):
        self.mean=training.mean.copy();self.scale=training.scale.copy()
        self.active=training.active.copy();self.loss_weight=training.loss_weight.copy()

    def _arrays(self,i):
        if i in self._open:
            self._open.move_to_end(i);return self._open[i]
        record=self.shards[i];folder=self.root/'shards'/record['task']['id']
        if i not in self._verified:
            for n in ('positions.npy','features.npy'):
                if sha(folder/n)!=record['files'][n]:raise ValueError(f'Changed training shard {folder/n}')
            self._verified.add(i)
        values={n:np.load(folder/f'{n}.npy',mmap_mode='r') for n in ('positions','features')}
        if len(self._open)==self.max_open:
            _,old=self._open.popitem(last=False)
            for v in old.values():v._mmap.close()
        self._open[i]=values;return values

    def batch(self,ids):
        ids=np.asarray(ids,np.int64)
        if ids.ndim!=1 or not len(ids) or ids.min()<0 or ids.max()>=len(self):raise IndexError('Invalid patch IDs')
        which=np.searchsorted(self.ends,ids,side='right')
        positions=np.empty((len(ids),80,3),np.float32);target=np.empty((len(ids),len(self.mean)),np.float32)
        for i in np.unique(which):
            take=np.flatnonzero(which==i);local=ids[take]-self.starts[i];arrays=self._arrays(int(i))
            if 'selected_indices' in self.shards[i]:local=self.shards[i]['selected_indices'][local]
            positions[take]=arrays['positions'][local];target[take]=arrays['features'][local]
        target=(target-self.mean)/self.scale
        if not np.isfinite(target).all():raise FloatingPointError('Nonfinite standardized descriptors')
        return dict(positions=positions,target=target)

    def audit_ids(self,per_material,seed):
        result={};rng=np.random.default_rng(seed)
        for material in sorted({s['task']['material'] for s in self.shards}):
            shards=np.array([i for i,s in enumerate(self.shards) if s['task']['material']==material])
            ends=np.cumsum([self.shards[i]['rows'] for i in shards]);starts=np.r_[0,ends[:-1]]
            draw=np.sort(rng.choice(ends[-1],min(per_material,ends[-1]),replace=False))
            local=np.searchsorted(ends,draw,side='right')
            result[material]=self.starts[shards[local]]+draw-starts[local]
        return result

    def close(self):
        for values in self._open.values():
            for value in values.values():value._mmap.close()
        self._open.clear()
