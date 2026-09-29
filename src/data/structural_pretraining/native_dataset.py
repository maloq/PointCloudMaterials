"""Stream geometry-only observations, using fixed training-calibrated material scales."""
from collections import OrderedDict
import json

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha
from src.project_runtime.paths import resolve_path


class NativeStructuralDataset:
    def __init__(self,root,role='train',paired=False,*,normalization,max_open_shards=32):
        if role not in ('train','selection'):
            raise ValueError('Structural fitting cannot access calibration or test sources')
        self.root=resolve_path(root)
        manifest=json.loads((self.root/'manifest.json').read_text())
        if manifest['state']!='complete' or sha(self.root/'plan.json')!=manifest['plan_sha256']:
            raise ValueError('Structural release incomplete or changed')
        self.plan=json.loads((self.root/'plan.json').read_text())
        self.identity=manifest['identity']
        if digest({k:v for k,v in self.plan.items() if k!='identity'})!=self.identity:
            raise ValueError('Structural release identity mismatch')
        self.normalization=normalization
        self.factors=material_factors(self.plan,normalization)
        self.observation_identity=digest(dict(release=self.identity,normalization=normalization,
            model_inputs=['center-relative normalized coordinates'],atom_channel='constant'))
        if [s['task'] for s in manifest['shards']]!=self.plan['tasks']:
            raise ValueError('Structural manifest omitted or replaced planned rows')
        sources={s['id']:s for s in self.plan['sources']}
        for shard in manifest['shards']:
            task=shard['task'];source=sources[task['source']]
            if (shard['identity']!=self.identity or shard['rows']!=task['stop']-task['start']
                    or shard['atomic_number']!=source['atomic_number']
                    or shard['atomic_number'] not in self.plan['config']['species']):
                raise ValueError(f'Invalid structural shard identity/species/count: {task["id"]}')
        key='paired' if paired else 'observed'
        self.shards=[s for s in manifest['shards'] if s['task']['split']==role and s['task'][key]]
        if not self.shards:raise ValueError(f'Empty structural role/view: {role}/{key}')
        self.ends=np.cumsum([s['rows'] for s in self.shards])
        self.starts=np.r_[0,self.ends[:-1]]
        self.paired=paired; self.role=role; self.max_open_shards=max_open_shards
        self._open=OrderedDict(); self._verified=set()
        self.source_names=sorted({s['task']['source'] for s in self.shards})

    def __len__(self):return int(self.ends[-1])

    def _arrays(self,index):
        if index in self._open:
            self._open.move_to_end(index);return self._open[index]
        shard=self.shards[index];directory=self.root/'shards'/shard['task']['id']
        names=('hot','cold') if self.paired else ('hot',)
        if index not in self._verified:
            for name in names:
                if sha(directory/f'{name}.npy')!=shard['files'][f'{name}.npy']:
                    raise ValueError(f'Changed structural coordinate shard: {directory}/{name}')
            self._verified.add(index)
        values={name:np.load(directory/f'{name}.npy',mmap_mode='r') for name in names}
        if len(self._open)==self.max_open_shards:
            _,old=self._open.popitem(last=False)
            for value in old.values():value._mmap.close()
        self._open[index]=values
        return values

    def batch(self,indices):
        ids=np.asarray(indices,dtype=np.int64)
        if ids.ndim!=1 or not len(ids) or ids.min()<0 or ids.max()>=len(self):
            raise IndexError('Require nonempty valid structural row IDs')
        which=np.searchsorted(self.ends,ids,side='right')
        result={name:np.empty((len(ids),80,3),np.float32) for name in (('hot','cold') if self.paired else ('hot',))}
        for index in np.unique(which):
            rows=np.flatnonzero(which==index);local=ids[rows]-self.starts[index]
            values=self._arrays(int(index));shard=self.shards[index]
            for name,value in values.items():
                result[name][rows]=value[local]*self.factors[shard['material']]
        return result

    def epoch(self,batch_size,epoch,seed,start_batch=0,shards_per_block=16):
        """One exact epoch; shuffled blocks bound random I/O and row-index memory."""
        rng=np.random.default_rng(np.random.SeedSequence([seed,epoch]))
        order=rng.permutation(len(self.shards));pending=np.empty(0,dtype=np.int64);step=0
        for begin in range(0,len(order),shards_per_block):
            block=order[begin:begin+shards_per_block]
            ids=np.concatenate([np.arange(self.starts[i],self.ends[i]) for i in block])
            rng.shuffle(ids);ids=np.r_[pending,ids]
            stop=len(ids)//batch_size*batch_size
            for offset in range(0,stop,batch_size):
                if step>=start_batch:yield step,ids[offset:offset+batch_size]
                step+=1
            pending=ids[stop:]
        if len(pending) and step>=start_batch:yield step,pending

    def close(self):
        for values in self._open.values():
            for value in values.values():value._mmap.close()
        self._open.clear()


def material_factors(plan,normalization):
    """Reuse the earlier fixed length normalization, anchored to native Al units.

    Material and potential remain provenance/preprocessing metadata. Neither
    their IDs nor the scale are returned to the encoder or its decoder.
    """
    if normalization['protocol']!='fixed_material_cutoff_al_reference_v1':
        raise ValueError('Require the declared geometry-only material normalization')
    catalog_path=resolve_path(plan['config']['source_catalog'])
    expected=plan['evidence']['catalog_sha256']
    if sha(catalog_path)!=expected or normalization['catalog_sha256']!=expected:
        raise ValueError('Material calibration source changed')
    catalog=json.loads(catalog_path.read_text())
    scales=normalization['scales_A']
    if scales!=catalog['scales'] or normalization['reference_scale_A']!=scales['Al']:
        raise ValueError('Material scales must match the recorded train-only calibration, with Al reference')
    if any(not np.isfinite(s) or s<=0 for s in scales.values()):
        raise ValueError('Material length scales must be finite and positive')
    sources={s['id']:s for s in catalog['sources']}
    excluded={s['lineage'] for s in plan['sources'] if s['split']=='selection'}
    excluded.update(s['lineage'] for s in plan['excluded_fixed_sources'])
    for material,rows in catalog['calibration'].items():
        for row in rows:
            source=sources[row['source']]
            if source['split']!='train' or source['lineage'] in excluded or source['material']!=material:
                raise ValueError(f'Nontraining material-scale calibration ancestor: {row["source"]}')
    return {material:np.float32(scales['Al']/scale) for material,scale in scales.items()}
