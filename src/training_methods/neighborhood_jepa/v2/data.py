"""Audited native-Al subset of immutable v1 graphs; only required views are loaded."""
import argparse
from collections import OrderedDict
from functools import partial
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import file_hash, digest, save_json
from src.data.structural_pretraining.support import REFERENCE_RADIUS
from src.data.structural_pretraining.batches import collate
from .contracts import RequiredViewPlan, LAYOUT
from .geometry import moments, blocks


def prepare(config):
    root = resolve_path(config['cache'])
    root.mkdir(parents=True,exist_ok=True)
    parent = resolve_path(config['parent_cache'])
    original = json.loads((parent/'manifest.json').read_text())
    plan = json.loads((parent/'plan.json').read_text())
    assay = json.loads(resolve_path(config['crystallization_plan']).read_text())
    conditions = {s['lineage']:s['temperature_K'] for s in assay['sources']}
    tasks = {t['shard']['task']['id']:t for t in plan['tasks']}
    producer = {str(p):file_hash(p) for p in [Path(__file__),Path(__file__).with_name('geometry.py'),Path(__file__).with_name('contracts.py')]}
    identity = digest(dict(parent=file_hash(parent/'manifest.json'), plan=file_hash(parent/'plan.json'),
        producer=producer, layout=LAYOUT.metadata(), protocol='native_al_lee_075ps_v2',
        assay=file_hash(resolve_path(config['crystallization_plan']))))
    path = root/'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['identity'] != identity:
            raise ValueError('V2 dataset identity changed; use a new cache')
        for r in saved['shards']:
            for name,sha in r['hashes'].items():
                if file_hash(parent/'shards'/r['id']/f'{name}.npy') != sha:
                    raise ValueError(f'Corrupt inherited array {r["id"]}/{name}')
            if file_hash(root/'moments'/f'{r["id"]}.npy') != r['moments_sha256']:
                raise ValueError('Corrupt fixed moments')
        return saved
    rows = []
    values = {name:[] for name in ('physical','tda')}
    training_moments = []
    for record in original['shards']:
        task = tasks[record['id']]
        source = task['source']
        if source['stratum'] != 'al_native':
            continue
        if source['material'] != 'Al' or source['potential'] != 'al-lee2003-meam':
            raise ValueError(f'Unexpected primary source: {source}')
        if source['lineage'] not in conditions:
            raise ValueError(f'Missing causal temperature: {source["lineage"]}')
        folder = parent/'shards'/record['id']
        for name,sha in record['hashes'].items():
            if file_hash(folder/f'{name}.npy') != sha:
                raise ValueError(f'Corrupt {record["id"]}/{name}')
        times = np.load(folder/'times.npy')
        np.testing.assert_allclose(times,[-.75,0,.75],atol=1e-6,rtol=0)
        arrays = {name:np.load(folder/f'{name}.npy',mmap_mode='r') for name in ('positions','offsets','views','query_atom_ids')}
        # e3nn CPU implementation uses vectorized chunks of complete snapshots.
        offsets = arrays['offsets']
        result = []
        for start in range(0,len(offsets)-1,128):
            end = min(start+128,len(offsets)-1)
            x = torch.from_numpy(np.array(arrays['positions'][offsets[start]:offsets[end]]))
            graph = torch.repeat_interleave(torch.arange(end-start),torch.tensor(np.diff(offsets[start:end+1])))
            result.append(moments(x,graph,end-start).numpy())
        fixed = np.concatenate(result)
        target_path = root/'moments'/f'{record["id"]}.npy'
        target_path.parent.mkdir(exist_ok=True)
        np.save(target_path,fixed)
        r = dict(record,group=0,temperature_K=conditions[source['lineage']],
            frame=task['shard']['task']['frame'],source_manifest_sha256=source['manifest_sha256'],
            original_dtype='float16 coordinates, reconstructed float32 graphs; precision not recovered',
            moments_sha256=file_hash(target_path))
        rows.append(r)
        if r['split'] == 'train':
            for name in values:
                values[name].append(np.load(folder/f'{name}.npy').reshape(-1,85 if name=='physical' else 144))
            training_moments.append(fixed[arrays['views'][:,1:,0]].reshape(-1,LAYOUT.equivariant_dim))
    train_roots = {r['lineage'] for r in rows if r['split']=='train'}
    selection_roots = {r['lineage'] for r in rows if r['split']=='selection'}
    locked_roots = {s['lineage'] for s in assay['sources'] if s.get('validation_role',s['split']) in ('test','calibration')}
    if train_roots & selection_roots or (train_roots|selection_roots)&locked_roots:
        raise ValueError('Ancestry overlap in primary encoder release')
    normalization = {}
    for name,parts in values.items():
        a = np.concatenate(parts).astype(np.float64)
        normalization[name] = dict(mean=a.mean(0).tolist(),std=np.maximum(a.std(0),1e-4).tolist())
    eq = torch.from_numpy(np.concatenate(training_moments))
    scales = torch.stack([b.square().mean((0,2)).sqrt().clamp_min(.001) for b in blocks(eq)])
    manifest = dict(state='complete',identity=identity,parent=str(parent),producer=producer,
        normalization=normalization,geometry_scales=scales.tolist(),groups=[['Al','al-lee2003-meam']],
        layout=LAYOUT.metadata(),shards=rows,lag_ps=.75,
        train_roots=sorted(train_roots),selection_roots=sorted(selection_roots),
        locked_overlap=[],selection_role='previously used development sources, not untouched test',
        condition='predictor receives known temperature; 400,450,500,510,520 K',
        support='inherited radius8, taper6–8, edge5; no halo; smooth moments <=8',
        exclusions='all shooting/non-native/non-Al sources; static data absent',
        continuity='solid-harmonic moments C2; inherited nearest80 TDA is not asserted continuous')
    save_json(path,manifest)
    return manifest


class Data(Dataset):
    def __init__(self, root, spec, all_views=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root/'manifest.json').read_text())
        self.parent = Path(self.manifest['parent'])
        self.plan = RequiredViewPlan.from_spec(spec,all_views)
        self.rows, self.train, self.selection = [], [], []
        self.arrays = OrderedDict()
        for r in self.manifest['shards']:
            for j in range(r['anchors']):
                index = len(self.rows)
                self.rows.append((r,j))
                (self.train if r['split']=='train' else self.selection).append(index)
        self.train_size = len(self.train)

    def __getstate__(self):
        state = dict(self.__dict__)
        state['arrays'] = OrderedDict()
        return state

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,index):
        record,row = self.rows[index]
        sid = record['id']
        if sid not in self.arrays:
            a = {p.stem:np.load(p,mmap_mode='r') for p in (self.parent/'shards'/sid).glob('*.npy')}
            a['moments'] = np.load(self.root/'moments'/f'{sid}.npy',mmap_mode='r')
            self.arrays[sid] = a
            if len(self.arrays)>16:
                self.arrays.popitem(last=False)
        self.arrays.move_to_end(sid)
        a = self.arrays[sid]
        views, fixed = [], []
        for t,j in self.plan.views:
            view = a['views'][row,t,j]
            lo,hi = a['offsets'][view:view+2]
            elo,ehi = a['edge_offsets'][view:view+2]
            views.append(dict(positions=np.array(a['positions'][lo:hi])[None],
                weights=np.array(a['weights'][lo:hi])[None],center=0,times=np.array([0.],np.float32),
                species=record['species'],log_scale=np.log(record['scale']/REFERENCE_RADIUS),
                edges=np.array(a['edges'][:,elo:ehi],dtype=np.int64),physical=np.zeros(85,np.float32),
                tda=np.zeros(144,np.float32),tda_valid=False))
            fixed.append(a['moments'][view])
        return dict(views=views,moments=np.array(fixed),position=np.array(a['query_positions'][row]),
            times=np.array(a['times']),physical=np.array(a['physical'][row]),tda=np.array(a['tda'][row]),
            query_atom_ids=np.array(a['query_atom_ids'][row]),frame=record['frame'],index=index,
            temperature_K=record['temperature_K'],group=0)


def pack(samples,microbatch):
    views = [v for s in samples for v in s['views']]
    batches = [collate(views[i:i+microbatch],'mace') for i in range(0,len(views),microbatch)]
    target = {name:torch.from_numpy(np.stack([s[name] for s in samples]))
              for name in ('moments','position','times','physical','tda','query_atom_ids')}
    for name in ('frame','index','group'):
        target[name] = torch.tensor([s[name] for s in samples],dtype=torch.long)
    target['temperature_K'] = torch.tensor([s['temperature_K'] for s in samples],dtype=torch.float32)
    return batches,target


class Batches:
    def __init__(self,data,size,seed,start,stop):
        self.data,self.size,self.seed,self.start,self.stop = data,size,seed,start,stop

    def __iter__(self):
        for step in range(self.start,self.stop):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed,step]))
            yield rng.choice(self.data.train,self.size,replace=False).tolist()

    def __len__(self):
        return self.stop-self.start


def loader(data,sampler,microbatch,workers=0):
    if workers:
        torch.multiprocessing.set_sharing_strategy('file_system')
    return DataLoader(data,batch_sampler=sampler,collate_fn=partial(pack,microbatch=microbatch),
        num_workers=workers,pin_memory=True,persistent_workers=workers>0,
        generator=torch.Generator().manual_seed(731),
        **({'prefetch_factor':2,'multiprocessing_context':'spawn'} if workers else {}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = prepare(json.loads(Path(args.config).read_text()))
    print(json.dumps({k:result[k] for k in ('identity','train_roots','selection_roots')}))
