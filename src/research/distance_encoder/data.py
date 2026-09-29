"""Resident immutable coordinates; learned MACE features are recomputed each update."""
import json
from pathlib import Path
import numpy as np
import torch
from src.data.fixed_cohort.protocol import sha,digest
from src.project_runtime.paths import resolve_path


class ResidentDataset:
    def __init__(self,config,role,device):
        root=resolve_path(config['labels']['root'])
        manifest=json.loads((root/'manifest.json').read_text())
        if manifest['state']!='complete' or sha(root/'plan.json')!=manifest['plan_sha256']:
            raise ValueError('Distance labels incomplete or changed')
        plan=json.loads((root/'plan.json').read_text())
        if digest({k:v for k,v in plan.items() if k!='identity'})!=manifest['identity']:
            raise ValueError('Distance release identity mismatch')
        if plan['structural_identity']!=config['structural_dataset']['identity'] or plan['fixed_identity']!=config['fixed_dataset']['identity']:
            raise ValueError('Wrong structural or source release')
        if role not in ('train','selection'):raise ValueError('Resident fitting may only open train/selection roles')
        shards=[s for s in plan['shards'] if s['task']['split']==role]
        n=sum(s['rows'] for s in shards)
        self.positions=torch.empty((n,80,3),dtype=torch.float32,device=device)
        self.distance=torch.empty(n,dtype=torch.float32,device=device)
        self.weight=torch.empty(n,dtype=torch.float32,device=device)
        self.sources=np.empty(n,np.int32);self.material=np.empty(n,np.int8)
        self.source_names=sorted({s['task']['source'] for s in shards})
        self.material_names=sorted({s['material'] for s in shards})
        counts={m:sum(s['rows'] for s in shards if s['material']==m) for m in self.material_names}
        self.target_counts={m:dict(rows=0,zero=0,within20=0,censored64=0) for m in self.material_names}
        source_counts={s:sum(t['rows'] for t in shards if t['task']['source']==s) for s in self.source_names}
        previous=None;distances=None;offset=0
        for i,shard in enumerate(shards):
            t=shard['task'];key=(t['source'],t['frame']);source=plan['sources'][t['source']]
            folder=Path(plan['structural_root'])/'shards'/t['id']
            for field in ('hot.npy','center_ids.npy'):
                if sha(folder/field)!=shard['files'][field]:raise ValueError(f'Changed structural data: {folder/field}')
            centers=np.load(folder/'center_ids.npy')
            np.testing.assert_array_equal(centers,source['source']['center_ids'][t['start']:t['stop']])
            if key!=previous:
                path=root/'sources'/t['source']/f'{t["frame"]}.npy'
                receipt=manifest['frames'][f'{t["source"]}/{t["frame"]}']
                if sha(path)!=receipt['sha256']:raise ValueError(f'Changed distances: {path}')
                distances=np.load(path);previous=key
            positions=np.load(folder/'hot.npy')*source['factor']
            target=distances[t['start']:t['stop']]*source['factor']
            target_count=self.target_counts[shard['material']]
            target_count['rows']+=len(target);target_count['zero']+=int((target==0).sum())
            target_count['within20']+=int((target<=20).sum());target_count['censored64']+=int((target>=64).sum())
            if positions.shape!=(shard['rows'],80,3) or not np.isfinite(positions).all() or np.isnan(target).any():
                raise ValueError(f'Invalid input/target: {t["id"]}')
            stop=offset+shard['rows'];sl=slice(offset,stop)
            self.positions[sl]=torch.from_numpy(positions).to(device)
            self.distance[sl]=torch.from_numpy(target).to(device)
            # Sampling traverses every row. Loss has equal material mass on
            # train; selection has equal source mass on its frozen Al sources.
            weight=n/(len(counts)*counts[shard['material']]) if role=='train' else n/(len(source_counts)*source_counts[t['source']])
            self.weight[sl]=weight
            self.sources[sl]=self.source_names.index(t['source']);self.material[sl]=self.material_names.index(shard['material'])
            offset=stop
            if i%1000==0:print(json.dumps(dict(stage='resident-coordinates',role=role,shards=i+1,total=len(shards),rows=offset)),flush=True)
        self.identity=manifest['identity'];self.n=n;self.counts=counts;self.role=role

    def batch(self,ids):
        return self.positions[ids],self.distance[ids],self.weight[ids]
