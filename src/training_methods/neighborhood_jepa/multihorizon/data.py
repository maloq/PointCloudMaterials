"""Tracked future-center graphs and fixed anchors from existing raw Al trajectories."""
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import partial
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import torch
from torch.utils.data import Dataset,DataLoader
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import source_arrays,chart,offsets,geometry_packet,file_hash,digest,save_json
from src.data.structural_pretraining.support import REFERENCE_RADIUS,OUTER_RADIUS,EDGE_CUTOFF,support_weights
from src.analysis.liquid_structure import persistence_image
from ..v2.contracts import RequiredViewPlan
from ..v2.data import fetch_by_shard
from ..v2.geometry import moments
from ..regularization.data import Data as OriginalData,pack as original_pack,worker_init


def view_plan(spec):
    base=RequiredViewPlan.from_spec(spec)
    return RequiredViewPlan(base.views+((3,0),(4,0),(5,0)),base.query_list)


def future_frames(timesteps,frame,timestep_fs,horizons):
    times=np.asarray(timesteps,dtype=np.float64)*timestep_fs/1000
    wanted=times[frame]+np.asarray(horizons)
    found=np.searchsorted(times,wanted-1e-6)
    valid=found<len(times)
    if not np.allclose(times[found[valid]],wanted[valid],atol=1e-6,rtol=0):
        raise ValueError(f'Requested physical horizons are absent: frame={frame}, targets={wanted}, indices={found}')
    return np.where(valid,found,-1),valid


def prepare_shard(args):
    root,parent,identity,record,source,horizons=args
    torch.set_num_threads(1)
    folder=Path(root)/'shards'/record['id'];receipt=folder/'complete.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        if saved['identity']!=identity:raise ValueError('Multihorizon shard identity changed')
        for name,sha in saved['hashes'].items():
            if file_hash(folder/f'{name}.npy')!=sha:raise ValueError(f'Corrupt future shard {folder}/{name}')
        return saved
    raw=source_arrays(source)
    atom_ids=np.load(Path(parent)/'shards'/record['id']/'query_atom_ids.npy')[:,0]
    lookup={int(v):i for i,v in enumerate(raw['atom_ids'])}
    centers=np.array([lookup[int(v)] for v in atom_ids])
    frames,available=future_frames(raw['timesteps'],record['frame'],source['timestep_fs'],horizons)
    n=len(centers);factor=REFERENCE_RADIUS/record['scale'];radius=OUTER_RADIUS/factor
    positions=[];edges=[];ptr=[0];eptr=[0];views=np.full((n,3),-1,np.int64)
    physical=np.zeros((n,3,85),np.float32);tda=np.zeros((n,3,144),np.float32)
    for h,frame in enumerate(frames):
        if frame<0:continue
        x,tree,box=chart(raw,int(frame),False)
        for row,center in enumerate(centers):
            ids=np.array(sorted(tree.query_ball_point(x[center],radius)),np.int64)
            ids=np.r_[center,ids[ids!=center]]
            local=offsets(x,center,ids,box)
            near=np.lexsort((raw['atom_ids'][ids],np.square(local.astype(np.float64)).sum(-1)))[:80]
            if len(near)!=80:raise ValueError(f'Incomplete future local support {record["id"]}/{row}/{h}')
            physical[row,h]=geometry_packet(local)
            tda[row,h]=persistence_image(local[near])
            normalized=(local*factor).astype(np.float32)
            normalized=normalized[np.linalg.norm(normalized,axis=-1)<OUTER_RADIUS]
            pairs=cKDTree(normalized).query_pairs(EDGE_CUTOFF,output_type='ndarray')
            edge=np.concatenate((pairs,pairs[:,::-1]),0).T.astype(np.int32)
            views[row,h]=len(positions);positions.append(normalized);edges.append(edge)
            ptr.append(ptr[-1]+len(normalized));eptr.append(eptr[-1]+edge.shape[1])
    x=np.concatenate(positions);graph=np.repeat(np.arange(len(positions)),np.diff(ptr))
    fixed=moments(torch.from_numpy(x),torch.from_numpy(graph),len(positions)).numpy()
    values=dict(positions=x,weights=support_weights(x),edges=np.concatenate(edges,1),
        offsets=np.array(ptr,np.int64),edge_offsets=np.array(eptr,np.int64),views=views,
        moments=fixed,physical=physical,tda=tda,valid=np.broadcast_to(available,(n,3)).copy(),
        center_atom_ids=atom_ids,frames=frames,times=np.array(horizons,np.float32))
    folder.mkdir(parents=True,exist_ok=True)
    for name,value in values.items():np.save(folder/f'{name}.npy',value,allow_pickle=False)
    result=dict(identity=identity,id=record['id'],anchors=n,split=record['split'],
        valid_counts=values['valid'].sum(0).tolist(),source_manifest_sha256=source['manifest_sha256'],
        hashes={name:file_hash(folder/f'{name}.npy') for name in values})
    save_json(receipt,result);return result


def prepare(config):
    base=resolve_path(config['cache']);manifest=json.loads((base/'manifest.json').read_text())
    parent=Path(manifest['parent']);plan=json.loads((parent/'plan.json').read_text())
    sources={t['source']['id']:t['source'] for t in plan['tasks']}
    root=resolve_path(config['future_cache']);root.mkdir(parents=True,exist_ok=True)
    producers={str(p):file_hash(p) for p in (Path(__file__),Path('src/data/structural_pretraining/prepare.py'),Path('src/analysis/liquid_structure.py'),Path('src/training_methods/neighborhood_jepa/v2/geometry.py'))}
    identity=digest(dict(base=manifest['identity'],horizons=config['horizons_ps'],producers=producers))
    receipts=[]
    with ProcessPoolExecutor(max_workers=config['prepare_workers']) as pool:
        pending=[pool.submit(prepare_shard,(str(root),str(parent),identity,r,sources[r['source']],config['horizons_ps'])) for r in manifest['shards']]
        for f in as_completed(pending):
            receipts.append(f.result());save_json(root/'status.json',dict(state='preparing',complete=len(receipts),total=len(pending)))
    receipts.sort(key=lambda r:r['id'])
    coverage={split:np.sum([r['valid_counts'] for r in receipts if r['split']==split],axis=0).tolist() for split in ('train','selection')}
    result=dict(state='complete',identity=identity,base_identity=manifest['identity'],parent=str(base),
        horizons_ps=config['horizons_ps'],producers=producers,shards=receipts,coverage=coverage,
        missing='Right boundary only; explicit mask, no clamping or fabricated targets',
        input='Current local structure only; future graphs are teacher targets, never predictor inputs',
        materials=['Al'],potential='al-lee2003-meam',train_roots=manifest['train_roots'],selection_roots=manifest['selection_roots'])
    save_json(root/'manifest.json',result);save_json(root/'status.json',dict(state='complete',coverage=coverage))
    print(json.dumps(coverage),flush=True)


class Data(Dataset):
    def __init__(self,config,spec):
        self.base=OriginalData(config,spec)
        for name in ('root','parent','manifest','rows','train','selection','train_size','extra','order_manifest','order_arrays'):
            setattr(self,name,getattr(self.base,name))
        self.plan=view_plan(spec);self.future_root=resolve_path(config['future_cache'])
        self.future_manifest=json.loads((self.future_root/'manifest.json').read_text())
        if self.future_manifest['state']!='complete' or self.future_manifest['base_identity']!=self.manifest['identity']:
            raise ValueError('Future graphs are incomplete or from another parent')
        if self.future_manifest['horizons_ps']!=spec['horizons_ps']:raise ValueError('Forecast horizon mismatch')
        self.future_arrays=OrderedDict()
    def __len__(self):return len(self.base)
    def __getitems__(self,indices):return fetch_by_shard(self,indices)
    def __getstate__(self):
        state=dict(self.__dict__);state['future_arrays']=OrderedDict();return state
    def future_array(self,sid):
        if sid not in self.future_arrays:
            self.future_arrays[sid]={p.stem:np.load(p,mmap_mode='r') for p in (self.future_root/'shards'/sid).glob('*.npy')}
            if len(self.future_arrays)>16:self.future_arrays.popitem(last=False)
        self.future_arrays.move_to_end(sid);return self.future_arrays[sid]
    def __getitem__(self,index):
        sample=self.base[index];record,row=self.rows[index];a=self.future_array(record['id'])
        if a['center_atom_ids'][row]!=sample['query_atom_ids'][0]:raise ValueError('Future center identity changed')
        fixed=[]
        for h in range(3):
            view=a['views'][row,h]
            if view<0:
                # Shape placeholder only; explicit validity removes all teacher losses.
                slot=self.base.plan.slot(1,0)
                sample['views'].append(sample['views'][slot]);fixed.append(sample['moments'][slot]);continue
            lo,hi=a['offsets'][view:view+2];elo,ehi=a['edge_offsets'][view:view+2]
            sample['views'].append(dict(positions=np.array(a['positions'][lo:hi])[None],weights=np.array(a['weights'][lo:hi])[None],
                center=0,times=np.array([0.],np.float32),species=record['species'],log_scale=np.log(record['scale']/REFERENCE_RADIUS),
                edges=np.array(a['edges'][:,elo:ehi],dtype=np.int64),physical=np.zeros(85,np.float32),tda=np.zeros(144,np.float32),tda_valid=False))
            fixed.append(a['moments'][view])
        sample['moments']=np.concatenate((sample['moments'],np.array(fixed)))
        sample.update(future_physical=np.array(a['physical'][row]),future_tda=np.array(a['tda'][row]),future_valid=np.array(a['valid'][row]))
        return sample


def pack(samples,microbatch):
    batches,target=original_pack(samples,microbatch)
    for name in ('future_physical','future_tda','future_valid'):target[name]=torch.from_numpy(np.stack([s[name] for s in samples]))
    return batches,target


def loader(data,sampler,microbatch,workers=0):
    if workers:torch.multiprocessing.set_sharing_strategy('file_system')
    return DataLoader(data,batch_sampler=sampler,collate_fn=partial(pack,microbatch=microbatch),
        num_workers=workers,pin_memory=True,persistent_workers=workers>0,worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(731),**({'prefetch_factor':2,'multiprocessing_context':'spawn'} if workers else {}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    prepare(json.loads(Path(a.config).read_text()))
