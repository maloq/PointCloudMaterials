"""Order anchors and a frozen random geometric reservoir on the existing Al release."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json,file_hash,digest
from src.data.structural_pretraining.batches import collate,move
from src.data.structural_pretraining.support import REFERENCE_RADIUS
from src.analysis.liquid_structure import bond_order,ORDER_NAMES
from ..v2.data import Data as BaseData,pack as base_pack
from ..v2.contracts import variants
from ..v2.model import Encoder
from src.training_methods.shared_pretraining.compilation import compile_encoder


def order_targets(positions,scale):
    """Same eight definitions as the baseline, computed only inside observed crop."""
    x=np.asarray(positions,dtype=np.float64)*(scale/REFERENCE_RADIUS)
    first=np.argsort(np.sum(x*x,axis=1),kind='stable')[:13]
    if len(first)!=13 or first[0]!=0:raise ValueError('Order target needs center first and 12 neighbors')
    delta=x[None]-x[first,None]
    nearest=np.argsort(np.sum(delta*delta,axis=-1),axis=1,kind='stable')[:,1:13]
    bonds=np.take_along_axis(delta,nearest[...,None],axis=1)
    value,_=bond_order(bonds[None],3.5)
    if not np.isfinite(value).all():raise FloatingPointError('Nonfinite order target')
    return value[0]


def prepare_shard(item):
    parent,out,record=item
    folder=Path(parent)/'shards'/record['id']
    a={k:np.load(folder/f'{k}.npy',mmap_mode='r') for k in ('views','offsets','positions')}
    values=[]
    for row in range(record['anchors']):
        pair=[]
        for t in (1,2):
            v=a['views'][row,t,0];lo,hi=a['offsets'][v:v+2]
            pair.append(order_targets(a['positions'][lo:hi],record['scale']))
        values.append(pair)
    result=np.asarray(values,np.float32);path=Path(out)/'order'/f'{record["id"]}.npy'
    np.save(path,result)
    return dict(id=record['id'],split=record['split'],sha256=file_hash(path))


def prepare(config):
    root=resolve_path(config['order_cache']);base=resolve_path(config['cache'])
    root.mkdir(parents=True,exist_ok=True);(root/'order').mkdir(exist_ok=True)
    manifest=json.loads((base/'manifest.json').read_text())
    identity=digest(dict(base=manifest['identity'],producer=file_hash(Path(__file__)),
                        bond_order=file_hash(Path('src/analysis/liquid_structure.py'))))
    path=root/'manifest.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['identity']!=identity:raise ValueError('Order cache producer changed')
        if old['state']=='complete':return old
    with ProcessPoolExecutor(max_workers=6) as pool:
        receipts=list(pool.map(prepare_shard,[(manifest['parent'],str(root),r) for r in manifest['shards']]))
    train=np.concatenate([np.load(root/'order'/f'{r["id"]}.npy').reshape(-1,8) for r in receipts if r['split']=='train']).astype(np.float64)
    result=dict(state='complete',identity=identity,base_identity=manifest['identity'],names=ORDER_NAMES,
        mean=train.mean(0).tolist(),std=train.std(0).clip(1e-4).tolist(),shards=receipts,
        support='Only encoder-observed local crop; center + its 12 nearest neighbors, their observed 12 bonds',
        parent=str(base),materials=['Al'],potential='al-lee2003-meam')
    save_json(path,result);return result


class Data(BaseData):
    def __init__(self,config,spec):
        super().__init__(resolve_path(config['cache']),spec)
        self.extra=resolve_path(config['order_cache'])
        self.order_manifest=json.loads((self.extra/'manifest.json').read_text())
        if self.order_manifest['base_identity']!=self.manifest['identity']:raise ValueError('Order/base identity mismatch')
        self.order_arrays={r['id']:np.load(self.extra/'order'/f'{r["id"]}.npy') for r in self.manifest['shards']}
        self.reservoir_arrays={}
        if spec['regularizer']=='epi':
            receipt=json.loads((self.extra/'reservoir.json').read_text())
            if receipt['state']!='complete' or receipt['base_identity']!=self.manifest['identity']:raise ValueError('Reservoir incomplete or mismatched')
            self.reservoir_arrays={r['id']:np.load(self.extra/'reservoir'/f'{r["id"]}.npy') for r in self.manifest['shards']}
    def __getitem__(self,index):
        sample=super().__getitem__(index);r,j=self.rows[index]
        sample['order']=self.order_arrays[r['id']][j]
        if self.reservoir_arrays:sample['reservoir']=self.reservoir_arrays[r['id']][j]
        return sample


def pack(samples,microbatch):
    batches,target=base_pack(samples,microbatch)
    target['order']=torch.from_numpy(np.stack([s['order'] for s in samples]))
    if 'reservoir' in samples[0]:target['reservoir']=torch.from_numpy(np.stack([s['reservoir'] for s in samples]))
    return batches,target


def worker_init(_):
    torch.set_num_threads(1);torch.multiprocessing.set_sharing_strategy('file_system')


def loader(data,sampler,microbatch,workers=0):
    if workers:torch.multiprocessing.set_sharing_strategy('file_system')
    return DataLoader(data,batch_sampler=sampler,collate_fn=partial(pack,microbatch=microbatch),
        num_workers=workers,pin_memory=True,persistent_workers=workers>0,worker_init_fn=worker_init,
        generator=torch.Generator().manual_seed(731),
        **({'prefetch_factor':2,'multiprocessing_context':'spawn'} if workers else {}))


@torch.no_grad()
def reservoir(config):
    torch.set_num_threads(1);torch.manual_seed(9173)
    root=resolve_path(config['order_cache']);(root/'reservoir').mkdir(exist_ok=True)
    spec=variants({'seed':1})[0];data=BaseData(resolve_path(config['cache']),spec)
    encoder=Encoder(channels=16).cuda().eval()
    projection=torch.linalg.qr(torch.randn(128,64,device='cuda'),mode='reduced').Q
    initialized=False
    for record in data.manifest['shards']:
        path=root/'reservoir'/f'{record["id"]}.npy'
        rows=[i for i,(r,_) in enumerate(data.rows) if r['id']==record['id']]
        values=[]
        for start in range(0,len(rows),64):
            packed,_=base_pack([data[i] for i in rows[start:start+64]],128)
            batch=move(packed[0],'cuda')
            if not initialized:compile_encoder(encoder,batch,'bf16');initialized=True
            with torch.autocast('cuda',dtype=torch.bfloat16):z=encoder(batch).float()[:,:128]
            values.append((z@projection).reshape(-1,2,64).cpu().numpy())
        np.save(path,np.concatenate(values))
        print(json.dumps(dict(shard=record['id'],anchors=len(rows))),flush=True)
    torch.save(dict(encoder=encoder.state_dict(),projection=projection.cpu(),seed=9173),root/'reservoir.pt')
    save_json(root/'reservoir.json',dict(state='complete',base_identity=data.manifest['identity'],
        checkpoint_sha256=file_hash(root/'reservoir.pt'),seed=9173,description='Frozen randomly initialized width16 MACE invariant state, fixed orthogonal 128-to-64 projection; no labels or learned parent weights',
        files={r['id']:file_hash(root/'reservoir'/f'{r["id"]}.npy') for r in data.manifest['shards']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['orders','reservoir']);p.add_argument('--config',required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());torch.set_num_threads(1)
    {'orders':prepare,'reservoir':reservoir}[a.phase](c)
