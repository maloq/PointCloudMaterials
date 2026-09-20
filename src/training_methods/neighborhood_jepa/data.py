"""Memory-mapped complete local graphs, mixed source groups and prefetched CPU batches."""
import json
from collections import OrderedDict
from pathlib import Path
from functools import partial
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader

# A 21-view prefetched batch contains hundreds of tensors; descriptor-per-storage
# transport exhausts the default 1024-FD limit before the bounded mmap cache does.
torch.multiprocessing.set_sharing_strategy('file_system')
from src.data.structural_pretraining.batches import collate
from src.data.structural_pretraining.support import REFERENCE_RADIUS
from src.training_methods.shared_pretraining.mixed import quotas


class NeighborhoodData(Dataset):
    def __init__(self,root,neighbors=6):
        self.root=Path(root);self.manifest=json.loads((self.root/'manifest.json').read_text());self.neighbors=neighbors
        if self.manifest['state']!='complete':raise ValueError('Incomplete neighborhood release')
        self.rows=[];self.groups=[[] for _ in self.manifest['groups']];self.selection=[];self.arrays=OrderedDict()
        for record in self.manifest['shards']:
            for row in range(record['anchors']):
                index=len(self.rows);self.rows.append((record,row))
                if record['split']=='train':self.groups[record['group']].append(index)
                elif record['split']=='selection':self.selection.append(index)
                else:raise ValueError(record['split'])
        self.train_size=sum(map(len,self.groups))
        self.baselines={}
        for name,width in [('physical',85),('tda',144)]:
            total=np.zeros((len(self.groups),width),np.float64);count=np.zeros(len(self.groups),np.int64)
            for record in self.manifest['shards']:
                if record['split']!='train':continue
                values=np.load(self.root/'shards'/record['id']/f'{name}.npy').reshape(-1,width)
                total[record['group']]+=values.sum(0,dtype=np.float64);count[record['group']]+=len(values)
            norm=self.manifest['normalization'][name]
            self.baselines[name]=((total/count[:,None]-np.array(norm['mean']))/np.array(norm['std'])).astype(np.float32)

    def __getstate__(self):
        state=dict(self.__dict__);state['arrays']=OrderedDict();return state

    def __len__(self):return len(self.rows)

    def __getitem__(self,index):
        record,row=self.rows[index];sid=record['id']
        if sid not in self.arrays:
            self.arrays[sid]={p.stem:np.load(p,mmap_mode='r') for p in (self.root/'shards'/sid).glob('*.npy')}
            if len(self.arrays)>16:self.arrays.popitem(last=False)
        self.arrays.move_to_end(sid);a=self.arrays[sid];views=[];bonds=[]
        for t in range(3):
            for k in range(self.neighbors+1):
                view=a['views'][row,t,k];lo,hi=a['offsets'][view:view+2];elo,ehi=a['edge_offsets'][view:view+2]
                x=np.array(a['positions'][lo:hi]);w=np.array(a['weights'][lo:hi])
                views.append(dict(positions=x[None],weights=w[None],center=0,times=np.array([0.],np.float32),
                    species=record['species'],log_scale=np.log(record['scale']/REFERENCE_RADIUS),
                    edges=np.array(a['edges'][:,elo:ehi],dtype=np.int64),
                    physical=np.zeros(85,np.float32),tda=np.zeros(144,np.float32),tda_valid=False))
                if t in (1,2) and k==0:
                    distance=np.square(x).sum(-1);ids=np.flatnonzero((distance>0)&(w>0));order=ids[np.argsort(distance[ids],kind='stable')[:12]]
                    if len(order)!=12:raise ValueError(f'Insufficient bonds in {record["id"]}/{row}')
                    bonds.append(x[order])
        return dict(views=views,position=np.array(a['query_positions'][row,:self.neighbors+1]),times=np.array(a['times']),
            physical=np.array(a['physical'][row]),tda=np.array(a['tda'][row]),bonds=np.stack(bonds),
            group=record['group'],source=record['source'],index=index)


def pack(samples,microbatch):
    views=[v for sample in samples for v in sample['views']]
    batches=[collate(views[i:i+microbatch],'mace') for i in range(0,len(views),microbatch)]
    target={name:torch.from_numpy(np.stack([s[name] for s in samples])) for name in ['position','times','physical','tda','bonds']}
    target['group']=torch.tensor([s['group'] for s in samples]);target['index']=torch.tensor([s['index'] for s in samples])
    return batches,target


class MixedBatches:
    def __init__(self,data,batch,seed,start,stop):self.data=data;self.batch=batch;self.seed=seed;self.start=start;self.stop=stop
    def __iter__(self):
        counts=quotas([len(g) for g in self.data.groups],self.batch,32)
        for step in range(self.start,self.stop):
            rng=np.random.default_rng(np.random.SeedSequence([self.seed,step]));indices=[]
            for group,count in zip(self.data.groups,counts):indices.extend(rng.choice(group,int(count),replace=False).tolist())
            rng.shuffle(indices);yield indices
    def __len__(self):return self.stop-self.start


def loader(data,sampler,microbatch,workers=4):
    return DataLoader(data,batch_sampler=sampler,collate_fn=partial(pack,microbatch=microbatch),num_workers=workers,
        pin_memory=True,persistent_workers=workers>0,generator=torch.Generator().manual_seed(731),**({'prefetch_factor':2,'multiprocessing_context':'spawn'} if workers else {}))
