"""Memory-mapped source shards, persistent atom histories and homogeneous batches."""
from collections import OrderedDict, defaultdict
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
import torch

from .prepare import ELEMENTS, REFERENCE_RADIUS
from src.data.predictive_memory.targets import taper
from src.models.encoders.structural import ATOMIC_NUMBERS


class Release:
    def __init__(self,root,materials=None,dynamic_only=False):
        self.root=Path(root); self.manifest=json.loads((self.root/'manifest.json').read_text())
        if self.manifest['state']!='complete':raise ValueError('Structural release is incomplete')
        if dynamic_only:
            self.manifest['shards']=[s for s in self.manifest['shards'] if not s['static']]
            self.manifest['identity']=dict(parent=self.manifest['identity'],dynamic_only=True,
                normalization='subset_training_endpoints_v1')
        if materials is not None:
            materials=sorted(set(materials))
            available={s['material'] for s in self.manifest['shards']}
            if not materials or not set(materials)<=available:
                raise ValueError(f'Requested materials {materials}; release contains {sorted(available)}')
            self.manifest['shards']=[s for s in self.manifest['shards'] if s['material'] in materials]
            self.manifest['identity']=dict(parent=self.manifest['identity'],materials=materials,
                normalization='subset_training_endpoints_v1')
        self.rows=[]; self.groups=defaultdict(list); self.selection=[]; self.arrays={}
        self.graphs=OrderedDict(); self.graph_bytes=0; self.max_graph_bytes=2*2**30
        for shard in self.manifest['shards']:
            name=shard['task']['id']; folder=self.root/'shards'/name
            self.arrays[name]={p.stem:np.load(p,mmap_mode='r',allow_pickle=False) for p in folder.glob('*.npy')}
            for row in range(shard['anchors']):
                index=len(self.rows); self.rows.append((name,row,shard))
                if shard['task']['split']=='selection':self.selection.append(index)
                elif shard['task']['split']=='train':
                    self.groups[(shard['material'],shard['potential'],shard['static'])].append(index)
                elif shard['task']['split'] not in ('calibration','test'):
                    raise ValueError(f'Unexpected structural split {shard["task"]["split"]} in {name}; '
                        'expected train, selection, calibration or test')
        self.group_keys=sorted(self.groups,key=str)
        self.group_weights=np.array([len(self.groups[k]) for k in self.group_keys],dtype=float)
        self.group_weights/=self.group_weights.sum()
        if materials is not None or dynamic_only:
            # Same endpoint/label population and std floor as prepare.finalize.
            # Do not retain moments from excluded materials, even though the
            # underlying immutable cache is shared with the broad study.
            parts={'physical':[],'tda':[]}
            for shard in self.manifest['shards']:
                if shard['task']['split']!='train':continue
                arrays=self.arrays[shard['task']['id']]
                views=arrays['views'][:,[2,4] if shard['static'] else [2,3,4]].ravel()
                parts['physical'].append(arrays['physical'][views])
                parts['tda'].append(arrays['tda'][arrays['tda_valid']])
            self.manifest['normalization']={}
            for name,values in parts.items():
                a=np.concatenate(values).astype(np.float64)
                self.manifest['normalization'][name]=dict(mean=a.mean(0).tolist(),
                    std=np.maximum(a.std(0),1e-4).tolist())

    def view(self,name,index,mace):
        key=(name,index,mace)
        if key in self.graphs:
            self.graphs.move_to_end(key);return self.graphs[key]
        a=self.arrays[name]; lo,hi=a['offsets'][index:index+2]
        x=np.array(a['positions'][lo:hi],copy=True)
        ids=np.array(a['atom_ids'][lo:hi],copy=True)
        result=dict(x=x,ids=ids,center=int(a['center_ids'][index]))
        if mace:result['physical_edges']=None # edges are generated after scale conversion
        return result

    def observation(self,index,which,history,mace):
        cache_key=(index,which,history,mace)
        if cache_key in self.graphs:
            self.graphs.move_to_end(cache_key)
            return self.graphs[cache_key]
        name,row,record=self.rows[index]; a=self.arrays[name]; mapping=a['views'][row]
        slot={'anchor':2,'spatial':4,'future':3,'previous':1}[which]
        if which in ('future','previous') and record['static']:raise ValueError('Static structures have no temporal neighbors')
        slots=[0,1,2] if history and not record['static'] and which=='anchor' else [slot]
        if any(mapping[i]<0 for i in slots):raise ValueError('Missing declared temporal observation')
        frames=[self.view(name,int(mapping[i]),mace) for i in slots]
        ids=np.unique(np.concatenate([f['ids'] for f in frames])); t=len(frames); n=len(ids)
        x=np.zeros((t,n,3),dtype=np.float32);w=np.zeros((t,n),dtype=np.float32)
        factor=REFERENCE_RADIUS/record['scale']
        for k,f in enumerate(frames):
            loc=np.searchsorted(ids,f['ids']); local=f['x']*factor
            x[k,loc]=local;w[k,loc]=taper(np.linalg.norm(local,axis=-1),15.,17.)
        center=frames[-1]['center']; loc=np.flatnonzero(ids==center)
        if len(loc)!=1 or not np.all(x[-1,loc[0]]==0):raise ValueError('Lost tracked center identity')
        times=(a['times'][slots] if t>1 else np.array([0.])).astype(np.float32)
        times=times-times[-1]
        target_index=int(mapping[slot])
        result=dict(positions=x,weights=w,center=int(loc[0]),times=times,
            species=ATOMIC_NUMBERS.index(ELEMENTS[record['material']]),log_scale=np.log(record['scale']/REFERENCE_RADIUS),
            physical=np.array(a['physical'][target_index]),tda=np.array(a['tda'][target_index]),
            tda_valid=bool(a['tda_valid'][target_index]),
            key=(name,target_index),factor=factor)
        if mace:
            pairs=cKDTree(x[0]).query_pairs(5.,output_type='ndarray')
            if len(pairs) and np.any(np.linalg.norm(x[0,pairs[:,0]]-x[0,pairs[:,1]],axis=-1)<=0):
                raise ValueError('Coincident atoms in the normalized MACE graph')
            result['edges']=np.concatenate((pairs,pairs[:,::-1]),axis=0).T.astype(np.int64)
        size=sum(a.nbytes for a in result.values() if isinstance(a,np.ndarray)); result['cache_bytes']=size
        while self.graphs and self.graph_bytes+size>self.max_graph_bytes:
            _,removed=self.graphs.popitem(last=False);self.graph_bytes-=removed['cache_bytes']
        self.graphs[cache_key]=result;self.graph_bytes+=size
        return result


def collate(samples,architecture,bond_order=False):
    lengths={len(s['times']) for s in samples}
    if len(lengths)!=1:raise ValueError('One packed batch must have one temporal grid length')
    t=lengths.pop(); b=len(samples); n=max(s['positions'].shape[1] for s in samples)
    x=np.zeros((b,t,n,3),np.float32); w=np.zeros((b,t,n),np.float32)
    species=np.zeros((b,n),np.int64)
    for i,s in enumerate(samples):
        count=s['positions'].shape[1];x[i,:,:count]=s['positions'];w[i,:,:count]=s['weights'];species[i,:count]=s['species']
    batch=dict(positions=torch.from_numpy(x),weights=torch.from_numpy(w),species=torch.from_numpy(species),
        centers=torch.tensor([s['center'] for s in samples]),times=torch.from_numpy(np.stack([s['times'] for s in samples])),
        log_scale=torch.tensor([s['log_scale'] for s in samples],dtype=torch.float32),
        physical=torch.from_numpy(np.stack([s['physical'] for s in samples])),
        tda=torch.from_numpy(np.stack([s['tda'] for s in samples])),
        tda_valid=torch.tensor([s['tda_valid'] for s in samples]))
    if architecture=='mace':
        if t!=1:raise ValueError('This structural MACE run is snapshot-only')
        xs=[];ws=[];zs=[];edges=[];graphs=[];centers=[];bonds=[];offset=0
        for i,s in enumerate(samples):
            p=s['positions'][0]; e=s['edges']+offset
            centers.append(offset+s['center'])
            if bond_order:
                d2=np.square(p-p[s['center']]).sum(-1)
                candidates=np.flatnonzero((d2>0)&(s['weights'][0]>0))
                if len(candidates)<12:
                    raise ValueError('Bond order requires 12 distinct supported neighbors of the tracked center')
                neighbors=candidates[np.argsort(d2[candidates],kind='stable')[:12]]
                bonds.append(neighbors+offset)
            xs.append(p);ws.append(s['weights'][0]);zs.append(np.full(len(p),s['species'],np.int64));edges.append(e)
            graphs.append(np.full(len(p),i,np.int64));offset+=len(p)
        batch.update(packed_positions=torch.from_numpy(np.concatenate(xs)),packed_weights=torch.from_numpy(np.concatenate(ws)),
            packed_species=torch.from_numpy(np.concatenate(zs)),node_graph=torch.from_numpy(np.concatenate(graphs)),
            edges=torch.from_numpy(np.concatenate(edges,axis=1)),packed_centers=torch.tensor(centers))
        if bond_order:
            from .bond_order import bond_order_targets
            vectors=batch['packed_positions'][torch.tensor(np.stack(bonds))]-batch['packed_positions'][batch['packed_centers']][:,None]
            batch['bond_order']=bond_order_targets(vectors)
    elif bond_order:
        raise ValueError('Equivariant bond-order training is implemented for MACE only')
    return batch


def move(batch,device):
    return {k:v.to(device,non_blocking=True) for k,v in batch.items()}
