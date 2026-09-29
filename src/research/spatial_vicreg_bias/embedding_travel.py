"""Frozen embedding probes for spatial and feature-neighbor path exploration."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster._kmeans import _labels_inertia_threadpool_limit
from src.data.fixed_cohort.protocol import sha, write_json


def write_asset(path,key,data,variable):
    encoded=json.dumps(data,separators=(',',':'),allow_nan=False).replace('<','\\u003c')
    temporary=path.with_suffix('.building.js')
    temporary.write_text(f'window.{variable}=window.{variable}||{{}};window.{variable}[{json.dumps(key)}]={encoded};\n')
    temporary.replace(path)


def payload(dest):return json.loads((dest/'index.html').read_text().split('const D=',1)[1].split(';\nconst palette',1)[0])
def asset(path):return json.loads(path.read_text().split('=',2)[2].rstrip(';\n'))


def infer(config,source,publication,dataset,lane,lanes):
    import torch
    from omegaconf import OmegaConf
    from src.research.spatial_vicreg_bias.train import PairEncoder
    c=json.loads(Path(config).read_text())
    pc=json.loads(Path(c['pacmap_config']).read_text())
    corr=json.loads(Path(pc['correspondence_config']).read_text())
    parent=json.loads(Path(corr['parent']).read_text())
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=Path(source);dest=Path(publication);d=payload(dest);folder=dest/'travel-data';folder.mkdir(exist_ok=True)
    identities=sorted({m['id'].removesuffix('-'+m['representation']) for m in d['md']['models']})
    selections={}
    for snapshot in d['md']['snapshots']:
        key=snapshot['key'];base=source/'data'/key
        with np.load(base/'physical.npz') as z:physical=dict(z)
        with np.load(base/'descriptor-labels.npz') as z:joint=z['joint']
        seed=int.from_bytes(hashlib.sha256(('travel-v1/'+key).encode()).digest()[:8],'little');rng=np.random.default_rng(seed)
        rows=set(rng.choice(len(joint),min(2048,len(joint)),replace=False).tolist())
        for cid in range(7):
            members=np.flatnonzero(joint==cid);rows.update(rng.choice(members,min(64,len(members)),replace=False).tolist())
        rows=np.array(sorted(rows));coords=physical['coords'][rows]
        md=asset(dest/'md-data'/Path(snapshot['asset']).name)
        box=np.array(md['box']) if dataset=='matched' else None
        tree=cKDTree(coords,boxsize=box);neighbors=tree.query(coords,k=17)[1][:,1:]
        geometry=dict(rows=rows.tolist(),atoms=physical['atom'][rows].tolist(),xyz=coords.tolist(),box=None if box is None else box.tolist(),neighbors=neighbors.tolist(),descriptor_clusters=joint[rows].tolist())
        selections[key]=(rows,geometry)
        if lane==0:write_asset(folder/(key+'-geometry.js'),key,geometry,'TRAVEL_GEOMETRY')
    entries={};receipts={}
    for identity in identities[lane::lanes]:
        run,epoch_text=identity.rsplit('-epoch',1);epoch=int(epoch_text)
        root=Path(parent['output'])/run;checkpoint=root/'checkpoints'/f'epoch-{epoch:02d}.pt'
        expected=json.loads((root/'analyses'/f'epoch-{epoch:02d}'/'technical/complete.json').read_text())['checkpoint_sha256']
        if sha(checkpoint)!=expected:raise ValueError('Changed checkpoint '+identity)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['epoch']!=epoch or saved['data_identity']!=corr['cache_identity']:raise ValueError('Wrong model identity')
        model=PairEncoder(OmegaConf.create(saved['recipe'])).cuda().eval();model.requires_grad_(False);model.load_state_dict(saved['model'],strict=True)
        centers={}
        for rep in ['encoder','projector']:
            with np.load(root/'analyses'/f'epoch-{epoch:02d}'/'data'/f'{rep}-k7-assignments.npz') as z:centers[rep]=z['centers']
        for snapshot in d['md']['snapshots']:
            key=snapshot['key'];base=source/'data'/key;rows,geometry=selections[key]
            patches=np.load(base/'patches.npy',mmap_mode='r');parts={'encoder':[],'projector':[]}
            labels_path=base/((key+'-' if dataset=='matched' else '')+identity+'-labels.npz')
            with np.load(labels_path) as z:labels={rep:z[rep][rows] for rep in parts}
            with torch.inference_mode():
                for start in range(0,len(rows),256):
                    x=torch.as_tensor(np.asarray(patches[rows[start:start+256]])/parent['geometry']['length_scale_A'],device='cuda')
                    z,y=model(x)
                    for rep,t in [('encoder',z),('projector',y)]:parts[rep].append(t.cpu().numpy())
                for rep in parts:
                    values=np.ascontiguousarray(np.concatenate(parts[rep]));replayed=_labels_inertia_threadpool_limit(values,np.ones(len(values),np.float32),centers[rep],n_threads=1,return_inertia=False)
                    disagreements=int(np.sum(replayed!=labels[rep]))
                    if disagreements:raise ValueError(f'Travel probe assignment replay differs: {key}/{identity}/{rep}: {disagreements}')
                    tensor=torch.as_tensor(values,device='cuda');distances=torch.cdist(tensor,tensor);distances.fill_diagonal_(float('inf'))
                    neighbors=distances.topk(16,largest=False).indices.cpu().numpy()
                    name=identity+'-'+rep;asset_key=key+'-'+name;out=folder/(asset_key+'.js')
                    write_asset(out,asset_key,dict(z=values.tolist(),neighbors=neighbors.tolist(),clusters=labels[rep].tolist()),'TRAVEL_EMBEDDINGS')
                    entries.setdefault(key,{})[name]=dict(key=asset_key,asset='../travel-data/'+out.name)
                    receipts[asset_key]=dict(checkpoint_sha256=expected,labels_sha256=sha(labels_path),asset_sha256=sha(out),rows=len(rows),replay_disagreements=disagreements)
            print(f'Travel {identity} {key}: {len(rows)} atoms',flush=True)
        del model;torch.cuda.empty_cache()
    write_json(folder/f'manifest-lane{lane}.json',entries)
    write_json(dest/f'technical/rendering/travel-lane{lane}.json',dict(entries=receipts,implementation_sha256=sha(__file__),neural_training=False,
        selection='2048 uniform atoms plus up to 64 per frozen joint descriptor cluster; seed fixed by snapshot; same rows for all checkpoints',
        inputs='frozen nearest-80 geometry, original fixed length normalization, no conditions or history',
        graph='undirected union of directed 16-nearest-neighbor edges, Euclidean distance; periodic minimum image for matched spatial edges; raw full embedding for feature edges'))


def publish(publication,lanes):
    dest=Path(publication);d=payload(dest);folder=dest/'travel-data';result={}
    for lane in range(lanes):
        records=json.loads((folder/f'manifest-lane{lane}.json').read_text())
        for key,models in records.items():result.setdefault(key,{}).update(models)
    expected={m['id'] for m in d['md']['models']}
    if set(result)!={s['key'] for s in d['md']['snapshots']}:raise ValueError('Missing travel snapshots')
    for key,models in result.items():
        if set(models)!=expected:raise ValueError('Missing travel models '+key)
    write_json(folder/'manifest.json',{key:dict(geometry=dict(key=key,asset='../travel-data/'+key+'-geometry.js'),models=models) for key,models in result.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('action',choices=['infer','publish']);p.add_argument('--publication',required=True);p.add_argument('--source');p.add_argument('--config');p.add_argument('--dataset',choices=['matched','static']);p.add_argument('--lane',type=int,default=0);p.add_argument('--lanes',type=int,default=1);a=p.parse_args()
    if a.action=='infer':infer(a.config,a.source,a.publication,a.dataset,a.lane,a.lanes)
    else:publish(a.publication,a.lanes)
