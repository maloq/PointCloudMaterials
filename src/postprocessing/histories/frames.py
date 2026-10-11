"""Shared source-frame preparation and resumable event-frame observations."""
import fcntl
from functools import lru_cache
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.postprocessing.io import geometry, read
from . import common


@lru_cache(maxsize=1)
def implementation():
    from src.experiment_runner.metric_docs import check_metric_docs
    return check_metric_docs(family='nucleus_histories')['nucleus_histories']['files']


def load(path,identity):
    receipt=path.with_suffix('.json')
    if not receipt.exists():return None
    r=read(receipt)
    if r['identity']!=identity or sha(path)!=r['sha256']:
        raise ValueError(f'Changed history frame checkpoint: {path}')
    with np.load(path,allow_pickle=False) as a:
        meta=json.loads(str(a['metadata']))
        for key in a.files:
            if key!='metadata':meta[key]=a[key]
    return meta


def save(path,identity,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:v for k,v in value.items() if isinstance(v,np.ndarray)}
    meta={k:None if k in arrays else v for k,v in value.items()}
    temporary=path.with_name(f'.{path.stem}.{os.getpid()}.building.npz')
    np.savez_compressed(temporary,metadata=np.asarray(json.dumps(meta,allow_nan=False)),**arrays)
    temporary.replace(path);write_json(path.with_suffix('.json'),dict(identity=identity,sha256=sha(path)))


@lru_cache(maxsize=4)
def anchor_plan(plan_path,sid):
    p=read(plan_path);s=next(i['source'] for i in p['sources'] if i['source_id']==sid)
    view=np.asarray(s['views']['010']);anchors=set();node_frames=set()
    from src.postprocessing.io import arrays
    atom_ids=arrays(s['parts'][0]['path'],s['parts'][0]['manifest_sha256'])['atom_ids']
    for e in p['events']:
        if e['source_id']!=sid:continue
        rows=np.searchsorted(atom_ids,e['first_members'])
        if not np.array_equal(atom_ids[rows],e['first_members']):raise ValueError('Changed history founder IDs')
        q,box=geometry(s,e['first_frame']);delta=q[rows]-np.asarray(e['centroid_A']);delta-=box*np.rint(delta/box)
        anchors.add(int(rows[np.argmin(np.linalg.norm(delta,axis=1))]))
        first=int(np.searchsorted(view,e['first_frame']));start=max(0,first-round(p['config']['lookback_ps']/.1))
        node_frames.update(int(f) for f in view[start:first+1])
    return np.asarray(sorted(anchors),np.int64),node_frames


class HistoryFrames:
    def __init__(self,c,p,item,access,atom_ids,owners,event):
        self.c,self.item,self.access,self.atom_ids,self.owners=c,item,access,atom_ids,owners
        self.anchors,self.node_frames=anchor_plan(str(common.root(c)/'technical/plan.json'),item['source_id'])
        self.identity=digest(dict(plan=p['identity'],source=item['source_id'],implementation=implementation()))
        self.shared=common.cache(c)/'shared-frames'/self.identity
        self.event=common.cache(c)/'event-frames'/self.identity/event['slug']
        self.restored=0;self.written=0

    def observation(self,frame):
        value=load(self.event/f'{frame:06d}.npz',digest([self.identity,str(self.event.name),frame]))
        if value is not None:self.restored+=1
        return value

    def checkpoint(self,frame,value):
        save(self.event/f'{frame:06d}.npz',digest([self.identity,str(self.event.name),frame]),value)
        self.written+=1

    def prepare(self,frame,observe):
        """One spatial tree and precursor partition for all anchors in this source."""
        path=self.shared/f'{frame:06d}.npz';identity=digest([self.identity,frame])
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            value=load(path,identity)
            if value is None:
                q,box,labels=observe()
                _,neighbors=cKDTree(q,boxsize=box).query(q[self.anchors],k=80,workers=1)
                nodes=[];offsets=[0];memberships=[]
                if frame in self.node_frames:
                    original=self.access.labels(frame)
                    for members in common.clusters(q,box,labels,self.c['neighbor_cutoff_A']):
                        old=original[members];positive=old[old>0]
                        family=self.owners.get((frame,int(np.bincount(positive).argmax()))) if len(positive) else None
                        record=common.node(self.item['source_id'],frame,self.item['source']['times_ps'][frame],members,self.atom_ids,family)
                        memberships.append(self.atom_ids[members]);offsets.append(offsets[-1]+len(members))
                        nodes.append(dict(record,atom_ids=None))
                value=dict(neighbors=neighbors,nodes=nodes,offsets=np.asarray(offsets,np.int64),
                           memberships=np.concatenate(memberships) if memberships else np.empty(0,self.atom_ids.dtype))
                save(path,identity,value)
        for i,node in enumerate(value['nodes']):
            node['atom_ids']=value['memberships'][value['offsets'][i]:value['offsets'][i+1]].tolist()
        return value
