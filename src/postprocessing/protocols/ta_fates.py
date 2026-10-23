"""Large-cell Ta extraction/reuse and the common birth/fate definitions."""
from functools import lru_cache, partial
import fcntl
from pathlib import Path
import shutil
import time

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.trajectories import array_sha256
from src.postprocessing.protocols import ancestry, external_audit as audit, external_data as data, fates, shooting_fates
from src.postprocessing.descriptors.ptm import full_labels
from src.postprocessing.protocols.shooting_data import read


def verify_inputs(plan, item):
    out=data.source_folder(plan,item['id'])
    out.mkdir(parents=True,exist_ok=True)
    for part in item['parts']:
        arrays=data.arrays(part['path'],part['manifest_sha256'])
        for name,a in arrays.items():
            if array_sha256(a)!=part['array_sha256'][name]:
                raise ValueError(f'Changed Ta input: {part["path"]}/{name}')
    write_json(out/'input-verified.json',dict(plan_identity=plan['identity'],parts=item['parts']))


def extract_frame(plan,item,frame):
    out=data.source_folder(plan,item['id'])
    out.mkdir(parents=True,exist_ok=True)
    name=f'ptm-{frame:04d}-{frame+1:04d}'
    path,receipt=out/(name+'.npz'),out/(name+'.json')
    identity=digest(dict(plan=plan['identity'],source=item['id'],frame=frame))
    if receipt.exists():
        old=read(receipt)
        if old['identity']!=identity or sha(path)!=old['sha256']:
            raise ValueError(f'Changed Ta PTM frame: {path}')
        return old
    started=time.monotonic()
    labels=full_labels(*data.geometry(item,frame),plan['config']['ptm_rmsd_cutoff'])
    temporary=path.with_suffix('.building.npz')
    np.savez_compressed(temporary,labels=labels[None])
    temporary.replace(path)
    value=dict(identity=identity,source=item['id'],frame=frame,sha256=sha(path),seconds=time.monotonic()-started)
    write_json(receipt,value)
    return value


def reused_graph(plan,item,identity):
    old=item['legacy'];folder=Path(old['folder']);out=data.source_folder(plan,item['id'])
    if read(folder/'ptm-complete.json')!=old['ptm_receipt'] or read(folder/'graph-complete.json')!=old['graph_receipt']:
        raise ValueError(f'Historical Ta receipt changed: {folder}')
    if old['ptm_receipt']['plan_identity']!=old['plan_identity']:
        raise ValueError(f'Wrong historical PTM population: {folder}')
    for name,expected in old['ptm_receipt']['files'].items():
        if sha(folder/name)!=expected:
            raise ValueError(f'Historical Ta PTM chunk changed: {folder/name}')
    if sha(folder/'graph.npz')!=old['graph_receipt']['sha256']:
        raise ValueError(f'Historical Ta graph changed: {folder}')
    shutil.copyfile(folder/'graph.npz',out/'graph.npz')
    write_json(out/'graph-complete.json',dict(identity=identity,sha256=sha(out/'graph.npz'),reused=old['graph_receipt']))
    write_json(out/'ptm-complete.json',dict(plan_identity=plan['identity'],reused=old))
    with np.load(out/'graph.npz') as values:
        return {k:values[k] for k in values.files}


class Observations(shooting_fates.Observations):
    def __init__(self,plan,item,graph):
        audit.FrameAccess.__init__(self,plan,item,graph)
        part=item['parts'][0]
        self.atom_ids=data.arrays(part['path'],part['manifest_sha256'])['atom_ids']
        self.cached_frame=lru_cache(maxsize=2)(partial(audit.FrameAccess.frame,self))
        self.chunk_count=1
        if item['legacy']:
            self.folder=Path(item['legacy']['folder'])
            self.chunk_count=item['legacy']['chunk_frames']

    def labels(self,frame):
        start=frame//self.chunk_count*self.chunk_count
        if start!=self.loaded_start:
            stop=min(start+self.chunk_count,self.source['frame_count'])
            with np.load(self.folder/f'ptm-{start:04d}-{stop:04d}.npz') as values:
                self.values=values['labels']
            self.loaded_start=start
        return self.values[frame-start]


def process(plan,item):
    out=data.source_folder(plan,item['id']);out.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(plan=plan['identity'],source=item['id']))
    done=out/'complete.json'
    with (out/'source.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if done.exists():
            value=read(done)
            if value['identity']!=identity or any(sha(out/n)!=h for n,h in value['files'].items()):
                raise ValueError(f'Changed Ta result: {out}')
            return value
        started=time.monotonic()
        verify_inputs(plan,item)
        if item['legacy']:
            graph=reused_graph(plan,item,identity)
        else:
            files={}
            for frame in range(item['frame_count']):
                result=extract_frame(plan,item,frame)
                files[f'ptm-{frame:04d}-{frame+1:04d}.npz']=result['sha256']
                write_json(out/'progress.json',dict(stage='PTM',frames=frame+1,total=item['frame_count'],seconds=time.monotonic()-started))
            write_json(out/'ptm-complete.json',dict(plan_identity=plan['identity'],files=files))
            graph=audit.graph_for_source(plan,item,identity)
        observations=Observations(plan,item,graph)
        events,roots,uncertain=ancestry.establish(graph,item['lineage']['thresholds'][0])
        audit.locate(item,graph,events,roots,observations)
        parents,_=fates.graph_links(graph)
        first=graph['frame'].copy()
        for node in range(1,len(first)):
            first[node]=min([int(first[node])]+[int(first[p]) for p,_,_ in parents[node]])
        for event in events:
            event['center_distances_A']=event['center_distances_A'].tolist()
            event['establishment_kind']=event['kind']
            event['initial_cluster_ancestry']=bool(first[event['birth_node']]==0)
            if event['initial_cluster_ancestry'] and not event['left_censored']:
                event['kind']='parent_present_embryo_growth'
        c=dict(plan['config']['fates']);dt=item['cadence_ps']
        c['primary_persistence_frames']=1+int(np.ceil(c['primary_persistence_ps']/dt-1e-9))
        c['growth_persistence_frames']=1+int(np.ceil(c['growth_persistence_ps']/dt-1e-9))
        c['dissolution_confirmation_frames']=int(np.ceil(c['dissolution_followup_ps']/dt-1e-9))
        transient,established,counts=shooting_fates.evaluate(c,item,graph,events,roots,uncertain,observations)
        for name,values in [('events',events),('transient-episodes',transient),('established-fates',established)]:
            write_json(out/(name+'.json'),values)
        names=['events.json','transient-episodes.json','established-fates.json','graph.npz','ptm-complete.json','input-verified.json']
        value=dict(identity=identity,source=item['id'],counts=counts,seconds=time.monotonic()-started,
            reused_ptm=bool(item['legacy']),files={n:sha(out/n) for n in names},
            actual_durations=dict(cadence_ps=dt,establishment_span_ps=1.5,
                primary_failed_span_ps=(c['primary_persistence_frames']-1)*dt,
                growth_span_ps=(c['growth_persistence_frames']-1)*dt,
                dissolution_followup_ps=c['dissolution_confirmation_frames']*dt))
        write_json(done,value)
        observations.cached_frame.cache_clear();data.arrays.cache_clear()
        return value
