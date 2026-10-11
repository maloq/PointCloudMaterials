"""Birth-centred precursor graphs without inventing additional nucleus events."""
from src.postprocessing.descriptors.ptm import producer_identity

from src.postprocessing.histories.common import root, cache, slug, precursor_links, local_environment, shape
from src.postprocessing.histories.frames import HistoryFrames
import argparse
from collections import defaultdict
import fcntl
from functools import lru_cache
import gzip
import json
import os
from pathlib import Path
import time

import numpy as np

from src.postprocessing.datasets.births import NucleiBirthDataset
from src.experiment_runner.artifacts import file_hash as sha,json_digest as digest,write_json
from src.experiment_runner.metric_docs import check_metric_docs,write_metric_rows
from src.project_runtime import paths as runtime_paths
from src.project_runtime.paths import resolve_path
from src.postprocessing.io import geometry, arrays
from src.postprocessing.descriptors.ptm import full_labels
from src.postprocessing.extraction import Frames
from src.postprocessing.io import read

FAMILY='nucleus_histories'


def prepare(c):
    with NucleiBirthDataset('010',expected_fingerprint=c['fingerprint']) as data:
        selected=list(data.events(selection='established',detector='ptm'))
        sources=[];events=[]
        for sid in sorted({e['source_id'] for e in selected}):
            s=data.source(sid)
            row=data.db.execute('''SELECT t.folder,t.files FROM tasks t JOIN views v USING(view_id)
                WHERE v.source_id=? AND v.cadence='010' AND t.detector='ptm' ''',(sid,)).fetchone()
            folder=Path(row['folder']);lineage=read(folder.parents[3]/'plan.json')
            parent=resolve_path(lineage['config']['parent_output'])/'technical/plan.json'
            if sha(parent)!=lineage['parent_plan_sha256']:raise ValueError('Changed component extraction plan')
            sources.append(dict(source_id=sid,source=s,labels=str(folder),label_files=json.loads(row['files']),parent=str(parent),parent_sha256=sha(parent)))
            for e in [e for e in selected if e['source_id']==sid]:
                full=data.event(e['event_id'])
                events.append(dict(full,slug=slug(e['event_id']),source_id=sid))
        examples=[];seen=set()
        sizes={s['source_id']:s['source']['atom_count'] for s in sources}
        for e in sorted(events,key=lambda e:(e['establishment_confirmation_ps']-e['first_time_ps'],e['event_id'])):
            if e['ancestor'] in seen or sizes[e['source_id']]>100000:continue
            examples.append(e['slug']);seen.add(e['ancestor'])
            if len(examples)==c['example_count']:break
        p=dict(config=c,sources=sources,events=events,examples=examples,
            scientific_implementation=check_metric_docs(family=FAMILY)[FAMILY]['files'],
            parent_fingerprint=data.fingerprint,
            semantics='One root per existing established birth event. New small components are precursor/history nodes, never nucleus entries.',
            scope='All PTM established 0.10 ps records, including sources without velocities; geometry-only annotation and viewing. Original roles retained.',
            example_selection='Shortest recorded embryo-to-establishment intervals on different <=100k-atom ancestors; illustrative, not a representative performance sample.')
        p['identity']=digest(p);dest=root(c)/'technical/plan.json'
        if dest.exists() and read(dest)!=p:raise ValueError('Birth-history protocol changed; use a new revision')
        write_json(dest,p);data.pin(root(c)/'technical/parent-pin.json')
    return p


@lru_cache(maxsize=2)
def context(plan_path,sid):
    p=read(plan_path);item=next(s for s in p['sources'] if s['source_id']==sid);s=item['source']
    parent=read(item['parent'])
    if sha(Path(item['parent']))!=item['parent_sha256']:raise ValueError('Component source changed')
    original=next(a for a in parent['sources'] if a['input_identity']==s['input_identity'])
    access=Frames(parent,original,'ptm')
    folder=Path(item['labels'])
    for name,h in item['label_files'].items():
        if sha(folder/name)!=h:raise ValueError('Changed original family lineage '+name)
    upstream=read(folder/'events.json');established={e['track']:e for e in upstream if e['establishment_confirmation_ps'] is not None}
    wanted={e['track'] for e in p['events'] if e['source_id']==sid}
    histories=defaultdict(list);owner={}
    with gzip.open(folder/'sizes.jsonl.gz','rt') as stream:
        for line in stream:
            fid=int(line.split(',',1)[0].split(':')[1])
            if fid not in established and fid not in wanted:continue
            row=json.loads(line)
            if fid in wanted:histories[fid].append(row)
            if fid in established and row['time_ps']>=established[fid]['establishment_confirmation_ps']-1e-8:
                owner[(row['frame'],row['largest_component'])]=fid
    ids=arrays(s['parts'][0]['path'],s['parts'][0]['manifest_sha256'])['atom_ids']
    return p,item,access,ids,histories,owner


def ptm(c,item,frame,q,box,access=None):
    producer=producer_identity()
    if access is not None:
        retained=access.ptm_labels(frame,c['ptm_rmsd_cutoff'],producer)
        if retained is not None:return retained
    folder=cache(c)/'ptm'/item['source_id'];folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{frame:06d}.npz';receipt=path.with_suffix('.json')
    identity=digest([item['source_id'],frame,c['ptm_rmsd_cutoff'],producer])
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            old=read(receipt)
            if old['identity']!=identity or sha(path)!=old['sha256']:raise ValueError('Changed history PTM cache')
            with np.load(path) as a:return a['labels']
        labels=full_labels(q,box,c['ptm_rmsd_cutoff'])
        temporary=path.with_suffix('.building.npz');np.savez_compressed(temporary,labels=labels);temporary.replace(path)
        write_json(receipt,dict(identity=identity,sha256=sha(path)));return labels


def event(c,event_id,*,blocking=True):
    plan_path=root(c)/'technical/plan.json';p=read(plan_path)
    e=next(e for e in p['events'] if e['slug']==event_id or e['event_id']==event_id)
    dest=root(c)/'analyses/births'/e['slug'];dest.mkdir(parents=True,exist_ok=True)
    identity=digest([p['identity'],e['event_id']])
    with (dest/'worker.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:return None
        if (dest/'complete.json').exists():
            old=read(dest/'complete.json')
            if old['identity']!=identity or any(sha(dest/n)!=h for n,h in old['files'].items()):raise ValueError('Changed birth-history packet')
            return old
        started=time.monotonic()
        _,item,access,atom_ids,histories,owners=context(str(plan_path),e['source_id']);s=item['source']
        view=np.asarray(s['views']['010']);times=np.asarray(s['times_ps']);first=int(np.searchsorted(view,e['first_frame']))
        if view[first]!=e['first_frame']:raise ValueError('Birth not on exact 0.1 ps view')
        confirmation=int(np.searchsorted(times[view],e['establishment_confirmation_ps']-1e-8))
        if abs(times[view[confirmation]]-e['establishment_confirmation_ps'])>1e-8:raise ValueError('Establishment timestamp mismatch')
        start=max(0,first-round(c['lookback_ps']/.1));early_end=min(len(view)-1,first+round(c['early_after_ps']/.1))
        late_start=max(0,confirmation-round(c['establishment_before_ps']/.1));late_end=min(len(view)-1,confirmation+round(c['establishment_after_ps']/.1))
        frame_slots=sorted(set(range(start,early_end+1))|set(range(late_start,late_end+1)))
        birth_rows=np.searchsorted(atom_ids,e['first_members'])
        if not np.array_equal(atom_ids[birth_rows],e['first_members']):raise ValueError('Founder IDs changed')
        q0,b0=geometry(s,int(view[first]));delta=q0[birth_rows]-np.asarray(e['centroid_A']);delta-=b0*np.rint(delta/b0)
        anchor=int(birth_rows[np.argmin(np.linalg.norm(delta,axis=1))])
        physical=digest([e['ancestor'],e['first_time_ps'],e['first_members'],q0[birth_rows].tolist(),b0.tolist()])
        movie={};frame_nodes={};nodes={};trace={r['frame']:r for r in histories[e['track']]}
        frame_cache=HistoryFrames(c,p,item,access,atom_ids,owners,e)
        anchor_index=int(np.searchsorted(frame_cache.anchors,anchor))
        if frame_cache.anchors[anchor_index]!=anchor:raise ValueError('Shared history anchor differs')
        for n,slot in enumerate(frame_slots):
            frame=int(view[slot]);observed=[]
            def observe():
                if not observed:
                    q,box=geometry(s,frame);observed.extend((q,box,ptm(c,item,frame,q,box,access)))
                return observed
            saved=frame_cache.observation(frame)
            if saved is None or slot<=first:
                shared=frame_cache.prepare(frame,observe)
            if slot<=first:frame_nodes[slot]=shared['nodes']
            if saved is None:
                q,box,labels=observe()
                d,local,patch,(q6,qbar6,coherence6)=local_environment(q,box,anchor,c['radius_A'],neighbors=shared['neighbors'][anchor_index])
                original=access.labels(frame);track=trace.get(frame)
                member_mask=np.zeros(len(local),bool) if track is None else original[local]==track['largest_component']
                saved=dict(frame=frame,time_ps=float(times[frame]),relative_time_ps=float(times[frame]-e['first_time_ps']),
                    positions=d[local].astype(np.float32),atom_ids=atom_ids[local],ptm=labels[local],
                    catalogued_family=member_mask,founder=np.isin(atom_ids[local],e['first_members']),
                    largest_family=0 if track is None else track['size'],q6=q6,qbar6=qbar6,
                    coherence6=coherence6,first_shell_A=float(np.linalg.norm(patch[1:13],axis=1).mean()),
                    crystal_fraction=float(np.isin(labels[local],[1,2,3]).mean()),box=box.tolist())
                frame_cache.checkpoint(frame,saved)
            movie[slot]=saved
            if n%10==0:write_json(dest/'progress.json',dict(state='classifying',completed=n+1,total=len(frame_slots),
                restored_frames=frame_cache.restored,new_frames=frame_cache.written,updated_unix=time.time()))
        anchor_node,nodes,edges=precursor_links(first,start,frame_nodes,e['first_members'],c['maximum_frame_gap'])
        for slot,frame in movie.items():
            ancestor_atoms={a for r in nodes.values() if r['frame']==frame['frame'] and r['path_status'] in ('direct','gap') for a in r['atom_ids']}
            frame['precursor']=np.isin(frame['atom_ids'],list(ancestor_atoms))
            cloud=frame['positions'][frame['precursor']|frame['catalogued_family']]
            frame.update(shape(cloud))
        all_nodes=sorted(nodes.values(),key=lambda r:(r['frame'],r['id']));direct=[n for n in all_nodes if n['path_status']=='direct']
        gap=[n for n in all_nodes if n['path_status'] in ('direct','gap')]
        known=sorted({n['known_established_family'] for n in all_nodes if n['known_established_family'] is not None})
        graph=dict(birth_event_id=e['event_id'],physical_birth_key=physical,root_node=anchor_node['id'],nodes=all_nodes,edges=edges,
            earliest_direct_ps=min(n['time_ps'] for n in direct),earliest_with_gaps_ps=min(n['time_ps'] for n in gap),
            lookback_start_ps=float(times[view[start]]),lookback_reaches_source_start=start==0,
            tail_left_censored=any(n['frame']==int(view[start]) for n in gap),known_established_connections=known,
            interpretation='Retrospective shared-atom structural ancestry; weak and gap links are uncertain. No new birth labels or causal inference.')
        write_json(dest/'graph.json',graph);write_json(dest/'birth.json',dict(e,physical_birth_key=physical,anchor_atom_id=int(atom_ids[anchor])))
        write_json(dest/'family-history.json',histories[e['track']])
        offsets=[0];packed=defaultdict(list);measurements=[]
        for slot,f in sorted(movie.items()):
            for key in ('positions','atom_ids','ptm','catalogued_family','founder','precursor'):packed[key].append(f[key])
            offsets.append(offsets[-1]+len(f['atom_ids']))
            measurements.append({k:v for k,v in f.items() if k not in packed})
        np.savez_compressed(dest/'movie.npz',offsets=np.asarray(offsets,np.int64),**{k:np.concatenate(v) for k,v in packed.items()})
        write_json(dest/'movie-frames.json',measurements)
        summary=dict(birth_event_id=e['event_id'],slug=e['slug'],source_id=e['source_id'],source=e['source'],ancestor=e['ancestor'],role=e['role'],
            physical_birth_key=physical,first_8_ps=e['first_time_ps'],establishment_start_ps=e['establishment_start_ps'],
            establishment_confirmation_ps=e['establishment_confirmation_ps'],earliest_direct_ps=graph['earliest_direct_ps'],
            earliest_with_gaps_ps=graph['earliest_with_gaps_ps'],tail_left_censored=graph['tail_left_censored'],precursor_nodes=len(all_nodes),
            weak_edges=sum(a['kind']=='weak_overlap' for a in edges),gap_edges=sum(a['kind']=='gap' for a in edges),
            known_established_connections=len(known),movie_frames=len(measurements),movie_atoms=offsets[-1])
        write_json(dest/'summary.json',summary)
        write_json(dest/'execution.json',dict(plan_identity=p['identity'],
            implementation=check_metric_docs(family=FAMILY)[FAMILY]['files'],
            path_resolver_sha256=sha(Path(runtime_paths.__file__)),
            descriptor_threads=int(os.environ.get('OVITO_THREAD_COUNT','1')),
            node=os.uname().nodename,job=os.environ.get('SLURM_JOB_ID'),
            array_task=os.environ.get('SLURM_ARRAY_TASK_ID'),seconds=time.monotonic()-started,
            restored_frames=frame_cache.restored,new_frames=frame_cache.written,frame_cache_identity=frame_cache.identity))
        files={n:sha(dest/n) for n in ('graph.json','birth.json','family-history.json','movie.npz','movie-frames.json','summary.json','execution.json')}
        done=dict(identity=identity,event_id=e['event_id'],slug=e['slug'],files=files);write_json(dest/'complete.json',done)
        print('birth history complete',e['slug'],len(all_nodes),'precursor nodes',flush=True);return done


def lane(c,index):
    p=read(root(c)/'technical/plan.json')
    for e in p['events'][index::c['lanes']]:event(c,e['event_id'])


def assist(c):
    """Claim unfinished packets without waiting for another worker's event."""
    from src.experiment_runner.execution import recorded_stage
    p=read(root(c)/'technical/plan.json')
    if p['config']!=c:raise ValueError('Assistant must use the existing history plan configuration')
    worker=os.environ.get('SLURM_JOB_ID',str(os.getpid()))+'-'+os.environ.get('SLURM_ARRAY_TASK_ID','0')
    with recorded_stage(root(c)/'technical'/f'assist-{worker}.json',
                        total=len(p['events']),completed=0,busy=0) as progress:
        completed=busy=0
        for e in p['events']:
            if (root(c)/'analyses/births'/e['slug']/'complete.json').exists():continue
            result=event(c,e['event_id'],blocking=False)
            if result is None:busy+=1
            else:completed+=1
            progress.update(completed=completed,busy=busy,last_event=e['slug'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('prepare','event','lane','assist'))
    p.add_argument('--config',required=True);p.add_argument('--event');p.add_argument('--lane',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID',0)))
    a=p.parse_args();c=read(a.config);check_metric_docs(family=FAMILY)
    if a.stage=='prepare':prepare(c)
    elif a.stage=='event':event(c,a.event)
    elif a.stage=='assist':assist(c)
    else:lane(c,a.lane)


if __name__=='__main__':main()
