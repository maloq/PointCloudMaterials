"""Queue, join and publish one history per original nucleus birth."""
import argparse
from collections import defaultdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np

from src.postprocessing.datasets.births import NucleiBirthDataset
from src.postprocessing.datasets.labels import NucleiBirthLabels
from src.experiment_runner.artifacts import file_hash as sha,json_digest as digest,write_json
from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,recorded_stage
from src.project_runtime.paths import REPO,resolve_path
from src.postprocessing.histories.common import root
from src.postprocessing.histories.al import context, lane, event
from src.postprocessing.visualization.births import render, index
from src.postprocessing.io import read

MODULE='src.postprocessing.histories.publish'


def join_labels(c,p,db):
    """Associate saved onsets with tracked families; an unknown fate stays unknown."""
    labels_path=resolve_path('${dataset:al-nuclei-birth}')/'additional_labels/first-crystal-010-v1/manifest.json'
    progress=root(c)/'technical/history-publication.json'
    while not labels_path.exists() or read(labels_path)['state']!='complete':
        write_json(progress,dict(state='waiting_for_sealed_appearance_labels',updated_unix=time.time()))
        time.sleep(30)
    with NucleiBirthDataset('010') as data,NucleiBirthLabels(data,'first-crystal-010-v1') as labels:
        for item in p['sources']:
            if digest(data.source(item['source_id']))!=digest(item['source']):raise ValueError('Pinned birth-history source changed')
        for item in p['sources']:
            sid=item['source_id'];events=[e for e in p['events'] if e['source_id']==sid]
            windows=list(labels.db.execute('SELECT window_id FROM windows WHERE source_id=? ORDER BY observation_time_ps',(sid,)))
            if not windows:continue
            _,_,access,ids,histories,_=context(str(root(c)/'technical/plan.json'),sid)
            # Source-local graph atom identity joins; shared nodes can support multiple births.
            graph_atoms=defaultdict(list)
            owner=defaultdict(list)
            for e in events:
                graph=read(root(c)/'analyses/births'/e['slug']/'graph.json')
                for n in graph['nodes']:
                    for atom in n['atom_ids']:graph_atoms[(n['frame'],atom)].append((e['event_id'],n['path_status']))
                for h in histories[e['track']]:owner[(h['frame'],h['largest_component'])].append(e['event_id'])
            tasks=defaultdict(list)
            for w in windows:
                wid=w['window_id'];packet=labels.window(wid);zero=int(packet['present_index'])
                for row in labels.db.execute('SELECT * FROM observations WHERE window_id=? AND at_risk=1',(wid,)):
                    i=int(np.searchsorted(packet['atom'],row['center_atom_id'])) if np.all(np.diff(packet['atom'])>=0) else int(np.flatnonzero(packet['atom']==row['center_atom_id'])[0])
                    if packet['atom'][i]!=row['center_atom_id']:raise ValueError('Appearance atom join failed')
                    for target,column in [('appearance','first_step'),('persistent','persistent_first_step')]:
                        step=row[column]
                        if step is None:continue
                        onset=zero+step;frame=int(packet['frames'][onset])
                        solid=np.isin(packet['labels'][i,onset],[1,2,3])&packet['consumed'][i]
                        if target=='persistent':solid&=np.isin(packet['labels'][i,onset:onset+3],[1,2,3]).all(0)
                        atoms=packet['members'][i,solid].astype(int).tolist()
                        if not atoms:raise ValueError('Saved positive onset has no supporting atoms')
                        tasks[frame].append((row['observation_id'],target,atoms))
            for frame,observations in sorted(tasks.items()):
                old_labels=access.labels(frame)
                for oid,target,atoms in observations:
                    matches=defaultdict(set)
                    for atom in atoms:
                        for eid,status in graph_atoms[(frame,atom)]:matches[(eid,'precursor_'+status)].add(atom)
                        pos=int(np.searchsorted(ids,atom))
                        if pos>=len(ids) or ids[pos]!=atom:raise ValueError('Onset atom missing from original source')
                        for eid in owner[(frame,int(old_labels[pos]))]:matches[(eid,'recorded_family')].add(atom)
                    distinct={k[0] for k in matches}
                    db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?)',(oid,target,sid,frame,len(distinct),'matched' if distinct else 'unassigned_within_reconstructed_coverage'))
                    for (eid,relation),shared in matches.items():
                        db.execute('INSERT INTO links VALUES (?,?,?,?,?,?)',(oid,target,eid,relation,len(shared),json.dumps(sorted(shared))))
            db.commit();write_json(progress,dict(state='linking_appearance_observations',source_id=sid,updated_unix=time.time()))
        # Sources with no established birth still retain their onset records and unknown association.
        included={s['source_id'] for s in p['sources']}
        for source in labels.db.execute('SELECT source_id FROM sources'):
            sid=source['source_id']
            if sid in included:continue
            for row in labels.db.execute('''SELECT o.* FROM observations o JOIN windows w USING(window_id)
                WHERE w.source_id=? AND o.at_risk=1''',(sid,)):
                for target,field in [('appearance','appearance_frame'),('persistent','persistent_frame')]:
                    if row[field] is not None:db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?)',(row['observation_id'],target,sid,row[field],0,'unassigned_within_reconstructed_coverage'))
        db.commit()
        return labels.manifest


def publish(c):
    r=root(c);p=read(r/'technical/plan.json');dest=resolve_path(c['publication']);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'publication.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (dest/'manifest.json').exists():
            old=read(dest/'manifest.json')
            if old['history_plan_identity']!=p['identity'] or sha(dest/'histories.sqlite')!=old['database_sha256']:raise ValueError('Published history identity changed')
            return old
        packets=[]
        for e in p['events']:
            folder=r/'analyses/births'/e['slug'];done=read(folder/'complete.json')
            if any(sha(folder/n)!=h for n,h in done['files'].items()):raise ValueError('Changed history before publication')
            packets.append((e,folder,done,read(folder/'summary.json')))
        build=dest/'history.building.sqlite'
        if build.exists():build.unlink() # Only our unpublished derived database; immutable packets stay intact.
        db=sqlite3.connect(build)
        db.executescript('''
            CREATE TABLE births(event_id TEXT PRIMARY KEY,slug TEXT UNIQUE,source_id TEXT,ancestor TEXT,
              role TEXT,physical_birth_key TEXT,path TEXT,summary_json TEXT,files_json TEXT);
            CREATE INDEX birth_physical ON births(physical_birth_key);
            CREATE TABLE observations(observation_id TEXT,target TEXT,source_id TEXT,frame INTEGER,
              matched_births INTEGER,status TEXT,PRIMARY KEY(observation_id,target));
            CREATE TABLE links(observation_id TEXT,target TEXT,event_id TEXT REFERENCES births,
              relation TEXT,shared_atoms INTEGER,atom_ids_json TEXT,PRIMARY KEY(observation_id,target,event_id,relation));
            CREATE INDEX links_birth ON links(event_id);
        ''')
        for e,folder,done,summary in packets:
            target=dest/'births'/e['slug'];target.mkdir(parents=True,exist_ok=True)
            for name,h in done['files'].items():
                path=target/name
                if not path.exists():shutil.copy2(folder/name,path)
                if sha(path)!=h:raise ValueError('Conflicting durable birth packet')
            db.execute('INSERT INTO births VALUES (?,?,?,?,?,?,?,?,?)',(e['event_id'],e['slug'],e['source_id'],e['ancestor'],e['role'],summary['physical_birth_key'],str(target.relative_to(dest)),json.dumps(summary),json.dumps(done['files'])))
        db.commit();label_manifest=join_labels(c,p,db)
        counts={
            'birth_records':db.execute('SELECT COUNT(*) FROM births').fetchone()[0],
            'physical_birth_keys':db.execute('SELECT COUNT(DISTINCT physical_birth_key) FROM births').fetchone()[0],
            'ancestors':db.execute('SELECT COUNT(DISTINCT ancestor) FROM births').fetchone()[0],
            'appearance_onset_records':db.execute("SELECT COUNT(*) FROM observations WHERE target='appearance'").fetchone()[0],
            'linked_appearance_onsets':db.execute("SELECT COUNT(*) FROM observations WHERE target='appearance' AND matched_births>0").fetchone()[0],
            'ambiguous_appearance_onsets':db.execute("SELECT COUNT(*) FROM observations WHERE target='appearance' AND matched_births>1").fetchone()[0]}
        db.close();build.replace(dest/'histories.sqlite')
        provenance=dest/'provenance';provenance.mkdir(exist_ok=True)
        shutil.copy2(r/'technical/plan.json',provenance/'plan.json');shutil.copy2(r/'technical/parent-pin.json',provenance/'parent-pin.json')
        write_json(provenance/'appearance-labels.json',label_manifest)
        manifest=dict(name='birth-tails-010-v1',state='complete',cadence='010',parent_fingerprint=c['fingerprint'],
            history_plan_identity=p['identity'],appearance_label_fingerprint=label_manifest['fingerprint'],counts=counts,
            database_sha256=sha(dest/'histories.sqlite'),publisher_sha256=sha(Path(__file__)),
            interpretation='One row per original established birth; precursor components and onsets are nested history, never new births. Unassigned is unknown fate, not dissolved. Multiple matches remain explicit.',
            provenance={str(f.relative_to(dest)):sha(f) for f in provenance.iterdir()})
        manifest['fingerprint']=digest(manifest);write_json(dest/'manifest.json',manifest)
        with (dest.parent/'index.lock').open('a') as master:
            fcntl.flock(master,fcntl.LOCK_EX);path=dest.parent/'index.json';registry=read(path) if path.exists() else dict(histories={})
            registry['histories'][manifest['name']]=dict(name=manifest['name'],cadence='010',path='histories/'+dest.name,fingerprint=manifest['fingerprint'])
            write_json(path,registry)
        write_json(r/'technical/history-publication.json',dict(state='complete',manifest=str(dest/'manifest.json'),counts=counts))
    return manifest


def examples(c):
    p=read(root(c)/'technical/plan.json')
    for slug in p['examples']:
        event(c,slug);render(c,slug);index(c)


def launch(c):
    r=root(c);tech=r/'technical';receipt=tech/'queue.json'
    if receipt.exists():raise FileExistsError('Queue already submitted: '+str(receipt))
    bundle=ExecutionBundle.freeze(REPO,tech/'code',c,directories=('src','configs/postprocessing','docs/metrics'),files=((REPO/'machine.local.yaml','machine.local.yaml'),(REPO/'configs/datasets.json','configs/datasets.json')))
    record=dict(owner='current-chat birth-centred histories',jobs={},plan_identity=read(tech/'plan.json')['identity'])
    env={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OVITO_THREAD_COUNT')}
    q=SlurmQueue(tech,bundle,MODULE,env,receipt,record,'al-birth-tails')
    with q.submission():
        job=q.submit('collect',['--array=0-'+str(c['lanes']-2),'--cpus-per-task=2','--mem=16G','--time='+c['time_limit']])
        q.submit('finish',['--cpus-per-task=2','--mem=16G','--time=24:00:00'],dependency='afterany:'+job)
    print(json.dumps(record),flush=True)


def finish(c):
    """Finish the deferred lane and any incomplete packets, then publish once."""
    r=root(c);p=read(r/'technical/plan.json');missing=[]
    for ix in range(c['lanes']):
        if any(not (r/'analyses/births'/e['slug']/'complete.json').exists() for e in p['events'][ix::c['lanes']]):missing.append(ix)
    write_json(r/'technical/continuation.json',dict(state='running',incomplete_lanes=missing))
    for ix in missing:lane(c,ix)
    examples(c);publish(c);index(c)
    write_json(r/'technical/continuation.json',dict(state='complete',incomplete_lanes=missing))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('launch','collect','examples','finish','finalize'));parser.add_argument('--config',required=True)
    args=parser.parse_args();c=read(args.config)
    if args.stage=='launch':launch(c)
    elif args.stage=='collect':
        with recorded_stage(root(c)/'technical'/('lane-'+os.environ['SLURM_ARRAY_TASK_ID']+'.json')):lane(c,int(os.environ['SLURM_ARRAY_TASK_ID']))
    elif args.stage=='examples':examples(c)
    elif args.stage=='finish':
        with recorded_stage(root(c)/'technical/finish.json'):finish(c)
    else:
        with recorded_stage(root(c)/'technical/finalize.json'):
            examples(c);publish(c);index(c)


if __name__=='__main__':main()
