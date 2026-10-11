"""Versioned observation-label supplements to the consolidated birth catalogue.

Publication copies completed labels, never runs PTM or changes nucleus events.
The parent catalogue keeps its fingerprint; each supplement has its own identity.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.experiment_runner.execution import ExecutionBundle, recorded_stage
from src.project_runtime.paths import REPO, resolve_path
from src.postprocessing.datasets.births import NucleiBirthDataset, connect, read, compact

MODULE = 'src.postprocessing.datasets.labels'
DOC = 'docs/datasets/nuclei_birth_labels.md'


def label_sets(data):
    """List supplements for this cadence without changing the parent reader."""
    path = data.root/'additional_labels/index.json'
    if not path.exists():
        return []
    return [v for v in read(path)['label_sets'].values() if v['cadence']==data.cadence]


class NucleiBirthLabels:
    """Read a supplement snapshot. Partial collections require explicit opt-in."""
    def __init__(self, data, name, *, expected_fingerprint=None, allow_partial=False):
        entry = next((v for v in label_sets(data) if v['name']==name), None)
        if entry is None:
            raise KeyError(f'Unknown label set for {data.cadence}: {name}')
        self.root = data.root/entry['path']
        with (self.root/'update.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_SH)
            self.manifest = read(self.root/'manifest.json')
            self.fingerprint = self.manifest['fingerprint']
            if expected_fingerprint is not None and self.fingerprint!=expected_fingerprint:
                raise ValueError('Additional label fingerprint changed')
            if not allow_partial and self.manifest['state']!='complete':
                raise ValueError('Additional labels are partial; use allow_partial=True only for explicit partial inspection')
            self.db = connect(self.root/'labels.sqlite', readonly=True)
            self.db.execute('BEGIN')
            if self.db.execute("SELECT value FROM metadata WHERE key='fingerprint'").fetchone()[0]!=self.fingerprint:
                self.db.close();raise ValueError('Interrupted additional-label publication')
        # A later cumulative parent release may contain the same unchanged sources.
        try:
            for s in self.db.execute('SELECT source_id,payload_sha256 FROM sources'):
                if digest(data.source(s['source_id']))!=s['payload_sha256']:
                    raise ValueError('Additional labels no longer match the parent source: '+s['source_id'])
        except BaseException:
            self.db.close();raise

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def observations(self, *, target='appearance', horizon_ps=1., cohort_only=True,
                     at_risk_only=True, source_id=None):
        if target not in ('appearance','persistent') or horizon_ps not in self.manifest['horizons_ps']:
            raise ValueError('Unknown target or observation horizon')
        if cohort_only and self.manifest['state']!='complete':
            raise ValueError('Final deduplicated cohort is not sealed; inspect with cohort_only=False')
        where=['l.target=?','l.horizon_ps=?'];args=[target,horizon_ps]
        if cohort_only:where.append('o.cohort_included=1')
        if at_risk_only:where.append('o.at_risk=1')
        if source_id is not None:where.append('w.source_id=?');args.append(source_id)
        sql='''SELECT o.*,w.source_id,w.observation_time_ps,w.original_frame,
            s.source,s.dataset,s.ancestor,s.role,l.target,l.horizon_ps,l.label
            FROM observations o JOIN windows w USING(window_id)
            JOIN sources s USING(source_id) JOIN labels l USING(observation_id)
            WHERE '''+' AND '.join(where)+' ORDER BY o.observation_id'
        for row in self.db.execute(sql,args):yield dict(row)

    def window(self, window_id):
        row=self.db.execute('SELECT packet_path,packet_sha256 FROM windows WHERE window_id=?',(window_id,)).fetchone()
        if row is None:raise KeyError(window_id)
        path=self.root/row['packet_path']
        if sha(path)!=row['packet_sha256']:raise ValueError('Changed durable label packet')
        with np.load(path,allow_pickle=False) as values:return {k:values[k].copy() for k in values.files}

    def pin(self, path):
        write_json(path,self.manifest)


def schema(db):
    db.executescript('''
      CREATE TABLE metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL);
      CREATE TABLE sources (source_id TEXT PRIMARY KEY,source TEXT,dataset TEXT,
        ancestor TEXT,role TEXT,payload_sha256 TEXT,potential_json TEXT);
      CREATE TABLE windows (window_id TEXT PRIMARY KEY,source_id TEXT REFERENCES sources,
        observation_time_ps REAL,original_frame INTEGER,candidates INTEGER,
        packet_path TEXT,packet_sha256 TEXT,producer_receipt_sha256 TEXT);
      CREATE TABLE observations (observation_id TEXT PRIMARY KEY,window_id TEXT REFERENCES windows,
        center_atom_id INTEGER,at_risk INTEGER,history_clear INTEGER,
        first_step INTEGER,appearance_lower_ps REAL,appearance_upper_ps REAL,appearance_frame INTEGER,
        persistent_first_step INTEGER,persistent_lower_ps REAL,persistent_upper_ps REAL,persistent_frame INTEGER,
        input_hash TEXT,cohort_included INTEGER,canonical_id TEXT,cohort_weight REAL);
      CREATE TABLE labels (observation_id TEXT REFERENCES observations,target TEXT,
        horizon_ps REAL,label INTEGER,PRIMARY KEY(observation_id,target,horizon_ps));
      CREATE INDEX windows_source ON windows(source_id);
      CREATE INDEX observations_window ON observations(window_id);
      CREATE INDEX labels_horizon ON labels(target,horizon_ps);
    ''')


def binding(c):
    producer=resolve_path(c['producer']);path=producer/'technical/plan.json';p=read(path)
    if digest({k:v for k,v in p.items() if k!='identity'})!=p['identity']:
        raise ValueError('Appearance producer plan changed')
    if p['config']['dataset']!='al-nuclei-birth-010' or p['config']['fingerprint']!=c['parent_fingerprint']:
        raise ValueError('Unexpected parent dataset or fingerprint')
    if p['config']['persistence_frames']!=3 or p['config']['ptm_rmsd_cutoff']!=.1:
        raise ValueError('Different appearance criterion needs another label set')
    spec=dict(config=c,producer_identity=p['identity'],producer_plan_sha256=sha(path),
        exporter_sha256=sha(Path(__file__)),definitions_sha256=sha(Path(__file__).resolve().parents[3]/DOC))
    spec['identity']=digest(spec)
    return p,spec


def put_window(c,p,db,task,dest):
    source=next(s['source'] for s in p['sources'] if s['source_id']==task['source_id'])
    origin=resolve_path(p['config']['cache'])/'windows'/task['id']
    receipt=read(origin/'complete.json')
    if receipt['identity']!=digest([p['identity'],task]) or any(sha(origin/n)!=h for n,h in receipt['files'].items()):
        raise ValueError('Changed completed appearance packet: '+task['id'])
    selected=read(origin/'selection.json')
    with np.load(origin/'observations.npz',allow_pickle=False) as raw:
        values={k:raw[k].copy() for k in raw.files if k!='state'}
        hashes=[hashlib.sha256(x.tobytes()).hexdigest() for x in raw['state']]
    if not np.array_equal(values['atom'],selected['center_atom_ids']) or not np.array_equal(values['members'],selected['member_atom_ids']):
        raise ValueError('Observation/member identities disagree with sealed selection')
    if not np.array_equal(values['consumed'],selected['consumed']):raise ValueError('Consumed atom mask changed')
    frames=np.asarray(source['views']['010'],np.int64)[task['slots']]
    times=np.asarray(source['times_ps'],np.float64)[frames]
    if not np.allclose(np.diff(times),.1,atol=1e-8,rtol=0):raise ValueError('Nonexact label timeline')
    zero=task['slots'].index(task['slot'])
    if selected['frame']!=int(frames[zero]):raise ValueError('Present frame mismatch')
    # Check the stored labels independently before durable publication. No PTM rerun.
    solid=np.isin(values['labels'],[1,2,3])&values['consumed'][:,None,:]
    clear=~solid[:,zero].any(1);history_clear=~solid[:,:zero+1].any(axis=(1,2))
    future=solid[:,zero+1:];hits=future.any(2)
    first=np.where(hits.any(1),hits.argmax(1)+1,-1)
    steps=int(round(max(p['config']['horizons_ps'])/.1))
    sustained=np.stack([future[:,i:i+3].all(1).any(1) for i in range(steps)],axis=1)
    persistent=np.where(sustained.any(1),sustained.argmax(1)+1,-1)
    for name,expected in [('clear',clear),('history_clear',history_clear),('first_step',first),('persistent_first_step',persistent)]:
        if not np.array_equal(values[name],expected):raise ValueError('Stored onset definition mismatch: '+name)
    values.update(frames=frames,times_ps=times,present_index=np.asarray(zero,np.int64))
    relative='windows/'+task['id']+'.npz';target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        with np.load(target,allow_pickle=False) as saved:
            if set(saved.files)!=set(values) or any(not np.array_equal(saved[k],v) for k,v in values.items()):
                raise ValueError('Conflicting immutable supplemental window')
    else:
        temporary=target.with_suffix('.building.npz');np.savez_compressed(temporary,**values);temporary.replace(target)
    db.execute('INSERT INTO windows VALUES (?,?,?,?,?,?,?,?)',
        (task['id'],task['source_id'],float(times[zero]),int(frames[zero]),len(clear),relative,sha(target),sha(origin/'complete.json')))
    for i,atom in enumerate(values['atom']):
        key=task['id']+':'+str(int(atom));at_risk=bool(clear[i])
        def onset(step):
            if not at_risk or step<1:return (None,None,None,None)
            return (int(step),float(times[zero+step-1]),float(times[zero+step]),int(frames[zero+step]))
        db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
            (key,task['id'],int(atom),int(at_risk),int(history_clear[i]),
             *onset(first[i]),*onset(persistent[i]),hashes[i],None,None,None))
        for name,step in [('appearance',first[i]),('persistent',persistent[i])]:
            for horizon in p['config']['horizons_ps']:
                value=int(0<step<=round(horizon/.1)) if at_risk else None
                db.execute('INSERT INTO labels VALUES (?,?,?,?)',(key,name,horizon,value))


def seal_cohort(p,db,dest):
    source=resolve_path(p['config']['cache']);receipt=read(source/'complete.json')
    if receipt['identity']!=p['identity'] or any(sha(source/n)!=h for n,h in receipt['files'].items()):
        raise ValueError('Changed sealed appearance cohort')
    rows=read(source/'rows.json');aliases=read(source/'aliases.json')
    db.execute('UPDATE observations SET cohort_included=0,canonical_id=NULL,cohort_weight=NULL')
    for row in rows:
        old=db.execute('SELECT o.input_hash,o.at_risk,s.ancestor,s.role,w.source_id FROM observations o '
            'JOIN windows w USING(window_id) JOIN sources s USING(source_id) WHERE observation_id=?',(row['id'],)).fetchone()
        if old is None or old['input_hash']!=row['input_hash'] or old['at_risk']!=1 or any(old[k]!=row[k] for k in ['ancestor','role','source_id']):
            raise ValueError('Final cohort differs from durable labels: '+row['id'])
        db.execute('UPDATE observations SET cohort_included=1,canonical_id=?,cohort_weight=? WHERE observation_id=?',
            (row['id'],row['weight'],row['id']))
    for alias in aliases:
        if db.execute('UPDATE observations SET canonical_id=? WHERE observation_id=? AND at_risk=1',
            (alias['retained'],alias['id'])).rowcount!=1:raise ValueError('Unmatched duplicate alias')
    count=db.execute('SELECT count(*) FROM observations WHERE at_risk=1').fetchone()[0]
    if count!=len(rows)+len(aliases):raise ValueError('At-risk observation not accounted for by sealed cohort')
    for name in ('rows.json','aliases.json','complete.json'):
        target=dest/'provenance'/('cohort-'+name);shutil.copyfile(source/name,target)
    return len(rows)


def register(dest,manifest):
    """Separate index leaves the event catalogue and cadence manifests unchanged."""
    index_root=dest.parent
    with (index_root/'index.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        index=read(index_root/'index.json') if (index_root/'index.json').exists() else dict(schema_version=1,label_sets={})
        index['label_sets'][manifest['name']]=dict(name=manifest['name'],dataset=manifest['dataset'],cadence='010',
            path=str(dest.relative_to(index_root.parent)),state=manifest['state'],fingerprint=manifest['fingerprint'],
            parent_fingerprint=manifest['parent_fingerprint'],counts=manifest['counts'])
        write_json(index_root/'index.json',index)


def refresh(c,p,spec):
    source=resolve_path(p['config']['cache'])
    if not (source/'complete.json').exists():
        raise ValueError('Publish once after label collection is sealed; it is still running')
    dest=resolve_path(c['output']);dest.mkdir(parents=True,exist_ok=True)
    with (dest/'update.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        old=read(dest/'manifest.json') if (dest/'manifest.json').exists() else None
        if old and old['binding']!=spec:raise ValueError('Label publication contract changed; use a new version')
        if old and old['state']=='complete':
            register(dest,old);return old
        staging=dest/f'.labels-{os.getpid()}.sqlite'
        if staging.exists():raise FileExistsError('Unfinished staging database requires inspection: '+str(staging))
        if old:
            if sha(dest/'labels.sqlite')!=old['files']['labels.sqlite']:raise ValueError('Changed published labels database')
            shutil.copyfile(dest/'labels.sqlite',staging)
        db=connect(staging)
        if not old:
            schema(db);(dest/'provenance').mkdir(exist_ok=True)
            with NucleiBirthDataset('010') as data:
                if dest.parent.parent!=data.root:raise ValueError('Label output must be inside the consolidated dataset')
                for item in p['sources']:
                    s=item['source']
                    if digest(data.source(item['source_id']))!=digest(s):raise ValueError('Parent source changed')
                    db.execute('INSERT INTO sources VALUES (?,?,?,?,?,?,?)',
                        (item['source_id'],s['id'],s['dataset'],s['ancestry_group'],s['role'],digest(s),compact(s['potential'])))
            producer=resolve_path(c['producer'])
            for source,name in [(producer/'technical/plan.json','producer-plan.json'),
                (producer/'technical/nuclei-birth-dataset.json','parent-pin.json'),
                (producer/'technical/code/docs/metrics/motion_first_appearance.md','label-definitions.md'),
                (Path(__file__),'exporter.py'),(Path(__file__).resolve().parents[3]/DOC,'dataset-guide.md')]:
                shutil.copyfile(source,dest/'provenance'/name)
        seen={r[0] for r in db.execute('SELECT window_id FROM windows')};added=0
        for task in p['tasks']:
            if task['id'] in seen:continue
            receipt=resolve_path(p['config']['cache'])/'windows'/task['id']/'complete.json'
            if receipt.exists():put_window(c,p,db,task,dest);added+=1
        windows=db.execute('SELECT count(*) FROM windows').fetchone()[0]
        ready=windows==len(p['tasks']) and (resolve_path(p['config']['cache'])/'complete.json').exists()
        if not ready:raise ValueError('Sealed collection is missing completed window packets')
        if old and not added and not ready:
            db.close();staging.unlink();register(dest,old);return old
        cohort_rows=seal_cohort(p,db,dest) if ready else None
        state='complete' if ready else 'partial'
        window_records=[dict(r) for r in db.execute('SELECT window_id,packet_sha256,producer_receipt_sha256 FROM windows ORDER BY window_id')]
        fingerprint=digest(dict(binding=spec['identity'],windows=window_records,state=state,
            cohort_sha256=sha(dest/'provenance/cohort-complete.json') if ready else None))
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('fingerprint',?)",(fingerprint,));db.commit()
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok' or db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Supplement database integrity failed')
        counts=dict(windows=windows,expected_windows=len(p['tasks']),candidate_observations=db.execute('SELECT count(*) FROM observations').fetchone()[0],
            at_risk_observations=db.execute('SELECT count(*) FROM observations WHERE at_risk=1').fetchone()[0],cohort_observations=cohort_rows,
            sources_with_labels=db.execute('SELECT count(DISTINCT source_id) FROM windows').fetchone()[0],source_contract=len(p['sources']))
        db.close();files={'labels.sqlite':sha(staging)}
        files.update({str(f.relative_to(dest)):sha(f) for f in sorted((dest/'provenance').iterdir())})
        files.update({'windows/'+r['window_id']+'.npz':r['packet_sha256'] for r in window_records})
        manifest=dict(schema_version=1,name=c['label_set'],dataset=c['dataset_id'],cadence='010',cadence_ps=.1,
            parent_dataset='al-nuclei-birth-010',parent_fingerprint=c['parent_fingerprint'],binding=spec,
            fingerprint=fingerprint,state=state,counts=counts,horizons_ps=p['config']['horizons_ps'],files=files,
            targets=['appearance','persistent'],missing='Unprocessed windows absent; present-crystalline candidates have null targets, never negative labels.',
            observation_key=['source_id','original_frame','center_atom_id'],event_id=None,
            updated_at=datetime.now(timezone.utc).isoformat())
        staging.replace(dest/'labels.sqlite');write_json(dest/'manifest.json',manifest)
    register(dest,manifest)
    (dest/'README.md').write_text('# First crystalline appearance: additional Al birth labels\n\n'
        f'Status: **{state}**. {windows}/{len(p["tasks"])} windows, {counts["at_risk_observations"]} at-risk observations.\n\n'
        'These labels belong to source/frame/center observations, not one label per established nucleus. '
        'Unprocessed windows are missing; present-crystalline candidates have null targets. '
        'The parent event catalogue and its fingerprint are unchanged.\n\n'
        '[Dataset guide](provenance/dataset-guide.md) · [Scientific definitions](provenance/label-definitions.md)\n')
    return manifest


def watch(c):
    p,spec=binding(c);progress=resolve_path(c['producer'])/'technical/additional-label-publication.json'
    with recorded_stage(progress,stage='waiting_for_sealed_collection') as record:
        complete=resolve_path(p['config']['cache'])/'complete.json'
        while not complete.exists():
            producer=read(resolve_path(c['producer'])/'technical/launch.json')
            command=Path(f'/proc/{producer["pid"]}/cmdline')
            if not command.exists() or b'src.research.structural_response.appearance' not in command.read_bytes():
                raise RuntimeError('Appearance producer stopped before sealing; no supplemental label set was published')
            time.sleep(30)
        record.update(stage='publishing_once')
        result=refresh(c,p,spec)
        record.update(label_state=result['state'],counts=result['counts'],fingerprint=result['fingerprint'])
        print(compact(dict(state=result['state'],**result['counts'])),flush=True)


def launch(c):
    # A single completion-triggered export; no periodic dataset writes.
    p,spec=binding(c)
    tech=resolve_path(c['producer'])/'technical';receipt=tech/'additional-label-launch.json'
    if receipt.exists():raise FileExistsError('Additional-label publisher already launched')
    bundle=ExecutionBundle.freeze(REPO,tech/'additional-label-code',c,directories=('src',),files=((REPO/DOC,DOC),))
    command=[sys.executable,'-u','-m',MODULE,'watch','--config',str(bundle.config_path)]
    env=dict(os.environ,PCM_PROJECT_ROOT=str(REPO),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    with (tech/'additional-label-publication.log').open('a') as log:
        child=subprocess.Popen(command,cwd=bundle.root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(receipt,dict(owner='current response/nucleus-recovery chat; user requested permanent additional labels',
        pid=child.pid,process_group=child.pid,host=os.uname().nodename,allocation=os.environ.get('SLURM_JOB_ID'),
        command=command,started_unix=time.time(),output=str(resolve_path(c['output'])),binding=spec))
    print(compact(dict(pid=child.pid,state='waiting_for_sealed_collection',publishes_once=True)),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['refresh','watch','launch','status','verify'])
    parser.add_argument('--config',required=True);args=parser.parse_args();c=read(args.config)
    if args.stage=='refresh':
        p,spec=binding(c);value=refresh(c,p,spec);print(compact(dict(state=value['state'],counts=value['counts'])))
    elif args.stage in ('watch','launch'):globals()[args.stage](c)
    else:
        dest=resolve_path(c['output'])
        if args.stage=='status' and not (dest/'manifest.json').exists():
            print(compact(dict(state='waiting_for_sealed_collection',published=False)));return
        m=read(dest/'manifest.json')
        if args.stage=='verify':
            with (dest/'update.lock').open('a') as lock:
                fcntl.flock(lock,fcntl.LOCK_SH);m=read(dest/'manifest.json')
                if any(sha(dest/n)!=h for n,h in m['files'].items()):raise ValueError('Additional label checksum failure')
        print(compact(dict(state=m['state'],fingerprint=m['fingerprint'],counts=m['counts'],verified=args.stage=='verify')))


if __name__=='__main__':main()
