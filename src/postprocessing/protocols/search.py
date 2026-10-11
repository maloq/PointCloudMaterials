"""Detached all-Al cluster searches with separately versioned exact cadences."""
from src.postprocessing.protocols import threshold_tracker
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha,json_digest as digest,write_json
from src.experiment_runner.execution import ExecutionBundle,SlurmQueue,recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs,write_metric_rows
from src.project_runtime.paths import REPO,resolve_path
from src.postprocessing.io import read
from src.postprocessing.sources import prepare
from src.postprocessing import extraction as search_clusters

FAMILY='al_nuclei_search'


def load(c):
    p=read(resolve_path(c['output'])/'technical/plan.json')
    if p['config']!=c or digest({k:v for k,v in p.items() if k!='identity'})!=p['identity']:
        raise ValueError('Search population/configuration changed; use its frozen version')
    return p


def initialize(path):
    global PLAN
    PLAN=load(read(path))


def run_task(stage,task):
    function=search_clusters.extract if stage=='extract' else threshold_tracker.track
    return function(PLAN,task)


def lane(path,stage,index):
    c=read(path);p=load(c);tasks=p['tasks'] if stage=='extract' else p['jobs']
    tasks=[t for t in tasks if t['lane']==index]
    root=resolve_path(c['output'])/'technical'
    with recorded_stage(root/f'{stage}-{index}.json',total=len(tasks),completed=0,
                        node=os.uname().nodename,job=os.environ.get('SLURM_JOB_ID')) as progress:
        with ProcessPoolExecutor(max_workers=c['workers_per_lane'],mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize,initargs=(path,)) as executor:
            pending=[executor.submit(run_task,stage,t) for t in tasks]
            for i,future in enumerate(as_completed(pending),1):
                r=future.result();progress.update(completed=i,last_source=r['source'])
                print(json.dumps(dict(completed=i,total=len(tasks),source=r['source'],seconds=r['seconds'],
                    counts=r.get('counts'))),flush=True)


def match_events(left,right):
    """One-to-one onset-membership matching with overlapping observed lifetimes."""
    candidates=[]
    for i,a in enumerate(left):
        aa=set(a['first_members'])
        for j,b in enumerate(right):
            tolerance=max(a['cadence_ps'],b['cadence_ps'])
            if a['first_time_ps']>b['last_high_time_ps']+tolerance or b['first_time_ps']>a['last_high_time_ps']+tolerance:continue
            bb=set(b['first_members']);shared=len(aa&bb)
            overlap=shared/min(len(aa),len(bb))
            if shared>=4 and overlap>=.25:candidates.append((overlap,shared,i,j))
    taken_a=set();taken_b=set();matches=[]
    for overlap,shared,i,j in sorted(candidates,reverse=True):
        if i in taken_a or j in taken_b:continue
        taken_a.add(i);taken_b.add(j);a,b=left[i],right[j]
        matches.append(dict(left_track=a['track'],left_episode=a['episode'],right_track=b['track'],right_episode=b['episode'],
            shared_onset_atoms=shared,overlap_fraction=overlap,onset_shift_ps=b['first_time_ps']-a['first_time_ps'],
            fate_agrees=a['fate']==b['fate'],origin_agrees=a['origin']==b['origin']))
    return matches,len(left)-len(taken_a),len(right)-len(taken_b)


def collect(c,partial=False):
    p=load(c);root=resolve_path(c['output']);destination=root/'analyses'/('progress-v1' if partial else 'catalogue-v2')
    results=[];events=[];missing=[];by_source=defaultdict(dict)
    sources={s['id']:s for s in p['sources']}
    for task in p['jobs']:
        folder=root/'technical/sources'/task['source']/task['cadence']/task['detector']
        if not (folder/'complete.json').exists():missing.append(task);continue
        done=read(folder/'complete.json')
        if done['identity']!=digest(dict(plan=p['identity'],task=task)) or any(sha(folder/n)!=h for n,h in done['files'].items()):
            raise ValueError(f'Changed result {folder}')
        rows=read(folder/'events.json');by_source[task['source']][(task['cadence'],task['detector'])]=rows
        events.extend({k:v for k,v in e.items() if k not in ('first_members','peak_members')} for e in rows)
        s=sources[task['source']]
        results.append(dict(**task,**done['counts'],seconds=done['seconds'],dataset=s['dataset'],
            ancestor=s['ancestry_group'],source_origin=s['origin'],native_cadence_ps=s['native_cadence_ps']))
    if missing and not partial:
        write_json(root/'technical/missing-results.json',missing)
        raise RuntimeError(f'{len(missing)} source/protocol/detector results missing; completed results are preserved')
    comparisons=[];matches=[]
    for source,groups in by_source.items():
        pairs=[(('010',d),('030',d)) for d in c['detectors']]
        pairs += [((cad,'ptm'),(cad,'coherent')) for cad in c['cadences']]
        for ka,kb in pairs:
            if ka not in groups or kb not in groups:continue
            mm,ua,ub=match_events(groups[ka],groups[kb])
            meta=dict(source=source,left_cadence=ka[0],right_cadence=kb[0],left_detector=ka[1],right_detector=kb[1])
            comparisons.append(dict(**meta,left_events=len(groups[ka]),right_events=len(groups[kb]),
                matched=len(mm),unmatched_left=ua,unmatched_right=ub))
            matches.extend(dict(**meta,**m) for m in mm)
    summary=[]
    for cadence in c['cadences']:
        for detector in c['detectors']:
            rows=[e for e in events if e['cadence_ps']==c['cadences'][cadence] and e['detector']==detector]
            for origin in sorted({r['origin'] for r in rows}):
                selected=[r for r in rows if r['origin']==origin]
                summary.append(dict(cadence_ps=c['cadences'][cadence],detector=detector,origin=origin,
                    events=len(selected),sources=len({r['source'] for r in selected}),
                    ancestor_groups=len({r['ancestor'] for r in selected}),
                    established=sum(r['establishment_confirmation_ps'] is not None for r in selected),
                    dissolved=sum(r['terminal_status']=='dissolved' for r in selected)))
    for name,rows in [('events',events),('source-results',results),('counts',summary),
                      ('coverage',p['coverage']),('paired-comparisons',comparisons),('matched-events',matches)]:
        if rows:
            columns=tuple(dict.fromkeys(key for row in rows for key in row))
            write_metric_rows(rows,destination,family=FAMILY,name=name,columns=columns)
    lines=['# Al cluster birth search','',f'Finished source/protocol/detector jobs: {len(results)} of {len(p["jobs"])}.',
        '', 'All detected embryo excursions are retained independently of predictor-history eligibility and liquid controls.',
        'Counts depend on detector and cadence. Paired views, descendants, precision copies and shooting branches are not independent melts.',
        '', '| Cadence ps | Detector | Origin | Events | Established | Dissolved |', '|---|---|---|---:|---:|---:|']
    lines += [f'| {r["cadence_ps"]} | {r["detector"]} | {r["origin"]} | {r["events"]} | {r["established"]} | {r["dissolved"]} |' for r in summary]
    lines += ['', 'Establishment is an operational size/persistence event, not an estimate of criticality.',
        'Raw MD defines temporal outcomes. Relaxed views, static structures, absent payloads, incompatible cadences and unverified remote copies remain in coverage.',
        '', '[Events](tables/events.csv) · [Coverage](tables/coverage.csv) · [Paired comparisons](tables/paired-comparisons.csv) · [Definitions](tables/METRICS.md)','']
    (destination/'README.md').write_text('\n'.join(lines))
    if not partial:write_json(root/'technical/complete.json',dict(identity=p['identity'],jobs=len(results),events=len(events),finished_at=datetime.now(timezone.utc).isoformat()))


def submit(path):
    c=read(path);p=load(c);check_metric_docs(family=FAMILY)
    technical=resolve_path(c['output'])/'technical'
    if (technical/'launch.json').exists():raise ValueError('Already submitted; resume from recorded frozen code')
    bundle=ExecutionBundle.freeze(REPO,technical/'code',c,directories=('src','configs','docs/metrics'))
    record=dict(identity=p['identity'],code=str(bundle.root),jobs={})
    threads=int(c.get('descriptor_threads',1))
    if threads<1:raise ValueError('descriptor_threads must be positive')
    env=dict(PCM_PROJECT_ROOT=str(REPO),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OVITO_THREAD_COUNT=str(threads),QT_QPA_PLATFORM='offscreen')
    queue=SlurmQueue(technical,bundle,'src.postprocessing.protocols.search',env,technical/'launch.json',record,'AL-NUCLEI')
    with queue.submission():
        extraction=queue.submit('extract',[f'--array=0-{c["extraction_lanes"]-1}',f'--cpus-per-task={c["workers_per_lane"]*threads}',
            '--mem=24G',f'--time={c["time_limit"]}'])
        analysis=queue.submit('track',[f'--array=0-{c["analysis_lanes"]-1}',f'--cpus-per-task={c["workers_per_lane"]}',
            '--mem=24G',f'--time={c["time_limit"]}'],dependency='afterok:'+extraction)
        queue.submit('collect',['--cpus-per-task=1','--mem=24G','--time=04:00:00'],dependency='afterok:'+analysis)
    return record


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['prepare','extract','track','collect','progress','submit'])
    parser.add_argument('--config',required=True);parser.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    args=parser.parse_args();c=read(args.config)
    if args.stage=='prepare':print(json.dumps(prepare(c)[1],indent=2))
    elif args.stage=='submit':print(json.dumps(submit(args.config),indent=2))
    elif args.stage in ('collect','progress'):collect(c,partial=args.stage=='progress')
    else:lane(args.config,args.stage,args.index)


if __name__=='__main__':main()
