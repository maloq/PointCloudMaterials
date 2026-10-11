"""Restart all-Al searches from sealed structural labels with revised lineage rules."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import multiprocessing
import os

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, resolve_path, machine, path_snapshot
from src.postprocessing.io import read
from src.postprocessing import lineage as lineage_tracker, extraction as search_clusters

FAMILY='al_nuclei_lineage'


@path_snapshot()
def prepare(c):
    root=resolve_path(c['output']);previous=resolve_path(c['parent_output']);parent=read(previous/'technical/plan.json')
    if (root/'technical/launch.json').exists():raise ValueError('Release already launched')
    extraction_only=c.get('parent_stage')=='extraction'
    if not extraction_only and read(previous/'technical/complete.json')['identity']!=parent['identity']:raise ValueError('Unsealed parent search')
    if digest({k:v for k,v in parent.items() if k!='identity'})!=parent['identity']:raise ValueError('Changed parent plan')
    check_metric_docs(family=FAMILY)
    jobs=[];loads=[0.]*c['lanes'];ranked=[]
    sources={s['id']:s for s in parent['sources']}
    for old in parent['jobs']:
        if extraction_only:
            source=sources[old['source']]
            ranked.append((source['atom_count']*len(source['views'][old['cadence']]),old))
            continue
        folder=previous/'technical/sources'/old['source']/old['cadence']/old['detector']
        receipt=read(folder/'complete.json')
        if receipt['identity']!=digest(dict(plan=parent['identity'],task=old)):raise ValueError(str(folder))
        ranked.append((receipt['seconds'],old))
    for cost,old in sorted(ranked,key=lambda x:-x[0]):
        lane=min(range(c['lanes']),key=lambda k:loads[k]);loads[lane]+=cost
        jobs.append(dict(source=old['source'],cadence=old['cadence'],detector=old['detector'],lane=lane))
    p=dict(config=c,parent_identity=parent['identity'],parent_plan_sha256=sha(previous/'technical/plan.json'),
        producer_files=check_metric_docs(family=FAMILY)[FAMILY]['files'],jobs=jobs,
        source_records=len(parent['sources']),cadence_source_records={k:sum(k in s['views'] for s in parent['sources']) for k in parent['config']['cadences']},
        population_policy=c['input_contract']['population'],
        prepared_at=datetime.now(timezone.utc).isoformat())
    p['identity']=digest(p);write_json(root/'technical/plan.json',p)
    return p


def load(path):
    c=read(path);root=resolve_path(c['output']);p=read(root/'technical/plan.json')
    if p['config']!=c or digest({k:v for k,v in p.items() if k!='identity'})!=p['identity']:raise ValueError('Lineage plan changed')
    if check_metric_docs(family=FAMILY)[FAMILY]['files']!=p['producer_files']:raise ValueError('Use the recorded frozen tracker')
    parent_path=resolve_path(c['parent_output'])/'technical/plan.json'
    if sha(parent_path)!=p['parent_plan_sha256']:raise ValueError('Extraction population changed')
    parent=read(parent_path)
    if parent['identity']!=p['parent_identity']:raise ValueError('Extraction identity changed')
    return p,parent


def initialize(path):
    global PLAN,PARENT
    PLAN,PARENT=load(path)


def execute(task):
    return lineage_tracker.run(PLAN,PARENT,task)


def execute_extraction(task):
    return search_clusters.extract(PARENT,task)


def extract(path,index):
    p,parent=load(path);c=parent['config'];root=resolve_path(p['config']['output'])/'technical'
    tasks=[t for t in parent['tasks'] if t['lane']==index]
    with recorded_stage(root/f'extract-{index}.json',total=len(tasks),completed=0,
                        job=os.environ.get('SLURM_JOB_ID'),node=os.uname().nodename) as progress:
        with ProcessPoolExecutor(max_workers=c['workers_per_lane'],mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(path,)) as pool:
            futures=[pool.submit(execute_extraction,t) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                result=f.result();progress.update(completed=i,last_source=result['source'])
                print(json.dumps(dict(completed=i,total=len(tasks),**result)),flush=True)


def lane(path,index):
    p,parent=load(path);c=p['config'];root=resolve_path(c['output'])/'technical'
    tasks=[t for t in p['jobs'] if t['lane']==index]
    with recorded_stage(root/f'track-{index}.json',total=len(tasks),completed=0,job=os.environ.get('SLURM_JOB_ID'),node=os.uname().nodename) as progress:
        with ProcessPoolExecutor(max_workers=c['workers_per_lane'],mp_context=multiprocessing.get_context('spawn'),initializer=initialize,initargs=(path,)) as pool:
            futures=[pool.submit(execute,t) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                result=f.result();progress.update(completed=i,last_source=result['source'])
                print(json.dumps(dict(completed=i,total=len(tasks),**result)),flush=True)


def match(left,right):
    """Exact onset-atom candidate index, then temporal and overlap criteria."""
    index=defaultdict(list)
    for j,e in enumerate(right):
        for atom in e['first_members']:index[atom].append(j)
    candidates=[]
    for i,a in enumerate(left):
        hits=Counter(j for atom in a['first_members'] for j in index.get(atom,()))
        for j,shared in hits.items():
            if shared<4:continue
            b=right[j];tol=max(a['cadence_ps'],b['cadence_ps'])
            if a['first_time_ps']>b['last_high_time_ps']+tol or b['first_time_ps']>a['last_high_time_ps']+tol:continue
            overlap=shared/min(len(a['first_members']),len(b['first_members']))
            if overlap>=.25:candidates.append((overlap,shared,i,j))
    used_a=set();used_b=set();rows=[]
    for overlap,shared,i,j in sorted(candidates,reverse=True):
        if i in used_a or j in used_b:continue
        used_a.add(i);used_b.add(j);a=left[i];b=right[j]
        rows.append(dict(left_track=a['track'],right_track=b['track'],shared_onset_atoms=shared,
            overlap_fraction=overlap,onset_shift_ps=b['first_time_ps']-a['first_time_ps'],
            origin_agrees=a['origin']==b['origin'],fate_agrees=a['fate']==b['fate']))
    return rows,len(left)-len(used_a),len(right)-len(used_b)


def collect(path):
    p,parent=load(path);root=resolve_path(p['config']['output']);dest=root/'analyses/catalogue-v3'
    by_source=defaultdict(list);results=[]
    for task in p['jobs']:
        folder=root/'technical/sources'/task['source']/task['cadence']/task['detector']
        receipt=read(folder/'complete.json')
        if receipt['identity']!=digest(dict(plan=p['identity'],task=task)) or any(sha(folder/n)!=v for n,v in receipt['files'].items()):raise ValueError(f'Changed or incomplete tracking: {folder}')
        results.append(dict(**task,**receipt['counts'],seconds=receipt['seconds']))
        by_source[task['source']].append((task,folder))
    counts=defaultdict(Counter);comparisons=[];matched=[];audits=[];total=[0]
    source_map={s['id']:s for s in parent['sources']}

    def rows():
        for sid,entries in by_source.items():
            groups={}
            for task,folder in entries:
                es=read(folder/'events.json');groups[(task['cadence'],task['detector'])]=[e for e in es if e['record_kind']=='root']
                for e in es:
                    total[0]+=1;key=(e['cadence_ps'],e['detector'],e['origin'])
                    v=counts[key];v['events']+=1;v['established']+=e['establishment_confirmation_ps'] is not None
                    v['dissolved']+=e['terminal_status']=='dissolved'
                    v['dissolved_before_establishment']+=e['terminal_status']=='dissolved' and e['establishment_confirmation_ps'] is None
                    v['descendant_established']+=e['descendant_establishment_confirmation_ps'] is not None
                    yield {k:v for k,v in e.items() if k!='first_members'}
                trajectory=read(folder/'trajectory.json');s=source_map[sid]
                audits.append(dict(source=sid,cadence=task['cadence'],detector=task['detector'],
                    dataset=s['dataset'],ancestor=s['ancestry_group'],source_origin=s['origin'],
                    initial_largest_cluster=trajectory[0]['largest_cluster'],max_largest_cluster=max(t['largest_cluster'] for t in trajectory),
                    atom_count=s['atom_count'],liquid_emergences=sum(e['origin']=='emergence_from_liquid' for e in es),
                    established_liquid_emergences=sum(e['origin']=='emergence_from_liquid' and e['establishment_confirmation_ps'] is not None for e in es),
                    established_composites_from_subcritical=sum(e['record_kind']=='composite' and not e['inherited_established_crystal'] and e['establishment_confirmation_ps'] is not None for e in es),
                    initially_present_records=sum(e['origin']=='parent_present' for e in es)))
            pairs=[(('010',d),('030',d)) for d in parent['config']['detectors']]
            pairs += [((cad,'ptm'),(cad,'coherent')) for cad in parent['config']['cadences']]
            for ka,kb in pairs:
                if ka not in groups or kb not in groups:continue
                mm,ua,ub=match(groups[ka],groups[kb]);meta=dict(source=sid,left_cadence=ka[0],right_cadence=kb[0],left_detector=ka[1],right_detector=kb[1])
                comparisons.append(dict(**meta,left_events=len(groups[ka]),right_events=len(groups[kb]),matched=len(mm),unmatched_left=ua,unmatched_right=ub))
                matched.extend(dict(**meta,**m) for m in mm)
            write_json(root/'technical/collection-progress.json',dict(last_source=sid,event_rows=total[0],comparisons=len(comparisons)))
    write_metric_rows(rows(),dest,family=FAMILY,name='events')
    summaries=[dict(cadence_ps=k[0],detector=k[1],origin=k[2],**v) for k,v in sorted(counts.items())]
    for name,rs in [('counts',summaries),('source-results',results),('growth-audit',audits),('paired-comparisons',comparisons),('matched-events',matched),('coverage',parent['coverage'])]:
        if rs:write_metric_rows(rs,dest,family=FAMILY,name=name,columns=tuple(dict.fromkeys(k for r in rs for k in r)))
    text=['# Revised Al birth and lineage catalogue','',f'Completed {len(results)} of {len(p["jobs"])} tracking jobs.',p['population_policy'],
        '', 'Birth origin, independent establishment, establishment through coalescence and terminal fate are separate fields.',
        'Composite families are tracked but are not additional births. Parent-present and uncertain-history observations remain explicit.',
        'Source records, detector definitions and cadence views overlap; totals are not independent physical nuclei or nucleation rates.',
        '', '| Cadence ps | Detector | Origin | Records | Established | Dissolved before establishment |',
        '|---|---|---|---:|---:|---:|']
    text += [f'| {r["cadence_ps"]} | {r["detector"]} | {r["origin"]} | {r["events"]} | {r["established"]} | {r["dissolved_before_establishment"]} |' for r in summaries]
    text += ['', '[Growth audit](tables/growth-audit.csv) · [Paired comparisons](tables/paired-comparisons.csv) · [Definitions](tables/METRICS.md) · [Coverage](tables/coverage.csv)','']
    (dest/'README.md').write_text('\n'.join(text))
    write_json(root/'technical/complete.json',dict(identity=p['identity'],jobs=len(results),events=total[0],finished_at=datetime.now(timezone.utc).isoformat()))


def submit(path):
    p,parent=load(path);c=p['config'];root=resolve_path(c['output'])/'technical'
    if (root/'launch.json').exists():raise ValueError('Already submitted; use owned launch receipt')
    bundle=ExecutionBundle.freeze(REPO,root/'code',c,directories=('src','configs','docs/metrics'))
    settings=machine();settings['catalog']=str(bundle.root/'configs/datasets.json')
    write_json(root/'machine.json',settings)
    record=dict(identity=p['identity'],parent_identity=p['parent_identity'],code=str(bundle.root),jobs={},
        machine_sha256=sha(root/'machine.json'),catalog_sha256=sha(settings['catalog']))
    threads=int(parent['config'].get('descriptor_threads',1))
    if threads<1:raise ValueError('descriptor_threads must be positive')
    env=dict(PCM_PROJECT_ROOT=str(REPO),PCM_MACHINE_CONFIG=str(root/'machine.json'),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OVITO_THREAD_COUNT=str(threads),QT_QPA_PLATFORM='offscreen')
    queue=SlurmQueue(root,bundle,'src.postprocessing.search',env,root/'launch.json',record,'AL-LINEAGE3')
    with queue.submission():
        dependency=None
        if c.get('parent_stage')=='extraction':
            pc=parent['config']
            extraction=queue.submit('extract',[f'--array=0-{pc["extraction_lanes"]-1}',f'--cpus-per-task={pc["workers_per_lane"]*threads}','--mem=32G',f'--time={pc["time_limit"]}'])
            dependency='afterany:'+extraction
        job=queue.submit('track',[f'--array=0-{c["lanes"]-1}',f'--cpus-per-task={c["workers_per_lane"]}','--mem=64G',f'--time={c["time_limit"]}'],dependency=dependency)
        queue.submit('collect',['--cpus-per-task=1','--mem=64G','--time=12:00:00'],dependency='afterany:'+job)
    return record


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=['prepare','submit','extract','track','collect'])
    parser.add_argument('--config',required=True);parser.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    a=parser.parse_args()
    if a.stage=='prepare':
        p=prepare(read(a.config));print(json.dumps(dict(identity=p['identity'],sources=p['source_records'],jobs=len(p['jobs']),cadences=p['cadence_source_records'])))
    elif a.stage=='submit':print(json.dumps(submit(a.config),indent=2))
    elif a.stage=='track':lane(a.config,a.index)
    elif a.stage=='extract':extract(a.config,a.index)
    else:collect(a.config)


if __name__=='__main__':main()
