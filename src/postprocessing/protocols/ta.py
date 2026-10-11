"""CPU queue for full-cell Ta births and disappearing crystal embryos."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows, snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from src.postprocessing.protocols.shooting_data import read
from src.postprocessing.protocols.ta_data import prepare, load
from src.postprocessing.protocols.ta_fates import process, extract_frame

FAMILY='ta_births'


def initialize(path):
    global PLAN
    PLAN=load(read(path))


def source_worker(item):
    with recorded_stage(resolve_path(PLAN['config']['output'])/f'technical/sources/{item["id"]}/stage.json',
                        source=item['id'],job=os.environ.get('SLURM_JOB_ID')):
        return process(PLAN,item)


def worker(path,index):
    c=read(path);p=load(c)
    selected=sorted((s for s in p['sources'] if s['lane']==index),
                    key=lambda s:(not bool(s['legacy']),s['atom_count'],s['id']))
    with recorded_stage(resolve_path(c['output'])/f'technical/lane-{index}.json',total=len(selected),completed=0,
                        job=os.environ.get('SLURM_JOB_ID'),node=os.uname().nodename) as progress:
        with ProcessPoolExecutor(max_workers=c['workers_per_lane'],mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize,initargs=(path,)) as pool:
            pending=[pool.submit(source_worker,s) for s in selected]
            for number,future in enumerate(as_completed(pending),1):
                value=future.result()
                progress.update(completed=number,last_source=value['source'],last_counts=value['counts'])
                print(json.dumps(dict(source=value['source'],counts=value['counts'],seconds=value['seconds'])),flush=True)


def collect(c):
    p=load(c);root=resolve_path(c['output']);transient=[];established=[];branches=[]
    for item in p['sources']:
        folder=root/'technical/sources'/item['id'];r=read(folder/'complete.json')
        if r['identity']!=digest(dict(plan=p['identity'],source=item['id'])) or any(sha(folder/n)!=h for n,h in r['files'].items()):
            raise ValueError(f'Incomplete/changed Ta source: {item["id"]}')
        transient.extend(read(folder/'transient-episodes.json'));established.extend(read(folder/'established-fates.json'))
        branches.append(dict(source=item['id'],campaign=item['campaign'],branch=item['branch_id'],parent=item['parent_id'],
            parent_sha256=item['parent_sha256'],ancestry_group=item['ancestry_group'],atoms=item['atom_count'],
            frames=item['frame_count'],duration_ps=item['branch_observed_duration_ps'],reused_ptm=r['reused_ptm'],
            **r['actual_durations'],**r['counts']))
    dest=root/'analyses/births-fates-v1';snapshot_metric_docs(dest,FAMILY)
    for name,rows in [('transient-episodes',transient),('established-fates',established),('branches',branches)]:
        if rows:write_metric_rows(rows,dest,family=FAMILY,name=name)
    counts=[]
    populations=[('establishments',established,'kind'),('established_fates',established,'fate_label'),
        ('unestablished_episodes',transient,'terminal_status'),
        ('brief_failed_candidates',[r for r in transient if r['brief_failed_candidate']],'terminal_status'),
        ('sustained_failed_candidates',[r for r in transient if r['primary_failed_candidate']],'terminal_status')]
    for parent in ['all',*sorted({s['parent_id'] for s in p['sources']})]:
        for population,rows,field in populations:
            selected=[r for r in rows if parent=='all' or r['parent_id']==parent]
            for label in sorted({r[field] for r in selected}):
                match=[r for r in selected if r[field]==label]
                counts.append(dict(parent=parent,population=population,label=label,count=len(match),
                    futures=len({r['source'] for r in match}),parents=len({r['parent_sha256'] for r in match}),
                    ancestry_groups=len({r['lineage'] for r in match})))
    if counts:write_metric_rows(counts,dest,family=FAMILY,name='counts')
    lines=['# Ta births and nucleus fates','',
        'Complete:30 full-cell trajectories from six archived parents; one conservative preparation-ancestry group.',
        '24 new shooting futures plus six earlier futures. Full-cell 0.5ps observations, existing Ta spatial cutoffs.',
        'Parent-present embryo growth, new isolated establishments, interface-associated establishments and uncertain origins remain separate.','',
        '| Population | Label | Events | Futures | Parents |','|---|---|---:|---:|---:|']
    lines += [f'| {r["population"]} | {r["label"]} | {r["count"]} | {r["futures"]} | {r["parents"]} |' for r in counts if r['parent']=='all']
    lines += ['', 'These are correlated, finite-observation harvest counts, not independent nuclei or nucleation rates.',
        'PTM FCC/HCP/BCC is the declared crystal definition; this does not label all possible Ta ordered phases.',
        'Strict crystal-free history eligibility, liquid controls and predictor fitting remain separate.','',
        '[Counts](tables/counts.csv) · [Established fates](tables/established-fates.csv) · [Transient episodes](tables/transient-episodes.csv) · [Definitions](tables/METRICS.md)','']
    (dest/'README.md').write_text('\n'.join(lines))
    write_json(root/'technical/complete.json',dict(identity=p['identity'],futures=len(branches),counts=counts))
    (root/'README.md').write_text('# Ta birth search\n\n[Completed analysis](analyses/births-fates-v1/README.md)\n')


def submit(path):
    c=read(path);check_metric_docs(family=FAMILY);p=load(c)
    technical=resolve_path(c['output'])/'technical'
    if (technical/'launch.json').exists():raise ValueError('Already submitted; resume using frozen source')
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,technical/'code',c,directories=('src','configs','docs/metrics'))
    record=dict(identity=p['identity'],code=str(bundle.root),futures=len(p['sources']),jobs={})
    queue=SlurmQueue(technical,bundle,'src.postprocessing.protocols.ta',
        dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             OVITO_THREAD_COUNT=str(c['ptm_threads']),QT_QPA_PLATFORM='offscreen'),technical/'launch.json',record,'TA-BIRTH')
    with queue.submission():
        job=queue.submit('worker',[f'--array=0-{c["lanes"]-1}',
            f'--cpus-per-task={c["workers_per_lane"]*c["ptm_threads"]}',f'--mem={c["lane_memory"]}',f'--time={c["time_limit"]}'])
        queue.submit('collect',['--cpus-per-task=1','--mem=16G','--time=01:00:00'],dependency='afterok:'+job)
    return record


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['prepare','source','frame','worker','collect','submit'])
    parser.add_argument('--config',required=True);parser.add_argument('--source-id');parser.add_argument('--frame',type=int,default=0)
    parser.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    args=parser.parse_args();c=read(args.config)
    if args.stage=='prepare':
        check_metric_docs(family=FAMILY);prepare(c)
        print(json.dumps(read(resolve_path(c['output'])/'technical/inventory.json'),indent=2))
    elif args.stage=='submit':print(json.dumps(submit(args.config),indent=2))
    elif args.stage=='worker':worker(args.config,args.index)
    elif args.stage in ('source','frame'):
        p=load(c);item=next(s for s in p['sources'] if s['id']==args.source_id)
        value=process(p,item) if args.stage=='source' else extract_frame(p,item,args.frame)
        print(json.dumps(value,indent=2))
    else:
        with recorded_stage(resolve_path(c['output'])/'technical/collector.json',job=os.environ.get('SLURM_JOB_ID')):
            collect(c)


if __name__=='__main__':main()
