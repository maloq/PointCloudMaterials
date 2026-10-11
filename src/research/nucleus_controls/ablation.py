"""Bounded box/thermostat ablation with frozen paired reference continuations."""
import argparse
from collections import defaultdict
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import time

import numpy as np

from src.data.fixed_cohort.protocol import sha,write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.research.nucleus_recovery import read,roots
from . import analysis,calibration,negative,queue
from .ablation_data import FAMILY,prepare
from .ablation_md import nvt
from .bias import paired_summary
from .negative_analysis import wilson

MODULE='src.research.nucleus_controls.ablation'
METRICS=['any_new_nucleus','local_new_nucleus','seed_established','crystal_fraction','largest_cluster_atoms']


def task_config(c,task):
    settings=deepcopy(c)
    if task['integrator']=='baoab':
        settings.update(timestep_fs=task['timestep_fs'],friction_fs=task['friction_fs'])
    elif task['integrator']=='nvt':
        settings['source_npt'].update(timestep_fs=task['timestep_fs'],momentum_every_steps=task['momentum_every_steps'])
    else:raise ValueError('Undeclared ablation integrator')
    return settings


def run_branch(c,plan,task,case,models,deadline):
    from src.research import nucleus_recovery_md as md
    settings=task_config(c,task)
    out,scratch,archive_root=roots(c)
    root=scratch/'branches'/task['name'];archive=archive_root/'branches'/task['name']
    if task['integrator']=='nvt' and not (archive/'complete.json').exists():
        root.mkdir(parents=True,exist_ok=True);archive.mkdir(parents=True,exist_ok=True)
        (root/'technical').mkdir(exist_ok=True)
        potential_files=md.files(c,root)
        for key,p in potential_files.items():
            if sha(p)!=plan['potential_sha256'][key]:raise ValueError('Changed pinned MEAM file')
        state=case['state'];initial_path=Path(state['initial_path'])
        if sha(initial_path)!=state['initial_sha256']:raise ValueError('Changed initial state')
        with np.load(initial_path) as values:initial={k:values[k].copy() for k in values.files}
        provenance=dict(task=task,case=case['key'],plan_identity=plan['identity'],
            original_source=case['raw'],original_frame=state['start_frame'],initial_sha256=state['initial_sha256'],
            encoder_input=None,predictor_input=None,physical_temperature_K=task['temperature_K'],
            states='native NVT float64 integration; float32 GPU forces; float64 scalar energy; thermostat state reset',
            momentum_removal_ps=.3,thermostat_ps=settings['source_npt']['thermostat_ps'],box='fixed initial cell',
            velocity_source='actual simultaneous saved velocities; no redraw, relaxation or perturbation',
            pid=os.getpid(),host=os.uname().nodename,allocation=os.environ.get('SLURM_JOB_ID'))
        write_json(root/'input.json',provenance);md.archive_file(root/'input.json',archive/'input.json')
        if not (root/'integration-complete.json').exists():
            observations=md.arrays(root,state['stop_frame']-state['start_frame']+1,case['atom_count'])
            if not nvt(settings,plan,task,state,case,root,archive,initial,observations,models,deadline):return False
            write_json(root/'integration-complete.json',dict(task=task['name'],plan_identity=plan['identity'],completed_unix=time.time()))
            del observations
        if not (root/'result.json').exists():analysis.analyze(settings,plan,task,case,root)
        md.publish(settings,plan,task,state,case,root,archive,initial)
    # Reuse unchanged BAOAB and the common-float16 classification producer.
    # Completed NVT archives bypass that producer's native integration dispatch.
    return calibration.run_branch(settings,plan,task,case,models,deadline)


def summarize(c):
    out,_,_=roots(c);plan=read(out/'technical/plan.json')
    with (out/'technical/summary.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if sha(out/'technical/reused.json')!=plan['reused_sha256']:raise ValueError('Changed reused results')
        reused=read(out/'technical/reused.json')
        refs=[read(out/'technical/reference'/f'{s["key"]}.json') for s in plan['cases']]
        records=[]
        for task in plan['tasks']:
            if not calibration.complete(c,plan,task):continue
            directory=out/'technical/common'/task['name'];r=read(directory/'result.json')
            if sha(directory/'result.json')!=read(directory/'complete.json')['result_sha256']:
                raise ValueError('Changed new branch result')
            records.append(dict(r,reused=False))
        groups=defaultdict(list)
        for r in refs+reused+records:groups[r['case'],r['arm']].append(r)
        required={'saved_reference':1,'npt_3fs':1,'baoab_100fs':2,
                  **{a['name']:a['replicates'] for a in c['arms']}}
        comparisons=[dict(name='saved_vs_'+arm,reference='saved_reference',restart=arm)
                     for arm in required if arm!='saved_reference']+c['contrasts']
        pairs=[];first=[];status=[]
        for contrast in comparisons:
            a,b=contrast['reference'],contrast['restart']
            for case in plan['cases']:
                left=groups[case['key'],a];right=groups[case['key'],b]
                if len(left)!=required[a] or len(right)!=required[b]:continue
                for h in c['evaluation_horizons_ps']:
                    row=dict(contrast=contrast['name'],reference_arm=a,restart_arm=b,time_ps=h,
                        ancestry=case['ancestry'],temperature_K=case['temperature_K'])
                    for metric in METRICS:
                        row['reference_'+metric]=float(np.mean([r[f'{metric}_{h:g}ps'] for r in left]))
                        row['restart_'+metric]=float(np.mean([r[f'{metric}_{h:g}ps'] for r in right]))
                    pairs.append(row)
        summaries=paired_summary(c,pairs,METRICS,['contrast','reference_arm','restart_arm','time_ps'])
        for arm,shots in required.items():
            status.append(dict(arm=arm,completed=sum(len(groups[s['key'],arm]) for s in plan['cases']),
                planned=len(plan['cases'])*shots,reused=arm in ('saved_reference','npt_3fs','baoab_100fs')))
            for h in c['evaluation_horizons_ps']:
                pairs=[]
                for case in plan['cases']:
                    rr=[r for r in groups[case['key'],arm] if r['replicate']==0]
                    if rr:pairs.append((groups[case['key'],'saved_reference'][0][f'any_new_nucleus_{h:g}ps'],rr[0][f'any_new_nucleus_{h:g}ps']))
                n=len(pairs);births=sum(v[1] for v in pairs);lo,hi=wilson(births,n)
                first.append(dict(arm=arm,time_ps=h,paired_states=n,planned_states=len(plan['cases']),
                    reference_births=sum(v[0] for v in pairs),restart_births=births,
                    restart_rate=births/n if n else None,restart_wilson95_low=lo,restart_wilson95_high=hi,
                    reference_only=sum(a and not b for a,b in pairs),restart_only=sum(b and not a for a,b in pairs)))
        for name,rows in [('branches',reused+records),('saved-reference',refs),('paired-comparison',summaries),('first-shot-birth-rates',first)]:
            if rows:write_metric_rows(rows,out,family=FAMILY,name=name)
        progress=dict(completed=len(records),planned=len(plan['tasks']),cases=len(plan['cases']),reused_continuations=len(reused),
            arms=status,scope=plan['scope'],updated_unix=time.time())
        write_json(out/'technical/progress.json',progress)
        if len(records)==len(plan['tasks']):
            complete_path=out/'technical/complete.json'
            if not complete_path.exists():write_json(complete_path,dict(plan_identity=plan['identity'],completed_unix=time.time(),branches=len(records)))
        (out/'README.md').write_text('# Box and thermostat continuation controls\n\n'
            f'{len(records)}/{len(plan["tasks"])} new 12 ps branches complete; 192 completed calibration branches reused. '
            'All conditions use the same 64 states/ancestors and original temperatures.\n\n'
            'NPT 3 fs versus NVT 3 fs changes box dynamics. NVT 3 fs versus 1 fs checks timestep. '
            'BAOAB 100/300/1000 fs changes damping with common random streams. '
            'NVT versus BAOAB additionally differs in integration, precision and momentum removal.\n\n'
            '[Paired contrasts](tables/paired-comparison.csv) · [First-shot rates](tables/first-shot-birth-rates.csv) · '
            '[Definitions](tables/METRICS.md) · [Live progress](technical/progress.json)\n\n'
            'Nine-hour shared deadline includes queue wait; partial comparisons use only matched complete states. '
            'No new independent ancestry. Dense, shooting, million-atom and relaxed data are outside this diagnostic.\n')
    return progress


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','launch','bounded','worker','summarize'])
    p.add_argument('--config',required=True);p.add_argument('--lane',type=int,default=0)
    a=p.parse_args();c=read(a.config)
    if a.stage=='prepare':prepare(c);print(json.dumps(summarize(c),indent=2))
    elif a.stage=='launch':negative.launch(c,module=MODULE,preparer=prepare,summarizer=summarize)
    elif a.stage=='worker':queue.worker(c,a.lane,summarizer=summarize,runner=run_branch,completion=calibration.complete)
    elif a.stage=='bounded':
        lanes=list(range(a.lane,a.lane+len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        queue.bounded(c,a.config,module=MODULE,summarizer=summarize,lane_ids=lanes)
    else:print(json.dumps(summarize(c),indent=2))


if __name__=='__main__':main()
