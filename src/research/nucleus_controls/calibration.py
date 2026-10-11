"""Matched-temperature continuation calibration on present-selected embryos."""
import argparse
from collections import defaultdict
from functools import lru_cache
import fcntl
import json
import os
import time

import numpy as np

from src.data.fixed_cohort.protocol import sha,write_json
from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.postprocessing.descriptors.ptm import full_labels
from src.research.nucleus_recovery import read,roots
from . import queue,negative,analysis
from .bias import paired_summary
from .calibration_data import FAMILY,prepare
from .negative_analysis import wilson

MODULE='src.research.nucleus_controls.calibration'


@lru_cache(maxsize=4)
def common_root(output):
    return resolve_path(output)/'technical/common'


def complete(c,plan,task):
    path=common_root(c['output'])/task['name']/'complete.json'
    if not path.exists():return False
    receipt=read(path)
    if receipt['plan_identity']!=plan['identity']:raise ValueError('Different calibration result')
    return True


def run_branch(c,plan,task,case,models,deadline):
    out,_,archive=roots(c);root=archive/'branches'/task['name']
    if not (root/'complete.json').exists():
        if not queue.run_branch(c,plan,task,case,models,deadline):return False
    raw=TemporalLAMMPSBinaryTrajectory.load(root/'trajectory_binary_float16');raw.verify_checksums()
    boxes=raw.box_high.astype(float)-raw.box_low.astype(float)
    q=np.stack([np.mod(v.astype(float),b) for v,b in zip(raw.positions,boxes)])
    labels=np.stack([full_labels(v,b,c['ptm']['rmsd_cutoff']) for v,b in zip(q,boxes)])
    result,graph,events,divergence=analysis.measure(c,case,task,q,boxes,labels)
    result.update(plan_identity=plan['identity'],measurement='common_float16')
    directory=out/'technical/common'/task['name'];directory.mkdir(parents=True,exist_ok=True)
    np.save(directory/'labels.npy',labels);np.savez_compressed(directory/'graph.npz',**graph)
    write_json(directory/'events.json',events);write_json(directory/'result.json',result)
    write_json(directory/'complete.json',dict(plan_identity=plan['identity'],result_sha256=sha(directory/'result.json'),
        archive_manifest_sha256=sha(raw.root/'manifest.json'),completed_unix=time.time()))
    return True


def summarize(c):
    out,_,_=roots(c);plan=read(out/'technical/plan.json')
    with (out/'technical/summary.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        refs={s['key']:read(out/'technical/reference'/f'{s["key"]}.json') for s in plan['cases']}
        groups=defaultdict(list);records=[];expected=defaultdict(list)
        for task in plan['tasks']:
            expected[task['case'],task['arms'][0]].append(task)
            if complete(c,plan,task):
                directory=out/'technical/common'/task['name'];r=read(directory/'result.json')
                if sha(directory/'result.json')!=read(directory/'complete.json')['result_sha256']:raise ValueError('Changed result')
                groups[task['case'],task['arms'][0]].append(r);records.append(r)
        paired=[];first=[]
        metrics=['any_new_nucleus','seed_established','crystal_fraction','largest_cluster_atoms']
        for (case,arm),tasks in expected.items():
            rs=groups[case,arm];ref=refs[case]
            if len(rs)!=len(tasks):continue
            for h in c['evaluation_horizons_ps']:
                base=dict(campaign='calibration',cohort='present_embryo',case=case,ancestry=ref['ancestry'],
                    temperature_K=ref['source_temperature_K'],arm=arm,time_ps=h,measurement='common_float16')
                for r in rs:
                    row=dict(**base)
                    for m in metrics:
                        row['reference_'+m]=ref[f'{m}_{h:g}ps'];row['restart_'+m]=r[f'{m}_{h:g}ps']
                    paired.append(row)
        summaries=paired_summary(c,paired,metrics,['arm','time_ps'])
        for arm in ['sourceNPT','sourceT']:
            for h in c['evaluation_horizons_ps']:
                pairs=[]
                for (case,a),tasks in expected.items():
                    if a!=arm:continue
                    wanted=min(tasks,key=lambda t:t['replicate'])['name']
                    rs=[r for r in groups[case,a] if r['task']==wanted]
                    if rs:pairs.append((refs[case][f'any_new_nucleus_{h:g}ps'],rs[0][f'any_new_nucleus_{h:g}ps']))
                n=len(pairs);old=sum(v[0] for v in pairs);new=sum(v[1] for v in pairs);lo,hi=wilson(new,n)
                first.append(dict(arm=arm,time_ps=h,paired_states=n,planned_states=len(plan['cases']),
                    reference_births=old,restart_births=new,restart_rate=new/n if n else None,restart_wilson95_low=lo,restart_wilson95_high=hi,
                    reference_only=sum(a and not b for a,b in pairs),restart_only=sum(b and not a for a,b in pairs)))
        if records:write_metric_rows(records,out,family=FAMILY,name='branches')
        write_metric_rows(list(refs.values()),out,family=FAMILY,name='saved-reference')
        if summaries:write_metric_rows(summaries,out,family=FAMILY,name='paired-comparison')
        write_metric_rows(first,out,family=FAMILY,name='first-shot-birth-rates')
        progress=dict(completed=len(records),planned=len(plan['tasks']),cases=len(plan['cases']),scope=plan['scope'],updated_unix=time.time())
        write_json(out/'technical/progress.json',progress)
        (out/'README.md').write_text('# Present-state-selected birth-rate calibration\n\n'
            f'{len(records)}/{len(plan["tasks"])} completed 12 ps continuations from {len(plan["cases"])} independent original sources.\n\n'
            'Selection uses current 8–63 atom embryos and past history, without future birth/fate labels. '
            'Both NPT and BAOAB use source temperature. All compared future labels use common float16 observation precision. '
            'NPT still resets the missing bath state; BAOAB still changes thermostat and box dynamics.\n\n'
            '[Paired comparison](tables/paired-comparison.csv) · [First-shot rates](tables/first-shot-birth-rates.csv) · '
            '[Saved futures](tables/saved-reference.csv) · [Definitions](tables/METRICS.md)\n\n'
            'A new shared three-hour deadline is recorded in technical/budget.json. Partial completion is provisional. '
            'This campaign overlaps prior ancestry and is not pooled as independent additional evidence.\n')
    return progress


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','launch','bounded','worker','summarize'])
    p.add_argument('--config',required=True);p.add_argument('--lane',type=int,default=0);a=p.parse_args();c=read(a.config)
    if a.stage=='prepare':prepare(c);print(json.dumps(summarize(c),indent=2))
    elif a.stage=='launch':negative.launch(c,module=MODULE,preparer=prepare,summarizer=summarize)
    elif a.stage=='worker':queue.worker(c,a.lane,summarizer=summarize,runner=run_branch,completion=complete)
    elif a.stage=='bounded':
        lanes=list(range(a.lane,a.lane+len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))))
        queue.bounded(c,a.config,module=MODULE,summarizer=summarize,lane_ids=lanes)
    else:print(json.dumps(summarize(c),indent=2))


if __name__=='__main__':main()
