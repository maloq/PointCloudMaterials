"""Source-level summaries for the negative-only supplement; no matched contrasts."""
from collections import defaultdict
import fcntl
import math
import time

import numpy as np

from src.data.fixed_cohort.protocol import write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.research.nucleus_recovery import read, roots
from .negative_data import FAMILY


def wilson(successes,n):
    if not n:return None,None
    z=1.959963984540054;p=successes/n;d=1+z*z/n
    center=(p+z*z/(2*n))/d
    radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0.,center-radius),min(1.,center+radius)


def summarize(c):
    out,_,archive=roots(c);plan=read(out/'technical/plan.json')
    with (out/'technical/summary.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        records=[];expected=defaultdict(list);groups=defaultdict(list)
        for t in plan['tasks']:
            case=next(s for s in plan['cases'] if s['key']==t['case'])
            expected[case['cohort'],t['arms'][0],case['key']].append(t)
            root=archive/'branches'/t['name']
            if not (root/'complete.json').exists():continue
            r=read(root/'result.json')
            if r['plan_identity']!=plan['identity']:raise ValueError('Result belongs to another plan')
            records.append(r);groups[r['cohort'],r['arm'],r['case']].append(r)
        references=[]
        for case in plan['cases']:
            path=out/'technical/reference'/f'{case["key"]}.json'
            if path.exists():references.append(read(path))
        metrics=[f'{m}_{h:g}ps' for h in c['evaluation_horizons_ps'] for m in
            ['any_new_nucleus','local_new_nucleus','seed_established','crystal_fraction','largest_cluster_atoms','reached_10pct_crystal']]+['seed_dissolved']
        summaries=[];first_shots=[]
        for cohort,arm in sorted({k[:2] for k in expected}):
            keys=[k for k in expected if k[:2]==(cohort,arm)]
            ready=[groups[k] for k in keys if len(groups[k])==len(expected[k])]
            for metric in metrics:
                values=np.array([np.mean([r[metric] for r in rs]) for rs in ready if rs[0][metric] is not None])
                row=dict(cohort=cohort,arm=arm,metric=metric,value=float(values.mean()) if len(values) else None,
                    ci95_low=None,ci95_high=None,interval='source bootstrap; undefined for constant samples',
                    completed_cases=len(ready),planned_cases=len(keys),metric_cases=len(values),
                    completed_continuations=sum(len(groups[k]) for k in keys),planned_continuations=sum(len(expected[k]) for k in keys))
                if len(values)>1 and np.ptp(values)>0:
                    rng=np.random.default_rng(c['seed'])
                    draws=values[rng.integers(len(values),size=(c['bootstrap_draws'],len(values)))].mean(1)
                    row['ci95_low'],row['ci95_high']=map(float,np.quantile(draws,[.025,.975]))
                summaries.append(row)
            # One predeclared first shot per ancestor avoids counting repeats as
            # independent evidence, and gives nonzero bounds for zero events.
            first=[]
            for k in keys:
                wanted=min(expected[k],key=lambda t:t['replicate'])['name']
                first.extend(r for r in groups[k] if r['task']==wanted)
            for h in c['evaluation_horizons_ps']:
                metric=f'any_new_nucleus_{h:g}ps';successes=sum(r[metric] for r in first);n=len(first)
                lo,hi=wilson(successes,n)
                first_shots.append(dict(cohort=cohort,arm=arm,horizon_ps=h,successes=successes,
                    independent_starting_states=n,planned_states=len(keys),rate=successes/n if n else None,
                    wilson95_low=lo,wilson95_high=hi,selection='first predeclared shot only'))
        if records:write_metric_rows(records,out,family=FAMILY,name='branches')
        if references:
            columns=tuple(dict.fromkeys(k for r in references for k in r))
            write_metric_rows(references,out,family=FAMILY,name='saved-reference',columns=columns)
        write_metric_rows(summaries,out,family=FAMILY,name='cohorts')
        write_metric_rows(first_shots,out,family=FAMILY,name='first-shot-birth-rates')
        progress=dict(completed=len(records),planned=len(plan['tasks']),cases=len(plan['cases']),
            references_measured=len(references),scope=plan['scope'],updated_unix=time.time())
        write_json(out/'technical/progress.json',progress)
        (out/'README.md').write_text('# Additional negative continuations\n\n'
            f'{len(records)}/{len(plan["tasks"])} completed 12 ps runs. '
            f'{plan["scope"]["new_liquid_ancestors"]} additional random-liquid ancestors; '
            f'{plan["scope"]["reused_dissolved_ancestors"]} previously selected failed-embryo states receive fresh noise shots.\n\n'
            'New liquid starts exclude every ancestor in the previous matched campaign. Future outcomes are not used for liquid selection. '
            'Repeated embryo shots add no independent starting states. New summaries remain separate from the previous matched comparison.\n\n'
            '[All-shot source means](tables/cohorts.csv) · [Independent first-shot birth rates](tables/first-shot-birth-rates.csv) · '
            '[Definitions](tables/METRICS.md)\n\n'
            'A shared three-hour deadline includes scheduler wait. Partial coverage is provisional and unfinished runs are excluded. '
            'Other Al cadences and cohorts remain outside this supplement.\n')
    return progress
