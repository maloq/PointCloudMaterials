"""Seal box and thermostat controls on the already selected 64-state cohort."""
from copy import deepcopy
from pathlib import Path
import shutil

import numpy as np

from src.data.fixed_cohort.protocol import digest,sha,write_json
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.nucleus_recovery import read,roots

FAMILY='nucleus_dynamics_ablation'
ALIASES={'sourceNPT':'npt_3fs','sourceT':'baoab_100fs'}


def prepare(c):
    check_metric_docs(family=FAMILY)
    out,_,archive=roots(c);path=out/'technical/plan.json'
    if path.exists():
        plan=read(path)
        if plan['config']!=c:raise ValueError('Existing dynamics ablation recipe differs')
        return plan
    parent=resolve_path(c['parent_analysis']);old=read(parent/'technical/plan.json')
    if old['identity']!=c['parent_identity']:raise ValueError('Changed calibration cohort')
    for key in ['duration_ps','evaluation_horizons_ps','source_npt','mass_amu','timestep_fs','meam_files',
                'ptm','lineage','establishment_atoms','persistence_ps','site_radius_A','dissolution_confirmation_frames']:
        if c[key]!=old['config'][key]:raise ValueError('Changed matched protocol field: '+key)
    cases=deepcopy(old['cases']);tasks=[];reused=[];inputs=[]
    def pin(p):
        inputs.append(dict(path=str(p),sha256=sha(p)))
    pin(parent/'technical/plan.json')
    for task in old['tasks']:
        directory=parent/'technical/common'/task['name']
        receipt=read(directory/'complete.json');result=read(directory/'result.json')
        if receipt['plan_identity']!=old['identity'] or sha(directory/'result.json')!=receipt['result_sha256']:
            raise ValueError('Incomplete or changed common-precision reference: '+task['name'])
        if result['measurement']!='common_float16':raise ValueError('Wrong comparison observation precision')
        pin(directory/'complete.json');pin(directory/'result.json')
        result.update(arm=ALIASES[result['arm']],reused=True)
        reused.append(result)
    if len(reused)!=192 or len(cases)!=64 or len({s['ancestry'] for s in cases})!=64:
        raise ValueError('Unexpected frozen reference population')
    seed_map={(t['case'],t['replicate']):t['seed'] for t in old['tasks'] if t['integrator']=='baoab'}
    for index,case in enumerate(cases):
        initial=Path(case['state']['initial_path'])
        if sha(initial)!=case['state']['initial_sha256']:raise ValueError('Changed original Q,V: '+case['key'])
        with np.load(initial) as values:
            if values['q'].dtype!=np.float32 or values['v_A_ps'].dtype!=np.float32:
                raise ValueError('Wrong decoded initial-state precision')
            if not all(np.isfinite(values[k]).all() for k in ['q','v_A_ps','box']):
                raise ValueError('Nonfinite original Q,V')
        pin(initial)
        target=out/'technical/reference'/f'{case["key"]}.json';target.parent.mkdir(parents=True,exist_ok=True)
        source=parent/'technical/reference'/target.name
        pin(source);shutil.copy2(source,target)
        for arm_index,arm in enumerate(c['arms']):
            for replicate in range(arm['replicates']):
                seed=seed_map[case['key'],replicate] if arm['integrator']=='baoab' else None
                tasks.append(dict(name=case['key']+'-'+arm['name']+f'-r{replicate}',case=case['key'],state=case['key'],
                    integrator=arm['integrator'],variant='as_saved',temperature_K=case['temperature_K'],
                    arms=[arm['name']],replicate=replicate,seed=seed,priority=(replicate,index,arm_index),
                    timestep_fs=arm['timestep_fs'],friction_fs=arm.get('friction_fs'),
                    momentum_every_steps=round(.3*1000/arm['timestep_fs']) if arm['integrator']=='nvt' else None))
    tasks.sort(key=lambda t:t['priority'])
    write_json(out/'technical/reused.json',reused)
    plan=dict(config=c,cases=cases,tasks=tasks,parent_plan_identity=old['identity'],inputs=inputs,
        reused_sha256=sha(out/'technical/reused.json'),potential_sha256=old['potential_sha256'],
        scope=dict(independent_ancestors=64,new_independent_ancestry_created=0,exact_starts_reused=64,
            selection='All 64 previously sealed present-selected states; no outcome-dependent subsampling',
            reused_continuations=192,new_continuations=len(tasks),
            deferred=['other Al cadences','dense and shooting descendants','million-atom data','relaxed views']))
    plan['identity']=digest(plan)
    write_json(path,plan);write_json(archive/'plan.json',plan)
    write_json(out/'technical/preparation-audit.json',dict(plan_identity=plan['identity'],initials_checked=64,
        branches_reused=192,new_branches=len(tasks),encoder_inputs=None,predictor_inputs=None,
        common_random_numbers='Same per-state replicate seed as parent BAOAB 100fs; reused across damping settings'))
    return plan
