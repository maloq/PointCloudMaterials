"""Dense fixed-grid held-out evaluation; no encoder fitting or outcome sampling."""
import argparse
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from src.data.structural_pretraining.prepare import digest, file_hash, save_json
from src.project_runtime.paths import resolve_path
from src.training_methods.shared_pretraining.queue import deadline_for_job
from . import assay
from .availability import fatal_failures, matched_assay_mask, requested_assay_frames
from .prepare import produce
from .queue import claim
from .selection import selected_runs,evaluation_exclusions


BASELINES = ('geometry_hot', 'geometry_cold', 'original_geometry', 'conditions')


def population_audit(pop, frames, onsets):
    """Count local atom onsets separately from independent simulation sources."""
    result = {}
    for role in np.unique(pop['role']):
        mask = pop['role'] == role
        positive = mask & (onsets > frames) & (onsets-frames <= 16)
        keys = np.c_[pop['source'][positive], pop['rows'][positive, 2], onsets[positive]]
        result[str(role)] = dict(windows=int(mask.sum()), sources=len(np.unique(pop['source'][mask])),
            positive_windows_12ps=int(positive.sum()), distinct_local_onsets_12ps=len(np.unique(keys, axis=0)),
            event_sources_12ps=len(np.unique(pop['source'][positive])))
    return result


def freeze(recipe):
    root = resolve_path(recipe['output']).resolve(); path = root/'technical/plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['evaluation_recipe'] != recipe:raise ValueError('Large-test recipe changed')
        return plan
    base_path = resolve_path(recipe['training_plan']).resolve()
    base = json.loads(base_path.read_text()); c = base['config']
    original = json.loads(resolve_path(c['assay_plan']).read_text())
    if recipe['heldout_stride_frames'] != 16:
        raise ValueError('This protocol uses disjoint 12 ps forecast windows')
    grid = list(range(original['anchors'][0], original['anchors'][-1]+1, 16))
    roles = dict(train=c['frames'], selection=grid, calibration=grid, test=grid)
    sources = [dict(s, pilot_fit=False) for s in base['sources']]
    if len({s['lineage'] for s in sources}) != len(sources):
        raise ValueError('Evaluation sources have shared simulation ancestry')
    checkpoints = {r['name']:str(base_path.parent/'runs'/r['name']) for r in c['runs']}
    runs = list(c['runs'])
    for alias, arm in [('pilot-hot','instantaneous'), ('pilot-hot-to-cold','hot_to_relaxed'),
                       ('pilot-cold','relaxed_to_relaxed')]:
        runs.append(dict(name=alias, arm=arm))
        checkpoints[alias] = str(resolve_path(recipe['pilot_output']).resolve()/'technical/runs'/arm)
    config = dict(c, **{k:recipe[k] for k in ('output','cache','scratch','archive')},
                  frames=sorted(set(f for fs in roles.values() for f in fs)), frames_by_role=roles,
                  training_frames=[], paired_parent_plan=str(base_path), runs=runs)
    tasks = [dict(id=f'{s["id"]}-{f}', source=s['id'], frame=f, anchor=f, priority=1)
             for s in sources for f in roles[s.get('validation_role',s['split'])]]
    plan = dict(config=config, sources=sources, tasks=tasks, evaluation_recipe=recipe,
                checkpoints=checkpoints, parent_training_identity=base['identity'],
                potential_files=base['potential_files'], assay_identity=base['assay_identity'])
    plan['identity'] = digest(plan)
    with np.load(resolve_path(c['population'])) as a:
        frames = np.asarray(original['anchors'])[a['rows'][:,1]]
        keep = matched_assay_mask(a['source'],frames,requested_assay_frames(plan))
        pop = {k:a[k][keep] for k in ('source','rows','role')}; frames = frames[keep]
    onset = np.empty(len(frames),np.int64)
    for sid in np.unique(pop['source']):
        ix = np.flatnonzero(pop['source']==sid)
        onset[ix] = np.load(resolve_path(original['config']['cache'])/str(sid)/'onset.npy')[pop['rows'][ix,2]]
    coverage = population_audit(pop,frames,onset)
    for role in ('selection','calibration','test'):
        if coverage[role]['positive_windows_12ps'] != coverage[role]['distinct_local_onsets_12ps']:
            raise ValueError('Held-out grid repeats local events')
    save_json(root/'technical/planned-cohort.json',dict(coverage=coverage,frames_by_role=roles,
              independence='Whole simulation source; atom onsets within one source are correlated'))
    (root/'technical/runs').mkdir(exist_ok=True)
    for name, folder in checkpoints.items():
        (root/'technical/runs'/name).symlink_to(folder,target_is_directory=True)
    save_json(path,plan)
    save_json(root/'technical/accelerator-config.json',dict(plan=str(path),training_config='',
        benchmark=recipe['benchmark'],seed=c['seed'],force_max_error_tolerance=recipe['force_max_error_tolerance']))
    return plan


def reuse(plan):
    """Copy verified local float32 clouds, without repeating completed quenches."""
    parent = json.loads(resolve_path(plan['config']['paired_parent_plan']).read_text())
    old = resolve_path(parent['config']['cache']); root = resolve_path(plan['config']['output'])/'technical'
    count = 0
    for task in plan['tasks']:
        if not (old/'cells'/task['id']/'complete.json').exists():continue
        with claim(root/'locks'/f'cell-{task["id"]}') as acquired:
            if acquired:produce(plan,task,1);count += 1
    save_json(root/'reuse.json',dict(verified_parent_cells=count))


def build(plan):
    deadline=deadline_for_job();root=resolve_path(plan['config']['output'])/'technical'
    while time.time()<deadline-300:
        if fatal_failures(plan):raise RuntimeError('Non-timeout relaxation failure')
        if assay.prepare(plan):
            pop=dict(np.load(root/'assay/population.npz'))
            original=json.loads(resolve_path(plan['config']['assay_plan']).read_text())
            frames=np.asarray(original['anchors'])[pop['rows'][:,1]]
            onset=frames+np.rint(pop['delay']/.75).astype(np.int64)
            save_json(root/'actual-cohort.json',population_audit(pop,frames,onset))
            return
        time.sleep(20)
    raise TimeoutError('Evaluation builder allocation ended before release')


def cached_features(plan,name):
    c=plan['config'];root=resolve_path(c['output'])/'technical/assay'
    old=resolve_path(plan['evaluation_recipe']['cached_encoders'][name]);record=json.loads((old/'record.json').read_text())
    if record['protected_overlap'] or record['population_sha256']!=file_hash(resolve_path(c['population'])):
        raise ValueError(f'Historical encoder population/ancestry mismatch: {name}')
    if file_hash(Path(record['checkpoint']))!=record['checkpoint_sha256']:
        raise ValueError(f'Historical checkpoint changed: {name}')
    original=dict(np.load(resolve_path(c['population'])));pop=dict(np.load(root/'population.npz'))
    z=np.empty((len(pop['source']),128),np.float32);hashes={}
    for sid in np.unique(pop['source']):
        previous=np.flatnonzero(original['source']==sid);current=np.flatnonzero(pop['source']==sid)
        lookup={tuple(row):i for i,row in enumerate(original['rows'][previous])}
        indices=np.array([lookup[tuple(row)] for row in pop['rows'][current]])
        path=old/'features'/f'{sid}.npy';receipt=json.loads(path.with_suffix('.json').read_text())
        sha=file_hash(path)
        if receipt['sha256']!=sha or receipt['checkpoint_sha256']!=record['checkpoint_sha256']:
            raise ValueError(f'Historical feature checksum mismatch: {name}/{sid}')
        values=np.load(path)
        if values.shape!=(len(previous),128):raise ValueError(f'Historical feature shape: {name}/{sid}')
        z[current]=values[indices];hashes[str(sid)]=sha
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite historical embedding')
    dest=root/name;dest.mkdir(exist_ok=True);np.save(dest/'features.npy',z)
    save_json(dest/'complete.json',dict(checkpoint_sha256=record['checkpoint_sha256'],
        feature_sha256=file_hash(dest/'features.npy'),population_sha256=file_hash(root/'population.npz'),
        reused_record=str(old/'record.json'),source_feature_sha256=hashes,input='original unrelaxed snapshot'))


def worker(plan,lane):
    from .evaluation_metrics import report
    root=resolve_path(plan['config']['output'])/'technical';deadline=deadline_for_job()
    fresh=[r['name'] for r in selected_runs(plan)]+['parent_hot','parent_cold']
    cached=list(plan['evaluation_recipe']['cached_encoders'])
    names=[*cached,*BASELINES,*fresh]
    while time.time()<deadline-300:
        if fatal_failures(plan):raise RuntimeError('Non-timeout relaxation failure')
        if not (root/'assay/ready.json').exists():time.sleep(20);continue
        remaining=False
        for name in names:
            receipt=root/'evaluation'/f'{name}.json'
            if receipt.exists() and json.loads(receipt.read_text())['state']=='complete':continue
            remaining=True
            with claim(root/'locks'/f'evaluate-{name}') as acquired:
                if not acquired:continue
                if name in plan['checkpoints']:
                    status=Path(plan['checkpoints'][name])/'status.json'
                    if not status.exists():continue
                    state=json.loads(status.read_text())['state']
                    if state=='failed':raise RuntimeError(f'Encoder failed: {name}')
                    if state!='complete':continue
                save_json(receipt,dict(state='running',name=name,lane=lane))
                if name not in BASELINES and not (root/f'assay/{name}/complete.json').exists():
                    if name in cached:cached_features(plan,name)
                    else:assay.extract(plan,name)
                assay.probes(plan,name)
                save_json(receipt,dict(state='complete',name=name,lane=lane))
                report(plan,bootstrap=False)
        if not remaining:
            report(plan,bootstrap=True)
            save_json(root/'evaluation-complete.json',dict(state='complete',representations=len(names),readouts=2*len(names),excluded=evaluation_exclusions(plan)))
            return
        time.sleep(20)
    raise TimeoutError('Large evaluation checkpointed at allocation limit')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['freeze','reuse','build','worker','report'])
    p.add_argument('--config',required=True);p.add_argument('--lane',default='0');a=p.parse_args()
    recipe=json.loads(resolve_path(a.config).read_text());plan=freeze(recipe);torch.set_num_threads(1)
    if a.stage=='reuse':reuse(plan)
    elif a.stage=='build':build(plan)
    elif a.stage=='worker':worker(plan,a.lane)
    elif a.stage=='report':
        from .evaluation_metrics import report
        report(plan,bootstrap=True)


if __name__=='__main__':
    code=0
    try:main()
    except BaseException:traceback.print_exc();code=1
    sys.stdout.flush();sys.stderr.flush()
    import os
    os._exit(code)
