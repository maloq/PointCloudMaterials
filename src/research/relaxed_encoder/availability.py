"""Terminal timeout exclusions shared by producers, caches and matched assays."""
import json
import fcntl
from pathlib import Path
import numpy as np
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json, file_hash


def is_timeout_record(record):
    # CPU/GPU producers persist repr(subprocess.TimeoutExpired), not free-form
    # messages. Do not turn arbitrary numerical/provenance failures into skips.
    return record['error'].startswith('TimeoutExpired(')


def skipped_cells(plan):
    root=resolve_path(plan['config']['output'])/'technical'
    cache=resolve_path(plan['config']['cache'])
    root.mkdir(parents=True,exist_ok=True)
    with (root/'skip-policy.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        for failure in (root/'failures').glob('*.json'):
            record=json.loads(failure.read_text());task=record['task'];sid=task['id']
            if not is_timeout_record(record):continue
            path=root/'skipped'/f'{sid}.json'
            # A completed recovery before the skip policy was deployed stays valid.
            if (cache/'cells'/sid/'complete.json').exists() and not path.exists():continue
            if not path.exists():
                save_json(path,dict(state='skipped',reason='relaxation_walltime_timeout',task=task,
                          failure=str(failure.resolve()),failure_sha256=file_hash(failure),error=record['error']))
    return {p.stem:json.loads(p.read_text()) for p in (root/'skipped').glob('*.json')}


def fatal_failures(plan):
    skipped_cells(plan)
    root=resolve_path(plan['config']['output'])/'technical'
    return [p for p in (root/'failures').glob('*.json') if not is_timeout_record(json.loads(p.read_text()))]


def training_availability(plan):
    c=plan['config'];cache=resolve_path(c['cache']);skips=skipped_cells(plan)
    included=[];excluded=[];pending=[]
    for source in plan['sources']:
        if not source['pilot_fit']:continue
        for frame in c.get('training_frames',c['frames']):
            ids=[f'{source["id"]}-{f}' for f in (frame,frame+1)]
            bad=[sid for sid in ids if sid in skips]
            if bad:
                excluded.append(dict(source=source['id'],frame=frame,skipped_cells=bad,
                    split=source.get('validation_role',source['split']),
                    anchors=len(source['pool_atom_ids'] if source.get('validation_role',source['split'])=='train' else source['center_atom_ids'])))
            elif all((cache/'cells'/sid/'complete.json').exists() for sid in ids):included.append((source,frame))
            else:pending.extend(sid for sid in ids if not (cache/'cells'/sid/'complete.json').exists())
    return included,excluded,pending


def requested_assay_frames(plan):
    c=plan['config']
    return {s['id']:c['frames_by_role'][s.get('validation_role',s['split'])]
            if 'frames_by_role' in c else c['frames'] for s in plan['sources']}


def assay_availability(plan):
    c=plan['config'];cache=resolve_path(c['cache']);skips=skipped_cells(plan);available={};pending=[]
    requested=requested_assay_frames(plan)
    for source in plan['sources']:
        sid=source['id'];available[sid]=[]
        for frame in requested[sid]:
            key=f'{sid}-{frame}'
            if key in skips:continue
            if (cache/'cells'/key/'complete.json').exists():available[sid].append(frame)
            else:pending.append(key)
    return available,pending,skips


def matched_assay_mask(sources,frames,available):
    return np.array([int(frame) in available[int(source)] for source,frame in zip(sources,frames,strict=True)],dtype=bool)
