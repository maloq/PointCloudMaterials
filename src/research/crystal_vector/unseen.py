"""Condition the fixed population on no interface atom in any encoder patch."""
import json
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import sha,write_json
from src.project_runtime.paths import resolve_path
from src.research.spatial_approach.evaluate import csv_rows


def apply_view(data):
    """Condition existing population probabilities; never delete exported rows."""
    data.eligibility=~data.meta['visible_context']
    data.view_audit={}
    for role,ids in data.split.items():
        selected=data.eligibility[ids];w=data.weights[role];mass=float(w[selected].sum())
        if mass<=0:raise ValueError(f'No non-visible interface population for {role}')
        data.split[role]=ids[selected];data.weights[role]=w[selected]/mass
        data.view_audit[role]=dict(original_rows=len(ids),retained_rows=int(selected.sum()),
            retained_original_mass=mass,retained_sources=int(len(np.unique(data.meta['source'][ids[selected]]))),
            finite_targets=int(np.isfinite(data.meta['distance'][ids[selected]]).sum()),
            within20A=int((data.meta['distance'][ids[selected]]<=20).sum()),
            inside_crystal=int(data.meta['inside_crystal'][ids[selected]].sum()))


def alarm_tables(values,meta,plan):
    from .evaluate import RADII
    from src.research.spatial_distance.confidence import first_alarm
    rows=[]
    for k,radius in enumerate(RADII):
        for threshold in (.5,.75,.95):
            warnings=[];away=[];eligible_lengths=[]
            for record in plan['paths']:
                if record['role']!='test':continue
                ids=np.flatnonzero(meta['path']==record['index'])
                ids=ids[np.argsort(meta['travel'][ids],kind='stable')]
                if len(ids)!=record['rows']:raise ValueError('Original scan rows must be retained')
                stop=np.flatnonzero(meta['visible_context'][ids] | meta['inside_crystal'][ids])
                prefix=ids[:int(stop[0])] if len(stop) else ids
                alarm=first_alarm(values['cdf'][prefix,k],threshold,2) if len(prefix)>=2 else None
                if record['kind']=='toward':
                    eligible_lengths.append(len(prefix))
                    warnings.append(np.nan if alarm is None else float(meta['distance'][prefix[alarm]]))
                else:away.append(alarm is not None)
            a=np.asarray(warnings);detected=np.isfinite(a)
            rows.append(dict(radius_A=radius,threshold=threshold,toward_paths=len(a),
                eligible_paths=int((np.asarray(eligible_lengths)>=2).sum()),detections=int(detected.sum()),
                misses=int((~detected).sum()),conditional_median_warning_A=float(np.median(a[detected])) if detected.any() else None,
                recall_at12A=float((a>=12).mean()),recall_at20A=float((a>=20).mean()),
                away_paths=len(away),away_alarms=int(np.sum(away)),away_alarm_rate=float(np.mean(away)),
                interface_visible_at_alarm=0,rule='two consecutive observations before first visible interface or crystal entry'))
    return rows


def paired_reference(values,meta,analysis,reference):
    """Pair frozen historical predictions with the unchanged original clear rows."""
    from .evaluate import point_tables
    ref=resolve_path(reference)/'analyses/localization-v1/technical'
    receipt=json.loads((ref/'predictions.json').read_text())
    if sha(ref/'predictions.npz')!=receipt['sha256']:
        raise ValueError('Changed reference predictions')
    if sha(resolve_path(reference)/'technical/best.pt')!=receipt['checkpoint_sha256']:
        raise ValueError('Changed reference checkpoint')
    with np.load(ref/'rows.npz') as a:old={k:a[k] for k in a.files}
    current=np.flatnonzero((meta['interface_parent_index']>=0)&~meta['visible_context'])
    previous=np.empty(len(current),dtype=np.int64)
    for sid in np.unique(meta['source'][current]):
        take=np.flatnonzero(meta['source'][current]==sid)
        old_ids=np.flatnonzero(old['source']==sid)
        previous[take]=old_ids[meta['interface_parent_index'][current[take]]]
    expected=np.flatnonzero(~old['visible_context'])
    if not np.array_equal(np.sort(previous),expected):
        raise ValueError('Original invisible comparison population changed')
    for field in ('source','atom','frame','kind','path','distance','visible_context','role'):
        if not np.array_equal(meta[field][current],old[field][previous]):
            raise ValueError(f'Paired original row field changed: {field}')
    with np.load(ref/'predictions.npz') as a:prior={k:a[k][previous] for k in a.files}
    rowmeta={k:v[current] for k,v in meta.items()}
    outputs=[[],[],[]]
    for model,pred in [('previous_vcreg',prior),('invisible_adaptation',{k:v[current] for k,v in values.items()})]:
        for collected,table in zip(outputs,point_tables(pred,rowmeta,True)):
            collected.extend(dict(model=model,**row) for row in table)
    for name,table in zip(('distance','direction','reliability'),outputs):
        csv_rows(analysis/'tables'/f'paired-original-unseen-{name}.csv',table)
    write_json(analysis/'technical/paired-reference.json',dict(reference=str(ref),
        predictions_sha256=receipt['sha256'],checkpoint_sha256=receipt['checkpoint_sha256'],
        rows=len(current),rule='all original invisible rows, exact source-local parent indices and labels verified'))


def export(values,meta,plan,analysis,reference):
    from .evaluate import point_tables
    mask=~meta['visible_context']
    tables=point_tables({k:v[mask] for k,v in values.items()},{k:v[mask] for k,v in meta.items()},True)
    for name,table in zip(('distance','direction','reliability'),tables):
        if table:csv_rows(analysis/'tables'/f'unseen-{name}.csv',table)
    csv_rows(analysis/'tables/unseen-alarms.csv',alarm_tables(values,meta,plan))
    paired_reference(values,meta,analysis,reference)
    write_json(analysis/'technical/unseen-scope.json',dict(exclusion='any interface atom in any of 25 radius-8 encoder patches',
        retained_rows=int(mask.sum()),total_rows=len(mask),original_rows_retained=True,
        scan_rule='prefix before first visible query, preserving consecutive original indices'))
    return {f"evaluation/unseen/{r['population']}/{r['role']}/{k}":v
            for r in tables[0] for k,v in r.items() if k not in ('population','role')}
