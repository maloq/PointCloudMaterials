"""Distance to an external crystal using only uncrystallized observed patches."""
import json
import numpy as np

from src.data.fixed_cohort.protocol import sha,write_json
from src.project_runtime.paths import resolve_path
from src.research.local_predictability.metrics import source_weights
from src.research.spatial_approach.evaluate import csv_rows


FILTER='liquid_no_visible_crystal'


def observation_mask(meta):
    # Established, past-confirmed crystals only. Subcritical/partially ordered
    # liquid structures are retained; no instantaneous PTM-order veto is used.
    return ~meta['inside_crystal'] & ~meta['crystal_visible_context']


def localization_mask(meta):
    return observation_mask(meta) & np.isfinite(meta['crystal_distance'])


def apply_view(data):
    meta=data.meta;data.eligibility=localization_mask(meta)
    ids=np.flatnonzero(data.eligibility)
    if meta['visible_context'][ids].any() or not meta['interface_exists'][ids].all():
        raise ValueError('Crystal-free observation has inconsistent interface metadata')
    if not np.array_equal(meta['distance'][ids],meta['crystal_distance'][ids]):
        raise ValueError('Cached interface distance differs from nearest-crystal distance in eligible liquid; relabel before fitting')
    data.view_audit={}
    for role,all_ids in data.split.items():
        keep=data.eligibility[all_ids];w=data.weights[role];mass=float(w[keep].sum())
        if mass<=0:raise ValueError(f'No crystal-free liquid localization examples for {role}')
        chosen=all_ids[keep];data.split[role]=chosen;data.weights[role]=w[keep]/mass
        data.view_audit[role]=dict(original_rows=len(all_ids),retained_rows=len(chosen),retained_original_mass=mass,
            retained_sources=int(len(np.unique(meta['source'][chosen]))),within20A=int((meta['distance'][chosen]<=20).sum()),
            within32A=int((meta['distance'][chosen]<=32).sum()),censored_at64A=int((meta['distance'][chosen]>=64).sum()),
            inside_crystal=int(meta['inside_crystal'][chosen].sum()),crystal_visible=int(meta['crystal_visible_context'][chosen].sum()),
            no_crystal_elsewhere=int((~np.isfinite(meta['crystal_distance'][chosen])).sum()),
            separate_absence_rows=int((observation_mask(meta)[all_ids]&~np.isfinite(meta['crystal_distance'][all_ids])).sum()))


def paired_reference(values,meta,analysis,reference):
    from .evaluate import point_tables
    root=resolve_path(reference);tech=root/'analyses/localization-v1/technical'
    receipt=json.loads((tech/'predictions.json').read_text())
    if sha(tech/'predictions.npz')!=receipt['sha256'] or sha(root/'technical/best.pt')!=receipt['checkpoint_sha256']:
        raise ValueError('Changed historical comparison predictions/checkpoint')
    with np.load(tech/'rows.npz') as a:old={k:a[k] for k in a.files}
    current=np.flatnonzero((meta['interface_parent_index']>=0)&localization_mask(meta));previous=np.empty(len(current),np.int64)
    for sid in np.unique(meta['source'][current]):
        take=np.flatnonzero(meta['source'][current]==sid)
        previous[take]=np.flatnonzero(old['source']==sid)[meta['interface_parent_index'][current[take]]]
    if not np.array_equal(np.sort(previous),np.flatnonzero(localization_mask(old))):
        raise ValueError('Original liquid-only comparison rows changed')
    for field in ('source','atom','frame','kind','path','role','distance','crystal_distance','crystal_visible_context','inside_crystal'):
        if not np.array_equal(meta[field][current],old[field][previous]):raise ValueError(f'Paired field changed: {field}')
    with np.load(tech/'predictions.npz') as a:prior={k:a[k][previous] for k in a.files}
    tables=[[],[],[]];rows={k:v[current] for k,v in meta.items()}
    for label,pred in [('previous_interface_vcreg',prior),('liquid_distance_vcreg',{k:v[current] for k,v in values.items()})]:
        for result,part in zip(tables,point_tables(pred,rows,True)):result.extend(dict(model=label,**r) for r in part)
    for name,table in zip(('distance','direction','reliability'),tables):csv_rows(analysis/'tables'/f'paired-original-liquid-{name}.csv',table)
    write_json(analysis/'technical/paired-reference.json',dict(reference=str(root),rows=len(current),
        checkpoint_sha256=receipt['checkpoint_sha256'],predictions_sha256=receipt['sha256']))


def export(values,data,plan,analysis,c):
    from .evaluate import RADII,point_tables
    from .unseen import alarm_tables
    meta=data.meta;eligible=data.eligibility;clear=observation_mask(meta)
    absence=clear&~np.isfinite(meta['crystal_distance'])
    rows,_,reliability=point_tables({k:v[absence] for k,v in values.items()},{k:v[absence] for k,v in meta.items()},False)
    csv_rows(analysis/'tables/absence-distance.csv',rows);csv_rows(analysis/'tables/absence-reliability.csv',reliability)
    scan_meta=dict(meta,visible_context=meta['crystal_visible_context'])
    alarms=alarm_tables(values,scan_meta,plan)
    for row in alarms:
        row['crystal_visible_at_alarm']=row.pop('interface_visible_at_alarm')
        row['rule']='two consecutive original queries before first established crystal in any input patch or crystal entry'
    csv_rows(analysis/'tables/liquid-alarms.csv',alarms)
    paired_reference(values,meta,analysis,c['comparison_reference'])
    train=data.split['train'];weight=data.weights['train'];d=meta['crystal_distance'][train]
    mean=float(weight@np.minimum(d,64));prob=np.asarray([weight@(d<=r) for r in RADII]);baselines=[]
    for track,original in [('original',(meta['interface_parent_index']>=0)),('expanded_and_original',np.ones(len(eligible),bool))]:
        for role in ('selection','calibration','test'):
            for population,kind in [('fixed_at_risk',0),('uniform',1)]:
                ids=np.flatnonzero(eligible&original&(meta['role']==role)&(meta['kind']==kind))
                if not len(ids):continue
                w=source_weights(meta['source'][ids]);target=np.minimum(meta['crystal_distance'][ids],64)
                for label,estimate,probability in [('trained',values['mean_A'][ids],values['cdf'][ids]),('constant_training_population',np.full(len(ids),mean),np.broadcast_to(prob,(len(ids),len(RADII))))]:
                    record=dict(track=track,role=role,population=population,model=label,rows=len(ids),sources=len(np.unique(meta['source'][ids])),
                        capped_mean_rmse_A=float(np.sqrt(w@(estimate-target)**2)))
                    for j,radius in enumerate(RADII):record[f'brier_within{radius}A']=float(w@(probability[:,j]-(meta['crystal_distance'][ids]<=radius))**2)
                    baselines.append(record)
    csv_rows(analysis/'tables/liquid-baselines.csv',baselines)
    write_json(analysis/'technical/liquid-scope.json',dict(selection=data.view_audit,observation='no established crystal in any actual patch; query outside crystal',
        task='distance to existing nearest external established crystal; equality to cached interface target verified',
        train_constant_distance_A=mean,train_constant_cdf=prob.tolist(),all_original_rows_preserved=True,
        absence='separate challenge only; not fitting or checkpoint selection; no claim of calibrated crystal-presence prediction'))
    return {f"evaluation/absence/{r['population']}/{r['role']}/{k}":v for r in rows for k,v in r.items() if k not in ('population','role')}
