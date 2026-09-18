"""Source-balanced audit of center-vector time evolution and spatial order."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

from src.data.structural_pretraining.prepare import file_hash, save_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.trajectory_stability.metrics import stratified_draws
from .model import STAGES, SECTORS
from .metrics import angular_metrics, direction_pair, unit_vectors, spatial_metrics, interval

BASELINES = ('centroid_7A', 'centroid_support', 'shape_axis_7A')
PRIMARY = 'block2_mlp_input/point_numerator'
SCALARS = ('coverage', 'mean_angle_deg', 'median_angle_deg', 'p95_angle_deg',
    'mean_axis_angle_deg', 'jump60_fraction', 'flip90_fraction', 'p1', 'p2')


def load_temporal(root, parent):
    items = []
    for source in parent['sources']:
        folder = root/'technical/temporal'/str(source['id'])
        receipt = json.loads((folder/'complete.json').read_text())
        if file_hash(folder/'features.npz') != receipt['sha256']:
            raise ValueError(f'Changed temporal features: {folder}')
        items.append((source, dict(np.load(folder/'features.npz'))))
    return items


def reference(items, config):
    train = np.concatenate([a['vectors'] for s, a in items if s['split'] == 'train']).astype(float)
    rms = np.sqrt(np.mean(np.sum(train*train, axis=-1), axis=0))
    channels = np.argmax(rms, axis=1)
    fields, rows = {}, []
    for si, stage in enumerate(STAGES):
        for ki, sector in enumerate(SECTORS):
            channel = int(channels[si, ki])
            name = f'{stage}/{sector}'
            for c in range(8):
                rows.append(dict(field=name, channel=c, rms_norm=rms[si,c,ki],
                    selected=c == channel, active=rms[si,c,ki] > 0,
                    threshold=config['direction_threshold_fraction']*rms[si,c,ki]))
            if rms[si,channel,ki] > 0:
                fields[name] = dict(stage=si, sector=ki, channel=channel,
                    rms=rms[si,channel,ki], threshold=config['direction_threshold_fraction']*rms[si,channel,ki])
    trainbase = np.concatenate([a['baseline'] for s, a in items if s['split'] == 'train']).astype(float)
    for bi, name in enumerate(BASELINES):
        scale = np.sqrt(np.mean(np.sum(trainbase[:,bi]**2, axis=-1)))
        fields[name] = dict(baseline=bi, rms=scale,
            threshold=config['direction_threshold_fraction']*scale)
    return fields, rows, rms


def get_vectors(a, spec):
    if 'baseline' in spec:
        v = a['baseline'][:,spec['baseline']].astype(float).copy()
        if spec['baseline'] == 2:
            v[a['shape_gap'] < .01] = 0  # axis undefined at a nearly degenerate top eigenvalue
        return v
    return a['vectors'][:,spec['stage'],spec['channel'],spec['sector']].astype(float)


def temporal_tables(items, fields, rms, config, cadence):
    sources, lag_rows, phases, diagnostics, channels, sensitivity = [], [], [], [], [], []
    for source, a in items:
        if source['split'] != 'test':
            continue
        meta = dict(source=source['id'], temperature_K=source['temperature_K'])
        nf, nc = len(a['frames']), len(a['centers'])
        labels = a['labels'].reshape(nf,nc)
        z = a['z'].reshape(nf,nc,-1).astype(float)
        dz = np.linalg.norm(np.diff(z, axis=0), axis=-1)
        for name, spec in fields.items():
            v = get_vectors(a, spec).reshape(nf,nc,3)
            threshold = spec['threshold']
            main = angular_metrics(v[:-1], v[1:], threshold)
            rotated = np.einsum('tci,tcij->tcj',v[:-1],a['cage_rotation'])
            corrected = angular_metrics(rotated,v[1:],threshold)
            sources.append(dict(meta,field=name,**main,
                cage_mean_angle_deg=corrected['mean_angle_deg'], cage_p1=corrected['p1'],
                cage_fit_rms_A=float(a['cage_fit_rms_A'].mean())))
            for lag in config['lag_frames']:
                lag_rows.append(dict(meta,field=name,lag_ps=cadence*lag,
                    **angular_metrics(v[:-lag],v[lag:],threshold)))
            _, norms, valid = unit_vectors(v,threshold)
            for label, code in [('unclassified',0),('FCC',1),('HCP',2),('BCC',3)]:
                mask = labels == code
                pair = mask[:-1] & mask[1:]
                if not mask.any():
                    continue
                phases.append(dict(meta,field=name,phase=label,observations=int(mask.sum()),
                    rms_relative_norm=float(np.sqrt(np.mean(norms[mask]**2))/spec['rms']),
                    **(angular_metrics(v[:-1][pair],v[1:][pair],threshold) if pair.any()
                        else dict.fromkeys(('pairs','valid_pairs',*SCALARS)))))
            _, angle, pairvalid = direction_pair(v[:-1],v[1:],threshold)
            dv = np.linalg.norm(np.diff(v,axis=0),axis=-1)
            detail = dict(meta,field=name,
                norm_rms=float(np.sqrt(np.mean(norms**2))), valid_fraction=float(valid.mean()),
                z_increment_spearman=float(spearmanr(dv.ravel(),dz.ravel()).statistic),
                cage_residual_angle_spearman=float(spearmanr(a['cage_fit_rms_A'][pairvalid],angle[pairvalid]).statistic))
            if 'baseline' not in spec:
                allv = a['vectors'][:,spec['stage'],:,spec['sector']].astype(float)
                u, _, ok = unit_vectors(allv, .1*rms[spec['stage'],:,spec['sector']])
                gram = np.einsum('nci,ncj->nij',u,u)
                eig = np.linalg.eigvalsh(gram)
                rowok = ok.sum(1)>0
                detail['channel_axis_rank1_fraction'] = float((eig[rowok,-1]/ok[rowok].sum(1)).mean())
                for bi, basename in enumerate(BASELINES):
                    b = get_vectors(a,fields[basename]).reshape(nf,nc,3)
                    vu, _, vok = unit_vectors(v,threshold)
                    bu, _, bok = unit_vectors(b,fields[basename]['threshold'])
                    take = vok&bok
                    cosine = np.sum(vu*bu,axis=-1)[take]
                    detail[f'align_{basename}_p2'] = float(np.mean((3*cosine*cosine-1)/2)) if len(cosine) else None
            diagnostics.append(detail)
            if name == PRIMARY:
                for fraction in (0.,.1,.3,.5,1.):
                    sensitivity.append(dict(meta,threshold_fraction=fraction,
                        **angular_metrics(v[:-1],v[1:],fraction*spec['rms'])))
        for si, stage in enumerate(STAGES):
            for ki, sector in enumerate(SECTORS):
                for c in range(8):
                    v = a['vectors'][:,si,c,ki].reshape(nf,nc,3)
                    channels.append(dict(meta,field=f'{stage}/{sector}',channel=c,
                        **angular_metrics(v[:-1],v[1:],config['direction_threshold_fraction']*rms[si,c,ki])))
    return dict(temporal_sources=sources,temporal_lags=lag_rows,phases=phases,
        diagnostics=diagnostics,temporal_channels=channels,threshold_sensitivity=sensitivity)


def spatial_tables(root, parent, fields, config):
    rows, frames, phase_rows = [], [], []
    for source in parent['sources']:
        if source['split'] != 'test':
            continue
        for frame in config['spatial_frames']:
            path = root/'technical/spatial'/f'{source["id"]}-{frame:04d}.npz'
            receipt = json.loads(path.with_suffix('.json').read_text())
            if file_hash(path) != receipt['sha256']:
                raise ValueError(f'Changed spatial panel: {path}')
            a = dict(np.load(path))
            n = len(a['atom_ids'])
            # Intersection fraction of the two native nearest-80 sets, including each center.
            ids = a['nearest80_ids']
            incidence = csr_matrix((np.ones(ids.size), (np.repeat(np.arange(n),80),ids.ravel())),
                shape=(n,int(ids.max())+1))
            overlap = (incidence@incidence.T).toarray()/80.
            frames.append(dict(source=source['id'],frame=frame,centers=n))
            for name, spec in fields.items():
                values = spatial_metrics(get_vectors(a,spec),spec['threshold'],a['labels'],
                    a['positions'],a['box'],config['distance_edges_A'],overlap)
                rows.extend([dict(source=source['id'],temperature_K=source['temperature_K'],
                    frame=frame,field=name,**v) for v in values])
                if name in (PRIMARY,'block1/point_numerator'):
                    for phase,code in [('unclassified',0),('FCC',1),('HCP',2)]:
                        values = spatial_metrics(get_vectors(a,spec),spec['threshold'],a['labels'],
                            a['positions'],a['box'],config['distance_edges_A'],overlap,phase_code=code)
                        phase_rows.extend([dict(source=source['id'],temperature_K=source['temperature_K'],
                            frame=frame,field=name,phase=phase,**v) for v in values])
    return rows, frames, phase_rows


def summarize(frame, keys, metrics, test_sources, draws):
    rows = []
    grouper = keys[0] if len(keys)==1 else keys
    for group, part in frame.groupby(grouper,sort=False):
        meta = dict(zip(keys,group if isinstance(group,tuple) else (group,),strict=True))
        per_source = part.groupby('source')[metrics].mean().reindex(test_sources)
        for metric in metrics:
            values = per_source[metric]
            if values.isna().any():
                rows.append(dict(meta,metric=metric,sources=int(values.notna().sum()),mean=None,low=None,high=None))
            else:
                rows.append(dict(meta,metric=metric,sources=len(values),**interval(values.to_numpy(),draws)))
    return rows


def report(config, parent):
    root = Path(config['output'])
    (root/'tables').mkdir(exist_ok=True); (root/'plots').mkdir(exist_ok=True)
    items = load_temporal(root,parent)
    fields, reference_rows, rms = reference(items,config)
    save_json(root/'technical/selected-directions.json',fields)
    tables = temporal_tables(items,fields,rms,config,parent['cadence_ps'])
    tables['reference_channels'] = reference_rows
    spatial, spatial_frames, spatial_phase = spatial_tables(root,parent,fields,config)
    tables['spatial_frames'] = spatial; tables['sampling'] = spatial_frames
    tables['spatial_phases'] = spatial_phase
    test = [s for s in parent['sources'] if s['split']=='test']
    draws = stratified_draws([s['temperature_K'] for s in test],config['bootstrap_draws'],config['seed'])
    ids = [s['id'] for s in test]
    tables['temporal_summary'] = summarize(pd.DataFrame(tables['temporal_sources']),['field'],
        [*SCALARS,'cage_mean_angle_deg','cage_p1'],ids,draws)
    tables['lag_summary'] = summarize(pd.DataFrame(tables['temporal_lags']),['field','lag_ps'],
        ['p1','p2','coverage'],ids,draws)
    tables['spatial_summary'] = summarize(pd.DataFrame(spatial),['field','distance_lo_A','distance_hi_A'],
        ['p1','p2','shuffle_p1','shuffle_p2','excess_p1','excess_p2','overlap80'],ids,draws)
    tables['sensitivity_summary'] = summarize(pd.DataFrame(tables['threshold_sensitivity']),['threshold_fraction'],
        ['coverage','mean_angle_deg','flip90_fraction','p1','p2'],ids,draws)
    tables['phase_summary'] = summarize(pd.DataFrame(tables['phases']),['field','phase'],
        ['coverage','mean_angle_deg','p1','p2','rms_relative_norm'],ids,draws)
    tables['spatial_phase_summary'] = summarize(pd.DataFrame(spatial_phase),['field','phase','distance_lo_A','distance_hi_A'],
        ['p1','p2','shuffle_p1','shuffle_p2','excess_p1','excess_p2'],ids,draws)
    snapshot_metric_docs(root,'gatr_equivariant')
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(root/'tables'/f'{name}.csv',index=False)
    from .plots import make_plots, make_explorer, findings
    make_plots(root,tables,items,fields)
    make_explorer(root,items,fields,config)
    findings(root,tables,fields,config,parent)
    print(f'Completed report: {root.resolve()}',flush=True)
