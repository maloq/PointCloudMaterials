"""Replay saved onset probabilities at 3/6/12 ps without refitting or selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.local_predictability.metrics import source_weights, weighted_scores
from src.research.robust_onset.metrics import horizon_index


HORIZONS = (3, 6, 12)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replay(prediction, records, targets, *, native=False, prefix=''):
    """Use the producer's exact row IDs and event bins; never align by position."""
    ix = prediction[prefix+'indices']
    source = np.array([records[i]['root'] for i in ix])
    expected_source = np.array([records[i]['source' if native else 'root'] for i in ix])
    if ix.ndim != 1 or ix.dtype.kind not in 'iu' or len(np.unique(ix)) != len(ix):
        raise ValueError('Prediction indices must be unique integer corpus row IDs')
    expected_ix = np.array([i for i,r in enumerate(records)
                           if r['split']=='development' and targets['at_risk'][i]])
    np.testing.assert_array_equal(ix, expected_ix)
    np.testing.assert_array_equal(prediction[prefix+'source'], expected_source)
    event = prediction[prefix+('event' if native else 'event_bin')]
    np.testing.assert_array_equal(event, targets['event_bin'][ix])
    risk = prediction[prefix+('risks' if native else 'risk')]
    if risk.shape != (len(ix),5) or not np.isfinite(risk).all() or np.any((risk<0)|(risk>1)):
        raise ValueError('Require finite cumulative probabilities for all five onset bins')
    if np.any(np.diff(risk,axis=1)<-1e-7) or np.any((event<0)|(event>5)):
        raise ValueError('Nonmonotone cumulative risk or invalid event/survival bin')
    scores = {h:weighted_scores(event<=horizon_index(h),risk[:,horizon_index(h)],source) for h in HORIZONS}
    return scores, dict(indices=ix,event=event,risk=risk,source=source,
                       temperature=np.array([records[i]['temperature_K'] for i in ix]))


def verify_saved(scores, saved):
    for h in HORIZONS:
        for metric in ('average_precision','brier','prevalence'):
            np.testing.assert_allclose(scores[h][metric],saved[h][metric],rtol=1e-10,atol=1e-12,
                                       err_msg=f'{h} ps {metric} differs from saved evaluation')


def bootstrap_pair(candidate, reference, *, draws, seed, horizon_ps):
    for key in ('indices','event','source','temperature'):
        np.testing.assert_array_equal(candidate[key],reference[key])
    k=horizon_index(horizon_ps);y=reference['event']<=k
    source=reference['source'];roots=np.unique(source);weight=source_weights(source)
    temp=np.array([reference['temperature'][source==s][0] for s in roots])
    rng=np.random.default_rng(seed);delta=[];absolute=[]
    for _ in range(draws):
        sampled=np.concatenate([rng.choice(roots[temp==t],int((temp==t).sum()),replace=True)
                                for t in np.unique(temp)])
        w=weight*np.array([(sampled==s).sum() for s in source])
        if not w[y].sum():
            continue
        ap=average_precision_score(y,candidate['risk'][:,k],sample_weight=w)
        absolute.append(ap)
        delta.append(ap-average_precision_score(y,reference['risk'][:,k],sample_weight=w))
    if not delta:
        raise ValueError(f'No positive-event bootstrap replicates at {horizon_ps} ps')
    return dict(horizon_ps=horizon_ps,AP_low=float(np.quantile(absolute,.025)),
                AP_high=float(np.quantile(absolute,.975)),delta_AP_low=float(np.quantile(delta,.025)),
                delta_AP_high=float(np.quantile(delta,.975)),valid_draws=len(delta),
                no_positive_draws=draws-len(delta))


def table(path, rows):
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run(config_path):
    c=json.loads(Path(config_path).read_text())
    if c['primary_horizon_ps']!=3 or c['horizons_ps']!=list(HORIZONS):
        raise ValueError('This review declares 3 ps primary and 6/12 ps secondary')
    output=resolve_path(c['output'])
    if output.exists():
        raise FileExistsError(f'Review output already exists: {output}')
    for directory in ('tables','technical','plots'):(output/directory).mkdir(parents=True,exist_ok=True)
    cache=resolve_path(c['cache']);records=json.loads((cache/'records.json').read_text())
    with np.load(cache/'targets.npz') as a:targets=dict(a)
    provenance={str(p):sha(p) for p in [Path(config_path),cache/'manifest.json',cache/'records.json',cache/'targets.npz']}
    rows=[];packets={};coverage=[]

    def add(collection, model, readout, metrics, prediction_path, *, native=False, prefix='', selection):
        provenance[str(prediction_path)]=sha(prediction_path)
        with np.load(prediction_path) as a:scores,packet=replay(dict(a),records,targets,native=native,prefix=prefix)
        verify_saved(scores,metrics)
        row=dict(collection=collection,model=model,readout=readout,primary_horizon_ps=3,
                 selection=selection,verification='replayed_saved_predictions')
        for h in HORIZONS:
            y=packet['event']<=horizon_index(h)
            row.update({f'AP{h}':scores[h]['average_precision'],f'Brier{h}':scores[h]['brier'],
                        f'prevalence{h}':scores[h]['prevalence'],f'positive_windows{h}':int(y.sum()),
                        f'positive_roots{h}':int(len(np.unique(packet['source'][y])))})
        row['prediction_path']=str(prediction_path);rows.append(row)
        packets[(collection,model,readout)]=packet

    for collection in c['collections']:
        name=collection['name'];root=resolve_path(collection['root'])
        files=sorted((root/'technical/evaluations').glob('*/metrics.json'))
        if len(files)!=collection['expected_models']:
            raise ValueError(f'{name}: expected {collection["expected_models"]} metric files, found {len(files)}')
        for path in files:
            m=json.loads(path.read_text());provenance[str(path)]=sha(path)
            complete=path.with_name('provenance.json' if collection['format']=='geoformer' else 'complete.json')
            if not complete.exists():raise ValueError(f'Missing evaluation completion receipt: {path}')
            provenance[str(complete)]=sha(complete)
            if collection['format']=='joint':
                receipt=json.loads(complete.read_text())
                if receipt['metrics_sha256']!=sha(path):raise ValueError(f'Changed metrics: {path}')
                add(name,m['arm'],'joint',{h:m['onset'][str(h)] for h in HORIZONS},path.with_name('predictions.npz'),
                    selection='encoder and joint head selected by tuning AP12 with retention gate')
                for readout in ('linear','mlp'):
                    add(name,m['arm'],readout,{h:m['probes'][readout]['horizons'][str(float(h))] for h in HORIZONS},
                        path.with_name(f'probe-{readout}.npz'),native=True,
                        selection='encoder selected by AP12; frozen probe selected by tuning NLL')
            elif collection['format'] in ('native','geoformer'):
                for readout in ('encoder','projector'):
                    key='prediction_'+readout
                    if key not in m:continue
                    add(name,path.parent.name,readout,{h:m[key]['conditional_hazard']['horizons'][str(float(h))] for h in HORIZONS},
                        path.with_name(f'{readout}-future-predictions.npz'),native=True,prefix='hazard_',
                        selection=collection['selection'])
            else:raise ValueError(f'Unknown declared producer: {collection["format"]}')
        coverage.append(dict(collection=name,models=len(files),missing_horizons=0))
    # Descriptor controls already contain every requested horizon, but their
    # original collector did not retain individual prediction arrays.
    path=resolve_path(c['descriptor_metrics']);m=json.loads(path.read_text());provenance[str(path)]=sha(path)
    controls=[]
    for name,metric in m['scores'].items():
        row=dict(model=name,selection='tuning NLL',verification='existing_metrics; no saved prediction array')
        for h in HORIZONS:
            row.update({f'AP{h}':metric['horizons'][str(float(h))]['average_precision'],
                        f'Brier{h}':metric['horizons'][str(float(h))]['brier']})
        controls.append(row)
    # This exact baseline is reused across all historical native hazard probes.
    p=resolve_path(c['current_physics_predictions'])
    with np.load(p) as a:scores,packet=replay(dict(a),records,targets,native=True,prefix='hazard_')
    provenance[str(p)]=sha(p)
    controls.append(dict(model='current-physics native hazard',selection='tuning NLL',verification='replayed_saved_predictions',
                         **{key:score for h in HORIZONS for key,score in [(f'AP{h}',scores[h]['average_precision']),(f'Brier{h}',scores[h]['brier'])]}))

    intervals=[]
    for comparison in c['paired_comparisons']:
        name=comparison['collection'];readout=comparison['readout'];ref=comparison['reference']
        for model in comparison['models']:
            result=bootstrap_pair(packets[(name,model,readout)],packets[(name,ref,readout)],
                                  draws=c['bootstrap_draws'],seed=c['seed'],horizon_ps=3)
            intervals.append(dict(collection=name,model=model,readout=readout,reference=ref,**result))
    counts=[]
    for split in ('fit','tune','development'):
        ix=np.array([i for i,r in enumerate(records) if r['split']==split and targets['at_risk'][i]])
        for h in HORIZONS:
            pos=ix[targets['event_bin'][ix]<=horizon_index(h)]
            counts.append(dict(split=split,horizon_ps=h,eligible_windows=len(ix),positive_windows=len(pos),
                               positive_roots=len({records[i]['root'] for i in pos})))
    snapshot_metric_docs(output,'onset_horizons')
    table(output/'tables/all-models.csv',rows);table(output/'tables/controls.csv',controls)
    table(output/'tables/paired-AP3.csv',intervals);table(output/'tables/event-counts.csv',counts)
    summary=[]
    for treatment in ('epi','vicreg','epi-variance'):
        selected=[r for r in rows if r['collection']=='mace_epi' and r['readout']=='encoder'
                  and r['model'] in [f'mace-{treatment}-s{s}-epoch024' for s in (123,456)]]
        if len(selected)!=2:raise ValueError(f'Missing matched final Epi seeds: {treatment}')
        summary.append(dict(objective=treatment,seeds=2,**{f'AP{h}':float(np.mean([r[f'AP{h}'] for r in selected])) for h in HORIZONS}))
    table(output/'tables/epi-final-means.csv',summary)
    lines=['# Onset horizon review: 3 ps primary','',
           'Saved checkpoints and readout selection are unchanged. This is retrospective scoring, not AP3-optimized training.',
           f'{len(rows)} model/readout evaluations replayed against saved probabilities; 3, 6 and 12 ps metrics verified.',
           'Historical metric exports are preserved. No encoder, probe or simulation was rerun.','',
           '**Only 3 development onset windows at 3 ps (3 roots); tuning has 2, fitting has 11.**',
           'AP measures ranking of onset within the horizon; it is not a guaranteed minimum warning time.','',
           '| Collection | Model | AP3 (primary) | AP6 | AP12 |','|---|---|---:|---:|---:|']
    for r in rows:
        if r['readout']=='joint':lines.append(f'| {r["collection"]} | {r["model"]} | {r["AP3"]:.4f} | {r["AP6"]:.4f} | {r["AP12"]:.4f} |')
    lines+=['','Final MACE/Epi: means of two seeds; separate current-physics-conditioned frozen hazards.','',
            '| Objective | AP3 | AP6 | AP12 |','|---|---:|---:|---:|']
    for r in summary:lines.append(f'| {r["objective"]} | {r["AP3"]:.4f} | {r["AP6"]:.4f} | {r["AP12"]:.4f} |')
    lines+=['','Full checkpoint/readout scores, including Geoformer, are in [all-models.csv](tables/all-models.csv).',
            'See [controls](tables/controls.csv), [event counts](tables/event-counts.csv), and [paired AP3 intervals](tables/paired-AP3.csv).',
            'The intervals resample whole roots within temperature; no-event draws are excluded and counted. They omit training-seed and model-search uncertainty.']
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    receipt=dict(state='complete',primary_horizon_ps=3,horizons_ps=list(HORIZONS),rows=len(rows),coverage=coverage,
                 config=c,inputs=provenance,no_refitting=True)
    (output/'technical/complete.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),rows=len(rows),coverage=coverage)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
