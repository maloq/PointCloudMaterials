"""Source-cluster uncertainty and explicit local-event coverage for the large assay."""
import json
from pathlib import Path

import numpy as np
import torch

from src.data.structural_pretraining.prepare import save_json
from src.experiment_runner.metric_docs import write_metric_table
from src.project_runtime.paths import resolve_path
from src.research.local_predictability.metrics import source_weights, cumulative_risk, hazard_loss
from .selection import selected_runs,evaluation_exclusions


def weighted_ap(actual, score, weight):
    order=np.argsort(-score,kind='stable');y=actual[order];w=weight[order]
    ends=np.flatnonzero(np.r_[score[order][1:]!=score[order][:-1],True])
    return _sorted_ap(y,w,ends)


def _sorted_ap(y,w,ends):
    positive=np.cumsum(w*y)[ends];total=np.cumsum(w)[ends]
    if positive[-1]<=0:return float('nan')
    precision=np.divide(positive,total,out=np.zeros_like(positive),where=total>0)
    return float(np.diff(np.r_[0,positive])@precision/positive[-1])


def source_draws(source, temperature, draws, seed):
    ids,inverse=np.unique(source,return_inverse=True)
    temperatures=np.array([temperature[np.flatnonzero(source==s)[0]] for s in ids])
    rng=np.random.default_rng(seed);counts=np.zeros((draws,len(ids)),np.int64)
    for temp in np.unique(temperatures):
        members=np.flatnonzero(temperatures==temp)
        picked=rng.choice(members,(draws,len(members)),replace=True)
        for row in range(draws):np.add.at(counts[row],picked[row],1)
    return counts,inverse


def bootstrap_values(pop, ids, logits, metrics, counts, inverse):
    source=pop['source'][ids];weight=source_weights(source)
    risk=cumulative_risk(torch.from_numpy(logits)).numpy().astype(np.float64)
    nll=hazard_loss(torch.from_numpy(logits),torch.from_numpy(pop['event'][ids])).numpy()
    actual=pop['event'][ids]<5;threshold=metrics['classification']['12.0']['threshold']
    hit=actual&(risk[:,-1]>=threshold)
    probability=np.diff(np.c_[np.zeros(len(ids)),risk],axis=1)
    midpoint=np.array([.375,1.875,4.5,7.5,10.5])
    predicted=(probability@midpoint)/np.maximum(risk[:,-1],1e-12)
    error=np.abs(predicted-pop['delay'][ids])
    order=np.argsort(-risk[:,-1],kind='stable');sorted_actual=actual[order]
    ends=np.flatnonzero(np.r_[risk[order[1:],-1]!=risk[order[:-1],-1],True])
    result={k:[] for k in ('event_nll','ap_12ps','recall_12ps','detected_timing_mae_ps',
                           'missed_event_fraction','timed_within_3ps_recall')}
    for count in counts:
        multiplicity=count[inverse];w=weight*multiplicity
        positive=w@actual;events=multiplicity@actual;detected=multiplicity@hit
        result['event_nll'].append(float(w@nll))
        result['ap_12ps'].append(_sorted_ap(sorted_actual,w[order],ends))
        result['recall_12ps'].append(float(w@hit/positive) if positive else np.nan)
        result['detected_timing_mae_ps'].append(float(multiplicity@(error*hit)/detected) if detected else np.nan)
        result['missed_event_fraction'].append(float(multiplicity@(actual&~hit)/events) if events else np.nan)
        result['timed_within_3ps_recall'].append(float(multiplicity@(hit&(error<=3))/events) if events else np.nan)
    return {k:np.array(v) for k,v in result.items()}


def interval(values):
    x=values[np.isfinite(values)]
    return dict(ci95=np.quantile(x,[.025,.975]).tolist() if len(x) else [None,None],
                valid_draws=len(x))


def report(plan, *, bootstrap):
    c=plan['config'];root=resolve_path(c['output']);fits={}
    excluded=evaluation_exclusions(plan)
    for path in (root/'readouts/technical/fits').glob('*/snapshot/*/metrics.json'):
        m=json.loads(path.read_text())
        if m['task']['encoder'] not in excluded:fits[m['task']['encoder'],m['task']['readout']]=(path,m)
    pop=dict(np.load(root/'technical/assay/population.npz'))
    ids=np.flatnonzero(pop['role']=='test');cal=np.flatnonzero(pop['role']=='calibration')
    coverage=json.loads((root/'technical/actual-cohort.json').read_text())
    cohort=coverage['test'];uncertainty={};draw_values={};gains={}
    if bootstrap:
        counts,inverse=source_draws(pop['source'][ids],pop['temperature'][ids],
                                   plan['evaluation_recipe']['bootstrap_draws'],c['seed'])
        for (name,kind),(path,m) in fits.items():
            with np.load(path.with_name('predictions.npz')) as p:
                np.testing.assert_array_equal(p['test_indices'],ids)
                np.testing.assert_array_equal(p['calibration_indices'],cal)
                draw_values[name,kind]=bootstrap_values(pop,ids,p['test'],m,counts,inverse)
            uncertainty[name+'--'+kind]={k:interval(v) for k,v in draw_values[name,kind].items()}
        for (name,kind),values in draw_values.items():
            for reference in ('hot-control','parent_cold','geometry_cold'):
                if (reference,kind) not in draw_values:continue
                base=draw_values[reference,kind]
                gains[f'{name}--{kind}--vs--{reference}']=dict(
                    event_nll_gain=interval(base['event_nll']-values['event_nll']),
                    ap_12ps_gain=interval(values['ap_12ps']-base['ap_12ps']))
        save_json(root/'technical/uncertainty.json',dict(uncertainty=uncertainty,paired_gains=gains))
        write_metric_table(dict(uncertainty=uncertainty,paired_gains=gains),root/'uncertainty',
                           family='relaxed_encoder_large_test',name='source-bootstrap')
    expected=2*(len(selected_runs(plan))+2+len(plan['evaluation_recipe']['cached_encoders'])+4)
    lines=['# Larger matched crystallization test', '',
        f'{cohort["windows"]} natural at-risk test windows; '
        f'{cohort["distinct_local_onsets_12ps"]} distinct local atom onsets within 12 ps, '
        f'in {cohort["event_sources_12ps"]} event-bearing sources out of {cohort["sources"]} test sources. '
        'Local atom onsets within a trajectory are correlated; they are not independent nucleation events.', '',
        'Selection, calibration and test origins use a fixed 12 ps cadence. No transition oversampling, '
        'test-source reassignment or selection by quench completion time. Original MD labels and source '
        'splits are retained. Training readouts retain the 15-origin training cohort. '
        'This is a historically used source split, not a newly untouched dataset.', '',
        '| Encoder / input | Readout | Event NLL | 12 ps AP | AUROC | Timing MAE (ps) | Misses / events |',
        '|---|---|---:|---:|---:|---:|---:|']
    def fmt(v):return 'undefined' if v is None else f'{v:.4f}'
    for (name,kind),(_,m) in sorted(fits.items()):
        t=m['timing']['12.0'];v=m['classification']['12.0']
        lines.append(f'| {name} | {kind} | {fmt(m["event_nll"])} | {fmt(v["average_precision"])} | '
                     f'{fmt(v["auroc"])} | {fmt(t["detected_timing_mae_ps"])} | {t["missed_windows"]}/{t["event_windows"]} |')
    lines+=['', 'All encoders are frozen. Neural heads and linear heads are compared separately. '
        'Calibration-only thresholds target 5% source-weighted false-positive rate. Timing MAE '
        'covers detections only; also report misses and timing-within-3-ps recall. '
        'Whole-source bootstrap intervals preserve temperature strata and exclude training-seed uncertainty. '
        'Legacy cached encoders use unrelaxed inputs; parent_hot/parent_cold isolate input relaxation.', '',
        f'Completed readouts: {len(fits)}/{expected}. Bootstrap exported: {bootstrap}. '
        'Full horizon metrics: readouts/tables/. Paired intervals: uncertainty/tables/.']
    if excluded:lines+=['','Excluded encoders: '+ '; '.join(f'{name}: {r["reason"]}' for name,r in excluded.items())]
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
