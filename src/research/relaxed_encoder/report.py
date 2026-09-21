"""Matched frozen-readout comparison, with whole-source uncertainty."""
import json
import numpy as np
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json
from src.research.local_predictability.metrics import stratified_bootstrap
from src.experiment_runner.metric_docs import write_metric_table
from .selection import evaluation_exclusions,selected_runs


def compare(base,other):
    if set(base['per_source'])!=set(other['per_source']):raise ValueError('Unpaired held-out sources')
    ids=sorted(base['per_source']);delta=np.array([base['per_source'][s]['event_nll']-other['per_source'][s]['event_nll'] for s in ids])
    temperatures=[base['per_source'][s]['temperature_K'] for s in ids]
    return dict(nll_gain=float(delta.mean()),ci95=stratified_bootstrap(delta[:,None],temperatures,draws=1000)[:,0].tolist())


def report(plan):
    c=plan['config'];root=resolve_path(c['output']);fits={}
    excluded=evaluation_exclusions(plan)
    reference=c.get('reference_run','instantaneous');expected=2*(len(selected_runs(plan))+6) if 'runs' in c else 18
    for p in (root/'readouts/technical/fits').glob('*/snapshot/*/metrics.json'):
        m=json.loads(p.read_text())
        if m['task']['encoder'] not in excluded:fits[m['task']['encoder'],m['task']['readout']]=m
    lines=['# Relaxed observations and targets: matched MACE study','',
        f'One seed; {sum(s["pilot_fit"] and s.get("validation_role",s["split"])=="train" for s in plan["sources"])} encoder-training sources. Observed origins in frames: {c["frames"]}. '
        'original independent source roles retained for frozen readouts. Limited event counts make this a pilot, not a final ranking. '
        'All readouts predict original MD onset at 0.75, 3, 6, 9 and 12 ps. Relaxed input requires per-frame preprocessing at deployment.', '',
        '| Encoder/input | Readout | Event NLL | 12 ps AP | Detected timing MAE (ps) | Missed / event windows |',
        '|---|---|---:|---:|---:|---|']
    gains={}
    if excluded:lines.insert(3,'Excluded encoders: '+ '; '.join(f'{name}: {r["reason"]}' for name,r in excluded.items()))
    exclusion=root/'technical/training-exclusions.json'
    assay=root/'technical/assay/ready.json'
    counts={}
    if exclusion.exists():counts['training_anchors']=json.loads(exclusion.read_text())['excluded_anchors']
    if assay.exists():counts['assay_windows']=json.loads(assay.read_text()).get('timeout_excluded_windows',0)
    if counts:
        lines[2]+=' Timeout exclusions (matched across all arms): '+', '.join(f'{k}={v}' for k,v in counts.items())+'.'
    for (name,kind),m in sorted(fits.items()):
        t=m['timing']['12.0'];ap=m['classification']['12.0']['average_precision'];mae=t['detected_timing_mae_ps']
        lines.append(f'| {name} | {kind} | {m["event_nll"]:.5f} | {ap if ap is not None else "undefined"} | {mae if mae is not None else "undefined"} | {t["missed_windows"]} / {t["event_windows"]} |')
        if (reference,kind) in fits:gains[name+'--'+kind]=compare(fits[reference,kind],m)
    lines+=['','Positive NLL gains favor the named method against instantaneous→instantaneous. Intervals resample whole test sources within temperature; they exclude seed uncertainty. '
        'Reconstruction scores across thermal versus quenched target domains do not establish a better encoder. '
        'The original-geometry control has its historical descriptor support; paired hot/cold geometry controls use exactly the same tracked 80 atoms as these encoders.',
        '',f'Completed readouts: {len(fits)}/{expected}. Reference: {reference}.']
    if 'runs' in c:
        lines+=['','## Encoder development diagnostics','','| Run | Physical | TDA | Raw rank | Correlation rank |','|---|---:|---:|---:|---:|']
        candidates=[]
        for run in selected_runs(plan):
            p=root/'technical/runs'/run['name']/'metrics.json'
            if not p.exists():continue
            m=json.loads(p.read_text());lines.append(f'| {run["name"]} | {m["physical"]:.4f} | {m["tda"]:.4f} | {m["invariant_effective_rank"]:.2f} | {m["invariant_correlation_effective_rank"]:.2f} |')
            if run['arm']=='relaxed_to_relaxed':candidates.append((m['selection_score'],run['name']))
        if candidates and len(candidates)==sum(r['arm']=='relaxed_to_relaxed' for r in selected_runs(plan)):
            score,name=min(candidates);save_json(root/'technical/development-selection.json',dict(name=name,score=score,criterion='Minimum cold-domain development Physical + .25 TDA; test scores never used; rank diagnostic only'))
    save_json(root/'technical/comparisons.json',gains)
    write_metric_table(gains,root/'comparison',family='relaxed_encoder',name='source-paired-gains')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
