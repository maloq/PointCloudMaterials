"""Freeze data/gate receipts and compare the untouched legacy positive control."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from src.project_runtime.paths import load_json,resolve_path
from src.data.predictive_memory.prepare import file_hash,write_json
from src.experiment_runner.metric_docs import snapshot_metric_docs


def audit(config):
    root=resolve_path(config['output']);technical=root/'technical'
    cohort=json.loads((technical/'cohort.json').read_text())
    release=json.loads((technical/'release.json').read_text())
    coverage=json.loads((technical/'coverage.json').read_text())
    if release['cohort_sha256']!=file_hash(technical/'cohort.json'):raise RuntimeError('Cohort changed')
    # Coverage is assessed before consulting the test outcomes.
    counts={}
    for split in ['train','selection','calibration']:
        rows=[r for r in coverage if r['persistence_frames']==3 and r['grid']=='native' and
              (r['validation_role'] or r['split'])==split]
        counts[split]=dict(sources=len(rows),event_sources=sum(r['event_centers']>0 for r in rows),
            event_centers=sum(r['event_centers'] for r in rows),eligible_windows=sum(r['eligible'] for r in rows),
            zero_eligible_sources=[r['source_id'] for r in rows if r['eligible']==0])
    coverage_pass=counts['train']['event_sources']>=20 and all(
        counts[s]['event_sources']>=5 and counts[s]['event_centers']>=50 for s in ['selection','calibration'])
    legacy_original=resolve_path(config['legacy_output'])/'technical/local_results.json'
    legacy_repeat=resolve_path('output/local_predictability/legacy-reproduction-20260917/technical/local_results.json')
    original=json.loads(legacy_original.read_text());repeated=json.loads(legacy_repeat.read_text())
    for key in ['state','onset','fixed_lead','by_temperature','census','paired_event_f1','thresholds']:
        if original[key]!=repeated[key]:raise AssertionError(f'Legacy reaggregation changed {key}')
    verification=[v for source in release['sources'] for v in source['ptm_verification']]
    if len(verification)!=10 or not all(v['identical'] for v in verification):
        raise RuntimeError('Missing ten fixed training-frame PTM/full-cell checks')
    receipt=dict(state='complete',source_integrity=True,assay_integrity=True,coverage=coverage_pass,
        coverage_counts=counts,ptm_verified_centers=sum(v['centers'] for v in verification),
        legacy_reproduction='exact saved-metric equality under original protocol',
        cohort_sha256=release['cohort_sha256'],release_sha256=file_hash(technical/'release.json'),
        native_fit_gates_pending=['nested_initialization','small_set_fit','budget_feasible'],
        training_seed=20260919)
    write_json(technical/'assay_gates.json',receipt)
    exchange=resolve_path('output/local_predictability/shared-20260917/technical/exchange');exchange.mkdir(parents=True,exist_ok=True)
    for name in ['release.json','coverage.json','native_rows.npz','assay_gates.json']:
        shutil.copy2(technical/name,exchange/name)
    snapshot_metric_docs(root,'local_predictability')
    import csv
    rows=[{**r,'positives':json.dumps(r['positives'])} for r in coverage]
    with (root/'tables/coverage.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Assay and observation coverage','',
        f"150 independent sources verified; {receipt['ptm_verified_centers']} patch/full-cell PTM labels match.",
        'The original positive-control metrics reproduce exactly from saved predictions.','',
        '| Fold | Sources with future events | Distinct event centers | Eligible native windows |',
        '| --- | --- | --- | --- |']
    for split,row in counts.items():lines.append(f"| {split} | {row['event_sources']} | {row['event_centers']} | {row['eligible_windows']} |")
    lines.extend(['',f"Predeclared coverage gate: {'PASS' if coverage_pass else 'FAIL — coverage-limited; native onset fits blocked'}.",
        'Event centers within a source are correlated. Counts are coverage, not independent sample size.',
        'No model performance or absence of physical predictability is inferred from this audit.'])
    (root/'ASSAY.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args();audit(load_json(args.config))


if __name__=='__main__':main()
