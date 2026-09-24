"""Audit frozen 12-ps onset predictions and bootstrap complete development sources."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.structural_state.common import sha, write_json
from src.research.structural_state.report import table


def weighted_ap(actual, scores, weights):
    """Noninterpolated AP for a matrix of row weights, including complete ties."""
    order = np.argsort(-scores, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order]) != 0), len(order)-1]
    weighted = weights[:, order]
    tp = np.cumsum(weighted * actual[order][None], axis=1)[:, ends]
    total = np.cumsum(weighted, axis=1)[:, ends]
    precision = np.divide(tp, total, out=np.zeros_like(tp), where=total > 0)
    numerator = (precision * np.diff(np.c_[np.zeros(len(tp)), tp], axis=1)).sum(1)
    return np.divide(numerator, tp[:, -1], out=np.full(len(tp), np.nan), where=tp[:, -1] > 0)


def run(run_root, output, draws=2000, seed=20260922):
    run_root, output = Path(run_root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    tech = run_root/'technical'
    collection = json.loads((tech/'collection.json').read_text())
    if collection['state'] != 'complete':
        raise ValueError('Review requires all four completed arms')
    identity = json.loads((tech/'identity.json').read_text())
    cache = resolve_path(identity['config']['cache'])
    assert sha(cache/'manifest.json') == identity['data_sha256']
    records = json.loads((cache/'records.json').read_text())
    manifest = json.loads((cache/'manifest.json').read_text())
    assert sha(cache/'records.json') == manifest['files']['records.json']
    inputs = {str(cache/'records.json'): sha(cache/'records.json')}
    for name in collection['completed']:
        fit = tech/'fits'/name
        receipt = json.loads((fit/'complete.json').read_text())
        assert receipt['identity'] == collection['identity'] and receipt['step'] == 4096
        for file, key in [('last.pt','checkpoint_sha256'), ('features.npz','feature_sha256')]:
            assert sha(fit/file) == receipt[key]
            inputs[str(fit/file)] = receipt[key]
    anchor = tech/'evaluation/B-relaxed/exported/hazard_mlp/predictions.npz'
    with np.load(anchor) as p:
        indices, source, event = (p[k] for k in ('indices','source','event'))
    assert all(records[i]['split'] == 'development' for i in indices)
    np.testing.assert_array_equal(source, [records[i]['source'] for i in indices])
    ids, inverse, counts = np.unique(source, return_inverse=True, return_counts=True)
    temperatures = np.array([records[indices[np.flatnonzero(source == s)[0]]]['temperature_K'] for s in ids])
    rng = np.random.default_rng(seed)
    sampled = np.concatenate([rng.choice(np.flatnonzero(temperatures == t),
        size=(draws, int((temperatures == t).sum()))) for t in np.unique(temperatures)], axis=1)
    multiplicity = np.array([np.bincount(row, minlength=len(ids)) for row in sampled])
    weights = np.vstack([np.ones(len(ids)), multiplicity])[:, inverse] / counts[inverse][None] / len(ids)
    np.testing.assert_allclose(weights.sum(1), 1.)
    actual = event < 5
    metrics, summaries = {}, []
    for path in sorted((tech/'evaluation').glob('*/*/hazard_*/predictions.npz')):
        arm, representation, readout = path.parts[-4:-1]
        readout = readout.removeprefix('hazard_')
        inputs[str(path)] = sha(path)
        mpath = path.parent/'metrics.json'
        inputs[str(mpath)] = sha(mpath)
        published = json.loads(mpath.read_text())
        with np.load(path) as p:
            for k, expected in [('indices',indices),('source',source),('event',event)]:
                np.testing.assert_array_equal(p[k], expected)
            risk, logits = p['risks'][:, -1], p['logits'].astype(float)
        assert np.isfinite(logits).all() and np.isfinite(risk).all()
        bins = np.arange(5)[None]
        nll = (np.logaddexp(0, logits)*(bins < event[:,None]) +
               np.logaddexp(0, -logits)*(bins == event[:,None])).sum(1)
        probability = risk.clip(1e-7, 1-1e-7)
        estimates = dict(average_precision=weighted_ap(actual, risk, weights),
            brier=weights @ (risk-actual)**2,
            log_loss=weights @ -(actual*np.log(probability) + (~actual)*np.log1p(-probability)),
            nll=weights @ nll)
        for i in (0, 1, 31):
            np.testing.assert_allclose(estimates['average_precision'][i],
                average_precision_score(actual, risk, sample_weight=weights[i]), rtol=1e-12, atol=1e-12)
        for metric, values in estimates.items():
            expected = published['nll'] if metric == 'nll' else published['horizons']['12.0'][metric]
            np.testing.assert_allclose(values[0], expected, rtol=1e-6, atol=1e-8)
            valid = np.isfinite(values[1:])
            low, high = np.quantile(values[1:][valid], [.025,.975])
            summaries.append(dict(arm=arm, representation=representation, readout=readout,
                metric=metric, estimate=float(values[0]), ci95_low=float(low), ci95_high=float(high),
                valid_draws=int(valid.sum()), selected_step=published['best_step']))
        metrics[arm, representation, readout] = estimates
    comparisons = []
    contrasts = [
        ('relaxed_teacher', 'A-observed','exported','C-relaxed-teacher','exported'),
        ('physical_distance', 'B-relaxed','exported','D-physical-distance','exported'),
        ('relaxed_input', 'A-observed','exported','B-relaxed','exported'),
        ('relaxed_training', 'B-relaxed','initial_exported','B-relaxed','exported'),
        ('relaxed_descriptor', 'B-relaxed','descriptor','B-relaxed','exported'),
        ('relaxed_temperature', 'B-relaxed','conditions','B-relaxed','exported')]
    for label, ra, rr, ca, cr in contrasts:
        for readout in ('linear','mlp'):
            for metric in ('average_precision','brier','log_loss','nll'):
                reference = metrics[ra,rr,readout][metric]
                candidate = metrics[ca,cr,readout][metric]
                change = candidate-reference
                valid = np.isfinite(change[1:])
                low, high = np.quantile(change[1:][valid], [.025,.975])
                comparisons.append(dict(contrast=label, readout=readout, metric=metric,
                    reference_arm=ra, reference_representation=rr, candidate_arm=ca,
                    candidate_representation=cr, reference=float(reference[0]), candidate=float(candidate[0]),
                    delta=float(change[0]), ci95_low=float(low), ci95_high=float(high), valid_draws=int(valid.sum())))
    snapshot_metric_docs(output, 'structural_state_onset_review')
    table(output/'tables/onset_uncertainty.csv', summaries)
    table(output/'tables/paired_onset.csv', comparisons)
    write_json(output/'technical/audit.json', dict(run=str(run_root), identity=collection['identity'],
        inputs=inputs, sources=len(ids), event_windows=int(actual.sum()), windows=len(actual),
        event_sources=int(sum(np.any(actual[source == s]) for s in ids)), draws=draws, seed=seed,
        paired_rows_verified=True, checkpoints_verified=True, point_metrics_recomputed=True))
    print(json.dumps(dict(output=str(output), sources=len(ids), rows=len(actual), event_windows=int(actual.sum())), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.run, args.output)
