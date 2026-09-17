"""Collect completed local-predictability studies and score deferred predictions on CPU."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from src.experiment_runner.metric_docs import snapshot_metric_docs
from .baselines import score_hazard
from .metrics import hazard_loss, source_weights, stratified_bootstrap


def aligned(reference, candidate, fields):
    """Return candidate row order, rejecting duplicates, absent rows or changed labels."""
    keys = ('source', 'center', 'anchor')
    a = list(zip(*(reference[k].tolist() for k in keys)))
    b = list(zip(*(candidate[k].tolist() for k in keys)))
    if len(set(a)) != len(a) or len(set(b)) != len(b) or set(a) != set(b):
        raise ValueError('Unpaired or duplicate source/center/anchor identities')
    lookup = {key: i for i, key in enumerate(b)}
    order = np.array([lookup[key] for key in a])
    for key in fields:
        np.testing.assert_array_equal(reference[key], candidate[key][order], err_msg=key)
    return order


def subset(arrays, split):
    index = arrays['split'] == split
    return {k: v[index] for k, v in arrays.items()}


def paired_rows(candidate, reference, values, temperatures, *, metric, horizon):
    """Equal-source differences and the established temperature-stratified interval."""
    sources = reference['source']
    ids = np.unique(sources)
    per_source = np.array([values[sources == sid].mean() for sid in ids])
    bounds = stratified_bootstrap(per_source, [temperatures[int(s)] for s in ids])
    return dict(candidate=candidate, metric=metric, horizon_ps=horizon,
                difference=float(per_source.mean()), ci95_lower=float(bounds[0]),
                ci95_upper=float(bounds[1]), sources=len(ids), rows=len(sources))


class Evidence:
    def __init__(self):
        self.inputs = {}

    def record(self, path):
        path = Path(path)
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        self.inputs[str(path)] = dict(sha256=digest.hexdigest(), bytes=path.stat().st_size)
        return path

    def json(self, path):
        return json.loads(self.record(path).read_text())

    def npz(self, path):
        with np.load(self.record(path)) as data:
            return {k: data[k] for k in data.files if k not in ('embeddings', 'state', 'indices')}


def table(root, name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (root / 'tables' / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect(config_path, root):
    torch.set_num_threads(2)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for directory in ('tables', 'technical', 'plots'):
        (root / directory).mkdir()
    evidence = Evidence()
    config = evidence.json(config_path)
    runs = {k: Path(v) for k, v in config['runs'].items()}
    plan = evidence.json(config['plan'])
    release = evidence.json(runs['descriptors'] / 'technical/release.json')
    temperatures = {s['id']: s['temperature_K'] for s in release['sources']}
    horizons = plan['sampling']['horizons_ps']
    status_files = dict(descriptors='baseline_status.json', native='status.json',
                        readouts='readout_status.json', packet_observability='observability_status.json',
                        raw_observability='raw_observability_status.json', h100_physical='screen_status.json',
                        rtx_physical='screen_status.json', comparison='repeat_status.json', topology='status.json')
    inventory = []
    for name, filename in status_files.items():
        status = evidence.json(runs[name] / 'technical' / filename)
        if status['state'] != 'complete':
            raise ValueError(f'{name} has no completed receipt: {status}')
        inventory.append(dict(study=name, path=str(runs[name]), state=status['state']))
    scores, onset_rows, predictions = {}, [], {}

    def score(name, arrays, grid, *, binary_horizon=None):
        arrays['y'] = arrays['event_bin']
        if not np.isfinite(arrays['probability']).all() or not np.isfinite(arrays['logits']).all():
            raise ValueError(f'Nonfinite predictions: {name}')
        local_plan = dict(plan, sampling=dict(plan['sampling']))
        if binary_horizon is not None:
            local_plan['sampling']['horizons_ps'] = [binary_horizon]
        masks = {s: np.flatnonzero(arrays['split'] == s) for s in ('selection', 'calibration', 'test')}
        if any(len(i) == 0 for i in masks.values()):
            raise ValueError(f'Missing evaluation fold: {name}')
        result = score_hazard(arrays, arrays['probability'], arrays['logits'], masks, local_plan)
        scores[name] = result
        rows = [dict(model=name, grid=grid, joint_event_nll=result['joint_event_nll']['test'], **r)
                for r in result['population'] if r['split'] == 'test']
        return rows

    # Native predictions are the row/label reference for every prospective comparison.
    for variant in ('snapshot', 'history12', 'repeat12'):
        name = f'mace-native-{variant}'
        arrays = evidence.npz(runs['native'] / f'technical/{variant}/predictions.npz')
        onset_rows += score(name, arrays, 'native')
        predictions[name] = subset(arrays, 'test')
        old = evidence.json(runs['native'] / f'technical/{variant}/result.json')
        np.testing.assert_allclose(scores[name]['joint_event_nll']['test'], old['joint_event_nll']['test'], atol=1e-7)
    reference = predictions['mace-native-snapshot']
    descriptor_results = evidence.json(runs['descriptors'] / 'technical/descriptor_results.json')
    for name, result in descriptor_results.items():
        if name.startswith('ridge-'):
            continue
        scores[f'dense-{name}'] = result
        onset_rows += [dict(model=name, grid='descriptor_3ps', joint_event_nll=result['joint_event_nll']['test'], **r)
                       for r in result['population'] if r['split'] == 'test']
        arrays = evidence.npz(runs['descriptors'] / f'technical/descriptors/{name}/predictions.npz')
        mask = np.isin(arrays['anchor'], release['native_anchors'])
        arrays = {k: v[mask] for k, v in arrays.items()}
        onset_rows += score(name, arrays, 'native')
        predictions[name] = subset(arrays, 'test')
    for variant in ('snapshot', 'history12', 'repeat12'):
        for kind in ('linear', 'mlp'):
            name = f'mace-frozen-{variant}-{kind}'
            arrays = evidence.npz(runs['readouts'] / f'technical/{variant}/{kind}/predictions.npz')
            onset_rows += score(name, arrays, 'native')
            predictions[name] = subset(arrays, 'test')
        parts = []
        for split in ('selection', 'calibration', 'test'):
            arrays = evidence.npz(runs['gatr_onset'] / f'technical/axial_gatr/onset/{variant}/{split}_predictions.npz')
            arrays['split'] = np.full(len(arrays['source']), split)
            arrays['temperature'] = np.array([temperatures[int(s)] for s in arrays['source']])
            parts.append(arrays)
        arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        name = f'gatr-native-{variant}'
        onset_rows += score(name, arrays, 'native')
        predictions[name] = subset(arrays, 'test')
    for name, arrays in predictions.items():
        aligned(reference, arrays, ('event_bin', 'split', 'temperature'))

    observability_rows = []
    for task in ('current-state', 'future-state-9ps', 'future-state-48ps',
                 'future-sequence-onset-9ps', 'future-sequence-onset-48ps'):
        case = evidence.json(runs['packet_observability'] / f'technical/{task}/task.json')
        for kind in ('linear', 'mlp'):
            arrays = evidence.npz(runs['packet_observability'] / f'technical/{task}/{kind}/predictions.npz')
            observability_rows += score(f'packet-{task}-{kind}', arrays,
                                       'native_all_states' if case['task'] == 'state' else 'native_at_risk',
                                       binary_horizon=case['horizon_ps'])
    parts = []
    for split in ('selection', 'calibration', 'test'):
        raw = evidence.npz(runs['raw_observability'] / f'technical/{split}_predictions.npz')
        arrays = dict(probability=raw['probability'], logits=raw['logits'], event_bin=raw['binary_target'],
                      source=raw['source_id'], center=raw['center_id'], anchor=raw['anchor'], split=raw['split'],
                      temperature=np.array([temperatures[int(s)] for s in raw['source_id']]))
        parts.append(arrays)
    arrays = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    packet = evidence.npz(runs['packet_observability'] / 'technical/current-state/mlp/predictions.npz')
    aligned(subset(packet, 'test'), subset(arrays, 'test'), ('event_bin',))
    observability_rows += score('raw-mace-current-state', arrays, 'native_all_states', binary_horizon=0)

    comparisons = [('mlp-packet_H12', 'mlp-packet_H0'), ('mlp-packet_H12', 'mlp-packet_repeat12'),
                   ('mlp-packet_plus_shell25_H0', 'mlp-packet_H0'), ('mlp-packet_H48', 'mlp-packet_H12')]
    for architecture in ('mace', 'gatr'):
        comparisons += [(f'{architecture}-native-history12', f'{architecture}-native-{v}') for v in ('snapshot', 'repeat12')]
    comparisons += [('mace-frozen-history12-mlp', 'mace-frozen-snapshot-mlp'),
                    ('mace-frozen-history12-mlp', 'mace-frozen-repeat12-mlp'),
                    ('mace-frozen-snapshot-mlp', 'mace-native-snapshot'),
                    ('mace-frozen-snapshot-mlp', 'mlp-packet_H0')]
    paired = []
    for candidate, ref in comparisons:
        a, b = predictions[candidate], predictions[ref]
        order = aligned(b, a, ('event_bin',))
        loss_a = hazard_loss(torch.from_numpy(a['logits'][order]), torch.from_numpy(b['event_bin'])).numpy()
        loss_b = hazard_loss(torch.from_numpy(b['logits']), torch.from_numpy(b['event_bin'])).numpy()
        row = paired_rows(candidate, b, loss_a - loss_b, temperatures, metric='joint_event_nll', horizon='all')
        paired.append(dict(reference=ref, **row))
        for h, horizon in enumerate(horizons):
            y = b['event_bin'] <= h
            pa, pb = a['probability'][order, h].astype(float), b['probability'][:, h].astype(float)
            ca, cb = np.clip(pa, 1e-7, 1-1e-7), np.clip(pb, 1e-7, 1-1e-7)
            errors = dict(brier=(pa-y)**2-(pb-y)**2,
                          log_loss=-(y*np.log(ca)+(1-y)*np.log1p(-ca)) + y*np.log(cb)+(1-y)*np.log1p(-cb))
            for metric, values in errors.items():
                row = paired_rows(candidate, b, values, temperatures, metric=metric, horizon=horizon)
                paired.append(dict(reference=ref, **row))

    physical, physical_rows = {}, []
    for name in (k for k in descriptor_results if k.startswith('ridge-')):
        physical[name] = evidence.npz(runs['descriptors'] / f'technical/descriptors/{name}/test_predictions.npz')
    for device in ('h100', 'rtx'):
        for encoder in ('mace', 'axial_gatr'):
            name = f'{device}-{encoder}'
            raw = evidence.npz(runs[f'{device}_physical'] / f'technical/{encoder}/physical_means/snapshot/test_predictions.npz')
            physical[name] = dict(source=raw['source'], center=raw['center'], anchor=raw['anchor'],
                                  prediction=raw['future'], target=raw['targets'][:, 1:])
            weight = source_weights(raw['source'])
            for label, values in [('present', (raw['present']-raw['targets'][:, 0])**2),
                                  ('future_mean', (raw['future']-raw['targets'][:, 1:])**2)]:
                physical_rows.append(dict(model=name, metric=label, horizon_ps='all', rows=len(weight),
                                          mse=float(weight @ values.reshape(len(weight), -1).mean(1))))
    reference_physical = physical['ridge-packet_H0']
    for name, arrays in physical.items():
        order = aligned(reference_physical, arrays, ())
        np.testing.assert_allclose(arrays['target'][order], reference_physical['target'], atol=1e-5, rtol=1e-5)
        error = (arrays['prediction'].astype(float)-arrays['target'])**2
        arrays['errors'] = error.mean(-1)
        weight = source_weights(arrays['source'])
        if name.startswith('ridge-'):
            physical_rows.append(dict(model=name, metric='future_mean', horizon_ps='all', rows=len(weight),
                                      mse=float(weight @ arrays['errors'].mean(1))))
        for h, horizon in enumerate(horizons):
            physical_rows.append(dict(model=name, metric='future', horizon_ps=horizon, rows=len(weight),
                                      mse=float(weight @ arrays['errors'][:, h])))
    for row in descriptor_results['ridge-packet_H0']['test']:
        if row['method'] == 'persistence' and row['block'] == 'all':
            physical_rows.append(dict(model='persistence', metric='future', horizon_ps=row['horizon_ps'], rows=7680, mse=row['mse']))
    physical_pairs = []
    for candidate, ref in [('h100-axial_gatr', 'h100-mace'), ('h100-mace', 'ridge-packet_H0'),
                           ('ridge-packet_H12', 'ridge-packet_H0')]:
        a, b = physical[candidate], physical[ref]
        order = aligned(b, a, ())
        differences = a['errors'][order] - b['errors']
        for h, values in [('all', differences.mean(1)), *zip(horizons, differences.T)]:
            physical_pairs.append(dict(reference=ref, **paired_rows(candidate, b, values, temperatures,
                                                                    metric='future_mse', horizon=h)))
    for name, rows in [('onset_test', onset_rows), ('observability', observability_rows),
                       ('paired_onset', paired), ('physical', physical_rows), ('paired_physical', physical_pairs),
                       ('completion', inventory)]:
        table(root, name, rows)
    for origin, filename, destination in [('topology', 'topology.csv', 'topology.csv'),
                                          ('comparison', 'h100_speed.csv', 'h100_speed.csv'),
                                          ('descriptors', 'paired_descriptor_differences.csv', 'paired_descriptor_original.csv')]:
        shutil.copyfile(evidence.record(runs[origin] / 'tables' / filename), root / 'tables' / destination)
        evidence.record(runs[origin] / 'tables/METRICS.md')
        evidence.record(runs[origin] / 'technical/metric-contract.json')
    evidence.json(config['h200_report'])
    shutil.copyfile(config['h200_report'], root / 'technical/h200_reported_results.json')
    h200_native = evidence.json(config['h200_native_report'])
    shutil.copyfile(config['h200_native_report'], root / 'technical/h200_native_reported_results.json')
    evidence.record(config['previous_report'])
    for path in config['supporting_artifacts']:
        evidence.record(path)
    for run in runs.values():
        for path in sorted((run / 'technical').glob('**/training.json')):
            evidence.record(path)
    captured = datetime.now(timezone.utc).isoformat()
    (root / 'technical/scores.json').write_text(json.dumps(scores, indent=2) + '\n')
    (root / 'technical/inputs.json').write_text(json.dumps(evidence.inputs, indent=2) + '\n')
    snapshot_metric_docs(root, 'local_predictability_summary')
    make_plots(root, onset_rows, physical_rows)
    (root / 'technical/status.json').write_text(json.dumps(dict(state='complete', captured_at=captured,
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        training_launched=False, test_rows_verified=len(reference['source']),
        physical_test_rows_verified=len(reference_physical['source']),
        h200_status=dict(training=h200_native['training_status'],
                         test_comparisons=h200_native['test_comparison_status'],
                         provenance='User-reported; raw native test comparisons not received')), indent=2) + '\n')
    print(f'Collected {len(onset_rows)} onset, {len(observability_rows)} observability and {len(physical_rows)} physical rows in {root}')


def make_plots(root, onset, physical):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['mlp-condition', 'mace-native-snapshot', 'mace-frozen-snapshot-mlp', 'gatr-native-history12',
             'mlp-packet_H0', 'mlp-packet_H12', 'mlp-packet_repeat12', 'mlp-packet_plus_shell25_H0']
    labels = ['Conditions', 'MACE native snapshot', 'MACE frozen + MLP', 'GATr native H12',
              'Packet snapshot + MLP', 'Packet H12 + MLP', 'Packet repeat + MLP', 'Packet + 25 Å context']
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for ax, horizon in zip(axes, [9, 48]):
        rows = [next(r for r in onset if r['grid']=='native' and r['model']==name and r['horizon_ps']==horizon) for name in names]
        ax.barh(labels, [r['average_precision'] for r in rows], color=['#8094a6']*4+['#2d947b']*4)
        ax.axvline(rows[0]['prevalence'], color='black', ls=':', label='Source-weighted prevalence')
        ax.set(title=f'{horizon} ps onset forecast', xlabel='Average precision (higher is better)', xlim=(0, .6))
        ax.legend(fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle('Same 4,691 test windows / 30 sources; one seed\nDescriptor fits used denser training origins; 25 Å context has wider input', fontsize=11)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(root / f'plots/onset_comparison.{extension}', dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, label in [('persistence', 'Persistence'), ('ridge-condition', 'Condition ridge'),
                        ('ridge-packet_H0', 'Packet snapshot ridge'), ('ridge-packet_H12', 'Packet H12 ridge'),
                        ('h100-mace', 'MACE snapshot'), ('h100-axial_gatr', 'GATr snapshot')]:
        rows = [r for r in physical if r['model']==name and r['metric']=='future']
        ax.plot([r['horizon_ps'] for r in rows], [r['mse'] for r in rows], 'o-', label=label)
    ax.set(xscale='log', xlabel='Forecast horizon (ps)', ylabel='Standardized physical MSE (lower is better)',
           title='All 7,680 test windows / 30 sources; same physical targets')
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(root / f'plots/physical_forecasts.{extension}', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    collect(args.config, args.output)


if __name__ == '__main__':
    main()
