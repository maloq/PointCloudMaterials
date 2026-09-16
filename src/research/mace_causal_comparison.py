"""Compare completed causal-state fits on paired held-out physical outcomes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json
from src.research.mace_velocity.train import GROUPS


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def verify_pair(reference, candidate):
    """Pair by exact producer identifiers and targets, never by row count alone."""
    for field in ('source_id', 'center_atom_id', 'anchor_ps', 'present_target', 'future_target'):
        if not np.array_equal(reference[field], candidate[field]):
            raise ValueError(f'Unpaired held-out predictions: {field}')


def source_values(rows):
    values = {}
    blocks = {}
    for row in rows:
        if row['value'] == '':
            continue
        key = (row['population'], row['method'], row['metric'], int(row['source_id']))
        if key in values:
            raise ValueError(f'Duplicate source metric: {key}')
        values[key] = float(row['value'])
        metric = row['metric']
        if metric.startswith(('present/', 'future/', 'delta/')) and metric.split('/')[-1] in GROUPS:
            group = (row['population'], row['method'], metric.rsplit('/', 1)[0]+'/block_mean', int(row['source_id']))
            blocks.setdefault(group, []).append(float(row['value']))
    for key, block in blocks.items():
        if len(block) != 6:
            raise ValueError(f'Incomplete six-block physical score: {key}: {len(block)}')
        values[key] = float(np.mean(block))
    return values


def summarize(records, draws, seed, pairs):
    """Average training seeds within source before source bootstrap/pairing."""
    rng = np.random.default_rng(seed)
    by_model = {}
    for record in records:
        key = (record['variant'], record['readout'], record['population'], record['method'], record['metric'])
        source = by_model.setdefault(key, {}).setdefault(record['source_id'], {})
        if record['seed'] in source:
            raise ValueError(f'Duplicate seed/source measurement: {key}')
        source[record['seed']] = record['value']

    def interval(a):
        low, high = np.quantile(a[rng.integers(len(a), size=(draws, len(a)))].mean(-1), [.025, .975])
        return dict(value=float(a.mean()), ci95_low=float(low) if len(a) > 1 else None,
                    ci95_high=float(high) if len(a) > 1 else None, sources=len(a))

    summary = []
    for key, sources in sorted(by_model.items()):
        seeds = {tuple(sorted(s)) for s in sources.values()}
        if len(seeds) != 1:
            raise ValueError(f'Missing seed/source measurements: {key}')
        summary.append(dict(zip(('variant', 'readout', 'population', 'method', 'metric'), key),
                            seeds=len(next(iter(seeds))), **interval(np.array([np.mean(list(s.values())) for s in sources.values()]))))
    differences = []
    for left, right in pairs:
        for key, sources in sorted(by_model.items()):
            variant, readout, population, method, metric = key
            if variant != left or method != 'encoder':
                continue
            other = by_model[(right, readout, population, method, metric)]
            if sources.keys() != other.keys() or any(sources[s].keys() != other[s].keys() for s in sources):
                raise ValueError(f'Unmatched paired sources/seeds: {left}, {right}, {key}')
            delta = np.array([np.mean([sources[s][seed]-other[s][seed] for seed in sources[s]]) for s in sources])
            differences.append(dict(left=left, right=right, readout=readout, population=population,
                metric=metric, seeds=len(next(iter(sources.values()))), **interval(delta)))
    return summary, differences


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No measurements to export: {path}')
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def readouts_for_variant(config, variant):
    selected = config.get('diagnostic_variants', [])
    if set(selected)-set(config['variants']):
        raise ValueError('History diagnostic variants must be declared compared encoders')
    readouts = list(config['readouts'])
    if variant in selected:
        for mode in ('state_constant', 'state_history'):
            if mode not in readouts:
                readouts.append(mode)
    return readouts


def run(config):
    root = Path(config['output'])
    if root.exists():
        raise FileExistsError(f'Preserve prior comparison: {root}')
    records, inputs, events = [], [], []
    reference = None
    reference_recipe = None
    reference_contract = None
    for recipe in config['recipes']:
        fit = load_json(recipe)
        matched = {k: v for k, v in fit.items() if k not in ('seed', 'output', 'smoothness')}
        matched['smoothness'] = {k: v for k, v in fit['smoothness'].items() if k != 'reference_checkpoint'}
        if reference_recipe is not None and matched != reference_recipe:
            raise ValueError(f'Unmatched data/architecture/training recipe: {recipe}')
        reference_recipe = matched
        for variant in config['variants']:
            base = Path(fit['output']).with_name(Path(fit['output']).name+'-'+variant)
            for readout in readouts_for_variant(config, variant):
                run_root = base if readout == 'joint' else base.with_name(base.name+'-probe-'+readout)
                status = json.loads((run_root/'technical/status.json').read_text())
                if status['state'] != 'complete':
                    raise ValueError(f'Fit is incomplete: {run_root}: {status}')
                contract = json.loads((run_root/'technical/metric-contract.json').read_text())
                if reference_contract is not None and contract['files'] != reference_contract:
                    raise ValueError(f'Encoder/metric implementation differs across fits: {run_root}')
                reference_contract = contract['files']
                path = run_root/'technical/test-predictions.npz'
                with np.load(path) as arrays:
                    current = {k: arrays[k] for k in ('source_id', 'center_atom_id', 'anchor_ps', 'present_target', 'future_target')}
                if reference is None:
                    reference = current
                else:
                    verify_pair(reference, current)
                # A has no trained joint future/hazard head. Its frozen probes
                # provide the fair future comparison.
                rows = read_csv(run_root/'tables/test-sources.csv')
                for (population, method, metric, source_id), value in source_values(rows).items():
                    if variant == 'A' and readout == 'joint' and metric.startswith(('future/', 'delta/', 'path/', 'hazard/')):
                        continue
                    records.append(dict(seed=fit['seed'], variant=variant, readout=readout, population=population,
                                        method=method, metric=metric, source_id=source_id, value=value))
                if not (variant == 'A' and readout == 'joint'):
                    for row in read_csv(run_root/'tables/test-events.csv'):
                        events.append(dict(seed=fit['seed'], variant=variant, readout=readout, **row))
                inputs.append(dict(path=str(run_root), seed=fit['seed'], variant=variant, readout=readout,
                    selected_step=status['selected_step'],
                    predictions_sha256=sha256(path), source_metrics_sha256=sha256(run_root/'tables/test-sources.csv'),
                    metric_contract=contract))
    # A's joint future values are absent by design, so compare only keys defined
    # by both models. All remaining source/seed pairings must be exact.
    # Extra, predeclared history-access diagnostics compare readouts within that
    # encoder; between-encoder comparisons use the common requested readouts.
    common = [r for r in records if r['readout'] in config['readouts']]
    keys = {(r['variant'], r['readout'], r['population'], r['method'], r['metric']) for r in common}
    summary, _ = summarize(records, config['bootstrap_draws'], config['seed'], [])
    differences = []
    for left, right in config['comparison_pairs']:
        left_keys = {key[1:] for key in keys if key[0] == left}
        right_keys = {key[1:] for key in keys if key[0] == right}
        for readout, population, method, metric in left_keys ^ right_keys:
            if not ('A' in (left, right) and readout == 'joint'
                    and metric.startswith(('future/', 'delta/', 'path/', 'hazard/'))):
                raise ValueError(f'Missing paired metric: {left}, {right}, {readout}, {metric}')
        paired = [r for r in common if r['variant'] in (left, right)
                  and (left, r['readout'], r['population'], r['method'], r['metric']) in keys
                  and (right, r['readout'], r['population'], r['method'], r['metric']) in keys]
        _, delta = summarize(paired, config['bootstrap_draws'], config['seed'], [(left, right)])
        differences.extend(delta)
    sufficiency = []
    if {'state_history', 'state_constant'} <= {r['readout'] for r in records}:
        paired = [dict(r, variant=r['readout'], readout=r['variant']) for r in records
                  if r['readout'] in ('state_history', 'state_constant') and r['method'] == 'encoder'
                  and r['metric'].startswith(('future/', 'delta/', 'path/', 'hazard/'))]
        _, sufficiency = summarize(paired, config['bootstrap_draws'], config['seed'], [('state_history', 'state_constant')])
    snapshot_metric_docs(root, 'mace_causal_comparison')
    for name, rows in [('sources', records), ('summary', summary), ('paired-differences', differences), ('events', events)]:
        write_csv(root/'tables'/f'{name}.csv', rows)
    if sufficiency:
        write_csv(root/'tables/sufficiency.csv', sufficiency)
    write_json(root/'technical/config.json', config)
    write_json(root/'technical/inputs.json', inputs)
    plot(root, summary, config['variants'])
    report(root, summary, differences, sufficiency, config)
    write_json(root/'technical/status.json', dict(state='complete', fits=len(inputs), seeds=len(config['recipes']),
                                                 test_sources=len(np.unique(reference['source_id']))))


def report(root, summary, differences, sufficiency, config):
    def measurement(variant, readout, metric):
        row, = [r for r in summary if (r['variant'], r['readout'], r['population'], r['method'], r['metric'])
                == (variant, readout, 'low_order', 'encoder', metric)]
        return f"{row['value']:.4f}"

    lines = ['# Causal MACE matched pilot', '',
        'Completed held-out comparison. Values below are seed means averaged equally across low-order sources;',
        'low order means current group qbar6 < 0.30, not a phase assignment.', '',
        '| Encoder | Present physical MSE | 9 ps future physical MSE | J at 0.75 ps |',
        '| --- | ---: | ---: | ---: |']
    for variant in config['variants']:
        values = [measurement(variant, 'nonlinear', 'present/block_mean'),
                  measurement(variant, 'nonlinear', 'future/9ps/block_mean'),
                  measurement(variant, 'joint', 'temporal/0.75ps/normalized_rms_jump')]
        lines.append('| '+variant+' | '+' | '.join(values)+' |')
    lines += ['', 'Physical errors use matched frozen nonlinear readouts and equal weight for the six target blocks.',
              'E is a second training phase initialized from D; its total training budget is larger.',
              'These are pilot fits, not an assumption of optimization convergence.', '',
              '![Physical comparison](plots/physical-comparison.png)', '',
              '| Comparison | 9 ps physical MSE difference | 95% source interval |',
              '| --- | ---: | --- |']
    for row in differences:
        if (row['readout'], row['population'], row['metric']) == ('nonlinear', 'low_order', 'future/9ps/block_mean'):
            bounds = 'undefined' if row['ci95_low'] is None else f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}]"
            lines.append(f"| {row['left']} minus {row['right']} | {row['value']:.4f} | {bounds} |")
    lines += ['', 'Negative error differences favor the first model. Intervals resample whole sources after',
              'averaging seeds; they do not quantify training-seed uncertainty.', '',
              'The [source table](tables/sources.csv) retains every seed and source. See the',
              '[paired differences](tables/paired-differences.csv), [event results](tables/events.csv),',
              '[metric definitions](tables/METRICS.md), and [input identities](technical/inputs.json).']
    if sufficiency:
        lines += ['', 'The [history-access diagnostic](tables/sufficiency.csv) compares matched predictors with',
                  'and without variable raw observed history alongside frozen z. A gain means accessible',
                  'predictive information was omitted from z; no gain does not prove sufficiency.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def plot(root, summary, variants):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [('present/block_mean', 'Present physical error', 'nonlinear'),
              ('future/0.75ps/block_mean', 'Future physical error, 0.75 ps', 'nonlinear'),
              ('future/3ps/block_mean', 'Future physical error, 3 ps', 'nonlinear'),
              ('future/9ps/block_mean', 'Future physical error, 9 ps', 'nonlinear'),
              ('temporal/0.75ps/normalized_rms_jump', 'Normalized jump J, 0.75 ps', 'joint'),
              ('future/9ps/block_mean', 'Linear future readout, 9 ps', 'linear')]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for ax, (metric, title, readout) in zip(axes.flat, panels, strict=True):
        for index, variant in enumerate(variants):
            row, = [r for r in summary if (r['variant'], r['readout'], r['population'], r['method'], r['metric'])
                    == (variant, readout, 'low_order', 'encoder', metric)]
            error = None if row['ci95_low'] is None else [[row['value']-row['ci95_low']], [row['ci95_high']-row['value']]]
            ax.errorbar(index, row['value'], yerr=error,
                        fmt='o', capsize=4, color='#276a8d')
        if metric.startswith('future/'):
            baseline = next(r for r in summary if r['readout'] == readout and r['population'] == 'low_order'
                            and r['method'] == 'persistence' and r['metric'] == metric)
            ax.axhline(baseline['value'], color='#b65c24', linestyle='--', label='Measured-state persistence')
            ax.legend(fontsize=8)
        if metric.startswith('temporal/'):
            ax.axhline(.10, color='#b65c24', linestyle='--', label='Declared criterion')
            ax.legend(fontsize=8)
        ax.set_xticks(range(len(variants)), [v.replace('repeated_anchor', 'repeat') for v in variants])
        ax.set_title(title); ax.set_ylabel('J' if metric.startswith('temporal/') else 'Standardized block-mean MSE')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Causal MACE: held-out low-order environments\nSeed means; 95% whole-source bootstrap intervals')
    fig.savefig(root/'plots/physical-comparison.png', dpi=180)
    fig.savefig(root/'plots/physical-comparison.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(load_json(parser.parse_args().config))
