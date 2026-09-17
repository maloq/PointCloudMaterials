"""Freeze a cross-study evidence snapshot without pooling different protocols."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import load_json, resolve_path
from src.research.mace_causal_comparison import source_values


def numeric(row):
    result = dict(row)
    for key in ('value', 'ci95_low', 'ci95_high'):
        if key in result:
            result[key] = float(result[key]) if result[key] != '' else None
    for key in ('seeds', 'sources'):
        if key in result:
            result[key] = int(result[key])
    return result


def cross_seed_mean(rows, model, metric, expected_seeds):
    chosen = [r for r in rows if r['model'] == model]
    if sorted(r['seed'] for r in chosen) != sorted(expected_seeds):
        raise ValueError(f'Cannot average an incomplete/duplicated seed cohort: {model}')
    return float(np.mean([r[metric] for r in chosen]))


def write_table(root, name, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (root/'tables'/f'{name}.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def collect(config, output):
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    for folder in ('technical', 'tables', 'plots'):
        (root/folder).mkdir()
    provenance = {}

    def read(path, kind='json'):
        path = Path(path)
        data = path.read_bytes()
        provenance[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return json.loads(data) if kind == 'json' else list(csv.DictReader(io.StringIO(data.decode())))

    snapshot = dict(causal={}, gaussian=[], memory=[], states=[], pairs=[], diagnostics=[], simulations=[])
    snapshot['h200'] = read(config['h200_report'])
    wanted = {'present/block_mean', 'future/0.75ps/block_mean', 'future/3ps/block_mean',
              'future/9ps/block_mean', 'temporal/0.75ps/normalized_rms_jump'}
    for cohort, directory in config['causal_comparisons'].items():
        directory = Path(directory)
        if read(directory/'technical/status.json')['state'] != 'complete':
            raise ValueError(f'Historical comparison incomplete: {directory}')
        snapshot['causal'][cohort] = {}
        for file in ('summary', 'paired-differences', 'sufficiency'):
            rows = read(directory/'tables'/f'{file}.csv', 'csv')
            selected = [numeric(r) for r in rows if r['metric'] in wanted]
            snapshot['causal'][cohort][file] = selected
            write_table(root, f'causal-{cohort}-{file}', selected)
        read(directory/'technical/metric-contract.json')

    for seed in config['gaussian_seeds']:
        for head in ('pilot', 'pilot-gaussian'):
            directory = resolve_path(f'output/mace_causal/{head}-seed{seed}-D')
            if read(directory/'technical/status.json')['state'] != 'complete':
                raise ValueError(f'Gaussian control is incomplete: {directory}')
            rows = read(directory/'tables/test-sources.csv', 'csv')
            values = source_values(rows)
            for population, metric in [('low_order', 'future/9ps/block_mean'),
                                       ('all', 'future/9ps/block_mean'),
                                       ('all', 'future/9ps/nll'), ('all', 'future/9ps/coverage_1sigma')]:
                selected = {source: value for (p, method, m, source), value in values.items()
                            if p == population and method == 'encoder' and m == metric}
                if not selected:
                    if head == 'pilot' and metric.endswith(('nll', 'coverage_1sigma')):
                        continue  # A deterministic head defines neither score.
                    raise ValueError(f'Missing Gaussian/control score: {directory}, {population}, {metric}')
                snapshot['gaussian'].append(dict(seed=seed, head=head, population=population,
                    metric=metric, sources=len(selected), value=float(np.mean(list(selected.values())))))
            read(directory/'technical/metric-contract.json')
    write_table(root, 'causal-gaussian-ablation', snapshot['gaussian'])

    for cohort, recipe in config['memory_recipes'].items():
        read(recipe)
        fit = load_json(recipe)
        directory = Path(fit['output'])
        modalities = ('x', 'xv') if cohort == 'pilot_seed17' else ('xv',)
        names = [f'{m}-H{h}' for m in modalities for h in (0, 12, 48)] + [f'{m}-H48-repeat' for m in modalities]
        for name in names:
            path = directory/name/'technical'
            status = read(path/'status.json') if (path/'status.json').exists() else dict(state='not_started')
            snapshot['states'].append(dict(cohort=cohort, model=name, **status))
            if status['state'] != 'complete':
                continue
            metrics = read(path/'metrics.json')
            if status['step'] != fit['training']['steps'] or metrics['trained_steps'] != status['step']:
                raise ValueError(f'Completed fit has a different budget: {path}')
            row = dict(cohort=cohort, model=name, seed=fit['seed'], width=fit['encoder']['channels'],
                       updates=status['step'], selected_update=metrics['selected_step'],
                       present_weight=fit['training']['present_weight'])
            for split in ('val', 'test'):
                for key, label in [('joint_nll', 'nll'), ('future_mse', 'future_mse'), ('present_mse', 'present_mse')]:
                    row[f'{split}_{label}'] = metrics[split][key]['mean']
            snapshot['memory'].append(row)
            read(path/'metric-contract.json')
        for component in ('comparison', 'diagnostics'):
            path = directory/component/'technical'
            if not (path/'status.json').exists():
                continue
            if read(path/'status.json')['state'] != 'complete':
                raise ValueError(f'Collector exists but is incomplete: {path}')
            metrics = read(path/'metrics.json')
            read(path/'metric-contract.json')
            if component == 'comparison':
                for name, record in metrics['paired_test'].items():
                    snapshot['pairs'].append(dict(cohort=cohort, comparison=name, gain=record['mean'],
                        ci95_low=record['ci95'][0], ci95_high=record['ci95'][1], sources=record['sources']))
            else:
                for name, record in metrics['models'].items():
                    row = dict(cohort=cohort, model=name)
                    for split in ('val', 'test'):
                        row[split+'_constant_state_nll_increase'] = record['interventions'][split]['mean_state_joint_nll_increase']['mean']
                        row[split+'_embedding_ridge_future_mse'] = record['embedding_future_ridge'][split]['future_mse']['mean']
                        row[split+'_embedding_ridge_present_mse'] = record['embedding_present_ridge'][split]['present_mse']['mean']
                    snapshot['diagnostics'].append(row)
                if cohort == 'pilot_seed17':
                    snapshot['physical_baselines'] = metrics['baselines']
    for key, name in [('memory', 'memory-fits'), ('states', 'fit-progress'), ('pairs', 'memory-paired-gains'), ('diagnostics', 'state-use')]:
        write_table(root, name, snapshot[key])

    reference = [r for r in snapshot['memory'] if r['cohort'] in ('pilot_seed17', 'pilot_seed18')]
    width_rows = []
    reported = snapshot['h200']['partial_observation']
    for model, wider in reported['width32_h200'].items():
        local = cross_seed_mean(reference, model, 'test_nll', [20260917, 20260918])
        if abs(local-reported['width16_h100_reported'][model]) > .00005:
            raise ValueError(f'H200 reference summary disagrees with local two-seed mean: {model}')
        width_rows.append(dict(model=model, seeds=2, updates=3000, width16_local_nll=local,
            width32_user_reported_nll=wider, width32_minus_width16=wider-local,
            width32_evidence='user-reported rounded mean; raw H200 artifacts unavailable'))
    snapshot['width_comparison'] = width_rows
    write_table(root, 'memory-width-comparison', width_rows)
    h = snapshot['h200']['causal_state']
    write_table(root, 'causal-h200-reported', [dict(model=m, width16=h['width16'][m], width32=h['width32'][m],
        seeds=3, sources=17, evidence='user-reported rounded means; interval endpoints unavailable') for m in h['width16']])

    simulation_root = Path(config['simulation_root'])
    manifest = read(simulation_root/'manifest.json')
    for record in manifest['runs']:
        path = simulation_root/record['run_dir']
        status = read(path/'status.json') if (path/'status.json').exists() else dict(state='not_started')
        row = dict(run_id=record['run_id'], split=record['split'], temperature_K=record['temperature_K'],
                   state=status['state'], published=False)
        if status['state'] == 'complete' and path.is_symlink():
            receipt = read(path.resolve().with_name(path.name+'.publication.json'))
            row['published'] = receipt['state'] == 'complete'
            # Numerical storage QC is allowed; sealed physical targets are never opened.
            if record['split'] != 'sealed_test':
                converted = read(path/'paired_conversion.json')
                row.update(frame_count=converted['frame_count'],
                    position_rms_A=converted['quantization']['positions']['rms'],
                    position_max_A=converted['quantization']['positions']['max_abs'])
        snapshot['simulations'].append(row)
    write_table(root, 'data-production-status', snapshot['simulations'])
    snapshot['captured_at_utc'] = datetime.now(timezone.utc).isoformat()
    snapshot['provenance'] = provenance
    (root/'technical/evidence.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    (root/'technical/inputs.json').write_text(json.dumps(provenance, indent=2)+'\n')
    snapshot_metric_docs(root, 'memory_research_summary')
    plots(snapshot, root)
    (root/'README.md').write_text('# Native causal-state and predictive-memory evidence\n\n'
        f"Snapshot captured at {snapshot['captured_at_utc']}.\n\n"
        'The interpretation is in [RESULTS.md](RESULTS.md). Tables use separate protocol definitions; '
        'see [METRICS.md](tables/METRICS.md). H200 means are user-reported and explicitly marked. '
        'Source identities and captured data are retained under technical/.\n')
    return snapshot


def plots(snapshot, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(root/'plots'/f'{name}.{suffix}', dpi=180, bbox_inches='tight')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    h = snapshot['h200']['causal_state']
    labels = ['Snapshot', 'Real history', 'Repeated frame']
    for width, color in [(16, '#356a9a'), (32, '#c17c20')]:
        axes[0].plot(range(3), [h[f'width{width}'][m] for m in ('snapshot', 'history', 'repeated_anchor')],
                     'o-', color=color, label=f'Width {width}')
    axes[0].set(xticks=range(3), xticklabels=labels, ylabel='9 ps physical MSE',
                title='Older causal study · 3 seeds\n17 low-order test sources')
    for label, field, color in [('Width 16 · local', 'width16_local_nll', '#356a9a'),
                                ('Width 32 · reported', 'width32_user_reported_nll', '#c17c20')]:
        axes[1].plot(range(4), [r[field] for r in snapshot['width_comparison']], 'o-', color=color, label=label)
    axes[1].set(xticks=range(4), xticklabels=['H=0', 'H=12', 'H=48', 'Repeat'], ylabel='Joint physical-path NLL',
                title='New memory study · 2 seeds\n30 test sources · 3,000 updates')
    for ax in axes:
        ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.text(.5, -.02, 'Lower is better. H200 points are rounded reported means; raw intervals were not supplied. Different protocols: do not compare vertical scales.', ha='center', fontsize=8)
    save(fig, 'width-comparisons')

    models = ['xv-H0', 'xv-H12', 'xv-H48', 'xv-H48-repeat']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for cohort, label, color in [('original_seed17_12000', 'Present weight 0.05', '#356a9a'),
                                 ('present1_seed17_12000', 'Present weight 1.0', '#c17c20')]:
        rows = {r['model']: r for r in snapshot['memory'] if r['cohort'] == cohort}
        if set(rows) != set(models):
            raise ValueError('Optimization plot requires both completed first-seed quartets')
        for ax, metric, ylabel in zip(axes, ('test_nll', 'test_future_mse'), ('Joint path NLL', 'Future physical MSE'), strict=True):
            ax.plot(range(4), [rows[m][metric] for m in models], 'o-', label=label, color=color)
            ax.set(xticks=range(4), xticklabels=['H=0', 'H=12', 'H=48', 'Repeat'], ylabel=ylabel)
    axes[1].axhline(snapshot['physical_baselines']['current_packet_ridge']['test']['future_mse']['mean'],
                    color='#478251', linestyle='--', label='Current-packet ridge')
    for ax in axes:
        ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.suptitle('Information-retention objective · seed 20260917 · 12,000 updates', y=1.03)
    save(fig, 'present-loss-followup')

    fig, ax = plt.subplots(figsize=(7, 4))
    for cohort, label, offset, color in [('original_seed17_12000', 'Present weight 0.05', -.08, '#356a9a'),
                                        ('present1_seed17_12000', 'Present weight 1.0', .08, '#c17c20')]:
        rows = {r['model']: r for r in snapshot['diagnostics'] if r['cohort'] == cohort}
        ax.plot(np.arange(4)+offset, [rows[m]['val_constant_state_nll_increase'] for m in models],
                'o-', label=label, color=color)
    ax.axhline(0, color='gray', linewidth=1)
    ax.set(xticks=range(4), xticklabels=['H=0', 'H=12', 'H=48', 'Repeat'],
           ylabel='Validation NLL increase after removing state', title='Does the fitted predictor use the exported state?')
    ax.legend(); ax.grid(axis='y', alpha=.2)
    fig.text(.5, -.02, 'Positive: replacing each state by its training mean hurts. This is a frozen-head intervention, not a sufficiency test.', ha='center', fontsize=8)
    save(fig, 'state-use')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    collect(load_json(args.config), resolve_path(args.output))
