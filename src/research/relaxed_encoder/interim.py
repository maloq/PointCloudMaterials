"""Evaluate completed checkpoints on an immutable, already prepared assay cohort."""
import argparse
import json
from pathlib import Path

from src.data.structural_pretraining.prepare import digest, file_hash, save_json
from src.experiment_runner.metric_docs import write_metric_table
from src.project_runtime.paths import resolve_path
from . import assay
from .report import compare


CONTROLS = ('parent_hot', 'parent_cold', 'geometry_hot', 'geometry_cold',
            'original_geometry', 'conditions')


def prepare(config):
    root = resolve_path(config['output']).resolve()
    source = resolve_path(config['training_output']).resolve()
    cohort = resolve_path(config['assay_output']).resolve()
    trained = json.loads((source/'technical/plan.json').read_text())
    observed = json.loads((cohort/'technical/plan.json').read_text())
    ready = json.loads((cohort/'technical/assay/ready.json').read_text())
    for key in ('scale', 'assay_plan', 'population', 'normalization_manifest', 'warm_checkpoint'):
        if trained['config'][key] != observed['config'][key]:
            raise ValueError(f'Training and existing assay disagree: {key}')
    old_sources = {s['id']: s for s in observed['sources']}
    for s in trained['sources']:
        old = old_sources[s['id']]
        for key in ('lineage', 'source_manifest_sha256', 'center_atom_ids'):
            if s[key] != old[key]:
                raise ValueError(f'Assay source changed: {s["id"]}, {key}')
        if s.get('validation_role', s['split']) != old.get('validation_role', old['split']):
            raise ValueError(f'Source split changed: {s["id"]}')
    runs = {r['name']: r for r in trained['config']['runs']}
    checkpoints = {}
    for name in config['runs']:
        status = json.loads((source/f'technical/runs/{name}/status.json').read_text())
        if status['state'] != 'complete':
            raise ValueError(f'Encoder training is not complete: {name}')
        checkpoints[name] = file_hash(source/f'technical/runs/{name}/best.pt')
    c = dict(trained['config'], output=str(root), cache=observed['config']['cache'],
             frames=ready['frames'], runs=[runs[n] for n in config['runs']])
    identity = dict(config=config, checkpoints=checkpoints,
                    training_identity=trained['identity'], assay_identity=ready['identity'],
                    population_sha256=file_hash(cohort/'technical/assay/population.npz'))
    plan = dict(config=c, sources=trained['sources'], identity=digest(identity),
                interim=identity, assay_counts=ready)
    path = root/'technical/plan.json'
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError('Interim evaluation identity changed')
        return plan
    (root/'technical/assay').mkdir(parents=True, exist_ok=True)
    (root/'technical/runs').mkdir(exist_ok=True)
    for name in config['runs']:
        (root/f'technical/runs/{name}').symlink_to(source/f'technical/runs/{name}', target_is_directory=True)
    for name in ('population.npz', 'hot-descriptors.npy', 'cold-descriptors.npy',
                 'hot-plan.json', 'cold-plan.json', 'ready.json'):
        (root/'technical/assay'/name).symlink_to(cohort/'technical/assay'/name)
    for name in ('parent_hot', 'parent_cold'):
        previous = cohort/'technical/assay'/name
        receipt = json.loads((previous/'complete.json').read_text())
        record = json.loads((previous/'record.json').read_text())
        if receipt['checkpoint_sha256'] != file_hash(resolve_path(c['warm_checkpoint'])):
            raise ValueError('Cached parent encoder checkpoint changed')
        if record['population_sha256'] != identity['population_sha256']:
            raise ValueError('Cached parent features belong to another population')
        if receipt['feature_sha256'] != file_hash(previous/'features.npy'):
            raise ValueError('Cached parent features changed')
        dest = root/'technical/assay'/name
        dest.mkdir()
        (dest/'features.npy').symlink_to(previous/'features.npy')
        save_json(dest/'complete.json', dict(receipt, reused_from=str(previous)))
    save_json(path, plan)
    return plan


def report(plan):
    root = resolve_path(plan['config']['output'])
    fits = {}
    for p in (root/'readouts/technical/fits').glob('*/snapshot/*/metrics.json'):
        m = json.loads(p.read_text())
        fits[m['task']['encoder'], m['task']['readout']] = m
    ready = plan['assay_counts']
    lines = ['# Preliminary crystallization readouts of expanded relaxed encoders', '',
             f'Fixed earlier assay cohort: {ready["counts"]["test"]} test windows, '
             f'{ready["events"]["test"]} positive by 12 ps; observation frames {ready["frames"]}. '
             'The cohort was fixed before these runs and is not selected by relaxation completion speed. '
             'This is separate from the larger pending 15-origin evaluation. One seed; '
             'use these sparse-event results as diagnostics, not hyperparameter selection.', '',
             'Each encoder is frozen. Matched linear and MLP hazard readouts use original source '
             'splits and original MD onset labels; thresholds are calibrated at 5% false-positive rate. '
             'Timing MAE includes detected event windows only; misses must be considered alongside it.', '',
             '| Encoder | Readout | Event NLL | 12 ps AP | AUROC | Timing MAE (ps) | Misses / events |',
             '|---|---|---:|---:|---:|---:|---:|']
    gains = {}
    def fmt(v):
        return 'undefined' if v is None else f'{v:.4f}'
    for (name, kind), m in sorted(fits.items()):
        cl, t = m['classification']['12.0'], m['timing']['12.0']
        lines.append(f'| {name} | {kind} | {fmt(m["event_nll"])} | '
                     f'{fmt(cl["average_precision"])} | {fmt(cl["auroc"])} | '
                     f'{fmt(t["detected_timing_mae_ps"])} | {t["missed_windows"]}/{t["event_windows"]} |')
        if ('hot-control', kind) in fits:
            gains[name+'--'+kind] = compare(fits['hot-control', kind], m)
    expected = 2*(len(plan['config']['runs'])+len(CONTROLS))
    lines += ['', f'Completed readouts: {len(fits)}/{expected}. '
              'Full horizon metrics (0.75, 3, 6, 9, 12 ps) are in readouts/tables/. '
              'Source-bootstrap NLL gains against hot-control are in comparison/tables/.']
    write_metric_table(gains, root/'comparison', family='relaxed_encoder', name='source-paired-gains')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'extract', 'probe', 'report'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    plan = prepare(config)
    root = resolve_path(config['output'])
    if args.stage == 'extract':
        for name in config['runs']:
            if not (root/f'technical/assay/{name}/complete.json').exists():
                assay.extract(plan, name)
    elif args.stage == 'probe':
        for name in (*config['runs'], *CONTROLS):
            assay.probes(plan, name)
            report(plan)
    elif args.stage == 'report':
        report(plan)


if __name__ == '__main__':
    main()
