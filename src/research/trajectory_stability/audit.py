"""Supplement completed exports with dimensions and stability; no retraining."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from .spectrum import analyze


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Changed input {path}: expected {expected}, got {actual}')
    return actual


def corpus_dynamics(z, corpus):
    """The structural-state producer verifies frames against a .75 ps timeline."""
    records = corpus.records
    source = np.array([r['root'] for r in records])
    atoms = np.array([r['center_atom_id'] for r in records])
    times = np.array([r['frame'] for r in records])*.75
    temperature = np.array([r['temperature_K'] for r in records])
    phase, risk = corpus.targets['phase'], corpus.targets['at_risk']
    # Here phase is the producer's Boolean crystalline indicator, not PTM IDs.
    if not np.isin(phase, [0, 1]).all():
        raise ValueError('Structural-state phase must be the binary crystal indicator')
    domains = dict(noncrystalline=phase == 0,
                   pre_onset_12ps=risk & (corpus.targets['event_bin'] < 5),
                   at_risk_no_onset_12ps=risk & (corpus.targets['event_bin'] == 5))
    domains.update({f'T{t:g}K': temperature == t for t in np.unique(temperature)})
    return analyze(z, source, atoms, times, corpus.split['fit'], corpus.split['development'],
                   lags_ps=[108., 120.], domains=domains)


def native(config):
    cache = resolve_path(config['cache'])
    manifest = json.loads((cache/'manifest.json').read_text())
    # No graph/position files are read or needed in this analysis-only adapter.
    evidence = {str(cache/'manifest.json'): sha(cache/'manifest.json')}
    for name in ('records.json', 'targets.npz'):
        evidence[str(cache/name)] = checked(cache/name, manifest['files'][name])
    records = json.loads((cache/'records.json').read_text())
    from src.research.structural_state.data import splits
    with np.load(cache/'targets.npz') as data:
        corpus = SimpleNamespace(records=records, split=splits(records), targets=dict(data))
    for folder_name in config['evaluations']:
        folder = resolve_path(folder_name)
        receipt = json.loads((folder/'complete.json').read_text())
        if receipt['future_identity'] != manifest['identity']:
            raise ValueError(f'Future assay identity mismatch: {folder}')
        path = folder/'embeddings/future.npz'
        proof = dict(evidence)
        proof[str(folder/'complete.json')] = sha(folder/'complete.json')
        proof[str(path)] = checked(path, receipt['feature_files']['future.npz'])
        with np.load(path) as features:
            for rep in receipt['extraction']['representations']:
                name = folder.name+'--'+rep
                yield name, corpus_dynamics(features[rep], corpus), dict(
                    protocol='four_snapshot_structural_screen', representation=rep,
                    input_hashes=proof, checkpoint_sha256=receipt['checkpoint_sha256'],
                    sampling='4 frames per atom; 108/120 ps intervals. Individual state rank <=3.')


def dense(config):
    root = resolve_path(config['root'])
    plan = json.loads((root/'technical/plan.json').read_text())
    evidence = {str(root/'technical/plan.json'): sha(root/'technical/plan.json')}
    methods = {name: [] for name in config['methods']}
    extra, extra_values, extra_evidence = [], {}, {}
    for entry in config.get('extra_exports', []):
        folder = resolve_path(entry['root'])
        receipt = json.loads((folder/'complete.json').read_text())
        if receipt['source_plan_sha256'] != evidence[str(root/'technical/plan.json')]:
            raise ValueError(f'Dense export source plan differs: {folder}')
        checked(folder/'task.json', receipt['task_sha256'])
        for name, expected in receipt['feature_files'].items():
            checked(folder/'embeddings'/name, expected)
        extra.append((entry, receipt))
        for rep in receipt['extraction']['representations']:
            key = 'dense-current--'+entry['name']+'--'+rep
            extra_values[key] = []
            extra_evidence[key] = dict(checkpoint_sha256=receipt['task']['checkpoint_sha256'],
                representation=rep, native_task=receipt['task'],
                extraction=receipt['extraction'], input_hashes={str(folder/'complete.json'): sha(folder/'complete.json'),
                    **{str(folder/'embeddings'/name): h for name, h in receipt['feature_files'].items()}})
    sources, atoms, times, phase, splits, temperatures = [], [], [], [], [], []
    from .report import features
    for i, record in enumerate(plan['sources']):
        folder = root/'technical/sources'/str(record['id'])
        receipt = json.loads((folder/'complete.json').read_text())
        path = folder/'observations.npz'
        evidence[str(path)] = checked(path, receipt['hashes']['observations.npz'])
        for name in config['methods']:
            if name in ('mace', 'gatr'):
                verification = json.loads((folder/f'{name}-verification.json').read_text())
                evidence[str(folder/f'{name}.npy')] = checked(folder/f'{name}.npy', verification['features_sha256'])
        with np.load(path) as a:
            f = features(folder, a)
            count = len(a['times_ps'])*len(a['centers'])
            for name in methods:
                if len(f[name]) != count:
                    raise ValueError(f'Dense feature rows do not match frame/atom product: {folder}/{name}')
                methods[name].append(f[name])
            for entry, receipt in extra:
                path = resolve_path(entry['root'])/'embeddings'/f'frame-{i:02d}.npz'
                with np.load(path) as exported:
                    for rep in receipt['extraction']['representations']:
                        values = exported[rep]
                        if len(values) != count:
                            raise ValueError(f'Wrong dense source row count: {path}/{rep}')
                        extra_values['dense-current--'+entry['name']+'--'+rep].append(values)
            sources.extend([record['id']]*count)
            atoms.extend(np.tile(a['centers'], len(a['times_ps'])))
            times.extend(np.repeat(a['times_ps'], len(a['centers'])))
            phase.extend(a['labels'].ravel())
            splits.extend([record['split']]*count)
            temperatures.extend([record['temperature_K']]*count)
    source, atom, time, phase, split, temperature = map(np.asarray, (sources, atoms, times, phase, splits, temperatures))
    domains = {'noncrystalline': ~np.isin(phase, [1, 2, 3])}
    domains.update({f'T{t:g}K': temperature == t for t in np.unique(temperature)})
    for method, parts in methods.items():
        yield 'dense-v6--'+method, analyze(np.concatenate(parts), source, atom, time,
            np.flatnonzero(split == 'train'), np.flatnonzero(split == 'test'),
            lags_ps=config['lags_ps'], domains=domains), dict(
                protocol=plan['protocol'], representation=method, input_hashes=evidence,
                sampling='0.75 ps test trajectories; historical v6 models, not current MACE checkpoints.')
    for name, parts in extra_values.items():
        yield name, analyze(np.concatenate(parts), source, atom, time,
            np.flatnonzero(split == 'train'), np.flatnonzero(split == 'test'),
            lags_ps=config['lags_ps'], domains=domains), dict(
                protocol=plan['protocol'], **extra_evidence[name], source_plan_sha256=evidence[str(root/'technical/plan.json')],
                sampling='Matched 0.75 ps observed MD; no relaxation or time interpolation. Historical reference/evaluation source split; descriptive transfer assay.')


def _csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def label(model):
    old = dict(mace='MACE v6', gatr='GATr v6', tda='TDA', soap='SOAP',
               bond_order='Bond order', radial='Radial', angular='Angular')
    if model.startswith('dense-v6--'):
        return old[model.removeprefix('dense-v6--')]
    name, representation = model.removeprefix('dense-current--').split('--')
    gf = re.fullmatch(r'gf-(mlp-cov1-factor1|visreg-mlp-factor1)-s(\d+)-0035', name)
    mace = re.fullmatch(r'mace-lr(1e-05|0.0001)-distance0-s(\d+)-4096', name)
    if gf:
        return f'Geoformer {"VISReg" if gf[1].startswith("visreg") else "VICReg"} s{gf[2]} ({representation})'
    if mace:
        return f'MACE LR {"1e-5" if mace[1] == "1e-05" else "1e-4"} s{mace[2]}'
    return model


def export(config):
    root = resolve_path(config['output'])
    if root.exists():
        raise FileExistsError(f'Use a fresh output to preserve metric definitions: {root}')
    for part in ('tables', 'plots', 'technical'):
        (root/part).mkdir(parents=True, exist_ok=True)
    (root/'technical/config.json').write_text(json.dumps(config, indent=2)+'\n')
    summary, ranks, tracks, eigenvalues = [], [], [], []
    with threadpool_limits(limits=1):
        for adapter, spec in [(native, config['native']), (dense, config['dense'])]:
            for name, result, evidence in adapter(spec):
                (root/'technical'/f'{name}.json').write_text(json.dumps(dict(metrics=result, evidence=evidence), indent=2, allow_nan=False)+'\n')
                for domain, block in result['domains'].items():
                    for population in ('dataset', 'reference', 'evaluation', 'within_track'):
                        spectrum = block[population]['spectrum']
                        if spectrum is None:
                            continue
                        ranks.append(dict(model=name, domain=domain, population=population,
                            **{k: v for k, v in spectrum.items() if k != 'eigenvalues'}))
                        eigenvalues.extend(dict(model=name, domain=domain, population=population, lag_ps=None,
                            component=i+1, eigenvalue=v) for i, v in enumerate(spectrum['eigenvalues']))
                    for lag, values in block['lags'].items():
                        row = dict(model=name, domain=domain, lag_ps=float(lag), pairs=values['pairs'], sources=values['sources'])
                        for key in ('rms_jump', 'domain_reference_rms_jump', 'p50_jump', 'p95_jump', 'p99_jump',
                                    'zero_increment_fraction', 'velocity_rms', 'drift_energy_fraction'):
                            row[key] = values[key] if values['pairs'] else None
                        for kind in ('movement', 'fluctuation'):
                            for key in ('participation_rank', 'entropy_rank', 'd90', 'd95', 'd99', 'numerical_rank', 'rank_ceiling'):
                                row[kind+'_'+key] = values[kind][key] if values['pairs'] else None
                            if values['pairs']:
                                eigenvalues.extend(dict(model=name, domain=domain, population=kind, lag_ps=float(lag),
                                    component=i+1, eigenvalue=v) for i, v in enumerate(values[kind]['eigenvalues']))
                        summary.append(row)
                for track in result['per_track']:
                    row = dict(model=name, **{k: v for k, v in track.items() if k not in ('state', 'movement', 'fluctuation', 'adjacent_lags_ps')})
                    row['adjacent_lags_ps'] = json.dumps(track['adjacent_lags_ps'])
                    for kind in ('state', 'movement', 'fluctuation'):
                        for key in ('participation_rank', 'entropy_rank', 'd95', 'numerical_rank', 'rank_ceiling'):
                            row[kind+'_'+key] = track[kind][key] if track[kind] else None
                    tracks.append(row)
                print('analyzed', name, flush=True)
    snapshot_metric_docs(root, 'embedding_dynamics')
    for filename, rows in [('stability', summary), ('ranks', ranks), ('per-track', tracks), ('eigenvalues', eigenvalues)]:
        _csv(root/'tables'/f'{filename}.csv', rows)
    comparison = []
    for row in summary:
        if row['domain'] != 'all' or not row['pairs']:
            continue
        def rank(domain, population):
            return next(r['participation_rank'] for r in ranks if r['model'] == row['model']
                        and r['domain'] == domain and r['population'] == population)
        comparison.append(dict(model=row['model'], label=label(row['model']), lag_ps=row['lag_ps'],
            dataset_rank=rank('all', 'dataset'), noncrystalline_dataset_rank=rank('noncrystalline', 'dataset'),
            evaluation_rank=rank('all', 'evaluation'), movement_rank=row['movement_participation_rank'],
            movement_d95=row['movement_d95'], rms_jump=row['rms_jump'],
            pairs=row['pairs'], sources=row['sources']))
    _csv(root/'tables/comparison.csv', comparison)
    plot(root, summary, ranks)
    if {r['lag_ps'] for r in comparison} == {.75}:
        write_lag075_report(root, comparison)
    (root/'technical/complete.json').write_text(json.dumps(dict(state='complete', models=len({r['model'] for r in ranks}),
        config_sha256=sha(root/'technical/config.json'), files={p.name: sha(p) for p in (root/'tables').iterdir()}), indent=2)+'\n')
    return root


def write_lag075_report(root, comparison):
    lines = ['# Matched embedding dynamics at 0.75 ps', '',
        '**Every temporal measurement in this report uses a lag of 0.75 ps.**', '',
        'Geoformer VICReg/VISReg and current MACE have been freshly exported on the same',
        'observed MD coordinates as the historical MACE/GATr and descriptor baselines.',
        'No new simulation, model training, relaxation or temporal interpolation was used.', '',
        'The common evaluation contains 40 tracked atoms across 10 Al sources, 801 frames',
        'per track, 32,040 states and 32,000 adjacent pairs. Five separate reference sources',
        'supply 420 observations for normalization. Dataset ranks use all 32,460 exported',
        'rows with equal total weight per source. Movement ranks and jumps use evaluation',
        'pairs only. The inherited reference split describes this assay, not every model\'s',
        'pretraining population.', '',
        '**Input qualification:** recent MACE checkpoints were trained on relaxed',
        'coordinates; here they receive observed MD. These are observed-input transfer',
        'measurements, not measurements along a relaxed trajectory or a rerun of onset AP.',
        'All models share source/atom/frame identities, but retain their native spatial',
        'support and numerical precision. Earlier 108/120 ps values are not mixed into',
        'this table. This is a descriptive comparison, without source confidence intervals.', '',
        '## How to read the dimensions', '',
        '**Movement d95** is the smallest number of principal directions retaining at',
        'least 95% of the source-weighted squared changes z(t+0.75)-z(t). For example,',
        'd95 = 5 means a five-dimensional linear subspace captures at least 95% of the',
        'movement energy. Each direction may combine all original embedding channels.',
        'The second moment is uncentered, so steady drift remains included.', '',
        'Dataset and movement ranks below are **participation ratios**: continuous',
        'effective dimensions describing energy concentration. They need not equal the',
        'integer d95 and are not nonlinear intrinsic manifold dimensions. Smaller jumps',
        'or ranks alone do not establish a more useful encoder.', '']
    for projector, heading in [(False, 'Encoder states and physical descriptors'), (True, 'Geoformer projectors (separate exports)')]:
        lines += ['## '+heading, '',
            '| Representation | Dataset rank | Noncrystalline rank | Movement rank | Movement d95 | RMS jump |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
        for r in comparison:
            if ('projector' in r['model']) != projector:
                continue
            lines.append(f'| {r["label"]} | {r["dataset_rank"]:.3f} | {r["noncrystalline_dataset_rank"]:.3f} | '
                         f'{r["movement_rank"]:.3f} | {r["movement_d95"]} | {r["rms_jump"]:.3f} |')
        lines.append('')
    lines += ['RMS jump is normalized by each model\'s own reference-population pair-distance',
        'scale. Noncrystalline means PTM outside FCC/HCP/BCC, not a verified equilibrium',
        'liquid label. Definitions, input identities and complete spectra are preserved below.', '',
        '- [Readable comparison CSV](tables/comparison.csv)',
        '- [All population/temperature stability measurements](tables/stability.csv)',
        '- [Dataset, reference, evaluation and within-track ranks](tables/ranks.csv)',
        '- [Individual atom trajectories](tables/per-track.csv)',
        '- [Full eigenvalue spectra](tables/eigenvalues.csv)',
        '- [Frozen metric definitions](tables/METRICS.md)',
        '- [Implementation hashes](technical/metric-contract.json)',
        '- Model-specific full metrics, checkpoint identities and native extraction checks are in `technical/`.', '',
        '![Matched 0.75 ps comparison](plots/dense.png)', '']
    (root/'RESULTS.md').write_text('\n'.join(lines))


def plot(root, summary, ranks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for prefix, caption in [('', 'Current exports: coarse-time measurements'), ('dense-', 'Matched dense trajectories')]:
        selected = [r for r in summary if r['domain'] == 'all' and r['pairs'] and (r['model'].startswith('dense-') == bool(prefix))]
        models = sorted({r['model'] for r in selected})
        if not models:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, max(4, len(models)*.35)), constrained_layout=True)
        single_lag = len({r['lag_ps'] for r in selected}) == 1
        for i, name in enumerate(models):
            points = sorted((r for r in selected if r['model'] == name), key=lambda r:r['lag_ps'])
            global_rank = next(r['participation_rank'] for r in ranks if r['model'] == name and r['domain'] == 'all' and r['population'] == 'evaluation')
            axes[0].scatter([global_rank], [i], marker='o', color='#2066a8')
            axes[0].scatter([points[0]['movement_participation_rank']], [i], marker='x', color='#d55e00')
            if single_lag:
                axes[1].barh(i, points[0]['rms_jump'], color='#2066a8')
            else:
                axes[1].plot([r['lag_ps'] for r in points], [r['rms_jump'] for r in points], 'o-', label=label(name))
        axes[0].set_yticks(range(len(models)), [label(m) for m in models], fontsize=7)
        axes[0].set_xlabel('Participation rank: evaluation state (blue), movement (orange)')
        if single_lag:
            axes[1].set(xlabel=f'Reference-normalized RMS jump at {selected[0]["lag_ps"]:g} ps', yticks=[])
        else:
            axes[1].set(xlabel='Physical lag (ps)', ylabel='Fit-reference normalized RMS jump', ylim=(0, None))
            axes[1].legend(fontsize=6)
        fig.suptitle(caption+' — no temporal smoothing')
        fig.savefig(root/'plots'/('dense.png' if prefix else 'coarse.png'), dpi=160)
        plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    export(json.loads(Path(args.config).read_text()))
