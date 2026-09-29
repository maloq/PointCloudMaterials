"""Measure native GeoFormer coordinates across physical crystal/liquid boundaries."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression

from src.experiment_runner.metric_docs import write_metric_table
from src.research.geoframe_continuity.analysis import load_model
from src.research.geoframe_evolution.evaluate import encode


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def checked(path, expected):
    observed = sha(path)
    if observed != expected:
        raise ValueError(f'Input identity changed: {path}: {observed} != {expected}')


def rho(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def spatial_reference(record, arrays, full, workers):
    points = np.load(record['file'])
    if not np.array_equal(points[arrays['rows']], arrays['coords']):
        raise ValueError(f'Atom identity/coordinate mismatch: {record["file"]}')
    tree = cKDTree(points)
    solid = np.isin(full['best_ptm'], [1, 2, 3]) & (full['rmsd'] <= 0.1)
    fraction = np.empty(len(points), dtype=np.float32)
    for first in range(0, len(points), 32768):
        rows = np.arange(first, min(first + 32768, len(points)))
        _, ids = tree.query(points[rows], k=15, workers=workers)
        if not np.array_equal(ids[:, 0], rows):
            raise ValueError('Duplicate full-cell positions invalidate neighbor identity.')
        fraction[rows] = solid[ids[:, 1:]].mean(1)
    margin = record['radius_A']
    safe = np.minimum(points - tree.mins, tree.maxes - points).min(1) > margin
    core = solid & (fraction >= 0.8) & safe
    liquid = ~solid & (fraction <= 0.1) & safe
    if min(core.sum(), liquid.sum()) < 100:
        raise ValueError(f'Insufficient bulk populations in frame {record["frame_index"]}')
    core_tree, liquid_tree = cKDTree(points[core]), cKDTree(points[liquid])

    def distance_coordinate(xyz):
        return (core_tree.query(xyz, workers=workers)[0]
                - liquid_tree.query(xyz, workers=workers)[0]) / 2

    n = record['anchor_count']
    d = distance_coordinate(arrays['coords'][:n])
    # Transects chosen using physical geometry only, never latent coordinates.
    anchor_rows = arrays['rows'][:n]
    distance, nearest_liquid = liquid_tree.query(points[anchor_rows], workers=workers)
    eligible = np.flatnonzero((arrays['split'][:n] == 1) & core[anchor_rows]
                             & (distance >= 12) & (distance <= 28))
    rng = np.random.default_rng(20260929 + record['frame_index'])
    transects, starts = [], []
    for j in rng.permutation(eligible):
        start = points[anchor_rows[j]].astype(float)
        if starts and np.linalg.norm(np.asarray(starts) - start, axis=1).min() < 25:
            continue
        end = liquid_tree.data[nearest_liquid[j]]
        unit = (end - start) / np.linalg.norm(end - start)
        requested_s = np.arange(0, np.linalg.norm(end - start) + 13, 1.0)
        line = start + requested_s[:, None] * unit
        if (np.minimum(line - tree.mins, tree.maxes - line).min() <= 2 * margin):
            continue
        perpendicular_distance, rows = tree.query(line, workers=workers)
        _, unique = np.unique(rows, return_index=True)
        rows = rows[np.sort(unique)]
        xyz = points[rows]
        actual_s = (xyz - start) @ unit
        order = np.argsort(actual_s)
        rows, xyz, actual_s = rows[order], xyz[order], actual_s[order]
        off_axis = np.linalg.norm(xyz - (start + actual_s[:, None] * unit), axis=1)
        distances, near = tree.query(xyz, k=80, workers=workers)
        if distances[:, -1].max() > margin:
            raise ValueError('A transect nearest-80 patch exceeds its declared radius.')
        if not np.array_equal(near[:, 0], rows):
            raise ValueError('Transect center identity mismatch.')
        clouds = (points[near] - points[rows, None]) / margin
        transects.append(dict(rows=rows, coords=xyz, s=actual_s, off_axis=off_axis,
                              distance=distance_coordinate(xyz), clouds=clouds,
                              solid=solid[rows], fraction=fraction[rows],
                              support_fraction=solid[near].mean(1)))
        starts.append(start)
        if len(transects) == 8:
            break
    _, near = tree.query(arrays['coords'][:n], k=80, workers=workers)
    return dict(distance=d, solid=solid[anchor_rows], fraction=fraction[anchor_rows],
                support_fraction=solid[near].mean(1), transects=transects,
                core_count=int(core.sum()), liquid_count=int(liquid.sum()))


def binned(x, y, edges):
    center, median, lower, upper = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        selected = (x >= lo) & (x < hi)
        if selected.sum() < 15:
            continue
        center.append((lo + hi) / 2)
        q = np.quantile(y[selected], [0.25, 0.5, 0.75])
        lower.append(q[0]); median.append(q[1]); upper.append(q[2])
    return np.asarray(center), np.asarray(median), np.asarray(lower), np.asarray(upper)


def run(config):
    import torch
    cfg = json.loads(Path(config).read_text())
    root = Path(cfg['output']); ref = Path(cfg['reference'])
    if (root / 'technical/summary.json').exists():
        raise FileExistsError(f'Completed analysis already exists: {root}')
    for part in ('technical', 'plots', 'tables', 'data'):
        (root / part).mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checked(cfg['checkpoint'], cfg['checkpoint_sha256'])
    checked(ref / 'manifest.json', cfg['reference_sha256'])
    manifest = json.loads((ref / 'manifest.json').read_text())
    model = load_model(cfg['checkpoint'])
    records = [r for r in manifest['frames'] if r['frame_index'] in cfg['frames']]
    if len(records) != len(cfg['frames']) or any(r['material'] != 'Al' for r in records):
        raise ValueError('This descriptive protocol requires the declared Al frames.')
    records = [dict(r, file=str(Path(cfg['repository_root']) / r['file'])) for r in records]
    frames = {}
    provenance = dict(config=cfg, job_id=os.environ.get('SLURM_JOB_ID'),
                      hostname=os.uname().nodename, files={}, purpose='descriptive; no fitting of encoder')
    for record in records:
        i = record['frame_index']
        checked(record['file'], record['input_sha256'])
        for filename in (f'frame-{i:02d}.npz', f'full-reference-{i:02d}.npz'):
            expected = (cfg['full_reference_files'][filename] if filename.startswith('full-')
                        else manifest['files'][filename])
            checked(ref / filename, expected)
            provenance['files'][str(ref / filename)] = expected
        with np.load(ref / f'frame-{i:02d}.npz') as f:
            arrays = {k: f[k] for k in f.files}
        with np.load(ref / f'full-reference-{i:02d}.npz') as f:
            full = {k: f[k] for k in f.files}
        field = spatial_reference(record, arrays, full, cfg['workers'])
        n = record['anchor_count']
        z = encode(model, arrays['clouds'][:n])
        # Compare the same batch shape. Different GEMM batch shapes need not
        # produce bitwise-identical floating-point results.
        repeat = encode(model, arrays['clouds'][:256])
        small = encode(model, arrays['clouds'][:64])
        same_error = float(np.max(np.abs(z[:256] - repeat)))
        shape_error = float(np.max(np.abs(z[:64] - small)))
        provenance.setdefault('repeatability', {})[str(i)] = dict(
            matched_batch_max_absolute_error=same_error,
            different_batch_max_absolute_error=shape_error)
        write(root / 'technical/repeatability.json', provenance['repeatability'])
        if not np.array_equal(z[:256], repeat):
            raise ValueError(f'Repeated identical-batch GeoFormer inference changed: {same_error}')
        if not np.allclose(z[:64], small, rtol=1e-4, atol=1e-5):
            raise ValueError(f'GeoFormer batch-shape sensitivity exceeds tolerance: {shape_error}')
        field.update(record=record, arrays=arrays, embeddings=z)
        for t, path in enumerate(field['transects']):
            path['embeddings'] = encode(model, path.pop('clouds'))
            np.savez_compressed(root / 'data' / f'frame-{i:02d}-transect-{t:02d}.npz', **path)
        np.savez_compressed(root / 'data' / f'frame-{i:02d}.npz', embeddings=z,
                            rows=arrays['rows'][:n], coords=arrays['coords'][:n],
                            split=arrays['split'][:n], distance=field['distance'],
                            solid=field['solid'], fraction=field['fraction'],
                            support_fraction=field['support_fraction'], order=arrays['order'][:n, 0])
        frames[i] = field
        print(f'Extracted {Path(record["file"]).stem}: {n} anchors, {len(field["transects"])} physical transects', flush=True)
    fit = frames[cfg['selection_frame']]
    fit_mask = fit['arrays']['split'][:fit['record']['anchor_count']] == 0
    metrics, selection, summaries = {}, {}, {}
    for rep_index, rep in enumerate(('encoder', 'projector')):
        zfit = fit['embeddings'][fit_mask, rep_index].astype(float)
        dfit = fit['distance'][fit_mask]
        mean, scale = zfit.mean(0), zfit.std(0)
        if (scale <= 1e-12).any():
            raise ValueError(f'Collapsed coordinate in {rep}; cannot standardize the declared full export.')
        train_rhos = np.asarray([rho(dfit, zfit[:, k]) for k in range(zfit.shape[1])])
        selected = np.argsort(-np.abs(train_rhos))[:6]
        sign = np.where(train_rhos >= 0, 1., -1.)
        crystal_fit = fit['solid'][fit_mask] & (fit['fraction'][fit_mask] >= .8)
        liquid_fit = ~fit['solid'][fit_mask] & (fit['fraction'][fit_mask] <= .1)
        cmean, lmean = zfit[crystal_fit].mean(0), zfit[liquid_fit].mean(0)
        axis = lmean - cmean
        if axis @ axis <= 1e-20:
            raise ValueError('Undefined bulk phase axis.')
        axis /= axis @ axis
        iso = [IsotonicRegression(increasing=bool(sign[k] > 0), out_of_bounds='clip').fit(dfit, zfit[:, k])
               for k in range(zfit.shape[1])]
        step = np.stack([zfit[~fit['solid'][fit_mask]].mean(0), zfit[fit['solid'][fit_mask]].mean(0)])
        selection[rep] = dict(coordinates_zero_based=selected.tolist(), sign=sign.tolist(),
                              mean=mean.tolist(), scale=scale.tolist(), crystal_mean=cmean.tolist(),
                              phase_axis=axis.tolist(), selection_frame=cfg['selection_frame'])
        for i, frame in frames.items():
            n = frame['record']['anchor_count']; test = frame['arrays']['split'][:n] == 1
            d = frame['distance'][test]; z = frame['embeddings'][test, rep_index].astype(float)
            phase = frame['solid'][test]; liquid = ~phase & (frame['fraction'][test] <= .1)
            clear = liquid & (frame['support_fraction'][test] == 0)
            prefix = f'frame_{i:02d}_{rep}'
            sm = dict(count=int(test.sum()), liquid_count=int(liquid.sum()),
                      strict_clear_count=int(clear.sum()), transect_count=len(frame['transects']),
                      phase_axis_distance_rho=rho(d, (z - cmean) @ axis),
                      selected_coordinates=selected.tolist())
            for k in range(z.shape[1]):
                predicted = iso[k].predict(d)
                mse = float(np.mean(((z[:, k] - predicted) / scale[k]) ** 2))
                step_mse = float(np.mean(((z[:, k] - step[phase.astype(int), k]) / scale[k]) ** 2))
                raw = dict(fit_rho=float(train_rhos[k]), test_rho=rho(d, z[:, k]),
                           liquid_rho=rho(d[liquid], z[liquid, k]),
                           strict_clear_rho=rho(d[clear], z[clear, k]),
                           isotonic_nmse=mse, binary_phase_nmse=step_mse,
                           isotonic_minus_binary_nmse=mse-step_mse)
                metrics[f'{prefix}_coordinate_{k:03d}'] = raw
                if k in selected[:3]:
                    sm[f'coordinate_{k:03d}'] = raw
            summaries[prefix] = sm
            fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
            edges = np.linspace(-25, 25, 26)
            for ax, k in zip(axes.flat, selected):
                y = sign[k] * (z[:, k] - mean[k]) / scale[k]
                take = np.random.default_rng(20260929).choice(len(d), min(500, len(d)), replace=False)
                ax.scatter(d[take], y[take], s=2, alpha=.16, color='#506b86')
                x, med, lo, hi = binned(d, y, edges)
                ax.fill_between(x, lo, hi, alpha=.22, color='#007c91', label='Middle 50%')
                ax.plot(x, med, color='#007c91', marker='.', label='Bin median')
                ax.axvline(0, color='gray', lw=.6)
                ax.set(xlim=(-25, 25), xlabel='Crystal → liquid distance coordinate (Å)',
                       ylabel='Signed coordinate / fit-region SD',
                       title=f'Coordinate {k} | held-region ρ={rho(d, y):.2f}')
            axes[0, 0].legend(fontsize=8)
            fig.suptitle(f'Archived GeoFormer epoch 34 · {rep} · {Path(frame["record"]["file"]).stem}\n'
                         'Coordinates chosen on the 174 ps fitting half; points/bands show real scatter, not uncertainty of the mean')
            for suffix in ('png', 'pdf'):
                fig.savefig(root / 'plots' / f'{prefix}-coordinate-profiles.{suffix}', dpi=160)
            plt.close(fig)
            if frame['transects']:
                fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
                for t, (ax, path) in enumerate(zip(axes.flat, frame['transects'])):
                    for k in selected[:3]:
                        y = sign[k] * (path['embeddings'][:, rep_index, k] - mean[k]) / scale[k]
                        ax.plot(path['s'], y, marker='.', ms=3, lw=.8, label=f'Coordinate {k}')
                        net = float(abs(y[-1] - y[0])); tv = float(np.abs(np.diff(y)).sum())
                        metrics[f'{prefix}_transect_{t:02d}_coordinate_{k:03d}'] = dict(
                            count=len(y), rho=rho(path['s'], y), total_variation=tv,
                            endpoint_change=net, excess_variation_ratio=tv/net if net > 1e-12 else None,
                            median_adjacent_jump=float(np.median(np.abs(np.diff(y)))),
                            max_off_axis_A=float(path['off_axis'].max()))
                    right = ax.twinx()
                    right.plot(path['s'], path['support_fraction'], 'k--', lw=.9, alpha=.6)
                    right.set(ylim=(-.05, 1.05), ylabel='Crystal fraction in input')
                    ax.set(xlabel='Position along physical transect (Å)', ylabel='Signed coordinate / fit SD', title=f'Transect {t + 1}')
                for ax in axes.flat[len(frame['transects']):]:
                    ax.set_visible(False)
                axes[0, 0].legend(fontsize=7)
                fig.suptitle(f'{rep} · {Path(frame["record"]["file"]).stem}: individual physical transects\n'
                             'Actual atom-centered outputs; no smoothing or interpolation of embeddings; black = observed crystalline fraction')
                for suffix in ('png', 'pdf'):
                    fig.savefig(root / 'plots' / f'{prefix}-transects.{suffix}', dpi=160)
                plt.close(fig)
    write(root / 'technical/selection.json', selection)
    write(root / 'technical/provenance.json', provenance)
    write_metric_table(metrics, root, family='spatial_vicreg_coordinates', name='coordinates')
    report = [
        '# GeoFormer epoch-34 coordinate transitions', '',
        'Completed descriptive inference on the archived checkpoint. Its saved recipe disables '
        'neighbor shifting and includes FactorVAE. This is not the proposed causal training comparison.', '',
        'Coordinates were selected on the fitting half of 174 ps and evaluated on the opposite '
        'spatial half of all three snapshots. These snapshots were in encoder training; this '
        'is not independent-source validation. Indices are zero-based.', '',
        '| Snapshot | Export | Coordinate | Held-region rho | Liquid-only rho | Crystal-clear rho |',
        '| --- | --- | ---: | ---: | ---: | ---: |',
    ]
    for i, frame in frames.items():
        for rep in ('encoder', 'projector'):
            prefix = f'frame_{i:02d}_{rep}'
            for k in selection[rep]['coordinates_zero_based'][:3]:
                row = metrics[f'{prefix}_coordinate_{k:03d}']
                values = ['undefined' if row[key] is None else f'{row[key]:.3f}'
                          for key in ('test_rho', 'liquid_rho', 'strict_clear_rho')]
                report.append(f'| {Path(frame["record"]["file"]).stem} | {rep} | {k} | ' + ' | '.join(values) + ' |')
    report += ['', 'Correlations retain the original coordinate sign. Figures orient selected '
               'coordinates toward increasing liquid distance. High pooled correlation can arise '
               'from phase separation or shared support; inspect individual paths and liquid-only '
               'results before calling a transition smooth.', '',
               'The distance coordinate is half the difference between distance to crystal core '
               'and distance to bulk-like liquid, not an exact interface distance. Bands show '
               'observation spread. See [frozen metric definitions](tables/METRICS.md).', '']
    for path in sorted((root / 'plots').glob('*.png')):
        report += [f'[{path.stem}](plots/{path.name})', '', f'![{path.stem}](plots/{path.name})', '']
    (root / 'README.md').write_text('\n'.join(report))
    write(root / 'technical/summary.json', summaries)
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    run(parser.parse_args().config)
