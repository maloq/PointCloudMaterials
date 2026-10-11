"""Local frozen-head TDA diagnostics on the existing spatial validation panel.

Invoke this file directly for inference so imports resolve inside the run's
frozen source bundle. Analysis is a separate, CPU-only invocation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def infer(args):
    # No live model/target producer is imported in this process.
    snapshot = args.run / 'technical/code'
    sys.path.insert(0, str(snapshot))
    import torch
    from src.research.spatial_architecture.model import Spatial
    from src.models.encoders.esen_runtime import configure
    from src.research.structural_response.model import views

    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    folder = args.run / 'analyses' / args.architecture / 'technical'
    checkpoint = folder / 'final.pt'
    completed = read(folder / 'complete.json')
    if sha(checkpoint) != completed['checkpoint_sha256']:
        raise ValueError('Final checkpoint checksum changed')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    c = saved['config']
    if saved['update'] != c['spatial_training']['updates']:
        raise ValueError('This comparison requires completed final checkpoints')
    files = read(args.cache / 'complete.json')['files']
    for name in ('spatial.json', 'spatial-selection.npz'):
        if sha(args.cache / name) != files[name]:
            raise ValueError('Changed frozen validation input: ' + name)
    panel = np.load(args.cache / 'spatial-selection.npz')
    meta = read(args.cache / 'spatial.json')
    if panel['positions'].shape != (4096, 80, 3):
        raise ValueError('Expected the original 4096-observation panel')
    weights = np.asarray(meta['weights'], np.float64)
    torch.manual_seed(c['seed'])
    model = Spatial(c, args.architecture).cuda()
    model.load_state_dict(saved['model'], strict=True)
    model.eval().requires_grad_(False)
    variant = c['execution_variants'][args.architecture]
    configure(model.geometry, args.architecture, variant)
    prediction = np.empty((4096, 2, len(weights)), np.float32)
    start_time = time.monotonic()
    with torch.no_grad():
        for start in range(0, 4096, 256):
            q = torch.from_numpy(panel['positions'][start:start+256]).cuda()
            for view, (x, keep) in enumerate(views(q, 'vicreg_descriptors', c['seed'] + start)):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    z = model.geometry(x, keep)
                prediction[start:start+256, view] = model.descriptors(z.float()).cpu().numpy()
            print(args.label, start + len(q), '/ 4096', flush=True)
    torch.cuda.synchronize()
    if not np.isfinite(prediction).all():
        raise FloatingPointError('Nonfinite frozen-head prediction')
    target = panel['target'].astype(np.float64)
    mse = ((prediction.astype(np.float64) - target[:, None]) ** 2).mean((0, 1))
    reconstructed = float(.5 * (mse * weights).sum())
    logged = read(folder / 'validation.json')['descriptor_half_mse']
    # CUDA reductions can differ slightly; reject a scientific discrepancy.
    if abs(reconstructed - logged) > 2e-4:
        raise ValueError(f'Validation reconstruction differs: {reconstructed} vs {logged}')
    dest = args.output / 'technical'
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / (args.label + '.npz'), prediction=prediction,
                        target=panel['target'], rows=panel['rows'])
    receipt = dict(label=args.label, architecture=args.architecture, run=str(args.run),
        identity=saved['identity'], checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        update=saved['update'], epochs=c['spatial_training']['epochs'],
        validation_input_sha256=files['spatial-selection.npz'],
        descriptor_metadata_sha256=files['spatial.json'], seed=c['seed'],
        precision='BF16 encoder, FP32 geometry/head; FP64 diagnostic reductions',
        variant=variant, evaluation_batch=256, views='original two jitter views at seed+batch_start',
        logged_descriptor_half_mse=logged, reconstructed_descriptor_half_mse=reconstructed,
        inference_seconds=time.monotonic()-start_time, frozen_source=str(snapshot),
        model_source=str(sys.modules[Spatial.__module__].__file__),
        diagnostic_sha256=sha(Path(__file__)), prediction_sha256=sha(dest/(args.label+'.npz')),
        resource_note='Brief local inference shares the node GPU; not a speed benchmark; no job changed')
    write(dest / (args.label + '.json'), receipt)
    print(json.dumps(receipt), flush=True)


def block_name(column):
    if column['family'] != 'tda':
        return column['family']
    match = re.fullmatch(r'tda/(n(?:32|80)_h[012])_(.+)', column['name'])
    if match is None:
        raise ValueError(column)
    prefix, suffix = match.groups()
    kind = ('spectrum' if re.fullmatch(r'(image|death)\d+', suffix)
            else 'betti' if suffix.startswith('betti') else 'statistics')
    return f'{prefix}_{kind}'


def analyze(args):
    sys.path.insert(0, str(REPO))
    from src.experiment_runner.metric_docs import write_metric_rows
    from scipy.stats import spearmanr
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    meta = read(args.cache / 'spatial.json')
    columns, weights = meta['descriptor_columns'], np.asarray(meta['weights'])
    manifest = read(args.descriptor_manifest)
    shards = [s for s in manifest['shards'] if s['task']['role']=='selection']
    ends = np.cumsum([s['task']['rows'] for s in shards])
    panel = np.load(args.cache / 'spatial-selection.npz')
    shard_ids = np.searchsorted(ends, panel['rows'], side='right')
    from collections import Counter
    population = dict(material_rows=dict(Counter(shards[i]['task']['material'] for i in shard_ids)),
                      sources=len({shards[i]['task']['source'] for i in shard_ids}),
                      ancestors=len({shards[i]['task']['lineage'] for i in shard_ids}),
                      manifest_sha256=sha(args.descriptor_manifest))
    if population['material_rows'] != {'Al': 4096}:
        raise ValueError('This audit expects the verified original Al validation panel')
    active = weights > 0
    groups = {}
    for index, column in enumerate(columns):
        groups.setdefault(block_name(column), []).append(index)
    feature_rows, group_rows, summary_rows, records = [], [], [], []
    previous = None
    scores = []
    for label in args.labels:
        p = args.output / 'technical' / label
        record = read(p.with_suffix('.json'))
        if sha(p.with_suffix('.npz')) != record['prediction_sha256']:
            raise ValueError('Changed prediction cache')
        values = np.load(p.with_suffix('.npz'))
        y, pred = values['target'].astype(np.float64), values['prediction'].astype(np.float64)
        if previous is not None and (not np.array_equal(previous['target'], values['target'])
                                     or not np.array_equal(previous['rows'], values['rows'])):
            raise ValueError('Models were evaluated on different observations/targets')
        previous = values
        mse = ((pred-y[:, None])**2).mean((0, 1))
        baseline, variance = (y*y).mean(0), y.var(0)
        for index, column in enumerate(columns):
            feature_rows.append(dict(model=label, feature=column['name'], block=block_name(column),
                active=bool(active[index]), weight=float(weights[index]), mse=mse[index],
                baseline_mse=baseline[index], target_variance=variance[index],
                mean_baseline_skill=1-mse[index]/baseline[index] if active[index] and baseline[index]>0 else None,
                r2=1-mse[index]/variance[index] if active[index] and variance[index]>1e-12 else None))
        model_scores = []
        for name, indices in groups.items():
            ids = np.array(indices)[active[indices]]
            if not len(ids):
                raise ValueError('No active outputs in block ' + name)
            row = dict(model=label, block=name, active_outputs=len(ids), total_outputs=len(indices),
                mse=float(mse[ids].mean()), baseline_mse=float(baseline[ids].mean()),
                target_variance=float(variance[ids].mean()),
                mean_baseline_skill=float(1-mse[ids].sum()/baseline[ids].sum()),
                r2=float(1-mse[ids].sum()/variance[ids].sum()))
            group_rows.append(row)
            if name != 'bond_order':
                model_scores.append(row['r2'])
        scores.append(model_scores)
        tda = np.array([col['family']=='tda' for col in columns]) & active
        summary_rows.append(dict(model=label, update=record['update'], epochs=record['epochs'],
            rows=len(y), active_tda_outputs=int(tda.sum()), inactive_tda_outputs=int((~active & np.array([col['family']=='tda' for col in columns])).sum()),
            descriptor_half_mse=float(.5*(mse*weights).sum()),
            tda_weighted_mse=float((mse[tda]*weights[tda]).sum()/weights[tda].sum()),
            tda_mean_baseline_skill=float(1-(mse[tda]*weights[tda]).sum()/(baseline[tda]*weights[tda]).sum()),
            bond_order_mse=float(mse[:40].mean())))
        records.append(record)
    rank_rows = [dict(first=args.labels[i], second=args.labels[j],
                     spearman=float(spearmanr(scores[i], scores[j]).statistic))
                 for i in range(len(scores)) for j in range(i+1,len(scores))]
    for name, rows in [('components', feature_rows), ('blocks', group_rows),
                       ('models', summary_rows), ('rank_agreement', rank_rows)]:
        write_metric_rows(rows, args.output, family='spatial_tda_difficulty', name=name)
    names = [k for k in groups if k!='bond_order']
    display = np.array(scores).T * 100
    fig, ax = plt.subplots(figsize=(9, 9), layout='constrained')
    picture = ax.imshow(display, vmin=0, vmax=100, cmap='viridis')
    ax.set_xticks(range(len(args.labels)), args.labels)
    ax.set_yticks(range(len(names)), [n.replace('_', ' · ') for n in names])
    for row in range(len(names)):
        for col in range(len(args.labels)):
            ax.text(col, row, f'{display[row,col]:.1f}%', ha='center', va='center',
                    color='black' if display[row,col]>65 else 'white')
    ax.set_title('TDA prediction: validation R²\nSame 4,096 Al patches / 15 sources; trained heads; 12 epochs; higher is better')
    fig.colorbar(picture, ax=ax, label='R² × 100', shrink=.7)
    plots = args.output / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / 'tda-difficulty.png', dpi=170)
    fig.savefig(plots / 'tda-difficulty.pdf')
    write(args.output / 'technical/analysis.json', dict(records=records, panel_sha256=sha(args.cache/'spatial-selection.npz'),
          descriptor_metadata_sha256=sha(args.cache/'spatial.json'),
          scope='Descriptive comparison of one seed per architecture; validation diagnostic, no refitting or selection',
          population=population,
          material_scope='Al-only spatial validation after mixed-material pretraining; not a nucleation evaluation'))
    print(json.dumps(dict(models=summary_rows, rank_agreement=rank_rows), indent=2))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('infer', 'analyze'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--architecture', choices=('mace', 'esen', 'tensornet'))
    parser.add_argument('--label')
    parser.add_argument('--labels', nargs='+')
    parser.add_argument('--descriptor-manifest', type=Path)
    args = parser.parse_args()
    (infer if args.stage=='infer' else analyze)(args)


if __name__ == '__main__':
    main()
