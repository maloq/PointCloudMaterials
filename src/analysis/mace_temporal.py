"""Held-out topology, history interventions and spatial maps for temporal MACE."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, normalize
import torch

from src.data_utils.mace_history import Histories
from src.data_utils.temporal_campaign import write_json
from src.models.encoders.pretrained_mace import PretrainedMACEEncoder
from src.training_methods.mace_temporal import TemporalLearner


def scores(target, prediction, materials):
    def measure(y, p):
        squared_error = np.square(p-y)
        return dict(mse=float(squared_error.mean()),
                    r2=float(1-squared_error.sum()/np.square(y-y.mean(0)).sum()))
    result = measure(target, prediction)
    result['by_material'] = {name: measure(target[materials == i], prediction[materials == i])
                             for i, name in enumerate(('Al', 'Mg', 'Ta'))}
    result['mean_material_r2'] = float(np.mean([v['r2'] for v in result['by_material'].values()]))
    return result


@torch.no_grad()
def infer(model, data, rows, cfg, view=0, intervention='history', spatial=False):
    embeddings, predictions = [], []
    size = cfg['microbatch_size']
    for start in range(0, len(rows), size):
        x, _, material = data.get(rows[start:start+size], view)
        if spatial:
            z = model(x[:, -1], material)
        else:
            if intervention == 'repeat_anchor':
                x = x[:, -1:].expand_as(x).contiguous()
            elif intervention == 'reverse_past':
                x = torch.cat((x[:, :-1].flip(1), x[:, -1:]), dim=1)
            elif intervention != 'history':
                raise ValueError(f'Unknown temporal intervention: {intervention}')
            z = model.encoder(x, material)
            predictions.append(model.tda(z).cpu().numpy())
        embeddings.append(z.cpu().numpy())
    return np.concatenate(embeddings), None if spatial else np.concatenate(predictions)


def ridge(train_z, train_y, val_z, alpha):
    probe = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='cholesky'))
    probe.fit(train_z, train_y)
    return probe.predict(val_z)


def effective_rank(z):
    eig = np.linalg.eigvalsh(np.cov(z.T)).clip(0)
    trace = eig.sum()
    if trace == 0:
        return 0.0
    p = eig[eig > 0]/trace
    return float(np.exp(-(p*np.log(p)).sum()))


def analyze(cfg):
    out = Path(cfg['output'])
    directory = out / 'analysis'
    directory.mkdir(exist_ok=True)
    with np.load(out / 'scaling.npz') as archive:
        scaling = dict(archive)
    data = Histories(cfg, scaling)
    train = np.random.default_rng(cfg['seed']+1).choice(
        data.indices['train'], cfg['analysis']['probe_train_anchors'], replace=False)
    val = data.indices['val']
    train_y = data.targets[torch.as_tensor(train, device='cuda'), 0].cpu().numpy()
    val_y = data.targets[torch.as_tensor(val, device='cuda'), 0].cpu().numpy()
    materials = data.materials[val]
    model = TemporalLearner(cfg).cuda().eval()
    results = {}
    for name, filename in (('initial', 'initial.pt'), ('trained', 'best.pt')):
        saved = torch.load(out / filename, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['model'], strict=True)
        train_z, _ = infer(model, data, train, cfg)
        val_z, predictions = infer(model, data, val, cfg)
        results[name+'_head'] = scores(val_y, predictions, materials)
        results[name+'_ridge'] = scores(val_y, ridge(train_z, train_y, val_z, cfg['analysis']['ridge_alpha']), materials)
        if name == 'trained':
            selected_epoch = saved['epoch']
            trained_z, trained_train_z, trained_predictions = val_z, train_z, predictions
        print('ANALYSIS', name, json.dumps(results[name+'_head']), flush=True)
    # Source-matched controls: a mean target per material, and the earlier trained
    # single-frame encoder with a freshly fitted probe in this run's target basis.
    mean_prediction = np.stack([train_y[data.materials[train] == m].mean(0) for m in materials])
    results['material_mean'] = scores(val_y, mean_prediction, materials)
    previous = torch.load(cfg['analysis']['single_frame_checkpoint'], map_location='cpu', weights_only=False)
    previous_cfg = previous['config']
    performance = dict(previous_cfg['performance'], compile_radial_mlp=False)
    spatial = PretrainedMACEEncoder(previous_cfg['pretrained_checkpoint'], performance=performance).cuda().eval()
    spatial.load_state_dict({k.removeprefix('encoder.'): v for k, v in previous['model'].items() if k.startswith('encoder.')}, strict=True)
    spatial_train, _ = infer(spatial, data, train, cfg, spatial=True)
    spatial_val, _ = infer(spatial, data, val, cfg, spatial=True)
    results['previous_single_frame_ridge'] = scores(val_y,
        ridge(spatial_train, train_y, spatial_val, cfg['analysis']['ridge_alpha']), materials)
    del spatial, previous
    interventions = {}
    for mode in ('repeat_anchor', 'reverse_past'):
        z, predicted = infer(model, data, val, cfg, intervention=mode)
        interventions[mode] = dict(scores=scores(val_y, predicted, materials),
            embedding_rms_change=float(np.sqrt(np.square(z-trained_z).mean())))
    view_features = [trained_z]
    for view in (1, 2, 3):
        view_features.append(infer(model, data, val, cfg, view=view)[0])
    view_features = np.stack(view_features, axis=1)
    geometry = {}
    rng = np.random.default_rng(cfg['seed'])
    for m, name in enumerate(('Al', 'Mg', 'Ta')):
        z = view_features[materials == m]
        random_distance = np.square(z[:, 0]-z[rng.permutation(len(z)), 0]).mean()
        geometry[name] = dict(effective_rank=effective_rank(z[:, 0]),
            spatial_neighbor_random_ratio=float(np.square(z[:, 0]-z[:, 1]).mean()/random_distance),
            temporal_neighbor_random_ratio=float(np.square(z[:, 0]-z[:, 2]).mean()/random_distance))
    raw_prediction = (trained_predictions*scaling['tda_std']) @ scaling['tda_components'] + scaling['tda_mean']
    raw_target = data.raw_targets[val, 0]
    results['trained_head_raw144'] = scores(raw_target, raw_prediction, materials)
    train_norm = StandardScaler().fit(trained_train_z)
    train_standard = train_norm.transform(trained_train_z)
    val_standard = train_norm.transform(trained_z)
    projection = PCA(n_components=2).fit(train_standard)
    projected = projection.transform(val_standard)
    clusters = KMeans(n_clusters=7, n_init=10, random_state=cfg['seed']).fit(normalize(train_standard))
    labels = clusters.predict(normalize(val_standard))
    np.savez_compressed(directory / 'embeddings.npz', train_rows=train, val_rows=val,
        train_embeddings=trained_train_z, val_view_embeddings=view_features, materials=materials,
        source_indices=data.sources[val], targets=val_y, head_predictions=trained_predictions,
        embedding_pca=projected, clusters=labels)
    torch.save(dict(encoder=model.encoder.state_dict(), config=cfg['encoder'], protocol='temporal80',
                    selected_epoch=selected_epoch), out / 'encoder.pt')
    report = dict(selected_epoch=selected_epoch, scores=results, interventions=interventions,
        representation=geometry, probe_train_anchors=len(train), validation_anchors=len(val),
        single_frame_reference=cfg['analysis']['single_frame_checkpoint'],
        limitations=cfg['data_limitations'])
    write_json(directory / 'metrics.json', report)
    records = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for domain in ('train', 'validation'):
        axes[0].plot([r['epoch'] for r in records], [r[domain]['tda_mse'] for r in records], label=domain)
    axes[0].set(xlabel='Epoch', ylabel='Standardized relaxed TDA MSE')
    axes[0].legend()
    for i, name in enumerate(('Al', 'Mg', 'Ta')):
        mask = materials == i
        axes[1].scatter(projected[mask, 0], projected[mask, 1], s=8, alpha=.7, label=name)
    axes[1].set(xlabel='Embedding PC1', ylabel='Embedding PC2', title='Held-out histories; train-fitted PCA')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(directory / 'training_and_embeddings.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, name in enumerate(('Al', 'Mg', 'Ta')):
        mask = materials == i
        axes[i].scatter(val_y[mask, 0], trained_predictions[mask, 0], s=8, alpha=.6)
        axes[i].set(xlabel='Relaxed TDA PC1 target', ylabel='Prediction', title=name)
    fig.tight_layout()
    fig.savefig(directory / 'topology_predictions.png', dpi=180)
    plt.close(fig)
    positions = np.concatenate([np.load(Path(r['directory']) / 'center_positions.npy') for r in data.records])[val]
    fig, ax = plt.subplots(figsize=(6, 5))
    mask = materials == 0
    scatter = ax.scatter(positions[mask, 0], positions[mask, 1], c=labels[mask], s=14, cmap='tab10', vmin=0, vmax=9)
    ax.set(xlabel='x (Å)', ylabel='y (Å)', title='Validation Al: sampled centers, projected through z')
    fig.colorbar(scatter, ax=ax, label='Descriptive latent cluster')
    fig.tight_layout()
    fig.savefig(directory / 'al_spatial_clusters.png', dpi=180)
    plt.close(fig)
    lines = ['# Temporal MACE training and analysis', '',
        f'Selected epoch: **{selected_epoch}**, by minimum validation relaxed-anchor TDA MSE.', '',
        '| Representation / predictor | MSE | Mean within-material R² |', '| --- | ---: | ---: |']
    for name in ('material_mean', 'initial_head', 'trained_head', 'initial_ridge', 'trained_ridge', 'previous_single_frame_ridge'):
        score = results[name]
        lines.append(f"| {name} | {score['mse']:.5f} | {score['mean_material_r2']:.4f} |")
    lines += ['', 'All rows above use the same standardized 32-component relaxed target. Ridge probes use '
        'the same training anchors, training-only feature scaling and fixed alpha. The previous '
        'single-frame encoder had a different objective and exposure budget; this is a reference, '
        'not a controlled architecture comparison.', '',
        '## Dependence on history', '', '| Input intervention | TDA MSE | Embedding RMS change |', '| --- | ---: | ---: |']
    for name, value in interventions.items():
        lines.append(f"| {name} | {value['scores']['mse']:.5f} | {value['embedding_rms_change']:.5f} |")
    lines += ['', 'These are inference interventions on the trained model, not retrained ablations. '
        'Repeating the anchor removes motion; reversing past frames changes order while retaining the anchor.', '',
        '## Representation', '', '| Material | Effective rank | Spatial/random distance | Temporal/random distance |',
        '| --- | ---: | ---: | ---: |']
    for name, value in geometry.items():
        lines.append(f"| {name} | {value['effective_rank']:.2f} | {value['spatial_neighbor_random_ratio']:.4f} | {value['temporal_neighbor_random_ratio']:.4f} |")
    lines += ['', '![Training and embeddings](training_and_embeddings.png)', '',
        '![Topology predictions](topology_predictions.png)', '', '![Al spatial clusters](al_spatial_clusters.png)', '',
        '## Scope', '', cfg['data_limitations'], '',
        'The spatial plot contains sampled validation centers, projected through z; clusters are '
        'descriptive, not identified phases. This evaluates real temporal windows and does not '
        'substitute duplicated snapshots for a full static analysis.', '',
        'Artifacts: [metrics](metrics.json), [embeddings](embeddings.npz), '
        '[selected encoder](../encoder.pt), [training summary](../training_summary.json).', '']
    (directory / 'RESULTS.md').write_text('\n'.join(lines))
    print('ANALYSIS_COMPLETE', json.dumps(report), flush=True)
