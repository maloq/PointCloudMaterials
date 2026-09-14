"""Shared train-only UMAP and exact additive crystal-score contributions."""

import importlib.metadata
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist
import torch
from umap import UMAP

from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json


def readout_parameters(probe, mean, scale):
    """Express the frozen ridge crystal-minus-other score in forecast-standardized units."""
    delta = probe['coefficients'][:, 1]-probe['coefficients'][:, 0]
    raw_weight = delta[:-1]/probe['std'].double()
    raw_bias = delta[-1]-probe['mean'].double()@raw_weight
    return dict(weight=(scale.double()*raw_weight).numpy(),
                bias=np.asarray((mean.double()@raw_weight+raw_bias).item()))


def channel_contributions(values, weight, history_frames=17):
    """Signed per-channel contribution to score change from mean observed history."""
    return (values-values[:history_frames].mean(axis=0))*weight


def time_blocks(time, coordinates, frames=8):
    """Nonoverlapping six-ps summaries of display coordinates; retain final partial bin."""
    return (np.asarray([np.mean(time[i:i+frames]) for i in range(0, len(time), frames)]),
            np.stack([np.median(coordinates[i:i+frames], axis=0) for i in range(0, len(time), frames)]))


def reuse_extraction(config, output):
    """Import the explicit retained PCA-extraction producer, leaving its results intact."""
    root = Path(config['reuse_extraction']); target = output/'technical'
    metadata = json.loads((root/'technical/projection.json').read_text())
    sample = root/'technical/training-projection.npz'
    if sha256(sample) != metadata['sample_and_projection_sha256']:
        raise ValueError(f'Retained training sample changed: {sample}')
    source_hashes = json.loads((root/'technical/input-hashes.json').read_text())
    plan = load_json(config['plan'])
    fit = next(r for r in plan['runs'] if r['name'] == 'history12_spatial_mixture4' and r['seed'] == config['fit_seed'])
    checkpoint = Path(fit['fit'])/'technical/best.pt'
    probe_path = Path(plan['local_assay'])/'local_crystal_probe.pt'
    for path in (checkpoint, probe_path):
        if sha256(path) != source_hashes[str(path)]:
            raise ValueError(f'Retained extraction used a different artifact: {path}')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    probe = torch.load(probe_path, map_location='cpu', weights_only=False)
    np.savez(target/'crystal-readout.npz', **readout_parameters(probe, payload['mean'], payload['scale']))
    examples = json.loads((root/'technical/examples.json').read_text())
    imported = {str(sample): sha256(sample)}
    for e in examples:
        path = root/'technical'/f"{e['category']}.npz"
        imported[str(path)] = sha256(path)
        with np.load(path) as arrays:
            # PCA coordinates are deliberately not imported into the new projection.
            np.savez_compressed(target/path.name, **{k: arrays[k] for k in arrays.files if not k.endswith('_pc')})
    with np.load(sample) as arrays:
        np.savez_compressed(target/'training-projection.npz',
                            standardized_sample=arrays['standardized_sample'], identities=arrays['identities'])
    write_json(target/'examples.json', examples)
    write_json(target/'input-hashes.json', dict(**source_hashes, **imported))
    return examples


def project(config, output, examples):
    """Fit on training only; transform all unique query embeddings in one fixed batch."""
    root = output/'technical'
    with np.load(root/'training-projection.npz') as a:
        reference, identities = a['standardized_sample'], a['identities']
    with np.load(root/'crystal-readout.npz') as a:
        weight, bias = a['weight'], a['bias']
    reducer = UMAP(n_components=2, n_neighbors=config['umap']['n_neighbors'],
                   min_dist=config['umap']['min_dist'], metric=config['umap']['metric'],
                   random_state=config['projection_seed'], transform_seed=config['projection_seed'], n_jobs=1)
    print(f'Fitting shared UMAP to {len(reference)} training embeddings only...', flush=True)
    reducer.fit(reference)
    records, blocks, index = [], [], []
    for e in examples:
        with np.load(root/f"{e['category']}.npz") as a:
            arrays = {key: a[key] for key in a.files}
        np.testing.assert_allclose(arrays['true_z'].astype(np.float64)@weight+bias,
                                   arrays['observed_margin'], atol=2e-5, rtol=2e-5)
        records.append(arrays)
        for key in ('full_true_z', *(m+'_mean_z' for m in config['models']), 'component_mean_z', 'sample_z'):
            shape = arrays[key].shape[:-1]
            index.append((len(records)-1, key, shape))
            blocks.append(arrays[key].reshape(-1, 256))
    query = np.concatenate(blocks)
    unique, inverse = np.unique(query, axis=0, return_inverse=True)
    print(f'Transforming {len(unique)} unique observed/predicted embeddings...', flush=True)
    coordinates = reducer.transform(unique)[inverse]
    if not np.isfinite(coordinates).all():
        raise FloatingPointError('UMAP generated nonfinite query coordinates.')
    offset = 0
    for record_index, key, shape in index:
        count = int(np.prod(shape))
        name = 'component_map' if key == 'component_mean_z' else key[:-2]+'_map'
        records[record_index][name] = coordinates[offset:offset+count].reshape(*shape, 2)
        offset += count
    for e, arrays in zip(examples, records):
        anchor = e['anchor']
        arrays['true_map'] = arrays['full_true_map'][anchor-16:anchor+13]
        np.savez_compressed(root/f"{e['category']}.npz", **arrays)
    channel_order = leaves_list(linkage(pdist(reference.T, metric='correlation'),
                                       method='average', optimal_ordering=True))
    influence = abs(weight)*reference.std(axis=0, ddof=1)
    top_channels = np.argsort(-influence, kind='stable')[:config['channel_count']]
    np.savez(root/'channel-layout.npz', weight=weight, bias=bias, order=channel_order,
             top_channels=top_channels, training_contribution_std=influence)
    joblib.dump(reducer, root/'umap.joblib')
    np.savez_compressed(root/'training-projection.npz', standardized_sample=reference,
                        identities=identities, coordinates=reducer.embedding_)
    info = dict(method='UMAP', dimensions=2, training_samples=len(reference),
        training_sources=len(np.unique(identities[:, 0])), parameters=config['umap'], seed=config['projection_seed'],
        package_version=importlib.metadata.version('umap-learn'), query_unique_rows=len(unique),
        sample_and_projection_sha256=sha256(root/'training-projection.npz'),
        model_sha256=sha256(root/'umap.joblib'), channel_layout_sha256=sha256(root/'channel-layout.npz'),
        semantics='Fit to training embeddings only. One transform batch of unique query rows; exact coordinates reused. '
                  'Nonlinear display coordinates, not physical distances or explained-variance components. '
                  'Transform the mean embedding itself; never average projected component coordinates.')
    write_json(root/'projection.json', info)
    print('UMAP projection and training-ranked channel layout ready.', flush=True)
    return info
