"""Connect measured local structures to observed and predicted embedding paths."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json
from src.research.forecast_crystallization.local_analyze import export_rows
from src.research.forecast_crystallization.local_metrics import first_sustained_onset, risk_windows
from src.research.forecast_spatial_mixture.evaluate import assay, directory
from src.training_methods.embedding_forecast.context_mixture import (
    build_forecaster, call_forecaster, crystal_probability, sample_trajectories,
)
from src.training_methods.embedding_forecast.spatial import pool_neighbors


CATEGORIES = ('correct', 'early', 'missed', 'false-alarm')


def example_masks(crystal, anchors, scores, threshold, cadence, persistence, negative_history):
    """Diagnostic cases with transitions 3–6 ps ahead, leaving visible post-onset truth."""
    onset = first_sustained_onset(crystal, persistence)
    eligible = risk_windows(crystal, onset, anchors, negative_history)
    delay = (onset[:, None]-anchors)*cadence
    actual = delay <= scores.shape[-1]*cadence
    predicted = scores.max(-1) >= threshold
    error = ((scores >= threshold).argmax(-1)+1)*cadence-delay
    visible_transition = actual & (delay >= 3) & (delay <= 6)
    return dict(correct=eligible & visible_transition & predicted & (abs(error) <= 1.5),
                early=eligible & visible_transition & predicted & (error < -3),
                missed=eligible & visible_transition & ~predicted,
                **{'false-alarm': eligible & ~actual & predicted}), onset


class Inputs:
    """Record every inspected artifact; verify existing digests when available."""

    def __init__(self):
        self.hashes = {}

    def record(self, path, expected=None):
        path = Path(path)
        key = str(path)
        if key not in self.hashes:
            self.hashes[key] = sha256(path)
        if expected is not None and self.hashes[key] != expected:
            raise ValueError(f'Visualization input changed: {path}; expected {expected}, '
                             f'observed {self.hashes[key]}')
        return path

    def json(self, path):
        return json.loads(self.record(path).read_text())


def select_examples(plan, run, data, inputs):
    root = directory(plan, run)
    prediction = inputs.json(root/'prediction.json')
    report = inputs.json(root/'results.json')
    threshold = next(r['threshold'] for r in report['results']['frame_crystal_probability']['onset']
                     if r['horizon_ps'] == 9 and r['persistence_frames'] == 3)
    inputs.record(root/'anchors.npy', prediction['anchors_sha256'])
    np.testing.assert_array_equal(np.load(root/'anchors.npy'), data['anchors'])
    examples = {}
    for source, crystal in zip(data['sources'], data['crystal']):
        if source['split'] != 'test':
            continue
        filename = f"source_{source['source_index']:03d}.npz"
        path = inputs.record(root/filename, prediction['scores_sha256'][filename])
        with np.load(path) as arrays:
            scores = arrays['frame_crystal_probability']
        masks, onset = example_masks(crystal, data['anchors'], scores, threshold,
                                     plan['cadence_ps'], 3, plan['negative_history_frames'])
        # One source per case: avoid four overlapping windows of the same transition.
        for category in CATEGORIES:
            if category not in examples and masks[category].any():
                center, column = np.argwhere(masks[category])[0]
                anchor = int(data['anchors'][column])
                crossing = scores[center, column] >= threshold
                examples[category] = dict(category=category, source=source, center=int(center),
                    anchor=anchor, onset_frame=int(onset[center]), threshold=threshold,
                    predicted_delay_ps=float((crossing.argmax()+1)*plan['cadence_ps'])
                    if crossing.any() else None,
                    cached_probability=scores[center, column].copy())
                break
        if len(examples) == len(CATEGORIES):
            break
    if set(examples) != set(CATEGORIES):
        raise ValueError(f'Missing diagnostic categories: {set(CATEGORIES)-set(examples)}')
    return [examples[key] for key in CATEGORIES]


def training_projection(cache, manifest, mean, scale, config, output, inputs):
    """Retain an equal embedding sample from each training source for the shared map."""
    samples, identities = [], []
    rng = np.random.default_rng(config['projection_seed'])
    for record in manifest['shards']:
        if record['split'] != 'train':
            continue
        centers = np.sort(rng.choice(record['centers'], config['projection_centers_per_source'], replace=False))
        frames = np.sort(rng.choice(record['frames'], config['projection_frames_per_center'], replace=False))
        values = np.load(cache/record['directory']/'embeddings.npy', mmap_mode='r')
        samples.append(values[centers[:, None], frames].astype(np.float32).reshape(-1, 256))
        identities.extend((record['source_index'], int(c), int(t)) for c in centers for t in frames)
    standardized = (np.concatenate(samples)-mean)/scale
    filename = output/'technical/training-projection.npz'
    np.savez_compressed(filename, standardized_sample=standardized, identities=np.asarray(identities))


def measured_clouds(trajectory, frames, atom_id):
    """Same instantaneous 80-nearest periodic neighborhood as the embedding producer, in Å."""
    center = np.searchsorted(trajectory.atom_ids, atom_id)
    np.testing.assert_equal(trajectory.atom_ids[center], atom_id)
    clouds, ids = [], []
    for frame in frames:
        low = trajectory.box_low[frame].astype(np.float64)
        lengths = trajectory.box_high[frame].astype(np.float64)-low
        points = np.mod(trajectory.positions[frame].astype(np.float64)-low, lengths)
        _, neighbors = cKDTree(points, boxsize=lengths).query(points[center], k=80, workers=1)
        np.testing.assert_equal(neighbors[0], center)
        offsets = points[neighbors]-points[center]
        offsets -= lengths*np.round(offsets/lengths)
        clouds.append(offsets.astype(np.float32))
        ids.append(trajectory.atom_ids[neighbors])
    return np.stack(clouds), np.stack(ids)


@torch.inference_mode()
def prepare(config, output):
    inputs = Inputs()
    plan = load_json(inputs.record(config['plan']))
    if plan['cadence_ps'] != .75 or plan['anchor_history_ps'] != 12 or plan['horizons_ps'][-1] != 9:
        raise ValueError('This visualization protocol requires the paired 12 ps history / 9 ps future study.')
    runs = [next(r for r in plan['runs'] if r['name'] == name and r['seed'] == config['fit_seed'])
            for name in config['models']]
    primary = next(r for r in runs if r['name'] == 'history12_spatial_mixture4')
    data = assay(plan)
    for name in ('local_observations.json', 'observed_scores.npy', 'physical_crystal_labels.npy',
                 'local_crystal_probe.pt', 'local_prediction_status.json'):
        inputs.record(data['root']/name)
    examples = select_examples(plan, primary, data, inputs)
    cache = Path(plan['embedding_cache'])
    manifest = inputs.json(cache/'manifest.json')
    spatial_manifest = inputs.json(Path(plan['spatial_cache'])/'manifest.json')
    spatial_records = {r['directory']: r for r in spatial_manifest['records']}
    config_fit = load_json(inputs.record(primary['config']))
    sources = load_json(inputs.record(config_fit['data']['sources_config']))['sources']
    probe = torch.load(data['root']/'local_crystal_probe.pt', weights_only=False, map_location='cpu')
    difference = probe['coefficients'][:, 1]-probe['coefficients'][:, 0]
    weight = difference[:-1]/probe['std'].double()
    bias = difference[-1]-probe['mean'].double()@weight
    models = {}
    for run in runs:
        fit = Path(run['fit'])/'technical'
        if inputs.json(fit/'status.json')['state'] != 'complete':
            raise ValueError(f'Incomplete fit: {fit}')
        pred = inputs.json(directory(plan, run)/'prediction.json')
        payload = torch.load(inputs.record(fit/'best.pt', pred['checkpoint_sha256']),
                             map_location='cpu', weights_only=False)
        if payload['cache_manifest_sha256'] != inputs.hashes[str(cache/'manifest.json')]:
            raise ValueError(f'Checkpoint cache mismatch: {fit}')
        if pred['probe_sha256'] != inputs.hashes[str(data['root']/'local_crystal_probe.pt')]:
            raise ValueError(f'Prediction crystal probe mismatch: {fit}')
        for name in ('model.py', 'context_mixture.py', 'spatial.py'):
            if name in payload['implementation_sha256']:
                inputs.record(Path(__file__).parents[2]/'training_methods/embedding_forecast'/name,
                              payload['implementation_sha256'][name])
        if models:
            torch.testing.assert_close(payload['mean'], mean, rtol=0, atol=0)
            torch.testing.assert_close(payload['scale'], scale, rtol=0, atol=0)
        mean, scale = payload['mean'], payload['scale']
        past = round(payload['config']['history_ps']/plan['cadence_ps'])
        model = build_forecaster(256, past+1, .75, plan['horizons_ps'], payload['variant'])
        model.load_state_dict(payload['model'], strict=True)
        models[run['name']] = (model.eval(), past, run, pred)
    if spatial_manifest['base_cache_manifest_sha256'] != inputs.hashes[str(cache/'manifest.json')]:
        raise ValueError('Spatial cache and embedding manifest differ.')
    training_projection(cache, manifest, mean.numpy(), scale.numpy(), config, output, inputs)
    from .trajectory_projection import readout_parameters
    np.savez(output/'technical/crystal-readout.npz', **readout_parameters(probe, mean, scale))
    for example_index, example in enumerate(examples):
        source = example['source']; index = source['source_index']
        source_record = sources[index]
        if source_record['name'] != source['name'] or source_record['preparation_seed'] != source['preparation_seed']:
            raise ValueError(f'Point-cloud source identity mismatch: {source}')
        shard = cache/source['directory']
        for filename, digest in source['checksums'].items():
            inputs.record(shard/filename, digest)
        label_dir = data['root']/'labels'/source['directory']
        for name in ('labels', 'embedding_rows', 'atom_ids'):
            inputs.record(label_dir/f'{name}.npy', source[f'{name}_sha256'])
        embedding_row = int(np.load(label_dir/'embedding_rows.npy')[example['center']])
        atom_id = int(np.load(label_dir/'atom_ids.npy')[example['center']])
        np.testing.assert_equal(np.load(shard/'atom_ids.npy')[embedding_row], atom_id)
        all_z = np.load(shard/'embeddings.npy', mmap_mode='r')
        anchor = example['anchor']; frames = np.arange(anchor-16, anchor+13)
        z = np.array(all_z[embedding_row], dtype=np.float32)
        standardized = (z-mean.numpy())/scale.numpy()
        full_time = np.load(shard/'time_ps.npy')
        labels = np.load(label_dir/'labels.npy')[example['center']]
        source_position = next(i for i, s in enumerate(data['sources']) if s['source_index'] == index)
        physical = data['crystal'][source_position, example['center']]
        trajectory = ShootingBinaryTrajectory.load(source_record['path'])
        inputs.record(trajectory.root/'manifest.json', source['trajectory_manifest_sha256'])
        trajectory_frames = np.load(shard/'frames.npy')[frames]
        np.testing.assert_array_equal(trajectory.timesteps[trajectory_frames], np.load(shard/'timesteps.npy')[frames])
        clouds, neighbor_ids = measured_clouds(trajectory, trajectory_frames, atom_id)
        neighbor_root = Path(plan['spatial_cache'])/source['directory']
        for filename, digest in spatial_records[source['directory']]['checksums'].items():
            inputs.record(neighbor_root/filename, digest)
        neighbors = np.load(neighbor_root/'neighbors.npy', mmap_mode='r')[embedding_row, anchor-16:anchor+1]
        columns = np.arange(anchor-16, anchor+1)
        # Use the assay's exact float32 pooling followed by cache-dtype storage.
        spatial = pool_neighbors(torch.from_numpy(np.array(all_z[:, columns], copy=True)),
                                 torch.from_numpy(neighbors.astype(np.int64))[None])[0].float().numpy()
        radii = np.load(neighbor_root/'radii_A.npy', mmap_mode='r')[embedding_row, anchor-16:anchor+1].copy()
        arrays = dict(time_ps=(frames-anchor)*.75, absolute_time_ps=full_time[frames],
            true_z=standardized[frames], full_true_z=standardized, full_time_ps=full_time,
            cloud_A=clouds, neighbor_atom_ids=neighbor_ids, physical_crystal=physical[frames],
            ptm_labels=labels[frames], full_physical_crystal=physical,
            observed_margin=data['observed'][source_position, example['center'], frames])
        for name, (model, past, run, pred) in models.items():
            history = torch.from_numpy(z[anchor-past:anchor+1])[None]
            batch = dict(spatial=torch.from_numpy(spatial[-past-1:])[None],
                         spatial_radii_A=torch.from_numpy(radii[-past-1:])[None])
            out = call_forecaster(model, (history-mean)/scale, batch, mean, scale)
            arrays[name+'_mean_z'] = out['mean'][0].numpy()
            filename = f'source_{index:03d}.npz'
            with np.load(inputs.record(directory(plan, run)/filename, pred['scores_sha256'][filename])) as saved:
                origin = int(np.flatnonzero(data['anchors'] == anchor)[0])
                cached_mean = saved['mean_margin'][example['center'], origin]
            new_margin = (out['mean'].double()@(scale.double()*weight)+mean.double()@weight+bias)[0].numpy()
            np.testing.assert_allclose(new_margin, cached_mean, atol=2e-4, rtol=2e-4,
                                       err_msg=f'CPU selected-window forecast differs from saved assay: {name}')
            arrays[name+'_margin'] = new_margin
            if name == primary['name']:
                torch.manual_seed(config['sample_seed']+example_index)
                samples = sample_trajectories(out, config['sample_paths'])[:, 0].numpy()
                arrays.update(component_mean_z=out['component_means'][0].numpy(),
                    component_std_z=out['component_std'][0].numpy(),
                    component_weight=out['mixture_logits'].softmax(-1)[0].numpy(), sample_z=samples)
                probability = crystal_probability(out, scale.double()*weight, mean.double()@weight+bias)[0].numpy()
                np.testing.assert_allclose(probability, example['cached_probability'], atol=2e-4, rtol=2e-4)
                arrays['crystal_probability'] = example.pop('cached_probability')
        example.update(atom_id=atom_id, embedding_row=embedding_row, anchor_time_ps=float(full_time[anchor]),
                       onset_delay_ps=float((example['onset_frame']-anchor)*.75)
                       if example['onset_frame'] < len(full_time) else None)
        np.savez_compressed(output/'technical'/f"{example['category']}.npz", **arrays)
        print(f"Extracted {example['category']}: source {index}, atom {atom_id}, "
              f"origin {example['anchor_time_ps']:.2f} ps", flush=True)
    write_json(output/'technical/examples.json', examples)
    write_json(output/'technical/input-hashes.json', inputs.hashes)
    return examples


def run(config, stage='all'):
    torch.set_num_threads(2)
    output = Path(config['output'])
    if stage == 'plots':
        if json.loads((output/'technical/config.json').read_text()) != config:
            raise ValueError('Rendering configuration differs from the retained extraction configuration.')
        examples = json.loads((output/'technical/examples.json').read_text())
        projection = json.loads((output/'technical/projection.json').read_text())
        if projection['method'] != 'UMAP':
            raise ValueError('Use the retained renderer-source to reproduce historical PCA figures.')
    else:
        if (output/'technical/config.json').exists():
            raise FileExistsError(f'Preserve the existing extraction; choose a fresh output: {output}')
        output = result_folders(output)
        write_json(output/'technical/config.json', config)
        from .trajectory_projection import project, reuse_extraction
        examples = reuse_extraction(config, output) if config['reuse_extraction'] else prepare(config, output)
        projection = project(config, output, examples)
        if stage == 'prepare':
            return
    from .trajectory_plots import render
    render(output, examples, projection)
    snapshot_metric_docs(output, 'forecast_spatial_mixture')
    rows = []
    for e in examples:
        with np.load(output/'technical'/f"{e['category']}.npz") as a:
            for t, pc, truth, margin in zip(a['time_ps'], a['true_map'], a['physical_crystal'], a['observed_margin']):
                rows.append(dict(case=e['category'], source_index=e['source']['source_index'], atom_id=e['atom_id'],
                    time_from_origin_ps=float(t), umap1=float(pc[0]), umap2=float(pc[1]),
                    ptm_crystal=int(truth), observed_crystal_margin=float(margin)))
    export_rows(output/'tables/observed-paths.csv', rows)
    write_json(output/'technical/status.json', dict(state='complete', examples=len(examples)))
    print(f'Completed structure / embedding gallery: {output}/index.html', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('all', 'prepare', 'plots'), default='all')
    args = parser.parse_args()
    run(load_json(args.config), args.stage)


if __name__ == '__main__':
    main()
