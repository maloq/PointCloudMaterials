"""Project completed context/mixture forecasts into the existing local PTM assay."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json
from src.research.forecast_crystallization.local_metrics import (
    classification, counts, first_sustained_onset, onset_metrics, risk_windows,
    select_threshold, source_bootstrap,
)
from src.training_methods.embedding_forecast.context_mixture import (
    build_forecaster, call_forecaster, crystal_probability,
)
from src.training_methods.embedding_forecast.spatial import pool_neighbors


def assay(plan):
    root = Path(plan['local_assay'])
    observations = json.loads((root/'local_observations.json').read_text())
    prior = json.loads((root/'local_prediction_status.json').read_text())
    for name, key in [('local_crystal_probe.pt', 'probe_sha256'),
                      ('observed_scores.npy', 'observed_scores_sha256'),
                      ('physical_crystal_labels.npy', 'physical_labels_sha256')]:
        if sha256(root/name) != prior[key]:
            raise ValueError(f'Frozen local assay artifact changed: {root/name}')
    chosen = [i for i, s in enumerate(observations['sources']) if s['split'] in ('val', 'test')]
    sources = [observations['sources'][i] for i in chosen]
    past = round(plan['anchor_history_ps']/plan['cadence_ps'])
    future = round(plan['horizons_ps'][-1]/plan['cadence_ps'])
    anchors = np.arange(past, observations['frames']-future-max(plan['persistence_frames'])+1)
    return dict(root=root, observations=observations, sources=sources, anchors=anchors,
        observed=np.load(root/'observed_scores.npy')[chosen],
        crystal=np.load(root/'physical_crystal_labels.npy')[chosen])


@torch.inference_mode()
def score_source(z, spatial, radii, anchors, model, mean, scale, weight, bias, batch_size):
    """Keep identical center/origin identities for both point and distribution readouts."""
    device = z.device
    offsets = torch.arange(1-model.history_steps, 1, device=device)
    anchor = torch.as_tensor(anchors, device=device)
    projected_weight = scale.double()*weight
    projected_bias = mean.double()@weight+bias
    shape = (len(z)*len(anchors), model.output_steps)
    scores = {'mean_margin': np.empty(shape, dtype=np.float32)}
    probabilistic = model.distribution == 'trajectory_mixture'
    if probabilistic:
        scores['frame_crystal_probability'] = np.empty(shape, dtype=np.float32)
    for start in range(0, shape[0], batch_size):
        flat = torch.arange(start, min(start+batch_size, shape[0]), device=device)
        center = flat//len(anchors)
        columns = anchor[flat % len(anchors), None]+offsets
        history = z[center[:, None], columns].float()
        batch = {}
        if spatial is not None:
            batch = dict(spatial=spatial[center[:, None], columns].float(),
                         spatial_radii_A=radii[center[:, None], columns])
        prediction = call_forecaster(model, (history-mean)/scale, batch, mean, scale)
        scores['mean_margin'][start:start+len(flat)] = (
            prediction['mean'].double()@projected_weight+projected_bias).cpu().numpy()
        if probabilistic:
            scores['frame_crystal_probability'][start:start+len(flat)] = crystal_probability(
                prediction, projected_weight, projected_bias).cpu().numpy()
    return {key: value.reshape(len(z), len(anchors), model.output_steps) for key, value in scores.items()}


def directory(plan, run):
    if 'local_directory' in run:
        return Path(run['local_directory'])
    return Path(plan['output'])/'technical/local'/f"{run['name']}-seed{run['seed']}"


def predict(plan, run):
    torch.set_num_threads(2)
    data = assay(plan)
    root = directory(plan, run)
    root.mkdir(parents=True)
    fit = Path(run['fit'])/'technical'
    if json.loads((fit/'status.json').read_text())['state'] != 'complete':
        raise ValueError(f'Local inference requires a completed fit: {fit}')
    checkpoint = fit/'best.pt'
    payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if payload['cache_manifest_sha256'] != data['observations']['cache_manifest_sha256']:
        raise ValueError(f'Checkpoint and local physical assay use different embeddings: {checkpoint}')
    for name in ('model.py', 'context_mixture.py', 'spatial.py'):
        if name in payload['implementation_sha256']:
            current = Path(__file__).resolve().parents[2]/'training_methods/embedding_forecast'/name
            if sha256(current) != payload['implementation_sha256'][name]:
                raise ValueError(f'Forecast implementation changed since fitting: {current}')
    device = plan['device']
    past = round(payload['config']['history_ps']/plan['cadence_ps'])
    model = build_forecaster(256, past+1, plan['cadence_ps'], plan['horizons_ps'], payload['variant']).to(device)
    model.load_state_dict(payload['model'], strict=True); model.eval()
    mean, scale = payload['mean'].to(device), payload['scale'].to(device)
    probe = torch.load(data['root']/'local_crystal_probe.pt', weights_only=False)
    difference = probe['coefficients'][:, 1]-probe['coefficients'][:, 0]
    weight = (difference[:-1]/probe['std'].double()).to(device)
    bias = (difference[-1]-probe['mean'].double()@(difference[:-1]/probe['std'].double())).to(device)
    spatial_model = payload['variant'].get('spatial_neighbors', 0) > 0
    if spatial_model:
        spatial_manifest = json.loads((Path(plan['spatial_cache'])/'manifest.json').read_text())
        if spatial_manifest['base_cache_manifest_sha256'] != payload['cache_manifest_sha256']:
            raise ValueError('Physical inference spatial cache differs from training embedding identities.')
        spatial_records = {r['directory']: r for r in spatial_manifest['records']}
    np.save(root/'anchors.npy', data['anchors'])
    hashes = {}; started = time.monotonic()
    for i, source in enumerate(data['sources']):
        cache = Path(plan['embedding_cache'])/source['directory']
        label_dir = data['root']/'labels'/source['directory']
        rows = np.load(label_dir/'embedding_rows.npy')
        np.testing.assert_array_equal(np.load(cache/'atom_ids.npy')[rows], np.load(label_dir/'atom_ids.npy'))
        values = np.load(cache/'embeddings.npy', mmap_mode='r')
        spatial, radii = None, None
        if spatial_model:
            spatial_dir = Path(plan['spatial_cache'])/source['directory']
            for name, digest in spatial_records[source['directory']]['checksums'].items():
                if sha256(spatial_dir/name) != digest:
                    raise ValueError(f'Spatial inference input changed: {spatial_dir/name}')
            all_z = torch.from_numpy(np.array(values, copy=True)).to(device)
            neighbors = torch.from_numpy(np.load(spatial_dir/'neighbors.npy')[rows].astype(np.int64)).to(device)
            spatial = pool_neighbors(all_z, neighbors)
            radii = torch.from_numpy(np.load(spatial_dir/'radii_A.npy')[rows]).to(device)
            z = all_z[torch.from_numpy(rows).to(device)]
            del all_z, neighbors
        else:
            z = torch.from_numpy(np.array(values[rows], copy=True)).to(device)
        scores = score_source(z, spatial, radii, data['anchors'], model, mean, scale, weight, bias,
                              plan['inference_batch_size'])
        path = root/f"source_{source['source_index']:03d}.npz"
        np.savez_compressed(path, **scores); hashes[path.name] = sha256(path)
        print(f"Local {run['name']} seed {run['seed']}: {i+1}/{len(data['sources'])} sources, "
              f'{time.monotonic()-started:.1f}s', flush=True)
    write_json(root/'prediction.json', dict(state='complete', run=run, checkpoint_sha256=sha256(checkpoint),
        cache_manifest_sha256=payload['cache_manifest_sha256'],
        probe_sha256=sha256(data['root']/'local_crystal_probe.pt'),
        scores_sha256=hashes, selected_epoch=payload['epoch']+1,
        parameters=sum(p.numel() for p in model.parameters()),
        score_types=list(scores), anchors_sha256=sha256(root/'anchors.npy')))


def assess(scores, data, plan, score_type):
    """Use validation thresholds; count a local first sustained PTM onset prospectively."""
    anchors, crystal = data['anchors'], data['crystal']
    sources = data['sources']; cadence = plan['cadence_ps']
    val = np.array([s['split'] == 'val' for s in sources]); test = ~val
    source_ids = np.array([s['source_index'] for s in sources])
    source_grid = np.broadcast_to(source_ids[:, None, None], scores.shape[:-1])
    onsets = {p: first_sustained_onset(crystal, p) for p in plan['persistence_frames']}
    event_rows, state_rows, fixed_rows, statistics = [], [], [], {}
    for horizon in plan['horizons_ps']:
        steps = round(horizon/cadence)
        state_score = scores[..., steps-1]
        truth = crystal[..., anchors+steps]
        threshold, val_f1 = select_threshold(truth[val].ravel(), state_score[val].ravel())
        metric = classification(truth[test].ravel(), state_score[test].ravel(), threshold)
        per_source = [counts(truth[s].ravel(), state_score[s].ravel() >= threshold) for s in np.flatnonzero(test)]
        metric['source_bootstrap'] = source_bootstrap(per_source, plan['bootstrap_repetitions'], plan['bootstrap_seed'])
        state_rows.append(dict(score_type=score_type, horizon_ps=horizon, threshold=threshold,
                               validation_f1=val_f1, **metric))
        for persistence, onset in onsets.items():
            risk = risk_windows(crystal, onset, anchors, plan['negative_history_frames'])
            actual = onset[..., None] <= anchors+steps
            val_mask = risk & val[:, None, None]; test_mask = risk & test[:, None, None]
            event_score = scores[..., :steps].max(axis=-1)
            threshold, val_f1 = select_threshold(actual[val_mask], event_score[val_mask])
            delay = (onset[..., None]-anchors)*cadence
            metric, per_source = onset_metrics(actual[test_mask], scores[..., :steps][test_mask], threshold,
                delay[test_mask], cadence, source_grid[test_mask], source_ids[test],
                plan['bootstrap_repetitions'], plan['bootstrap_seed'])
            event_rows.append(dict(score_type=score_type, horizon_ps=horizon, persistence_frames=persistence,
                threshold=threshold, validation_f1=val_f1, **metric))
            statistics[f'{score_type}_{horizon}ps_p{persistence}'] = per_source
            source, center = np.nonzero(test[:, None] & (onset-steps >= anchors[0]) & (onset-steps <= anchors[-1]))
            origin = onset[source, center]-steps-anchors[0]
            eligible = risk[source, center, origin]
            source, center, origin = source[eligible], center[eligible], origin[eligible]
            metric, _ = onset_metrics(np.ones(len(source), dtype=bool), scores[source, center, origin, :steps],
                threshold, np.full(len(source), horizon), cadence, source_ids[source], source_ids[test],
                plan['bootstrap_repetitions'], plan['bootstrap_seed'])
            fixed_rows.append(dict(score_type=score_type, horizon_ps=horizon, persistence_frames=persistence, **metric))
    return dict(state=state_rows, onset=event_rows, fixed_lead=fixed_rows), statistics


def analyze(plan, run):
    data = assay(plan); root = directory(plan, run)
    prediction = json.loads((root/'prediction.json').read_text())
    if prediction['state'] != 'complete':
        raise ValueError(f'Local scores are incomplete: {root}')
    np.testing.assert_array_equal(np.load(root/'anchors.npy'), data['anchors'])
    arrays = []
    for source in data['sources']:
        name = f"source_{source['source_index']:03d}.npz"
        if sha256(root/name) != prediction['scores_sha256'][name]:
            raise ValueError(f'Local predicted scores changed: {root/name}')
        with np.load(root/name) as archive:
            arrays.append({key: archive[key] for key in archive.files})
    report = dict(run=run, prediction=prediction, protocol=plan, results={})
    clustered = {}
    for key in prediction['score_types']:
        result, stats = assess(np.stack([a[key] for a in arrays]), data, plan, key)
        report['results'][key] = result; clustered.update(stats)
    write_json(root/'results.json', report)
    np.savez(root/'source-statistics.npz', **clustered)
    write_json(root/'status.json', dict(state='complete'))
    print(f"Local analysis complete: {run['name']} seed {run['seed']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--stage', choices=('predict', 'analyze', 'all'), default='all')
    args = parser.parse_args(); plan = load_json(args.plan)
    selected = [r for r in plan['runs'] if r['name'] == args.run and r['seed'] == args.seed]
    if len(selected) != 1:
        raise ValueError(f'Expected exactly one planned run: {args.run}, seed {args.seed}')
    if args.stage in ('predict', 'all'):
        predict(plan, selected[0])
    if args.stage in ('analyze', 'all'):
        analyze(plan, selected[0])


if __name__ == '__main__':
    main()
