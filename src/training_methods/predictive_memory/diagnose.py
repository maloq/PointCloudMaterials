"""Audit state use and fit train-only linear physical-prediction controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.predictive_memory.prepare import file_hash, write_json
from src.data.predictive_memory.windows import MemoryDataset
from src.experiment_runner.metric_docs import write_metric_table
from src.project_runtime.paths import load_json
from .compare import fit_names
from .objective import PathHeads, fit_scaler, mean_path_scores, physical_scores
from .train import bootstrap_sources


RIDGE_PENALTIES = (.001, .01, .1, 1., 10., 100., 1000., 10000.)


def ridge_readout(features, targets, train, validation, penalties=RIDGE_PENALTIES):
    """SVD ridge: train-only scaling/intercept; select penalty on validation MSE."""
    x, y = np.asarray(features, dtype=np.float64), np.asarray(targets, dtype=np.float64)
    center = x[train].mean(0)
    scale = np.maximum(x[train].std(0), 1e-8)
    x = (x-center)/scale
    y_center = y[train].mean(0)
    u, singular, vt = np.linalg.svd(x[train], full_matrices=False)
    projected = u.T@(y[train]-y_center)
    candidates = []
    for alpha in penalties:
        if alpha <= 0:
            raise ValueError('Diagnostic ridge penalties must be positive')
        coefficient = vt.T@((singular/(singular**2+alpha))[:, None]*projected)
        prediction = x[validation]@coefficient+y_center
        candidates.append((float(np.mean((prediction-y[validation])**2)), alpha, coefficient))
    error, alpha, coefficient = min(candidates, key=lambda value: value[0])
    state = dict(center=torch.from_numpy(center), scale=torch.from_numpy(scale),
                 intercept=torch.from_numpy(y_center), coefficient=torch.from_numpy(coefficient))
    selection = dict(alpha=alpha, validation_mse=error,
                     penalty_validation_mse={str(a): e for e, a, _ in candidates})
    return torch.from_numpy(x@coefficient+y_center), state, selection


def verify_export(dataset, exported, split):
    expected = dataset.indices[split]
    if exported['indices'] != expected or len(exported['embeddings']) != len(expected):
        raise ValueError(f'Exported {split} embeddings must match immutable release row indices')
    keys = lambda row: (row['source_id'], row['center_id'], row['anchor'])
    if [keys(row) for row in exported['rows']] != [keys(dataset.rows[i]) for i in expected]:
        raise ValueError(f'Exported {split} source/center/anchor tuples do not match release')


@torch.no_grad()
def state_interventions(head, embeddings, training_embeddings, condition, present, future):
    """Keep the head and temperature fixed; remove sample-specific state information."""
    actual = physical_scores(head(embeddings, condition), present, future)
    constant = training_embeddings.mean(0, keepdim=True).expand_as(embeddings)
    replaced = physical_scores(head(constant, condition), present, future)
    scores = {}
    for key in ('joint_nll', 'future_mse', 'present_mse'):
        scores[key] = actual[key]
        scores[f'mean_state_{key}'] = replaced[key]
        scores[f'mean_state_{key}_increase'] = replaced[key]-actual[key]
    return scores


def diagnose(config, modalities=('x', 'xv')):
    torch.set_num_threads(2)
    root = Path(config['output'])
    destination = root/'diagnostics'
    technical = destination/'technical'
    technical.mkdir(parents=True, exist_ok=True)
    if (technical/'status.json').exists():
        raise FileExistsError(f'Retain completed diagnostics; output already exists: {destination}')
    dataset = MemoryDataset(config, 0.)  # Targets only: never assemble atomic inputs here.
    present, future, condition = dataset.targets(list(range(len(dataset.rows))), 'cpu')
    train = dataset.indices['train']
    center, scale = fit_scaler(present[train], future[train])
    present, future = (present-center)/scale, (future-center)/scale
    validation = [i for i in dataset.indices['val'] if dataset.rows[i]['anchor'] == config['anchor_frames'][1]]
    sources = np.array([row['source_id'] for row in dataset.rows])
    retained = dict(indices=dataset.indices, normalizer=(center, scale), readouts={}, interventions={})
    metrics = dict(baselines={}, models={})
    provenance = dict(release_sha256=dataset.release_sha256, fits={})

    def summarize(scores, indices):
        return {key: bootstrap_sources(value.detach().numpy(), sources[indices],
                    config['bootstrap_draws'], config['seed']) for key, value in scores.items()}

    def readout(name, features, targets, *, current=False):
        predictions, state, selection = ridge_readout(features, targets.flatten(1), train, validation)
        predictions = predictions.reshape(targets.shape)
        result = dict(selection=selection)
        for split, indices in dataset.indices.items():
            scores = dict(present_mse=(predictions[indices]-targets[indices]).square().mean(1)) if current else \
                mean_path_scores(predictions[indices], targets[indices])
            result[split] = summarize(scores, indices)
        retained['readouts'][name] = dict(state=state, predictions=predictions, selection=selection)
        return result

    metrics['baselines']['temperature_ridge'] = readout('temperature_ridge', condition, future)
    # This baseline explicitly observes both positions and measured velocities.
    metrics['baselines']['current_packet_ridge'] = readout('current_packet_ridge',
        torch.cat((present, condition), 1), future)
    baseline_prediction = retained['readouts']['temperature_ridge']['predictions']
    packet_prediction = retained['readouts']['current_packet_ridge']['predictions']
    test = dataset.indices['test']
    gain = (baseline_prediction[test]-future[test]).square().mean((1, 2)) \
        -(packet_prediction[test]-future[test]).square().mean((1, 2))
    metrics['paired_test_packet_gain_over_temperature'] = summarize(dict(future_mse_gain=gain), test)

    for name in fit_names(modalities):
        path = root/name/'technical'
        status = json.loads((path/'status.json').read_text())
        if status['state'] != 'complete' or status['step'] != config['training']['steps']:
            raise ValueError(f'Diagnostics require completed declared budget: {name}: {status}')
        checkpoint = torch.load(path/'best.pt', weights_only=False, map_location='cpu')
        if checkpoint['release_sha256'] != dataset.release_sha256 or checkpoint['config'] != config:
            raise ValueError(f'Checkpoint configuration or release mismatch: {name}')
        for actual, expected in zip(checkpoint['normalizer'], (center, scale), strict=True):
            # Old normalizers were computed on CUDA; allow reduction-order roundoff.
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        exports = torch.load(path/'evaluation.pt', weights_only=True, map_location='cpu')
        z = torch.empty(len(dataset.rows), config['encoder']['output_dim'])
        for split, exported in exports.items():
            verify_export(dataset, exported, split)
            z[exported['indices']] = exported['embeddings']
        if set(exports) != set(dataset.indices):
            raise ValueError(f'Missing or extra exported splits: {name}')
        head = PathHeads(config['encoder']['output_dim'], len(config['future_lags_ps']), **config['mixture'])
        head.load_state_dict({key.removeprefix('heads.'): value for key, value in checkpoint['model'].items()
                              if key.startswith('heads.')})
        head.eval()
        result = dict(selected_step=checkpoint['step'], interventions={})
        retained['interventions'][name] = {}
        for split in ('val', 'test'):
            indices = dataset.indices[split]
            scores = state_interventions(head, z[indices], z[train], condition[indices],
                                          present[indices], future[indices])
            saved = torch.tensor([row['joint_nll'] for row in exports[split]['rows']])
            torch.testing.assert_close(scores['joint_nll'], saved, atol=2e-5, rtol=2e-5,
                                       msg=f'Recomputed {name}/{split} NLL differs from frozen evaluation')
            result['interventions'][split] = summarize(scores, indices)
            retained['interventions'][name][split] = scores
        result['embedding_future_ridge'] = readout(name+'/future', torch.cat((z, condition), 1), future)
        result['embedding_present_ridge'] = readout(name+'/present', z, present, current=True)
        prediction = retained['readouts'][name+'/future']['predictions']
        gains = (baseline_prediction[test]-future[test]).square().mean((1, 2)) \
            -(prediction[test]-future[test]).square().mean((1, 2))
        result['paired_test_ridge_gain_over_temperature'] = summarize(dict(future_mse_gain=gains), test)
        metrics['models'][name] = result
        provenance['fits'][name] = {file: file_hash(path/file) for file in ('best.pt', 'evaluation.pt')}
        print(f"Diagnosed {name}: val constant-state NLL increase "
              f"{result['interventions']['val']['mean_state_joint_nll_increase']['mean']:.6g}", flush=True)
    torch.save(retained, technical/'diagnostics.pt')
    write_json(technical/'provenance.json', provenance)
    write_json(technical/'metrics.json', metrics)
    write_metric_table(metrics, destination, family='predictive_memory')
    lines = ['# State-use and physical-readout diagnostics', '',
        'Replacing the embedding with its training mean keeps the learned head and temperature fixed. '
        'Positive NLL increase means removing the state hurts. This intervention is not a retrained '
        'condition-only model or a mutual-information estimate.', '',
        '| Model | Validation NLL increase | Test NLL increase | Test embedding-ridge future MSE |',
        '|---|---:|---:|---:|']
    for name, result in metrics['models'].items():
        lines.append(f"| {name} | {result['interventions']['val']['mean_state_joint_nll_increase']['mean']:.5f} "
                     f"| {result['interventions']['test']['mean_state_joint_nll_increase']['mean']:.5f} "
                     f"| {result['embedding_future_ridge']['test']['future_mse']['mean']:.5f} |")
    lines += ['', '| Physical baseline | Validation future MSE | Test future MSE |', '|---|---:|---:|']
    for name, result in metrics['baselines'].items():
        lines.append(f"| {name} | {result['val']['future_mse']['mean']:.5f} | {result['test']['future_mse']['mean']:.5f} |")
    lines += ['', 'Ridge penalties use only the middle validation anchor; scaling and coefficients use only '
        'training sources. The current-packet baseline includes measured velocities and is observation-matched '
        'only to xv, not x. No history-packet baseline is fitted here. All test sources were previously examined; '
        'this is exploratory. Source intervals and per-block/per-lag errors are retained in tables and JSON. '
        'The saved linear readouts are diagnostic predictors, not replacement encoders.']
    (destination/'README.md').write_text('\n'.join(lines)+'\n')
    write_json(technical/'status.json', dict(state='complete', fits=len(metrics['models'])))
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--modalities', nargs='+', choices=('x', 'xv'), default=['x', 'xv'])
    args = parser.parse_args()
    diagnose(load_json(args.config), modalities=args.modalities)
