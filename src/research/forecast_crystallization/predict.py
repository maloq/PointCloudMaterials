"""Fit train-only physical readouts and assess frozen forecasts on val/test sources."""

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import torch

from src.experiment_runner.registry import sha256, write_json
from src.training_methods.embedding_forecast.model import EmbeddingForecaster

EVENTS = ('nucleation', 'growth')
EVENT_FIELDS = ('onset_frame', 'growth_frame')


def labels(metadata, frames):
    target = np.zeros((len(metadata), frames, len(EVENTS)), dtype=bool)
    for source in metadata:
        for event, field in enumerate(EVENT_FIELDS):
            if source[field] is not None:
                target[source['index'], source[field]:, event] = True
    return target


def readout_scores(values, readouts):
    standardized = (values.astype(np.float64) - readouts['mean']) / readouts['scale']
    return expit(standardized @ np.array(readouts['coefficients']).T + readouts['intercepts'])


def fit_readouts(config, observations, features, fractions):
    target = labels(observations['sources'], observations['frames'])
    indices = {split: [s['index'] for s in observations['sources'] if s['split'] == split]
               for split in ('train', 'val', 'test')}
    x = {split: features[ids].reshape(-1, features.shape[-1]).astype(np.float64) for split, ids in indices.items()}
    y = {split: target[ids].reshape(-1, len(EVENTS)) for split, ids in indices.items()}
    scaler = StandardScaler().fit(x['train'])
    x = {split: scaler.transform(values) for split, values in x.items()}
    readouts = dict(mean=scaler.mean_.tolist(), scale=scaler.scale_.tolist(), coefficients=[], intercepts=[], selection={})
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for event, name in enumerate(EVENTS):
            candidates = []
            for c in config['logistic_C']:
                model = LogisticRegression(C=c, max_iter=4000, tol=1e-7).fit(x['train'], y['train'][:, event])
                loss = log_loss(y['val'][:, event], model.predict_proba(x['val'])[:, 1])
                candidates.append((loss, c, model))
            loss, c, model = min(candidates, key=lambda row: row[0])
            readouts['coefficients'].append(model.coef_[0].tolist())
            readouts['intercepts'].append(float(model.intercept_[0]))
            readouts['selection'][name] = dict(C=c, validation_log_loss=float(loss),
                candidates=[dict(C=v[1], validation_log_loss=float(v[0])) for v in candidates])
    candidates = []
    for alpha in config['ridge_alpha']:
        model = Ridge(alpha=alpha).fit(x['train'], fractions[indices['train']].reshape(-1))
        loss = mean_squared_error(fractions[indices['val']].reshape(-1), model.predict(x['val']))
        candidates.append((loss, alpha, model))
    loss, alpha, model = min(candidates, key=lambda row: row[0])
    readouts['fraction'] = dict(coefficient=model.coef_.tolist(), intercept=float(model.intercept_),
                                alpha=alpha, validation_mse=float(loss))
    diagnostics = {}
    for event, name in enumerate(EVENTS):
        p = expit(x['test'] @ np.array(readouts['coefficients'][event]) + readouts['intercepts'][event])
        actual = y['test'][:, event]
        near = np.zeros_like(target[:, :, event])
        for s in observations['sources']:
            onset = s[EVENT_FIELDS[event]]
            if onset is not None:
                near[s['index'], max(0, onset-12):onset+13] = True
        mask = near[indices['test']].reshape(-1)
        diagnostics[name] = dict(test_snapshot_accuracy=float(np.mean((p >= .5) == actual)),
            test_snapshot_auroc=float(roc_auc_score(actual, p)),
            test_snapshot_average_precision=float(average_precision_score(actual, p)),
            near_event_snapshot_accuracy=float(np.mean((p[mask] >= .5) == actual[mask])),
            near_event_snapshot_auroc=float(roc_auc_score(actual[mask], p[mask])))
    readouts['diagnostics'] = diagnostics
    return readouts


def anchors_for_source(source, frames, past, future):
    # Evaluate only origins before at least one event; censored sources retain all origins.
    origins = np.arange(past, frames-future)
    eligible = np.zeros(len(origins), dtype=bool)
    for field in EVENT_FIELDS:
        eligible |= True if source[field] is None else origins < source[field]
    return origins[eligible]


@torch.inference_mode()
def forecast_source(z, origins, model, mean, scale, batch_anchors):
    predictions = []
    steps = model.history_steps
    offsets = torch.arange(-steps+1, 1, device=z.device)
    for start in range(0, len(origins), batch_anchors):
        current = torch.from_numpy(origins[start:start+batch_anchors]).to(z.device)
        history = z[:, current[:, None]+offsets].permute(1, 0, 2, 3).reshape(-1, steps, z.shape[-1]).float()
        output = model((history-mean)/scale)['mean']
        pooled = output.reshape(len(current), len(z), model.output_steps, z.shape[-1]).mean(dim=1)
        predictions.append((pooled*scale+mean).cpu().numpy())
    return np.concatenate(predictions)


def predict(config):
    root = Path(config['output']) / 'technical'
    observations = json.loads((root / 'observations.json').read_text())
    if (root / 'prediction_status.json').exists():
        raise FileExistsError(f'Prediction attempt already recorded: {root}')
    features = np.load(root / 'observed_mean_embeddings.npy')
    with np.load(root / 'physical_progress.npz') as z:
        fractions, clusters = z['crystalline_fraction'], z['largest_cluster_atoms']
    with threadpool_limits(limits=config['cpu_threads']):
        readouts = fit_readouts(config, observations, features, fractions)
    write_json(root / 'readouts.json', readouts)
    print('Readouts selected on validation:', readouts['selection'], flush=True)
    print('Observed snapshot diagnostics:', readouts['diagnostics'], flush=True)
    past = round(config['anchor_history_ps']/config['cadence_ps'])
    future = round(config['horizons_ps'][-1]/config['cadence_ps'])
    output = root / 'scores'
    output.mkdir()
    selected = [s for s in observations['sources'] if s['split'] in ('val', 'test')]
    source_ids, origins, baseline = [], [], {name: [] for name in
        ('persistence', 'history_mean', 'linear_embedding', 'oracle_future', 'physical_persistence', 'physical_trend')}
    relative = np.arange(-past, 1, dtype=np.float64)
    relative -= relative.mean()
    times = np.arange(1, future+1)
    for source in selected:
        index = source['index']
        anchor = anchors_for_source(source, observations['frames'], past, future)
        source_ids.extend([index]*len(anchor)); origins.extend(anchor)
        history = features[index, anchor[:, None]+np.arange(-past, 1)]
        slope = (history * relative[None, :, None]).sum(axis=1) / np.square(relative).sum()
        future_features = dict(persistence=np.repeat(history[:, -1:, :], future, axis=1),
            history_mean=np.repeat(history.mean(axis=1, keepdims=True), future, axis=1),
            linear_embedding=history[:, -1:, :]+times[None, :, None]*slope[:, None, :],
            oracle_future=features[index, anchor[:, None]+times])
        for name, values in future_features.items():
            baseline[name].append(readout_scores(values, readouts))
        physical = np.stack((np.log1p(clusters[index]), fractions[index]), axis=-1)
        history = physical[anchor[:, None]+np.arange(-past, 1)]
        slope = (history * relative[None, :, None]).sum(axis=1) / np.square(relative).sum()
        baseline['physical_persistence'].append(np.repeat(history[:, -1:, :], future, axis=1))
        baseline['physical_trend'].append(history[:, -1:, :]+times[None, :, None]*slope[:, None, :])
    np.savez(output / 'identities.npz', source=np.array(source_ids, dtype=np.int64), anchor=np.array(origins, dtype=np.int64))
    for name, values in baseline.items():
        np.save(output / (name+'.npy'), np.concatenate(values))
    del baseline
    torch.set_num_threads(config['cpu_threads'])
    for spec in observations['models']:
        if sha256(Path(spec['checkpoint'])) != spec['sha256']:
            raise ValueError(f'Frozen forecast checkpoint changed: {spec["checkpoint"]}')
        payload = torch.load(spec['checkpoint'], map_location='cpu', weights_only=False)
        if payload['config']['history_ps'] != config['anchor_history_ps']:
            raise ValueError('Physical assay uses the declared common history length.')
        model = EmbeddingForecaster(features.shape[-1], past+1, config['cadence_ps'], config['horizons_ps'], payload['variant']).to(config['device'])
        model.load_state_dict(payload['model'], strict=True); model.eval()
        mean, scale = payload['mean'].to(config['device']), payload['scale'].to(config['device'])
        scores = []
        for i, source in enumerate(selected):
            anchor = anchors_for_source(source, observations['frames'], past, future)
            values = np.load(Path(config['cache']) / source['embedding_directory'] / 'embeddings.npy', mmap_mode='r')
            z = torch.from_numpy(np.array(values, copy=True)).to(config['device'])
            prediction = forecast_source(z, anchor, model, mean, scale, config['anchors_per_batch'])
            scores.append(readout_scores(prediction, readouts))
            del z
            print(f'{spec["name"]}: {i+1}/{len(selected)} sources; {source["split"]}; {len(anchor)} origins', flush=True)
        np.save(output / (spec['name']+'.npy'), np.concatenate(scores))
        del model, payload
    write_json(root / 'prediction_status.json', dict(state='complete', origins=len(origins), methods=8,
        readouts_sha256=sha256(root / 'readouts.json'), score_files={p.name: sha256(p) for p in output.iterdir()}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    predict(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
