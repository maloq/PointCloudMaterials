"""Fit a local crystal readout and project frozen embedding forecasts through it."""

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from src.experiment_runner.registry import sha256, write_json
from src.research.smooth_temporal_encoder.evaluate import structure_fit
from src.training_methods.embedding_forecast.model import EmbeddingForecaster


def load_embeddings(config, record):
    root = Path(config['output']) / 'technical'
    rows = np.load(root / 'labels' / record['directory'] / 'embedding_rows.npy')
    values = np.load(Path(config['cache']) / record['directory'] / 'embeddings.npy', mmap_mode='r')
    return np.array(values[rows], copy=True)


def load_labels(root, record):
    path = root / 'labels' / record['directory'] / 'labels.npy'
    if sha256(path) != record['labels_sha256']:
        raise ValueError(f'Local PTM labels changed: {path}')
    return np.isin(np.load(path), [1, 2, 3])


@torch.inference_mode()
def forecast_scores(z, anchors, model, mean, scale, weight, bias, batch_size):
    """Return [center, origin, future] readout margins; no true futures enter the model."""
    device = z.device
    offsets = torch.arange(1-model.history_steps, 1, device=device)
    a = torch.as_tensor(anchors, device=device)
    projected_weight = scale.double() * weight
    projected_bias = mean.double() @ weight + bias
    output = np.empty((len(z)*len(anchors), model.output_steps), dtype=np.float32)
    for start in range(0, len(output), batch_size):
        flat = torch.arange(start, min(start+batch_size, len(output)), device=device)
        centers, origin = flat // len(anchors), a[flat % len(anchors)]
        history = z[centers[:, None], origin[:, None]+offsets].float()
        predicted = model((history-mean)/scale)['mean']
        output[start:start+len(flat)] = (predicted.double() @ projected_weight + projected_bias).cpu().numpy()
    return output.reshape(len(z), len(anchors), model.output_steps)


def predict(config):
    torch.set_num_threads(1)
    root = Path(config['output']) / 'technical'
    observations = json.loads((root / 'local_observations.json').read_text())
    if (root / 'local_prediction_status.json').exists():
        raise FileExistsError(f'Local prediction attempt already completed: {root}')
    write_json(root / 'prediction_config.json', config)
    sources = observations['sources']
    stride = config['probe_frame_stride']
    features, labels = {}, {}
    for split in ('train', 'val', 'test'):
        chosen = [s for s in sources if s['split'] == split]
        features[split] = torch.from_numpy(np.concatenate([
            load_embeddings(config, s)[:, ::stride].reshape(-1, 256) for s in chosen
        ])).to(config['device']).float()
        labels[split] = np.concatenate([load_labels(root, s)[:, ::stride].reshape(-1) for s in chosen]).astype(np.int64)
    metrics, probe, _ = structure_fit(features['train'], labels['train'], features['val'], labels['val'],
                                      features['test'], labels['test'])
    np.testing.assert_array_equal(probe['classes'].numpy(), [0, 1])
    torch.save(probe, root / 'local_crystal_probe.pt')
    write_json(root / 'probe_fit.json', dict(**metrics, frame_stride=stride,
        examples={k: len(v) for k, v in labels.items()},
        positives={k: int(v.sum()) for k, v in labels.items()},
        protocol='Class-balanced ridge; alpha selected on validation macro F1; fixed decision margin zero.'))
    print('Local crystal probe:', metrics, flush=True)
    del features, labels
    difference = probe['coefficients'][:, 1] - probe['coefficients'][:, 0]
    weight = (difference[:-1] / probe['std'].double()).to(config['device'])
    bias = (difference[-1] - probe['mean'].double() @ (difference[:-1]/probe['std'].double())).to(config['device'])
    observed_scores = []
    physical_labels = []
    for source in sources:
        z = torch.from_numpy(load_embeddings(config, source)).to(config['device'])
        observed_scores.append((z.double() @ weight + bias).cpu().numpy().astype(np.float32))
        physical_labels.append(load_labels(root, source))
    np.save(root / 'observed_scores.npy', np.stack(observed_scores))
    np.save(root / 'physical_crystal_labels.npy', np.stack(physical_labels))
    past = round(config['anchor_history_ps']/config['cadence_ps'])
    future = round(config['horizons_ps'][-1]/config['cadence_ps'])
    # Reserve confirmation frames even for the longest persistence sensitivity.
    anchors = np.arange(past, observations['frames']-future-max(config['sensitivity_persistence_frames'])+1)
    np.save(root / 'anchors.npy', anchors)
    selected = [s for s in sources if s['split'] in ('val', 'test')]
    checkpoint_root = root / 'checkpoints'
    checkpoint_root.mkdir()
    frozen = []
    for spec in config['models']:
        original = Path(spec['checkpoint'])
        checkpoint = checkpoint_root / (spec['name'] + '.pt')
        shutil.copy2(original, checkpoint)
        digest = sha256(checkpoint)
        if digest != sha256(original):
            raise RuntimeError(f'Checkpoint changed during snapshot: {original}')
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if payload['cache_manifest_sha256'] != observations['cache_manifest_sha256']:
            raise ValueError(f'Checkpoint/cache mismatch: {original}')
        model_path = Path(__file__).resolve().parents[2] / 'training_methods/embedding_forecast/model.py'
        if sha256(model_path) != payload['implementation_sha256']['model.py']:
            raise ValueError(f'Forecast model implementation differs from checkpoint: {original}')
        if payload['config']['history_ps'] != config['anchor_history_ps']:
            raise ValueError(f'Checkpoint history differs from assay: {original}')
        model = EmbeddingForecaster(256, past+1, config['cadence_ps'], config['horizons_ps'], payload['variant']).to(config['device'])
        model.load_state_dict(payload['model'], strict=True)
        model.eval()
        mean, scale = payload['mean'].to(config['device']), payload['scale'].to(config['device'])
        directory = root / 'scores' / spec['name']
        directory.mkdir(parents=True)
        hashes = {}
        started = time.monotonic()
        for index, source in enumerate(selected):
            z = torch.from_numpy(load_embeddings(config, source)).to(config['device'])
            scores = forecast_scores(z, anchors, model, mean, scale, weight, bias, config['inference_batch_size'])
            path = directory / f'source_{source["source_index"]:03d}.npy'
            np.save(path, scores)
            hashes[path.name] = sha256(path)
            print(f'{spec["name"]}: {index+1}/{len(selected)} sources; {time.monotonic()-started:.1f} s', flush=True)
        frozen.append(dict(name=spec['name'], checkpoint=str(checkpoint), sha256=digest,
            selected_epoch=payload['epoch']+1, original=str(original), score_sha256=hashes,
            parameters=sum(p.numel() for p in model.parameters()), variant=payload['variant']))
        del model, payload
    write_json(root / 'local_prediction_status.json', dict(state='complete', models=frozen,
        origins=len(anchors), centers_per_source=config['centers_per_source'],
        probe_sha256=sha256(root / 'local_crystal_probe.pt'),
        observed_scores_sha256=sha256(root / 'observed_scores.npy'),
        physical_labels_sha256=sha256(root / 'physical_crystal_labels.npy')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    predict(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
