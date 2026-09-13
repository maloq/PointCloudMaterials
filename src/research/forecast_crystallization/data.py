"""Pool existing embeddings and align independently measured crystal progress."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.registry import sha256, write_json
from src.training_methods.embedding_forecast.data import verify_cache


def persistent_onset(above, persistence):
    """First frame of the first fully observed persistent run; None means censored."""
    for end in range(persistence - 1, len(above)):
        if above[end - persistence + 1:end + 1].all():
            return end - persistence + 1
    return None


def prepare(config):
    root = result_folders(config['output']) / 'technical'
    if (root / 'observations.json').exists():
        raise FileExistsError(f'Physical observations already prepared: {root}')
    torch.set_num_threads(config['cpu_threads'])
    cache = Path(config['cache'])
    manifest = verify_cache(cache)
    sources = json.loads(Path(config['sources_config']).read_text())['sources']
    records = manifest['shards']
    pooled = np.empty((len(records), records[0]['frames'], manifest['embedding_dim']), dtype=np.float32)
    fractions = np.empty(pooled.shape[:2], dtype=np.float32)
    clusters = np.empty(pooled.shape[:2], dtype=np.int64)
    metadata = []
    for index, record in enumerate(records):
        source = sources[record['source_index']]
        if source['name'] != record['name'] or source['preparation_seed'] != record['preparation_seed']:
            raise ValueError(f'Embedding/physical source identity differs: {record["name"]}')
        outcome_path = Path(source['outcome'])
        outcome = json.loads(outcome_path.read_text())
        progress_path = outcome_path.parent / 'crystallization_progress.npz'
        if sha256(progress_path) != outcome['progress_artifact']['sha256']:
            raise ValueError(f'Physical-label checksum differs from the producer: {progress_path}')
        frames = np.load(cache / record['directory'] / 'frames.npy')
        with np.load(progress_path) as progress:
            np.testing.assert_array_equal(progress['time_ps'][frames], frames * manifest['cadence_ps'])
            fractions[index] = progress['crystalline_fraction'][frames]
            clusters[index] = progress['largest_crystalline_cluster_atoms'][frames]
        onset = persistent_onset(clusters[index] >= config['nucleus_atoms'], config['persistence_frames'])
        observed_time = None if onset is None else onset * manifest['cadence_ps']
        if observed_time != outcome['nucleation_onset_time_ps']:
            raise ValueError(f'Recomputed onset disagrees with campaign outcome: {outcome_path}')
        growth = persistent_onset(fractions[index] >= config['growth_fraction'], config['persistence_frames'])
        values = np.load(cache / record['directory'] / 'embeddings.npy', mmap_mode='r')
        z = torch.from_numpy(np.array(values, copy=True)).to(config['device']).float()
        pooled[index] = z.mean(dim=0).cpu().numpy()
        del z
        metadata.append(dict(index=index, source_index=record['source_index'], name=record['name'],
            split=record['split'], temperature_K=record['temperature_K'], centers=record['centers'],
            onset_frame=onset, growth_frame=growth, progress_sha256=sha256(progress_path),
            outcome_sha256=sha256(outcome_path), embedding_directory=record['directory']))
        print(f'Prepared {index + 1}/{len(records)} sources; {record["name"]}; onset_ps={observed_time}', flush=True)
    np.save(root / 'observed_mean_embeddings.npy', pooled)
    np.savez(root / 'physical_progress.npz', crystalline_fraction=fractions, largest_cluster_atoms=clusters)
    checkpoints = root / 'checkpoints'
    checkpoints.mkdir()
    frozen = []
    for model in config['models']:
        original = Path(model['checkpoint'])
        path = checkpoints / (model['name'] + '.pt')
        shutil.copy2(original, path)
        digest = sha256(path)
        if digest != sha256(original):
            raise RuntimeError(f'Checkpoint changed while snapshotting: {original}; use a new explicit attempt.')
        payload = torch.load(path, map_location='cpu', weights_only=False)
        if payload['cache_manifest_sha256'] != sha256(cache / 'manifest.json'):
            raise ValueError(f'Forecast checkpoint uses a different embedding cache: {original}')
        frozen.append(dict(name=model['name'], checkpoint=str(path), sha256=digest,
                           original=str(original), selected_epoch=payload['epoch'] + 1,
                           history_ps=payload['config']['history_ps'], variant=payload['variant']))
    write_json(root / 'observations.json', dict(sources=metadata, models=frozen,
        cadence_ps=manifest['cadence_ps'], frames=pooled.shape[1],
        cache_manifest_sha256=sha256(cache / 'manifest.json'),
        pooled_sha256=sha256(root / 'observed_mean_embeddings.npy'),
        physical_progress_sha256=sha256(root / 'physical_progress.npz')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    prepare(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
