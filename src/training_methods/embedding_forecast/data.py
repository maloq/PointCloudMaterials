"""Encode the independent MEAM sources once; batch windows from mmap shards."""

import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.spatiotemporal_views import local_views
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json, resolve_path, resolve_config


def frame_steps(time_ps, cadence_ps):
    steps = int(round(time_ps / cadence_ps))
    if time_ps < 0 or not np.isclose(steps * cadence_ps, time_ps, rtol=0, atol=1e-9):
        raise ValueError(f"Time {time_ps} ps is not a nonnegative multiple of {cadence_ps} ps.")
    return steps


def source_records(config):
    """Use the existing denoising source selection and the independent-melt producer."""
    source_config = load_json(config['sources_config'])
    if source_config['protocol'] not in ('denoising80', 'embedding_forecast_sources'):
        raise ValueError('Forecast preparation expects a denoising80 or embedding_forecast_sources selection.')
    records = source_config['sources']
    seen_paths, seen_seeds = set(), set()
    campaigns = {}
    for source in records:
        path = str(resolve_path(source['path']).resolve())
        seed = source['preparation_seed']
        if path in seen_paths or seed in seen_seeds:
            raise ValueError(f"Duplicate trajectory or melt lineage: {source['name']}; split by whole source.")
        seen_paths.add(path)
        seen_seeds.add(seed)
        campaign_path = source['campaign_manifest']
        if campaign_path not in campaigns:
            campaigns[campaign_path] = json.loads(Path(campaign_path).read_text())
        campaign = campaigns[campaign_path]
        if campaign['campaign_type'] != 'independently_melted_boundary_parent_sources':
            raise ValueError(f'Unsupported lineage producer: {campaign_path}')
        produced = next(r for r in campaign['runs'] if r['run_id'] == source['name'])
        split = {'optimization': 'train', 'model_selection': 'val', 'final_validation': 'test'}
        if (split[produced['source_split']] != source['split'] or
                produced['preparation_seed'] != seed or
                (resolve_path(campaign_path).parent / produced['run_dir']).resolve() != Path(path).parent):
            raise ValueError(f"Source selection disagrees with campaign lineage: {source['name']}")
        if (source['cadence_ps'] != config['cadence_ps'] or
                campaign['protocol']['sample_interval_ps'] != config['cadence_ps'] or
                campaign['protocol']['timestep_fs'] / 1000 != source['timestep_ps']):
            raise ValueError(f"Physical timeline differs from forecast config: {source['name']}")
    return records


def load_snapshot_encoder(checkpoint, device):
    """Restore the actual original-VICReg snapshot encoder, excluding its projector."""
    from omegaconf import OmegaConf
    from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
    from src.utils.model_utils import load_model_from_checkpoint, resolve_config_path

    directory, name = resolve_config_path(str(resolve_path(checkpoint)))
    cfg = OmegaConf.load(Path(directory) / f'{name}.yaml')
    if cfg.encoder.name != 'PretrainedMACEGeometry':
        raise ValueError('Forecast targets require a single-frame PretrainedMACEGeometry checkpoint; '
                         'a temporal encoder would change the target definition and overlap future windows.')
    cfg.encoder.kwargs.performance.compile_radial_mlp = False
    cfg.encoder.kwargs.activation_checkpointing = False
    model = load_model_from_checkpoint(checkpoint, cfg, device=device, module=VICRegModule)
    model.eval().requires_grad_(False)
    return model.encoder, float(cfg.data.normalization_radius_A)


def verify_cache(root):
    root = resolve_path(root)
    migration = root / 'storage_migration.json'
    if migration.exists() and json.loads(migration.read_text())['state'] != 'complete':
        raise RuntimeError(f'Embedding storage migration must finish before use: {migration}')
    manifest = json.loads((root / 'manifest.json').read_text())
    if manifest['state'] != 'complete':
        raise ValueError(f'Incomplete embedding cache: {root}')
    assignments = {}
    for shard in manifest['shards']:
        lineage = shard['preparation_seed']
        if lineage in assignments and assignments[lineage] != shard['split']:
            raise ValueError(f'Forecast train/validation/test lineage leakage: melt seed {lineage}')
        assignments[lineage] = shard['split']
        directory = root / shard['directory']
        for filename, digest in shard['checksums'].items():
            if sha256(directory / filename) != digest:
                raise ValueError(f'Embedding artifact checksum changed: {directory / filename}')
    return manifest


def prepare_cache(config, device):
    config = resolve_config(config)
    storage_dtype = np.dtype(config.get('storage_dtype', 'float32'))
    if storage_dtype not in (np.dtype('float32'), np.dtype('float16')):
        raise ValueError(f'Unsupported embedding storage dtype: {storage_dtype}')
    records = source_records(config)
    root = Path(config['cache'])
    root.mkdir(parents=True, exist_ok=True)
    migration = root / 'storage_migration.json'
    if migration.exists() and json.loads(migration.read_text())['state'] != 'complete':
        raise RuntimeError(f'Embedding storage migration must finish before preparation: {migration}')
    checkpoint = Path(config['checkpoint'])
    from src.utils.model_utils import resolve_config_path
    directory, name = resolve_config_path(str(checkpoint))
    protocol = dict(config=config, checkpoint_sha256=sha256(checkpoint),
                    checkpoint_config_sha256=sha256(Path(directory) / f'{name}.yaml'),
                    sources_sha256=sha256(Path(config['sources_config'])),
                    producer_sha256=sha256(Path(__file__)),
                    source_manifests=[sha256(Path(s['path']) / 'manifest.json') for s in records],
                    campaign_manifests=[sha256(Path(s['campaign_manifest'])) for s in records])
    signature = root / 'protocol.json'
    if signature.exists():
        if json.loads(signature.read_text()) != protocol:
            raise ValueError(f'Embedding cache protocol changed: {root}; choose a new cache path.')
    else:
        write_json(signature, protocol)
    if (root / 'manifest.json').exists():
        return verify_cache(root)

    encoder, radius = load_snapshot_encoder(str(checkpoint), device)
    past = frame_steps(config['history_ps'], config['cadence_ps'])
    future = frame_steps(config['future_ps'], config['cadence_ps'])
    margin = frame_steps(config['anchor_margin_ps'], config['cadence_ps'])
    shards = []
    for source_index, source in enumerate(records):
        trajectory = ShootingBinaryTrajectory.load(source['path'])
        expected = config['cadence_ps'] / source['timestep_ps']
        if not np.all(np.diff(trajectory.timesteps) == expected):
            raise ValueError(f"Irregular or incorrect physical timeline: {source['path']}")
        centers = np.sort(np.random.default_rng(np.random.SeedSequence(
            [config['seed'], source['preparation_seed']])).choice(
                trajectory.atom_count, config['centers_per_source'], replace=False))
        previous_stop = -1
        for context, anchor in enumerate(source['anchors']):
            start, stop = anchor - past - margin, anchor + future + margin + 1
            if start < 0 or stop > trajectory.frame_count or start < previous_stop:
                raise ValueError(f"Invalid/overlapping segment [{start}, {stop}) in {source['name']}")
            previous_stop = stop
            shard_dir = root / f'source_{source_index:03d}_segment_{context:02d}'
            shard_dir.mkdir(exist_ok=True)
            completion = shard_dir / 'manifest.json'
            if completion.exists():
                shard = json.loads(completion.read_text())
                for filename, digest in shard['checksums'].items():
                    if sha256(shard_dir / filename) != digest:
                        raise ValueError(f'Changed completed embedding shard: {shard_dir / filename}')
                shards.append(shard)
                continue
            embeddings = np.lib.format.open_memmap(shard_dir / 'embeddings.npy', mode='w+',
                dtype=storage_dtype, shape=(len(centers), stop - start, 256))
            for column, frame in enumerate(range(start, stop)):
                low = trajectory.box_low[frame].astype(np.float64)
                lengths = trajectory.box_high[frame].astype(np.float64) - low
                points = np.mod(trajectory.positions[frame].astype(np.float64) - low, lengths)
                tree = cKDTree(points, boxsize=lengths)
                # Reselect the local neighborhood at each time; track only the center identity.
                # This never uses an anchor/future neighbor selection to construct past inputs.
                clouds = local_views(points, tree, lengths, centers, num_points=80, radius=radius)
                with torch.inference_mode():
                    for batch_start in range(0, len(centers), config['encoder_batch_size']):
                        batch = slice(batch_start, batch_start + config['encoder_batch_size'])
                        z = encoder(torch.from_numpy(clouds[batch]).to(device)).float()
                        stored = z.cpu().numpy().astype(storage_dtype)
                        if not np.isfinite(stored).all():
                            raise FloatingPointError(f"Nonfinite {storage_dtype} embedding: {source['name']}, frame {frame}")
                        embeddings[batch, column] = stored
            embeddings.flush()
            np.save(shard_dir / 'atom_ids.npy', trajectory.atom_ids[centers])
            np.save(shard_dir / 'frames.npy', np.arange(start, stop, dtype=np.int64))
            np.save(shard_dir / 'timesteps.npy', trajectory.timesteps[start:stop])
            np.save(shard_dir / 'time_ps.npy', trajectory.timesteps[start:stop] * source['timestep_ps'])
            files = ('embeddings.npy', 'atom_ids.npy', 'frames.npy', 'timesteps.npy', 'time_ps.npy')
            shard = dict(directory=shard_dir.name, source_index=source_index, name=source['name'],
                preparation_seed=source['preparation_seed'], split=source['split'],
                temperature_K=source['temperature_K'], context=context, centers=len(centers),
                frames=stop-start, checksums={f: sha256(shard_dir / f) for f in files})
            write_json(completion, shard)
            shards.append(shard)
            print(f"Encoded {shard_dir.name}: {len(centers)} centers x {stop-start} frames", flush=True)
    manifest = dict(state='complete', protocol=protocol, cadence_ps=config['cadence_ps'],
        embedding_dim=256, representation='frozen snapshot encoder, before VICReg projector',
        neighborhood='instantaneous periodic 80 nearest atoms; fixed center atom identity',
        shards=shards)
    write_json(root / 'manifest.json', manifest)
    return manifest


class WindowDataset(Dataset):
    """Batch-indexed windows; storage is (center, time, channel), with no window copies."""

    def __init__(self, root, manifest, split, history_ps, future_ps, stride_ps, anchor_history_ps):
        self.root = resolve_path(root)
        self.records = [r for r in manifest['shards'] if r['split'] == split]
        self.cadence_ps = manifest['cadence_ps']
        self.history_steps = frame_steps(history_ps, self.cadence_ps) + 1
        self.history_skip = frame_steps(anchor_history_ps, self.cadence_ps) + 1 - self.history_steps
        if self.history_skip < 0:
            raise ValueError('History exceeds the common anchor grid; increase anchor_history_ps and reprepare segments.')
        self.future_steps = frame_steps(future_ps, self.cadence_ps)
        self.stride = frame_steps(stride_ps, self.cadence_ps)
        if self.future_steps < 1 or self.stride < 1:
            raise ValueError('Forecast horizon and window stride must each be at least one frame.')
        self.dim = manifest['embedding_dim']
        self.windows = np.array([(r['frames'] - self.history_skip - self.history_steps - self.future_steps) // self.stride + 1
                                 for r in self.records], dtype=np.int64)
        if not self.records or np.any(self.windows < 1):
            raise ValueError(f'No complete {history_ps}+{future_ps} ps windows for split {split}: {self.root}')
        self.ends = np.cumsum(self.windows * [r['centers'] for r in self.records])
        self._arrays = {}

    def __len__(self):
        return int(self.ends[-1])

    def arrays(self, shard):
        if shard not in self._arrays:
            directory = self.root / self.records[shard]['directory']
            self._arrays[shard] = {name: np.load(directory / f'{name}.npy', mmap_mode='r')
                                  for name in ('embeddings', 'frames', 'atom_ids')}
        return self._arrays[shard]

    def __getstate__(self):
        return dict(self.__dict__, _arrays={})

    def __getitem__(self, indices):
        rows = np.asarray(indices, dtype=np.int64)
        shards = np.searchsorted(self.ends, rows, side='right')
        starts = np.r_[0, self.ends[:-1]]
        length = self.history_steps + self.future_steps
        values = np.empty((len(rows), length, self.dim), dtype=np.float32)
        sources, atoms, frames, temperatures = (np.empty(len(rows), dtype=dtype)
            for dtype in (np.int64, np.int64, np.int64, np.float32))
        for shard in np.unique(shards):
            mask = shards == shard
            local = rows[mask] - starts[shard]
            centers, window = np.divmod(local, self.windows[shard])
            columns = self.history_skip + window[:, None] * self.stride + np.arange(length)
            arrays = self.arrays(int(shard))
            values[mask] = arrays['embeddings'][centers[:, None], columns]
            sources[mask] = self.records[shard]['source_index']
            temperatures[mask] = self.records[shard]['temperature_K']
            atoms[mask] = arrays['atom_ids'][centers]
            frames[mask] = arrays['frames'][columns[:, self.history_steps - 1]]
        return dict(history=torch.from_numpy(values[:, :self.history_steps]),
                    future=torch.from_numpy(values[:, self.history_steps:]),
                    source=torch.from_numpy(sources), atom_id=torch.from_numpy(atoms),
                    anchor_frame=torch.from_numpy(frames), temperature_K=torch.from_numpy(temperatures))


def window_loader(dataset, batch_size, workers, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    sampler = RandomSampler(dataset, generator=generator) if shuffle else SequentialSampler(dataset)
    worker_options = dict(multiprocessing_context='spawn') if workers > 0 else {}
    return DataLoader(dataset, sampler=BatchSampler(sampler, batch_size, drop_last=False),
                      batch_size=None, num_workers=workers, pin_memory=torch.cuda.is_available(),
                      persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(seed + 1), **worker_options)


def fit_scaling(dataset, floor_fraction):
    """Streaming moments of each unique training embedding, never of held-out data."""
    total = np.zeros(dataset.dim, dtype=np.float64)
    square = total.copy()
    count = 0
    for index in range(len(dataset.records)):
        z = dataset.arrays(index)['embeddings']
        for start in range(0, len(z), 32):
            values = np.asarray(z[start:start + 32], dtype=np.float64).reshape(-1, dataset.dim)
            total += values.sum(0)
            square += np.square(values).sum(0)
            count += len(values)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean**2, 0))
    floor = floor_fraction * np.sqrt(np.mean(std**2))
    if floor <= 0 or not np.isfinite(std).all():
        raise ValueError('Training embeddings have no finite variance; forecasting/collapse metrics are undefined.')
    return mean.astype(np.float32), np.maximum(std, floor).astype(np.float32)
