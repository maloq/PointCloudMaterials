"""Content-bound inference caches with the collectors' sample-aligned array contract."""

import hashlib
import json
from pathlib import Path
import threading
import zipfile

import numpy as np
from omegaconf import OmegaConf

from src.data.trajectories.lammps import resolve_temporal_lammps_artifact
from src.experiment_runner.artifacts import file_hash, implementation_hashes
from src.project_runtime.paths import resolve_path
from .output_layout import write_json


def _static_cache_files(data):
    settings = data.get('sample_cache')
    if not settings or not settings['enabled']:
        return []
    root = resolve_path(settings['cache_dir'])
    metadata = root / 'metadata.json'
    if not metadata.exists():
        return []  # The datamodule prepares this cache before the final inference spec.
    record = json.loads(metadata.read_text())
    caches = [(root, record)]
    local = settings.get('local_cache_dir')
    if local is not None and str(local).strip():
        from src.data.static import PointCloudDataset
        staged = resolve_path(local)
        matches, _ = PointCloudDataset._cache_copy_matches(
            source_cache_dir=root, staged_cache_dir=staged, metadata=record,
        )
        if matches:
            caches.append((staged, json.loads((staged / 'metadata.json').read_text())))
    paths = []
    for directory, cache_record in caches:
        paths.append(directory / 'metadata.json')
        for shard in cache_record['shards']:
            paths.append(directory / shard['samples_path'])
            if shard['coords_path'] is not None:
                paths.append(directory / shard['coords_path'])
    if data.get('atomic_context') is not None:
        for shard in record['shards']:
            directory = resolve_path(data['atomic_context']['cache_dir'])
            receipt = directory / (shard['file'] + '.json')
            if receipt.exists():
                paths.extend((receipt, directory / (shard['file'] + '.context.npy')))
    return paths


def _input_hashes(data):
    """Fingerprint the concrete files consumed by each maintained data producer."""
    kind = data['kind'].strip().lower()
    if kind == 'synthetic':
        root = resolve_path(data['synthetic']['root_dir'])
        paths = [directory / name for directory in sorted(root.iterdir()) if directory.is_dir()
                 for name in ('atoms.npy', 'atoms_full.npy', 'metadata.json', 'phase_mapping.json')]
    elif kind == 'static':
        sources = data.get('data_sources') or [data]
        paths = []
        for source in sources:
            names = source['data_files']
            names = [names] if isinstance(names, str) else names
            for name in names:
                path = resolve_path(source['data_path']) / name
                # The OFF reader consumes an existing NPY conversion when present.
                converted = path.with_suffix('.npy')
                paths.append(converted if path.suffix.lower() == '.off' and converted.exists() else path)
        paths.extend(_static_cache_files(data))
    elif kind == 'temporal_lammps':
        source = resolve_temporal_lammps_artifact(data['dump_file'])
        cache = (resolve_path(data['cache_dir']) if data.get('cache_dir') is not None
                 else Path(data['dump_file']).expanduser().resolve().with_suffix('.temporal_cache'))
        paths = [source]
        if source.is_dir():
            manifest = source / 'manifest.json'
            arrays = json.loads(manifest.read_text())['arrays']
            paths = [manifest, *(source / record['file'] for record in arrays.values())]
        elif (cache / 'manifest.json').exists():
            paths.extend(cache / name for name in (
                'manifest.json', 'positions.npy', 'atom_ids.npy', 'atom_types.npy',
                'timesteps.npy', 'box_low.npy', 'box_high.npy',
            ))
        if data.get('precompute_neighbor_indices', False):
            for receipt in cache.glob('neighbor_indices_*.json'):
                paths.extend((receipt, receipt.with_suffix('.npy')))
    elif kind in ('relaxed_histories', 'spatiotemporal_binary'):
        root = resolve_path(data['cache_dir'])
        manifest = root / 'manifest.json'
        paths = [manifest, *(root / name for name in json.loads(manifest.read_text())['checksums'])]
    else:
        raise ValueError(f'Inference cache has no input contract for data.kind={kind!r}')
    return {str(path.resolve()): file_hash(path) for path in sorted(set(paths))}


def _collector_hashes():
    repository = Path(__file__).resolve().parents[2]
    # Bind model construction and data preprocessing as well as array collection.
    paths = [path for folder in ('src/models', 'src/data', 'src/data_utils', 'src/utils', 'src/training_methods')
             for path in (repository / folder).rglob('*.py')]
    paths += [repository / f'src/analysis/{name}.py' for name in (
        'inference_cache', 'utils', 'pipeline_runtime', 'temporal_real', 'temporal_dense',
        'dynamic_motif_cache')]
    paths += list((repository / 'src/analysis').glob('*adapter.py'))
    paths += [repository / name for name in ('src/research/supervised_onset/model.py',
                                            'src/research/encoder_context/geometry.py')]
    return implementation_hashes(*(str(path.relative_to(repository)) for path in sorted(paths)))


def _build_inference_cache_spec(*, checkpoint_path, cfg, inference_batch_size,
                              max_batches_latent, max_samples_total, seed_base,
                              temporal_real_selection=None, temporal_sequence_inference=None,
                              collector_mode='generic'):
    checkpoint = Path(checkpoint_path).resolve()
    stat = checkpoint.stat()
    config = OmegaConf.to_container(cfg, resolve=True)
    inputs = (dict(kind='temporal_lammps', dump_file=temporal_real_selection['dump_file'],
                   cache_dir=temporal_real_selection['cache_dir'],
                   precompute_neighbor_indices=temporal_real_selection['precompute_neighbor_indices'])
              if temporal_real_selection is not None else config['data'])
    return dict(
        version=9,
        checkpoint=dict(path=str(checkpoint), size_bytes=stat.st_size,
                        mtime_ns=stat.st_mtime_ns, sha256=file_hash(checkpoint)),
        model_type=str(cfg.model_type), model_config=config, data_config=config['data'],
        input_sha256=_input_hashes(inputs), implementation_sha256=_collector_hashes(),
        checkpoint_batch_size=int(cfg.batch_size), inference_batch_size=int(inference_batch_size),
        max_batches_latent=max_batches_latent, max_samples_total=max_samples_total,
        seed_base=int(seed_base), rng_strategy='seed_once_per_collection_v1', collect_coords=True,
        collector_mode=str(collector_mode), temporal_real_selection=temporal_real_selection,
        temporal_sequence_inference=temporal_sequence_inference,
    )


def _inference_cache_spec_hash(spec):
    payload = json.dumps(spec, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _inference_cache_paths(out_dir, cache_filename):
    data = Path(out_dir) / cache_filename
    return data, data.with_suffix(data.suffix + '.meta.json')


def _validate_inference_cache_arrays(cache):
    missing = {'inv_latents', 'eq_latents', 'phases', 'coords', 'instance_ids'} - cache.keys()
    if missing:
        raise ValueError(f'Inference cache is missing collector fields: {sorted(missing)}')
    latents = cache['inv_latents']
    if latents.ndim != 2 or min(latents.shape) == 0:
        raise ValueError(f'Expected nonempty (samples, channels) invariant latents, got {latents.shape}')
    rows = len(latents)
    for name, values in cache.items():
        if values.ndim == 0 or len(values) not in (0, rows):
            raise ValueError(f'Cache {name}: expected {rows} aligned rows or absent optional values, '
                             f'got {values.shape}')
    if cache['coords'].ndim != 2 or cache['coords'].shape[1] != 3:
        raise ValueError(f'Cache coords must have shape (samples, 3), got {cache["coords"].shape}')
    for name in ('phases', 'instance_ids', 'anchor_frame_indices', 'sample_index'):
        if name in cache and (cache[name].ndim != 1 or
                              (len(cache[name]) and cache[name].dtype.kind not in 'iu')):
            raise ValueError(f'Cache {name} must be a one-dimensional integer array')
    _validate_invariant_latent_values(latents)


def _validate_invariant_latent_values(inv_latents):
    for start in range(0, len(inv_latents), 65536):
        chunk = inv_latents[start:start + 65536].astype(np.float32, copy=False)
        invalid = ~np.isfinite(chunk).all(axis=1) | (np.linalg.norm(chunk, axis=1) <= 1e-8)
        if invalid.any():
            raise ValueError(f'Invalid encoder output at row {start + np.flatnonzero(invalid)[0]}: '
                             'nonfinite or zero-norm embedding; recompute inference.')


def _array_schema(cache):
    return {name: dict(shape=list(values.shape), dtype=values.dtype.str)
            for name, values in cache.items()}


def _load_inference_cache(*, out_dir, cache_filename, expected_spec):
    data, metadata = _inference_cache_paths(out_dir, cache_filename)
    if not data.exists() and not metadata.exists():
        return None, f'cache file does not exist: {data}'
    if not data.exists() or not metadata.exists():
        raise RuntimeError(f'Incomplete inference cache: require both {data} and {metadata}')
    meta = json.loads(metadata.read_text())
    expected = _inference_cache_spec_hash(expected_spec)
    if meta['spec_sha256'] != expected:
        return None, f'cache spec mismatch: expected sha256={expected}, found {meta["spec_sha256"]}'
    if file_hash(data) != meta['data_sha256']:
        raise ValueError(f'Inference cache contents changed: {data}')
    try:
        with np.load(data, allow_pickle=False) as arrays:
            cache = {name: arrays[name] for name in arrays.files}
        _validate_inference_cache_arrays(cache)
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f'Inference cache validation failed for {data}: {exc}') from exc
    if _array_schema(cache) != meta['arrays'] or len(cache['inv_latents']) != meta['num_samples']:
        raise ValueError(f'Inference cache array schema differs from its receipt: {data}')
    return cache, f'loaded cache from {data}'


def _save_inference_cache(*, out_dir, cache_filename, cache, spec):
    _validate_inference_cache_arrays(cache)
    data, metadata = _inference_cache_paths(out_dir, cache_filename)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    temporary = data.with_suffix(data.suffix + '.tmp.npz')
    temporary_metadata = metadata.with_suffix(metadata.suffix + '.tmp')
    _save_npz_with_progress(temporary, cache)
    write_json(temporary_metadata, dict(
        spec=spec, spec_sha256=_inference_cache_spec_hash(spec), data_sha256=file_hash(temporary),
        num_samples=len(cache['inv_latents']), arrays=_array_schema(cache), storage='npz_uncompressed',
    ))
    temporary.replace(data)
    temporary_metadata.replace(metadata)


def _save_npz_with_progress(path, arrays):
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(30):
            print(f'[analysis][cache] Still writing {path.name}...', flush=True)

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        np.savez(path, **arrays)
    finally:
        stop.set()
        thread.join(timeout=1)


def discard_inference_cache(out_dir, cache_filename):
    """Retain reconstruction provenance before removing completed inference arrays."""
    data, metadata = _inference_cache_paths(out_dir, cache_filename)
    if metadata.is_file():
        retained = Path(out_dir) / 'retention'
        retained.mkdir(exist_ok=True)
        archive = retained / f'{metadata.name}.{file_hash(metadata)}.json'
        archive.write_bytes(metadata.read_bytes())
        if file_hash(archive) != file_hash(metadata):
            raise RuntimeError(f'Could not verify retained inference specification: {archive}')
        (retained / 'README.md').write_text(
            '# Removed inference caches\n\n'
            'The adjacent metadata preserves checkpoint, input, implementation, sampling and seed identities. '
            'Rebuild with the original full analysis in a new output directory. '
            'Keep the selected checkpoint and original datasets. Prediction arrays are not removed.\n')
    for path in (data, metadata):
        path.unlink(missing_ok=True)
