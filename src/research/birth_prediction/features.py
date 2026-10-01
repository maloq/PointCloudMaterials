"""Reuse rich descriptors and exact frozen scalar MACE exports on common histories."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.research.equivariant_context.cache import RetainedCache
from src.research.liquid_predictability.descriptors import patch_descriptors
from .data import load, read


def descriptors(c, index):
    root = resolve_path(c['cache'])
    manifest = read(root / 'manifest.json')
    for item in manifest['sources'][index::c['prepare_tasks']]:
        folder = root / 'sources' / str(item['source'])
        positions = np.load(folder / 'positions.npy', mmap_mode='r')
        binding = dict(source=item['identity'], positions_sha256=item['files']['positions.npy'],
                       descriptor_sha256=sha(Path(patch_descriptors.__code__.co_filename)))
        identity = digest(binding)
        receipt = folder / 'descriptor-complete.json'
        if receipt.exists():
            old = read(receipt)
            if old['identity'] != identity or sha(folder / 'descriptors.npy') != old['sha256']:
                raise ValueError('Changed frozen rich features')
            continue
        # Empty sources retain a valid zero-row artifact with the same schema.
        sample = positions[0] if len(positions) else np.load(root / 'positions.npy', mmap_mode='r')[0]
        _, names = patch_descriptors(sample)
        output = np.lib.format.open_memmap(folder / 'descriptors.npy', mode='w+',
                                          dtype=np.float32, shape=(len(positions), len(names)))
        begin = 0
        progress = folder / 'descriptor-progress.json'
        for i, xyz in enumerate(positions):
            values, actual = patch_descriptors(xyz)
            if actual != names:
                raise ValueError('Rich-descriptor schema changed between observed patches')
            output[i] = values
            if (i + 1) % 128 == 0:
                output.flush()
                write_json(progress, dict(identity=identity, patches=i + 1, total=len(positions)))
        output.flush()
        del output
        write_json(receipt, dict(identity=identity, columns=names, sha256=sha(folder / 'descriptors.npy')))
        print(json.dumps(dict(source=item['source'], patches=len(positions), descriptors=len(names))), flush=True)


def seal_descriptors(c):
    root = resolve_path(c['cache'])
    manifest = read(root / 'manifest.json')
    banks, names, receipts = [], None, []
    for item in manifest['sources']:
        folder = root / 'sources' / str(item['source'])
        receipt = read(folder / 'descriptor-complete.json')
        if sha(folder / 'descriptors.npy') != receipt['sha256']:
            raise ValueError('Changed rich descriptor shard')
        if names is not None and names != receipt['columns']:
            raise ValueError('Different source descriptor schemas')
        names = receipt['columns']
        banks.append(np.load(folder / 'descriptors.npy', mmap_mode='r'))
        receipts.append(receipt)
    np.save(root / 'descriptors.npy', np.concatenate(banks))
    write_json(root / 'descriptor-manifest.json', dict(columns=names, sources=receipts,
               dataset_identity=manifest['identity'], sha256=sha(root / 'descriptors.npy')))


def encoder_metadata(c, model):
    _, _, manifest = load(c)
    return dict(dataset_identity=manifest['identity'], positions_sha256=manifest['files']['positions.npy'],
                model=model, inference_sha256=sha(Path(__file__).with_name('infer.py')))


def encode(c, index):
    model = c['encoders'][index]
    metadata = encoder_metadata(c, model)
    key = digest(metadata)
    pool = RetainedCache(resolve_path(c['feature_cache']), 6)
    root = resolve_path(c['output']) / 'technical'
    root.mkdir(parents=True, exist_ok=True)
    with pool.lease(key, deadline=time.time() + 8 * 3600, metadata=metadata) as cache:
        request = dict(model=model, positions=str(resolve_path(c['cache']) / 'positions.npy'),
                       destination=str(cache / 'states.npy'), batch_size=c['inference_batch_size'])
        path = root / f'inference-{model["name"]}.json'
        write_json(path, request)
        if not (cache / 'complete.json').exists():
            # Execute the recorded training source in a fresh interpreter. Old
            # VICReg producers differ from today's model files and must not mix.
            subprocess.run([sys.executable, str(Path(__file__).with_name('infer.py')), str(path)],
                           check=True, cwd=resolve_path(model['producer']),
                           env=dict(os.environ, TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'))
            write_json(cache / 'complete.json', dict(metadata=metadata, sha256=sha(cache / 'states.npy')))
        done = read(cache / 'complete.json')
        if done['metadata'] != metadata or sha(cache / 'states.npy') != done['sha256']:
            raise ValueError('Changed frozen MACE states')
        write_json(root / f'encoder-cache-{model["name"]}.json', dict(key=key, metadata=metadata,
                   path=str(cache), states_sha256=done['sha256']))


def load_bank(c, name):
    if name == 'descriptors':
        root = resolve_path(c['cache'])
        record = read(root / 'descriptor-manifest.json')
        if sha(root / 'descriptors.npy') != record['sha256']:
            raise ValueError('Changed rich feature bank')
        return np.load(root / 'descriptors.npy'), record['columns']
    model = next(m for m in c['encoders'] if m['name'] == name)
    meta = encoder_metadata(c, model)
    key = digest(meta)
    pool = RetainedCache(resolve_path(c['feature_cache']), 6)
    with pool.lease(key, deadline=time.time() + 3600, metadata=meta, shared=True) as root:
        record = read(root / 'complete.json')
        if record['metadata'] != meta or sha(root / 'states.npy') != record['sha256']:
            raise ValueError('Frozen encoder feature cache missing or changed; rerun encode')
        states = np.load(root / 'states.npy')  # resident copy survives later cache eviction
    return states, [f'{name}/z{i}' for i in range(states.shape[1])]


def history_packet(bank, rows, remove, columns):
    n = rows['indices'].shape[1] - remove
    values = bank[rows['indices'][:, :n]][:, :, columns]
    packet = np.concatenate((values.reshape(len(values), -1), values.mean(1), values.std(1),
                             values[:, -1] - values[:, 0]), axis=1)
    if not np.isfinite(packet).all():
        raise FloatingPointError('Nonfinite history features')
    return packet.astype(np.float32)
