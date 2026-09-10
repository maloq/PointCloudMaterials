"""Reuse completed Al denoising and temporal80 targets without new minimization."""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import shutil

import numpy as np

from src.analysis.liquid_structure import persistence_image
from src.data_utils.mace_denoising import completed, signature
from src.data_utils.temporal_campaign import write_json
from src.simulation.relaxation import sha256


def verified_manifest(path):
    path = Path(path)
    record = json.loads(path.read_text())
    for name, digest in record['checksums'].items():
        if sha256(path.parent / name) != digest:
            raise ValueError(f'Changed completed input: {path.parent / name}')
    return record


def prepare_existing(cfg):
    root = Path(cfg['cache'])
    root.mkdir(parents=True, exist_ok=True)
    records = []
    assignments = {}
    producer = {str(p): sha256(p) for p in (Path(__file__), Path('src/analysis/liquid_structure.py'))}
    filenames = ('histories.npy', 'relaxed.npy', 'targets.npy', 'hot_targets.npy',
                 'centers.npy', 'neighbor_ids.npy', 'frames.npy')
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for item in cfg['reuse_records']:
            if item['source_index'] in assignments and assignments[item['source_index']] != item['split']:
                raise ValueError(f"Source crosses training/evaluation splits: {item['name']}")
            assignments[item['source_index']] = item['split']
            original = verified_manifest(item['manifest'])
            source = Path(item['manifest']).parent
            provenance = dict(selection=item, input_manifest_sha256=sha256(item['manifest']), producer=producer,
                              original_split=original['split'])
            if item['format'] == 'temporal80':
                paired = verified_manifest(item['paired_manifest'])
                provenance['paired_manifest_sha256'] = sha256(item['paired_manifest'])
                provenance['offsets'] = original['signature']['frame_offsets_ps']
            elif item['format'] == 'denoising80':
                provenance['offsets'] = original['provenance']['offsets']
            else:
                raise ValueError(f"Unsupported owned cache format: {item['format']}")
            sig = signature(provenance)
            directory = root / item['name']
            directory.mkdir(exist_ok=True)
            saved = completed(directory, sig)
            if saved is not None:
                records.append(saved)
                continue
            write_json(Path(cfg['output']) / 'status.json', dict(state='reusing_completed_targets',
                completed_shards=len(records), total_shards=len(cfg['reuse_records']), shard=item['name']))
            if item['format'] == 'denoising80':
                for name in filenames:
                    shutil.copyfile(source / name, directory / name)
                frame = original['frame']
            else:
                paired_dir = Path(item['paired_manifest']).parent
                # temporal80 trains on view zero. Its other views are not independent examples.
                histories = np.load(source / 'histories.npy', mmap_mode='r')[:, 0]
                clouds = np.load(paired_dir / 'clouds.npy', mmap_mode='r')
                targets = np.load(source / 'targets.npy', mmap_mode='r')[:, 0]
                np.testing.assert_array_equal(histories[:, -1], clouds[:, 0])
                np.testing.assert_array_equal(targets, np.load(paired_dir / 'tda.npy', mmap_mode='r')[:, 0])
                frames = np.load(source / 'frames.npy')[:, 0]
                np.testing.assert_array_equal(frames, np.tile(frames[0], (len(frames), 1)))
                hot_targets = np.stack(list(pool.map(persistence_image, histories[:, -1].astype(np.float32), chunksize=32)))
                arrays = dict(histories=histories, relaxed=clouds[:, 4], targets=targets, hot_targets=hot_targets,
                    centers=np.load(source / 'ids.npy')[:, 0], neighbor_ids=np.load(source / 'neighbor_ids.npy')[:, 0],
                    frames=frames[0])
                for name, value in arrays.items():
                    np.save(directory / f'{name}.npy', value)
                frame = int(frames[0, -1])
            saved = dict(name=item['name'], directory=str(directory), signature=sig, provenance=provenance,
                source_index=item['source_index'], split=item['split'], temperature_K=item['temperature_K'],
                potential=item['potential'], minimizer=item['minimizer'], lineage=item['lineage'],
                frame=frame, count=original['count'],
                checksums={name: sha256(directory / name) for name in filenames})
            write_json(directory / 'manifest.json', saved)
            records.append(saved)
            print('REUSED_TARGETS', item['name'], saved['count'], item['potential'], item['minimizer'], flush=True)
    write_json(root / 'manifest.json', dict(protocol='denoising80_reuse', shards=records,
        context=cfg['data_context'], limitations=cfg['data_limitations'],
        target='Existing converged full-cell targets; original hot-selected atom identities; no target averaging.',
        sampling='Frozen list of available artifacts; whole-source split declared before fitting.'))
