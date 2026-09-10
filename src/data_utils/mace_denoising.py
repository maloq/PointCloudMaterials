"""Independent-source Al histories, full-cell relaxed targets and frozen MACE caches."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from src.analysis.liquid_structure import persistence_image
from src.data_utils.conversion.relaxation import read_relaxed
from src.data_utils.mace_history import history_clouds
from src.data_utils.mace_relaxed import paired_clouds
from src.data_utils.shooting_binary import ShootingBinaryTrajectory
from src.data_utils.temporal_campaign import write_json
from src.simulation.relaxation import relax_frame, sha256


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def completed(directory, expected):
    path = directory / 'manifest.json'
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if saved['signature'] != expected:
        raise ValueError(f'Configuration or producer changed for {path}; use a new cache.')
    for name, digest in saved['checksums'].items():
        if sha256(directory / name) != digest:
            raise ValueError(f'Cache checksum mismatch: {directory / name}')
    return saved


def prepare(cfg):
    """Use the independent MEAM producer's original source splits and atom IDs."""
    root = Path(cfg['cache'])
    root.mkdir(parents=True, exist_ok=True)
    seen = set()
    for source in cfg['sources']:
        lineage = source['preparation_seed']
        if lineage in seen:
            raise ValueError(f'Repeated independent melt seed {lineage}; source splits would leak.')
        seen.add(lineage)
        expected = {'optimization': 'train', 'model_selection': 'val', 'final_validation': 'test'}
        if source['split'] != expected[source['source_split']]:
            raise ValueError(f"Changed source split for {source['name']}")
    producer = {str(p): sha256(p) for p in (
        Path(__file__), Path('src/data_utils/mace_history.py'),
        Path('src/data_utils/mace_relaxed.py'), Path('src/analysis/liquid_structure.py'),
        Path('src/simulation/relaxation.py'))}
    records = []
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        for source_index, source in enumerate(cfg['sources']):
            trajectory = ShootingBinaryTrajectory.load(source['path'])
            np.testing.assert_allclose(np.diff(trajectory.timesteps) * source['timestep_ps'],
                                       source['cadence_ps'], rtol=0, atol=1e-9)
            steps = np.rint(np.array(cfg['frame_offsets_ps']) / source['cadence_ps']).astype(np.int64)
            np.testing.assert_allclose(steps * source['cadence_ps'], cfg['frame_offsets_ps'], rtol=0, atol=1e-9)
            campaign = json.loads(Path(source['campaign_manifest']).read_text())
            lib, parameters = cfg['relaxation']['potential_files']
            if (sha256(lib) != campaign['potential']['library_sha256'] or
                    sha256(parameters) != campaign['potential']['parameter_sha256']):
                raise ValueError(f"Relaxation potential differs from source {source['name']}")
            # Completed producer artifacts include checksums for all exact identities and positions.
            trajectory.verify_checksums()
            for frame in source['anchors']:
                name = f"{source['name']}_frame{frame}"
                directory = root / name
                directory.mkdir(exist_ok=True)
                provenance = dict(source=source, source_manifest_sha256=sha256(trajectory.root / 'manifest.json'),
                    centers=cfg['centers_per_frame'], seed=cfg['seed'], offsets=cfg['frame_offsets_ps'],
                    relaxation=cfg['relaxation'], producer=producer)
                sig = signature(provenance)
                saved = completed(directory, sig)
                if saved is not None:
                    records.append(saved)
                    continue
                write_json(Path(cfg['output']) / 'status.json', dict(state='preparing_targets',
                    completed_frames=len(records), total_frames=sum(len(s['anchors']) for s in cfg['sources']), frame=name))
                work = Path(cfg['output']) / 'relaxed_frames' / source['name'] / str(frame)
                if not (work / 'metadata.json').exists():
                    relax_frame(trajectory, frame, work, cfg['relaxation'])
                relaxed, metadata = read_relaxed(work)
                rng = np.random.default_rng(np.random.SeedSequence([cfg['seed'], source_index, frame]))
                centers = rng.choice(trajectory.atom_count, cfg['centers_per_frame'], replace=False)
                low = trajectory.box_low[frame].astype(np.float64)
                lengths = trajectory.box_high[frame].astype(np.float64) - low
                hot = trajectory.positions[frame].astype(np.float64) - low
                anchor, quenched, neighbors, errors = paired_clouds(hot, relaxed-low, lengths, centers)
                identities = trajectory.atom_ids[neighbors]
                history, history_error = history_clouds(trajectory, centers, identities, frame, steps)
                np.testing.assert_array_equal(history[:, -1], anchor)
                target = np.stack(list(pool.map(persistence_image, quenched.astype(np.float32), chunksize=16)))
                hot_target = np.stack(list(pool.map(persistence_image, anchor.astype(np.float32), chunksize=16)))
                arrays = dict(histories=history, relaxed=quenched, targets=target, hot_targets=hot_target,
                    centers=centers, neighbor_ids=identities, frames=frame+steps)
                for key, value in arrays.items():
                    np.save(directory / f'{key}.npy', value)
                saved = dict(name=name, directory=str(directory), signature=sig, provenance=provenance,
                    source_index=source_index, split=source['split'], temperature_K=source['temperature_K'],
                    frame=frame, count=len(centers), local_quantization_max_A=max(*errors, history_error),
                    fmax_eV_per_A=metadata['fmax_eV_per_A'], relaxation_seconds=metadata['seconds'],
                    checksums={f'{key}.npy': sha256(directory / f'{key}.npy') for key in arrays})
                # Keep text until local targets AND verified global float16 storage exist.
                with (work / 'conversion_stdout.log').open('w') as log:
                    subprocess.run([sys.executable, 'scripts/convert_trajectory.py', 'relaxation', str(work)],
                                   check=True, stdout=log)
                write_json(directory / 'manifest.json', saved)
                records.append(saved)
                print('DENOISING_FRAME', name, metadata['seconds'], flush=True)
    write_json(root / 'manifest.json', dict(protocol='denoising80', shards=records,
        frame_offsets_ps=cfg['frame_offsets_ps'], sampling='Uniform centers; fixed times; no outcome-based selection.',
        target='Relaxed full-cell anchor, fixed hot-selected 80 identities; centered float16 offsets.'))


@torch.no_grad()
def cache_features(cfg):
    """One frozen spatial pass shared by every loss/temporal ablation."""
    from src.models.encoders.pretrained_mace import PretrainedMACEEncoder
    records = json.loads((Path(cfg['cache']) / 'manifest.json').read_text())['shards']
    root = Path(cfg['feature_cache'])
    root.mkdir(parents=True, exist_ok=True)
    model = PretrainedMACEEncoder(cfg['pretrained_checkpoint'], performance=cfg['performance']).cuda().eval()
    model.requires_grad_(False)
    torch.save(dict(model=model.state_dict(), checkpoint_sha256=sha256(cfg['pretrained_checkpoint']),
                    performance=cfg['performance']), root / 'frozen_mace.pt')
    result = []
    for index, record in enumerate(records):
        source = Path(record['directory'])
        directory = root / record['name']
        directory.mkdir(exist_ok=True)
        sig = signature(dict(data_manifest_sha256=sha256(source / 'manifest.json'),
            checkpoint_sha256=sha256(cfg['pretrained_checkpoint']), performance=cfg['performance'],
            encoder_sha256=sha256('src/models/encoders/pretrained_mace.py')))
        saved = completed(directory, sig)
        if saved is not None:
            result.append(saved)
            continue
        history = np.load(source / 'histories.npy')
        relaxed = np.load(source / 'relaxed.npy')
        clouds = np.concatenate((history, relaxed[:, None]), axis=1)
        flat = clouds.reshape(-1, 80, 3)
        values = []
        for start in range(0, len(flat), cfg['feature_batch_size']):
            x = torch.from_numpy(flat[start:start+cfg['feature_batch_size']].astype(np.float32)).cuda()
            values.append(model.raw_node_features(x, torch.zeros(len(x), dtype=torch.long, device='cuda')).cpu().numpy())
        nodes = np.concatenate(values).reshape(len(clouds), clouds.shape[1], 80, 256)
        pooled = nodes.mean(2)
        hot_nodes = nodes[:, :-1].astype(np.float16)
        if not np.isfinite(hot_nodes).all():
            raise FloatingPointError(f'Frozen atom feature float16 overflow in {directory}')
        np.save(directory / 'pooled.npy', pooled)
        np.save(directory / 'nodes.npy', hot_nodes)
        saved = dict(record, directory=str(directory), data_directory=str(source), signature=sig,
            atom_feature_quantization_max=float(np.abs(nodes[:, :-1]-hot_nodes.astype(np.float32)).max()),
            checksums={f'{key}.npy': sha256(directory / f'{key}.npy') for key in ('pooled', 'nodes')})
        write_json(directory / 'manifest.json', saved)
        result.append(saved)
        write_json(Path(cfg['output']) / 'status.json', dict(state='caching_frozen_mace',
            completed_frames=index+1, total_frames=len(records)))
        print('FROZEN_FRAME', record['name'], flush=True)
    write_json(root / 'manifest.json', dict(protocol='denoising80', shards=result,
        pretrained_checkpoint_sha256=sha256(cfg['pretrained_checkpoint']), backbone_frozen=True,
        pooled_order=['past_4', 'past_3', 'past_2', 'past_1', 'anchor', 'relaxed_anchor']))
