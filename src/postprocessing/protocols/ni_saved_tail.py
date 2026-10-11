"""Retain verified Ni observations saved beyond an operational stop endpoint."""
import fcntl
from pathlib import Path
import shutil

import numpy as np

from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.project_runtime.paths import resolve_path
from src.postprocessing.io import read
from src.postprocessing.sources import part


def include_saved_tail(c, record, source):
    """Append already-written observations; never simulate or change the stop record."""
    archive = resolve_path(c['archive'])/record['id']
    outcome = read(archive/'complete.json')
    if outcome['termination'] != 'hard_hold_cap':
        return source
    control = archive/'hard-stop'
    if not read(control/'terminated.json')['verified_publication']:
        raise ValueError('Ni source must have stopped before its saved tail is inventoried')
    destination = control/'saved-tail'; destination.mkdir(exist_ok=True)
    base = ShootingBinaryTrajectory.load(source['parts'][0]['path'])
    original_end = int(base.timesteps[-1]); interval = round(.1/c['timestep_ps'])
    binding = dict(source=record['id'],completion_sha256=sha(archive/'complete.json'),
                   stopping_certificate_sha256=sha(control/'certificate.json'))
    receipt = destination/'complete.json'
    with (destination/'archive.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            saved = read(receipt)
            if saved['binding'] != binding:
                raise ValueError('Ni saved-tail completion or stopping provenance changed')
            retained = saved['chunks']
        else:
            retained = []
            scratch = resolve_path(c['scratch'])/record['id']/'chunks'
            if not scratch.is_dir():
                raise FileNotFoundError(f'Uninventoried stopped Ni chunks are missing: {scratch}')
            for manifest in sorted(scratch.glob('chunk-*/manifest.json')):
                binary = ShootingBinaryTrajectory.load(manifest.parent)
                if int(binary.timesteps[-1]) <= original_end:
                    continue
                binary.verify_checksums()
                conversion = manifest.parent/'conversion.json'
                record_hashes = dict(manifest_sha256=sha(manifest),conversion_sha256=sha(conversion))
                target = destination/manifest.parent.name
                if not target.exists():
                    building = target.with_name(target.name+'.building')
                    shutil.copytree(manifest.parent,building,dirs_exist_ok=True)
                    ShootingBinaryTrajectory.load(building).verify_checksums()
                    if (sha(building/'manifest.json') != record_hashes['manifest_sha256']
                            or sha(building/'conversion.json') != record_hashes['conversion_sha256']):
                        raise ValueError('Ni saved-tail copy differs from its verified native conversion')
                    building.rename(target)
                retained.append(dict(path=str(target),original_path=str(manifest.parent),**record_hashes))
        previous = original_end
        added = 0
        for item in retained:
            path = Path(item['path'])
            if (sha(path/'manifest.json') != item['manifest_sha256']
                    or sha(path/'conversion.json') != item['conversion_sha256']):
                raise ValueError('Archived Ni saved-tail provenance changed')
            binary = ShootingBinaryTrajectory.load(path); binary.verify_checksums()
            if not (np.array_equal(binary.atom_ids,base.atom_ids)
                    and np.array_equal(binary.atom_types,base.atom_types)):
                raise ValueError('Ni saved-tail atom identity differs from its source')
            indices = np.flatnonzero(binary.timesteps > original_end)
            steps = binary.timesteps[indices]
            if not len(steps) or not np.array_equal(steps,np.arange(previous+interval,int(steps[-1])+1,interval)):
                raise ValueError('Ni saved-tail timeline is not the exact continuous native 0.10-ps sequence')
            pi = len(source['parts']); source['parts'].append(part(path,c['timestep_ps']*1000))
            source['frames'].extend([[pi,int(i)] for i in indices])
            source['times_ps'].extend((steps*c['timestep_ps']).tolist())
            previous = int(steps[-1]); added += len(indices)
        n = len(source['frames'])
        source.update(frame_count=n,sampled_frames=list(range(n)),views={'010':list(range(n))})
        if added:
            source['input_identity'] = digest(dict(arrays=[p['array_sha256'] for p in source['parts']],frames=source['frames']))
        sealed = dict(binding=binding,chunks=retained,additional_frames=added,
                      published_endpoint_ps=original_end*c['timestep_ps'],full_saved_endpoint_ps=previous*c['timestep_ps'])
        if receipt.exists() and read(receipt) != sealed:
            raise ValueError('Archived Ni saved-tail inventory changed')
        write_json(receipt,sealed)
        source['saved_tail'] = dict(receipt=str(receipt),sha256=sha(receipt),additional_frames=added)
    return source
