"""Bounded-memory float16 export, verified against every float32 consumer value."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat

from src.data.conversion.shooting_text import iter_lammps_shooting_frames_for_conversion
from src.data.trajectories.shooting import (FORMAT_NAME, SCHEMA_VERSION, ShootingBinaryTrajectory,
                                          _array_description, _array_sha256)
from src.project_runtime.transfer import write_json
from src.simulation.campaigns.birth_sources import now, sha256

PROTOCOL = 'al_dense_replay_001ps_v1'


def convert_streaming_pair(root, metadata, steps, *, chunk_frames=32, delete_source=False):
    """Write ordinary shooting-binary arrays without retaining a full float32 tree."""
    root = Path(root); source = root/'trajectory.lammpstrj'
    if sha256(source) != metadata['source_sha256']:
        raise ValueError(f'Completed source dump checksum changed: {source}')
    if chunk_frames < 1:
        raise ValueError('Chunk frame count must be positive')
    steps = tuple(steps); atoms = metadata['atom_count']; frames = len(steps)
    target = root/'trajectory_binary_float16'; building = root/'.trajectory_binary_float16.building'
    if target.exists() or building.exists():
        raise FileExistsError(f'Inspect existing streaming conversion before rerunning: {root}')
    building.mkdir()
    shape = (frames, atoms, 3)
    lows = np.empty((frames,3),dtype=np.float32); highs = np.empty_like(lows)
    reference_hash = {name:hashlib.sha256() for name in ('positions','velocities')}
    stored_hash = {name:hashlib.sha256() for name in reference_hash}
    errors = {name:dict(max_abs=0.,square_sum=0.,coordinate_count=0) for name in reference_hash}
    handles = {}
    identity = None; kinds = None
    pending = {name:[] for name in reference_hash}
    written = 0

    def flush_chunk():
        nonlocal written
        if not pending['positions']:
            return
        n = len(pending['positions'])
        for name, items in pending.items():
            reference = np.stack(items)
            encoded = reference.astype(np.float16)
            if not np.isfinite(reference).all() or not np.isfinite(encoded).all():
                raise ValueError(f'Nonfinite/overflowing {name} in frames {written}:{written+n}')
            reference_hash[name].update(reference.tobytes())
            payload = encoded.tobytes(); stored_hash[name].update(payload)
            handle = handles[name]; offset = handle.tell()
            handle.write(payload); handle.flush()
            handle.seek(offset)
            observed = np.frombuffer(handle.read(len(payload)),dtype=np.float16).reshape(encoded.shape)
            if not np.array_equal(observed,encoded):
                raise ValueError(f'Incorrect stored rounding for {name}, frame {written}')
            handle.seek(offset+len(payload))
            delta = observed.astype(np.float64)-reference
            if name == 'positions':
                lengths = highs[written:written+n]-lows[written:written+n]
                delta -= lengths[:,None,:]*np.rint(delta/lengths[:,None,:])
            error = errors[name]
            error['max_abs'] = max(error['max_abs'],float(np.abs(delta).max()))
            error['square_sum'] += float(np.square(delta).sum())
            error['coordinate_count'] += delta.size
            items.clear()
        written += n
        write_json(root/'technical/conversion_progress.json',dict(frames_written=written,
                   expected_frames=frames,chunk_frames=chunk_frames,updated_at=now()))

    try:
        for name in reference_hash:
            handle = (building/f'{name}.npy').open('x+b')
            handles[name] = handle
            npformat.write_array_header_2_0(handle,dict(descr=np.dtype('float16').str,
                                                       fortran_order=False,shape=shape))
        for index, frame in enumerate(iter_lammps_shooting_frames_for_conversion(
                source,timesteps=steps,atom_count=atoms,exact_timeline=True)):
            if identity is None:
                identity=frame.atom_ids.copy(); kinds=frame.atom_types.copy()
            if not np.array_equal(frame.atom_ids,identity) or not np.array_equal(frame.atom_types,kinds):
                raise ValueError(f'Atom identity/types changed at timestep {frame.timestep}')
            lows[index]=frame.box_low; highs[index]=frame.box_high
            for name in pending:
                pending[name].append(getattr(frame,name))
            if len(pending['positions']) == chunk_frames:
                flush_chunk()
        flush_chunk()
        if written != frames:
            raise ValueError(f'Wrong frame count: {written}, expected {frames}')
        for handle in handles.values():
            handle.flush(); os.fsync(handle.fileno())
    finally:
        for handle in handles.values():
            handle.close()
    arrays = {}
    for name in stored_hash:
        # Check persisted NPY payload with bounded reads; manifests use array-byte
        # hashes, exactly as the established shooting-binary reader requires.
        with (building/f'{name}.npy').open('rb') as handle:
            if npformat.read_magic(handle) != (2,0):
                raise ValueError('Unexpected NPY header version')
            observed_shape, fortran, dtype = npformat.read_array_header_2_0(handle)
            if observed_shape != shape or fortran or dtype != np.dtype('float16'):
                raise ValueError(f'Wrong stored NPY contract: {name}')
            actual = hashlib.sha256()
            while block := handle.read(16*1024*1024):
                actual.update(block)
        if actual.hexdigest() != stored_hash[name].hexdigest():
            raise ValueError(f'Stored {name} checksum differs from every verified rounded chunk')
        arrays[name] = dict(file=f'{name}.npy',dtype='float16',shape=list(shape),sha256=actual.hexdigest())
    for name, values in dict(timesteps=np.asarray(steps,dtype=np.int64),box_low=lows,box_high=highs,
                             atom_ids=identity,atom_types=kinds).items():
        np.save(building/f'{name}.npy',values,allow_pickle=False)
        persisted=np.load(building/f'{name}.npy',allow_pickle=False)
        if not np.array_equal(values,persisted):
            raise ValueError(f'Stored exact identity/timeline/box array changed: {name}')
        arrays[name]=_array_description(values,f'{name}.npy',sha256=_array_sha256(persisted))
    quantization = {name:dict(max_abs=e['max_abs'],rms=(e['square_sum']/e['coordinate_count'])**.5,
        coordinate_count=e['coordinate_count'],units='angstrom' if name=='positions' else 'angstrom_per_ps',
        periodic_minimum_image=name=='positions') for name,e in errors.items()}
    stat = source.stat()
    provenance=dict(protocol=metadata['protocol'],root_lineage=metadata['root_lineage'],split=metadata['split'],
                    source_sha256=metadata['source_sha256'],timestep_ps=metadata['timestep_ps'])
    manifest=dict(format=FORMAT_NAME,schema_version=SCHEMA_VERSION,state='complete',created_at=now(),
        storage_dtype='float16',atom_count=atoms,frame_count=frames,first_timestep=steps[0],last_timestep=steps[-1],
        coordinate_convention='positions are wrapped float32 consumer coordinates relative to box_low in the half-open periodic interval [0, box_high-box_low)',
        velocity_units='angstrom_per_ps',provenance=provenance,arrays=arrays,
        source=dict(trajectory_path=str(source),size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    semantic_float32_sha256={name:h.hexdigest() for name,h in reference_hash.items()}))
    write_json(building/'manifest.json',manifest)
    building.rename(target)
    loaded=ShootingBinaryTrajectory.load(target)
    if loaded.frame_count != frames or loaded.atom_count != atoms:
        raise ValueError('Exported consumer contract differs')
    report=dict(state='complete',protocol=metadata['protocol'],source_sha256=metadata['source_sha256'],
        source_deleted=False,frame_count=frames,atom_count=atoms,
        float16_manifest_sha256=sha256(target/'manifest.json'),quantization=quantization,
        float32_reference_retained=False,float32_semantic_sha256=manifest['source']['semantic_float32_sha256'],
        precision_reference='Every float16 rounding value verified against the established float32 consumer in bounded chunks; native restarts retained',
        conversion_protocol='streaming_paired_float16_v1',chunk_frames=chunk_frames)
    write_json(root/'paired_conversion.json',report)
    if delete_source:
        source.unlink(); report['source_deleted']=True
        write_json(root/'paired_conversion.json',report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--delete-source',action='store_true')
    parser.add_argument('--chunk-frames',type=int,default=32)
    args=parser.parse_args(argv)
    metadata=json.loads((args.directory/'metadata.json').read_text())
    if metadata['protocol'] != PROTOCOL or metadata['state'] != 'dynamics_complete':
        raise ValueError('Dense 0.01 ps conversion requires completed declared dynamics')
    if (metadata['timestep_ps'],metadata['sample_interval_steps'],metadata['measurement_steps']) != (.002,5,300000):
        raise ValueError('Wrong 600 ps / 0.01 ps dense Al timeline')
    print(json.dumps(convert_streaming_pair(args.directory,metadata,range(0,300001,5),
                    chunk_frames=args.chunk_frames,delete_source=args.delete_source),indent=2))


if __name__ == '__main__':
    main()
