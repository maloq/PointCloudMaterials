"""Verify native Q/V NPZ observations and publish the standard binary format."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from src.experiment_runner.artifacts import file_hash,write_json
from src.data.trajectories.shooting import (FORMAT_NAME,SCHEMA_VERSION,ShootingBinaryTrajectory,
                                           _array_description,_array_sha256)


def convert(source,target,*,delete_source=False):
    source,target=Path(source),Path(target)
    if target.exists():
        result=ShootingBinaryTrajectory.load(target);result.verify_checksums()
        receipt=json.loads((target/'conversion.json').read_text())
        if source.exists() and file_hash(source)!=receipt['source_sha256']:raise ValueError('Changed NPZ source')
    else:
        checksum=file_hash(source)
        with np.load(source,allow_pickle=False) as data:
            required={'positions','velocities','box_low','box_high','timesteps','atom_ids','atom_types','provenance'}
            if set(data.files)!=required:raise ValueError(f'Native Q/V fields differ: {data.files}')
            arrays={k:data[k].copy() for k in required-{'provenance'}}
            provenance=json.loads(str(data['provenance']))
        q,v=arrays['positions'],arrays['velocities'];steps=arrays['timesteps'];frames,atoms,three=q.shape
        if three!=3 or v.shape!=q.shape or q.dtype!=np.float64 or v.dtype!=np.float64:
            raise ValueError('Native Q and V must be float64 (frames,atoms,3)')
        if steps.dtype!=np.int64 or steps.shape!=(frames,) or np.any(np.diff(steps)<=0):raise ValueError('Invalid exact steps')
        if not np.array_equal(arrays['atom_ids'],np.arange(1,atoms+1)) or arrays['atom_ids'].dtype!=np.int64:
            raise ValueError('Stable atom IDs 1..N required')
        if arrays['atom_types'].dtype!=np.int32 or arrays['atom_types'].shape!=(atoms,):raise ValueError('Invalid types')
        for key in ['box_low','box_high']:
            if arrays[key].shape!=(frames,3) or not np.isfinite(arrays[key]).all():raise ValueError('Invalid box')
            arrays[key]=arrays[key].astype(np.float32)
        box=arrays['box_high']-arrays['box_low']
        if np.any(box<=0):raise ValueError('Invalid cell lengths')
        # Input positions are already relative to the native box origin. Wrap
        # against the stored float32 box before quantizing, as normal readers do.
        arrays['positions']=np.mod(q,box[:,None,:])
        errors={};semantic={}
        for key in ['positions','velocities']:
            original=arrays[key]
            if not np.isfinite(original).all():raise ValueError('Nonfinite native observations')
            encoded=original.astype(np.float16)
            if not np.isfinite(encoded).all():raise ValueError('Float16 observation overflow')
            difference=encoded.astype(np.float64)-original
            if key=='positions':difference-=box[:,None,:]*np.rint(difference/box[:,None,:])
            errors[key]=dict(max_abs=float(abs(difference).max()),rms=float(np.sqrt(np.mean(difference**2))),
                             reference_dtype='float64',units='angstrom' if key=='positions' else 'angstrom_per_ps')
            semantic[key]=_array_sha256(original);arrays[key]=encoded
        building=target.with_name('.'+target.name+'.building');building.mkdir(parents=True)
        descriptions={}
        for key,values in arrays.items():
            np.save(building/(key+'.npy'),values,allow_pickle=False)
            observed=np.load(building/(key+'.npy'),allow_pickle=False)
            if not np.array_equal(values,observed):raise IOError(f'Write mismatch {key}')
            descriptions[key]=_array_description(values,key+'.npy',sha256=_array_sha256(values))
        manifest=dict(format=FORMAT_NAME,schema_version=SCHEMA_VERSION,state='complete',
            created_at=datetime.now(timezone.utc).isoformat(),storage_dtype='float16',
            coordinate_convention='Wrapped Cartesian coordinates relative to box_low; decode and wrap against float32 box',
            velocity_units='angstrom_per_ps',atom_count=atoms,frame_count=frames,
            first_timestep=int(steps[0]),last_timestep=int(steps[-1]),
            source=dict(path=str(source.resolve()),sha256=checksum,semantic_native_sha256=semantic),
            provenance=provenance,arrays=descriptions)
        write_json(building/'manifest.json',manifest)
        receipt=dict(state='verified',source_sha256=checksum,source_deleted=False,quantization=errors,
                     native_checkpoints='Retained separately at float64/native LAMMPS precision')
        write_json(building/'conversion.json',receipt)
        ShootingBinaryTrajectory.load(building).verify_checksums();building.rename(target)
    if delete_source and source.exists():
        if file_hash(source)!=receipt['source_sha256']:raise ValueError('NPZ source changed before deletion')
        source.unlink();receipt['source_deleted']=True;write_json(target/'conversion.json',receipt)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(__doc__);p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--delete-source',action='store_true');a=p.parse_args(argv)
    print(json.dumps(convert(a.source,a.output,delete_source=a.delete_source)))
