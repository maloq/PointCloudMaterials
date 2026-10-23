"""Archive a stopped orthogonal LAMMPS position/velocity dump as verified float16 or float32.

Preserves the observed timeline, including incomplete final-frame bytes separately.
Does not change scientific completion states or native restart precision.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import mmap
import os
from pathlib import Path

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha256
from src.data.trajectories import array_sha256, shooting, lammps
from src.project_runtime.transfer import write_json

MARKER = b'ITEM: TIMESTEP\n'
COLUMNS = [('id','type','x','y','z'), ('id','type','x','y','z','vx','vy','vz'), ('id','type','vx','vy','vz')]


def signature(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def header(mapped, offset):
    mapped.seek(offset)
    lines = [mapped.readline().decode('ascii').strip() for _ in range(9)]
    if lines[0] != 'ITEM: TIMESTEP' or lines[2] != 'ITEM: NUMBER OF ATOMS' or lines[4] != 'ITEM: BOX BOUNDS pp pp pp':
        raise ValueError(f'Unsupported dump header at byte {offset}: {lines}')
    columns = tuple(lines[8].removeprefix('ITEM: ATOMS ').split())
    if columns not in COLUMNS:
        raise ValueError(f'Unsupported columns at byte {offset}: {columns}')
    bounds = np.array([np.fromstring(s, sep=' ') for s in lines[5:8]], dtype=np.float32)
    if bounds.shape != (3,2) or not np.isfinite(bounds).all() or np.any(bounds[:,1] <= bounds[:,0]):
        raise ValueError(f'Invalid bounds at byte {offset}')
    return int(lines[1]), int(lines[3]), bounds, columns, mapped.tell()


def convert(source, *, output=None, delete_source=False, positions_binary=None, storage_dtype="float16"):
    if storage_dtype not in ("float16", "float32"):
        raise ValueError(f"Unsupported storage dtype: {storage_dtype}")
    source = Path(source).absolute()
    target = Path(output).absolute() if output else source.with_name(source.name+'.'+storage_dtype)
    receipt = target/'conversion.json'
    if target.exists():
        report = json.loads(receipt.read_text())
        if report['state'] != 'verified' or report['source_path'] != str(source):
            raise ValueError(f'Existing conversion does not match source: {target}')
        if json.loads((target/'manifest.json').read_text())['storage_dtype'] != storage_dtype:
            raise ValueError(f'Existing conversion has different precision: {target}')
        module = shooting if report['has_velocities'] else lammps
        cls = module.ShootingBinaryTrajectory if report['has_velocities'] else module.TemporalLAMMPSBinaryTrajectory
        cls.load(target).verify_checksums()
        if report['incomplete_tail']:
            if sha256(target/'incomplete_final_frame.bin') != report['incomplete_tail']['sha256']:
                raise ValueError('Incomplete-frame preservation checksum mismatch')
        if source.exists():
            before = signature(source)
            if sha256(source) != report['source_sha256'] or signature(source) != before:
                raise ValueError(f'Source changed since conversion: {source}')
            if delete_source:
                source.unlink(); report['source_deleted'] = True
                write_json(receipt,report)
        return report
    before = signature(source)
    if before[2] == 0:
        raise ValueError(f'Empty trajectory: {source}')
    building = target.with_name('.'+target.name+'.building')
    if building.exists():
        raise FileExistsError(f'Inspect unfinished conversion: {building}')
    # Hash and locate frame boundaries in one bounded sequential pass.
    with source.open('rb') as stream:
        marker = stream.readline()
    if marker not in (MARKER, b'ITEM: TIMESTEP\r\n'):
        raise ValueError(f'Unsupported initial frame marker: {source}')
    digest=hashlib.sha256(); offsets=[]; consumed=0; overlap=b''
    with source.open('rb') as stream:
        while block := stream.read(16*1024*1024):
            digest.update(block); joined=overlap+block; base=consumed-len(overlap); start=0
            while (found := joined.find(marker,start)) >= 0:
                offset=base+found
                if not offsets or offset>offsets[-1]:offsets.append(offset)
                start=found+len(marker)
            consumed+=len(block);overlap=joined[-(len(marker)-1):]
    source_hash=digest.hexdigest()
    with source.open('rb') as f, mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as mapped:
        if not offsets or offsets[0] != 0:
            raise ValueError(f'No exact initial frame marker: {source}')
        first = header(mapped,0); atoms=first[1]; columns=first[3]
        velocity_only = columns == COLUMNS[2]
        has_velocities = len(columns)==8 or velocity_only
        if velocity_only and positions_binary is None:
            raise ValueError('A velocity-only dump requires --positions-binary for exact matching')
        # A stopped writer may leave its last frame incomplete. Preserve its exact
        # bytes separately, never pretend it is an observed complete frame.
        last_offset=offsets[-1]; tail=None
        try:
            last=header(mapped,last_offset)
            values=np.fromstring(mapped[last[4]:].decode('ascii'),sep=' ')
            complete_last = values.size == last[1]*len(last[3]) and mapped[-1:] == b'\n'
        except (ValueError, UnicodeError):
            complete_last=False
        if not complete_last:
            tail=bytes(mapped[last_offset:]); offsets.pop()
        if not offsets:
            raise ValueError(f'No complete frames: {source}; original retained')
        frames=len(offsets); building.mkdir(parents=True)
        names=['velocities'] if velocity_only else ['positions']+(['velocities'] if has_velocities else [])
        arrays={name:np.lib.format.open_memmap(building/(name+'.npy'),mode='w+',dtype=storage_dtype,shape=(frames,atoms,3)) for name in names}
        expected={name:hashlib.sha256() for name in names}
        errors={name:dict(max_abs=0.,square_sum=0.,count=0) for name in names}
        steps=[]; lows=[]; highs=[]; ids=np.arange(1,atoms+1,dtype=np.int64); kinds=None
        for i,offset in enumerate(offsets):
            step,n,bounds,cols,start=header(mapped,offset)
            if n != atoms or cols != columns or (steps and step<=steps[-1]):
                raise ValueError(f'Changed schema or nonmonotonic timeline at {source}, frame {i}')
            end=offsets[i+1] if i+1<frames else (last_offset if tail is not None else len(mapped))
            raw=np.fromstring(mapped[start:end].decode('ascii'),sep=' ')
            if raw.size != atoms*len(columns) or not np.isfinite(raw).all():
                raise ValueError(f'Incomplete/nonfinite interior frame {i} in {source}')
            table=raw.reshape(atoms,len(columns));table=table[np.argsort(table[:,0],kind='stable')]
            if not np.array_equal(table[:,0],ids) or not np.array_equal(table[:,1],table[:,1].astype(np.int32)):
                raise ValueError(f'Invalid IDs/types in {source}, frame {i}')
            if kinds is None: kinds=table[:,1].astype(np.int32)
            if not np.array_equal(table[:,1],kinds):raise ValueError('Atom types changed')
            low,high=bounds[:,0],bounds[:,1]; lengths=high-low
            positions=np.mod(table[:,2:5].astype(np.float32)-low,lengths)
            positions=np.minimum(positions,np.nextafter(lengths,np.zeros(3,dtype=np.float32)))
            refs={'velocities':table[:,2:5].astype(np.float32)} if velocity_only else {'positions':positions}
            if has_velocities and not velocity_only:refs['velocities']=table[:,5:8].astype(np.float32)
            for name,reference in refs.items():
                encoded=reference.astype(storage_dtype)
                if not np.isfinite(encoded).all():raise ValueError(f'{storage_dtype} overflow: {source}, {name}')
                arrays[name][i]=encoded;expected[name].update(encoded.tobytes())
                delta=encoded.astype(np.float64)-reference
                if name=='positions':delta-=lengths*np.rint(delta/lengths)
                e=errors[name];e['max_abs']=max(e['max_abs'],float(abs(delta).max()));e['square_sum']+=float(np.square(delta).sum());e['count']+=delta.size
            steps.append(step);lows.append(low);highs.append(high)
            if i%100==0:
                write_json(building/'progress.json',dict(frames_written=i+1,frames=frames,source=str(source)))
        for array in arrays.values():array.flush()
        for name,value in dict(timesteps=np.array(steps,dtype=np.int64),box_low=np.array(lows,dtype=np.float32),box_high=np.array(highs,dtype=np.float32),atom_ids=ids,atom_types=kinds).items():
            np.save(building/(name+'.npy'),value,allow_pickle=False)
            if not np.array_equal(np.load(building/(name+'.npy')),value):raise ValueError(f'Exact array verification failed: {name}')
        if velocity_only:
            paired=lammps.TemporalLAMMPSBinaryTrajectory.load(positions_binary)
            paired.verify_checksums()
            for name in ['timesteps','box_low','box_high','atom_ids','atom_types']:
                if not np.array_equal(np.load(building/(name+'.npy')),np.load(Path(positions_binary)/(name+'.npy'))):
                    raise ValueError(f'Position/velocity identity or timeline mismatch: {name}')
            if paired.positions.shape != (frames,atoms,3) or paired.positions.dtype != np.dtype(storage_dtype):
                raise ValueError('Position/velocity shape or precision mismatch')
            os.link(Path(positions_binary)/'positions.npy',building/'positions.npy')
        descriptions={}
        for p in building.glob('*.npy'):
            with p.open('rb') as handle:os.fsync(handle.fileno())
            a=np.load(p,mmap_mode='r');digest=array_sha256(a)
            if p.stem in expected and digest!=expected[p.stem].hexdigest():raise ValueError(f'Persisted rounded values differ: {p}')
            descriptions[p.stem]=dict(file=p.name,dtype=str(a.dtype),shape=list(a.shape),sha256=digest)
        tail_record=None
        if tail is not None:
            p=building/'incomplete_final_frame.bin';p.write_bytes(tail)
            digest=hashlib.sha256(tail).hexdigest()
            if sha256(p)!=digest:raise ValueError('Incomplete-tail verification failed')
            tail_record=dict(source_offset=last_offset,size_bytes=len(tail),sha256=digest)
    if signature(source)!=before:raise ValueError(f'Source changed during conversion: {source}; original retained')
    quantization={name:dict(max_abs=e['max_abs'],rms=(e['square_sum']/e['count'])**.5,coordinate_count=e['count']) for name,e in errors.items()}
    manifest=dict(format=shooting.FORMAT_NAME if has_velocities else lammps.FORMAT_NAME,schema_version=1,state='complete',storage_dtype=storage_dtype,atom_count=atoms,frame_count=frames,first_timestep=steps[0],last_timestep=steps[-1],atom_columns=list(columns),coordinate_convention='wrapped float32 consumer coordinates relative to box_low; decode stored coordinates to float32 and wrap again',velocity_units='angstrom_per_ps',source=dict(path=str(source),sha256=source_hash),provenance=dict(producer='src.data.conversion.offline_dump',positions_binary=str(positions_binary) if velocity_only else None,scientific_completion='unchanged; conversion complete does not mean dynamics complete',quantization=quantization,incomplete_tail=tail_record),arrays=descriptions)
    write_json(building/'manifest.json',manifest)
    report=dict(state='verified',source_path=str(source),source_sha256=source_hash,source_signature=before,source_allocated_bytes=source.stat().st_blocks*512,has_velocities=has_velocities,frame_count=frames,first_timestep=steps[0],last_timestep=steps[-1],quantization=quantization,incomplete_tail=tail_record,source_deleted=False,completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(building/'conversion.json',report)
    cls=shooting.ShootingBinaryTrajectory if has_velocities else lammps.TemporalLAMMPSBinaryTrajectory
    cls.load(building).verify_checksums()
    os.rename(building,target)
    if delete_source:
        if signature(source)!=before:raise ValueError('Source changed before deletion')
        source.unlink();report['source_deleted']=True;write_json(receipt,report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('source',type=Path);p.add_argument('--output',type=Path);p.add_argument('--delete-source',action='store_true')
    p.add_argument('--positions-binary',type=Path)
    p.add_argument('--storage-dtype', choices=('float16','float32'), default='float16')
    a=p.parse_args(argv);print(json.dumps(convert(a.source,output=a.output,delete_source=a.delete_source,positions_binary=a.positions_binary,storage_dtype=a.storage_dtype),indent=2))

if __name__=='__main__':main()
