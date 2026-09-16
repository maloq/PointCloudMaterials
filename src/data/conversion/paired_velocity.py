"""Convert selected, exactly atom/time-matched legacy position and velocity dumps.

Original dumps remain untouched. The normal shooting converter verifies the new
float16 arrays; this wrapper also measures coordinate and velocity quantization.
"""
import argparse
import mmap
from pathlib import Path

import numpy as np

from src.data.trajectories.shooting import convert_shooting_trajectory
from src.experiment_runner.registry import sha256, write_json


def dump_index(path):
    with Path(path).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        rows, start = [], 0
        while (offset := mapped.find(b'ITEM: TIMESTEP\n', start)) >= 0:
            mapped.seek(offset+15)
            step = int(mapped.readline())
            rows.append((step, offset))
            start = mapped.tell()
    if not rows or any(b[0] <= a[0] for a,b in zip(rows[:-1],rows[1:])):
        raise ValueError(f'Empty, repeated or nonmonotonic dump timeline: {path}')
    return rows


def selected_tables(path, rows, columns, atom_count):
    result = {}
    with Path(path).open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        for step, offset in rows:
            mapped.seek(offset)
            header = [mapped.readline().decode('ascii').strip() for _ in range(9)]
            if header[:4] != ['ITEM: TIMESTEP',str(step),'ITEM: NUMBER OF ATOMS',str(atom_count)] or header[4] != 'ITEM: BOX BOUNDS pp pp pp' or header[8] != 'ITEM: ATOMS '+' '.join(columns):
                raise ValueError(f'Invalid paired dump header: {path}, step={step}: {header}')
            low_high = np.array([np.fromstring(h,sep=' ') for h in header[5:8]])
            start = mapped.tell(); end = mapped.find(b'ITEM: TIMESTEP\n',start)
            table = np.fromstring(mapped[start:len(mapped) if end<0 else end].decode('ascii'),sep=' ')
            if table.size != atom_count*len(columns) or not np.isfinite(table).all():
                raise ValueError(f'Incomplete/nonfinite paired dump: {path}, step={step}')
            table = table.reshape(atom_count,len(columns)); table=table[np.argsort(table[:,0])]
            np.testing.assert_array_equal(table[:,0],np.arange(1,atom_count+1))
            np.testing.assert_array_equal(table[:,1],np.ones(atom_count))
            result[step]=(low_high,table)
    return result


def convert(positions, velocities, output, *, frame_count, atom_count):
    pindex, vindex = dump_index(positions), dump_index(velocities)
    np.testing.assert_array_equal([s for s,_ in pindex],[s for s,_ in vindex])
    # Reject truncated endpoints even when the requested interior frames exist.
    selected_tables(positions,[pindex[0],pindex[-1]],['id','type','x','y','z'],atom_count)
    selected_tables(velocities,[vindex[0],vindex[-1]],['id','type','vx','vy','vz'],atom_count)
    anchors = np.unique(np.linspace(1,len(pindex)-1,frame_count+2,dtype=int)[1:-1])
    chosen = np.unique(np.r_[anchors-1,anchors])
    p = selected_tables(positions,[pindex[i] for i in chosen],['id','type','x','y','z'],atom_count)
    v = selected_tables(velocities,[vindex[i] for i in chosen],['id','type','vx','vy','vz'],atom_count)
    target=Path(output);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.parent/(target.name+'-selected-source.lammpstrj')
    if temporary.exists() or target.exists(): raise FileExistsError(f'Preserve conversion: {target}')
    errors={'positions':[],'velocities':[]}
    with temporary.open('w') as stream:
        for step in p:
            bounds, pp=p[step];vbounds,vv=v[step]
            np.testing.assert_array_equal(bounds,vbounds);np.testing.assert_array_equal(pp[:,:2],vv[:,:2])
            stream.write(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{atom_count}\nITEM: BOX BOUNDS pp pp pp\n')
            np.savetxt(stream,bounds,fmt='%.17g');stream.write('ITEM: ATOMS id type x y z vx vy vz\n')
            np.savetxt(stream,np.c_[pp,vv[:,2:]],fmt=['%d','%d']+['%.9g']*6)
            lengths=(bounds[:,1]-bounds[:,0]).astype(np.float32)
            wrapped=np.mod(pp[:,2:].astype(np.float32)-bounds[:,0].astype(np.float32),lengths)
            for key,array in [('positions',wrapped),('velocities',vv[:,2:].astype(np.float32))]:
                delta=array.astype(np.float16).astype(np.float32)-array
                if key=='positions': delta-=lengths*np.round(delta/lengths)
                errors[key].append(dict(max_abs=float(abs(delta).max()),rms=float(np.sqrt(np.mean(delta.astype(float)**2)))))
    result=convert_shooting_trajectory(temporary,target,timesteps=list(p),atom_count=atom_count,
        storage_dtype='float16',provenance=dict(protocol='selected_paired_velocity_v1',
        source_hashes={str(Path(q).resolve()):sha256(Path(q)) for q in (positions,velocities)},
        selected_original_frame_indices=chosen.tolist(),anchor_indices_in_selection=np.searchsorted(chosen,anchors).tolist(),
        original_frame_count=len(pindex),quantization_error=errors))
    result.verify_checksums()
    write_json(target.parent/(target.name+'-conversion.json'),dict(state='complete',manifest_sha256=sha256(target/'manifest.json'),quantization=errors))
    # Keep the selected converter source (small, exact reproduction input).
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--positions',required=True);parser.add_argument('--velocities',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--frames',type=int,default=3)
    parser.add_argument('--atoms',type=int,required=True)
    args=parser.parse_args(argv)
    convert(args.positions,args.velocities,args.output,frame_count=args.frames,atom_count=args.atoms)
    return 0


if __name__=='__main__': raise SystemExit(main())
