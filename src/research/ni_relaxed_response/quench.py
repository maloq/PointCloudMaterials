"""Indexed full-cell quenches, preserving pre-quantization local observations."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.relaxed_targets.worker import AbsolutePositions,publish
from src.simulation.relaxation import relax_frame
from src.data.conversion.relaxation import read_relaxed,convert
from src.data.relaxation_inventory import import_archive
from .data import read,root,resolve_path,sha,write_json,digest,Source,native_path


def finish(trajectory,frame,work,archive,settings,centers,neighbors,velocity,runner,*,reuse=True,receipt_fields=None):
    work=Path(work);archive=Path(archive)
    if (archive/'complete.json').exists():
        done=read(archive/'complete.json')
        if sha(archive/'patches.npz')!=done['patches_sha256']:raise ValueError('Changed quenched observations')
        with np.load(archive/'patches.npz') as a:return {k:a[k] for k in a.files},done
    work.mkdir(parents=True,exist_ok=True)
    if not (work/'metadata.json').exists():relax_frame(trajectory,frame,work,settings,reuse=reuse,runner=runner)
    if not (work/'patches.npz').exists():
        xyz,meta=read_relaxed(work);box=np.asarray(meta['box_high'])-meta['box_low']
        q=xyz[neighbors]-xyz[neighbors[:,:1]];q-=box*np.rint(q/box)
        np.savez_compressed(work/'patches.npz',positions=q.astype(np.float32),velocities=velocity[neighbors].astype(np.float32),center_atom_ids=centers,neighbor_atom_ids=trajectory.atom_ids[neighbors])
    if not (work/'conversion.json').exists():convert(work,delete_source=True,local_cloud_dtype='float32')
    meta=read(work/'metadata.json');done=dict(patches_sha256=sha(work/'patches.npz'),metadata_sha256=sha(work/'metadata.json'),fmax=meta['fmax_eV_per_A'],seconds=meta['seconds'],reuse=meta.get('reuse'),**(receipt_fields or {}))
    write_json(work/'complete.json',done);publish(work,archive)
    if reuse:import_archive(archive)
    with np.load(archive/'patches.npz') as a:return {k:a[k] for k in a.files},done


def native(c,p,task,runner):
    archive=native_path(c,task)
    if (archive/'complete.json').exists():
        done=read(archive/'complete.json')
        if done['task']!=task or sha(archive/'patches.npz')!=done['patches_sha256']:raise ValueError('Changed matched native quench')
        return done
    item=next(s for s in p['sources'] if s['key']==task['source']);s=item['source'];part,frame=s['frames'][task['frame']]
    raw=ShootingBinaryTrajectory.load(s['parts'][part]['path']);absolute=SimpleNamespace(**vars(raw),atom_count=raw.atom_count);absolute.positions=AbsolutePositions(raw)
    reader=Source(s);_,neighbors=reader.patches(task['frame'],task['centers']);_,velocity,_=reader.frame(task['frame'])
    _,done=finish(absolute,frame,resolve_path(c['relaxation_scratch'])/'native'/task['key'],archive,c['quench_settings'],np.asarray(task['centers']),neighbors,velocity,runner,receipt_fields=dict(task=task))
    return done


def virtual(c,key,qv,box,atom_ids,center,neighbors,runner,provenance,*,reuse=True):
    archive=resolve_path(c['relaxation_archive'])/'response'/key
    work=resolve_path(c['relaxation_scratch'])/'response'/key
    source=work/'source';source.mkdir(parents=True,exist_ok=True)
    metadata=dict(provenance,positions_sha256=digest(qv[:,:3].tolist()),box=box.tolist(),units='angstrom and angstrom_per_ps',state_dtype='float64',source='pre-quantization MD integration checkpoint')
    if (source/'manifest.json').exists() and read(source/'manifest.json')!=metadata:raise ValueError('Changed virtual response state')
    write_json(source/'manifest.json',metadata)
    trajectory=SimpleNamespace(root=source,positions=qv[None,:,:3],box_low=np.zeros((1,3)),box_high=box[None],atom_ids=atom_ids,atom_types=np.ones(len(atom_ids),dtype=np.int32),atom_count=len(atom_ids),timesteps=np.array([0],dtype=np.int64))
    return finish(trajectory,0,work,archive,c['quench_settings'],np.array([center]),neighbors[None],qv[:,3:],runner,reuse=reuse)
