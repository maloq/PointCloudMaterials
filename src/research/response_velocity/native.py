"""Actual same-frame MD velocities for the fixed response and all64 cohorts."""
from collections import defaultdict
import fcntl
import hashlib

import numpy as np
from scipy.spatial import cKDTree
import torch

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import centered
from src.postprocessing.protocols.extract import raw_source
from src.postprocessing.io import frame_geometry
from src.project_runtime.paths import resolve_path
from .common import root,read,sha,save,write_json,digest


def attach_native(c,states):
    if c['velocity_fields']!=1 or c['temperature_mode']!='source':
        raise ValueError('Native response recipe requires one saved velocity and source temperature')
    _,release=read_release(c['fixed_release'])
    if release['identity']!=c['fixed_identity']:raise ValueError('Different fixed source release')
    sources={s['id']:s for s in release['sources']};provenance=[]
    for state in states:
        item=sources[state['source']];raw=raw_source(item);frame=state['frame']
        if raw.manifest['velocity_units']!='angstrom_per_ps':raise ValueError('Wrong original velocity units')
        if sha(raw.root/'manifest.json')!=state['source_manifest_sha256']:raise ValueError('Different raw parent')
        if hashlib.sha256(np.ascontiguousarray(raw.positions[frame]).tobytes()).hexdigest()!=state['source_frame_sha256']:
            raise ValueError('Different saved coordinate frame')
        if item['role']!=state['role'] or item['lineage']!=state['lineage']:raise ValueError('Changed parent role/ancestry')
        order=state['atom_rows'].numpy()
        if not np.array_equal(raw.atom_ids[order],state['atom_ids'].numpy()):raise ValueError('Changed native atom ordering')
        physical_time=float(raw.timesteps[frame])*item['timestep_fs']/1000
        if abs(physical_time-state['time_ps'])>1e-9:raise ValueError('Native Q,V timeline differs')
        v=np.asarray(raw.velocities[frame],np.float64)[order]/1000.
        if v.shape!=tuple(state['q'].shape) or not np.isfinite(v).all():raise ValueError('Invalid full-cell velocities')
        state['v_A_fs']=torch.from_numpy(v.copy())
        state['temperature_K']=float(item['audit_metadata']['temperature_K'])
        state['native_velocity_sha256']=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
        provenance.append(dict(parent=state['index'],source=state['source'],role=state['role'],frame=frame,
            time_ps=physical_time,raw=str(raw.root),manifest_sha256=state['source_manifest_sha256'],
            raw_velocity=raw.manifest['arrays']['velocities'],velocity_A_fs_sha256=state['native_velocity_sha256'],
            temperature_K=state['temperature_K'],position_storage=str(raw.positions.dtype),velocity_storage=str(raw.velocities.dtype)))
    write_json(root(c)/'technical/native-inputs.json',dict(sources=provenance,
        conversion='saved A/ps divided by 1000 in float64; no redraw, rescale or COM removal',
        encoder_inputs='Q or QV local80 only',predictor_inputs='z128 only; no temperature/time',
        source_temperatures=sorted(set(s['temperature_K'] for s in states))))


def static_velocities(c,arrays,fit,held):
    """Retain every released row and its exact pre-existing nearest80 ordering."""
    directory=resolve_path(c['clustering']['velocity_cache']);directory.mkdir(parents=True,exist_ok=True)
    rows=np.r_[fit,held]
    identity=digest(dict(fixed=c['fixed_identity'],assay_manifest=sha(resolve_path(c['clustering']['assay']).parent/'manifest.json'),
        rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),protocol='actual_native_all64_v1'))
    path=directory/'velocities.npy';receipt=directory/'complete.json'
    with (directory/'prepare.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            done=read(receipt)
            if done['identity']!=identity or done['sha256']!=sha(path):raise ValueError('Changed native all64 velocity bank')
            return np.load(path,mmap_mode='r')
        fixed,release=read_release(c['fixed_release'])
        if release['identity']!=c['fixed_identity']:raise ValueError('Changed native all64 source release')
        sources={s['id']:s for s in release['sources']}
        folder=directory/'sources';folder.mkdir(exist_ok=True);provenance=[]
        output=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(len(rows),80,3))
        selected_sources=arrays['source'][rows];selected_frames=arrays['frame'][rows];selected_atoms=arrays['atom'][rows]
        for sid in sorted(np.unique(selected_sources)):
            take=np.flatnonzero(selected_sources==sid);part=folder/f'{sid}.npz';meta=folder/f'{sid}.json'
            if meta.exists():
                record=read(meta)
                if record['identity']!=identity or record['sha256']!=sha(part):raise ValueError('Changed native velocity shard')
                with np.load(part) as saved:
                    if not np.array_equal(saved['rows'],rows[take]):raise ValueError('Changed all64 row ordering')
                    output[take]=saved['velocities']
                provenance.append(record);continue
            item=sources[int(sid)];raw=raw_source(item)
            if raw.manifest['velocity_units']!='angstrom_per_ps':raise ValueError('Unexpected all64 velocity units')
            original_neighbors={}
            for section in ['benchmark','structural']:
                if section=='structural' and item['role'] not in ['train','selection']:continue
                native=fixed/section/'sources'/str(sid)
                ids=np.load(native/'neighbor_ids.npy',mmap_mode='r')
                for index,frame in enumerate(np.load(native/'frames.npy')):original_neighbors[int(frame)]=ids[index]
            velocities=np.empty((len(take),80,3),np.float32)
            for frame in sorted(np.unique(selected_frames[take])):
                subset=np.flatnonzero(selected_frames[take]==frame);indices=take[subset]
                atom_ids=selected_atoms[indices];centers=np.searchsorted(raw.atom_ids,atom_ids)
                if not np.array_equal(raw.atom_ids[centers],atom_ids):raise ValueError('Missing released center atom')
                q,box=frame_geometry(raw,int(frame))
                if int(frame) in original_neighbors:
                    lookup={atom:i for i,atom in enumerate(item['center_atom_ids'])}
                    neighbors=np.searchsorted(raw.atom_ids,original_neighbors[int(frame)][[lookup[int(a)] for a in atom_ids]])
                else:
                    neighbors=cKDTree(q,boxsize=box).query(q[centers],k=80,workers=1)[1]
                if not np.array_equal(centered(q,box,centers,neighbors),arrays['parents'][rows[indices],:80]):
                    raise ValueError(f'Native all64 patch correspondence failed: source {sid}, frame {frame}')
                velocities[subset]=np.asarray(raw.velocities[int(frame),neighbors],np.float64)/1000.
            if not np.isfinite(velocities).all():raise FloatingPointError('Nonfinite all64 velocities')
            np.savez(part,rows=rows[take],velocities=velocities)
            record=dict(identity=identity,source=int(sid),role=item['role'],rows=len(take),sha256=sha(part),
                manifest_sha256=item['manifest_sha256'],raw_velocity=raw.manifest['arrays']['velocities'],
                exact_position_correspondence=True)
            write_json(meta,record);provenance.append(record);output[take]=velocities
            print('native all64 velocities',sid,len(take),flush=True)
        output.flush();del output
        write_json(receipt,dict(identity=identity,sha256=sha(path),sources=provenance,training_rows=len(fit),test_rows=len(held),
            velocities='actual simultaneous same-atom MD A/ps converted to A/fs, stored float32 for neural inputs'))
    return np.load(path,mmap_mode='r')
