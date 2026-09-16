"""Atom-matched local coordinate/velocity pairs and instantaneous labels."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

from src.analysis.liquid_structure import persistence_image
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.registry import sha256, write_json
from src.models.encoders.mace_context import inner_weights
from src.research.mace_local_state.physics import group_observables
from .inventory import read


MOTION_NAMES = ['relative_speed_squared', 'relative_speed_fourth_moment', 'divergence_squared',
                'deviatoric_strain_squared', 'rotation_squared', 'nonaffine_velocity_squared',
                'divergence', 'radial_flux', 'radial_velocity_third_moment']


def motion_observables(x, v):
    radius=np.linalg.norm(x-x[0],axis=1); ids=np.flatnonzero(radius<7.)
    w=inner_weights(radius[ids]); w/=w.sum()
    r=x[ids].astype(float);r-=w@r
    u=v[ids].astype(float);u-=w@u
    a=np.linalg.solve(r.T@(w[:,None]*r)+.001*np.eye(3),r.T@(w[:,None]*u))
    divergence=np.trace(a); strain=.5*(a+a.T)-np.eye(3)*divergence/3
    rotation=.5*(a-a.T); residual=u-r@a
    speed2=np.sum(u*u,axis=1)
    radial=np.sum(r*u,axis=1)/np.sqrt(np.sum(r*r,axis=1)+.25)
    return np.array([w@speed2,w@(speed2**2),divergence**2,np.sum(strain**2),np.sum(rotation**2),
                     w@np.sum(residual**2,axis=1),divergence,w@radial,w@(radial**3)],dtype=np.float32)


def labels(pair):
    x,v=pair
    return np.r_[group_observables(x),persistence_image(x[:80]),motion_observables(x,v)].astype(np.float32)


def local_clouds(points, velocities, lengths, centers, radius):
    if min(lengths) <= 2*radius:
        raise ValueError(f'Candidate halo overlaps its periodic images: {lengths}')
    points=np.mod(points.astype(float),lengths)
    tree=cKDTree(points,boxsize=lengths)
    neighbors=tree.query_ball_point(points[centers],radius,workers=1)
    nearest=tree.query(points[centers],k=80,workers=1)[1]
    result=[]
    for center,ids,nn in zip(centers,neighbors,nearest,strict=True):
        ids=np.r_[nn,np.setdiff1d(ids,nn)]
        if ids[0]!=center: raise ValueError(f'Tracked atom is not first: {center}')
        x=points[ids]-points[center];x-=lengths*np.round(x/lengths)
        v=velocities[ids].astype(np.float32)
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise FloatingPointError('Nonfinite source position or velocity')
        result.append((x.astype(np.float32),v))
    return result


def prepare(config):
    cache=Path(config['cache']);cache.mkdir(parents=True,exist_ok=True)
    root=Path(config['output'])/'technical'
    inventory=read(root/'inventory.json');records=[];started=time.monotonic()
    rng=np.random.default_rng(config['seed'])
    # Spawn avoids inherited BLAS/thread state in scientific label workers.
    with ProcessPoolExecutor(config['label_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        for number,source in enumerate(inventory['records']):
            output=cache/f'source-{source["id"]:04d}.npz'
            sidecar=output.with_suffix('.json')
            # Resume only complete, verified preparation units, preserving RNG.
            centers=rng.choice(source['atom_count'],config['centers_per_frame'],replace=False)
            if sidecar.exists():
                saved=read(sidecar)
                if saved['config_sha256']!=config_digest(config) or saved['sha256']!=sha256(output):
                    raise ValueError(f'Changed preparation unit: {sidecar}')
                records.append(saved);continue
            if output.exists(): raise FileExistsError(f'Unfinished cache requires investigation: {output}')
            if source['format']=='paired_dump_conversion':
                path=Path(source['path'])
                if not path.exists():
                    subprocess.run([sys.executable,'scripts/convert_trajectory.py','paired-velocity',
                        '--positions',source['positions_dump'],'--velocities',source['velocities_dump'],
                        '--output',str(path),'--frames',str(config['frames_per_trajectory']),
                        '--atoms',str(source['atom_count'])],check=True)
                trajectory=ShootingBinaryTrajectory.load(path)
                anchors=np.array(trajectory.manifest['provenance']['anchor_indices_in_selection'])
                source_steps=trajectory.timesteps
            elif source['format']=='shooting_binary':
                trajectory=ShootingBinaryTrajectory.load(source['path'])
                if sha256(Path(source['path'])/'manifest.json')!=source['manifest_sha256']:
                    raise ValueError(f'Changed source manifest: {source["path"]}')
                eligible=np.array(source['eligible_frames'])
                n=min(len(eligible),config['frames_per_trajectory'])
                anchors=eligible[np.linspace(0,len(eligible)-1,n+2,dtype=int)[1:-1]]
                anchors=np.unique(anchors)
                source_steps=trajectory.timesteps
            elif source['format']=='legacy_npz':
                with np.load(source['path']) as values:
                    positions=values['positions_A'];velocities=values['velocities_A_per_ps']
                    cells=values['cell_vectors_A'];source_steps=values['step']
                if positions.dtype!=np.float32 or velocities.dtype!=np.float32 or positions.shape!=velocities.shape or positions.shape[1:]!=(source['atom_count'],3):
                    raise ValueError(f'Unexpected legacy producer schema: {source["path"]}')
                if not np.allclose(cells,cells*np.eye(3),rtol=0,atol=0):
                    raise ValueError(f'Nonorthogonal legacy cell: {source["path"]}')
                eligible=np.array(source['eligible_frames']);n=min(len(eligible),config['frames_per_trajectory'])
                anchors=np.unique(eligible[np.linspace(0,len(eligible)-1,n+2,dtype=int)[1:-1]])
            else: raise ValueError(f'Unknown producer: {source["format"]}')
            clouds=[];pair_records=[]
            for frame in anchors:
                views=[]
                for f in (frame,frame-1):
                    if source['format']=='legacy_npz':
                        p,v,lengths=positions[f],velocities[f],np.diag(cells[f])
                    else:
                        # Shooting positions are already relative to box_low.
                        p,v=trajectory.positions[f],trajectory.velocities[f]
                        lengths=trajectory.box_high[f].astype(float)-trajectory.box_low[f]
                    views.append(local_clouds(p,v,lengths,centers,config['candidate_radius_A']))
                for j,center in enumerate(centers):
                    clouds.extend([views[0][j],views[1][j]])
                    pair_records.append(dict(center_atom_id=int(center)+1,frame=int(frame),
                        timestep=int(source_steps[frame]),previous_timestep=int(source_steps[frame-1]),
                        lag_ps=float(source_steps[frame]-source_steps[frame-1])*source['timestep_fs']/1000))
            targets=np.stack(list(pool.map(labels,clouds)))
            if targets.shape!=(len(clouds),169) or not np.isfinite(targets).all():
                raise FloatingPointError(f'Invalid physical labels at {source["path"]}: {targets.shape}')
            np.savez(output,positions=np.concatenate([x for x,v in clouds]),
                velocities=np.concatenate([v for x,v in clouds]),
                pointers=np.cumsum([0]+[len(x) for x,v in clouds]),targets=targets)
            record=dict(source=source,file=output.name,sha256=sha256(output),pairs=pair_records,
                config_sha256=config_digest(config),source_metadata_sha256=sha256(Path(source['path'])/'manifest.json') if source['format']!='legacy_npz' else sha256(Path(source['path'])))
            write_json(sidecar,record);records.append(record)
            status=dict(state='preparing',sources=number+1,total=len(inventory['records']),
                        local_pairs=sum(len(r['pairs']) for r in records),elapsed_seconds=time.monotonic()-started)
            write_json(root/'prepare-status.json',status)
            print('PREPARE',status,flush=True)
    write_json(cache/'manifest.json',dict(state='complete',records=records,config=config,
        inventory_sha256=sha256(root/'inventory.json'),config_sha256=config_digest(config)))
    write_json(root/'prepare-status.json',dict(state='complete',sources=len(records),
        local_pairs=sum(len(r['pairs']) for r in records),elapsed_seconds=time.monotonic()-started))


def config_digest(config):
    import hashlib
    return hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()


def load_cache(config):
    root=Path(config['cache']);manifest=read(root/'manifest.json')
    clouds,targets,metadata=[],[],[]
    for record in manifest['records']:
        path=root/record['file']
        if sha256(path)!=record['sha256']: raise ValueError(f'Changed cache: {path}')
        with np.load(path) as arrays:
            x,v,p=arrays['positions'],arrays['velocities'],arrays['pointers']
            clouds.extend([(x[a:b],v[a:b]) for a,b in zip(p[:-1],p[1:],strict=True)])
            targets.append(arrays['targets'])
        metadata.extend([dict(source_id=record['source']['id'],split=record['source']['split'],
            lineage=record['source']['lineage'],balance_group=record['source']['balance_group'],**pair) for pair in record['pairs']])
    if len(clouds)!=2*len(metadata): raise ValueError('Current/previous pairing was lost')
    return clouds,np.concatenate(targets),metadata
