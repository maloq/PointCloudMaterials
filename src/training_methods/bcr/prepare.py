"""Prepare full-radius snapshot patches without targets from future trajectories."""
import hashlib
import json
from pathlib import Path
import numpy as np
from src.project_runtime.paths import resolve_path
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
from .data import extract_patch,coordinate_uncertainty,allowed_levels,audit_corruptions,balanced_subset
from .runtime import identity


def prepare(config,output):
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if (root/'manifest.json').exists():raise FileExistsError('Preparation is immutable; choose a new cache')
    roles={};potentials=set();patches=[];records=[];uncertainty=0.;heldout_uncertainty=0.;ids_all=[];images_all=[]
    for source in config['sources']:
        lineage=source['root'];split=source['split']
        if split not in ('train','development','test'):raise ValueError(split)
        if lineage in roles and roles[lineage]!=split:raise ValueError(f'Root split leakage: {lineage}')
        roles[lineage]=split;potentials.add(tuple(source['potential_sha256']))
        path=resolve_path(source['path'])
        loader={'shooting_binary':ShootingBinaryTrajectory,'temporal_lammps_binary':TemporalLAMMPSBinaryTrajectory}[source['format']]
        raw=loader.load(path)
        if source['species']!='Al' or not (raw.atom_types==1).all():raise ValueError('BCR-v1 supports pure Al from one potential')
        # Validate lineage from its producer, not an invented per-velocity root.
        metadata=json.loads(resolve_path(source['provenance_manifest']).read_text())
        if 'shared_liquid_source' in metadata:
            actual=metadata['shared_liquid_source']['prepared_liquid_sha256']
            if lineage!=actual:raise ValueError('Shared liquid descendants must share the prepared-liquid root hash')
        else:
            actual=metadata
            for field in source['root_evidence_field']:actual=actual[field]
            if str(actual)!=lineage:raise ValueError('Root differs from its declared producer metadata field')
        for frame in config['frames']:
            native=raw.positions[frame]
            if split=='train':uncertainty=max(uncertainty,coordinate_uncertainty(native))
            else:heldout_uncertainty=max(heldout_uncertainty,coordinate_uncertainty(native))
            cell=np.diag((raw.box_high[frame]-raw.box_low[frame]).astype(float))
            rng=np.random.default_rng(np.random.SeedSequence([config['seed'],source['source'],frame]))
            centers=rng.choice(len(raw.atom_ids),config['centers_per_frame'],replace=False)
            for center in centers:
                patch=extract_patch(native,cell,int(center),raw.atom_ids,config['radius_A'],config.get('max_atoms'))
                patches.append(patch['positions']);ids_all.append(patch['atom_ids']);images_all.append(patch['images'])
                records.append(dict(root=lineage,source=source['source'],split=split,block=frame//config['block_frames'],frame=frame,
                    time_ps=float(raw.timesteps[frame]*source['timestep_fs']/1000),center_atom_id=int(raw.atom_ids[center]),
                    temperature_K=source['temperature_K'],species='Al',cell_A=cell.tolist(),native_dtype=str(native.dtype),
                    source_manifest_sha256=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest(),
                    potential_sha256=source['potential_sha256'],path=str(path)))
    if len(potentials)!=1:raise ValueError('BCR-v1 requires one generating potential')
    train=[i for i,r in enumerate(records) if r['split']=='train']
    # Preparation draws equally many centers/frames per source. Median over a
    # source-balanced collection; do not normalize each individual patch.
    d0=float(np.median([np.linalg.norm(patches[i][1:],axis=-1).min() for i in train]))
    n_ref=float(np.mean([len(patches[i]) for i in train]))
    levels=allowed_levels(config['noise_levels'],d0,uncertainty)
    if min(levels)*d0<10*heldout_uncertainty:raise ValueError('Held-out storage is too coarse for the training-calibrated noise grid')
    offsets=np.r_[0,np.cumsum([len(p) for p in patches])]
    np.savez(root/'patches.npz',positions=np.concatenate(patches),offsets=offsets,atom_ids=np.concatenate(ids_all),images=np.concatenate(images_all))
    result=dict(config=config,records=records,d0=d0,n_ref=n_ref,radius_A=config['radius_A'],coordinate_units='angstrom',
        uncertainty_A=uncertainty,uncertainty_method='conservative center-relative native-ULP bound, not empirical RMS',
        noise_levels=levels,dropped_noise_levels=[s for s in config['noise_levels'] if s not in levels],
        overflow_count=0,roots=len(roles),sources=len(config['sources']),anchors=len(records),
        test_status=config['test_status'],preprocessing='bcr-radius-images-v1',
        patches_sha256=hashlib.sha256((root/'patches.npz').read_bytes()).hexdigest(),
        corruption_audit=audit_corruptions([patches[i] for i in balanced_subset(records,train,32,key='source')],levels,d0,config['radius_A']))
    result['identity']=identity(result);(root/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
