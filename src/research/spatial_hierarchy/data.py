"""Current-frame surroundings for the exact original focal observations."""
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from src.project_runtime.paths import resolve_path,dataset_path
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.research.structural_state.data import Corpus
from src.research.robust_onset.data import patches
from src.research.trajectory_stability.native_dense import pack_observations
from src.research.robust_onset.common import sha,write_json
from .model import region_tokens


RADII={'local':[4.,6.,8.],'near':[8.,10.,12.],'wide':[8.,12.,16.]}


def canonical(x):
    return x[np.lexsort((x[:,2],x[:,1],x[:,0]))]


def prepare(study):
    corpus=Corpus(study);root=study.context_cache;root.mkdir(parents=True,exist_ok=True)
    selection_path=resolve_path(study.config['context_selection'])
    signature=dict(parent=sha(study.cache/'manifest.json'),selection=sha(selection_path),
        producer=sha(__file__),basis=sha(__file__.replace('data.py','model.py')),radii=RADII)
    path=root/'manifest.json'
    if path.exists():
        prior=json.loads(path.read_text())
        if prior['signature']!=signature:raise ValueError('Context recipe changed: use a new cache')
        for name,h in prior['files'].items():
            if sha(root/name)!=h:raise ValueError(f'Context input changed: {name}')
        prepare_dense(study)
        return prior
    selection=json.loads(selection_path.read_text());lookup={s['id']:s for s in selection['sources']}
    with np.load(study.cache/'observed-graphs.npz') as a:old=patches(dict(a))
    all_patches=[None]*len(corpus.records);proof={};max_error=0.
    for sid in sorted({r['source'] for r in corpus.records}):
        s=lookup[sid];raw=ShootingBinaryTrajectory.load(dataset_path(s['dataset'])/s['relative_trajectory_path'])
        if sha(raw.root/'manifest.json')!=s['manifest_sha256']:raise ValueError(f'Changed raw source {sid}')
        ix=[i for i,r in enumerate(corpus.records) if r['source']==sid]
        if any(corpus.records[i]['root']!=s['lineage'] for i in ix):raise ValueError('Root ancestry changed')
        proof[str(sid)]=dict(manifest_sha256=s['manifest_sha256'],frames={})
        for frame in sorted({corpus.records[i]['frame'] for i in ix}):
            if abs(raw.timesteps[frame]*s['timestep_fs']/1000-frame*.75)>1e-8:raise ValueError('Current-frame time mismatch')
            box=(raw.box_high[frame]-raw.box_low[frame]).astype(float)
            if box.min()<=32:raise ValueError('16 A ball requires explicit multiple periodic images in this cell')
            positions=np.mod(raw.positions[frame].astype(float)-raw.box_low[frame],box)
            tree=cKDTree(positions,boxsize=box)
            indices=[i for i in ix if corpus.records[i]['frame']==frame]
            centers=np.searchsorted(raw.atom_ids,[corpus.records[i]['center_atom_id'] for i in indices])
            np.testing.assert_array_equal(raw.atom_ids[centers],[corpus.records[i]['center_atom_id'] for i in indices])
            for i,center,rows in zip(indices,centers,tree.query_ball_point(positions[centers],16.,return_sorted=True),strict=True):
                rows=np.r_[center,np.asarray(rows)[np.asarray(rows)!=center]]
                x=positions[rows]-positions[center];x-=box*np.round(x/box);x=x.astype(np.float32)
                x=x[np.linalg.norm(x,axis=1)<16.];x[0]=0
                core=x[np.linalg.norm(x,axis=1)<8.]
                np.testing.assert_allclose(canonical(core),canonical(old[i]),atol=1e-4,rtol=0,
                    err_msg=f'Focal geometry differs for source={sid} frame={frame} atom={raw.atom_ids[center]}')
                max_error=max(max_error,float(np.max(abs(canonical(core)-canonical(old[i])))))
                all_patches[i]=x
            payload=np.ascontiguousarray(raw.positions[frame]).tobytes()+box.tobytes()+raw.atom_ids.tobytes()
            proof[str(sid)]['frames'][str(frame)]=hashlib.sha256(payload).hexdigest()
        print('prepared current surroundings',sid,flush=True)
    np.savez(root/'patches.npz',positions=np.concatenate(all_patches),offsets=np.r_[0,np.cumsum([len(x) for x in all_patches])])
    np.savez(root/'tokens.npz',**{k:region_tokens(all_patches,radii) for k,radii in RADII.items()})
    write_json(root/'records.json',corpus.records)
    receipt=dict(state='complete',signature=signature,files={p:sha(root/p) for p in ['patches.npz','tokens.npz','records.json']},
        raw_current_frames=proof,maximum_focal_coordinate_replay_error_A=max_error,
        rows=len(all_patches),radii_A=RADII,context='Observed current positions only; no future or relaxed positions',
        parent_identity=corpus.manifest['identity'],new_simulations=0)
    write_json(path,receipt);prepare_dense(study);return receipt


def prepare_dense(study):
    """Existing dense parent has ~16.87 A support; prior native export kept 10 A."""
    root=resolve_path(study.config['dense_root']);out=resolve_path(study.config['dense_inputs']);out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'technical/plan.json').read_text())
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text())
        if manifest['source_plan_sha256']!=sha(root/'technical/plan.json'):raise ValueError('Dense parent changed')
        for f,h in manifest['files'].items():
            if sha(out/f)!=h:raise ValueError('Dense 16 A input changed')
        return
    manifest=dict(source_plan_sha256=sha(root/'technical/plan.json'),radius_A=16.,producer_sha256=sha(__file__),files={})
    for i,s in enumerate(plan['sources']):
        folder=root/'technical/sources'/str(s['id']);receipt=json.loads((folder/'complete.json').read_text())
        for name in ('positions.npy','atom_ids.npy','observations.npz'):
            if sha(folder/name)!=receipt['hashes'][name]:raise ValueError(f'Changed dense parent: {folder/name}')
        with np.load(folder/'observations.npz') as obs:
            a=pack_observations(np.load(folder/'positions.npy',mmap_mode='r'),np.load(folder/'atom_ids.npy',mmap_mode='r'),obs,radius=16.)
        name=f'frame-{i:02d}.npz';np.savez(out/name,**a);manifest['files'][name]=sha(out/name)
        print('prepared dense surroundings',s['id'],flush=True)
    write_json(manifest_path,manifest)
