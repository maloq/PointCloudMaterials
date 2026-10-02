"""Extract real parent neighborhoods without changing source roles or identities."""
import hashlib
import shutil
import numpy as np
import torch

from src.data.fixed_cohort.dataset import read_release
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.project_runtime.paths import dataset_path, resolve_path
from .common import root, read, sha, write_json, save


def prepare(c):
    out = root(c)/'technical'
    receipt = out/'parents.json'
    if receipt.exists():
        record = read(receipt)
        if record['config'] != c['sampling'] or record['fixed_identity'] != c['fixed_identity']:
            raise ValueError('Prepared sampling changed')
        if sha(out/'parents.pt') != record['sha256']:
            raise ValueError('Prepared parent tensors changed')
        return torch.load(out/'parents.pt', weights_only=False)
    _, plan = read_release(c['fixed_release'])
    if plan['identity'] != c['fixed_identity']:
        raise ValueError('Unexpected fixed Al64 release')
    states = []
    for role in ('train', 'selection', 'test'):
        sources = [s for s in plan['sources'] if s['role'] == role]
        for rank, source in enumerate(sources):
            frame = c['sampling']['frames'][rank % len(c['sampling']['frames'])]
            center_id = source['center_atom_ids'][(rank*c['sampling']['center_stride']) % 64]
            raw = ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
            if sha(raw.root/'manifest.json') != source['manifest_sha256']:
                raise ValueError(f'Changed source manifest {source["id"]}')
            ids = np.asarray(raw.atom_ids)
            center_row = int(np.searchsorted(ids, center_id))
            if ids[center_row] != center_id or not np.all(raw.atom_types == 1):
                raise ValueError('Wrong center identity or non-Al atom types')
            xyz = np.asarray(raw.positions[frame], dtype=np.float64)
            box = np.asarray(raw.box_high[frame]-raw.box_low[frame], dtype=np.float64)
            if c['halo']['reference_A'] >= box.min()/2:
                raise ValueError('Environment sphere crosses half the periodic cell')
            x = xyz-xyz[center_row]
            x -= box*np.round(x/box)
            distance = np.linalg.norm(x, axis=1)
            order = np.lexsort((ids, distance))
            if order[0] != center_row or distance[order[79]] >= 8:
                raise ValueError(f'Expected80 neighbors inside8A: source {source["id"]}')
            selected = order[distance[order] < c['halo']['reference_A']]
            # Atom order starts with the exact nearest80; fixed for all derivatives.
            q = torch.from_numpy(x[selected].copy())
            rng = np.random.default_rng(c['seed']+source['id'])
            v = rng.normal(size=(79,3,2)); v -= v.mean(0, keepdims=True)
            v = np.linalg.qr(v.reshape(-1,2))[0].reshape(79,3,2)
            basis = torch.zeros(80,3,2,dtype=torch.float64)
            basis[1:] = torch.from_numpy(v)
            index = len(states)
            states.append(dict(index=index, source=source['id'], role=role, lineage=source['lineage'],
                frame=frame, center_atom_id=int(center_id), original_atoms=len(xyz),
                q=q, atom_rows=torch.from_numpy(selected.copy()), atom_ids=torch.from_numpy(ids[selected].copy()),
                basis=basis, box=torch.from_numpy(box), source_manifest_sha256=source['manifest_sha256'],
                source_frame_sha256=hashlib.sha256(np.ascontiguousarray(raw.positions[frame]).tobytes()).hexdigest(),
                parent_potential='Lee2003 MEAM', parent_storage_dtype=str(raw.positions.dtype),
                time_ps=float(raw.timesteps[frame]*source['timestep_fs']/1000)))
            print(f'Prepared {index+1}/135: {role} source{source["id"]}, environment atoms={len(q)}', flush=True)
    if [sum(s['role']==r for s in states) for r in ('train','selection','test')] != [90,15,30]:
        raise ValueError('Expected every90/15/30 non-calibration fixed source exactly once')
    save(out/'parents.pt', states)
    write_json(receipt, dict(config=c['sampling'], fixed_identity=c['fixed_identity'],
        sha256=sha(out/'parents.pt'), parents=[{k:v for k,v in s.items() if not torch.is_tensor(v)} for s in states],
        integrity='source manifest verified; selected raw frame independently hashed; no reread of complete multi-GB trajectories'))
    archive=resolve_path(c['simulation_archive']);archive.mkdir(parents=True,exist_ok=True)
    for name in ('parents.pt','parents.json'):shutil.copy2(out/name,archive/name)
    return states


def environment(state, radius, dtype=torch.float32):
    keep = state['q'].norm(dim=-1) < radius
    if not keep[:80].all():
        raise ValueError('Environment does not contain the full input')
    q = state['q'][keep].to(device='cuda', dtype=dtype)
    basis = torch.zeros(len(q),3,2,device='cuda',dtype=dtype)
    basis[:80] = state['basis'].to(device='cuda',dtype=dtype)
    rows = state['atom_rows'][keep].cuda()
    return q.flatten(), basis.reshape(-1,2), rows
