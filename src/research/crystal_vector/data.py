"""Frozen Al64 rows, causal crystal vectors, and rotation-covariant patch covering."""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin import ancestry


def covering(relative, atom_rows, shells):
    """Periodic displacements from the query; distance-only farthest coverage.

    Candidates are ordered by persistent identity, resolving exact geometric ties.
    Euclidean coverage uses query-relative minimum images inside radius 24; this
    entire ball fits within half the native Al box. No orientation frame is used.
    """
    r2 = np.einsum('ij,ij->i', relative, relative)
    selected = []
    centers = [np.zeros(3)]
    for low, high, count in shells:
        ids = np.flatnonzero((r2 > low**2) & (r2 <= high**2))
        if len(ids) < count:
            raise ValueError(f'Insufficient context atoms in shell {low}/{high}: {len(ids)}')
        xyz = relative[ids]
        nearest = np.min(np.sum((xyz[:, None]-np.asarray(centers)[None])**2, -1), axis=1)
        for _ in range(count):
            k = int(nearest.argmax())
            selected.append(int(ids[k])); centers.append(xyz[k])
            nearest = np.minimum(nearest, np.sum((xyz-xyz[k])**2, -1))
            nearest[k] = -1.
    return atom_rows[selected], relative[selected]


def context_atoms(points, box, tree, center, shells):
    radius = max(s[1] for s in shells)
    if (box <= 2*radius).any():
        raise ValueError('Context covering assumes its support fits inside half the periodic box')
    candidates = np.sort(tree.query_ball_point(points[center], radius))
    relative = points[candidates]-points[center]
    relative -= box*np.rint(relative/box)
    rows, positions = covering(relative, candidates, shells)
    return np.r_[center, rows], np.vstack((np.zeros(3), positions)).astype(np.float32)


def make_plan(config_path):
    c = json.loads(Path(config_path).read_text())
    fixed, release = read_release(c['fixed_dataset']['root'])
    if release['identity'] != c['fixed_dataset']['identity']:
        raise ValueError('Fixed cohort identity changed')
    root = resolve_path(c['dataset']['root']); root.mkdir(parents=True, exist_ok=True)
    audit_config = resolve_path(c['dataset']['audit_config'])
    parent = resolve_path(c['dataset']['scan_parent'])
    paths = []
    for item in release['sources']:
        folder = parent/'technical/sources'/str(item['id'])
        receipt = json.loads((folder/'complete.json').read_text())
        if sha(folder/'paths.json') != receipt['files']['paths.json']:
            raise ValueError(f'Changed path identities: {folder}')
        for record in json.loads((folder/'paths.json').read_text())['paths']:
            paths.append(dict(record, index=len(paths)))
    binding = dict(protocol='crystal_vector_snapshot_al64_v1', dataset=c['dataset'],
        fixed_identity=release['identity'], population_sha256=sha(fixed/'benchmark/population.npz'),
        audit_config_sha256=sha(audit_config), sources=release['sources'], paths=paths,
        implementation={p.name:sha(p) for p in [Path(__file__), Path(ancestry.__file__)]})
    binding['identity'] = digest(binding)
    dest = root/'plan.json'
    if dest.exists() and json.loads(dest.read_text()) != binding:
        raise ValueError('Preparation identity changed; use a fresh dataset root')
    write_json(dest, binding)
    return root


def prepare_source(config_path, sid):
    c = json.loads(Path(config_path).read_text()); root = resolve_path(c['dataset']['root'])
    plan = json.loads((root/'plan.json').read_text())
    folder = root/'sources'/str(sid); folder.mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        record = json.loads((folder/'complete.json').read_text())
        if record['identity'] != plan['identity'] or any(sha(folder/n) != h for n,h in record['files'].items()):
            raise ValueError(f'Changed prepared source {sid}')
        return record
    fixed, _ = read_release(c['fixed_dataset']['root'])
    with np.load(fixed/'benchmark/population.npz') as a:
        pop = {k:a[k] for k in ('source','frame','atom','role','sample_id')}
    item = next(s for s in plan['sources'] if s['id'] == sid)
    ids = np.flatnonzero(pop['source'] == sid)
    audit = resolve_path(c['dataset']['audit'])/'technical/sources'/str(sid)
    if sha(audit/'graph.npz') != json.loads((audit/'graph-complete.json').read_text())['sha256']:
        raise ValueError(f'Changed crystal graph {sid}')
    with np.load(audit/'graph.npz') as a: graph = {k:a[k] for k in a.files}
    settings = json.loads(resolve_path(c['dataset']['audit_config']).read_text())
    events, roots, _ = ancestry.establish(graph, settings['lineage']['thresholds'][0])
    access = ancestry.GeometryAccess(settings, item, audit, graph)
    ptm_receipt = json.loads((audit/'ptm-complete.json').read_text())
    parent = resolve_path(c['dataset']['scan_parent'])/'technical/sources'/str(sid)
    with np.load(parent/'labels.npz') as a:
        previous_distance = dict(zip(a['rows'].tolist(), a['distance'].tolist()))
    source_paths = [p for p in plan['paths'] if p['source'] == sid]
    rng = np.random.default_rng(c['dataset']['uniform_seed']+sid)
    banks, values = [], []; bank_offset = 0; checked_chunks = set()
    for frame in np.unique(pop['frame'][ids]):
        frame = int(frame); chunk = settings['ptm']['chunk_frames']
        begin = frame//chunk*chunk; stop = min(item['frame_count'], begin+chunk)
        filename = f'ptm-{begin:04d}-{stop:04d}.npz'
        if filename not in checked_chunks:
            if sha(audit/filename) != ptm_receipt['files'][filename]: raise ValueError(f'Changed PTM {sid}/{filename}')
            checked_chunks.add(filename)
        points, box, dense = access.frame(frame)
        tree = cKDTree(points, boxsize=box)
        original = ids[pop['frame'][ids] == frame]
        atoms = list(np.searchsorted(access.raw.atom_ids, pop['atom'][original]))
        if not np.array_equal(access.raw.atom_ids[atoms], pop['atom'][original]): raise ValueError('Atom identity mismatch')
        metadata = dict(row=list(original), kind=[0]*len(atoms), path=[-1]*len(atoms), travel=[0.]*len(atoms))
        if item['role'] in ('train','selection'):
            extra = rng.choice(len(points), c['dataset']['uniform_per_frame'], replace=False)
            atoms.extend(extra.tolist())
            for key,val in [('row',-1),('kind',1),('path',-1),('travel',0.)]: metadata[key].extend([val]*len(extra))
        for path in [p for p in source_paths if p['frame'] == frame]:
            if sha(parent/path['file']) != path['sha256']: raise ValueError('Changed scan path')
            with np.load(parent/path['file']) as a:
                scan = np.searchsorted(access.raw.atom_ids, a['atom'])
                if not np.array_equal(access.raw.atom_ids[scan], a['atom']): raise ValueError('Scan atom identity mismatch')
                atoms.extend(scan.tolist()); metadata['travel'].extend(a['travel_A'].tolist())
                metadata['row'].extend([-1]*len(scan)); metadata['kind'].extend([2]*len(scan))
                metadata['path'].extend([path['index']]*len(scan))
        atoms = np.asarray(atoms); unique_atoms, query_inverse = np.unique(atoms, return_inverse=True)
        nodes = np.flatnonzero((graph['frame'] == frame) & (graph['size'] >= 64))
        known = [n for n in nodes if any(events[r-1]['confirmation_frame'] <= frame for r in roots[n])]
        solid = np.isin(dense, known) if known else np.zeros(len(points), bool)
        crystal_rows = np.flatnonzero(solid)
        distance = np.full(len(atoms), np.inf, np.float32); direction = np.zeros((len(atoms),3),np.float32)
        ambiguous = np.zeros(len(atoms),bool)
        if len(crystal_rows):
            dd, nn = cKDTree(points[solid], boxsize=box).query(points[atoms], k=2, workers=1)
            delta = points[crystal_rows[nn[:,0]]]-points[atoms]; delta -= box*np.rint(delta/box)
            distance = dd[:,0].astype(np.float32)
            direction = (delta/np.maximum(dd[:,0,None],1e-12)).astype(np.float32)
            ambiguous = np.abs(dd[:,1]-dd[:,0]) <= c['dataset']['tie_tolerance_A']
        if not np.allclose(distance[:len(original)], [previous_distance[int(r)] for r in original], atol=1e-4, rtol=1e-6):
            raise ValueError(f'Frozen distance label changed: source {sid}, frame {frame}')
        chosen = [context_atoms(points,box,tree,int(atom),c['dataset']['shells']) for atom in unique_atoms]
        query = np.stack([v[0] for v in chosen])[query_inverse]
        actual = np.stack([v[1] for v in chosen])[query_inverse]
        unique_patches, inverse = np.unique(query, return_inverse=True)
        neighbors = tree.query(points[unique_patches],k=80,workers=1)[1]
        if not np.array_equal(neighbors[:,0],unique_patches): raise ValueError('Nearest-80 center ordering changed')
        xyz = points[neighbors]-points[unique_patches,None]; xyz -= box*np.rint(xyz/box)
        visible = (solid[neighbors] & (np.linalg.norm(xyz,axis=-1)<8)).any(1)[inverse].reshape(query.shape)
        record = dict(indices=inverse.reshape(query.shape).astype(np.int64)+bank_offset,
            actual=actual, distance=distance, direction=direction, ambiguous=ambiguous,
            direction_valid=(distance>0)&(distance<64)&~ambiguous,
            visible_local=visible[:,0], visible_context=visible.any(1),
            source=np.full(len(atoms),sid,np.int32),frame=np.full(len(atoms),frame,np.int32),
            atom=access.raw.atom_ids[atoms],role=np.full(len(atoms),item['role']),
            **{k:np.asarray(v) for k,v in metadata.items()})
        values.append(record); banks.append(xyz.astype(np.float32)); bank_offset += len(xyz)
    np.save(folder/'positions.npy',np.concatenate(banks))
    fields = {k:np.concatenate([v[k] for v in values]) for k in values[0]}
    np.savez(folder/'rows.npz',**fields)
    record = dict(identity=plan['identity'],source=sid,rows=len(fields['atom']),patches=bank_offset,
        counts={str(k):int((fields['kind']==k).sum()) for k in (0,1,2)},
        files={n:sha(folder/n) for n in ('positions.npy','rows.npz')})
    write_json(folder/'complete.json',record)
    return record


def prepare(config_path):
    root = make_plan(config_path); c = json.loads(Path(config_path).read_text())
    plan = json.loads((root/'plan.json').read_text()); records=[]
    with ProcessPoolExecutor(max_workers=c['dataset']['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(prepare_source,str(Path(config_path).resolve()),s['id']) for s in plan['sources']]
        for future in as_completed(futures):
            record=future.result(); records.append(record)
            progress=dict(state='preparing',completed=len(records),total=len(futures),source=record['source'])
            write_json(root/'state.json',progress); print(json.dumps(progress),flush=True)
    records.sort(key=lambda r:r['source'])
    write_json(root/'manifest.json',dict(state='complete',identity=plan['identity'],plan_sha256=sha(root/'plan.json'),sources=records))
    write_json(root/'state.json',dict(state='complete',sources=len(records),rows=sum(r['rows'] for r in records)))


class ResidentContexts:
    def __init__(self,config,device):
        import torch
        root=resolve_path(config['dataset']['root']); manifest=json.loads((root/'manifest.json').read_text())
        if manifest['state']!='complete' or sha(root/'plan.json')!=manifest['plan_sha256']: raise ValueError('Dataset not sealed')
        self.identity=manifest['identity']; self.root=root
        n=sum(s['patches'] for s in manifest['sources'])
        self.positions=torch.empty((n,80,3),dtype=torch.float32,device=device)
        offset=0;rows=[]
        for s in manifest['sources']:
            folder=root/'sources'/str(s['source'])
            for name,checksum in s['files'].items():
                if sha(folder/name)!=checksum:raise ValueError(f'Changed prepared coordinates/labels: {folder/name}')
            xyz=np.load(folder/'positions.npy',mmap_mode='r')
            self.positions[offset:offset+len(xyz)].copy_(torch.tensor(xyz,device=device))
            with np.load(folder/'rows.npz') as a:r={k:a[k] for k in a.files}
            r['indices']+=offset;rows.append(r);offset+=len(xyz)
        self.meta={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
        self.indices=torch.tensor(self.meta['indices'],device=device)
        self.actual=torch.tensor(self.meta['actual'],device=device)
        self.distance=torch.tensor(self.meta['distance'],device=device)
        self.direction=torch.tensor(self.meta['direction'],device=device)
        self.valid=torch.tensor(self.meta['direction_valid'],device=device)
        self.split={role:np.flatnonzero((self.meta['role']==role)&(self.meta['kind']<2)) for role in ('train','selection','calibration','test')}
        self.weights={}
        for role,ids in self.split.items():
            weight=np.zeros(len(ids))
            kinds=np.unique(self.meta['kind'][ids])
            for kind in kinds:
                loc=np.flatnonzero(self.meta['kind'][ids]==kind)
                sources,inv,count=np.unique(self.meta['source'][ids[loc]],return_inverse=True,return_counts=True)
                if config['dataset'].get('expanded_clear_only'):
                    denominators={s['source']:s['population_counts'][str(kind)] for s in manifest['sources']}
                    count=np.asarray([denominators[int(s)] for s in sources])
                weight[loc]=1/(len(kinds)*len(sources)*count[inv])
            self.weights[role]=weight
        if config.get('observation_filter')=='no_visible_interface':
            from .unseen import apply_view
            apply_view(self)
        elif config.get('observation_filter')=='liquid_no_visible_crystal':
            from .liquid import apply_view
            apply_view(self)

    def batch(self,ids):
        import torch
        index=torch.as_tensor(ids,device=self.indices.device)
        patches,inverse=torch.unique(self.indices[index].flatten(),sorted=True,return_inverse=True)
        return dict(positions=self.positions[patches],inverse=inverse.reshape(-1,25),actual=self.actual[index],
            distance=self.distance[index],direction=self.direction[index],valid=self.valid[index])


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    prepare(p.parse_args().config)
