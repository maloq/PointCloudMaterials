"""A separate, sealed crystal-side interface-layer target on the fixed Al64 cohort."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin import ancestry
from .data import context_atoms
from .sampling import RandomBatches


def interface_mask(points, box, solid, cutoff, minimum_disordered_size):
    """Crystal atoms adjacent to a sufficiently large non-crystal component.

    Connectivity and adjacency both use the existing Al 3.6-A neighbor graph.
    A finite atom layer is the target, not an inferred continuum dividing surface.
    """
    boundary = np.zeros(len(points), bool)
    liquid = np.flatnonzero(~solid)
    accepted = np.zeros(len(points), bool)
    if not solid.any() or not len(liquid):
        return boundary, accepted
    pairs = cKDTree(points[liquid], boxsize=box).query_pairs(cutoff, output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(liquid), len(liquid))).tocsr()
    _, components = connected_components(graph, directed=False)
    sizes = np.bincount(components)
    accepted[liquid] = sizes[components] >= minimum_disordered_size
    if accepted.any():
        distance = cKDTree(points[accepted], boxsize=box).query(points[solid], workers=1)[0]
        boundary[solid] = distance <= cutoff
    return boundary, accepted


def nearest_target(points, box, atoms, reference, tolerance):
    rows = np.flatnonzero(reference)
    distance = np.full(len(atoms), np.inf, np.float32)
    direction = np.zeros((len(atoms), 3), np.float32)
    ambiguous = np.zeros(len(atoms), bool)
    if len(rows):
        dd, nn = cKDTree(points[rows], boxsize=box).query(points[atoms], k=[1, 2], workers=1)
        delta = points[rows[nn[:, 0]]] - points[atoms]
        delta -= box * np.rint(delta / box)
        distance = dd[:, 0].astype(np.float32)
        direction = (delta / np.maximum(dd[:, 0, None], 1e-12)).astype(np.float32)
        ambiguous = np.abs(dd[:, 1] - dd[:, 0]) <= tolerance
    return dict(distance=distance, direction=direction, ambiguous=ambiguous,
                direction_valid=(distance > 0) & (distance < 64) & ~ambiguous)


def make_plan(config_path):
    c = json.loads(Path(config_path).read_text())
    base = resolve_path(c['target']['geometry_parent'])
    manifest = json.loads((base / 'manifest.json').read_text())
    if manifest['state'] != 'complete' or sha(base / 'manifest.json') != c['target']['parent_manifest_sha256']:
        raise ValueError('Interface geometry parent changed or is incomplete')
    parent = json.loads((base / 'plan.json').read_text())
    if sha(base / 'plan.json') != manifest['plan_sha256'] or parent['fixed_identity'] != c['fixed_dataset']['identity']:
        raise ValueError('Changed geometry/source contract')
    root = resolve_path(c['dataset']['root']); root.mkdir(parents=True, exist_ok=True)
    plan = dict(protocol=c['protocol'], target=c['target'], dataset=c['dataset'],
        fixed_identity=parent['fixed_identity'], sources=parent['sources'], paths=parent['paths'],
        parent_manifest_sha256=sha(base / 'manifest.json'),
        implementation={str(p): sha(p) for p in (Path(__file__), Path(ancestry.__file__))})
    # Stable relative producer names allow frozen-code workers to reproduce the plan.
    plan['implementation'] = {Path(k).name: v for k, v in plan['implementation'].items()}
    plan['identity'] = digest(plan)
    dest = root / 'plan.json'
    if dest.exists() and json.loads(dest.read_text()) != plan:
        raise ValueError('Interface preparation identity changed; use a new root')
    write_json(dest, plan)
    return root


def prepare_source(config_path, sid):
    c = json.loads(Path(config_path).read_text()); root = resolve_path(c['dataset']['root'])
    plan = json.loads((root / 'plan.json').read_text())
    folder = root / 'sources' / str(sid); folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        record = json.loads((folder / 'complete.json').read_text())
        if record['identity'] != plan['identity'] or any(sha(folder/n) != h for n,h in record['files'].items()):
            raise ValueError(f'Changed interface source {sid}')
        return record
    base = resolve_path(c['target']['geometry_parent']) / 'sources' / str(sid)
    receipt = json.loads((base / 'complete.json').read_text())
    if any(sha(base/n) != h for n,h in receipt['files'].items()):
        raise ValueError(f'Changed parent geometry {sid}')
    with np.load(base / 'rows.npz') as a: old = {k:a[k] for k in a.files}
    item = next(s for s in plan['sources'] if s['id'] == sid)
    audit = resolve_path(c['dataset']['audit']) / 'technical/sources' / str(sid)
    if sha(audit/'graph.npz') != json.loads((audit/'graph-complete.json').read_text())['sha256']:
        raise ValueError(f'Changed crystal graph {sid}')
    with np.load(audit/'graph.npz') as a: graph = {k:a[k] for k in a.files}
    settings = json.loads(resolve_path(c['dataset']['audit_config']).read_text())
    parent_plan = json.loads((base.parent.parent/'plan.json').read_text())
    if sha(resolve_path(c['dataset']['audit_config'])) != parent_plan['audit_config_sha256']:
        raise ValueError('Crystal lineage settings changed')
    if settings['lineage']['neighbor_cutoff_A'] != c['target']['neighbor_cutoff_A']:
        raise ValueError('Interface must use the declared Al neighbor graph')
    events, roots, _ = ancestry.establish(graph, settings['lineage']['thresholds'][0])
    access = ancestry.GeometryAccess(settings, item, audit, graph)
    ptm = json.loads((audit/'ptm-complete.json').read_text()); checked = set()
    values = []; extra_banks = []; offset = receipt['patches']; frames = []
    rng = np.random.default_rng(c['dataset']['uniform_seed'] + sid)
    for frame in np.unique(old['frame']):
        frame = int(frame); chunk = settings['ptm']['chunk_frames']
        begin = frame//chunk*chunk; stop = min(item['frame_count'], begin+chunk)
        filename = f'ptm-{begin:04d}-{stop:04d}.npz'
        if filename not in checked:
            if sha(audit/filename) != ptm['files'][filename]: raise ValueError(f'Changed PTM {sid}/{filename}')
            checked.add(filename)
        points, box, dense = access.frame(frame); tree = cKDTree(points, boxsize=box)
        ids = np.flatnonzero(old['frame'] == frame)
        record = {k:v[ids].copy() for k,v in old.items()}
        record['parent_index'] = ids
        atoms = np.searchsorted(access.raw.atom_ids, record['atom'])
        if not np.array_equal(access.raw.atom_ids[atoms], record['atom']): raise ValueError('Query identity mismatch')
        nodes = np.flatnonzero((graph['frame'] == frame) & (graph['size'] >= 64))
        known = [n for n in nodes if any(events[r-1]['confirmation_frame'] <= frame for r in roots[n])]
        solid = np.isin(dense, known) if known else np.zeros(len(points), bool)
        if not np.array_equal(solid[atoms], record['distance'] == 0):
            raise ValueError(f'Confirmed phase changed: {sid}/{frame}')
        boundary, accepted = interface_mask(points, box, solid, c['target']['neighbor_cutoff_A'],
                                            c['target']['minimum_disordered_component_atoms'])
        # Existing train/selection uniform queries are reused exactly. New outcome-blind
        # uniform held-out queries supply an interior assay, absent from fixed-at-risk rows.
        if item['role'] in ('calibration', 'test'):
            extra = rng.choice(len(points), c['dataset']['uniform_per_frame'], replace=False)
            chosen = [context_atoms(points,box,tree,int(atom),c['dataset']['shells']) for atom in extra]
            queries = np.stack([v[0] for v in chosen]); actual = np.stack([v[1] for v in chosen])
            patches, inverse = np.unique(queries, return_inverse=True)
            neighbors = tree.query(points[patches],k=80,workers=1)[1]
            xyz = points[neighbors]-points[patches,None]; xyz -= box*np.rint(xyz/box)
            crystal = nearest_target(points,box,extra,solid,c['dataset']['tie_tolerance_A'])
            visible = (solid[neighbors] & (np.linalg.norm(xyz,axis=-1)<8)).any(1)[inverse].reshape(queries.shape)
            extra_record = dict(indices=inverse.reshape(queries.shape).astype(np.int64)+offset, actual=actual,
                **crystal,visible_local=visible[:,0],visible_context=visible.any(1),
                source=np.full(len(extra),sid,np.int32),frame=np.full(len(extra),frame,np.int32),
                atom=access.raw.atom_ids[extra],role=np.full(len(extra),item['role']),
                row=np.full(len(extra),-1),kind=np.ones(len(extra),int),path=np.full(len(extra),-1),
                travel=np.zeros(len(extra)),parent_index=np.full(len(extra),-1))
            if record.keys() != extra_record.keys(): raise ValueError('Unexpected parent row schema')
            record = {k:np.concatenate((v,extra_record[k])) for k,v in record.items()}
            atoms = np.r_[atoms,extra]; extra_banks.append(xyz.astype(np.float32)); offset += len(xyz)
        record['crystal_distance'] = record['distance'].copy()
        record['crystal_visible_local'] = record['visible_local'].copy()
        record['crystal_visible_context'] = record['visible_context'].copy()
        record.update(nearest_target(points,box,atoms,boundary,c['dataset']['tie_tolerance_A']))
        record['inside_crystal'] = solid[atoms]
        record['interface_member'] = boundary[atoms]
        record['interface_exists'] = np.full(len(atoms),boundary.any())
        if not np.array_equal(record['distance']==0,record['interface_member']):
            raise ValueError('Only interface members may have zero distance')
        if (record['distance']+1e-4 < record['crystal_distance']).any():
            raise ValueError('A subset of crystal cannot be closer than the full crystal')
        # Recover exact context atom identities from the sealed geometry offsets.
        centers = np.mod(points[atoms,None]+record['actual'],box).reshape(-1,3)
        error, query = tree.query(centers,workers=1)
        if error.max() > 1e-4: raise ValueError(f'Context geometry no longer matches atoms: {sid}/{frame}')
        unique, inverse = np.unique(query,return_inverse=True)
        dd, neighbors = tree.query(points[unique],k=80,workers=1)
        visible = (boundary[neighbors] & (dd<8)).any(1)[inverse].reshape(-1,25)
        record['visible_local'] = visible[:,0]; record['visible_context'] = visible.any(1)
        values.append(record)
        interior = record['inside_crystal'] & ~record['interface_member']
        frames.append(dict(frame=frame,crystal_atoms=int(solid.sum()),accepted_disordered_atoms=int(accepted.sum()),
            interface_atoms=int(boundary.sum()),interior_queries=int(interior.sum()),
            interior_finite_queries=int((interior & np.isfinite(record['distance'])).sum()),
            max_interior_distance_A=float(record['distance'][interior & np.isfinite(record['distance'])].max())
                if (interior & np.isfinite(record['distance'])).any() else None))
    fields = {k:np.concatenate([v[k] for v in values]) for k in values[0]}
    retained = fields['parent_index'] >= 0
    if not np.array_equal(fields['parent_index'][retained],np.arange(len(old['atom']))):
        raise ValueError('Changed original query order')
    for k in ('indices','actual','atom','row','source','frame','role','kind','path','travel'):
        if not np.array_equal(fields[k][retained],old[k]): raise ValueError(f'Changed parent {k}')
    destination = folder/'positions.npy'
    if extra_banks:
        original = np.load(base/'positions.npy',mmap_mode='r')
        bank = np.lib.format.open_memmap(destination,mode='w+',dtype=np.float32,shape=(offset,80,3))
        bank[:len(original)] = original; start = len(original)
        for xyz in extra_banks: bank[start:start+len(xyz)] = xyz; start += len(xyz)
        bank.flush(); del bank
    elif not destination.exists():
        destination.symlink_to(base/'positions.npy')
    np.savez(folder/'rows.npz',**fields); write_json(folder/'frames.json',frames)
    record = dict(identity=plan['identity'],source=sid,rows=len(fields['atom']),patches=offset,
        parent_rows=receipt['rows'],added_uniform_rows=int((fields['parent_index']<0).sum()),
        counts={str(k):int((fields['kind']==k).sum()) for k in (0,1,2)},
        phase_counts=dict(inside=int(fields['inside_crystal'].sum()),layer=int(fields['interface_member'].sum()),
            interior=int((fields['inside_crystal'] & ~fields['interface_member']).sum())),
        parent_receipt_sha256=sha(base/'complete.json'),
        files={n:sha(folder/n) for n in ('positions.npy','rows.npz','frames.json')})
    write_json(folder/'complete.json',record)
    print(json.dumps(dict(stage='interface-source-complete',source=sid,rows=record['rows'],phase_counts=record['phase_counts'])),flush=True)
    return record


def seal(config_path):
    c = json.loads(Path(config_path).read_text()); root = resolve_path(c['dataset']['root'])
    plan = json.loads((root/'plan.json').read_text()); records=[]; rows=[]
    for item in plan['sources']:
        folder=root/'sources'/str(item['id']); record=json.loads((folder/'complete.json').read_text())
        if record['identity']!=plan['identity'] or any(sha(folder/n)!=h for n,h in record['files'].items()):
            raise ValueError(f'Incomplete/changed source {item["id"]}')
        records.append(record)
        with np.load(folder/'rows.npz') as a: rows.append({k:a[k] for k in ('role','kind','source','distance','inside_crystal','interface_member')})
    meta={k:np.concatenate([r[k] for r in rows]) for k in rows[0]}
    ids=np.flatnonzero((meta['role']=='train') & (meta['kind']<2)); weight=np.zeros(len(ids))
    for kind in (0,1):
        loc=np.flatnonzero(meta['kind'][ids]==kind)
        sources,inv,count=np.unique(meta['source'][ids[loc]],return_inverse=True,return_counts=True)
        weight[loc]=1/(2*len(sources)*count[inv])
    sampler=RandomBatches(SimpleNamespace(split={'train':ids},weights={'train':weight}),c['batch_size'])
    coverage=sampler.audit(meta['distance'],meta['inside_crystal'])
    write_json(root/'sampling.json',coverage)
    counts=[]
    for role in ('train','selection','calibration','test'):
        for kind in (0,1,2):
            take=(meta['role']==role)&(meta['kind']==kind)
            counts.append(dict(role=role,kind=kind,rows=int(take.sum()),
                layer=int((take & meta['interface_member']).sum()),
                interior=int((take & meta['inside_crystal'] & ~meta['interface_member']).sum()),
                no_interface=int((take & ~np.isfinite(meta['distance'])).sum())))
    if not any(x['role']=='test' and x['kind']==1 and x['interior']>0 for x in counts):
        raise ValueError('No held-out uniform interior examples; interface experiment cannot answer its question')
    write_json(root/'manifest.json',dict(state='complete',identity=plan['identity'],plan_sha256=sha(root/'plan.json'),sources=records))
    write_json(root/'state.json',dict(state='complete',rows=len(meta['distance']),counts=counts,sampling=coverage))
    return root
