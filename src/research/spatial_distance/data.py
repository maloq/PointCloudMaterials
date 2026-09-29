"""Uniform-atom augmentation restricted to frozen train/selection ancestors."""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial import cKDTree
from src.data.fixed_cohort.protocol import sha, write_json
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin import ancestry
from src.research.crystallization_origin.extract import raw_source
from src.research.structured_context.geometry import representatives, stencil
from src.research.spatial_approach.data import observations
from .common import study


def source_task(config, sid):
    s = study(config); c = s.config
    item = next(p for p in s.plan['sources'] if p['id'] == sid)
    if item['role'] not in ('train','selection'):
        raise ValueError('Augmentation may not consume calibration or test sources')
    folder = s.geometry/str(sid); folder.mkdir(exist_ok=True)
    receipt = folder/'complete.json'
    if receipt.exists():
        result = json.loads(receipt.read_text())
        if result['identity'] != s.identity or any(sha(folder/n)!=h for n,h in result['files'].items()):
            raise ValueError(f'Changed augmented source {sid}')
        return result
    audit = resolve_path(c['audit'])/'technical/sources'/str(sid)
    if sha(audit/'graph.npz') != json.loads((audit/'graph-complete.json').read_text())['sha256']:
        raise ValueError(f'Changed graph {sid}')
    with np.load(audit/'graph.npz') as a: graph = {k:a[k] for k in a.files}
    audit_config = json.loads(resolve_path(c['audit_config']).read_text())
    events, roots, _ = ancestry.establish(graph, audit_config['lineage']['thresholds'][0])
    access = ancestry.GeometryAccess(audit_config,item,audit,graph)
    raw = raw_source(item)
    rng = np.random.default_rng(c['seed']+sid)
    frames = np.unique(s.pop['frame'][s.pop['source']==sid])
    records = []
    for frame in frames:
        frame = int(frame)
        points,box,dense = access.frame(frame)
        nodes = np.flatnonzero((graph['frame']==frame)&(graph['size']>=64))
        known = [n for n in nodes if any(events[r-1]['confirmation_frame']<=frame for r in roots[n])]
        solid = np.isin(dense,known) if known else np.zeros(len(points),bool)
        tree = cKDTree(points,boxsize=box)
        # Uniform atoms, no label/distance oversampling or importance correction.
        atoms = rng.choice(len(points), c['augmentation']['atoms_per_frame'], replace=False)
        queries = np.stack([representatives(points,int(a),tree,box,stencil())[0] for a in atoms])
        actual = points[queries]-points[atoms,None]; actual -= box*np.rint(actual/box)
        distance = cKDTree(points[solid],boxsize=box).query(points[atoms],workers=1)[0] if solid.any() else np.full(len(atoms),np.inf)
        observed,_,inverse,xyz = observations(points,box,tree,queries,solid,access.labels[frame-access.chunk_start])
        path = folder/f'{frame}.npz'
        np.savez_compressed(path, positions=xyz.astype(np.float32), inverse=inverse, actual=actual.astype(np.float32),
            atom=raw.atom_ids[atoms], distance=distance.astype(np.float32), **observed)
        records.append(dict(source=sid,role=item['role'],frame=frame,rows=len(atoms),file=path.name,sha256=sha(path)))
    result = dict(identity=s.identity,source=sid,records=records,graph_sha256=sha(audit/'graph.npz'),
                  files={r['file']:r['sha256'] for r in records})
    write_json(receipt,result)
    return result


def prepare(config):
    s = study(config); results=[]
    items=[p for p in s.plan['sources'] if p['role'] in ('train','selection')]
    with ProcessPoolExecutor(max_workers=s.config['cpu_workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(source_task,config,p['id']) for p in items]
        for f in as_completed(futures):
            result=f.result();results.append(result)
            print(json.dumps(dict(stage='uniform-spatial-centers',completed=len(results),total=len(items),source=result['source'])),flush=True)
    write_json(s.technical/'prepared.json',dict(identity=s.identity,sources=results,
        records=[r for result in sorted(results,key=lambda x:x['source']) for r in result['records']]))
