"""Build centered multi-material observations without changing source trajectories."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import eval_legendre

from src.data.predictive_memory.targets import taper, rbf
from src.project_runtime.paths import dataset_path, resolve_path

ELEMENTS = {'Mg': 12, 'Al': 13, 'Ti': 22, 'Zr': 40, 'Ta': 73}
REFERENCE_RADIUS = 9.192189


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def geometry_packet(x):
    """85 geometry-only channels, matching the geometric native-packet primitives."""
    x = np.asarray(x, dtype=np.float64)
    radius = np.linalg.norm(x, axis=-1)
    keep = (radius > 0) & (radius < 7)
    x, radius = x[keep], radius[keep]
    if len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Geometry target requires at least two finite neighbors within 7 A')
    support = taper(radius, 5., 7.); w = support/support.sum()
    a, b = np.triu_indices(len(x), 1); pw = support[a]*support[b]
    direction = x/radius[:, None]
    cosine = np.clip((direction[a]*direction[b]).sum(-1), -1, 1)
    angular = np.array([np.dot(pw, eval_legendre(l, cosine))/pw.sum() for l in range(1, 17)])
    moments = [support.sum(), taper(radius, 0, 3).sum(), w@radius, w@radius**2, w@radius**3]
    value = np.concatenate((rbf(radius, support, 0, 7, 32),
        rbf(np.linalg.norm(x[a]-x[b], axis=-1), pw, 0, 14, 32), angular, moments))
    if value.shape != (85,) or not np.isfinite(value).all():
        raise FloatingPointError('Invalid geometry packet')
    return value.astype(np.float32)


def source_arrays(source):
    path = resolve_path(source['path'])
    if source['kind'] == 'static':
        if path.stat().st_size != source['bytes'] or path.stat().st_mtime_ns != source['mtime_ns']:
            raise ValueError(f'Static source changed after calibration: {path}')
        x = np.load(path, mmap_mode='r', allow_pickle=False)
        if x.ndim != 2 or x.shape[1] != 3 or x.dtype != np.float32:
            raise ValueError(f'Unexpected static coordinate schema: {path}')
        return dict(positions=x, atom_ids=np.arange(len(x), dtype=np.int64))
    manifest = json.loads((path/'manifest.json').read_text())
    if file_hash(path/'manifest.json') != source['manifest_sha256']:
        raise ValueError(f'Source manifest changed: {path}')
    if manifest['state'] != 'complete' or manifest['format'] not in (
            'pointcloudmaterials.shooting_trajectory', 'pointcloudmaterials.temporal_lammps_trajectory'):
        raise ValueError(f'Unsupported or incomplete trajectory: {path}')
    arrays = {}
    for name in ('positions', 'atom_ids', 'atom_types', 'timesteps', 'box_low', 'box_high'):
        spec = manifest['arrays'][name]
        a = np.load(path/spec['file'], mmap_mode='r', allow_pickle=False)
        if list(a.shape) != spec['shape'] or a.dtype.name != spec['dtype']:
            raise ValueError(f'Changed {name} schema: {path}')
        arrays[name] = a
    if np.any(arrays['atom_types'] != 1):
        raise ValueError(f'Pure-element source contains unexpected atom types: {path}')
    return arrays


def chart(arrays, frame, static):
    if static:
        points = np.asarray(arrays['positions'], dtype=np.float32)
        return points, cKDTree(points, balanced_tree=False), None
    lengths = (arrays['box_high'][frame]-arrays['box_low'][frame]).astype(np.float64)
    points = np.mod(arrays['positions'][frame].astype(np.float64), lengths)
    return points, cKDTree(points, boxsize=lengths, balanced_tree=False), lengths


def offsets(points, center, rows, box):
    x = points[rows].astype(np.float64)-points[center].astype(np.float64)
    if box is not None:
        x -= box*np.round(x/box)
    return x.astype(np.float32)


def registry_sources(config):
    registry = json.loads(Path(config['registry']).read_text())
    entries = {d['id']: d for d in registry['datasets']}
    cohort = json.loads(Path(config['cohort']).read_text())
    relaxed = json.loads(Path(config['ancestry_plan']).read_text())
    blocked = set(relaxed['split_conflicts_held_out'])
    sources = []
    for row in cohort['sources']:
        role = row['validation_role'] if row['split']=='val' else row['split']
        if role not in ('train', 'selection') or row['lineage'] in blocked:
            continue
        sources.append(dict(id=f"native_{row['id']}", material='Al', kind='dynamic',
            path=str(dataset_path(row['dataset'])/row['relative_trajectory_path']),
            manifest_sha256=row['manifest_sha256'], lineage=row['lineage'],
            split=role, stratum='al_native', potential='al-lee2003-meam',
            timestep_fs=row['timestep_fs'], frame_count=row['frame_count'],
            center_ids=row['center_atom_ids']))
    for row in relaxed['sources']:
        if not row['training_eligible'] or row['family'] not in ('position_shooting', 'nested_first_passage'):
            continue
        sources.append(dict(id=row['id'], material='Al', kind='dynamic', path=row['trajectory'],
            manifest_sha256=row['manifest_sha256'], lineage=row['lineage'], split='train',
            stratum='al_shooting', potential='al-lee2003-meam', timestep_fs=row['timestep_fs'],
            frame_count=row['frame_count']))
    selected = [('al-eam-six-24ps', 'Al', 'al_other', 1.),
        ('al_meam_1m_450K_400ps_with_melt_20260913T205405Z', 'Al', 'al_other', 1.),
        ('mg-eam-six-24ps', 'Mg', 'mg', 1.),
        ('ti-meam-source-and-six-branches', 'Ti', 'ti', 1.),
        ('ta-eam-five-24ps', 'Ta', 'ta', 2.),
        ('ta_initial_model_1m_24ps_npt_20260905', 'Ta', 'ta', 2.)]
    seen = set()
    for identifier, material, stratum, timestep in selected:
        entry = entries[identifier]
        records = json.loads((Path(config['registry']).parent/entry['records_file']).read_text())['records']
        for record in records:
            if record['kind'] != 'trajectory' or not record['trajectory']['usable_binary']:
                continue
            t = record['trajectory']
            if t['frame_count'] < 20:
                continue
            signature = t['content_signature_from_manifest']
            if signature in seen:
                continue
            seen.add(signature)
            sources.append(dict(id=signature[:20], material=material, kind='dynamic',
                path=str(Path(record['path']).parent), manifest_sha256=record['sha256'],
                lineage=f'{material}-archived-root', split='train', stratum=stratum,
                potential=entry['potential_ids'][0], timestep_fs=timestep,
                frame_count=t['frame_count']))
    for material, stratum in [('Al','al_other'), ('Mg','mg'), ('Ta','ta'), ('Zr','zr')]:
        entry = entries[material]
        for array in entry['loose_arrays']:
            if not array['path'].endswith('.npy'):
                continue
            path = Path(entry['location']['resolved'])/array['path']
            sources.append(dict(id=digest([material, array['path']])[:20], material=material,
                kind='static', path=str(path), lineage=f'{material}-archived-root', split='train',
                stratum=stratum, potential='unknown-static', frame_count=1,
                bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, file_sha256=file_hash(path)))
    train_lineages = {s['lineage'] for s in sources if s['split'] == 'train'}
    heldout = {s['lineage'] for s in cohort['sources'] if s['split'] != 'train'}
    if train_lineages & heldout:
        raise ValueError(f'Pretraining leaks held-out ancestry: {train_lineages & heldout}')
    if sum(s['split']=='selection' for s in sources)!=15:
        raise ValueError('Expected the original fifteen selection sources')
    return sources


def calibrate(sources, seed):
    rng = np.random.default_rng(seed); scales = {}; evidence = {}
    for material in ELEMENTS:
        candidates = [s for s in sources if s['material'] == material and s['split'] == 'train']
        candidates = [candidates[i] for i in rng.choice(len(candidates), min(8, len(candidates)), replace=False)]
        distances = []; rows = []
        for i, source in enumerate(candidates):
            arrays = source_arrays(source); static = source['kind'] == 'static'
            frame = 0 if static else min(2, source['frame_count']-1)
            x, tree, box = chart(arrays, frame, static)
            eligible = np.arange(len(x))
            if static:
                eligible = np.flatnonzero(((x > x.min(0)+25) & (x < x.max(0)-25)).all(-1))
            n = 4000//len(candidates)+(i < 4000 % len(candidates))
            if len(eligible) < n:
                raise ValueError(f'Insufficient interior calibration points: {source["id"]}')
            centers = rng.choice(eligible, n, replace=False)
            distances.extend(tree.query(x[centers], k=160, workers=1)[0][:, -1].tolist())
            rows.append(dict(source=source['id'], frame=frame, center_ids=arrays['atom_ids'][centers].tolist()))
            del tree, x, arrays
        scales[material] = float(np.quantile(distances, .995)*1.02)
        evidence[material] = rows
    return scales, evidence


def build_plan(config):
    root = resolve_path(config['release']); destination = root/'plan.json'
    if destination.exists():
        plan = json.loads(destination.read_text())
        if plan['config'] != config:
            raise ValueError('Existing immutable release has a different config')
        for path, expected in plan['producer_hashes'].items():
            if file_hash(path)!=expected:
                raise ValueError(f'Cannot mix target producer revisions: {path}')
        return plan
    sources = registry_sources(config)
    if 'calibration_plan' in config:
        reference=json.loads(resolve_path(config['calibration_plan']).read_text())
        if reference['sources']!=sources or reference['config']['seed']!=config['seed']:
            raise ValueError('Calibration reference has different source exclusions or seed')
        scales,calibration=reference['scales'],reference['calibration']
    else:
        scales, calibration = calibrate(sources, config['seed'])
    rng = np.random.default_rng(config['seed']); tasks = []
    for stratum, count in config['counts'].items():
        subset = [i for i,s in enumerate(sources) if s['split']=='train' and s['stratum']==stratum]
        by_root = defaultdict(list)
        for i in subset:
            by_root[sources[i]['lineage']].append(i)
        roots = sorted(by_root); rng.shuffle(roots)
        remaining = count; iteration = 0
        while remaining:
            candidates = by_root[roots[iteration % len(roots)]]
            dynamic = [i for i in candidates if sources[i]['kind']=='dynamic']
            static = [i for i in candidates if sources[i]['kind']=='static']
            if dynamic and static:
                candidates = static if iteration % 5 == 0 else dynamic
            index = int(rng.choice(candidates)); source = sources[index]
            n = min(remaining, config['rows_per_task'])
            frame = 0 if source['kind']=='static' else int(rng.integers(2, source['frame_count']-1))
            tasks.append(dict(source=index, frame=frame, count=n, seed=int(rng.integers(2**31)), split='train'))
            remaining -= n; iteration += 1
    for index, source in enumerate(sources):
        if source['split'] == 'selection':
            for frame in (104, 504):
                tasks.append(dict(source=index, frame=frame, count=len(source['center_ids']),
                                  seed=int(rng.integers(2**31)), split='selection'))
    for i,t in enumerate(tasks):
        t['id'] = f'{i:06d}'
    plan = dict(protocol='structural_neighbors_v1', config=config, sources=sources, tasks=tasks,
        scales=scales, calibration=calibration, reference_radius=REFERENCE_RADIUS,
        source_evidence=dict(registry=file_hash(config['registry']), cohort=file_hash(config['cohort']),
                             ancestry=file_hash(config['ancestry_plan'])),
        producer_hashes={p:file_hash(p) for p in ('src/data/structural_pretraining/prepare.py',
            'src/data/predictive_memory/targets.py','src/analysis/liquid_structure.py')})
    plan['identity'] = digest(plan)
    save_json(destination, plan)
    return plan


def prepare_task(arguments):
    root, source, task, scale, identity = arguments
    root = Path(root); folder = root/'shards'/task['id']; receipt = folder/'complete.json'
    if receipt.exists():
        r = json.loads(receipt.read_text())
        if r['identity'] != identity:
            raise ValueError(f'Changed shard identity: {folder}')
        return r
    from src.analysis.liquid_structure import persistence_image
    started = time.monotonic(); rng = np.random.default_rng(task['seed'])
    arrays = source_arrays(source); static = source['kind']=='static'; frame = task['frame']
    if static:
        frames = [0]; times = [0.]
    else:
        frames = [frame-2, frame-1, frame, frame+1]
        steps = arrays['timesteps'][frames]
        if np.any(np.diff(steps) <= 0):
            raise ValueError(f'Non-increasing recorded temporal neighborhood: {source["id"]}, {frame}')
        times = ((steps-steps[2])*source['timestep_fs']/1000.).tolist()
    radius = 17*scale/REFERENCE_RADIUS
    current_x, current_tree, current_box = chart(arrays, frame, static)
    if current_box is not None and np.min(current_box) <= 4*radius:
        raise ValueError(f'Observation radius exceeds local periodic chart: {source["id"]}')
    eligible = np.arange(len(current_x))
    if static:
        eligible = np.flatnonzero(((current_x > current_x.min(0)+1.3*radius) &
                                  (current_x < current_x.max(0)-1.3*radius)).all(-1))
    if task['split']=='selection':
        index = {int(v):i for i,v in enumerate(arrays['atom_ids'])}
        centers = np.array([index[i] for i in source['center_ids']])
    else:
        centers = rng.choice(eligible, task['count'], replace=False)
    partners = []
    for center in centers:
        nearby = current_tree.query_ball_point(current_x[center], radius*.25)
        nearby = [i for i in nearby if i != center]
        if not nearby:
            raise ValueError(f'No spatial partner for {source["id"]}, atom row {center}')
        partners.append(int(rng.choice(nearby)))
    n = len(centers); labelled = rng.random(n) < .25
    if task['split']=='selection':
        labelled[:] = True
    positions = []; ids = []; ptr = [0]; mappings = np.full((n, 5), -1, dtype=np.int32)
    targets = []; tda = []; tda_valid = []; view_centers = []; view_steps = []

    def add_view(x, tree, box, center, slot, row, step, with_targets):
        neighbors = np.array(sorted(tree.query_ball_point(x[center], radius)), dtype=np.int64)
        local = offsets(x, center, neighbors, box)
        distance2 = np.square(local.astype(np.float64)).sum(-1)
        nearest = np.lexsort((arrays['atom_ids'][neighbors], distance2))[:80]
        if len(nearest) != 80 or np.sqrt(distance2.max()) >= radius+1e-4:
            raise ValueError(f'Incomplete local target support: {source["id"]}, row {center}')
        mappings[row, slot] = len(targets)
        positions.append(local); ids.append(arrays['atom_ids'][neighbors].astype(np.int64)); ptr.append(ptr[-1]+len(local))
        targets.append(geometry_packet(local) if with_targets else np.zeros(85, np.float32))
        valid = bool(labelled[row] and with_targets)
        tda.append(persistence_image(local[nearest]) if valid else np.zeros(144, np.float32))
        tda_valid.append(valid); view_centers.append(int(arrays['atom_ids'][center])); view_steps.append(int(step))

    for row, (center, partner) in enumerate(zip(centers, partners, strict=True)):
        step = 0 if static else arrays['timesteps'][frame]
        add_view(current_x, current_tree, current_box, center, 2, row, step, True)
        add_view(current_x, current_tree, current_box, partner, 4, row, step, True)
    del current_x, current_tree
    if not static:
        for slot in (0, 1, 3):
            f = frames[slot]; x, tree, box = chart(arrays, f, False)
            for row, center in enumerate(centers):
                add_view(x, tree, box, center, slot, row, arrays['timesteps'][f], slot==3)
            del x, tree
    folder.mkdir(parents=True, exist_ok=True)
    values = dict(positions=np.concatenate(positions), atom_ids=np.concatenate(ids), offsets=np.array(ptr,dtype=np.int64),
        views=mappings, physical=np.stack(targets), tda=np.stack(tda), tda_valid=np.array(tda_valid),
        center_ids=np.array(view_centers,dtype=np.int64), steps=np.array(view_steps,dtype=np.int64),
        times=np.asarray(times,dtype=np.float64))
    hashes = {}
    for name,value in values.items():
        path = folder/f'{name}.npy'; temporary = folder/f'{name}.tmp.npy'
        np.save(temporary, value, allow_pickle=False); temporary.replace(path); hashes[name] = file_hash(path)
    r = dict(identity=identity, task=task, source=source['id'], material=source['material'],
        potential=source['potential'], scale=scale, static=static, anchors=n, views=len(targets),
        atoms=ptr[-1], labelled_views=int(sum(tda_valid)), seconds=time.monotonic()-started, hashes=hashes)
    save_json(receipt, r)
    return r


def finalize(root, plan):
    records = []; pstats = []; tstats = []
    for task in plan['tasks']:
        folder = root/'shards'/task['id']; r = json.loads((folder/'complete.json').read_text())
        records.append(r)
        if task['split']=='train':
            mapping = np.load(folder/'views.npy'); valid_views = mapping[:, [2,4] if r['static'] else [2,3,4]].ravel()
            pstats.append(np.load(folder/'physical.npy')[valid_views])
            mask = np.load(folder/'tda_valid.npy'); tstats.append(np.load(folder/'tda.npy')[mask])
    p = np.concatenate(pstats).astype(np.float64); t = np.concatenate(tstats).astype(np.float64)
    normalization = {name:dict(mean=a.mean(0).tolist(), std=np.maximum(a.std(0),1e-4).tolist()) for name,a in [('physical',p),('tda',t)]}
    result = dict(state='complete', identity=plan['identity'], normalization=normalization, shards=records,
        sources=plan['sources'], scales=plan['scales'], config=plan['config'])
    save_json(root/'manifest.json', result)
    return result


def run(config, workers):
    root = resolve_path(config['release']); plan = build_plan(config)
    # Selection first, then interleave strata so useful coverage is visible early.
    groups = defaultdict(list)
    for task in plan['tasks']:
        groups[plan['sources'][task['source']]['stratum'] if task['split']=='train' else 'selection'].append(task)
    ordered = [v[i] for i in range(max(map(len,groups.values()))) for v in groups.values() if i<len(v)]
    args = [(str(root),plan['sources'][t['source']],t,plan['scales'][plan['sources'][t['source']]['material']],plan['identity']) for t in ordered]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(prepare_task,a) for a in args]
        for completed,future in enumerate(as_completed(futures),1):
            r = future.result()
            status = dict(state='preparing', completed_shards=completed, total_shards=len(args), last_shard=r['task']['id'])
            save_json(root/'status.json',status); print(json.dumps(status),flush=True)
    result = finalize(root,plan)
    save_json(root/'status.json',dict(state='complete', shards=len(result['shards'])))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args(); config=json.loads(Path(args.config).read_text()); run(config,args.workers)


if __name__=='__main__':
    main()
