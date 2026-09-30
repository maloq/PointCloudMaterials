"""Frozen interface cluster comparison on the six relaxed, nonperiodic Al snapshots."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import html
from importlib.metadata import version
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster._kmeans import _labels_inertia_threadpool_limit

from src.data.fixed_cohort.protocol import sha, write_json
from src.project_runtime.paths import resolve_path
from src.research.crystal_vector.interface import interface_mask
from .dense_md import block
from .viewer_payload import md_template, write_asset, write_page

FAMILIES = {'tda': 'TDA clusters', 'bond_order': 'Bond-order clusters',
            'cna': 'CNA clusters', 'joint': 'Joint descriptor clusters'}


def settings(config):
    c = json.loads(Path(config).read_text())
    if c['protocol'] != 'static_al_interface_v1':
        raise ValueError('Wrong static interface protocol')
    for key in ('output', 'publication', 'static_data', 'grid_reference', 'pacmap_config'):
        path = Path(c[key])
        # A submitted config already records its real publication destination.
        # Rebinding it against a frozen code tree would redirect repo outputs.
        c[key] = str(path.resolve() if path.is_absolute() else resolve_path(c[key]).resolve())
    pc = json.loads(Path(c['pacmap_config']).read_text())
    corr = json.loads(Path(pc['correspondence_config']).read_text())
    parent = json.loads(Path(corr['parent']).read_text())
    return c, pc, corr, parent


def assign(x, centers):
    x = np.ascontiguousarray(x, dtype=np.float32)
    return _labels_inertia_threadpool_limit(x, np.ones(len(x), np.float32),
                                          centers, n_threads=1, return_inertia=False).astype(np.uint8)


def descriptor_space(targets, model):
    x = np.ascontiguousarray((targets[:, model['columns']]-model['mean'])/model['sd'], dtype=np.float32)
    x /= model['balance']
    return x


def prepare(config, index):
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    from src.analysis.representative_structures import _build_ovito_data_collection
    c, pc, corr, parent = settings(config)
    frame = c['frames'][index]; tag = f'{frame}ps'
    folder = Path(c['output'])/'data'/tag; (folder/'chunks').mkdir(parents=True, exist_ok=True)
    if (folder/'complete.json').exists():
        raise FileExistsError(f'Already prepared: {folder}')
    started = time.monotonic(); source = Path(c['static_data'])/(tag+'.npy')
    grid = Path(c['grid_reference'])/'data'/f'population-{tag}.npz'
    protocol = json.loads((Path(c['grid_reference'])/'technical/protocol.json').read_text())
    original = next(r for r in protocol['frames'] if r['frame'] == tag)
    if sha(source) != original['sha256']:
        raise ValueError(f'Changed source snapshot: {source}')
    points = np.asarray(np.load(source), np.float64)
    if points.shape != (1048576, 3) or not np.isfinite(points).all():
        raise ValueError(f'Unexpected static coordinates: {source}')
    with np.load(grid) as z:
        atoms = z['source_atom_rows']; coords = z['coords']
    if len(atoms) != original['centers'] or len(np.unique(atoms)) != len(atoms):
        raise ValueError(f'Changed static grid: {grid}')
    if not np.array_equal(points[atoms], coords):
        raise ValueError(f'Grid coordinates do not match source rows: {grid}')
    low, high = points.min(0), points.max(0)
    tree = cKDTree(points)
    radius, neighbors = tree.query(coords, k=80, workers=c['workers'])
    clearance = np.minimum(coords-low, high-coords).min(1)
    if np.any(radius[:, -1] >= clearance):
        raise ValueError(f'Incomplete nonperiodic neighborhoods: {tag}; do not silently discard grid rows')
    patches = (points[neighbors]-points[atoms, None]).astype(np.float32)
    if not np.all(patches[:, 0] == 0):
        raise ValueError('Nearest-80 patches must start with their central atom')
    np.save(folder/'patches.npy', patches)
    print(f'{tag}: {len(atoms)} grid centers / {len(points)} context atoms; full-source PTM', flush=True)
    data = _build_ovito_data_collection(points)
    data.apply(PolyhedralTemplateMatchingModifier(rmsd_cutoff=0.1))
    ptm = np.asarray(data.particles['Structure Type'], np.uint8)
    solid = np.isin(ptm, [1, 2, 3]); crystalline = np.flatnonzero(solid)
    large = np.zeros(len(points), bool)
    if len(crystalline):
        pairs = cKDTree(points[crystalline]).query_pairs(3.6, output_type='ndarray')
        graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])),
                           shape=(len(crystalline), len(crystalline))).tocsr()
        _, component = connected_components(graph, directed=False)
        large[crystalline] = np.bincount(component)[component] >= 64
    boundary, accepted = interface_mask(points, None, solid, 3.6, 64)
    interior = np.minimum(points-low, high-points).min(1) >= c['outer_boundary_guard_A']
    boundary &= large & interior
    distance = np.full(len(atoms), np.inf, np.float32)
    if boundary.any():
        distance[:] = cKDTree(points[boundary]).query(coords, workers=c['workers'])[0]
    near = distance <= 12; ss = solid[atoms]; aa = accepted[atoms]
    region = np.full(len(atoms), 4, np.uint8)
    region[near & ss] = 1; region[near & aa] = 2; region[near & ~ss & ~aa] = 3
    region[boundary[atoms]] = 0; region[~np.isfinite(distance)] = 5
    sample = np.sort(np.random.default_rng(c['sample_seed']+frame).choice(len(atoms), c['projection_samples_per_frame'], replace=False))
    np.savez_compressed(folder/'physical.npz', coords=coords, atom=atoms, ptm=ptm[atoms],
        solid=ss, large_crystal=large[atoms], accepted_disorder=aa, boundary=boundary[atoms],
        distance=distance, region=region, support_fraction=solid[neighbors].mean(1),
        bounds=np.stack([low, high], 1), sample=sample)
    np.savez_compressed(folder/'full-source-ptm.npz', ptm=ptm, boundary=boundary, accepted_disorder=accepted)
    chunks = [(i, min(i+c['chunk_atoms'], len(atoms))) for i in range(0, len(atoms), c['chunk_atoms'])]
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        jobs = [pool.submit(block, str(folder), lo, hi) for lo, hi in chunks]
        for n, job in enumerate(as_completed(jobs), 1):
            job.result()
            if n % 40 == 0: print(f'{tag}: descriptors {n}/{len(jobs)} chunks', flush=True)
    targets = np.concatenate([np.load(folder/'chunks'/f'{lo:06d}-{hi:06d}.npy') for lo, hi in chunks])
    np.save(folder/'targets.npy', targets)
    labels = {}; bindings = {}
    for family in FAMILIES:
        model = Path(corr['output'])/'data'/f'interface12-{family}-k7-descriptor-model.npz'
        with np.load(model) as z: labels[family] = assign(descriptor_space(targets, z), z['centers'])
        bindings[str(model)] = sha(model)
    np.savez_compressed(folder/'descriptor-labels.npz', **labels)
    write_json(folder/'complete.json', dict(frame=frame, rows=len(atoms), context_atoms=len(points),
        source=str(source), source_sha256=sha(source), grid=str(grid), grid_sha256=sha(grid),
        physical_sha256=sha(folder/'physical.npz'), targets_sha256=sha(folder/'targets.npy'),
        patches_sha256=sha(folder/'patches.npy'), descriptor_models=bindings,
        neural_training=False, periodic=False, relaxation='inherent configurations, already relaxed',
        interface_boundary_atoms=int(boundary.sum()), implementation_sha256=sha(__file__),
        seconds=time.monotonic()-started))
    print(f'Completed static geometry and descriptors: {tag}', flush=True)


def sampled(c):
    parts = []
    for frame in c['frames']:
        folder = Path(c['output'])/'data'/f'{frame}ps'
        receipt = json.loads((folder/'complete.json').read_text())
        if sha(folder/'physical.npz') != receipt['physical_sha256']:
            raise ValueError(f'Changed physical data: {frame}')
        with np.load(folder/'physical.npz') as p:
            ids = p['sample']; part = {k: p[k][ids] for k in ('atom', 'distance', 'solid', 'region', 'support_fraction', 'ptm')}
        part.update(frame=np.full(len(ids), frame), source=np.zeros(len(ids), int), grid_row=ids)
        parts.append(part)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def project(c, x, name, title):
    import pacmap
    import faiss
    import numba
    pc = json.loads(Path(c['pacmap_config']).read_text())
    if version('pacmap') != pc['pacmap']['version']: raise ValueError('Changed PaCMAP version')
    faiss.omp_set_num_threads(2); numba.set_num_threads(2)
    a = sampled(c); out = Path(c['output'])
    params = {k: v for k, v in pc['pacmap'].items() if k != 'version'}
    for population in ('all_static', 'interface20'):
        ids = np.arange(len(x)) if population == 'all_static' else np.flatnonzero(a['distance'] <= 20)
        if len(ids) < 100: raise ValueError(f'Too few interface projection points: {len(ids)}')
        stem = name+'-'+population; ys = {}; started = time.monotonic()
        for dim in (2, 3):
            print(f'PaCMAP {stem}: {len(ids)} rows, {dim} dimensions', flush=True)
            ys[dim] = pacmap.PaCMAP(n_components=dim, **params).fit_transform(np.asarray(x[ids], np.float32), init='pca')
            if not np.isfinite(ys[dim]).all(): raise ValueError(f'Nonfinite PaCMAP {stem}')
        path = out/'data'/f'{stem}.npz'
        np.savez_compressed(path, pacmap2=ys[2], pacmap3=ys[3], sample_row=ids,
                            frame=a['frame'][ids], atom=a['atom'][ids], grid_row=a['grid_row'][ids])
        write_json(out/'technical/views'/f'{stem}.json', dict(name=stem, title=title+' / '+population,
            rows=len(ids), features=x.shape[1], params=params, coordinate_sha256=sha(path),
            population=population, seconds=time.monotonic()-started, neural_training=False))


def infer(config, index):
    import torch
    from omegaconf import OmegaConf
    from .train import PairEncoder
    c, pc, corr, parent = settings(config); out = Path(c['output'])
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    item = c['models'][index]; run, epoch = item['run'], item['epoch']; identity = f'{run}-epoch{epoch}'
    root = Path(parent['output'])/run; checkpoint = root/'checkpoints'/f'epoch-{epoch:02d}.pt'
    original = json.loads((root/'analyses'/f'epoch-{epoch:02d}'/'technical/complete.json').read_text())
    if sha(checkpoint) != original['checkpoint_sha256']: raise ValueError('Changed frozen checkpoint')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved['epoch'] != epoch or saved['data_identity'] != corr['cache_identity']:
        raise ValueError('Unexpected frozen checkpoint identity')
    torch.manual_seed(saved['seed']); np.random.seed(saved['seed'])
    model = PairEncoder(OmegaConf.create(saved['recipe'])).cuda().eval(); model.requires_grad_(False)
    model.load_state_dict(saved['model'], strict=True)
    centers = {}; bindings = {}; features = {k: [] for k in ('encoder', 'projector')}
    for rep in features:
        path = root/'analyses'/f'epoch-{epoch:02d}'/'data'/f'{rep}-k7-assignments.npz'
        with np.load(path) as z: centers[rep] = z['centers']
        bindings[str(path)] = sha(path)
    for frame in c['frames']:
        folder = out/'data'/f'{frame}ps'; patches = np.load(folder/'patches.npy', mmap_mode='r')
        complete = json.loads((folder/'complete.json').read_text())
        if sha(folder/'patches.npy') != complete['patches_sha256']: raise ValueError('Changed static patches')
        with np.load(folder/'physical.npz') as p: ids = p['sample']
        values = {k: np.empty(len(patches), np.uint8) for k in features}
        retained = {k: np.empty((len(ids), centers[k].shape[1]), np.float32) for k in features}
        with torch.inference_mode():
            for start in range(0, len(patches), 256):
                stop = min(start+256, len(patches)); selected = (ids >= start) & (ids < stop)
                x = torch.as_tensor(np.asarray(patches[start:stop])/parent['geometry']['length_scale_A'], device='cuda')
                z, y = model(x)
                for rep, tensor in (('encoder', z), ('projector', y)):
                    xx = tensor.cpu().numpy(); values[rep][start:stop] = assign(xx, centers[rep])
                    retained[rep][selected] = xx[ids[selected]-start]
        np.savez_compressed(folder/(identity+'-labels.npz'), **values)
        for rep in features: features[rep].append(retained[rep])
        print(f'Frozen inference complete: {identity}/{frame}ps, {len(patches)} centers', flush=True)
    del model; torch.cuda.empty_cache()
    write_json(out/'technical'/f'{identity}-inference.json', dict(checkpoint=str(checkpoint),
        checkpoint_sha256=sha(checkpoint), centroid_bindings=bindings, neural_training=False,
        encoder_inputs='centered nonperiodic nearest-80 relaxed coordinates / fixed Al length scale',
        predictor_inputs=None, history=0, motion=False, conditions=[], training_only_teacher=None,
        feature_storage='sampled vectors held only in RAM until projection; no persistent feature bank',
        device=torch.cuda.get_device_name(), precision='float32, TF32 disabled'))
    for rep in features:
        project(c, np.concatenate(features[rep]), identity+'-'+rep, f'{run} epoch {epoch} {rep}')


def descriptors(config):
    c, pc, corr, parent = settings(config); parts = []
    for frame in c['frames']:
        folder = Path(c['output'])/'data'/f'{frame}ps'
        with np.load(folder/'physical.npz') as p: ids = p['sample']
        r = json.loads((folder/'complete.json').read_text())
        if sha(folder/'targets.npy') != r['targets_sha256']: raise ValueError('Changed descriptor targets')
        parts.append(np.load(folder/'targets.npy', mmap_mode='r')[ids])
    targets = np.concatenate(parts)
    for family, title in FAMILIES.items():
        path = Path(corr['output'])/'data'/f'interface12-{family}-k7-descriptor-model.npz'
        with np.load(path) as z: x = descriptor_space(targets, z)
        project(c, x, 'descriptors-'+family, title)


def publish(config):
    from plotly.offline import get_plotlyjs
    from .correspondence import agreement, heatmap
    from .pacmap_views import list_values, static_plot
    from src.experiment_runner.metric_docs import write_metric_table
    c, pc, corr, parent = settings(config); out = Path(c['output']); dest = Path(c['publication'])
    for part in ('plots', 'interactive', 'assets', 'md-data', 'technical/views'): (out/part).mkdir(parents=True, exist_ok=True)
    (out/'assets/plotly.min.js').write_text(get_plotlyjs())
    md = dict(snapshots=[], models=[], default_model='S1-seed17-epoch24-encoder')
    fields = {title: [] for title in FAMILIES.values()}; neural = {}; metrics = {}; sample = sampled(c)
    for item in c['models']:
        identity = f'{item["run"]}-epoch{item["epoch"]}'
        for rep in ('encoder', 'projector'):
            key = identity+'-'+rep; title = f'{item["run"]} epoch {item["epoch"]} {rep} K=7'
            md['models'].append(dict(id=key, title=title, representation=rep, snapshots={}))
            neural[title] = []
    for frame in c['frames']:
        tag = f'{frame}ps'; folder = out/'data'/tag
        p = dict(np.load(folder/'physical.npz')); ids = p['sample']; labels = dict(np.load(folder/'descriptor-labels.npz'))
        n = len(p['atom']); md['snapshots'].append(dict(key=tag, frame=frame, source=0, count=n,
            title=f'Al {frame} ps · {n:,} grid centers', asset=f'../md-data/{tag}.js'))
        write_asset(out/'md-data'/f'{tag}.js', tag, dict(source=0, frame=frame, count=n,
            bounds=p['bounds'].tolist(), x=np.round(p['coords'][:, 0], 4).tolist(),
            y=np.round(p['coords'][:, 1], 4).tolist(), z=np.round(p['coords'][:, 2], 4).tolist(),
            fields={FAMILIES[k]: v.tolist() for k, v in labels.items()}))
        for family, title in FAMILIES.items(): fields[title].append(labels[family][ids])
        populations = dict(all_grid=np.ones(n, bool), interface12=p['distance'] <= 12,
            crystal_interface_layer=p['boundary'], nearby_disorder=(p['distance'] <= 12) & p['accepted_disorder'],
            small_disordered_pockets=(p['distance'] <= 12) & ~p['solid'] & ~p['accepted_disorder'],
            crystal_free_nearby_disorder=(p['distance'] <= 20) & p['accepted_disorder'] & (p['support_fraction'] == 0))
        for lo, hi in ((0, 3.6), (3.6, 8), (8, 12), (12, 20)):
            populations[f'disorder_{lo:g}_{hi:g}A'] = (p['distance'] > lo) & (p['distance'] <= hi) & p['accepted_disorder']
        fm = dict(rows=n, interface12_rows=int(populations['interface12'].sum()), models={})
        for item in c['models']:
            identity = f'{item["run"]}-epoch{item["epoch"]}'; nn = dict(np.load(folder/(identity+'-labels.npz')))
            assetkey = tag+'-'+identity; asset = '../md-data/'+assetkey+'.js'
            write_asset(out/'md-data'/(assetkey+'.js'), assetkey, {k: v.tolist() for k, v in nn.items()}, 'MD_NEURAL')
            for rep, values in nn.items():
                model = next(m for m in md['models'] if m['id'] == identity+'-'+rep)
                model['snapshots'][tag] = dict(asset=asset, key=assetkey); neural[model['title']].append(values[ids])
                rm = {}; fm['models'][model['id']] = rm
                for family, cl in labels.items():
                    rm[family] = {}
                    for population, mask in populations.items():
                        record = dict(rows=int(mask.sum()), defined=False)
                        if mask.sum() >= 2:
                            record = agreement(cl[mask], values[mask], np.zeros(mask.sum(), int), 7)
                            record['defined'] = True
                        rm[family][population] = record
                    if family == 'joint' and rm[family]['interface12']['defined']:
                        heatmap(np.asarray(rm[family]['interface12']['contingency']),
                            out/'plots'/f'correspondence-{tag}-{model["id"]}.png', f'{tag} / {model["title"]} / interface ≤12 Å')
        metrics[tag] = fm
    fields = {k: np.concatenate(v) for k, v in {**fields, **neural}.items()}
    signed = np.clip(np.where(sample['solid'], -1, 1)*sample['distance'], -20, 20)
    signed[~np.isfinite(sample['distance'])] = np.nan
    fields.update({'PTM type': sample['ptm'], 'Physical region': sample['region'],
                   'Input crystal fraction': sample['support_fraction'], 'Interface distance (Å, clipped ±20)': signed})
    rows = []; template = md_template(static=True)
    for receipt in sorted((out/'technical/views').glob('*.json')):
        r = json.loads(receipt.read_text()); name = r['name']; path = out/'data'/f'{name}.npz'
        if sha(path) != r['coordinate_sha256']: raise ValueError('Changed projection data')
        with np.load(path) as z:
            ids = z['sample_row']; y2 = z['pacmap2']; y3 = z['pacmap3']
            if not np.array_equal(z['atom'], sample['atom'][ids]) or not np.array_equal(z['frame'], sample['frame'][ids]):
                raise ValueError('Static projection row mismatch')
        title = r['title']; own = next((m for m in md['models'] if name.startswith(m['id']+'-')), None)
        md['default_model'] = own['id'] if own else 'S1-seed17-epoch24-encoder'
        first = own['title'] if own else FAMILIES[name.split('-')[1]]
        ordered = {first: fields[first], **fields}; selected = {k: v[ids] for k, v in ordered.items()}
        static_plot(y2, selected, title, out/'plots'/f'{name}.png')
        payload = dict(title=title, y2=np.round(y2, 6).tolist(), y3=np.round(y3, 6).tolist(),
            **{k: list_values(sample[k][ids]) for k in ('source', 'frame', 'atom', 'distance', 'solid', 'region')},
            fields={k: list_values(v) for k, v in selected.items()}, md=md)
        write_page(out/'interactive'/f'{name}.html', template, payload)
        rows.append(f'<tr><td>{html.escape(title)}</td><td>{len(ids):,}</td><td><a href="plots/{name}.png">2D panels</a></td><td><a href="interactive/{name}.html">2D / 3D PaCMAP + two MD views</a></td></tr>')
    write_json(out/'technical/metrics.json', metrics)
    write_metric_table(metrics, out, family='static_interface')
    description = ('Six relaxed Al snapshots: 166, 170, 174, 175, 177 and 240 ps. Dense MD uses the established static grid '
        '(684,723 centers total; 1,048,576 context atoms per snapshot). PaCMAP uses the same 4,000 uniform centers per snapshot '
        'in every space; frame and region filters are available. All neural and descriptor cluster models remain frozen from '
        'the earlier MD study. This is a descriptive transfer analysis, not held-out prediction or new training. '
        'The static data have no recorded periodic box or cross-snapshot atom identities. PTM-unclassified does not prove liquid. '
        'Cluster colors agree between PaCMAP and MD within each feature space; independent cluster numbers are not physical matches. '
        'Two MD panels, no linked atom events. Use the z sliders to expose the interior.')
    page = ('<!doctype html><meta charset="utf-8"><title>Six static Al snapshots — interface clusters</title>'
        '<style>body{font:16px system-ui;margin:32px;max-width:1400px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}</style>'
        '<h1>Six static Al snapshots: neural and rich descriptor clusters</h1><p>'+description+'</p>'
        '<p><a href="tables/metrics.csv">Correspondence metrics</a> · <a href="tables/METRICS.md">Definitions</a></p>'
        '<table><tr><th>Feature space / population</th><th>Samples</th><th>Static</th><th>Interactive</th></tr>'+''.join(rows)+'</table>')
    (out/'index.html').write_text(page); (out/'README.md').write_text('# Six static Al interface comparison\n\n'+description+'\n\n[Gallery](index.html) · [Metrics](tables/metrics.csv)\n')
    for part in ('assets', 'md-data', 'interactive', 'plots', 'tables', 'technical'):
        shutil.copytree(out/part, dest/part, dirs_exist_ok=True, ignore=shutil.ignore_patterns('queue'))
    for part in ('index.html', 'README.md'): shutil.copy2(out/part, dest/part)
    write_json(out/'technical/complete.json', dict(views=len(rows), snapshots=6, dense_centers=684723,
        neural_training=False, publication=str(dest), implementation_sha256=sha(__file__)))
    shutil.copy2(out/'technical/complete.json', dest/'technical/complete.json')
    print(f'Published static Al comparison: {dest}', flush=True)


def submit(config):
    from src.experiment_runner.metric_docs import check_metric_docs
    c, pc, corr, parent = settings(config); check_metric_docs(family='static_interface')
    out = Path(c['output']); tech = out/'technical/queue'; code = tech/'code'
    if (tech/'launch.json').exists(): raise FileExistsError('Static analysis already submitted')
    original = Path(c['pacmap_config']).parent/'code'
    shutil.copytree(original, code, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    repo = Path(__file__).resolve().parents[3]
    for relative in ('src/research/spatial_vicreg_bias/static_md.py', 'src/research/spatial_vicreg_bias/dense_md.py',
                     'src/research/spatial_vicreg_bias/viewer_payload.py',
                     'src/data/fixed_cohort/protocol.py', 'src/experiment_runner/artifacts.py',
                     'src/research/spatial_vicreg_bias/pacmap_md_view.html', 'src/experiment_runner/metric_docs.py'):
        shutil.copy2(repo/relative, code/relative)
    shutil.copytree(repo/'docs/metrics', code/'docs/metrics', dirs_exist_ok=True)
    # The copied model source is the exact original frozen inference implementation.
    write_json(tech/'code-hashes.json', {str(p.relative_to(code)): sha(p) for p in (code/'src').rglob('*') if p.is_file()})
    for part in ('technical/views', 'data', 'plots', 'interactive', 'assets'): (out/part).mkdir(parents=True, exist_ok=True)
    cfg = tech/'config.json'; write_json(cfg, c)
    launch = dict(neural_training=False, config=str(cfg), jobs=[])
    for stage, cpus, memory, hours, array in [('prepare', c['workers'], 32, 4, '0-5%3'),
        ('infer', 4, 16, 3, '0-1%1'), ('descriptors', 4, 12, 2, None), ('publish', 4, 16, 1, None)]:
        script = tech/(stage+'.sbatch')
        resources = ['#SBATCH --partition='+c['gpu_partition'], '#SBATCH --gres=gpu:1'] if stage == 'infer' else ['#SBATCH --partition=CPU']
        command = [sys.executable, '-u', '-m', 'src.research.spatial_vicreg_bias.static_md', stage, '--config', str(cfg)]
        suffix = ' --index "$SLURM_ARRAY_TASK_ID"' if array else ''
        script.write_text('\n'.join(['#!/bin/bash', '#SBATCH --job-name=SVB-static-'+stage, *resources,
            f'#SBATCH --cpus-per-task={cpus}', f'#SBATCH --mem={memory}G', f'#SBATCH --time={hours:02d}:00:00',
            f'#SBATCH --output={tech}/{stage}-%A_%a.log', *([f'#SBATCH --array={array}'] if array else []),
            'set -euo pipefail', 'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=2',
            'export OVITO_THREAD_COUNT=1 QT_QPA_PLATFORM=offscreen TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1',
            'cd '+shlex.quote(str(code)), shlex.join(command)+suffix])+'\n')
        args = ['sbatch', '--parsable']
        if stage in ('infer', 'descriptors'): args += ['--dependency=afterok:'+launch['jobs'][0]['job']]
        if stage == 'publish': args += ['--dependency=afterok:'+':'.join(j['job'] for j in launch['jobs'][1:])]
        job = subprocess.check_output(args+[str(script)], text=True).strip().split(';')[0]
        launch['jobs'].append(dict(stage=stage, job=job, script=str(script))); write_json(tech/'launch.json', launch)
    dest = Path(c['publication']); dest.mkdir(parents=True, exist_ok=True)
    (dest/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Six static Al snapshots — analysis submitted</h1>'
        '<p>166, 170, 174, 175, 177 and 240 ps. Frozen neural and rich descriptor clusters; two dense MD panels per snapshot; '
        '2D and 3D PaCMAP. This page will be replaced with results after Slurm jobs finish.</p><pre>'+html.escape(json.dumps(launch, indent=2))+'</pre>')
    print(json.dumps(launch, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('stage', choices=['submit', 'prepare', 'infer', 'descriptors', 'publish'])
    p.add_argument('--config', required=True); p.add_argument('--index', type=int)
    args = p.parse_args()
    if args.stage in ('prepare', 'infer'): globals()[args.stage](args.config, args.index)
    else: globals()[args.stage](args.config)
