"""Reproducible PNG figures from completed, development-selected forecasts.

No predictor fitting; CPU replay only. UMAP is fitted on training-source features.
The saved test population and the exact original input builder remain authoritative.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from src.project_runtime.paths import resolve_path, dataset_path
from src.research.crystallization_transfer.data import Corpus, representatives
from src.research.context_night.context import LOCAL_COLUMNS, information_values
from src.research.local_predictability.metrics import threshold_at_fpr
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.data.structural_pretraining.support import REFERENCE_RADIUS, OUTER_RADIUS
from src.experiment_runner.metric_docs import write_metric_table
from .runtime import make_model


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def assert_producer(config):
    paths = ['src/research/crystallization_paths/' + name + '.py'
             for name in ('model', 'refined_model', 'runtime')]
    paths += ['src/research/crystallization_transfer/' + name + '.py'
              for name in ('attention', 'model', 'data')]
    paths += ['src/research/context_night/context.py']
    hashes = {}
    for p in paths:
        hashes[p] = digest(p)
        if hashes[p] != digest(resolve_path(config['producer']) / p):
            raise ValueError(f'Forecast producer changed: {p}; replay its frozen implementation')
    return hashes


def choose_examples(corpus, prediction):
    """Explicit descriptive selection, never a model-selection criterion."""
    ids = prediction['test_indices']; source = corpus.source_ids[ids]
    event = prediction['test_event']; error = prediction['test_path_scores'][:, :, 1:4].mean((1, 2))
    threshold = threshold_at_fpr(prediction['calibration_event'] < 16,
        prediction['calibration_cdf'][:, 15], corpus.source_ids[prediction['calibration_indices']], .05)
    masks = [('Early onset', event < 16), ('Later onset', (event >= 16) & (event < 64)),
             ('No onset within 96 ps', event == 128),
             ('Missed early onset', (event < 16) & (prediction['test_cdf'][:, 15] < threshold))]
    selected = []; used = set()
    for label, mask in masks:
        # First the median-error window per source, then the median across sources.
        medians = []
        for sid in np.unique(source[mask]):
            if int(sid) in used:
                continue
            rows = np.flatnonzero(mask & (source == sid))
            rows = rows[np.lexsort((ids[rows], error[rows]))]
            medians.append(int(rows[len(rows)//2]))
        if not medians:
            raise ValueError(f'No independent illustrative source for {label}')
        medians.sort(key=lambda i: (error[i], int(ids[i])))
        row = medians[len(medians)//2]; index = int(ids[row]); sid, ai, ci, temp = corpus.rows[index]
        used.add(sid)
        selected.append(dict(label=label, index=index, test_row=row, source=sid,
            center=ci, frame=corpus.plan['anchors'][ai], temperature_K=temp,
            onset_ps=float((event[row]+1)*.75) if event[row] < 128 else None,
            direct_path_error=float(error[row]), direct_12ps_threshold=float(threshold)))
    return selected


class Replay:
    """Small batches using the original corpus input producer and context layout."""
    def __init__(self, plan, corpus):
        self.plan = plan; self.corpus = corpus
        self.sources = {s['id']: s for s in plan['sources']}
        original = json.loads(resolve_path(plan['config']['reuse_plan']).read_text())
        self.assay = resolve_path(original['config']['assay_cache'])
        self.info = {}; self.states = {}

    def state(self, sid):
        if sid not in self.states:
            a = self.corpus.arrays[sid]
            z = np.load(resolve_path(self.plan['config']['future_cache']) / str(sid) / 'center.npy')
            self.states[sid] = np.concatenate((z, a['packet'][:, ::4].transpose(1, 0, 2)[:199],
                a['order'][:, ::4].transpose(1, 0, 2)[:199],
                np.isin(a['labels'][:, ::4], [1, 2, 3]).T[:199, :, None]), -1).astype(np.float32)
        return self.states[sid]

    def observed(self, indices, spec):
        if spec.get('motion_input', False) or spec['equivariant'] or spec['information_context'] == 'both_new':
            raise ValueError('This figure recipe is the fixed scalar-context family comparison')
        result = self.corpus.inputs(indices, spec)
        result = {k: result[k] for k in ('features', 'geometry', 'condition')}
        result['features'] = result['features'][..., :128]
        info = []
        for index in indices:
            sid, ai, ci, _ = self.corpus.rows[index]; frame = self.plan['anchors'][ai]
            if sid not in self.info:
                source = self.sources[sid]; path = self.assay / source['shard']
                if digest(path) != source['shard_sha256']:
                    raise ValueError(f'Changed context shard {sid}')
                with np.load(path) as a:
                    np.testing.assert_array_equal(a['atom_ids'], source['center_atom_ids'])
                    local = np.concatenate((a['packet'], a['order']), -1)[..., LOCAL_COLUMNS]
                    shell = a['shell'][..., [0, 1, 6, 7]]
                    self.info[sid] = torch.tensor(np.concatenate((local, shell), -1)[:, :665:4].transpose(1, 0, 2)[None])
            rows = torch.tensor([[0, frame, ci, 0]])
            info.append(torch.cat((information_values(self.info[sid], rows), torch.zeros((1, 128))), -1))
        result['information'] = torch.cat(info)
        return result


def aggregate(config, corpus, predictions):
    """Equal source weighting and paired source bootstrap; no window independence."""
    ids = predictions['direct']['test_indices']; sources = corpus.source_ids[ids]
    roots = np.unique(sources); rng = np.random.default_rng(config['seed'])
    bootstrap = rng.integers(len(roots), size=(config['bootstrap_draws'], len(roots)))
    values = {}; metrics = {}
    for name, p in predictions.items():
        np.testing.assert_array_equal(p['test_indices'], ids)
        np.testing.assert_array_equal(p['test_event'], predictions['direct']['test_event'])
        brier = (p['test_cdf']-(np.arange(128)[None] >= p['test_event'][:, None]))**2
        arrays = dict(physical=p['test_path_scores'][:, :, 1], brier=brier)
        if name == 'direct':
            arrays['persistence'] = p['test_persistence_scores'][:, :, 1]
        for key, array in arrays.items():
            root_scores = np.stack([array[sources == sid].mean(0) for sid in roots])
            mean = root_scores.mean(0); low, high = np.quantile(root_scores[bootstrap].mean(1), [.025, .975], axis=0)
            values[f'{name}_{key}'] = np.stack((mean, low, high))
            metrics[f'{name}_{key}'] = dict(mean_over_horizon=float(mean.mean()),
                points={str((i+1)*(.75 if key == 'brier' else 3)): float(v) for i, v in enumerate(mean)})
    return values, metrics


def embedding_map(config, plan, replay, cases, forecasts, root):
    from sklearn.preprocessing import StandardScaler
    import umap
    train = []; test = []; metadata = []; rng = np.random.default_rng(config['seed'])
    for s in plan['sources']:
        role = s.get('validation_role', s['split'])
        if role not in ('train', 'test'):
            continue
        sid = s['id']; a = replay.corpus.arrays[sid]
        # Timelines are sampled independently of model errors, labels and event times.
        flat = np.sort(rng.choice(199*16, config['umap_per_source'], replace=False)); fi, ci = flat//16, flat%16
        path = resolve_path(plan['config']['future_cache']) / str(sid) / 'center.npy'
        state = np.load(path, mmap_mode='r'); z = np.array(state[fi, ci])
        if role == 'train':
            train.append(z)
        else:
            test.append(z)
            metadata.extend(zip([sid]*len(fi), fi*3, ci, [s['temperature_K']]*len(fi),
                a['order'][ci, fi*4, 1], np.isin(a['labels'][ci, fi*4], [1, 2, 3]).astype(int)))
    train = np.concatenate(train); test = np.concatenate(test)
    scaler = StandardScaler().fit(train)
    mapper = umap.UMAP(n_neighbors=config['umap_neighbors'], min_dist=config['umap_min_dist'],
        random_state=config['seed'], transform_seed=config['seed'], n_jobs=1, metric='euclidean')
    mapper.fit(scaler.transform(train)); test_xy = mapper.transform(scaler.transform(test))
    arrays = dict(test_xy=test_xy, metadata=np.array(metadata), train_mean=scaler.mean_, train_scale=scaler.scale_)
    all_z = []; sizes = []
    for i, case in enumerate(cases):
        f = case['frame']//4; actual = replay.state(case['source'])[f-16:f+33, case['center'], :128]
        all_z.append(actual); sizes.append((f'actual_{i}', len(actual)))
        for name, result in forecasts.items():
            # Transform the 128D predictive mean, not an average of 2D projections.
            z = result['paths'][i].mean(0)[:, :128]
            all_z.append(z); sizes.append((f'{name}_{i}', len(z)))
    joined = np.concatenate(all_z); xy = mapper.transform(scaler.transform(joined)); offset = 0
    for key, n in sizes:
        arrays[key] = xy[offset:offset+n]; offset += n
    np.savez_compressed(root/'technical/umap.npz', **arrays)
    import joblib
    joblib.dump(dict(scaler=scaler, umap=mapper), root/'technical/umap.joblib')
    save_json(root/'technical/umap-method.json', dict(training_rows=len(train), test_rows=len(test),
        train_sources=sum(s.get('validation_role', s['split']) == 'train' for s in plan['sources']),
        dimensions=128, seed=config['seed'], neighbors=config['umap_neighbors'], min_dist=config['umap_min_dist'],
        fit_population='64 outcome-independent states per training source over the 0–594 ps timeline; no test fitting',
        interpretation='Visualization only; 2D distances and forecast proximity are not predictive error metrics'))


def context_clouds(plan, replay, case, root):
    from scipy.spatial import cKDTree
    source = replay.sources[case['source']]
    raw = ShootingBinaryTrajectory.load(dataset_path(source['dataset']) / source['relative_trajectory_path'])
    if digest(raw.root/'manifest.json') != source['manifest_sha256']:
        raise ValueError('Raw point-cloud manifest changed')
    frame = case['frame']; center_id = source['center_atom_ids'][case['center']]
    center = int(np.searchsorted(raw.atom_ids, center_id))
    if raw.atom_ids[center] != center_id:
        raise ValueError('Tracked atom identity missing')
    box = (raw.box_high[frame]-raw.box_low[frame]).astype(float)
    points = np.mod(raw.positions[frame].astype(float), box); tree = cKDTree(points, boxsize=box)
    ids = representatives(points, center, tree, box)
    rel = points[ids]-points[center]; rel -= box*np.round(rel/box)
    a = replay.corpus.arrays[source['id']]
    np.testing.assert_allclose(rel, a['relative'][frame//4, case['center']], atol=1e-5, rtol=0)
    radius = OUTER_RADIUS*plan['scale']/REFERENCE_RADIUS
    full_ids = np.array(tree.query_ball_point(points[center], 25+radius))
    full = points[full_ids]-points[center]; full -= box*np.round(full/box)
    clouds = dict(full=full, representatives=rel, representative_atom_ids=np.array(raw.atom_ids[ids]))
    counts = []
    for j, atom in enumerate(ids):
        keep = np.array(tree.query_ball_point(points[atom], radius)); x = points[keep]-points[atom]; x -= box*np.round(x/box)
        x = x[np.linalg.norm(x, axis=1) < radius]
        g = int(a['mapping'][frame//4, case['center'], j]); lo, hi = a['offsets'][g:g+2]
        np.testing.assert_equal(len(x), hi-lo)
        clouds[f'local_{j}'] = x; counts.append(len(x))
    np.savez_compressed(root/'technical/context-clouds.npz', **clouds)
    save_json(root/'technical/context-method.json', dict(source=source['id'], frame=frame, time_ps=frame*.75,
        tracked_atom_id=int(center_id), local_radius_A=radius, atom_counts=counts,
        representative_distances_A=np.linalg.norm(rel, axis=1).tolist(),
        raw_manifest=str(raw.root/'manifest.json'), dtype=str(raw.positions.dtype),
        construction='Center + nearest-seeded farthest-point sampling of 3 atoms in (0,12] and 3 in (12,25] A, separately at each observed frame',
        geometry='minimum-image displacement in the source periodic box; each representative has its own local MACE crop'))


def prepare(config):
    root = resolve_path(config['output']); tech = root/'technical'; tech.mkdir(parents=True, exist_ok=True)
    if (tech/'prepared.json').exists():
        saved = json.loads((tech/'prepared.json').read_text())
        if saved['config'] != config:
            raise ValueError('Figure configuration changed; choose a new output directory')
        return
    # Corpus memory-maps the same 150 per-source caches as the training producer.
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(max(soft, 8192), hard), hard))
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    producer = assert_producer(config); base = resolve_path(config['input'])/'technical'
    plan = json.loads((base/'plan.json').read_text()); corpus = Corpus(plan); replay = Replay(plan, corpus)
    predictions = {}; hashes = {}
    for name, run in config['models'].items():
        folder = base/'runs'/run
        if json.loads((folder/'status.json').read_text())['state'] != 'complete':
            raise ValueError(f'Incomplete model {run}')
        with np.load(folder/'predictions.npz') as a:
            predictions[name] = {k: a[k] for k in a.files if not k.startswith('calibration_path') and not k.startswith('calibration_persistence')}
        hashes[name] = dict(checkpoint=digest(folder/'best.pt'), predictions=digest(folder/'predictions.npz'))
    cases = choose_examples(corpus, predictions['direct']); save_json(tech/'examples.json', cases)
    print('Selected four distinct test sources:', [c['source'] for c in cases], flush=True)
    curves, metrics = aggregate(config, corpus, predictions); np.savez_compressed(tech/'aggregate.npz', **curves)
    write_metric_table(metrics, root, family='trajectory_figures', name='forecast_curves')
    histories = {}; forecasts = {}; checks = {}
    for i, c in enumerate(cases):
        f = c['frame']//4
        histories[f'state_{i}'] = replay.state(c['source'])[f-16:f+33, c['center']]
    np.savez_compressed(tech/'observed.npz', **histories)
    for name, run in config['models'].items():
        folder = base/'runs'/run; saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        model = make_model(saved['spec']).eval(); model.load_state_dict(saved['model'], strict=True)
        with np.load(folder/'sample-trajectories.npz') as a:
            check_ids = a['indices'][:2].tolist(); target = a['target'][:2]
            saved_paths = a['paths'][:2].astype(np.float32)
        ids = [c['index'] for c in cases] + check_ids
        observed = replay.observed(ids, saved['spec'])
        torch.manual_seed(config['seed'])
        with torch.no_grad():
            paths, cdf = model.forecast(observed, samples=config['samples'], diffusion_steps=saved['spec']['diffusion_steps'])
        mean = saved['mean'].numpy(); scale = saved['scale'].numpy()
        target_replay = []
        for index in check_ids:
            sid, ai, ci, _ = corpus.rows[index]; f = plan['anchors'][ai]//4
            target_replay.append((replay.state(sid)[f+1:f+33, ci]-mean)/scale)
        np.testing.assert_allclose(target_replay, target, atol=1e-6, rtol=1e-5)
        if name in ('direct', 'ar_mse'):
            np.testing.assert_allclose(paths[-2:].numpy(), saved_paths, atol=.001, rtol=.002)
            expected_cdf = predictions[name]['test_cdf'][[c['test_row'] for c in cases]]
            np.testing.assert_allclose(cdf[:4].numpy(), expected_cdf, atol=3e-5, rtol=1e-4)
            checks[name] = dict(cdf_max_absolute_error=float(abs(cdf[:4].numpy()-expected_cdf).max()),
                saved_float16_path_max_error=float(abs(paths[-2:].numpy()-saved_paths).max()))
        value = dict(paths=paths[:4].numpy()*scale+mean,
            cdf=predictions[name]['test_cdf'][[c['test_row'] for c in cases]],
            history_ps=np.array(saved['spec']['history_ps']))
        forecasts[name] = value; np.savez_compressed(tech/f'{name}-examples.npz', **value)
        print('Replayed', name, 'on CPU', flush=True)
    save_json(tech/'replay-checks.json', checks)
    context_clouds(plan, replay, cases[0], root)
    print('Actual context point clouds verified against original cache', flush=True)
    embedding_map(config, plan, replay, cases, forecasts, root)
    save_json(tech/'prepared.json', dict(config=config, producer_hashes=producer, inputs=hashes,
        feature_checkpoint_sha256=plan['checkpoint_sha256'], plan_sha256=digest(base/'plan.json'),
        test_windows=len(predictions['direct']['test_indices']), test_sources=30,
        example_selection='Median-error source-balanced examples in predefined event strata; final case deliberately illustrates a missed alarm',
        probabilistic_paths='64 fresh CPU draws for illustration; event CDFs and aggregate scores use original archived evaluation predictions'))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    p.add_argument('--stage', choices=('all', 'prepare', 'render'), default='all'); args = p.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if args.stage in ('prepare', 'all'):
        prepare(config)
    if args.stage in ('render', 'all'):
        from .figure_render import render
        render(config)


if __name__ == '__main__':
    main()
