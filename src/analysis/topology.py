"""Relaxed-topology diagnostics as a stage of the checkpoint analysis pipeline.

Model loading, inference collection/caching and plotting are owned by the main
pipeline. This stage adds supervised readout scores on explicit source splits.
"""

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf, open_dict
import torch

from src.data_utils.topology_targets import raw_prediction, transform_target
from src.experiment_runner.registry import sha256
from .inference_cache import _build_inference_cache_spec, _load_inference_cache, _save_inference_cache
from .output_layout import write_json
from .topology_dataset import RelaxedTopologyAnalysisDataset, topology_dataloader
from .topology_metrics import paired_source_gain, ridge_predictions, score
from .utils import gather_inference_batches


def _measure(prediction, dataset):
    base = dataset.base
    rows, contexts = base.indices, dataset.contexts
    temperatures = np.array([base.records[c]['temperature_K'] for c in contexts])
    result, errors = score(prediction, base.raw_targets[rows], contexts, temperatures, base.scaling)
    for key, values in [('frame', [base.records[c]['frame'] for c in contexts]),
                        ('source', [base.records[c]['source'] for c in contexts])]:
        values = np.asarray(values)
        result['by_'+key] = {}
        for value in np.unique(values):
            mask = values == value
            result['by_'+key][str(value)] = score(prediction[mask], base.raw_targets[rows][mask],
                contexts[mask], temperatures[mask], base.scaling)[0]
    return result, errors


def run_topology_analysis(*, model, cfg, analysis_cfg, checkpoint_path, out_dir, main_cache, step):
    settings = OmegaConf.select(analysis_cfg, 'topology')
    if settings is None or not settings.enabled:
        return {}
    if model is None:
        raise ValueError('Topology diagnostics require model inference; disable figure_only.')
    step('Evaluating relaxed topology with source-held-out linear probes')
    data_cfg = OmegaConf.merge(cfg, {'data': settings.data, 'tda': {'target': 'blocks'}})
    with open_dict(data_cfg):
        data_cfg.data.input_mode = ('history' if cfg.encoder.name == 'PretrainedMACEHistoryGeometry' else 'anchor')
        data_cfg.data.source_manifest_sha256 = sha256(Path(data_cfg.data.cache_dir)/'manifest.json')
        data_cfg.data.analysis_identity = 'atom_id_v1'
    root = Path(out_dir)/'topology'
    root.mkdir(exist_ok=True)
    datasets = {s: RelaxedTopologyAnalysisDataset(data_cfg, s) for s in ('train','val','test')}
    mode = 'temporal' if data_cfg.data.input_mode == 'history' else 'static_anchor'

    def collect(dataset, split, intervention='real'):
        cache_cfg = OmegaConf.create(OmegaConf.to_container(data_cfg, resolve=True))
        with open_dict(cache_cfg):
            cache_cfg.data.analysis_split = split
            cache_cfg.data.history_intervention = intervention
        spec = _build_inference_cache_spec(checkpoint_path=checkpoint_path, cfg=cache_cfg,
            inference_batch_size=settings.batch_size, max_batches_latent=None, max_samples_total=None,
            seed_base=analysis_cfg.runtime.seed_base, collector_mode='generic',
            temporal_sequence_inference={'mode':mode})
        name = f'{split}_{intervention}_inference.npz'
        cache, message = _load_inference_cache(out_dir=root, cache_filename=name, expected_spec=spec)
        if cache is None:
            print(f'[analysis][topology] {split}/{intervention}: {message}', flush=True)
            if split == 'test' and intervention == 'real' and cfg.data.kind == 'relaxed_histories':
                np.testing.assert_array_equal(main_cache['instance_ids'], dataset.atom_ids)
                cache = main_cache
            else:
                cache = gather_inference_batches(model,
                    topology_dataloader(dataset, settings.batch_size, settings.num_workers), str(model.device),
                    max_batches=None, max_samples_total=None, collect_coords=True,
                    seed_base=analysis_cfg.runtime.seed_base, verbose=True,
                    temporal_sequence_mode=mode)
            if bool(OmegaConf.select(settings, 'save_inference_cache', default=True)):
                _save_inference_cache(out_dir=root, cache_filename=name, cache=cache, spec=spec)
        if len(cache['inv_latents']) != len(dataset):
            raise ValueError(f'Topology cache omitted rows: {split}/{intervention}')
        return cache['inv_latents']

    embeddings = {s:collect(d, s) for s,d in datasets.items()}
    train = datasets['train'].base
    scaling = train.scaling
    features = np.concatenate(list(embeddings.values()))
    targets = np.concatenate([d.base.raw_targets[d.indices] for d in datasets.values()])
    transformed = transform_target(targets, scaling, 'blocks')
    boundaries = np.cumsum([0]+[len(d) for d in datasets.values()])
    fit_rows = np.arange(boundaries[1])
    mean = train.raw_targets[train.indices].mean(0)

    def head(features):
        with torch.inference_mode():
            prediction = model.tda_head(torch.from_numpy(features).to(model.device)).cpu().numpy()
        return raw_prediction(prediction, scaling, cfg.tda.target)

    results = {}
    for split, lo, hi in zip(datasets, boundaries[:-1], boundaries[1:]):
        if split == 'train':
            continue
        dataset = datasets[split]
        ridge = raw_prediction(ridge_predictions(features, transformed, fit_rows, np.arange(lo,hi),
            settings.ridge_alpha), scaling, 'blocks')
        predictions = dict(projector_ridge=ridge, training_mean=np.broadcast_to(mean, ridge.shape))
        predictions['prediction'] = ridge if model.tda_head is None else head(embeddings[split])
        measured = {name:_measure(prediction, dataset) for name,prediction in predictions.items()}
        results[split] = {name:value[0] for name,value in measured.items()}
        sources = np.array([dataset.base.records[c]['source'] for c in dataset.contexts])
        results[split]['ridge_vs_head'] = paired_source_gain(measured['prediction'][1],
            measured['projector_ridge'][1], sources, analysis_cfg.runtime.seed_base)
        if split == 'test' or bool(OmegaConf.select(settings, 'save_validation_predictions', default=True)):
            np.savez_compressed(root/f'{split}_predictions.npz',
                predictions=predictions['prediction'], ridge_predictions=ridge,
                targets=dataset.base.raw_targets[dataset.indices], embeddings=embeddings[split],
                errors=measured['prediction'][1], ridge_errors=measured['projector_ridge'][1],
                indices=dataset.indices, contexts=dataset.contexts, sources=sources)

    interventions = {}
    if data_cfg.data.input_mode == 'history':
        for name in settings.history_interventions:
            dataset = RelaxedTopologyAnalysisDataset(data_cfg, 'test', name)
            projected = collect(dataset, 'test', name)
            intervention_features = np.concatenate((embeddings['train'], projected))
            intervention_targets = np.concatenate((transformed[:boundaries[1]],
                transformed[boundaries[2]:]))
            ridge = raw_prediction(ridge_predictions(intervention_features, intervention_targets, fit_rows,
                np.arange(boundaries[1], len(intervention_features)), settings.ridge_alpha), scaling, 'blocks')
            # Fit the identical real-training probe; intervene only at test inference.
            interventions[name] = dict(projector_ridge=_measure(ridge, dataset)[0])
            if model.tda_head is not None:
                interventions[name]['prediction'] = _measure(head(projected), dataset)[0]

    result = dict(state='complete', checkpoint=str(checkpoint_path),
        checkpoint_sha256=sha256(Path(checkpoint_path)),
        variant=OmegaConf.select(cfg,'experiment_variant',default=cfg.experiment_name),
        seed=cfg.seed_everything, results=results, history_interventions=interventions,
        primary_prediction='TDA head' if model.tda_head is not None else 'training-only projector ridge',
        source_manifest_sha256=data_cfg.data.source_manifest_sha256,
        sample_counts={s:len(d) for s,d in datasets.items()},
        selection='Original best combined-validation-loss checkpoint; target scaling and probes use training sources only.',
        limitation='The six held-out sources were examined in the previous frozen-MACE study; this is a follow-up cohort.')
    write_json(root/'metrics.json', result)
    flat = {f'{s}/{key}':values['prediction'][key] for s,values in results.items()
            for key in ('balanced_mse','mean_within_frame_r2')}
    flat.update({f'{s}/projector_ridge_mse':values['projector_ridge']['balanced_mse'] for s,values in results.items()})
    import wandb
    if wandb.run is not None:
        wandb.run.summary.update(flat)
    result['flat_metrics'] = flat
    return result


def collect(root, specification):
    cfg = json.loads(Path(specification).read_text())
    report_root = cfg.get('report_root')
    if report_root is not None:
        manifests = sorted(Path(report_root).glob('*/technical/source.json')) + sorted(Path(report_root).glob('*/source.json'))
        files = [p.parent/'metrics.json' for p in manifests]
    else:
        files = sorted(set(Path(root).glob('*/**/analysis_standard/**/analysis_metrics.json'))
                       | set(Path(root).glob('*/**/technical/analysis_metrics.json')))
    groups = {name:[] for name in cfg['variants']}
    for path in files:
        result = json.loads(path.read_text())['topology']
        if result['variant'] in groups:
            groups[result['variant']].append((result, path))
    results, errors, ridge_errors = {}, {}, {}
    reference_sources = None
    for name, runs in groups.items():
        if sorted(r['seed'] for r,_ in runs) != sorted(cfg['seeds']):
            raise ValueError(f'Incomplete/duplicate seed results for {name}: {[r["seed"] for r,_ in runs]}')
        metrics = [r['results']['test']['prediction'] for r,_ in runs]
        results[name] = dict(mse_mean=float(np.mean([m['balanced_mse'] for m in metrics])),
            mse_std=float(np.std([m['balanced_mse'] for m in metrics])),
            within_frame_r2=float(np.mean([m['mean_within_frame_r2'] for m in metrics])),
            ridge_mse_mean=float(np.mean([r['results']['test']['projector_ridge']['balanced_mse'] for r,_ in runs])),
            ridge_mse_std=float(np.std([r['results']['test']['projector_ridge']['balanced_mse'] for r,_ in runs])))
        seed_errors, seed_ridge_errors = [], []
        for _, path in runs:
            source_dir = (Path(json.loads((path.parent/'source.json').read_text())['analysis_directory'])
                          if report_root is not None else path.parent)
            with np.load(source_dir/'topology/test_predictions.npz') as saved:
                if reference_sources is None:
                    reference_sources = saved['sources']
                    reference_rows = saved['indices']
                np.testing.assert_array_equal(reference_sources, saved['sources'])
                np.testing.assert_array_equal(reference_rows, saved['indices'])
                seed_errors.append(saved['errors'])
                seed_ridge_errors.append(saved['ridge_errors'])
        errors[name] = np.mean(seed_errors, axis=0)
        ridge_errors[name] = np.mean(seed_ridge_errors, axis=0)
    comparisons = {f'{candidate}_versus_{reference}':paired_source_gain(errors[reference], errors[candidate],
        reference_sources, cfg['bootstrap_seed']) for reference,candidate in cfg['comparisons']}
    ridge_comparisons = {f'{candidate}_versus_{reference}':paired_source_gain(
        ridge_errors[reference], ridge_errors[candidate], reference_sources, cfg['bootstrap_seed'])
        for reference,candidate in cfg['comparisons']}
    from src.experiment_runner.artifacts import result_folders
    output = result_folders(Path(root)/'comparison')
    write_json(output/'technical/metrics.json', dict(results=results, comparisons=comparisons,
                                         ridge_comparisons=ridge_comparisons))
    lines = ['# MEAM comparison in the original VICReg trainer', '',
        '| Variant | Test balanced MSE | Within-frame R² | Projector ridge MSE |', '|---|---:|---:|---:|']
    for name, r in results.items():
        lines.append(f'| {name} | {r["mse_mean"]:.6f} ± {r["mse_std"]:.6f} | {r["within_frame_r2"]:.4f} | {r["ridge_mse_mean"]:.6f} ± {r["ridge_mse_std"]:.6f} |')
    lines += ['', '| Comparison | Error reduction | 95% source-bootstrap interval |', '|---|---:|---:|']
    for name,r in comparisons.items():
        lo,hi = r['source_bootstrap_95_percent_interval']
        lines.append(f'| {name} | {100*r["relative_mse_reduction"]:.2f}% | [{100*lo:.2f}%, {100*hi:.2f}%] |')
    lines += ['', '| Same ridge readout comparison | Error reduction | 95% source-bootstrap interval |', '|---|---:|---:|']
    for name,r in ridge_comparisons.items():
        lo,hi = r['source_bootstrap_95_percent_interval']
        lines.append(f'| {name} | {100*r["relative_mse_reduction"]:.2f}% | [{100*lo:.2f}%, {100*hi:.2f}%] |')
    lines += ['', 'Intervals resample six whole source trajectories after averaging seeds. Comparisons are exploratory; this cohort was already evaluated in the frozen-MACE experiment.']
    from src.experiment_runner.metric_docs import write_metric_table
    write_metric_table(dict(results=results, comparisons=comparisons, ridge_comparisons=ridge_comparisons),
                       output, family='topology', name='comparison')
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(output/'RESULTS.md', flush=True)
