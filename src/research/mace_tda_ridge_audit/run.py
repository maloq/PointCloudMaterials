"""Fresh checkpoint inference, loss intervention, and independent ridge reproduction."""

import argparse
import ast
import csv
import json
from pathlib import Path
import tarfile
import time
from unittest.mock import patch

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import default_collate

from src.analysis.topology_metrics import ridge_predictions, score
from src.data.relaxed_histories import RelaxedHistoryDataset
from src.data_utils.topology_targets import raw_prediction, transform_target
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json, resolve_path
from src.training_methods.contrastive_learning.vicreg_module import VICRegModule
from src.utils.model_utils import load_model_from_checkpoint
from .math import balanced_errors, paired_interval, ridge_predict


def read_json(path):
    """Read producer data verbatim; path expansion is reserved for the run recipe."""
    return json.loads(Path(path).read_text())


def load_recipe(path):
    config = load_json(path)
    for item in config['runs']:
        report = Path(item['report'])
        metrics = read_json(report)['topology']
        source = read_json(report.parent/'source.json')
        item['checkpoint'] = str(resolve_path(metrics['checkpoint']))
        item['historical_predictions'] = str(resolve_path(source['analysis_directory'])/'topology/test_predictions.npz')
    return config


def source_audit(checkpoint):
    record = read_json(checkpoint.parent / 'run_record.json')
    comparisons = [
        ('src/training_methods/contrastive_learning/vicreg_module.py',
         'src/training_methods/contrastive_learning/vicreg_module.py',
         ['_spatiotemporal_step', '_weighted_total_loss']),
        ('src/training_methods/base_ssl_module.py',
         'src/training_methods/shared/base_ssl_module.py',
         ['_weighted_total_loss', '_output_representation']),
        ('src/data_utils/relaxed_histories.py', 'src/data/relaxed_histories.py', ['__getitem__']),
    ]
    results = {}
    with tarfile.open(record['source_snapshot']) as archive:
        for old, current, names in comparisons:
            trees = [ast.parse(archive.extractfile(old).read()), ast.parse(Path(current).read_text())]
            for name in names:
                functions = [next(n for n in ast.walk(tree)
                                  if isinstance(n, ast.FunctionDef) and n.name == name) for tree in trees]
                equal = ast.dump(functions[0]) == ast.dump(functions[1])
                if not equal:
                    raise ValueError(f'Historical function changed: {checkpoint}: {current}:{name}')
                results[current + ':' + name] = equal
    return results


def data_audit(config, root):
    cache = Path(config['cache'])
    manifest = read_json(cache/'manifest.json')
    for name, digest in manifest['checksums'].items():
        if sha256(cache/name) != digest:
            raise ValueError(f'Prepared input checksum mismatch: {cache/name}')
    original = read_json(manifest['protocol']['source_manifest'])
    if sha256(Path(manifest['protocol']['source_manifest'])) != manifest['protocol']['source_sha256']:
        raise ValueError('Original relaxed-data manifest checksum changed')
    targets = np.load(cache/'targets.npy')
    scaling = dict(np.load(cache/'scaling.npz'))
    contexts, sources, splits, temperatures, points, raw_original = [], [], [], [], [], []
    for record, parent in zip(manifest['shards'], original['shards'], strict=True):
        if record['source'] != parent['source_index'] or record['split'] != parent['split']:
            raise ValueError(f'Prepared source/split differs from original: {record}')
        for name in ['targets.npy', 'histories.npy']:
            if sha256(Path(parent['directory'])/name) != parent['checksums'][name]:
                raise ValueError(f'Original {name} checksum changed: {parent["directory"]}')
        views = np.load(cache/record['views'], mmap_mode='r')
        original_views = np.load(Path(parent['directory'])/'histories.npy', mmap_mode='r')
        np.testing.assert_array_equal(views[:, 0], original_views)
        points.append(views[:, 0, -1].astype(np.float32) / manifest['protocol']['radius_A'])
        raw_original.append(np.load(Path(parent['directory'])/'targets.npy'))
        for items, value in [(contexts, record['context']), (sources, record['source']),
                             (splits, record['split']), (temperatures, record['temperature_K'])]:
            items.extend([value] * record['samples'])
    np.testing.assert_array_equal(targets, np.concatenate(raw_original))
    sources, contexts, temperatures, splits = map(np.asarray, (sources, contexts, temperatures, splits))
    indices = {s: np.flatnonzero(splits == s) for s in ['train', 'val', 'test']}
    source_sets = {s: set(sources[ids]) for s, ids in indices.items()}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if source_sets[a] & source_sets[b]:
            raise ValueError(f'Source leakage between {a} and {b}')
    train_targets = targets[indices['train']]
    std = np.array([train_targets[:, a:b].var(axis=0).mean() ** .5 for a,b in [(0,16),(16,80),(80,144)]])
    expected_scale = np.sqrt(std**2 + (std.max()*manifest['protocol']['block_scale_floor_fraction'])**2)
    np.testing.assert_allclose(scaling['block_scale'], expected_scale, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(scaling['tda_mean'], train_targets.mean(axis=0), rtol=1e-6, atol=1e-7)
    # Deliberately damage held-out labels: neither fitted scale nor ridge weights may use them.
    write_json(root/'technical/data-audit.json', dict(
        cache_manifest_sha256=sha256(cache/'manifest.json'), verified_files=len(manifest['checksums']),
        original_anchor_and_target_arrays_equal=True, train_only_target_scaling_verified=True,
        sample_counts={s:len(ids) for s,ids in indices.items()},
        source_counts={s:len(ids) for s,ids in source_sets.items()},
        source_disjoint=True, block_scale=scaling['block_scale'].tolist()))
    return dict(points=np.concatenate(points), targets=targets, scaling=scaling,
                sources=sources, contexts=contexts, temperatures=temperatures, indices=indices)


def gradient_audit(model, payload, cfg):
    dataset = RelaxedHistoryDataset(cfg, 'train')
    locations = np.linspace(0, len(dataset)-1, 12, dtype=int)
    batch = default_collate([dataset[int(i)] for i in locations])
    params = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    results = []
    gradients = []
    for permute in [False, False, True]:
        model.load_state_dict(payload['state_dict'], strict=True)
        model.train()
        torch.manual_seed(711)
        changed = dict(batch)
        if permute:
            changed['tda_targets'] = batch['tda_targets'].roll(1, dims=0)
        changed['tda_targets'] = changed['tda_targets'].clone().requires_grad_(True)
        captured = {}
        def capture(**kwargs):
            captured.update(kwargs['losses'])
            return model._weighted_total_loss(kwargs['losses'])
        with patch.object(model, '_finish_ssl_step', capture), patch.object(model, '_log_metric', lambda *a, **k: None):
            loss = model._spatiotemporal_step(changed, 0, 'train')
        target_gradient, = torch.autograd.grad(loss, changed['tda_targets'], retain_graph=True, allow_unused=True)
        grads = torch.autograd.grad(loss, [p for _,p in params], retain_graph='tda' in captured, allow_unused=True)
        gradients.append({n:g.detach().cpu() for (n,p),g in zip(params,grads) if g is not None})
        info = dict(permuted_targets=permute, total=float(loss.detach()),
                    losses={k:float(v.detach()) for k,v in captured.items()},
                    target_gradient_norm=None if target_gradient is None else float(target_gradient.norm()))
        if 'tda' in captured:
            tda_grads = torch.autograd.grad(captured['tda'], [p for _,p in params], allow_unused=True)
            norms = {}
            for prefix in ['encoder.', 'vicreg.projector.', 'tda_head.']:
                squared = sum(float(g.detach().double().square().sum()) for (n,p),g in zip(params,tda_grads)
                              if n.startswith(prefix) and g is not None)
                norms[prefix] = squared**.5
            if any(v == 0 or not np.isfinite(v) for v in norms.values()):
                raise ValueError(f'TDA gradients do not reach expected modules: {norms}')
            info['tda_only_gradient_norms'] = norms
        results.append(info)
    differences = {k:float((gradients[0][k]-gradients[2][k]).abs().max()) for k in gradients[0]}
    gradient_scale = sum(float(g.double().square().sum()) for g in gradients[0].values())**.5
    relative_changes = [sum(float((gradients[0][k]-grads[k]).double().square().sum())
                            for k in gradients[0])**.5 / gradient_scale for grads in gradients[1:]]
    if not cfg.tda.enabled:
        # CUDA scatter reductions are not bitwise deterministic. Compare against
        # a repeated unchanged-target control and require negligible relative drift.
        if ('tda' in results[0]['losses'] or any(r['target_gradient_norm'] is not None for r in results) or max(relative_changes) > 1e-4 or
                not np.allclose([r['total'] for r in results], results[0]['total'], rtol=1e-6, atol=1e-7)):
            raise ValueError(f'Disabled-TDA target intervention failed: {results}; gradient relative changes={relative_changes}')
    elif (results[0]['total'] == results[2]['total'] or relative_changes[1] < 1e-4
          or any(r['target_gradient_norm'] is None or r['target_gradient_norm'] == 0 for r in results)):
        raise ValueError('Enabled-TDA objective is insensitive to topology labels')
    model.load_state_dict(payload['state_dict'], strict=True)
    model.eval()
    return dict(cases=results, maximum_gradient_change=max(differences.values()),
                unchanged_target_gradient_relative_drift=relative_changes[0],
                permuted_target_gradient_relative_change=relative_changes[1],
                tda_enabled=bool(cfg.tda.enabled), target_intervention_passed=True,
                note='Original training-step forwards: real targets twice, then permuted targets; identical weights, RNG and inputs. No optimizer step. Direct target-gradient graph check; allow 1e-4 relative CUDA parameter-gradient rounding.')


@torch.no_grad()
def extract(model, data, batch_size, root):
    projector, encoder = [], []
    started = time.monotonic()
    for start in range(0, len(data['points']), batch_size):
        x = torch.from_numpy(data['points'][start:start+batch_size]).to(model.device)
        z, h, _ = model(x)
        if z.shape != (len(x),128) or h.shape != (len(x),256):
            raise ValueError(f'Wrong projector/encoder shapes: {z.shape}, {h.shape}')
        if start == 0:
            torch.testing.assert_close(model.encoder(x), h, rtol=1e-5, atol=1e-6)
        projector.append(z.cpu().numpy()); encoder.append(h.cpu().numpy())
        if start % (batch_size*25) == 0:
            print('EXTRACT', root.name, start, len(data['points']), round(time.monotonic()-started,1), flush=True)
    arrays = dict(projector=np.concatenate(projector), encoder=np.concatenate(encoder))
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise FloatingPointError('Nonfinite extracted embeddings')
    np.savez(root/'features.npz', **arrays)
    return arrays


def evaluate(arrays, data, reference, root, config):
    ids, target, scaling = data['indices'], data['targets'], data['scaling']
    transformed = transform_target(target, scaling, 'blocks')
    scores, errors = [], {}
    historical = np.load(reference)
    for key, actual in [('indices', ids['test']), ('sources', data['sources'][ids['test']]),
                        ('contexts', data['contexts'][ids['test']]), ('targets', target[ids['test']])]:
        np.testing.assert_array_equal(historical[key], actual)
    reproduction = {}
    for name, features in arrays.items():
        for split in ['val','test']:
            evaluation = ids[split]
            repository_prediction = raw_prediction(ridge_predictions(features, transformed,
                ids['train'], evaluation, 1.), scaling, 'blocks')
            independent = ridge_predict(features[ids['train']], target[ids['train']], features[evaluation])
            row_errors, block_errors = balanced_errors(independent, target[evaluation], scaling['block_scale'])
            repository_metric, repository_errors = score(repository_prediction, target[evaluation],
                data['contexts'][evaluation], data['temperatures'][evaluation], scaling)
            independent_repository_errors = balanced_errors(repository_prediction, target[evaluation], scaling['block_scale'])[0]
            np.testing.assert_allclose(repository_errors, independent_repository_errors, rtol=1e-12, atol=1e-12)
            difference = abs(row_errors.mean()-repository_metric['balanced_mse'])
            if difference > 1e-5:
                raise ValueError(f'Independent ridge differs materially: {name}/{split}: {difference}')
            scores.append(dict(representation=name, split=split, mse=float(row_errors.mean()),
                repository_mse=repository_metric['balanced_mse'], solver_mse_difference=float(difference),
                H0=float(block_errors[:,0].mean()), H1=float(block_errors[:,1].mean()), H2=float(block_errors[:,2].mean())))
            errors[name+'_'+split] = row_errors
            np.save(root/f'{name}_{split}_predictions.npy', independent)
            if name == 'projector' and split == 'test':
                old_errors = balanced_errors(historical['ridge_predictions'], historical['targets'], scaling['block_scale'])[0]
                np.testing.assert_allclose(old_errors, historical['ridge_errors'], rtol=1e-12, atol=1e-12)
                delta = repository_metric['balanced_mse'] - float(old_errors.mean())
                if abs(delta) > 1e-5:
                    raise ValueError(f'Fresh checkpoint evaluation does not reproduce historical score: {delta}')
                reproduction = dict(historical_mse=float(old_errors.mean()), fresh_repository_mse=repository_metric['balanced_mse'],
                    fresh_independent_mse=float(row_errors.mean()), delta=float(delta),
                    projector_max_abs_difference=float(np.abs(features[evaluation]-historical['embeddings']).max()),
                    projector_relative_l2=float(np.linalg.norm(features[evaluation]-historical['embeddings']) / np.linalg.norm(historical['embeddings'])))
        # Negative control: break the input/target relation during fitting only.
        shuffle = np.random.default_rng(config['seed']).permutation(ids['train'])
        prediction = ridge_predict(features[ids['train']], target[shuffle], features[ids['test']])
        scores.append(dict(representation=name+'_shuffled_training_targets', split='test',
            mse=float(balanced_errors(prediction, target[ids['test']], scaling['block_scale'])[0].mean())))
    mean_prediction = np.broadcast_to(target[ids['train']].mean(axis=0), target[ids['test']].shape)
    scores.append(dict(representation='training_mean', split='test',
        mse=float(balanced_errors(mean_prediction, target[ids['test']], scaling['block_scale'])[0].mean())))
    np.savez(root/'errors.npz', **errors, test_sources=data['sources'][ids['test']])
    write_json(root/'scores.json', scores)
    write_json(root/'reproduction.json', reproduction)
    return scores


def summarize(root, config):
    rows, checks = [], {}
    for run in config['runs']:
        directory = root/'technical'/run['name']
        checks[run['name']] = read_json(directory/'reproduction.json')
        rows.extend(dict(method=run['method'], seed=run['seed'], **r) for r in read_json(directory/'scores.json'))
    write_json(root/'technical/reproduction.json', checks)
    comparisons = {}
    for representation in ['projector', 'encoder']:
        grouped = {}
        for method in ['vicreg','tda']:
            selected = sorted([r for r in config['runs'] if r['method']==method], key=lambda r:r['seed'])
            grouped[method] = [np.load(root/'technical'/r['name']/'errors.npz')[representation+'_test'] for r in selected]
        sources = np.load(root/'technical'/config['runs'][0]['name']/'errors.npz')['test_sources']
        comparisons[representation] = paired_interval(grouped['vicreg'], grouped['tda'], sources,
            seed=config['seed'], draws=config['bootstrap_draws'])
    write_json(root/'technical/comparisons.json', comparisons)
    snapshot_metric_docs(root, 'mace_tda_ridge_audit')
    fields = ['method','seed','representation','split','mse','repository_mse','solver_mse_difference','H0','H1','H2']
    with (root/'tables/scores.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,4.4),constrained_layout=True)
    for ax,representation in zip(axes,['projector','encoder']):
        for i,method in enumerate(['vicreg','tda']):
            selected=sorted([r for r in rows if r['representation']==representation and r['method']==method and r['split']=='test'],key=lambda r:r['seed'])
            ax.scatter([i]*len(selected),[r['mse'] for r in selected],s=55,color=['#345995','#d87532'][i],zorder=3)
            ax.plot([i-.18,i+.18],[np.mean([r['mse'] for r in selected])]*2,color='black',lw=2)
        for seed in sorted({r['seed'] for r in rows}):
            points=[next(r['mse'] for r in rows if r['representation']==representation and r['method']==m and r['split']=='test' and r['seed']==seed) for m in ['vicreg','tda']]
            ax.plot([0,1],points,color='#999999',lw=1,zorder=1)
        ax.set_xticks([0,1],['VICReg','VICReg + TDA']);ax.set_xlim(-.4,1.4)
        ax.set_title('128D projector: original claim' if representation=='projector' else '256D encoder: forecasting representation')
        ax.set_ylabel('Held-out balanced ridge MSE ↓');ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'plots/ridge-retest.png',dpi=180);fig.savefig(root/'plots/ridge-retest.pdf');plt.close(fig)
    write_json(root/'technical/status.json',dict(state='complete',runs=len(config['runs'])))


def precision_audit(root, config):
    """Reapply the original analysis's TF32 setting to the saved fresh encodings."""
    from src.training_methods.shared.vicreg import VICRegLoss
    torch.set_num_threads(config['cpu_threads']);torch.cuda.set_device(config['device'])
    cache = Path(config['cache'])
    manifest = read_json(cache/'manifest.json')
    splits = np.concatenate([np.repeat(r['split'],r['samples']) for r in manifest['shards']])
    train, test = np.flatnonzero(splits=='train'), np.flatnonzero(splits=='test')
    targets = np.load(cache/'targets.npy');scaling = dict(np.load(cache/'scaling.npz'))
    transformed = transform_target(targets, scaling, 'blocks')
    results = []
    for item in config['runs']:
        checkpoint=Path(item['checkpoint']);directory=root/'technical'/item['name']
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=OmegaConf.load(checkpoint.parent/'.hydra/config.yaml')
        projector=VICRegLoss.from_config(cfg,input_dim=256).to(config['device']).eval()
        projector.load_state_dict({k.removeprefix('vicreg.'):v for k,v in payload['state_dict'].items() if k.startswith('vicreg.')},strict=True)
        features=np.load(directory/'features.npz')['encoder']
        projected=[]
        torch.set_float32_matmul_precision('high')
        with torch.no_grad():
            for start in range(0,len(features),config['batch_size']):
                projected.append(projector.project_features(torch.from_numpy(features[start:start+config['batch_size']]).to(config['device'])).cpu().numpy())
        projected=np.concatenate(projected)
        historical=np.load(item['historical_predictions'])
        predicted=raw_prediction(ridge_predictions(projected,transformed,train,test,1.),scaling,'blocks')
        mse=float(balanced_errors(predicted,targets[test],scaling['block_scale'])[0].mean())
        old_mse=float(balanced_errors(historical['ridge_predictions'],targets[test],scaling['block_scale'])[0].mean())
        record=dict(method=item['method'],seed=item['seed'],historical_mse=old_mse,
            matching_precision_mse=mse,delta=mse-old_mse,
            projector_relative_l2=float(np.linalg.norm(projected[test]-historical['embeddings'])/np.linalg.norm(historical['embeddings'])),
            projector_max_abs_difference=float(np.abs(projected[test]-historical['embeddings']).max()))
        if cfg.tda.enabled:
            head=torch.nn.Sequential(torch.nn.Linear(128,256),torch.nn.SiLU(),torch.nn.Linear(256,144)).to(config['device']).eval()
            head.load_state_dict({k.removeprefix('tda_head.'):v for k,v in payload['state_dict'].items() if k.startswith('tda_head.')},strict=True)
            with torch.no_grad():prediction=raw_prediction(head(torch.from_numpy(projected[test]).to(config['device'])).cpu().numpy(),scaling,'blocks')
            record['fresh_trained_head_mse']=float(balanced_errors(prediction,targets[test],scaling['block_scale'])[0].mean())
            record['historical_trained_head_mse']=float(balanced_errors(historical['predictions'],targets[test],scaling['block_scale'])[0].mean())
        results.append(record)
        print('PRECISION',json.dumps(record),flush=True)
    write_json(root/'technical/precision-audit.json',results)
    precision_root=root/'precision'
    snapshot_metric_docs(precision_root,'mace_tda_ridge_audit')
    fields=list(dict.fromkeys(key for row in results for key in row))
    with (precision_root/'tables/scores.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(results)


def run(config, stage):
    root = Path(config['output'])
    for name in ['technical','tables','plots']:(root/name).mkdir(parents=True,exist_ok=True)
    if stage == 'summarize':
        summarize(root,config);return
    if stage == 'precision':
        precision_audit(root,config);return
    write_json(root/'technical/status.json',dict(state='running',stage='data-audit'))
    torch.set_num_threads(config['cpu_threads']);torch.set_float32_matmul_precision('highest')
    # cuEquivariance launches kernels on the current device, including autotuning.
    torch.cuda.set_device(config['device'])
    data = data_audit(config,root)
    reference_payloads = {}
    for item in config['runs']:
        directory = root/'technical'/item['name'];directory.mkdir(exist_ok=True)
        checkpoint = Path(item['checkpoint'])
        report = read_json(item['report'])['topology']
        if sha256(checkpoint) != report['checkpoint_sha256']:
            raise ValueError(f'Checkpoint does not match original report: {checkpoint}')
        historical_source = source_audit(checkpoint)
        payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg = OmegaConf.load(checkpoint.parent/'.hydra/config.yaml')
        if bool(cfg.tda.enabled) != (item['method']=='tda') or bool(payload['hyper_parameters']['tda']['enabled']) != bool(cfg.tda.enabled):
            raise ValueError(f'TDA flag contradicts model grouping or checkpoint: {checkpoint}')
        cfg.data.cache_dir=config['cache']
        cfg.encoder.kwargs.activation_checkpointing=False
        if cfg.representation_source != 'vicreg_projector':raise ValueError('Historical ridge must use projector')
        model=load_model_from_checkpoint(str(checkpoint),cfg,device=config['device'],module=VICRegModule)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(),payload['state_dict'][name],rtol=0,atol=0,msg=name)
        if (model.tda_head is not None) != bool(cfg.tda.enabled):raise ValueError('Loaded TDA head presence mismatch')
        prefix_states={prefix:{k:v for k,v in payload['state_dict'].items() if k.startswith(prefix)} for prefix in ['encoder.','vicreg.projector.']}
        parameter_difference={}
        if item['method']=='vicreg':reference_payloads[item['seed']]=prefix_states
        else:
            for prefix,state in prefix_states.items():
                base=reference_payloads[item['seed']][prefix]
                difference=sum(float((v.double()-base[k].double()).square().sum()) for k,v in state.items())**.5
                if difference==0:raise ValueError(f'Duplicate VICReg/TDA {prefix} weights')
                parameter_difference[prefix]=difference
        write_json(directory/'checkpoint-audit.json',dict(checkpoint_sha256=sha256(checkpoint),
            epoch=payload['epoch'],global_step=payload['global_step'],tda=OmegaConf.to_container(cfg.tda,resolve=True),
            source_functions_match_training_snapshot=historical_source,
            strict_loaded_state_exact=True,paired_state_difference_norm=parameter_difference))
        write_json(root/'technical/status.json',dict(state='running',stage='gradient-and-extraction',current=item['name']))
        write_json(directory/'gradient-audit.json',gradient_audit(model,payload,cfg))
        arrays=extract(model,data,config['batch_size'],directory)
        scores=evaluate(arrays,data,Path(item['historical_predictions']),directory,config)
        print('SCORES',item['name'],json.dumps(scores),flush=True)
        del model,payload;torch.cuda.empty_cache()
    summarize(root,config)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['all','summarize','precision','initialization','direct'],default='all');args=parser.parse_args()
    configuration=load_json(args.config) if args.stage in ('initialization','direct') else load_recipe(args.config)
    try:
        if args.stage=='direct':
            from .direct import run_direct
            run_direct(configuration)
        elif args.stage=='initialization':
            from .initialization import run_initialization
            run_initialization(configuration)
        else:
            run(configuration,args.stage)
    except Exception as error:
        write_json(Path(configuration['output'])/'technical/status.json',dict(state='failed',error=repr(error)))
        raise
