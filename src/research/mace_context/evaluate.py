"""Matched held-out topology reconstruction and controlled rank-boundary crossings."""

import csv
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from src.data_utils.topology_targets import fit_targets
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from .data import load_clouds, read_json
from .engine import encode, load_model
from src.research.mace_tda_ridge_audit.math import balanced_errors, ridge_path, paired_interval


def crossing_clouds(cloud, epsilon):
    order = np.argsort(np.linalg.norm(cloud-cloud[0],axis=1),kind='stable')
    a, b = order[79:81]
    radius = np.linalg.norm(cloud[[a,b]]-cloud[0],axis=1)
    directions = (cloud[[a,b]]-cloud[0])/radius[:,None]
    midpoint = radius.mean()
    values = []
    for sign in [-1,1]:
        x = cloud.copy()
        x[[a,b]] = cloud[0]+directions*(midpoint+sign*epsilon*np.array([1,-1]))[:,None]
        # The baseline must actually reselect k nearest atoms on each side.
        values.append(x[np.argsort(np.linalg.norm(x-x[0],axis=1),kind='stable')])
    return values


def extract(config, mode, *, model=None, trained=False):
    if mode not in config['modes']:
        raise ValueError('Choose an explicit context mode')
    root = Path(config['output'])/'technical'/f'{"trained" if trained else "frozen"}-{mode}'
    root.mkdir(parents=True,exist_ok=True)
    if (root/'features.npz').exists():
        raise FileExistsError(f'Preserve extracted context features: {root}')
    if model is None:
        model, _ = load_model(config)
    model.eval()
    manifest = read_json(Path(config['cache'])/'manifest.json')
    if manifest['state'] != 'complete':
        raise ValueError('Context cache is incomplete')
    probes = np.load(Path(config['diagnostics'])/'technical/probes.npz')
    temporal = np.load(Path(config['diagnostics'])/'technical/temporal.npz')
    z = np.full((len(probes['z']),256),np.nan,dtype=np.float32)
    tz = np.full((temporal['z'].shape[0]*17,256),np.nan,dtype=np.float32)
    started = time.monotonic()
    for index, record in enumerate(manifest['records']):
        path = Path(config['cache'])/record['file']
        if sha256(path) != record['sha256']:
            raise ValueError(f'Context input changed: {path}')
        clouds = load_clouds(path)[::3]
        z[record['rows']] = encode(config,model,clouds,mode).cpu().numpy()
        write_json(root/'status.json',dict(state='running',stage='probes',context=index+1,total=90))
        if index % 10 == 0:
            print(f'EXTRACT {mode} context {index+1}/90',flush=True)
    boundaries = []
    for record in manifest['temporal']:
        path = Path(config['cache'])/record['file']
        if sha256(path) != record['sha256']:
            raise ValueError(f'Temporal input changed: {path}')
        clouds = load_clouds(path)
        tz[record['rows']] = encode(config,model,clouds,mode).cpu().numpy()
        # Four tracked atoms per test context, fixed before measuring effects.
        boundaries.extend(clouds[8*8:8*8+4])
    if not np.isfinite(z).all() or not np.isfinite(tz).all():
        raise ValueError(f'Incomplete feature extraction: {mode}')
    epsilons = [0.1,0.01,0.001,0.0001]
    crossings = []
    for epsilon in epsilons:
        clouds = [x for cloud in boundaries for x in crossing_clouds(cloud,epsilon)]
        crossings.append(encode(config,model,clouds,mode).cpu().numpy().reshape(len(boundaries),2,256))
    np.savez(root/'features.npz', z=z, temporal_z=tz.reshape(*temporal['z'].shape),
             crossing_z=np.stack(crossings),epsilons=epsilons)
    write_json(root/'provenance.json',dict(mode=mode,trained=trained,config=config,
        initial_checkpoint_sha256=sha256(Path(config['checkpoint'])),
        cache_manifest_sha256=sha256(Path(config['cache'])/'manifest.json'),
        elapsed_seconds=time.monotonic()-started))
    write_json(root/'status.json',dict(state='complete',mode=mode,trained=trained))
    print(f'EXTRACT COMPLETE {mode} trained={trained}',flush=True)


def scores(config, name, features, probes, temporal, destination):
    z = features['z'].astype(np.float64)
    ids = {split:np.flatnonzero(probes['split']==split) for split in ['train','val','test']}
    train, val, test = (ids[s] for s in ['train','val','test'])
    variance = z[train].var(axis=0).mean()
    singular = np.linalg.svd(z[train]-z[train].mean(0),compute_uv=False)
    out = dict(name=name, train_feature_variance=float(variance),
               feature_participation_rank=float(np.sum(singular**2)**2/np.sum(singular**4)),tda={},temporal=[],boundary=[])
    evaluation = np.concatenate([z[train],z[val],z[test],features['temporal_z'].reshape(-1,256)])
    nt, nv, ns = len(train),len(val),len(test)
    errors = {}
    for target in ['hot','relaxed']:
        y = probes[target]
        scales = fit_targets(y[train],32,.05)['block_scale']
        path = ridge_path(z[train],y[train],evaluation,config['ridge_alphas'])
        alpha = min(config['ridge_alphas'],key=lambda a:balanced_errors(path[a][nt:nt+nv],y[val],scales)[0].mean())
        prediction = path[alpha]
        level = {}
        for split, offset, rows in [('train',0,train),('val',nt,val),('test',nt+nv,test)]:
            error, blocks = balanced_errors(prediction[offset:offset+len(rows)],y[rows],scales)
            baseline = balanced_errors(np.broadcast_to(y[train].mean(0),y[rows].shape),y[rows],scales)[0].mean()
            level[split] = dict(mse=float(error.mean()),mean_baseline_mse=float(baseline),
                               mean_baseline_reduction=float(1-error.mean()/baseline),block_mse=blocks.mean(0).tolist())
            if split == 'test':
                errors[target] = error
        shuffled = y[train][np.random.default_rng(config['seed']).permutation(nt)]
        shuffled_path = ridge_path(z[train],shuffled,np.concatenate([z[val],z[test]]),config['ridge_alphas'])
        shuffled_alpha = min(config['ridge_alphas'],key=lambda a:balanced_errors(shuffled_path[a][:nv],y[val],scales)[0].mean())
        out['tda'][target] = dict(alpha=alpha,alpha_at_grid_edge=alpha in (config['ridge_alphas'][0],config['ridge_alphas'][-1]),
            levels=level,shuffled_test_mse=float(balanced_errors(shuffled_path[shuffled_alpha][nv:],y[test],scales)[0].mean()))
        within_prediction = prediction[nt+nv:nt+nv+ns].copy()
        within_target = y[test].astype(np.float64).copy()
        for context in np.unique(probes['context'][test]):
            mask = probes['context'][test] == context
            within_prediction[mask] -= within_prediction[mask].mean(0)
            within_target[mask] -= within_target[mask].mean(0)
        within_error = balanced_errors(within_prediction,within_target,scales)[0].mean()
        within_baseline = balanced_errors(np.zeros_like(within_target),within_target,scales)[0].mean()
        out['tda'][target]['test_within_context_reduction'] = float(1-within_error/within_baseline)
        if target == 'hot':
            temporal_prediction = prediction[nt+nv+ns:].reshape(*temporal['hot'].shape)
            for lag in config['temporal_lags_steps']:
                dz = features['temporal_z'][:,lag:].astype(np.float64)-features['temporal_z'][:,:-lag]
                dy = temporal['hot'][:,lag:]-temporal['hot'][:,:-lag]
                dp = temporal_prediction[:,lag:]-temporal_prediction[:,:-lag]
                error = balanced_errors(dp.reshape(-1,144),dy.reshape(-1,144),scales)[0].mean()
                baseline = balanced_errors(np.zeros_like(dy).reshape(-1,144),dy.reshape(-1,144),scales)[0].mean()
                out['temporal'].append(dict(lag_ps=lag*.75,relative_latent_mse=float(np.mean(dz**2)/variance),
                    tda_increment_mse=float(error),tda_persistence_mse=float(baseline),
                    tda_increment_reduction=float(1-error/baseline)))
    natural = np.mean(np.diff(features['temporal_z'].astype(np.float64),axis=1)**2)
    # These four targets were produced alongside the original probes from the
    # nearest twelve atoms, independently of the 80-atom TDA patch boundary.
    observable_names = ['q4','q6','nearest_shell_density','mean_r12_A']
    y = probes['geometry'][:, -4:].astype(np.float64)
    scale = y[train].std(axis=0)
    structural_path = ridge_path(z[train],y[train],evaluation,config['ridge_alphas'])
    alpha = min(config['ridge_alphas'],key=lambda a:np.mean(((structural_path[a][nt:nt+nv]-y[val])/scale)**2))
    prediction = structural_path[alpha]
    out['structural'] = dict(alpha=alpha,observables={})
    for column, observable in enumerate(observable_names):
        errors_by_split = {}
        for split, offset, rows in [('train',0,train),('val',nt,val),('test',nt+nv,test)]:
            mse = np.mean((prediction[offset:offset+len(rows),column]-y[rows,column])**2)
            baseline = np.mean((y[rows,column]-y[train,column].mean())**2)
            errors_by_split[split] = dict(mse=float(mse),mean_baseline_reduction=float(1-mse/baseline))
        out['structural']['observables'][observable] = errors_by_split
    structural_temporal = prediction[nt+nv+ns:].reshape(temporal['z'].shape[0],17,4)
    observed_temporal = temporal['observables'][:,:,2:]
    for row, lag in zip(out['temporal'],config['temporal_lags_steps'],strict=True):
        predicted_change=structural_temporal[:,lag:]-structural_temporal[:,:-lag]
        actual_change=observed_temporal[:,lag:]-observed_temporal[:,:-lag]
        reduction=1-np.mean((predicted_change-actual_change)**2,axis=(0,1))/np.mean(actual_change**2,axis=(0,1))
        row['structural_increment_reductions']=dict(zip(observable_names,reduction.tolist(),strict=True))
    for epsilon, crossing in zip(features['epsilons'],features['crossing_z'],strict=True):
        energy = np.mean((crossing[:,1].astype(np.float64)-crossing[:,0])**2)
        out['boundary'].append(dict(epsilon_A=float(epsilon),relative_latent_mse=float(energy/variance),
                                  fraction_of_075ps_energy=float(energy/natural)))
    np.savez(destination/'errors.npz', **errors, test_sources=probes['source'][test])
    write_json(destination/'scores.json',out)
    return out, errors


def summarize(config):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(config['output'])
    probes = np.load(Path(config['diagnostics'])/'technical/probes.npz')
    temporal = np.load(Path(config['diagnostics'])/'technical/temporal.npz')
    names = [f'frozen-{m}' for m in config['modes']]+[f'trained-{m}' for m in config['training_modes']]
    results, all_errors = [], {}
    for name in names:
        directory = root/'technical'/name
        if not (directory/'status.json').exists() or read_json(directory/'status.json')['state'] != 'complete':
            continue
        result, errors = scores(config,name,np.load(directory/'features.npz'),probes,temporal,directory)
        results.append(result);all_errors[name]=errors
    if not results:
        raise ValueError('No complete extractions to summarize')
    comparisons = {}
    for name in all_errors:
        reference = name.split('-')[0]+'-mean80'
        if name != reference and reference in all_errors:
            comparisons[name] = {target:paired_interval(all_errors[reference][target][None],all_errors[name][target][None],
                probes['source'][probes['split']=='test'],seed=config['seed'],draws=config['bootstrap_draws']) for target in ['hot','relaxed']}
    (root/'plots').mkdir(exist_ok=True);(root/'tables').mkdir(exist_ok=True)
    snapshot_metric_docs(root,'mace_context')
    rows = [dict(method=r['name'],hot_test_mse=r['tda']['hot']['levels']['test']['mse'],
                relaxed_test_mse=r['tda']['relaxed']['levels']['test']['mse'],
                hot_alpha=r['tda']['hot']['alpha'],relaxed_alpha=r['tda']['relaxed']['alpha'],
                boundary_fraction=r['boundary'][-1]['fraction_of_075ps_energy'],
                increment_reduction_075ps=r['temporal'][0]['tda_increment_reduction'],
                q6_test_reduction=r['structural']['observables']['q6']['test']['mean_baseline_reduction'],
                feature_rank=r['feature_participation_rank']) for r in results]
    with (root/'tables/comparison.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(root/'technical/summary.json',dict(results=results,comparisons=comparisons))
    completed = [r['name'] for r in results]
    progress = dict(state='complete' if set(completed)==set(names) else 'partial',
                    completed=completed,pending=[name for name in names if name not in completed])
    write_json(root/'technical/comparison-status.json',progress)
    fig,axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    for r in results:
        axes[0].plot([b['epsilon_A'] for b in r['boundary']],[b['fraction_of_075ps_energy'] for b in r['boundary']],'.-',label=r['name'])
        axes[2].plot([t['lag_ps'] for t in r['temporal']],[t['tda_increment_reduction'] for t in r['temporal']],'.-',label=r['name'])
    axes[0].set(xscale='log',yscale='log',xlabel='Crossing displacement parameter (A)',ylabel='Embedding jump / natural 0.75 ps energy',title='80th/81st atom crossing')
    labels=[r['name'].replace('frozen-','F: ').replace('trained-','T: ') for r in results]
    x=np.arange(len(results));width=.38
    axes[1].bar(x-width/2,[r['tda']['hot']['levels']['test']['mse'] for r in results],width,label='Hot 80-atom TDA')
    axes[1].bar(x+width/2,[r['tda']['relaxed']['levels']['test']['mse'] for r in results],width,label='Relaxed TDA')
    axes[1].set_xticks(x,labels,rotation=60,ha='right',fontsize=8)
    axes[1].set(ylabel='Balanced TDA MSE (lower is better)',title='Held-out source reconstruction');axes[1].legend(fontsize=8)
    axes[2].set(xlabel='Physical lag (ps)',ylabel='TDA increment error reduction vs persistence',title='Physical change remains measurable')
    axes[0].legend(fontsize=7)
    for ax in axes:ax.grid(alpha=.2)
    fig.savefig(root/'plots/context-comparison.png',dpi=180);fig.savefig(root/'plots/context-comparison.pdf');plt.close(fig)
    lines=['# MACE context pilot','',
       'Matched 5,760 anchors: 18 training, 6 validation and 6 test simulations. One encoder initialization; exploratory cohort already used in earlier diagnostics.',
       '', '| Method | Hot TDA MSE | Relaxed TDA MSE | Boundary / 0.75 ps energy |', '|---|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['method']} | {row['hot_test_mse']:.6f} | {row['relaxed_test_mse']:.6f} | {row['boundary_fraction']:.3g} |")
    lines += ['', '![Context comparison](plots/context-comparison.png)', '',
       'Readouts are float64 SVD ridge regressions, with alpha chosen on validation sources only. TDA labels still describe the original hard 80-atom support. A smooth embedding need not reproduce every discontinuity of that label definition.', '',
       'Training is a matched eight-epoch warm-start pilot at learning rate 1e-4 and batch 256, using the original spatial/temporal VICReg objective, projector, mirror and jitter settings. Encoder gradients are replayed in graph microbatches; the complete batch enters VICReg and projector BatchNorm. Validation VICReg loss selects the checkpoint. No TDA labels train the encoder.', '',
       'The halo uses complete native two-hop 5 A neighborhoods. Inner pooling has unit weight to 5 A and a quintic taper to zero at 7 A. The center variant returns the tracked center node. The 18 A candidate crop includes a checked augmentation margin. Feature width remains 256.', '',
       'Training continuation and feature extraction status are retained separately under technical/. No forecasting model has been retrained in this pilot.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    local=Path(config['local_output']);local.mkdir(parents=True,exist_ok=True)
    for sub in ['plots','tables']:
        shutil.copytree(root/sub,local/sub,dirs_exist_ok=True)
    shutil.copy2(root/'README.md',local/'README.md')
    (local/'technical').mkdir(exist_ok=True)
    for name in ['summary.json','metric-contract.json','verification.json','comparison-status.json']:
        if (root/'technical'/name).exists():shutil.copy2(root/'technical'/name,local/'technical'/name)
    print(rows,flush=True)
