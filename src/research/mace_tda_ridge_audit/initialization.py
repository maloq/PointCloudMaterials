"""Frozen random/MLIP controls using the audited encoder-space ridge protocol."""

import csv
import hashlib
import inspect
from pathlib import Path
import shutil
import time

import numpy as np
from omegaconf import OmegaConf
import torch

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from src.models.encoders.pretrained_mace import PretrainedMACEGeometryEncoder
from src.project_runtime.paths import resolve_path
from .math import balanced_errors, paired_interval, ridge_predict, ridge_path
from .run import data_audit, load_recipe, read_json


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def random_native(pretrained, seed):
    """Invoke MACE's native constructor; never transfer learned MLIP tensors."""
    from mace.tools.scripts_utils import extract_config_mace_model
    config = extract_config_mace_model(pretrained)
    # These energy-only quantities are unused by the encoder, but neutralize them
    # as well so the saved random artifact contains no fitted atomic offsets.
    config['atomic_energies'] = np.zeros_like(config['atomic_energies'])
    config['atomic_inter_scale'] = np.ones_like(config['atomic_inter_scale'])
    config['atomic_inter_shift'] = np.zeros_like(config['atomic_inter_shift'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = type(pretrained)(**config).float().eval()
    original = dict(pretrained.named_parameters())
    if set(original) != set(dict(model.named_parameters())):
        raise ValueError('Random architecture changed its learned parameter names')
    checks = {}
    for name, value in model.named_parameters():
        other = original[name]
        if value.shape != other.shape or torch.equal(value, other):
            raise ValueError(f'Random initialization retained MLIP parameters or changed shape: {name}')
        checks[name] = dict(shape=list(value.shape), equal_to_mlip=False,
                           difference_l2=float((value.detach()-other.detach()).double().norm()))
    return model, dict(seed=seed, parameters=checks,
        parameter_count=sum(p.numel() for p in model.parameters()),
        initialization='Fresh native MACE constructor, with no state_dict transfer',
        avg_num_neighbors=config['avg_num_neighbors'],
        shared_priors='Same architecture, cutoff, element table, analytic geometric bases and MLIP neighbor-count normalization scalar. Energy offsets neutralized; energy heads unused.',
        constructor_file=inspect.getfile(type(model)),
        config_extractor_file=inspect.getfile(extract_config_mace_model))


@torch.no_grad()
def extract_frozen(checkpoint, data, config, radius, directory):
    performance = config['performance']
    model = PretrainedMACEGeometryEncoder(str(checkpoint), radius, performance)
    model.to(config['device']).requires_grad_(False).eval()
    before = state_digest(model)
    features = []
    started = time.monotonic()
    for start in range(0, len(data['points']), config['batch_size']):
        x = torch.from_numpy(data['points'][start:start+config['batch_size']]).to(config['device'])
        output = model(x)
        if output.shape != (len(x), 256) or not torch.isfinite(output).all():
            raise ValueError(f'Invalid frozen encoder output: {directory}: {output.shape}')
        features.append(output.cpu().numpy())
        if start == 0:
            # Check accelerated weight conversion against the native e3nn model
            # on actual inputs, independently of the conversion's tensor layout.
            from src.models.encoders.pretrained_mace import PretrainedMACEEncoder
            native = PretrainedMACEEncoder(str(checkpoint), accelerated=False).to(config['device']).eval()
            raw = native.raw_features(x[:3]*radius, torch.zeros(3, dtype=torch.long, device=x.device))
            torch.testing.assert_close(raw, output[:3], rtol=2e-3, atol=2e-5)
            parity = dict(relative_l2=float((raw-output[:3]).norm()/raw.norm()),
                          max_abs=float((raw-output[:3]).abs().max()))
            del native
        if start % (config['batch_size']*40) == 0:
            print('ENCODE', directory.name, start, len(data['points']), flush=True)
    if state_digest(model) != before:
        raise ValueError('Frozen encoder state changed during inference')
    result = np.concatenate(features)
    np.savez(directory/'features.npz', encoder=result)
    write_json(directory/'inference.json', dict(state_sha256=before, frozen_state_unchanged=True,
        native_accelerated_parity=parity, performance=performance, seconds=time.monotonic()-started,
        shape=list(result.shape), features_sha256=sha256(directory/'features.npz')))
    del model
    torch.cuda.empty_cache()
    return result


def evaluate_frozen(features, data, config, directory, method, seed):
    train, val, test = [data['indices'][name] for name in ('train','val','test')]
    target, scales = data['targets'], data['scaling']['block_scale']
    rows = []
    predictions = ridge_path(features[train],target[train],features[np.concatenate([val,test])],config['ridge_alphas'])
    independent = ridge_predict(features[train],target[train],features[np.concatenate([val,test])],1.)
    np.testing.assert_allclose(predictions[1.],independent,rtol=1e-7,atol=1e-9)
    for alpha, prediction in predictions.items():
        for split, ids, output in [('val',val,prediction[:len(val)]),('test',test,prediction[len(val):])]:
            errors, blocks = balanced_errors(output, target[ids], scales)
            rows.append(dict(method=method, seed=seed, alpha=alpha, split=split,
                mse=float(errors.mean()), **dict(zip(['H0','H1','H2'],blocks.mean(axis=0).tolist()))))
    # Selection sees validation scores only; the test never chooses alpha.
    best = min((r for r in rows if r['split']=='val'), key=lambda r:r['mse'])['alpha']
    primary = predictions[1.][len(val):]
    selected = predictions[best][len(val):]
    errors = dict(test_sources=data['sources'][test], test_indices=test, targets=target[test],
        alpha1_predictions=primary, selected_predictions=selected,
        alpha1_errors=balanced_errors(primary,target[test],scales)[0],
        selected_errors=balanced_errors(selected,target[test],scales)[0])
    shuffled = target[train][np.random.default_rng(config['seed']).permutation(len(train))]
    control = ridge_predict(features[train], shuffled, features[test])
    control_mse = float(balanced_errors(control,target[test],scales)[0].mean())
    std = features[train].astype(np.float64).std(axis=0)
    standardized = (features[train].astype(np.float64)-features[train].astype(np.float64).mean(axis=0))/np.where(std==0,1,std)
    singular = np.linalg.svd(standardized,compute_uv=False)
    info = dict(method=method, seed=seed, selected_alpha=best,
                shuffled_train_mse=control_mse, constant_channels=int((std==0).sum()),
                svd_normal_equation_max_abs_difference=float(np.abs(predictions[1.]-independent).max()),
                feature_participation_rank=float(singular.dot(singular)**2/np.sum(singular**4)))
    write_json(directory/'scores.json',rows)
    write_json(directory/'readout.json',info)
    np.savez(directory/'predictions.npz',**errors)
    print('RESULT',info, 'alpha1',float(errors['alpha1_errors'].mean()),
          'selected',float(errors['selected_errors'].mean()),flush=True)
    return rows, info, errors


def summarize_initialization(root, config, results, mean_mse):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = [row for result in results for row in result[0]]
    snapshot_metric_docs(root, 'mace_tda_ridge_audit')
    with (root/'tables/scores.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    methods=['random','mlip','vicreg','tda']
    grouped={method:[result for result in results if result[1]['method']==method] for method in methods}
    summary={method:dict(
        seeds=[r[1]['seed'] for r in group],
        alpha1_mse=float(np.mean([r[2]['alpha1_errors'].mean() for r in group])),
        selected_mse=float(np.mean([r[2]['selected_errors'].mean() for r in group])),
        selected_alphas=[r[1]['selected_alpha'] for r in group],
        shuffled_train_mse=float(np.mean([r[1]['shuffled_train_mse'] for r in group])))
        for method,group in grouped.items()}
    intervals={}
    for before,after in [('random','mlip'),('mlip','vicreg'),('mlip','tda'),('vicreg','tda')]:
        for metric in ['alpha1_errors','selected_errors']:
            intervals[f'{before}_to_{after}_{metric}']=paired_interval(
                np.stack([r[2][metric] for r in grouped[before]]),
                np.stack([r[2][metric] for r in grouped[after]]),results[0][2]['test_sources'],
                seed=config['seed'],draws=config['bootstrap_draws'])
    write_json(root/'technical/summary.json',dict(groups=summary,comparisons=intervals,training_mean_mse=mean_mse))
    labels=['Random\n3 seeds','MLIP only\n1 checkpoint','MLIP + VICReg\n3 seeds','MLIP + VICReg + TDA\n3 seeds']
    colors=['#9b6a41','#2276a8','#27816c','#7958a2']
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),constrained_layout=True)
    for ax,metric,title in zip(axes,['alpha1_errors','selected_errors'],['Original ridge: alpha = 1','Ridge alpha chosen on validation']):
        for i,method in enumerate(methods):
            values=[r[2][metric].mean() for r in grouped[method]]
            ax.bar(i,np.mean(values),color=colors[i],width=.64,alpha=.8)
            ax.scatter(np.linspace(i-.09,i+.09,len(values)),values,s=28,color='black',zorder=3)
            ax.text(i,max(values)*1.025,f'{np.mean(values):.5f}',ha='center',fontsize=10)
        ax.set_xticks(range(4),labels,fontsize=9);ax.set_title(title)
        ax.set_ylabel('Balanced topology MSE (lower is better)');ax.grid(axis='y',alpha=.2)
        ax.set_ylim(0,max(r[2][metric].mean() for r in results)*1.18)
    fig.suptitle('Where is topology information available? Same 256D features, same held-out simulations',fontsize=13)
    fig.savefig(root/'plots/initialization-comparison.png',dpi=180)
    fig.savefig(root/'plots/initialization-comparison.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),constrained_layout=True)
    for ax,block in zip(axes,['H0','H1','H2']):
        for i,method in enumerate(methods):
            values=[r[block] for r in rows if r['method']==method and r['split']=='test' and r['alpha']==1.]
            ax.bar(i,np.mean(values),color=colors[i],alpha=.8)
            ax.scatter(np.linspace(i-.09,i+.09,len(values)),values,color='black',s=20,zorder=3)
        ax.set_title(block);ax.set_xticks(range(4),['Random','MLIP','VICReg','+TDA'],fontsize=9)
        ax.set_ylabel('Block-normalized MSE');ax.grid(axis='y',alpha=.2)
    fig.suptitle('Topology blocks, fixed ridge alpha = 1')
    fig.savefig(root/'plots/topology-blocks.png',dpi=180);fig.savefig(root/'plots/topology-blocks.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),constrained_layout=True)
    for ax,split in zip(axes,['val','test']):
        for method,color,label in zip(methods,colors,['Random','MLIP only','MLIP + VICReg','MLIP + VICReg + TDA']):
            scores=[np.mean([r['mse'] for r in rows if r['method']==method and r['split']==split and r['alpha']==alpha])
                    for alpha in config['ridge_alphas']]
            ax.plot(config['ridge_alphas'],scores,marker='.',label=label,color=color)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Ridge penalty alpha')
        ax.set_ylabel('Balanced topology MSE');ax.set_title('Validation: selects alpha' if split=='val' else 'Test: evaluation only')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Regularization sensitivity of frozen encoder readouts')
    fig.savefig(root/'plots/ridge-regularization.png',dpi=180);fig.savefig(root/'plots/ridge-regularization.pdf');plt.close(fig)
    lines=['# Frozen MACE initialization controls','',
        'All models use the same 256D raw encoder output, 23,040 neighborhoods and original 18/6/6 source split. Ridge alone fits training topology labels; encoders remain frozen.',
        '', '| Encoder | Fixed alpha 1 MSE | Validation-selected MSE | Selected alpha(s) |',
        '|---|---:|---:|---|']
    for method,value in summary.items():
        lines.append(f"| {method} | {value['alpha1_mse']:.9f} | {value['selected_mse']:.9f} | {value['selected_alphas']} |")
    lines += ['',f'Training-mean baseline: {mean_mse:.9f}. Random/VICReg/TDA average three initializations; MLIP is one fixed checkpoint.',
        '', '![Initialization comparison](plots/initialization-comparison.png)',
        '', '![Topology blocks](plots/topology-blocks.png)',
        '', 'Random weights come from the native constructor with no MLIP parameter transfer. Shared architecture includes the geometric basis, cutoff and neighbor-count normalization. Energy offsets are neutralized and energy readouts are unused.',
        '', 'Test-source bootstrap comparisons and per-seed readouts are in technical/summary.json and tables/scores.csv. This previously examined cohort is exploratory; these are descriptor reconstruction errors, not crystallization classification accuracies.',
        '', 'Reproduce: `python -m src.research.mace_tda_ridge_audit.run --config configs/analysis/mace_tda_initialization.json --stage initialization` using a fresh output directory.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')


def run_initialization(config):
    root=Path(config['output'])
    for name in ['technical','tables','plots']:(root/name).mkdir(parents=True,exist_ok=True)
    if (root/'technical/summary.json').exists():
        raise FileExistsError(f'Preserve completed initialization results; choose a fresh output: {root}')
    torch.set_num_threads(config['cpu_threads']);torch.set_float32_matmul_precision('highest')
    torch.cuda.set_device(config['device'])
    write_json(root/'technical/config.json',config)
    write_json(root/'technical/status.json',dict(state='running',stage='data-audit'))
    data=data_audit(config,root)
    reference=load_recipe(config['reference_recipe']);prior=Path(reference['output'])
    if read_json(root/'technical/data-audit.json') != read_json(prior/'technical/data-audit.json'):
        raise ValueError('Initialization controls do not share the audited data protocol')
    manifest=read_json(Path(config['cache'])/'manifest.json')
    radius=manifest['protocol']['radius_A']
    original=Path(config['pretrained_checkpoint'])
    first_cfg=OmegaConf.load(Path(reference['runs'][0]['checkpoint']).parent/'.hydra/config.yaml')
    config['performance']=OmegaConf.to_container(first_cfg.encoder.kwargs.performance,resolve=True)
    write_json(root/'technical/config.json',config)
    source=torch.load(original,map_location='cpu',weights_only=False).float().eval()
    provenance=dict(mlip_checkpoint=str(original),mlip_sha256=sha256(original),reference_runs=[])
    results=[]
    for method,seed in [('mlip',None)]+[('random',s) for s in config['random_seeds']]:
        directory=root/'technical'/f'{method}-{seed}';directory.mkdir(exist_ok=True)
        write_json(root/'technical/status.json',dict(state='running',stage='encode',current=directory.name))
        checkpoint=original
        if method=='random':
            native,audit=random_native(source,seed)
            checkpoint=directory/'random.model';torch.save(native,checkpoint)
            audit['state_sha256']=state_digest(native)
            audit['checkpoint_sha256']=sha256(checkpoint)
            for key in ['constructor_file','config_extractor_file']:
                audit[key+'_sha256']=sha256(Path(audit[key]))
            write_json(directory/'initialization.json',audit)
            del native
        features=extract_frozen(checkpoint,data,config,radius,directory)
        results.append(evaluate_frozen(features,data,config,directory,method,seed))
    for item in reference['runs']:
        directory=root/'technical'/item['name'];directory.mkdir(exist_ok=True)
        old=prior/'technical'/item['name']
        cfg=OmegaConf.load(Path(item['checkpoint']).parent/'.hydra/config.yaml')
        if OmegaConf.to_container(cfg.encoder.kwargs.performance,resolve=True) != config['performance']:
            raise ValueError(f'Reference encoder numerical settings differ: {item["name"]}')
        if sha256(resolve_path(cfg.encoder.kwargs.pretrained_checkpoint)) != provenance['mlip_sha256']:
            raise ValueError(f'Trained comparator used a different MLIP initialization: {item["name"]}')
        if cfg.encoder.kwargs.reference_radius_A != radius:
            raise ValueError('Reference encoder coordinate scaling differs')
        features=np.load(old/'features.npz')['encoder']
        if features.shape != (len(data['points']),256) or not np.isfinite(features).all():
            raise ValueError(f'Invalid retained encoder features: {old}')
        result=evaluate_frozen(features,data,config,directory,item['method'],item['seed'])
        np.testing.assert_allclose(result[2]['alpha1_errors'],np.load(old/'errors.npz')['encoder_test'],rtol=1e-8,atol=1e-10)
        results.append(result)
        provenance['reference_runs'].append(dict(name=item['name'],features=str(old/'features.npz'),
            features_sha256=sha256(old/'features.npz'),checkpoint=read_json(old/'checkpoint-audit.json')))
    write_json(root/'technical/provenance.json',provenance)
    ids=data['indices'];target=data['targets']
    mean_mse=float(balanced_errors(np.broadcast_to(target[ids['train']].mean(axis=0),target[ids['test']].shape),
        target[ids['test']],data['scaling']['block_scale'])[0].mean())
    summarize_initialization(root,config,results,mean_mse)
    contract=read_json(root/'technical/metric-contract.json')
    for relative in contract['files']:
        destination=root/'technical/source'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(relative,destination)
    write_json(root/'technical/status.json',dict(state='complete',encoders=len(results)))
