"""Read-only scale, noise and competing-gradient audit of frozen local80 fits."""
import argparse
import math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read, sha, save, write_json

FAMILY = 'local_response_audit'
PREFIXES = [('20fs', 0, 128), ('100fs_joint', 128, 256), ('full', 0, 256)]


def rms(x):
    return float(x.double().square().mean().sqrt())


def audit_data(c, run, out, data):
    train = [r for r in data['parents'] if r['role'] == 'train']
    v = torch.stack([r['basis'] for r in train]).double()
    lengths = v[:, 1:].norm(dim=2)
    geometry = []
    for eps in c['epsilons_A']:
        shifts = eps * lengths
        geometry.append(dict(epsilon_A=eps, moving_atoms=79, collective_norm_A=eps,
            atom_vector_rms_A=rms(shifts), coordinate_rms_A=eps/math.sqrt(79*3),
            atom_vector_median_A=float(shifts.median()), atom_vector_p95_A=float(torch.quantile(shifts,.95)),
            atom_vector_max_A=float(shifts.max()), displacement_relative_to_2p8A=100*rms(shifts)/2.8))
    write_metric_rows(geometry, out, family=FAMILY, name='disturbance')
    basis_norm=v.flatten(1,2).norm(dim=1)
    write_json(out/'technical/basis.json', dict(norm_min=float(basis_norm.min()),norm_max=float(basis_norm.max()),
        center_max=float(v[:,0].abs().max()),center_of_mass_sum_max=float(v[:,1:].sum(1).abs().max()),
        cross_direction_dot_max=float((v[...,0]*v[...,1]).sum((1,2)).abs().max())))
    scale = data['scale'].double(); center = data['center'].double()
    response_scale = float(data['response_scale'])
    y = torch.stack([r['values'][:8] for r in train]).double()
    h = torch.stack([r['responses'] for r in train]).double()
    scales, shifts, noises = [], [], []
    for prefix,lo,hi in PREFIXES:
        raw_h=h[:,:,lo:hi];norm_h=raw_h/scale[None,None,lo:hi,None]
        norm_y=(y[:,:,lo:hi]-center[None,None,lo:hi])/scale[None,None,lo:hi]
        scales.append(dict(prefix=prefix,parents=len(train),shots=8,
            raw_value_rms=rms(y[:,:,lo:hi]),feature_scale_min=float(scale[lo:hi].min()),
            feature_scale_median=float(scale[lo:hi].median()),feature_scale_max=float(scale[lo:hi].max()),
            raw_branch_response_rms_per_A=rms(raw_h),raw_mean_response_rms_per_A=rms(raw_h.mean(1)),
            standardized_mean_response_rms_per_A=rms(norm_h.mean(1)),global_response_scale_per_A=response_scale,
            normalized_target_value_half_mse=.5*rms(norm_y.mean(1))**2,
            normalized_target_response_half_mse=.5*rms(norm_h.mean(1)/response_scale)**2))
        for eps in c['epsilons_A']:
            shifts.append(dict(prefix=prefix,epsilon_A=eps,
                raw_mean_feature_change_rms_linear=eps*rms(raw_h.mean(1)),
                standardized_mean_feature_change_rms_linear=eps*rms(norm_h.mean(1)),
                raw_mean_plus_minus_difference_rms_linear=2*eps*rms(raw_h.mean(1))))
        for label,x in [('value',norm_y),('response',norm_h)]:
            total=float(x.mean(1).square().mean());noise=float((x.var(1,unbiased=True)/x.shape[1]).mean())
            noises.append(dict(prefix=prefix,target=label,parents=len(train),shots=x.shape[1],
                mean_target_square=total,estimated_noise_variance_of_mean=noise,
                estimated_signal_square=total-noise,noise_fraction_of_mean_square=noise/total))
    for name,rows in [('response-scale',scales),('linearized-feature-change',shifts),('shot-noise',noises)]:
        write_metric_rows(rows,out,family=FAMILY,name=name)
    histories=[];training=[]
    for seed in [20261002,20261003,20261004]:
        for arm in ['values8','responses8','values32','values8_time']:
            folder=run/'analyses'/f'{arm}-seed-{seed}'
            history=pd.read_csv(folder/'tables/learning.csv');done=read(folder/'technical/complete.json')
            hist=history.assign(arm=arm,seed=seed)
            histories.append(hist)
            best=history[history.epoch==done['selected_epoch']].iloc[0]
            training.append(dict(arm=arm,seed=seed,selected_epoch=done['selected_epoch'],updates=len(history),
                max_gradient_norm=float(history.gradient_norm.max()),clipped_updates=int((history.gradient_norm>10).sum()),
                first_value_half_mse=float(history.iloc[0].train_value_nll-.5*math.log(2*math.pi)),
                first_response_half_mse=float(history.iloc[0].train_response_half_mse),
                best_epoch_train_value_half_mse=float(best.train_value_nll-.5*math.log(2*math.pi)),
                best_epoch_train_response_half_mse=float(best.train_response_half_mse),
                selection_nll=done['selection_nll']))
    write_metric_rows(training,out,family=FAMILY,name='training-audit')
    return pd.concat(histories,ignore_index=True)


def gradients(c,run,out,data,config):
    from src.research.local_response.model import Predictor, initialize, responses
    torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32=False
    records=[r for r in data['parents'] if r['role']=='train']
    device=c['device'];micro=c['microbatch']
    q=torch.stack([r['q'] for r in records]).to(device)
    basis=torch.stack([r['basis'] for r in records]).to(device)
    target=torch.stack([(r['values'][:8].mean(0)-data['center'])/data['scale'] for r in records]).to(device)
    htarget=torch.stack([r['responses'].mean(0)/data['scale'][:,None] for r in records]).to(device)
    hs=data['response_scale'].to(device)
    results=[]
    parameter_counts=None
    for seed in config['fit_seeds']:
        # Initialization and both endpoints of the response arm expose the
        # actual competition, without optimizing or choosing a new checkpoint.
        for arm,checkpoint in [('responses8','initial'),('responses8','best'),('responses8','last'),('values8','best')]:
            cache=out/'technical'/f'gradient-{arm}-{seed}-{checkpoint}.pt'
            if cache.exists():
                results.extend(torch.load(cache,weights_only=False));continue
            torch.manual_seed(seed)
            model=Predictor(config).to(device)
            checkpoint_hash='reconstructed_from_recorded_seed'
            if checkpoint=='initial':
                initialize(model,q,micro);epoch=0
            else:
                file=run/'analyses'/f'{arm}-seed-{seed}'/'technical'/f'{checkpoint}.pt'
                checkpoint_hash=sha(file)
                p=torch.load(file,map_location=device,weights_only=False)
                if checkpoint=='best':
                    model.load_state_dict(p['model']);epoch=p['epoch']
                else:
                    model.load_state_dict(p['model']);epoch=p['epoch']
            model.eval()
            named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
            parameters=[p for _,p in named]
            parameter_counts=dict(total=sum(p.numel() for p in parameters),
                encoder=sum(p.numel() for name,p in named if name.startswith('encoder.')),
                head=sum(p.numel() for name,p in named if name.startswith('head.')))
            gv=[torch.zeros_like(p) for p in parameters];gh=[torch.zeros_like(p) for p in parameters]
            lv_sum=lh_sum=0.
            for begin in range(0,len(q),micro):
                sl=slice(begin,begin+micro);fraction=len(q[sl])/len(q)
                pred=model(q[sl]);lv=.5*(pred-target[sl]).square().mean()*fraction
                value_grad=torch.autograd.grad(lv,parameters,allow_unused=True)
                for accum,g in zip(gv,value_grad):
                    if g is not None:accum.add_(g.detach())
                response=responses(model,q[sl],basis[sl],create_graph=True)
                lh=.5*((response-htarget[sl])/hs).square().mean()*fraction
                response_grad=torch.autograd.grad(lh,parameters,allow_unused=True)
                for accum,g in zip(gh,response_grad):
                    if g is not None:accum.add_(g.detach())
                lv_sum+=float(lv.detach());lh_sum+=float(lh.detach())
            stage=[]
            for group in ['all','encoder','head']:
                chosen=[i for i,(name,p) in enumerate(named) if group=='all' or name.startswith(group+'.')]
                a=torch.cat([gv[i].flatten().double() for i in chosen]);b=torch.cat([gh[i].flatten().double() for i in chosen])
                an=float(a.norm());bn=float(b.norm())
                stage.append(dict(arm=arm,seed=seed,checkpoint=checkpoint,epoch=epoch,group=group,
                    value_half_mse=lv_sum,response_half_mse=lh_sum,value_gradient_norm=an,
                    response_gradient_norm=bn,response_to_value_gradient_ratio=bn/max(an,1e-30),
                    gradient_cosine=float(a@b/max(an*bn,1e-30)),combined_gradient_norm=float((a+b).norm()),
                    checkpoint_sha256=checkpoint_hash))
            save(cache,stage);results.extend(stage)
            print(stage[0],flush=True)
            del model,gv,gh,a,b,response,parameters,named
            torch.cuda.empty_cache()
    write_metric_rows(results,out,family=FAMILY,name='gradient-balance')
    if parameter_counts:write_json(out/'technical/parameter-counts.json',parameter_counts)
    return pd.DataFrame(results)


def plot(out,history,gradient):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True)
    for ax,seed in zip(axes,[20261002,20261003,20261004]):
        for arm,color,label in [('values8','#367d9e','Values only'),('responses8','#bf623f','Values + responses')]:
            d=history[(history.seed==seed)&(history.arm==arm)]
            ax.plot(d.epoch,d.selection_nll,color=color,label=label,lw=1.5)
            best=d.loc[d.selection_nll.idxmin()]
            ax.scatter(best.epoch,best.selection_nll,color=color,marker='*',s=130,zorder=4)
            ax.annotate(f'selected {int(best.epoch)}',(best.epoch,best.selection_nll),
                        xytext=(4,10 if arm=='values8' else -18),textcoords='offset points',fontsize=9,color=color)
        ax.set(title=f'Training seed {seed}',xlabel='Optimizer updates (= epochs)')
    axes[0].set_ylabel('Validation feature NLL — lower is better');axes[0].legend(frameon=False)
    fig.suptitle('Early stopping selects two response models before their later recovery can be assessed')
    fig.tight_layout(rect=(0,0,1,.93))
    plots=out/'plots';plots.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(plots/f'learning-curves.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    d=gradient[(gradient.group=='all')&(gradient.arm=='responses8')]
    names=['initial','best','last'];x=np.arange(3)
    for seed,offset,color in zip([20261002,20261003,20261004],[-.16,0,.16],['#367d9e','#bf623f','#428c65']):
        r=d[d.seed==seed].set_index('checkpoint').loc[names]
        axes[0].plot(x+offset,r.response_to_value_gradient_ratio,'o-',color=color,label=str(seed))
        axes[1].plot(x+offset,r.gradient_cosine,'o-',color=color)
    axes[0].axhline(1,color='grey',ls='--');axes[1].axhline(0,color='grey',ls='--')
    axes[0].set(ylabel='Response / value parameter-gradient norm',title='Weight1 does not mean equal training influence')
    axes[1].set(ylabel='Cosine of the two parameter gradients',title='Negative cosine means competing update directions',ylim=(-1.05,1.05))
    for ax in axes:ax.set_xticks(x,names)
    axes[0].legend(frameon=False)
    fig.suptitle('Full90-parent gradients at frozen checkpoints; no optimization performed')
    fig.tight_layout(rect=(0,0,1,.93))
    for ext in ['png','pdf']:fig.savefig(plots/f'gradient-balance.{ext}',dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    args=p.parse_args();c=read(args.config);run=resolve_path(c['run'])
    binding=read(run/'technical/binding.json')
    if binding['identity']!=c['identity']:raise ValueError('Unexpected experiment identity')
    for name,expected in binding['implementation'].items():
        if sha(Path(name))!=expected:raise ValueError(f'Bound scientific producer changed: {name}')
    receipt=read(run/'technical/data-complete.json')
    if sha(receipt['cache'])!=receipt['sha256']:raise ValueError('Label bank changed')
    data=torch.load(receipt['cache'],map_location='cpu',weights_only=False)
    out=run/'analyses'/c['analysis'];out.mkdir(parents=True,exist_ok=True)
    write_json(out/'technical/provenance.json',dict(config=c,binding_identity=binding['identity'],
        training_bank_sha256=receipt['sha256'],implementation_sha256=sha(Path(__file__)),
        purpose='Read-only diagnostics; train ancestors only for scale/gradient calculations; no fits or new W&B runs'))
    torch.set_num_threads(2)
    history=audit_data(c,run,out,data)
    gradient=gradients(c,run,out,data,binding['config'])
    plot(out,history,gradient)
    print('Audit complete:',out,flush=True)


if __name__=='__main__':main()
