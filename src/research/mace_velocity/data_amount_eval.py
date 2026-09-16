"""Source-held-out physical and motion scores for native encoder learning curves."""
import csv
from pathlib import Path

import numpy as np
import torch

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.mace_local_state.motion import projection_residual, time_differences
from .data_amount_data import batch
from .data_amount_model import normalized_targets, objective
from .train import GROUPS, encode


def extract(config, model, heads, directions, sources, ids, norm, device):
    clouds,raw,times=batch(sources,ids)
    model.eval(); heads.eval(); directions.eval()
    z=encode(config,model,clouds,device)
    y=normalized_targets(raw,norm,device)
    tt=torch.as_tensor(times,dtype=torch.float64,device=device)
    with torch.no_grad():
        _,loss=objective(config,heads,directions,z,y,tt,1.)
        predicted=heads(z).reshape(len(ids),4,9,169)
        state=z[:,:256].reshape(-1,9,256)
        delta,_,_,bend=time_differences(state,tt)
        basis=directions(state[:,:-1].reshape(-1,256))
        residual=projection_residual(delta.flatten(0,1),basis).reshape(len(ids),4,8)
    return dict(z=z.cpu().numpy().reshape(len(ids),4,9,304),
        error=(predicted-y.reshape(len(ids),4,9,169)).square().cpu().numpy(),
        raw_target=raw.reshape(len(ids),4,9,169),time_ps=times.reshape(len(ids),4,9),
        direction_residual=residual.cpu().numpy(),
        bend_squared=bend.square().sum(-1).cpu().numpy().reshape(len(ids),4,7),
        source_ids=np.asarray(ids),loss=loss)


def reference_trace(z, mask, within_context=False):
    # Inputs source x center x time x channel; contexts are source/time.
    if within_context:
        values=z.transpose(0,2,1,3).reshape(-1,4,z.shape[-1])
        eligible=mask.transpose(0,2,1).reshape(-1,4)
        variances=[v[m].var(0).sum() for v,m in zip(values,eligible,strict=True) if m.sum()>=2]
        if not variances: raise ValueError('No eligible core contexts with two local groups')
        result=float(np.mean(variances))
    else:
        if mask.sum()<2: raise ValueError('Insufficient reference population')
        result=float(z[mask].var(0).sum())
    if not np.isfinite(result) or result<=1e-12: raise ValueError(f'Collapsed core reference: {result}')
    return result


def summarize(config, result, reference, fit, split):
    rows=[]; source_rows=[]
    z=result['z']; dz=np.diff(z[...,:256],axis=2); d2=(dz**2).sum(-1)
    full2=(np.diff(z,axis=2)**2).sum(-1)
    for population in ('all','low_order'):
        low=population=='low_order'
        mask=result['raw_target'][...,4]<config['low_order_threshold'] if low else np.ones(z.shape[:3],bool)
        rmask=reference['raw_target'][...,4]<config['low_order_threshold'] if low else np.ones(reference['z'].shape[:3],bool)
        trace=reference_trace(reference['z'][...,:256],rmask)
        within=reference_trace(reference['z'][...,:256],rmask,True)
        full_trace=reference_trace(reference['z'],rmask)
        edges=mask[:,:,1:] & mask[:,:,:-1]
        triples=mask[:,:,2:] & mask[:,:,1:-1] & mask[:,:,:-2]
        entries=[]
        for i,sid in enumerate(result['source_ids']):
            row=dict(seed=fit['seed'],training_sources=fit['count'],split=split,population=population,
                     source_id=int(sid),observations=int(mask[i].sum()),pairs=int(edges[i].sum()),
                     triples=int(triples[i].sum()))
            for name,section in GROUPS.items():
                row[name]=float(result['error'][i][mask[i]][:,section].mean()) if mask[i].any() else None
            row['increment_energy']=float(d2[i][edges[i]].mean()) if edges[i].any() else None
            row['direction_residual']=float(result['direction_residual'][i][edges[i]].mean()) if edges[i].any() else None
            row['full_increment_energy']=float(full2[i][edges[i]].mean()) if edges[i].any() else None
            row['bend_energy']=float(result['bend_squared'][i][triples[i]].mean()) if triples[i].any() else None
            entries.append(row); source_rows.append(row)
        def average(key):
            values=[r[key] for r in entries if r[key] is not None]
            if not values: raise ValueError(f'No eligible sources: {split}, {population}, {key}')
            return float(np.mean(values))
        energy=average('increment_energy')
        jumps=np.sqrt(d2[edges]/(2*trace))
        source_energy=np.array([r['increment_energy'] for r in entries if r['increment_energy'] is not None])
        rng=np.random.default_rng(config['core_seed'])
        draws=rng.integers(len(source_energy),size=(config['bootstrap_draws'],len(source_energy)))
        ci=np.quantile(np.sqrt(source_energy[draws].mean(1)/(2*trace)),[.025,.975])
        row=dict(seed=fit['seed'],training_sources=fit['count'],training_observations=36*fit['count'],
                 split=split,population=population,sources=len(entries),
                 eligible_sources=sum(r['pairs']>0 for r in entries),observations=int(mask.sum()),
                 pairs=int(edges.sum()),lag_ps=.75,**{k:average(k) for k in GROUPS},
                 jump_rms=float(np.sqrt(energy/(2*trace))),jump_p95=float(np.quantile(jumps,.95)),
                 jump_max=float(jumps.max()),jump_rms_ci95_lower=float(ci[0]),jump_rms_ci95_upper=float(ci[1]),
                 within_context_jump_rms=float(np.sqrt(energy/(2*within))),
                 full_embedding_jump_rms=float(np.sqrt(average('full_increment_energy')/(2*full_trace))),
                 bend_rms=float(np.sqrt(average('bend_energy')/(2*trace))),
                 learned_rank8_explained_energy=1-average('direction_residual')/max(energy,1e-12),
                 core_structure_trace=trace,core_within_context_trace=within)
        row['physical_mean_mse']=float(np.mean([row[k] for k in list(GROUPS)[:4]]))
        rows.append(row)
    return rows,source_rows


def export(config, rows, source_rows, root):
    root=Path(root); snapshot_metric_docs(root,'mace_data_amount')
    for name,values in [('quality',rows),('source_quality',source_rows)]:
        with (root/'tables'/f'{name}.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,7),constrained_layout=True)
    metrics=[('physical_mean_mse','Physical error'),('bond_order','Bond-order error'),
             ('instantaneous_TDA_H1','Instantaneous H1 topology error'),('jump_rms','0.75 ps RMS jump'),
             ('within_context_jump_rms','Jump / local within-context spread'),
             ('learned_rank8_explained_energy','Motion energy in 8 learned directions')]
    for ax,(metric,title) in zip(axes.flat,metrics,strict=True):
        for population,color in [('all','tab:blue'),('low_order','tab:orange')]:
            for seed in config['seeds']:
                selected=sorted((r for r in rows if r['split']=='development_test' and r['population']==population
                    and r['seed']==seed),key=lambda r:r['training_sources'])
                if selected:
                    ax.plot([r['training_sources'] for r in selected],[r[metric] for r in selected],'-o',color=color,
                            alpha=.7,label=population if seed==config['seeds'][0] else None)
        ax.set_title(title);ax.set_xlabel('Independent training trajectories');ax.grid(alpha=.2)
    axes[1,0].axhline(.1,color='black',ls=':',label='Requested jump target')
    axes[0,0].legend();fig.suptitle('End-to-end native MACE encoder: matched optimizer updates')
    fig.savefig(root/'plots/learning_curves.png',dpi=160);fig.savefig(root/'plots/learning_curves.pdf');plt.close(fig)
