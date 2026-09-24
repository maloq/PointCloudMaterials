"""Paired source-level endpoint review; no checkpoint-selected significance claims."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_runner.metric_docs import write_metric_table
from .reference import write_json


def paired_mean_ci(delta, seed=20260923):
    delta=np.asarray(delta,dtype=float)
    if delta.shape != (15,) or not np.isfinite(delta).all():
        raise ValueError(f'Require fifteen paired finite development-source effects: {delta.shape}')
    rng=np.random.default_rng(seed)
    draws=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1)
    return dict(mean=float(delta.mean()),ci95=np.quantile(draws,[.025,.975]).tolist(),sources=15)


def run(source, output):
    source=Path(source);root=Path(output)
    for part in ('plots','tables','technical'):(root/part).mkdir(parents=True,exist_ok=True)
    names=['epoch-011','epoch-034']; metrics={}
    for name in names:
        metrics[name]=json.loads((source/'technical/evaluations'/name/'metrics.json').read_text())
    results={}
    for name in names:
        folder=source/'technical/evaluations'/name
        base=np.load(folder/'current-physics-future-predictions.npz')
        source_ids=base['hazard_source'];event=base['hazard_event']<5
        roots=np.unique(source_ids)
        if len(roots)!=15:raise ValueError('Onset assay no longer has 15 development sources.')
        brier0=np.square(base['hazard_risks'][:,-1]-event)
        for rep in ('encoder','projector'):
            p=metrics[name]['prediction_'+rep]
            mse=p['future_residual_9ps']['per_source'];constant=p['residual_constant']['per_source']
            if set(mse)!=set(constant):raise ValueError('Future regression sources differ.')
            mse_delta=[mse[k]-constant[k] for k in sorted(mse)]
            a=np.load(folder/f'{rep}-future-predictions.npz')
            for key in ('hazard_indices','hazard_source','hazard_event'):
                np.testing.assert_array_equal(a[key],base[key])
            brier=np.square(a['hazard_risks'][:,-1]-event)
            brier_delta=[float((brier[source_ids==s]-brier0[source_ids==s]).mean()) for s in roots]
            results[name+'_'+rep]=dict(future_mse_minus_current_baseline=paired_mean_ci(mse_delta),
                onset_brier_minus_current_physics=paired_mean_ci(brier_delta),
                onset_ap=p['conditional_hazard']['horizons']['12.0']['average_precision'],
                selected_hazard_step=p['conditional_hazard']['best_step'])
    # Confusion-style display of unsupervised groups versus independent contexts.
    labels=['Liquid other','Crystal interior','Mixed boundary','Al planar fault','Internal defect','Five-fold proxy','Ordered liquid']
    for name in ['archived-epoch34',*names]:
        data=json.loads((source/'technical/evaluations'/name/'metrics.json').read_text())
        fig,axes=plt.subplots(1,3,figsize=(16,6),constrained_layout=True)
        for ax,index,material in zip(axes,[2,4,7],['Al','Ta','Zr']):
            count=np.array(data[f'frame_{index:02d}_{material}_projector']['cluster_context_counts'])
            total=count.sum(0)
            frac=np.divide(count,total[None],out=np.full(count.shape,np.nan),where=total[None]>0)
            im=ax.imshow(frac,vmin=0,vmax=1,cmap='Blues',aspect='auto')
            ax.set(xticks=range(7),xticklabels=[f'{s}\nn={n}' for s,n in zip(labels,total)],
                   yticks=range(7),yticklabels=[f'C{k+1}' for k in range(7)],title=material,
                   ylabel='Unsupervised embedding cluster')
            ax.tick_params(axis='x',rotation=60)
            for i,j in zip(*np.where(np.isfinite(frac))):
                if frac[i,j]>.04:ax.text(j,i,f'{frac[i,j]:.2f}',ha='center',va='center',color='white' if frac[i,j]>.65 else 'black',fontsize=8)
        fig.colorbar(im,ax=axes,shrink=.5,label='Fraction of reference class assigned to cluster')
        fig.suptitle(f'{name}: can K=7 separate interfaces, faults and different liquid environments?\nColumn-normalized held-out counts; white missing columns have no test examples. Cluster IDs are arbitrary.')
        fig.savefig(root/'plots'/f'{name}-cluster-context.png',dpi=160);plt.close(fig)
    write_json(root/'technical/endpoint-comparisons.json',results)
    write_json(root/'technical/inputs.json',{str(source/'technical/evaluations'/n/'metrics.json'):
        hashlib.sha256((source/'technical/evaluations'/n/'metrics.json').read_bytes()).hexdigest() for n in names})
    write_metric_table(results,root,family='geoframe_evolution_review',name='endpoint-comparisons')
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.source,a.output)
