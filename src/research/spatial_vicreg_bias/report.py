"""Publish fixed-endpoint trajectories; never choose a winning checkpoint."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_runner.metric_docs import write_metric_table
from src.research.structural_state.common import write_json
from .data import load


def run(config):
    c=load(config);root=Path(c['output']);dest=root/'comparison'
    for part in ('plots','tables','technical'):(dest/part).mkdir(parents=True,exist_ok=True)
    result={};coverage={};fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for seed in c['training']['seed_values']:
        for alpha in c['training']['alphas']:
            name=f'S{alpha:g}-seed{seed}';result[name]={};coverage[name]={}
            epochs=[];series=[[],[],[],[]]
            for epoch in c['assay']['epochs']:
                p=root/name/'analyses'/f'epoch-{epoch:02d}'/'technical/complete.json'
                coverage[name][str(epoch)]=p.exists()
                if not p.exists():continue
                r=json.loads(p.read_text())['metrics']['encoder']['k7'];epochs.append(epoch)
                block=r['readouts']['cohorts']
                values=dict(clear_tda_cluster_skill=block['clear_input']['tracks']['uniform']['test']['models']['cluster_means']['tda']['train_mean_skill'],
                    clear_tda_conditional_gain=block['clear_input']['tracks']['uniform']['test']['conditional_cluster_gain']['tda']['delta_normalized_mse'],
                    clear_effective_rank=r['spatial']['clear_input']['effective_rank'],
                    neighbor_distance=r['spatial']['all']['neighbor_squared_distance'])
                result[name][str(epoch)]=values
                for i,v in enumerate(values.values()):series[i].append(v)
            for ax,values in zip(axes.flat,series):
                ax.plot(epochs,values,marker='.',alpha=.7,label=name,color={0.:'C0',.5:'C1',1.:'C2'}[alpha])
    titles=['Crystal-free input: cluster → TDA skill','Crystal-free input: gain beyond phase/density',
            'Crystal-free input: effective rank','Neighbor squared distance / train variance']
    for ax,title in zip(axes.flat,titles):ax.set(title=title,xlabel='Completed full passes');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(dest/'plots'/f'encoder-trajectories.{ext}',dpi=180)
    plt.close(fig)
    write_metric_table(result,dest,family='spatial_vicreg_bias',name='fixed-epoch-comparison')
    write_json(dest/'technical/coverage.json',coverage);write_json(dest/'technical/summary.json',result)
    complete=all(all(v.values()) for v in coverage.values())
    (dest/'README.md').write_text('# Spatial-neighbor VICReg mechanism study\n\n'
        +('All planned encoder assays are complete.\n' if complete else 'Partial results: consult technical/coverage.json for missing assays.\n')
        +'\nThe fixed endpoints are epochs 12 and 24. Curves display every paired seed; '
        'they do not select a checkpoint. Training used geometry only, no future labels. '
        'Cluster information is evaluated on fixed held-out source roles. The strict PTM-free subset '
        'is an evaluation stratum, never an encoder training filter. Full per-feature, per-source errors '
        'and conditional-gain intervals are retained in each checkpoint analysis.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);run(p.parse_args().config)
