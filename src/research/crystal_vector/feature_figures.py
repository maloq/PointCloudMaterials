"""Render saved feature-dominance tables without recomputing scientific metrics."""
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.project_runtime.paths import resolve_path


def render(root):
    root=resolve_path(root);tables=root/'tables';plots=root/'plots';plots.mkdir(exist_ok=True)
    concentration=pd.read_csv(tables/'feature-concentration.csv')
    probes=pd.read_csv(tables/'probe-scores.csv')
    head=pd.read_csv(tables/'frozen-head-scores.csv')
    colors={'train':'#3275ad','test':'#cd6335'}
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for j,field in enumerate(('local_z','context_z')):
        ax=axes[0,j]
        for role in ('train','test'):
            part=concentration[(concentration.field==field)&(concentration.role==role)]
            ax.plot(part.top_k,part.training_PC_variance_share*100,'o-',label=role.title(),color=colors[role])
        ax.set(xscale='log',xticks=[1,2,4,8,16],xticklabels=['1','2','4','8','16'],ylim=(0,102),
            xlabel='Number of leading training PCs',ylabel='Variance explained (%)',title='Local embedding' if j==0 else 'Context state')
        ax.grid(alpha=.2);ax.legend()
        ax=axes[1,j];designs=['all_PC','top_PC_1','top_PC_2','top_PC_8','drop_PC_1']
        for off,role in enumerate(('train','test')):
            part=probes[(probes.field==field)&(probes.role==role)&(probes.visibility=='all')&(probes.radius_A==20)].set_index('design')
            ax.bar(np.arange(len(designs))+(off-.5)*.36,[part.loc[d,'log_loss'] for d in designs],.36,label=role.title(),color=colors[role])
        ax.set(xticks=np.arange(len(designs)),xticklabels=['All PCs','Top 1','Top 2','Top 8','Remove 1'],ylabel='20 Å proximity log loss ↓',xlabel='Refitted linear readout input')
        ax.grid(axis='y',alpha=.2);ax.legend()
    fig.suptitle('Strong feature concentration, similar train/test behavior',fontsize=15)
    fig.tight_layout();fig.savefig(plots/'feature-concentration-and-readouts.png',dpi=180);plt.close(fig)

    designs=['original','scalar_clip_3sd','vectors_zero','scalar_top_PC_1','scalar_top_PC_8','scalar_drop_PC_1','scalar_mean']
    labels=['Original','Clip scalar tails','Zero vectors','Keep scalar PC1','Keep scalar PCs 1–8','Remove scalar PC1','Mean scalars']
    fig,ax=plt.subplots(figsize=(11,5))
    for off,role in enumerate(('train','test')):
        part=head[(head.role==role)&(head.visibility=='all')&(head.radius_A==20)].set_index('treatment')
        ax.barh(np.arange(len(designs))+(off-.5)*.36,[part.loc[d,'predictive_objective'] for d in designs],.36,label=role.title(),color=colors[role])
    ax.set(yticks=np.arange(len(designs)),yticklabels=labels,xlabel='Predictive objective ↓ (frozen head; no refit)',title='Reliance on the leading scalar direction transfers to held-out sources')
    ax.invert_yaxis();ax.grid(axis='x',alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(plots/'frozen-predictor-interventions.png',dpi=180);plt.close(fig)
    (root/'technical/figures.json').write_text(json.dumps(dict(source='saved CSV tables only',figures=[p.name for p in plots.glob('*.png')]),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);a=p.parse_args();render(a.root)
