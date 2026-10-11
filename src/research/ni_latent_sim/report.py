"""Publish existing scores and compact scientific figures, without refitting."""
import csv
import numpy as np
from .artifacts import write_rows as write_metric_rows, table_path, export_root
from .data import root,write_json,FAMILY


def run(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=export_root(root(c)/'analyses/comparison-v1');rows=[];birth=[]
    for arm in c['arms']:
        folder=root(c)/'analyses'/arm['name']
        for target,name in [(rows,'free_rollout'),(birth,'nucleus_prediction')]:
            with table_path(folder,name).open() as stream:
                for row in csv.DictReader(stream):
                    target.append(dict(row,dimension=arm['dimension'],history=arm['history'],response=arm['response'],seed=arm['seed']))
    for values,name in [(rows,'free_rollout'),(birth,'nucleus_prediction')]:
        write_metric_rows(values,out,family=FAMILY,name=name,columns=list(values[0]))
    plots=out/'plots';plots.mkdir(parents=True,exist_ok=True)
    for values,metric,field,space,filename,xname in [(rows,'energy_score','space','descriptors','future-distribution','horizon_ps'),(birth,'ap','task','nucleus64','nucleus64-ap','lead_ps')]:
        fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True)
        for ax,(dim,hist) in zip(axes.flat,[(16,1),(16,4),(64,1),(64,4)]):
            for response in [False,True]:
                selected=[r for r in values if r['dimension']==dim and r['history']==hist and r['response']==response and r['metric']==metric and r[field]==space]
                times=sorted({float(r[xname]) for r in selected})
                lookup={(r['seed'],float(r[xname])):float(r['value']) for r in selected}
                scores=np.array([[lookup[seed,t] for t in times] for seed in [c['seed'],c['seed']+1]])
                ax.plot(times,scores.mean(0),'o-',label='Values + responses' if response else 'Values')
                ax.fill_between(times,scores.min(0),scores.max(0),alpha=.15)
            ax.set_title(f'{dim} variables × {hist} observations');ax.set_xlabel('Lead before nucleus (ps)' if xname=='lead_ps' else 'Free rollout (ps)');ax.set_ylabel('AP' if metric=='ap' else 'Energy score (lower is better)');ax.grid(alpha=.2)
        axes.flat[0].legend();fig.suptitle('Two training seeds; shading is seed range, not confidence interval');fig.tight_layout();fig.savefig(plots/(filename+'.png'),dpi=170);fig.savefig(plots/(filename+'.pdf'));plt.close(fig)
    write_json(out/'technical/complete.json',dict(state='complete',models=len(c['arms']),source='existing frozen score tables',no_new_fits=True))
