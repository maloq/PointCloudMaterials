"""Readable matched panels, exact evidence links and visible pending/failed stages."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.geoframe_evolution.review import paired_mean_ci
from .common import load_config,write


def summarize(name,metrics,representation,reference,source,task=None):
    p=metrics['prediction_'+representation];h=p['conditional_hazard'];h12=h['horizons']['12.0']
    row=dict(model=name,representation=representation,material='Al',status='complete',
        input_support='80-neighbor GeoFrame' if task is None else task['support'],
        checkpoint_step='' if task is None else task['step'],
        precision='float32' if task is None else task['precision'],
        liquid_order_r2=None,liquid_topology_r2=None,boundary_ap=None,fault_ap=None,
        nonbulk_ami=None,nonbulk_spatial_auc=None,liquid_rank=None,
        future_residual_mse=p['future_residual_9ps']['groups']['all']['mse'],
        primary_horizon_ps=3,onset_ap3=h['horizons']['3.0']['average_precision'],
        onset_ap6=h['horizons']['6.0']['average_precision'],onset_ap12=h12['average_precision'],
        onset_brier3=h['horizons']['3.0']['brier'],onset_brier6=h['horizons']['6.0']['brier'],
        onset_brier12=h12['brier'],hazard_selected_step=h['best_step'],
        brier_delta=None,brier_ci_low=None,brier_ci_high=None,future_mse_delta=None,future_ci_low=None,future_ci_high=None,
        source=str(source))
    paths=dict(liquid_order_r2=('liquid_order','mean_r2'),liquid_topology_r2=('liquid_topology','mean_r2'),
               boundary_ap=('context','ap','2'),fault_ap=('context','ap','3'),
               nonbulk_ami=('nonbulk_cluster_ami',),nonbulk_spatial_auc=('nonbulk_spatial','auc'),
               liquid_rank=('liquid_collapse','effective_rank'))
    for key,path in paths.items():
        values=[]
        for i in range(3):
            value=metrics[f'frame_{i:02d}_Al_{representation}']
            for field in path:value=value.get(field) if isinstance(value,dict) else None
            if value is not None:values.append(value)
        row[key]=float(np.mean(values)) if values else None
    a=np.load(Path(source)/f'{representation}-future-predictions.npz');base=np.load(reference)
    for k in ('hazard_indices','hazard_source','hazard_event'):np.testing.assert_array_equal(a[k],base[k])
    event=a['hazard_event']<5;sources=a['hazard_source'];delta=(a['hazard_risks'][:,-1]-event)**2-(base['hazard_risks'][:,-1]-event)**2
    score=paired_mean_ci([delta[sources==s].mean() for s in np.unique(sources)])
    row.update(brier_delta=score['mean'],brier_ci_low=score['ci95'][0],brier_ci_high=score['ci95'][1])
    errors=p['future_residual_9ps']['per_source'];constant=p['residual_constant']['per_source']
    score=paired_mean_ci([errors[s]-constant[s] for s in sorted(errors)])
    row.update(future_mse_delta=score['mean'],future_ci_low=score['ci95'][0],future_ci_high=score['ci95'][1])
    return row


def report(config):
    root=Path(config['output']);reuse=Path(config['reuse_geoframe']);rows=[];tasks=[]
    base=reuse/'technical/evaluations/epoch-034/current-physics-future-predictions.npz'
    for folder in sorted((reuse/'technical/evaluations').iterdir()):
        if not (folder/'metrics.json').exists():continue
        m=json.loads((folder/'metrics.json').read_text())
        for rep in ('encoder','projector'):rows.append(summarize('geoframe-'+folder.name,m,rep,base,folder))
    for t in config['tasks']:
        folder=root/'technical/evaluations'/t['name'];status='pending'
        if (folder/'complete.json').exists():
            status='complete';m=json.loads((folder/'metrics.json').read_text())
            receipt=json.loads((folder/'complete.json').read_text())
            for rep in receipt['extraction']['representations']:rows.append(summarize(t['name'],m,rep,base,folder,t))
        elif (folder/'failed.json').exists():status='failed'
        elif (folder/'task.json').exists():status='running_or_interrupted'
        tasks.append(dict(name=t['name'],step=t['step'],status=status,kind=t['kind'],materials=t['materials']))
    for part in ('plots','tables','technical'):(root/part).mkdir(parents=True,exist_ok=True)
    snapshot_metric_docs(root,'encoder_screen')
    with (root/'tables/summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    counts={state:sum(t['status']==state for t in tasks) for state in ['complete','pending','running_or_interrupted','failed']}
    write(root/'technical/summary.json',dict(tasks=tasks,counts=counts,rows=rows,excluded=config['excluded']))
    # Fixed endpoint reference controls avoid letting 35 correlated GeoFrame
    # epochs dominate a cross-family figure. Full trajectory remains in the table.
    selected=[r for r in rows if not r['model'].startswith('geoframe-epoch-') or r['model'] in ('geoframe-epoch-011','geoframe-epoch-034')]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(17,max(5,len(selected)*.25)),constrained_layout=True)
    for ax,key,title in zip(axes,['liquid_order_r2','fault_ap','nonbulk_ami','brier_delta'],
        ['Liquid order R² ↑','Planar-fault AP ↑','Nonbulk cluster AMI ↑','12ps Brier Δ vs physics ↓']):
        for i,r in enumerate(selected):
            if r[key] is not None:ax.scatter(r[key],i,s=20,color='#176d91' if r['representation']=='encoder' else '#b55834')
            if key=='brier_delta':ax.plot([r['brier_ci_low'],r['brier_ci_high']],[i,i],color='gray',lw=.8)
        ax.set(title=title,yticks=range(len(selected)),yticklabels=[r['model']+' / '+r['representation'] for r in selected] if ax is axes[0] else [])
        ax.invert_yaxis();ax.grid(alpha=.2);ax.axvline(0,color='gray',lw=.6)
    fig.suptitle('Fixed Al centers and future targets; native inputs/precision differ\nStatic means across three frames; Brier intervals resample 15 reused development roots, not training seeds')
    fig.savefig(root/'plots/comparison.png',dpi=150);plt.close(fig)
    columns=['model','representation','onset_ap3','onset_ap6','onset_ap12','onset_brier3','liquid_order_r2','liquid_topology_r2','fault_ap','boundary_ap','nonbulk_ami','nonbulk_spatial_auc','onset_brier12','hazard_selected_step','brier_delta','future_mse_delta']
    def cell(v):return 'undefined' if v is None else f'{v:.5g}' if isinstance(v,float) else html.escape(str(v))
    table='<table><thead><tr>'+''.join('<th>'+k+'</th>' for k in columns)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+cell(r[k])+'</td>' for k in columns)+'</tr>' for r in rows)+'</tbody></table>'
    tasks_html='<ul>'+''.join('<li>'+html.escape(t['name'])+': '+t['status']+'</li>' for t in tasks)+'</ul>'
    page='''<!doctype html><meta charset="utf-8"><title>Encoder physical screen</title><style>body{font:16px system-ui;margin:30px;color:#153340}table{border-collapse:collapse;font-size:12px}td,th{padding:7px;border-bottom:1px solid #ddd}th{position:sticky;top:0;background:#eef5f7}img{max-width:100%}input{padding:10px;width:50%}</style><h1>Encoder physical screen</h1>'''
    page+='<p>'+html.escape(str(counts))+' new evaluations; 37 prior GeoFrame checkpoints reused.</p>'
    page+='<p>Primary onset metric is AP at 3 ps; 6 ps and 12 ps are secondary. Historical checkpoint and NLL readout selections are unchanged. Native input supports and numerical precision differ. All snapshot probes share physical labels/centers. Static Al is a transductive screen, and the 15 future development roots have been repeatedly inspected. Positive AP alone does not establish calibrated prediction. Step 0 is the constant-risk control. Blank/undefined classes are not zeros.</p><p><a href="tables/summary.csv">Summary CSV</a> · <a href="tables/METRICS.md">Metric definitions</a> · <a href="../../../docs/encoder_screen.md">Protocol and execution</a></p>'
    galleries=[]
    for t in config['tasks']:
        path=root/'technical/evaluations'/t['name']/'figures.json'
        if path.exists():
            galleries.extend('<a href="'+html.escape(f)+'">'+html.escape(Path(f).stem)+'</a>' for f in json.loads(path.read_text())['plots'])
    page+='<p>Cached spatial/UMAP/context panels: '+' · '.join(galleries)+'</p>'
    page+='<img src="plots/comparison.png"><p><input id="q" placeholder="Filter model / representation"></p>'+table+'<h2>Queue</h2>'+tasks_html
    page+='''<script>document.getElementById('q').oninput=e=>document.querySelectorAll('tbody tr').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script>'''
    tmp=root/'index.building.html';tmp.write_text(page);tmp.replace(root/'index.html')
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    print(report(load_config(p.parse_args().config)))
