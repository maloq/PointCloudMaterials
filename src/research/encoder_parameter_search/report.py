"""Separate material, representation, training-time and matched-seed views."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.encoder_screen.common import write
from src.research.encoder_screen.report import summarize
from .queue import read,evaluation_config


def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def decisions(rows,config):
    """Require the declared effect in both seeds; pending evidence never passes."""
    final=[r for r in rows if r['milestone']==(35 if r['family']=='geoframe' else 4096)]
    results=[]
    baselines={'geoframe':'gf-mlp-cov1-factor1','mace':'mace-lr1e-05-distance0'}
    for family in ('geoframe','mace'):
        seeds=sorted({i['seed'] for i in config['fits'] if i['family']==family})
        materials=['Al','Ta','Zr'] if family=='geoframe' else ['Al']
        for recipe in sorted({i['recipe'] for i in config['fits'] if i['family']==family}):
            for rep in (['encoder','projector'] if family=='geoframe' else ['encoder']):
                checks=[];predictive=[];effects=[]
                for seed in seeds:
                    candidate=[r for r in final if (r['recipe'],r['seed'],r['representation'])==(recipe,seed,rep)]
                    baseline=[r for r in final if (r['recipe'],r['seed'],r['representation'])==(baselines[family],seed,rep)]
                    if {r['material'] for r in candidate}!=set(materials) or {r['material'] for r in baseline}!=set(materials):continue
                    ratios=[];retain=[];aps=[];spatial=[];coverage=True
                    for material in materials:
                        a=next(r for r in baseline if r['material']==material);b=next(r for r in candidate if r['material']==material)
                        ratios.append(b['liquid_order_neighbor_nmse']/a['liquid_order_neighbor_nmse'])
                        retain += [b[k]/a[k] for k in ['liquid_order_nmse','liquid_topology_nmse']]
                        for k in ['nonbulk_boundary_ap','nonbulk_fault_ap']:
                            if a[k] is not None:
                                if b[k] is None:coverage=False
                                else:aps.append(b[k]-a[k])
                        if b['spatial_auc'] is None or a['spatial_auc'] is None:coverage=False
                        else:spatial.append(b['spatial_auc']-a['spatial_auc'])
                    checks.append(coverage and bool(aps) and bool(spatial) and np.mean(ratios)<=.99 and max(retain)<=1.02 and min(aps)>=-.02 and min(spatial)>=-.02)
                    al=next(r for r in candidate if r['material']=='Al')
                    predictive.append(al['brier_delta']<0 and al['future_mse_delta']<=0 and al['hazard_selected_step']>0)
                    effects.append(dict(seed=seed,neighbor_change_percent=float(100*(np.mean(ratios)-1)),
                        worst_retention_ratio=float(max(retain)),coverage=coverage,
                        minimum_ap_change=float(min(aps)) if aps else None,minimum_spatial_change=float(min(spatial)) if spatial else None))
                results.append(dict(family=family,recipe=recipe,representation=rep,baseline=baselines[family],
                    paired_seeds=len(checks),required_seeds=len(seeds),
                    structural_promotion=bool(len(checks)==len(seeds) and all(checks)),
                    predictive_promotion=bool(len(checks)==len(seeds) and all(checks) and all(predictive)),effects=effects))
    # Descriptive Pareto set within a family/export, only after both seeds exist.
    # No weighted scalar score trades a fault against liquid fidelity or risk.
    vectors={}
    for r in results:
        if r['paired_seeds']!=r['required_seeds']:continue
        selected=[v for v in final if (v['recipe'],v['representation'])==(r['recipe'],r['representation'])]
        al=[v for v in selected if v['material']=='Al']
        vector=[mean([v[k] for v in selected]) for k in ['liquid_order_nmse','liquid_topology_nmse','liquid_order_neighbor_nmse']]
        vector += [-mean([v[k] for v in al]) for k in ['nonbulk_boundary_ap','nonbulk_fault_ap']]
        vector += [-mean([v['spatial_auc'] for v in selected]),mean([v['brier_delta'] for v in al]),mean([v['future_mse_delta'] for v in al])]
        if all(v is not None and np.isfinite(v) for v in vector):vectors[(r['recipe'],r['representation'])]=np.array(vector)
    for r in results:
        key=(r['recipe'],r['representation']);r['descriptive_pareto']=None
        if key not in vectors:continue
        competitors=[vectors[(v['recipe'],v['representation'])] for v in results
            if v['family']==r['family'] and v['representation']==r['representation'] and (v['recipe'],v['representation']) in vectors]
        r['descriptive_pareto']=not any(np.all(v<=vectors[key]) and np.any(v<vectors[key]) for v in competitors)
    return results


def report(path):
    c=read(path);root=Path(c['output']);ec=evaluation_config(c)
    for part in ('tables','plots','technical'):(root/part).mkdir(parents=True,exist_ok=True)
    reference=Path(ec['reuse_geoframe'])/'technical/evaluations/epoch-034/current-physics-future-predictions.npz'
    rows=[];status=[]
    for item in c['fits']:
        stage=root/'technical/tasks'/item['name']
        state='complete' if (stage/'complete.json').exists() else 'failed' if (stage/'failed.json').exists() else 'pending_or_running'
        status.append(dict(name=item['name'],state=state))
        for milestone in item['milestones']:
            name=f'{item["name"]}-{milestone:04d}';folder=root/'technical/evaluations'/name
            extra=root/'technical/supplements'/name/'technical/metrics.json'
            if not (folder/'complete.json').exists() or not extra.exists():continue
            receipt=json.loads((folder/'complete.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
            supplement=json.loads(extra.read_text())
            for rep in receipt['extraction']['representations']:
                original=summarize(name,metrics,rep,reference,folder,receipt['task'])
                for material in receipt['task']['materials']:
                    selected=[v for k,v in supplement.items() if k.endswith(f'_{material}_{rep}')]
                    base=[v for k,v in metrics.items() if k.startswith('frame_') and k.endswith(f'_{material}_{rep}')]
                    row=dict(fit=item['name'],recipe=item['recipe'],family=item['family'],seed=item['seed'],
                        milestone=milestone,representation=rep,material=material,
                        liquid_order_nmse=mean([v['liquid_order'].get('embedding_nmse') for v in selected]),
                        liquid_topology_nmse=mean([v['liquid_topology'].get('embedding_nmse') for v in selected]),
                        liquid_order_neighbor_nmse=mean([v['liquid_order'].get('embedding_neighbor_nmse') for v in selected]),
                        liquid_order_retrieval_gain=mean([v['liquid_order'].get('retrieval_gain_vs_density') for v in selected]),
                        nonbulk_boundary_ap=mean([v['nonbulk_context']['ap'].get('2') for v in selected]),
                        nonbulk_fault_ap=mean([v['nonbulk_context']['ap'].get('3') for v in selected]),
                        spatial_auc=mean([v['nonbulk_spatial']['auc'] for v in base]),
                        liquid_rank=mean([v['liquid_collapse']['effective_rank'] for v in base]),
                        perturbation_p95_001=mean([v['continuity_fixed_candidates_center_fixed']['0.01']['normalized_p95'] for v in base]))
                    for field in ['onset_ap12','onset_brier12','hazard_selected_step','brier_delta','brier_ci_low','brier_ci_high','future_mse_delta','future_ci_low','future_ci_high']:
                        row[field]=original[field] if material=='Al' else None
                    rows.append(row)
    snapshot_metric_docs(root,'encoder_parameter_search')
    if rows:
        with (root/'tables/training-comparison.csv').open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    rules=decisions(rows,c)
    write(root/'technical/selection-rules.json',rules)
    write(root/'technical/summary.json',dict(fits=status,rows=rows,selection=rules))
    counts={s:sum(r['state']==s for r in status) for s in ['complete','failed','pending_or_running']}
    page='<!doctype html><meta charset="utf-8"><title>Encoder parameter search</title><style>body{font:16px system-ui;margin:28px;color:#193340}table{border-collapse:collapse;font-size:12px}th,td{padding:7px;border-bottom:1px solid #ddd}img{max-width:100%}th{position:sticky;top:0;background:#eef5f7}</style><h1>Matched encoder parameter search</h1>'
    page+=f'<p>{counts}: 20 GeoFrame fits (16 VICReg + 4 VISReg; 35 complete passes), 8 MACE fits (4096 sampled updates).</p>'
    page+='<p><a href="RESULTS.md">Evidence and selection criteria</a> · <a href="tables/training-comparison.csv">All measurements</a> · <a href="tables/METRICS.md">Definitions</a></p><p>Separate structure, interface, liquid-neighbor, coherence and conditional prediction criteria. Static snapshots and forecast development sources are reused; this is parameter selection, not final validation. Ta/Zr liquid proxies do not establish nuclei.</p>'
    page+='<p><a href="technical/selection-rules.json">Predeclared two-seed promotion checks</a>: '+str(sum(r['structural_promotion'] for r in rules))+' structural promotions; '+str(sum(r['predictive_promotion'] for r in rules))+' also pass the predictive screen. Pending pairs cannot pass.</p>'
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for family in ['geoframe','mace']:
            for item in [i for i in c['fits'] if i['family']==family]:
                v=[r for r in rows if r['fit']==item['name'] and r['material']=='Al' and r['representation']=='encoder']
                for ax,key,title in zip(axes,['liquid_order_nmse','nonbulk_fault_ap','brier_delta'],['Liquid-order error ↓','Nonbulk fault AP ↑','Brier difference vs physics ↓']):
                    ax.plot([r['milestone']/(35 if family=='geoframe' else 4096) for r in v],[r[key] for r in v],marker='.',alpha=.7,label=item['name'])
                    ax.set(title=title,xlabel='Fraction of declared training budget');ax.grid(alpha=.2)
        axes[-1].axhline(0,color='black',lw=.6);fig.savefig(root/'plots/training-curves.png',dpi=150);plt.close(fig)
        page+='<img src="plots/training-curves.png"><p>Every line is one fit; use the table for recipe, seed and material identity. MACE update fractions and GeoFrame pass fractions are not equal data budgets.</p>'
        page+='<table><tr>'+''.join('<th>'+k+'</th>' for k in rows[0])+'</tr>'
        for row in rows:
            page+='<tr>'+''.join('<td>'+('undefined' if v is None else f'{v:.5g}' if isinstance(v,float) else html.escape(str(v)))+'</td>' for v in row.values())+'</tr>'
        page+='</table>'
    page+='<ul>'+''.join('<li>'+s['name']+': '+s['state']+'</li>' for s in status)+'</ul>'
    page+='<h2>Final checkpoint spatial panels</h2><p>Eight times the original spatial samples; marker diameter halved. Numerical cohorts stay fixed.</p>'
    for picture in sorted((root/'plots').glob('*-encoder-*.png')):
        page+=f'<details><summary>{html.escape(picture.stem)}</summary><img loading="lazy" src="plots/{picture.name}"></details>'
    temporary=root/'index.html.writing';temporary.write_text(page);temporary.replace(root/'index.html')
    return counts


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);print(report(p.parse_args().config))
