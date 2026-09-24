"""Freeze an explicit interim native-screen evidence cut and compute supplements."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from src.research.encoder_screen.common import load_config,write,sha
from src.research.encoder_screen.report import summarize
from src.experiment_runner.metric_docs import snapshot_metric_docs
from .assess import supplement


def review(config_path,output):
    c=load_config(config_path);root=Path(c['output']);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    selected=[t for t in c['tasks'] if (root/'technical/evaluations'/t['name']/'complete.json').exists()]
    write(out/'technical/cut.json',dict(config_sha256=sha(config_path),completed=[t['name'] for t in selected],total=len(c['tasks'])))
    rows=[]
    reference=Path(c['reuse_geoframe'])/'technical/evaluations/epoch-034/current-physics-future-predictions.npz'
    for task in selected:
        folder=root/'technical/evaluations'/task['name'];dest=out/'technical/models'/task['name']
        scores=supplement(folder,c['reference'],dest)
        m=json.loads((folder/'metrics.json').read_text());complete=json.loads((folder/'complete.json').read_text())
        for rep in complete['extraction']['representations']:
            row=summarize(task['name'],m,rep,reference,folder,task)
            row.pop('source');row.update(family=task['kind'])
            frames=[v for k,v in scores.items() if k.endswith('_Al_'+rep)]
            for family in ['liquid_order','liquid_topology']:
                for key in ['embedding_nmse','density_nmse','embedding_neighbor_nmse','retrieval_gain_vs_density']:
                    row[family+'_'+key]=float(np.mean([v[family][key] for v in frames]))
            for key,id in [('nonbulk_boundary_ap','2'),('nonbulk_fault_ap','3')]:
                values=[v['nonbulk_context']['ap'][id] for v in frames if v['nonbulk_context']['ap'].get(id) is not None]
                row[key]=float(np.mean(values)) if values else None
            rows.append(row)
        print('reviewed',task['name'],flush=True)
    snapshot_metric_docs(out,'encoder_parameter_search')
    with (out/'tables/snapshot-review.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    associations=[]
    # One raw export per checkpoint; no correlated 35-epoch sequence in this cut.
    for family in ['all']+sorted({r['family'] for r in rows}):
        group=[r for r in rows if r['representation']=='encoder' and (family=='all' or r['family']==family)]
        if len(group)<4:continue
        for x in ['liquid_order_r2','nonbulk_fault_ap','nonbulk_spatial_auc','liquid_order_embedding_neighbor_nmse','liquid_rank']:
            for y in ['brier_delta','future_mse_delta','onset_ap12']:
                valid=[r for r in group if r[x] is not None and r[y] is not None]
                if len(valid)<4 or np.std([r[x] for r in valid])==0 or np.std([r[y] for r in valid])==0:continue
                rho=float(spearmanr([r[x] for r in valid],[r[y] for r in valid]).statistic)
                associations.append(dict(family=family,n=len(valid),x=x,y=y,spearman=rho))
    with (out/'tables/exploratory-associations.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['family','n','x','y','spearman']);w.writeheader();w.writerows(associations)
    write(out/'technical/review.json',dict(rows=rows,associations=associations))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();review(a.config,a.output)
