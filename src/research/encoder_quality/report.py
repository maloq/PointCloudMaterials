"""Dense spatial panels and an honest completion/quality overview."""
import argparse
import csv
import html
import json
from pathlib import Path
import numpy as np
from src.research.structural_state.common import write_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from .common import load


def static_plots(config,spec,model,frames,cache,device,encode):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from threadpoolctl import threadpool_limits
    from umap import UMAP
    root=Path(config['output']);plots=root/'plots';plots.mkdir(exist_ok=True)
    colors=plt.get_cmap('tab10')(np.arange(7));cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,7.5),7)
    paths=[]
    for index in (2,4,7):
        z,clusters,record,km=frames[index]
        with np.load(Path(config['reference'])/f'frame-{index:02d}.npz') as values:a=dict(values)
        with np.load(Path(config['dense_inputs'])/f'labels-{index:02d}.npz') as values:d=dict(values)
        n=record['anchor_count'];labels=a['context'][:n,1]
        np.testing.assert_array_equal(labels[d['original_indices']],d['context'][:len(d['original_indices'])])
        with np.load(Path(config['dense_inputs'])/f'frame-{index:02d}.npz') as values:x=values['nearest80']
        x=x*(config['normalization']['reference_scale_A']/config['normalization']['scales_A'][record['material']])
        features=encode(model,x,config['batch_size'],device)
        # Extend the actual metric fit; a second fit can choose another local minimum.
        with threadpool_limits(limits=1):
            np.testing.assert_array_equal(km.predict(z[:n]),clusters)
            dense_clusters=np.r_[clusters[d['original_indices']],km.predict(features)]
        if len(dense_clusters)!=8*len(d['original_indices']):raise ValueError('Wrong spatial plot density')
        xy=UMAP(n_neighbors=30,min_dist=.1,random_state=20260923,n_jobs=1).fit_transform(z[:n])
        np.save(cache/f'umap-{index:02d}.npy',xy)
        fig,axes=plt.subplots(2,2,figsize=(11,10),constrained_layout=True)
        for ax,field,title in [(axes[0,0],d['context'],'Physical reference context'),(axes[0,1],dense_clusters,'Embedding K=7')]:
            ax.scatter(*d['coords'][:,:2].T,c=field,cmap=cmap,norm=norm,s=1.25)
            ax.set(title=title,xlabel='x (Å)',ylabel='y (Å)',aspect='equal')
        for ax,field,title in [(axes[1,0],labels,'UMAP: physical context'),(axes[1,1],clusters,'UMAP: embedding clusters')]:
            ax.scatter(*xy.T,c=field,cmap=cmap,norm=norm,s=3)
            ax.set(title=title,xticks=[],yticks=[])
        names=['Liquid other','Crystal interior','Mixed neighborhood','Al planar fault','Nontemplate interior','Fivefold proxy','Ordered liquid']
        handles=[plt.Line2D([],[],marker='o',linestyle='',color=colors[j],label=label) for j,label in enumerate(names)]
        fig.legend(handles=handles,loc='outside lower center',ncol=4,fontsize=8,title='Physical reference only; embedding cluster colors are arbitrary')
        fig.suptitle(f'{spec["name"]} — {record["material"]} {Path(record["file"]).stem}\n'
            f'{len(dense_clusters):,} spatial points (8×); native 128-D metrics; UMAP is exploratory\n'
            'All static inputs relaxed; hot-trained models shown as transfer',fontsize=11)
        path=plots/f'{spec["name"]}-{record["material"]}.png';fig.savefig(path,dpi=150);plt.close(fig)
        paths.append(str(path.relative_to(root)))
    write_json(root/'technical/evaluations'/spec['name']/'figures.json',dict(plots=paths))


def collect(config):
    root=Path(config['output']);rows=[];statuses=[];gallery=[]
    for model in config['models']:
        folder=root/'technical/evaluations'/model['name']
        state='complete' if (folder/'complete.json').exists() else 'failed' if (folder/'failed.json').exists() else 'running' if (folder/'state.json').exists() else 'queued'
        statuses.append(dict(model=model['name'],state=state))
        if (folder/'figures.json').exists():gallery.extend(json.loads((folder/'figures.json').read_text())['plots'])
        static_path=folder/'static-metrics.json'
        if static_path.exists():
            metrics=json.loads(static_path.read_text())
            for material in ('Al','Ta','Zr'):
                values=[v for key,v in metrics.items() if f'_{material}_' in key]
                if not values:continue
                def mean(path):
                    selected=[]
                    for v in values:
                        for key in path:v=v[key]
                        if v is not None:selected.append(v)
                    return float(np.mean(selected)) if selected else None
                rows.append(dict(model=model['name'],domain=model['domain'],kind=model['kind'],material=material,
                    frames=len(values),liquid_neighbor_nmse=mean(('supplement','liquid_order','embedding_neighbor_nmse')),
                    liquid_order_nmse=mean(('supplement','liquid_order','embedding_nmse')),
                    nonbulk_accuracy=mean(('supplement','nonbulk_context','balanced_accuracy')),
                    spatial_auc=mean(('nonbulk_spatial','auc')),liquid_rank=mean(('liquid_collapse','effective_rank'))))
    (root/'tables').mkdir(exist_ok=True);snapshot_metric_docs(root,'encoder_quality')
    if rows:
        with (root/'tables/structural.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    predictions=[]
    for model in config['models']:
        path=root/'technical/evaluations'/model['name']/'predictive-metrics.json'
        if not path.exists():continue
        metrics=json.loads(path.read_text())
        for name,record in metrics['readouts'].items():
            row=dict(model=model['name'],domain=model['domain'],readout=name)
            for h in ('3','6','12'):
                scores=record['scores'][h]['test']
                row.update({f'{scale}_{key}_{h}ps':scores[scale][key] for scale in ('raw','calibrated') for key in ('log_loss','brier','average_precision')})
            predictions.append(row)
    if predictions:
        with (root/'tables/prediction.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(predictions[0]));writer.writeheader();writer.writerows(predictions)
    write_json(root/'technical/summary.json',dict(statuses=statuses,structural_rows=len(rows),predictive_rows=len(predictions)))
    def table(values):
        if not values:return '<p>Pending</p>'
        keys=list(values[0]);head=''.join('<th>'+html.escape(k)+'</th>' for k in keys)
        body=''.join('<tr>'+''.join('<td>'+html.escape(f'{r[k]:.5g}' if isinstance(r[k],float) else str(r[k]))+'</td>' for k in keys)+'</tr>' for r in values)
        return '<table><tr>'+head+'</tr>'+body+'</table>'
    page='<!doctype html><meta charset="utf-8"><title>Latest MACE quality</title><style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:7px;border-bottom:1px solid #ddd}img{max-width:100%}</style>'
    page+='<h1>Latest native MACE: structural and predictive information</h1><p>Frozen encoders. No temperature/time inputs. Observed and relaxed prediction tracks remain separate. One fitted seed per recipe. Static snapshots are exploratory; onset chiefly measures existing-crystal arrival, not nucleus birth.</p>'
    page+='<p><a href="tables/structural.csv">Structural CSV</a> · <a href="tables/prediction.csv">Prediction CSV</a> · <a href="tables/METRICS.md">Definitions</a></p>'+table(statuses)
    page+='<h2>Structural metrics</h2>'+table(rows)+'<h2>Predictive proper scores and diagnostics</h2>'+table(predictions)
    for path in gallery:page+='<h3>'+html.escape(Path(path).stem)+'</h3><img loading="lazy" src="'+html.escape(path)+'">'
    (root/'index.html').write_text(page)
    return statuses


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();print(collect(load(args.config)))
