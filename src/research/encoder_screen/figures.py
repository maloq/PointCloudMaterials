"""CPU-only figure pass using checked completed embedding artifacts."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .common import load_config,checked,write


def render(config,task,dense=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    from umap import UMAP
    from src.research.geoframe_evolution.reference import CONTEXT_NAMES
    root=Path(config['output']);folder=root/'technical/evaluations'/task['name']
    receipt=json.loads((folder/'complete.json').read_text());metrics=json.loads((folder/'metrics.json').read_text())
    reference=Path(config['reference']);manifest=json.loads((reference/'manifest.json').read_text())
    context_labels=['Liquid other','Crystal interior','Mixed crystalline neighborhood','Al planar fault','Nontemplate crystal interior','Five-fold liquid proxy','Ordered-liquid candidate']
    paths=[];colors=plt.get_cmap('tab10')(np.arange(7));cmap=ListedColormap(colors);norm=BoundaryNorm(np.arange(-.5,7.5),7)
    for i in [2,4,7]:
        rec=manifest['frames'][i]
        if rec['material'] not in task['materials']:continue
        artifact=folder/'embeddings'/f'frame-{i:02d}.npz';checked(artifact,receipt['feature_files'][artifact.name])
        features=np.load(artifact);a=np.load(reference/f'frame-{i:02d}.npz');n=rec['anchor_count'];xyz=a['coords'][:n]
        slab=abs(xyz[:,2]-np.median(xyz[:,2]))<np.ptp(xyz[:,2])*.12
        for rep in receipt['extraction']['representations']:
            z=features[rep][:n];clusters=np.load(folder/f'clusters-{i:02d}-{rep}.npy');labels=a['context'][:n,1]
            count=np.array(metrics[f'frame_{i:02d}_{rec["material"]}_{rep}']['cluster_context_counts']);total=count.sum(0)
            fraction=np.divide(count,total[None],out=np.full(count.shape,np.nan),where=total[None]>0)
            xyfile=folder/f'umap-{i:02d}-{rep}.npy'
            if xyfile.exists():xy=np.load(xyfile)
            else:
                xy=UMAP(n_neighbors=30,min_dist=.1,random_state=20260923,n_jobs=1).fit_transform(z);np.save(xyfile,xy)
            spatial_xyz=xyz[slab];spatial_labels=labels[slab];spatial_clusters=clusters[slab]
            density_caption=''
            if dense:
                from sklearn.cluster import KMeans
                d=np.load(root/config['spatial_plot']['inputs']/f'labels-{i:02d}.npz')
                new=np.load(folder/'dense8-embeddings'/f'frame-{i:02d}.npz')[rep]
                km=KMeans(n_clusters=7,n_init=5,random_state=20260923).fit(z[a['split'][:n]==0])
                np.testing.assert_array_equal(km.predict(z),clusters)
                spatial_xyz=d['coords'];spatial_labels=d['context'];spatial_clusters=np.r_[clusters[d['original_indices']],km.predict(new)]
                if len(spatial_xyz)!=config['spatial_plot']['factor']*len(d['original_indices']):raise ValueError('Spatial panel is not exactly the requested factor')
                density_caption=f'Spatial panels: {len(spatial_xyz):,} atoms (8× original); same slab. UMAP/counts: original fixed cohort.'
            fig,axes=plt.subplots(2,2,figsize=(13,11),constrained_layout=True)
            for ax,field,title in [(axes[0,0],spatial_labels,'Physical reference context'),(axes[0,1],spatial_clusters,'Embedding K=7')]:
                ax.scatter(spatial_xyz[:,0],spatial_xyz[:,1],c=field,cmap=cmap,norm=norm,s=config['spatial_plot']['marker_area_points2'] if dense else 8)
                ax.set(title=title,xlabel='x (Å)',ylabel='y (Å)',aspect='equal')
            cluster_handles=[plt.Line2D([],[],marker='o',linestyle='',color=colors[j],label=f'C{j+1}') for j in range(7)]
            axes[0,1].legend(handles=cluster_handles,title='Arbitrary cluster IDs',fontsize=7,ncol=4,loc='upper right')
            axes[1,0].scatter(*xy.T,c=labels,cmap=cmap,norm=norm,s=3);axes[1,0].set(title='UMAP colored by physical context',xticks=[],yticks=[])
            axes[1,1].imshow(fraction,cmap='Blues',vmin=0,vmax=1,aspect='auto')
            axes[1,1].set(title='Held-out cluster/context fractions',xticks=range(7),xticklabels=[f'{name}\nn={v}' for name,v in zip(context_labels,total)],yticks=range(7),yticklabels=[f'C{k+1}' for k in range(7)])
            axes[1,1].tick_params(axis='x',rotation=55,labelsize=8)
            for a0,b0 in zip(*np.where(fraction>.05)):axes[1,1].text(b0,a0,f'{fraction[a0,b0]:.2f}',ha='center',va='center',fontsize=8,color='white' if fraction[a0,b0]>.6 else 'black')
            handles=[plt.Line2D([],[],marker='o',linestyle='',color=colors[j],label=label) for j,label in enumerate(context_labels)]
            fig.legend(handles=handles,loc='outside lower center',ncol=4,fontsize=9)
            fig.suptitle(task['name']+' / '+rep+' / '+rec['material']+' '+Path(rec['file']).stem+'\n'+(density_caption if dense else 'Fixed atom reference; refitted UMAP is exploratory; candidate liquid labels do not establish nuclei'),fontsize=12)
            path=root/'plots'/f'{task["name"]}-{rep}-{i:02d}.png';fig.savefig(path,dpi=150);plt.close(fig);paths.append(str(path.relative_to(root)))
    write(folder/'figures.json',dict(state='complete',plots=paths));return paths


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--name');p.add_argument('--watch',action='store_true');a=p.parse_args();config=load_config(a.config);root=Path(config['output'])
    deadline=float('inf')
    if a.watch and 'SLURM_JOB_ID' in os.environ:
        from src.training_methods.shared_pretraining.queue import deadline_for_job
        deadline=deadline_for_job()
    while True:
        unresolved=0
        for task in config['tasks']:
            if a.name and task['name']!=a.name:continue
            folder=root/'technical/evaluations'/task['name']
            if (folder/'figures.json').exists() or (folder/'figures-failed.json').exists() or (folder/'failed.json').exists():continue
            if not (folder/'complete.json').exists():unresolved+=1;continue
            try:
                render(config,task)
                from .report import report
                with (root/'technical/report.lock').open('a') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);report(config)
            except Exception as exc:
                write(folder/'figures-failed.json',dict(error=repr(exc),traceback=traceback.format_exc()))
                print(traceback.format_exc(),flush=True)
        if not a.watch or not unresolved or time.time()>deadline-300:break
        time.sleep(20)


if __name__=='__main__':main()
