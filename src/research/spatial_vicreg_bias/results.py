"""Summarize completed frozen assays; no encoder, clusterer or readout is refitted."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.experiment_runner.metric_docs import write_metric_table
from src.research.structural_state.common import sha,write_json
from .data import load


def run(config,output):
    c=load(config);root=Path(c['output']);out=Path(output)
    for part in ('plots','tables','technical'):(out/part).mkdir(parents=True,exist_ok=True)
    if (out/'technical/complete.json').exists():raise FileExistsError(out)
    aroot=Path(c['cache'])/'assay'
    a={k:np.load(aroot/f'{k}.npy') for k in ('role','source','frame','atom','uniform','visible_A','ptm','support_fraction')}
    masks=dict(all=(a['role']=='test')&a['uniform'])
    masks['clear_input']=masks['all']&~a['visible_A']
    masks['mixed_noncrystal_center']=masks['all']&a['visible_A']&~np.isin(a['ptm'],[1,2,3])
    seeds=c['training']['seed_values'];alphas=c['training']['alphas'];epochs=c['assay']['epochs']
    curves={};endpoints={};occupancy={};bindings={};source_errors={};names=None;source_ids=None
    for seed in seeds:
        for alpha in alphas:
            name=f'S{alpha:g}-seed{seed}';curves[name]={};occupancy[name]={}
            fit=json.loads((root/name/'technical/complete.json').read_text())
            if fit['epochs']!=24 or fit['updates']!=108552:raise ValueError(f'Incomplete fit: {name}')
            for epoch in epochs:
                folder=root/name/'analyses'/f'epoch-{epoch:02d}'
                receipt=folder/'technical/complete.json';bindings[str(receipt)]=sha(receipt)
                saved=json.loads(receipt.read_text())['metrics'];curves[name][str(epoch)]={}
                for rep in ('encoder','projector'):
                    m=saved[rep]['k7'];read=m['readouts']['cohorts']['clear_input']['tracks']['uniform']['test']
                    curves[name][str(epoch)][rep]=dict(
                        normalized_neighbor_distance=m['spatial']['all']['neighbor_squared_distance'],
                        clear_variance_ratio=m['spatial']['clear_input']['variance_relative_to_training'],
                        clear_effective_rank=m['spatial']['clear_input']['effective_rank'],
                        width_A=m['profiles']['transition_width_10_90_A'],
                        clear_cluster_tda_skill=read['models']['cluster_means']['tda']['train_mean_skill'],
                        clear_continuous_tda_skill=read['models']['continuous_embedding']['tda']['train_mean_skill'])
                    if epoch in (4,12,24):
                        path=folder/'data'/f'{rep}-k7-source-errors.npz';bindings[str(path)]=sha(path)
                        columns=json.loads((folder/'technical'/f'{rep}-k7-readout.json').read_text())['descriptor_names']
                        if names is None:names=columns
                        if names!=columns:raise ValueError('Descriptor columns differ across frozen assays')
                        prefix='clear_input/uniform/test/'
                        with np.load(path) as z:
                            ids=z[prefix+'source_ids']
                            if source_ids is None:source_ids=ids
                            if not np.array_equal(source_ids,ids):raise ValueError('Source rows differ')
                            for model,key in [('mean','mean_source_mse'),('continuous','continuous_embedding/source_mse'),('cluster','cluster_means/source_mse')]:
                                source_errors[(seed,alpha,epoch,rep,model)]=z[prefix+key].astype(float)
                    if epoch!=24:continue
                    endpoints.setdefault(rep,{})[name]=dict(
                        clear={model:{f:v['train_mean_skill'] for f,v in values.items()}
                               for model,values in read['models'].items()},
                        all_cluster={f:v['train_mean_skill'] for f,v in m['readouts']['cohorts']['all']['tracks']['uniform']['test']['models']['cluster_means'].items()},
                        k_panel_clear_tda={str(k):saved[rep][f'k{k}']['readouts']['cohorts']['clear_input']['tracks']['uniform']['test']['models']['cluster_means']['tda']['train_mean_skill'] for k in c['assay']['ks']})
                    occupancy[name][rep]={}
                    for k in (7,10):
                        path=folder/'data'/f'{rep}-k{k}-assignments.npz';bindings[str(path)]=sha(path)
                        with np.load(path) as z:
                            for key in ('source','frame','atom'):
                                if not np.array_equal(z[key],a[key]):raise ValueError(f'Changed assignment population: {path}, {key}')
                            labels=z['cluster']
                        record={}
                        for cohort,mask in masks.items():
                            count=np.bincount(labels[mask],minlength=k)
                            record[cohort]=dict(rows=int(mask.sum()),largest_cluster_fraction=float(count.max()/count.sum()),
                                occupied_clusters=int(np.count_nonzero(count)),counts=count.tolist())
                        occupancy[name][rep][str(k)]=record
    tda=np.asarray([n.startswith('tda/') for n in names])
    rng=np.random.default_rng(20260930);draw=rng.integers(len(source_ids),size=(10000,len(source_ids)))
    paired={}
    # Average the paired seeds first, then bootstrap entire source IDs.
    # These intervals condition on the three trained seeds, not an infinite seed population.
    for rep in ('encoder','projector'):
        for family in ('geometry','bond_order','cna','tda'):
            columns=np.asarray([n.startswith(family+'/') and '/density' not in n for n in names])
            def loss(alpha,epoch,model):
                return np.stack([source_errors[(s,alpha,epoch,rep,model)][:,columns].mean(1) for s in seeds]).mean(0)
            baseline=loss(0,24,'mean')
            cases={
                'same_center_continuous_gain_epoch4_to24':(loss(0,4,'continuous'),loss(0,24,'continuous')),
                'same_center_advantage_over_neighbors_epoch24':(loss(1,24,'continuous'),loss(0,24,'continuous')),
                'same_center_continuous_advantage_over_clusters_epoch24':(loss(0,24,'cluster'),loss(0,24,'continuous'))}
            for key,(first,second) in cases.items():
                per_source=first-second
                delta=float(per_source.mean()/baseline.mean())
                samples=per_source[draw].mean(1)/baseline[draw].mean(1)
                paired.setdefault(rep,{}).setdefault(family,{})[key]=dict(delta_skill=delta,
                    ci95=np.quantile(samples,[.025,.975]).tolist(),sources=len(source_ids),seeds=len(seeds))
    nulls={}
    for name in ('diffusion-0','diffusion-1','diffusion-2','diffusion-4','local-q6','averaged-q6'):
        p=root/'nulls/analyses'/name/'technical/complete.json';bindings[str(p)]=sha(p)
        m=json.loads(p.read_text())['metrics']['k7']['readouts']['cohorts']
        nulls[name]={cohort:{f:v['train_mean_skill'] for f,v in m[cohort]['tracks']['uniform']['test']['models']['cluster_means'].items()}
                     for cohort in ('all','noncrystal_center','clear_input')}
    result=dict(curves=curves,endpoints=endpoints,occupancy=occupancy,paired=paired,nulls=nulls,
        coverage=dict(encoders=9,epochs_each=24,checkpoint_assays=63,representations_per_assay=2,
                      test_sources=len(source_ids),uniform_test_rows=int(masks['all'].sum()),clear_test_rows=int(masks['clear_input'].sum())))
    write_json(out/'technical/summary.json',result);write_json(out/'technical/input-hashes.json',bindings)
    write_metric_table(result,out,family='spatial_vicreg_results',name='completed-study')
    fig,axes=plt.subplots(1,3,figsize=(14,4.2))
    for alpha in alphas:
        color={0:'C0',.5:'C1',1:'C2'}[alpha]
        names_run=[f'S{alpha:g}-seed{s}' for s in seeds]
        for ax,key in zip(axes[:2],('clear_cluster_tda_skill','clear_continuous_tda_skill')):
            v=np.asarray([[curves[n][str(e)]['encoder'][key] for e in epochs] for n in names_run])*100
            ax.plot(epochs,v.mean(0),marker='.',label=f'Neighbor weight {alpha:g}',color=color)
            ax.fill_between(epochs,v.min(0),v.max(0),color=color,alpha=.15)
        d=[curves[n]['24']['encoder']['normalized_neighbor_distance'] for n in names_run]
        axes[2].bar(str(alpha),np.mean(d),color=color,alpha=.7)
        axes[2].scatter([str(alpha)]*3,d,color='black',s=16,zorder=3)
    axes[0].set(title='Global K=7 clusters lose liquid resolution',xlabel='Completed epochs',ylabel='Liquid TDA prediction skill (%)')
    axes[1].set(title='Continuous embeddings retain more structure',xlabel='Completed epochs',ylabel='Liquid TDA prediction skill (%)')
    axes[2].set(title='Neighbor alignment contracts pair distances',xlabel='Neighbor alignment weight',ylabel='Neighbor squared distance / training variance')
    axes[0].legend(fontsize=8);axes[0].axhline(0,color='grey',lw=.5)
    fig.suptitle('Fixed held-out Al sources · raw encoder · shading: range across three paired seeds',fontsize=11)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(out/'plots'/f'mechanism-summary.{ext}',dpi=180)
    plt.close(fig)
    write_json(out/'technical/complete.json',dict(state='complete',refitted_models=0,source_bootstrap_draws=10000,
        definition='post-completion saved-output review; fixed primary endpoint retained'))
    print(json.dumps(dict(coverage=result['coverage'],paired=paired),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output)
