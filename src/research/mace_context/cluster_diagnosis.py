"""Matched static label coherence and saved-feature ablations; no encoder training."""

import argparse
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import pandas as pd
from scipy.spatial import cKDTree
import torch
from sklearn.metrics import adjusted_rand_score

from src.analysis.liquid_structure import bond_order, ORDER_NAMES
from src.experiment_runner.artifacts import result_folders, write_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256
from src.experiment_runner.tracking import tracked_run
from src.project_runtime.paths import load_json
from src.training_methods.shared.vicreg import VICRegLoss
from src.vis_tools.latent_analysis_vis import fit_clustering_model, predict_clustering_model


def agreement(labels, near, mask):
    valid = mask[:, None] & mask[near]
    a = np.broadcast_to(labels[:, None], near.shape)[valid]
    b = labels[near][valid]
    if len(a) == 0:
        raise ValueError('No eligible spatial edges')
    same = float(np.mean(a == b))
    chance = float(np.dot(np.bincount(a, minlength=7), np.bincount(b, minlength=7))/len(a)**2)
    return dict(same_label=same, chance=chance,
                adjusted_agreement=(same-chance)/(1-chance) if chance < 1 else None,
                edges=len(a), centers=int(mask.sum()))


def ptm(points):
    from ovito.data import DataCollection, Particles
    from ovito.modifiers import PolyhedralTemplateMatchingModifier
    data = DataCollection()
    particles = Particles(count=len(points))
    particles.create_property('Position', data=points)
    data.objects.append(particles)
    data.apply(PolyhedralTemplateMatchingModifier(rmsd_cutoff=.1))
    return np.asarray(data.particles['Structure Type']).copy()


def projector(config):
    cfg = OmegaConf.load(config['initial_config'])
    loss = VICRegLoss.from_config(cfg, input_dim=256)
    saved = torch.load(config['joint_checkpoint'], map_location='cpu', weights_only=False)
    state = {k.removeprefix('vicreg.'): v for k,v in saved['model_state'].items() if k.startswith('vicreg.')}
    loss.load_state_dict(state, strict=True)
    loss.eval()
    return loss.projector


def project(module, x):
    with torch.inference_mode():
        return np.concatenate([module(torch.as_tensor(v.copy())).numpy() for v in np.array_split(x, max(1, len(x)//4096))])


def run(config):
    root = result_folders(config['output'])
    torch.set_num_threads(config['cpu_threads'])
    rng = np.random.default_rng(config['seed'])
    source = Path(config['analysis'])/'technical'
    with np.load(source/'analysis_inference_cache.npz') as cache:
        z, coords = cache['inv_latents'], cache['coords']
    metadata = json.loads((Path(config['sample_cache'])/'metadata.json').read_text())
    head = projector(config)
    features = dict(dual=z, inner=z[:,:256], center=z[:,256:], projector=project(head,z[:,:256]))
    frames, existing, offset, fit_rows = [], [], 0, []
    references = {k: [] for k in config['references']}
    for shard in metadata['shards']:
        name = Path(shard['file']).stem
        n = shard['count']; section = slice(offset, offset+n); c = coords[section]
        current = np.load(source/f'snapshots/{name}/md_space/local_structure_coords_clusters.npz')
        np.testing.assert_array_equal(c, current['coords'])
        existing.append(current['clusters'])
        for method, reference in config['references'].items():
            values = np.load(Path(reference)/f'snapshots/{name}/md_space/local_structure_coords_clusters.npz')
            dist, ids = cKDTree(values['coords']).query(c)
            np.testing.assert_allclose(dist, 0, atol=1e-6, rtol=0)
            if len(np.unique(ids)) != n:
                raise ValueError('Archived center matching is not one-to-one')
            references[method].append(values['clusters'][ids])
        tree = cKDTree(c)
        near = tree.query(c, k=7, workers=4)[1][:,1:]
        selected = rng.choice(n, config['fit_per_frame'], replace=False)
        fit_rows.extend(offset+selected)
        p = Path(config['static_source'])/shard['file']
        points = np.load(p).astype(np.float64); atoms = cKDTree(points)
        dist, center_ids = atoms.query(c)
        np.testing.assert_allclose(dist, 0, atol=1e-7, rtol=0)
        structural_type = ptm(points)[center_ids]
        neighbors = atoms.query(c[selected], k=13, workers=4)[1]
        neighbors2 = atoms.query(points[neighbors].reshape(-1,3), k=13, workers=4)[1][:,1:]
        vectors = (points[neighbors2]-points[neighbors].reshape(-1,1,3)).reshape(-1,13,12,3)
        observables = bond_order(vectors, 3.7)[0]
        np.savez(root/f'technical/physical-{name}.npz', ptm=structural_type, sampled_rows=selected, observables=observables)
        frames.append(dict(name=name, section=section, near=near, selected=selected,
                           types=structural_type, observables=observables, source_sha256=sha256(p)))
        offset += n
        print('Prepared',name,n,'PTM Other fraction',float(np.mean(structural_type==0)),flush=True)
    if offset != len(z):
        raise ValueError('Static cache and metadata lengths differ')
    fit_rows = np.asarray(fit_rows)
    labels = {'current_full_fit': np.concatenate(existing)}
    labels.update({name: np.concatenate(parts) for name,parts in references.items()})
    np.save(root/'technical/clustering-fit-rows.npy', fit_rows)
    variants = config['variants']
    fit_info = {}
    for name, variant in variants.items():
        values = features[variant['features']]
        fitted, _, info = fit_clustering_model(values[fit_rows], 7, random_state=config['seed'],
            method=variant['method'], l2_normalize=variant['l2_normalize'], standardize=True,
            pca_variance=variant['pca_variance'], pca_max_components=variant['pca_max_components'])
        y = np.concatenate([predict_clustering_model(block, fitted) for block in np.array_split(values, 24)])
        labels[name] = y; fit_info[name] = info
        np.save(root/f'technical/labels-{name}.npy', y)
        print('Fit',name,'PCs',info['pca_components'],'silhouette',info['silhouette_cosine'],flush=True)
    spatial, physical, continuous, block = [], [], [], []
    for frame in frames:
        sl, near, selected = frame['section'], frame['near'], frame['selected']
        other = frame['types']==0
        for method, y in labels.items():
            local = y[sl]
            for region, mask in [('all',np.ones(len(local),bool)),('PTM_other',other)]:
                spatial.append(dict(frame=frame['name'], model=method, region=region, **agreement(local,near,mask)))
                rows = selected[mask[selected]]; o = frame['observables'][mask[selected]].astype(np.float64)
                yy = local[rows]
                for col, obs in enumerate(ORDER_NAMES):
                    grand = o[:,col].mean(); total = np.square(o[:,col]-grand).sum()
                    between = sum(np.sum(yy==k)*(o[yy==k,col].mean()-grand)**2 for k in np.unique(yy))
                    physical.append(dict(frame=frame['name'], model=method, region=region, observable=obs,
                        explained_variance=between/total if total>0 else None, samples=len(yy)))
        for region, mask in [('all',np.ones(len(other),bool)),('PTM_other',other)]:
            a,b = np.where(mask[:,None]&mask[near]); choose=rng.choice(len(a),min(20000,len(a)),replace=False)
            a,b = a[choose],near[a[choose],b[choose]]
            eligible = np.flatnonzero(mask); random = rng.choice(eligible,len(a),replace=True)
            for name,v in features.items():
                v=v[sl].astype(np.float64)
                std=v[mask].std(0); std[std==0]=1
                numerator=np.mean(np.square((v[a]-v[b])/std))
                denominator=np.mean(np.square((v[a]-v[random])/std))
                continuous.append(dict(frame=frame['name'],model=name,region=region,neighbor_over_random=numerator/denominator))
            v=z[sl].astype(np.float64);v/=np.linalg.norm(v,axis=1,keepdims=True)
            # Repository preprocessing: normalize rows, standardize channels, PCA.
            scaled=v/np.maximum(v.std(0),1e-30)
            energies=[np.square(scaled[a,s]-scaled[b,s]).sum() for s in [slice(0,256),slice(256,512)]]
            block.append(dict(frame=frame['name'],region=region,center_fraction_spatial_increment=energies[1]/sum(energies)))
    temporal=[]
    saved=np.load(config['temporal_features']); probes=np.load(config['probes'])
    train=probes['split']=='train'
    for name in ['inner','center','dual','projector']:
        section={'inner':slice(0,256),'center':slice(256,512),'dual':slice(None),'projector':slice(0,256)}[name]
        training=saved['z'][train,section].astype(np.float64)
        t=saved['temporal_z'][...,section].astype(np.float64)
        crossing=saved['crossing_z'][...,section].astype(np.float64)
        if name=='projector':
            training=project(head,training.astype(np.float32)).astype(np.float64)
            t=project(head,t.reshape(-1,256).astype(np.float32)).reshape(144,17,128).astype(np.float64)
            crossing=project(head,crossing.reshape(-1,256).astype(np.float32)).reshape(4,72,2,128).astype(np.float64)
        variance=np.mean(training.var(0)); one=np.mean(np.square(t[:,1:]-t[:,:-1]))
        for lag in [1,2,4,8,16]:
            temporal.append(dict(model=name,lag_ps=lag*.75,
                change_over_train_variance=float(np.mean(np.square(t[:,lag:]-t[:,:-lag]))/variance),
                crossing_1e4_over_075ps=float(np.mean(np.square(crossing[-1,:,0]-crossing[-1,:,1]))/one)))
    snapshot_metric_docs(root,'analysis')
    tables=dict(spatial=spatial,physical=physical,continuous=continuous,blocks=block,temporal=temporal)
    for name,rows in tables.items():pd.DataFrame(rows).to_csv(root/f'tables/{name}.csv',index=False)
    write_json(root/'technical/fit-info.json',fit_info)
    write_json(root/'technical/protocol.json',dict(config=config,
        frames=[{k:v for k,v in f.items() if k in ['name','source_sha256']} for f in frames],
        caveats='Descriptive same-source ablations. Fit 6000 centers per frame, predict every center. Archived fits used full original grid. PTM Other is not a liquid phase ground truth. Static time frames are not atom-tracked trajectories. Temporal lag scores use the separate retained six-source trajectory cohort. No encoder or physical head is trained.'))
    make_plots(root,frames,coords,labels,tables)
    write_json(root/'technical/status.json',dict(state='complete',models=list(labels)))


def make_plots(root,frames,coords,labels,tables):
    models=['current_full_fit','gf_vicreg_best','inner_standard','projector_standard']
    fig,axes=plt.subplots(1,4,figsize=(18,4.8),layout='constrained')
    frame=frames[0]; c=coords[frame['section']]; mid=(c[:,2].min()+c[:,2].max())/2
    mask=np.abs(c[:,2]-mid)<2.5
    for ax,name in zip(axes,models,strict=True):
        ax.scatter(c[mask,0],c[mask,1],c=labels[name][frame['section']][mask],s=9,cmap='tab10',vmin=0,vmax=9)
        ax.set(title=name,xlabel='x (Å)',ylabel='y (Å)',aspect='equal')
    fig.suptitle('166 ps: identical centers in a 5 Å slab; cluster colors are model-specific')
    fig.savefig(root/'plots/spatial-slice-comparison.png',dpi=170);plt.close(fig)
    df=pd.DataFrame(tables['spatial']);df=df[df.region=='PTM_other']
    pivot=df.pivot(index='frame',columns='model',values='adjusted_agreement')
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    pivot.plot.bar(ax=ax,width=.85)
    ax.set(ylabel='Neighbor label agreement above chance',title='Disordered regions: six-nearest-center graph, PTM Other at both endpoints')
    ax.legend(fontsize=7,ncol=3);fig.savefig(root/'plots/disordered-coherence.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args();config=load_json(args.config)
    root=result_folders(config['output'])
    with tracked_run(root/'technical/execution',kind='research',configs=[Path(args.config)],
                     command=[sys.executable,*sys.argv],question='Why do stable MACE embeddings give incoherent liquid clusters?'):
        run(config)


if __name__=='__main__':main()
