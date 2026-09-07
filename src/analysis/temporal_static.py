"""Full saved-center static analysis of a selected temporal-campaign encoder."""
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import adjusted_rand_score
import torch

from src.analysis.liquid_structure import bond_order, persistence_image
from src.data_utils.temporal_campaign import ROOT, write_json
from src.training_methods.temporal_campaign import Learner
from src.vis_tools.latent_analysis_vis import compute_kmeans_labels
from src.vis_tools.md_cluster_plot import save_interactive_md_plot


def geometry(cloud):
    """Same local, nonperiodic patch assay as the liquid-SRO benchmark producer."""
    tree=cKDTree(cloud)
    _,neighbors=tree.query(cloud[:13],k=13)
    vectors=cloud[neighbors[:,1:]]-cloud[:13,None]
    order,connections=bond_order(vectors[None],3.7)
    return order[0],connections[0],persistence_image(cloud[:65])


@torch.inference_mode()
def run(campaign,cfg,out):
    started=time.monotonic()
    campaign_out=ROOT/campaign['output']
    selected=json.loads((campaign_out/'selected_runs.json').read_text())
    trial=min((r for r in selected if r['name']==cfg['hypothesis']),key=lambda r:r['selection_score'])
    key=f"{trial['stage']}_{trial['name']}_seed{trial['seed']}"
    checkpoint=campaign_out/'checkpoints'/f'{key}.pt'
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=Learner(trial['hypothesis'],campaign).cuda().eval()
    model.load_state_dict(payload['model'])
    write_json(out/'checkpoint.json',dict(path=str(checkpoint),sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        selection='Lowest pre-existing balanced validation motion MSE among confirmed density_predictive seeds; no static-data selection.',training=trial,
        representation='Online 128D representation including its learned output map and LayerNorm; no forecast, motion head or EMA teacher.'))
    cache=ROOT/cfg['centers_cache'];metadata=json.loads((cache/'metadata.json').read_text())
    reference=ROOT/cfg['reference_static'];coverage=json.loads((reference/'coverage.json').read_text())
    old_coords=np.load(reference/'coords.npy',mmap_mode='r');old_meta=np.load(reference/'metadata.npz')
    n=metadata['total_samples'];points_count=model.representation.points
    latents=open_memmap(out/'embeddings.npy',mode='w+',dtype=np.float32,shape=(n,campaign['latent_dim']))
    coords=open_memmap(out/'coords.npy',mode='w+',dtype=np.float32,shape=(n,3))
    frame_ids=np.empty(n,np.int16);frames=[];offset=0;assay_rows=[];assay_clouds=[]
    rng=np.random.default_rng(cfg['seed'])
    for frame_id,shard in enumerate(metadata['shards']):
        start_time=time.monotonic();centers=np.load(cache/shard['coords_path'])
        source=next(s for s in metadata['request']['sources'] if s['name']==shard['source'])
        path=ROOT/source['files'][0]['path'];sha=hashlib.sha256(path.read_bytes()).hexdigest()
        previous=coverage['frames'][frame_id]
        if previous['source']!=shard['source'] or previous['source_sha256']!=sha:
            raise ValueError(f'Static source changed since reference analysis: {path}')
        if not np.array_equal(centers,old_coords[offset:offset+len(centers)]):
            raise ValueError(f'Analysis centers differ from reference: {path}')
        points=np.load(path);tree=cKDTree(points,balanced_tree=False)
        chosen=np.sort(rng.choice(len(centers),cfg['assay_per_frame'],replace=False));margin=float('inf')
        for start in range(0,len(centers),cfg['batch_size']):
            c=centers[start:start+cfg['batch_size']]
            distance,ids=tree.query(c,k=points_count+1,workers=cfg['workers'])
            if distance[:,0].max()>1e-5:raise ValueError(f'Saved centers do not match source atoms: {path}/{start}')
            margin=min(margin,float(distance[:,-1].min()-8.))
            if margin<=0:raise ValueError(f'Nearest-{points_count} patch truncates the 8 Angstrom density support: {path}/{start}, margin={margin}')
            cloud=(points[ids[:,:points_count]]-c[:,None]).astype(np.float32)
            x=torch.tensor(cloud,device='cuda');material=torch.zeros(len(x),dtype=torch.long,device='cuda')
            z=model.representation(x,material).cpu().numpy()
            if not np.isfinite(z).all():raise FloatingPointError(f'Nonfinite embeddings: {path}/{start}')
            latents[offset+start:offset+start+len(c)]=z
            take=chosen[(chosen>=start)&(chosen<start+len(c))]
            assay_rows.extend((offset+take).tolist());assay_clouds.extend(cloud[take-start])
        coords[offset:offset+len(centers)]=centers;frame_ids[offset:offset+len(centers)]=frame_id
        frames.append(dict(source=shard['source'],offset=offset,count=len(centers),path=str(path),source_sha256=sha,
            excluded_neighbor_margin_A=margin,elapsed_seconds=time.monotonic()-start_time))
        offset+=len(centers);latents.flush();coords.flush()
        write_json(out/'coverage.json',dict(expected=n,processed=offset,frames=frames,matched_reference_coordinates=True))
        print('ENCODED',shard['source'],len(centers),frames[-1]['elapsed_seconds'],flush=True)
    if offset!=n or not np.array_equal(frame_ids,old_meta['source_ids']):raise ValueError('Static frame coverage/order mismatch')
    ptm=old_meta['ptm_labels'];np.savez(out/'metadata.npz',frame_ids=frame_ids,ptm_labels=ptm)
    del model,payload,x;torch.cuda.empty_cache()
    print('CLUSTERING',n,'centers',flush=True)
    clusters,info=compute_kmeans_labels(np.asarray(latents),cfg['clusters'],random_state=cfg['seed'],method='spherical_kmeans',
        standardize=True,l2_normalize=True,pca_variance=.99,pca_max_components=64,return_info=True)
    np.save(out/'clusters.npy',clusters)
    keep=('silhouette_cosine','silhouette_euclidean','davies_bouldin','calinski_harabasz',
        'cluster_validation_sample_size','cluster_counts','pca_components')
    metrics={k:info[k] for k in keep};metrics['fit_sample_count']=n
    covariance=np.cov(latents,rowvar=False);eig,vec=np.linalg.eigh(covariance);eig=np.maximum(eig,0)
    probabilities=eig/eig.sum();positive=probabilities[probabilities>0]
    metrics.update(effective_rank=float(np.exp(-(positive*np.log(positive)).sum())),largest_pc_fraction=float(probabilities[-1]),
        cluster_ptm_ari=float(adjusted_rand_score(ptm,clusters)))
    reference_metrics=json.loads((reference/'metrics.json').read_text())
    comparisons=[]
    for name,old in reference_metrics.items():
        labels=np.load(reference/f'{name}.clusters.npy')
        comparisons.append(dict(model=name,cosine_silhouette=old['silhouette_cosine'],effective_rank=old['effective_rank'],
            cluster_ptm_ari=old['cluster_ptm_ari'],ari_with_predictive_density=float(adjusted_rand_score(labels,clusters))))
    comparisons.append(dict(model=cfg['hypothesis'],cosine_silhouette=metrics['silhouette_cosine'],effective_rank=metrics['effective_rank'],
        cluster_ptm_ari=metrics['cluster_ptm_ari'],ari_with_predictive_density=1.))
    pd.DataFrame(comparisons).to_csv(out/'reference_comparison.csv',index=False)
    write_json(out/'metrics.json',metrics)
    pd.crosstab(pd.Series(frame_ids,name='frame_id'),pd.Series(clusters,name='cluster')).to_csv(out/'frame_cluster_counts.csv')
    pd.crosstab(pd.Series(clusters,name='cluster'),pd.Series(ptm,name='ptm')).to_csv(out/'cluster_ptm_counts.csv')
    print('ASSAYS',len(assay_rows),'uniformly sampled centers',flush=True)
    with ProcessPoolExecutor(max_workers=cfg['workers'],mp_context=mp.get_context('spawn')) as pool:
        assays=list(pool.map(geometry,assay_clouds,chunksize=16))
    order,connections,tda=(np.stack(v) for v in zip(*assays));rows=np.array(assay_rows)
    np.savez(out/'geometry_assay.npz',rows=rows,order=order,connections=connections,tda=tda,clusters=clusters[rows],frame_ids=frame_ids[rows],ptm=ptm[rows])
    columns=['q4','q6','w4','w6','qbar6','mean_bond_coherence','density','soft_coordination']
    profiles=pd.DataFrame(order,columns=columns)
    profiles['H1_persistence_image_mass']=tda[:,16:80].sum(1);profiles['H2_persistence_image_mass']=tda[:,80:].sum(1)
    profiles['coherent_bonds_0.70']=connections[:,1]
    profiles['cluster']=clusters[rows];profiles['frame_id']=frame_ids[rows];profiles['ptm']=ptm[rows]
    profiles.to_csv(out/'sampled_geometry.csv',index=False)
    profiles.groupby(['frame_id','cluster']).agg(['count','mean','std']).to_csv(out/'cluster_geometry_by_frame.csv')
    profiles[profiles.ptm==0].groupby(['frame_id','cluster']).agg(['count','mean','std']).to_csv(out/'ptm_other_geometry_by_frame.csv')
    # Keep plots on common cluster colors, shared across all six frames.
    palette=plt.get_cmap('tab10');mean=np.asarray(latents).mean(0);axes2=vec[:,-2:][:,::-1]
    pca=(np.asarray(latents)-mean)@axes2;np.save(out/'pca2.npy',pca.astype(np.float32))
    figure,axes=plt.subplots(2,3,figsize=(14,8),sharex=True,sharey=True)
    for ax,frame in zip(axes.flat,frames):
        ids=np.arange(frame['offset'],frame['offset']+frame['count']);chosen=rng.choice(ids,min(len(ids),10000),replace=False)
        ax.scatter(pca[chosen,0],pca[chosen,1],s=1,c=palette(clusters[chosen]),rasterized=True)
        ax.set_title(frame['source']);ax.set_xlabel('PC1');ax.set_ylabel('PC2')
    figure.suptitle('Predictive density: raw embedding PCA; colors are shared k=7 clusters')
    figure.tight_layout();figure.savefig(out/'embedding_pca.png',dpi=160);plt.close(figure)
    figure,axes=plt.subplots(2,3,figsize=(14,9))
    for ax,frame in zip(axes.flat,frames):
        section=slice(frame['offset'],frame['offset']+frame['count']);c=np.asarray(coords[section]);labels=clusters[section]
        mask=np.abs(c[:,2]-np.median(c[:,2]))<cfg['slice_half_width_A']
        ax.scatter(c[mask,0],c[mask,1],c=palette(labels[mask]),s=4,rasterized=True)
        ax.set(title=frame['source'],xlabel='x (Angstrom)',ylabel='y (Angstrom)',aspect='equal')
        folder=out/'snapshots'/Path(frame['source']).stem/'md_space';folder.mkdir(parents=True)
        np.savez_compressed(folder/'local_structure_coords_clusters.npz',coords=c,clusters=labels)
        save_interactive_md_plot(c,labels,folder/'clusters_3d.html',max_points=cfg['html_max_points'],
            title=f"Predictive density: {frame['source']}; sampled display, full labels in NPZ")
    figure.suptitle('Central slices: shared cluster IDs, no assigned phase identities')
    figure.tight_layout();figure.savefig(out/'spatial_slices.png',dpi=160);plt.close(figure)
    features=['q6','qbar6','mean_bond_coherence','H1_persistence_image_mass','H2_persistence_image_mass']
    figure,axes=plt.subplots(1,len(features),figsize=(16,4))
    for ax,feature in zip(axes,features):
        ax.boxplot([profiles.loc[profiles.cluster==k,feature] for k in range(cfg['clusters'])],showfliers=False)
        ax.set(title=feature,xticks=range(1,cfg['clusters']+1),xticklabels=range(cfg['clusters']),xlabel='Cluster')
    figure.suptitle('Continuous geometry on a uniform sample of 512 centers/frame; pooled across frames')
    figure.tight_layout();figure.savefig(out/'geometry_profiles.png',dpi=160);plt.close(figure)
    summary=f'''# Predictive density: full static Al

Encoded and clustered all {n:,} saved regular-grid centers across six snapshots.
These are the existing analysis centers with boundary exclusions, not every atom.
Selected checkpoint: `{key}`, chosen by pre-existing validation motion score.
Inputs are the center plus 192 nearest atoms in physical Angstrom, with the
training density cutoff of 8 Angstrom; excluded-neighbor support was verified.
Coordinates, source hashes and frame order match the previous full-static assay.

Spherical k-means k=7 uses every embedding with standardization, row normalization
and the repository's 99%-variance PCA (at most 64 components). Internal clustering
scores use the fixed {metrics['cluster_validation_sample_size']:,}-row diagnostic subset.
Cosine silhouette: **{metrics['silhouette_cosine']:.4f}**; effective rank:
**{metrics['effective_rank']:.2f}/128**; largest raw PC: **{100*metrics['largest_pc_fraction']:.1f}%**.
Cluster/PTM adjusted Rand index: **{metrics['cluster_ptm_ari']:.4f}**.

PTM is an imperfect geometric reference, not ground-truth phase identity. No PTM
classifier was trained. Continuous bond-order and alpha-persistence descriptors
use {len(rows):,} uniformly sampled centers (512/frame), not the complete dataset.
Sample counts and per-frame/per-cluster distributions are retained, including
PTM-Other subsets. Cluster separation or different TDA profiles alone does not
establish a new motif or future crystallization propensity. Static snapshots
provide no temporal forecast validation; no artificial lag was supplied.

The older comparison rows reuse the same centers and clustering settings but
different representations/training. Silhouette scores across different latent
spaces are descriptive, not a common supervised accuracy metric.

![Spatial slices](spatial_slices.png)
![Embedding PCA](embedding_pca.png)
![Continuous geometry](geometry_profiles.png)

[Metrics](metrics.json), [reference comparison](reference_comparison.csv),
[frame populations](frame_cluster_counts.csv), [PTM contingency](cluster_ptm_counts.csv),
[sampled geometry](sampled_geometry.csv), [within-frame profiles](cluster_geometry_by_frame.csv)
and [PTM-Other profiles](ptm_other_geometry_by_frame.csv) provide numerical results.
Full embeddings, coordinates and labels are NPY files in this directory.
Interactive displays subsample at most {cfg['html_max_points']:,} centers per frame;
the adjacent NPZ files contain every saved center. The renderer uses Plotly's CDN.

'''
    for frame in frames:
        name=Path(frame['source']).stem;summary+=f'- [{name} interactive 3D](snapshots/{name}/md_space/clusters_3d.html)\n'
    (out/'RESULTS.md').write_text(summary)
    print('COMPLETE',n,'centers',time.monotonic()-started,'seconds',flush=True)


def main(campaign,config_path):
    cfg=json.loads(config_path.read_text());out=ROOT/cfg['output'];out.mkdir(parents=True,exist_ok=False)
    write_json(out/'config.json',cfg)
    write_json(out/'provenance.json',dict(source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ('src/analysis/temporal_static.py','src/training_methods/temporal_campaign.py','src/models/encoders/atomic_graph.py','src/models/encoders/smooth_density.py')}))
    status=dict(state='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat());write_json(out/'status.json',status)
    try:
        run(campaign,cfg,out);status.update(state='complete',finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:write_json(out/'status.json',status)
