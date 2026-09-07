"""Matched spatial coherence diagnosis on the six repository static Al frames."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.vis_tools.latent_analysis_vis import compute_kmeans_labels


def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    roots={k:Path(v) for k,v in cfg['analyses'].items()}
    raw_cache=np.load(roots['density_raw']/'analysis_inference_cache.npz')
    old_cache=np.load(roots['old_geoframe_projector']/'analysis_inference_cache.npz')
    coords=raw_cache['coords'];old_coords=old_cache['coords']
    z=raw_cache['inv_latents']
    metrics=json.loads((roots['density_raw']/'analysis_metrics.json').read_text())
    frames=metrics['real_md_qualitative']['frames']
    labels={name:np.concatenate([np.load(root/'snapshots'/f['output_name']/'md_space/local_structure_coords_clusters.npz')['clusters'] for f in frames]) for name,root in roots.items()}
    projected=Path(cfg['projected_analysis'])
    if not np.array_equal(coords,np.load(projected/'coords.npy')):raise ValueError('Projected analysis coordinates do not match')
    labels['density_projected']=np.load(projected/'clusters.npy')
    # One explicit clustering-only control: retain eight PCs as in the older run.
    labels['density_raw_pca8'],info=compute_kmeans_labels(z,7,random_state=123,method='spherical_kmeans',
        standardize=True,l2_normalize=True,pca_variance=1.,pca_max_components=8,return_info=True)
    np.save(out/'density_raw_pca8_clusters.npy',labels['density_raw_pca8'])
    ptm=np.load(projected/'metadata.npz')['ptm_labels']
    features={'old_geoframe_projector':old_cache['inv_latents'],'density_raw':z,
        'density_projected':np.load(projected/'embeddings.npy',mmap_mode='r')}
    results=[];feature_results=[];alignment=[];offset=0;rng=np.random.default_rng(20260907)
    for frame in frames:
        total=frame['num_samples'];sl=slice(offset,offset+total)
        distance,old_indices=cKDTree(old_coords[sl]).query(coords[sl])
        common=np.flatnonzero(distance<1e-6);old_indices=old_indices[common]
        if len(np.unique(old_indices))!=len(common):raise ValueError('Shared-center alignment is not one-to-one')
        c=coords[sl][common];n=len(common)
        alignment.append(dict(frame=frame['output_name'],original_centers=total,shared_centers=n,max_alignment_error=float(distance[common].max())))
        _,near=cKDTree(c).query(c,k=7,workers=4);near=near[:,1:]
        other=ptm[sl][common]==0;inside=other[:,None]&other[near]
        for name,y in labels.items():
            y=y[sl][old_indices if name=='old_geoframe_projector' else common]
            for region,mask,edge_mask in [('all',np.ones(n,dtype=bool),np.ones(near.shape,dtype=bool)),('PTM_other',other,inside)]:
                a=np.broadcast_to(y[:,None],near.shape)[edge_mask];b=y[near][edge_mask]
                agreement=float((a==b).mean())
                # Endpoint marginals account for degree and region-boundary selection.
                chance=float(np.dot(np.bincount(a,minlength=7)/len(a),np.bincount(b,minlength=7)/len(b)))
                results.append(dict(frame=frame['output_name'],model=name,region=region,centers=int(mask.sum()),edges=len(a),
                    same_label_fraction=agreement,chance_fraction=chance,excess_agreement=(agreement-chance)/(1-chance)))
        sample=rng.choice(n,min(10000,n),replace=False);random_neighbors=rng.integers(n,size=(len(sample),6))
        for name,value in features.items():
            v=np.asarray(value[sl])[old_indices if name=='old_geoframe_projector' else common];std=np.maximum(v.std(0),1e-3)
            local=np.square((v[sample,None]-v[near[sample]])/std).mean()
            shuffled=np.square((v[sample,None]-v[random_neighbors])/std).mean()
            feature_results.append(dict(frame=frame['output_name'],model=name,neighbor_mse_over_random=float(local/shuffled)))
        offset+=total
    pd.DataFrame(results).to_csv(out/'spatial_label_coherence.csv',index=False)
    pd.DataFrame(feature_results).to_csv(out/'continuous_spatial_variation.csv',index=False)
    summary={name:json.loads((root/'analysis_metrics.json').read_text())['clustering']['cluster_fit_info_by_k']['7'] for name,root in roots.items()}
    summary['density_raw_pca8']=info
    (out/'clustering_metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps(dict(config=cfg,matched_centers=sum(a['shared_centers'] for a in alignment),alignment=alignment,neighbors=6,
        label_metric='Directed six-nearest-center graph within each snapshot; adjusted for endpoint label marginals. PTM_other includes edges with both endpoints labeled Other by the existing imperfect assay.',
        continuous_metric='Mean standardized squared difference to six spatial neighbors / random same-frame neighbors, on 10,000 fixed centers per frame. Smaller means greater spatial coherence, not automatically better physics.',
        limitations='No independent phase labels; seven clusters remain imposed. Same sources and correlated neighborhoods; this is descriptive diagnosis, not a causal architecture experiment.'),indent=2)+'\n')
