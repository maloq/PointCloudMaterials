"""Linked, source-bound static structure exploration over frozen encoder clusters."""
import argparse
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import sph_harm_y

from src.project_runtime.paths import load_json, resolve_path
from src.experiment_runner.artifacts import write_json
from src.experiment_runner.result_records import file_hash, location
from src.experiment_runner.metric_docs import check_metric_docs, snapshot_metric_docs, write_metric_table
from .liquid_structure import bond_order, ORDER_NAMES
from .representative_structures import _build_ovito_data_collection


def table(root, name, rows):
    contract = snapshot_metric_docs(root, 'cluster_explorer')
    target = root/'tables'/f'{name}.csv'
    pd.DataFrame(rows).to_csv(target,index=False)
    write_json(root/'technical/table-contracts'/f'{name}.json',dict(table=f'tables/{name}.csv',
        sha256=file_hash(target),family='cluster_explorer',contract=str(contract.relative_to(root)),
        definitions='tables/metric-definitions/cluster_explorer.md'))


def local_order(points, tree, centers):
    parts=[]
    for start in range(0,len(centers),256):
        ids=tree.query(centers[start:start+256],k=13,workers=4)[1]
        neighbors=tree.query(points[ids].reshape(-1,3),k=13,workers=4)[1][:,1:]
        vectors=(points[neighbors]-points[ids].reshape(-1,1,3)).reshape(-1,13,12,3)
        parts.append(bond_order(vectors,3.7)[0])
    return np.concatenate(parts)


def q6_units(points, tree, atoms):
    result=[]
    for start in range(0,len(atoms),2048):
        centers=points[atoms[start:start+2048]]
        ids=tree.query(centers,k=13,workers=4)[1][:,1:]
        vectors=points[ids]-centers[:,None];r=np.linalg.norm(vectors,axis=-1)
        theta=np.arccos(np.clip(vectors[...,2]/r,-1,1));phi=np.arctan2(vectors[...,1],vectors[...,0])
        q=np.stack([sph_harm_y(6,m,theta,phi).mean(1) for m in range(-6,7)],-1)
        norm=np.linalg.norm(q,axis=-1)
        if np.any(norm<1e-14):raise ValueError('Undefined q6 orientation in radial sample')
        result.append(q/norm[:,None])
    return np.concatenate(result)


def atlas_selection(samples, rng, per_band):
    rows=[]
    for cluster, group in samples.groupby('cluster',sort=True):
        # Stable rank thirds also work for highly tied crystalline samples.
        ranked=group.sort_values(['mean_q6_coherence','sample_index'])
        for band,ids in zip(('low','middle','high'),np.array_split(ranked.index,3),strict=True):
            chosen=rng.choice(ids,min(per_band,len(ids)),replace=False)
            for index in sorted(chosen):
                row=samples.loc[index]
                rows.append(dict(cluster=int(cluster),band=band,sample_index=int(row.sample_index),
                    frame=str(row.frame),mean_q6_coherence=float(row.mean_q6_coherence)))
    return rows


def compute(c):
    check_metric_docs(family='cluster_explorer')
    run=Path(c['source_run']);standard=run/'analyses'/c['source_analysis']/'data'
    order=run/'analyses'/c['order_analysis'];root=run/'analyses'/c['analysis_name']
    if (root/'data/metrics.json').exists():raise FileExistsError(f'Use render or a new revision: {root}')
    (root/'data').mkdir(parents=True,exist_ok=True);(root/'technical').mkdir(exist_ok=True)
    protocol=json.loads((order/'technical/protocol.json').read_text())
    receipt=json.loads((run/'run.json').read_text())
    source=next(a for a in receipt['analyses'] if a['id']==protocol['source_analysis_id'])
    source_order=next(a for a in receipt['analyses'] if a['source']==location(order/'data'))
    samples=pd.read_csv(order/'tables/local_order_samples.csv',dtype={'frame':str})
    if file_hash(order/'tables/local_order_samples.csv') != next(a['sha256'] for a in source_order['artifacts'] if a['relative']=='tables/local_order_samples.csv'):
        raise ValueError('Local-order sample table differs from published evidence')
    rng=np.random.default_rng(c['seed']);atlas=atlas_selection(samples,rng,c['atlas_per_band'])
    with np.load(standard/'analysis_inference_cache.npz') as a:
        z=a['inv_latents'];all_coords=a['coords']
    if z.shape!=(len(all_coords),128) or not np.isfinite(z).all():raise ValueError('Expected finite native 128-D embeddings')
    znorm=np.linalg.norm(z,axis=1,keepdims=True)
    if np.any(znorm==0):raise ValueError('Cosine distance undefined for zero embedding')
    z=z/znorm
    os.environ['OVITO_THREAD_COUNT']=str(c['cpu_threads'])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='.*OVITO.*PyPI')
        from ovito.modifiers import PolyhedralTemplateMatchingModifier
    pieces=[];retrieval=[];radial=[];frames=[];offset=0
    for fi,record in enumerate(protocol['frames']):
        frame=record['frame'];path=resolve_path(record['file'])
        if file_hash(path)!=record['sha256']:raise ValueError(f'Changed source: {path}')
        pop_path=order/'data'/f'population-{frame}.npz'
        expected=next(a['sha256'] for a in source_order['artifacts'] if a['relative']==pop_path.name)
        if file_hash(pop_path)!=expected:raise ValueError(f'Changed PTM population: {frame}')
        with np.load(pop_path) as a:pop={k:a[k] for k in a.files}
        coords=pop['coords'];labels=pop['clusters'];n=len(coords)
        np.testing.assert_array_equal(all_coords[offset:offset+n],coords)
        points=np.load(path).astype(np.float64);tree=cKDTree(points)
        print(f'{frame}: linked geometry, order and retrieval',flush=True)
        subset=samples[samples.frame==frame]
        spatial=np.sort(rng.choice(n,min(n,c['spatial_samples_per_frame']),replace=False))
        chosen=set(spatial.tolist())|set(subset.frame_row.astype(int))
        # Exact cosine neighbors over ALL saved centers of this same frame.
        for query in [a for a in atlas if a['frame']==frame]:
            q=query['sample_index'];local=q-offset
            similarity=np.clip(z[offset:offset+n]@z[q],-1,1)
            distances=np.linalg.norm(coords-coords[local],axis=1)
            eligible=np.flatnonzero(distances>=c['retrieval_exclusion_A'])
            best=eligible[np.argsort(-similarity[eligible],kind='stable')[:c['retrieval_neighbors']]]
            if len(best)!=c['retrieval_neighbors']:raise ValueError('Insufficient distant retrieval candidates')
            for rank,neighbor in enumerate(best,1):
                chosen.add(int(neighbor))
                retrieval.append(dict(query=q,sample_index=offset+int(neighbor),kind='embedding',rank=rank,
                    cosine_distance=float(1-similarity[neighbor]),separation_A=float(distances[neighbor]),
                    candidate_count=len(eligible),frame=frame))
            # Fixed physical calipers; absence is reported, never silently widened.
            query_order=subset[subset.sample_index==q].iloc[0]
            controls=subset[(abs(subset.mean_q6_coherence-query_order.mean_q6_coherence)<=.05)&
                (abs(subset.qbar6-query_order.qbar6)<=.04)&
                (abs(subset.density_r12/query_order.density_r12-1)<=.10)]
            ids=controls.frame_row.to_numpy(int)
            ids=ids[(distances[ids]>=c['retrieval_exclusion_A'])&~np.isin(ids,best)]
            query['matched_control_candidates']=len(ids)
            for rank,neighbor in enumerate(rng.choice(ids,min(c['retrieval_neighbors'],len(ids)),replace=False),1):
                retrieval.append(dict(query=q,sample_index=offset+int(neighbor),kind='order-matched',rank=rank,
                    cosine_distance=float(1-similarity[neighbor]),separation_A=float(distances[neighbor]),
                    candidate_count=len(ids),frame=frame))
        ids=np.array(sorted(chosen),int);selected_coords=coords[ids]
        values=local_order(points,tree,selected_coords)
        order_lookup={int(row):i for i,row in enumerate(ids)}
        for row in subset.itertuples():
            np.testing.assert_allclose(values[order_lookup[row.frame_row]],np.array([getattr(row,k) for k in ORDER_NAMES]),rtol=1e-6,atol=1e-6)
        data=_build_ovito_data_collection(points)
        modifier=PolyhedralTemplateMatchingModifier(rmsd_cutoff=0,output_rmsd=True)
        for structure in modifier.structures:structure.enabled=int(structure.id) in [1,2,3,4]
        data.apply(modifier)
        raw=np.asarray(data.particles['Structure Type'],np.int8).copy()
        rmsd=np.asarray(data.particles['RMSD'],np.float32).copy()
        np.testing.assert_array_equal(raw[pop['source_atom_rows']],pop['ptm_best_type'])
        np.testing.assert_allclose(rmsd[pop['source_atom_rows']],pop['ptm_rmsd'],rtol=1e-5,atol=1e-6)
        nearest=tree.query(selected_coords,k=64,workers=4)[1]
        pieces.append(dict(sample_index=offset+ids,frame=np.full(len(ids),fi,np.int16),clusters=labels[ids],
            coords=selected_coords,source_atom_rows=pop['source_atom_rows'][ids],
            spatial=np.isin(ids,spatial),stratified=np.isin(ids,subset.frame_row.to_numpy()),
            order=values,points=(points[nearest]-selected_coords[:,None]).astype(np.float32),
            neighbor_type=raw[nearest],neighbor_rmsd=rmsd[nearest],neighbor_atoms=nearest,
            ptm_best_type=raw[nearest[:,0]],ptm_rmsd=rmsd[nearest[:,0]]))
        # Radial statistics use full physical spheres, independent of the 64-atom display.
        balls=tree.query_ball_point(coords[subset.frame_row.to_numpy(int)],c['radial_edges_A'][-1],workers=4)
        atoms=np.unique(np.concatenate(balls));units=q6_units(points,tree,atoms)
        lookup=np.full(len(points),-1,int);lookup[atoms]=np.arange(len(atoms))
        for row,neighbors in zip(subset.itertuples(),balls,strict=True):
            neighbors=np.array(neighbors);center=int(row.source_atom_row)
            radius=np.linalg.norm(points[neighbors]-points[center],axis=1)
            alignment=(units[lookup[neighbors]]@units[lookup[center]].conj()).real
            for lo,hi in zip(c['radial_edges_A'][:-1],c['radial_edges_A'][1:],strict=True):
                mask=(radius>lo)&(radius<=hi);nn=int(mask.sum())
                crystalline=np.isin(raw[neighbors[mask]],[1,2,3])&(rmsd[neighbors[mask]]<=.1)
                radial.append(dict(frame=frame,cluster=int(row.cluster),sample_index=int(row.sample_index),
                    inner_A=lo,outer_A=hi,atoms=nn,crystalline_fraction=float(crystalline.mean()) if nn else None,
                    q6_alignment=float(alignment[mask].mean()) if nn else None))
        frames.append(dict(frame=frame,source_sha256=record['sha256'],population=n,
            displayed_uniform=len(spatial),inspectable=len(ids),stratified=len(subset),
            population_sha256=file_hash(pop_path)))
        offset+=n
        write_json(root/'technical/progress.json',dict(state='running',frames=frames))
        del data,tree,points,units
    joined={k:np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
    np.savez_compressed(root/'data/inspection.npz',**joined)
    radial_df=pd.DataFrame(radial);profiles=[]
    for (cluster,lo,hi),rows in radial_df.groupby(['cluster','inner_A','outer_A']):
        for measure in ('crystalline_fraction','q6_alignment'):
            values=rows[measure].dropna()
            profiles.append(dict(cluster=int(cluster),inner_A=lo,outer_A=hi,measure=measure,
                neighborhoods=len(values),mean=float(values.mean()) if len(values) else None,
                q25=float(values.quantile(.25)) if len(values) else None,
                q75=float(values.quantile(.75)) if len(values) else None))
    table(root,'radial_neighborhoods',radial);table(root,'radial_profiles',profiles)
    table(root,'embedding_neighbors',retrieval);table(root,'atlas_samples',atlas)
    inspect_rows=[]
    for i,sample in enumerate(joined['sample_index']):
        inspect_rows.append(dict(sample_index=int(sample),frame=frames[int(joined['frame'][i])]['frame'],
            cluster=int(joined['clusters'][i]),source_atom_row=int(joined['source_atom_rows'][i]),
            spatial=bool(joined['spatial'][i]),stratified=bool(joined['stratified'][i]),
            **{key:float(joined['order'][i,j]) for j,key in enumerate(ORDER_NAMES)}))
    table(root,'inspection_samples',inspect_rows)
    metrics=dict(population=offset,inspectable=len(joined['sample_index']),spatial=int(joined['spatial'].sum()),
        radial_neighborhoods=len(samples),atlas_samples=len(atlas),retrieval_queries=len(atlas),
        queries_without_matched_controls=sum(a['matched_control_candidates']==0 for a in atlas))
    write_json(root/'data/atlas.json',atlas);write_json(root/'data/metrics.json',metrics)
    write_metric_table(metrics,root,family='cluster_explorer',name='scores')
    write_json(root/'technical/protocol.json',dict(config=c,frames=frames,source_analysis_id=source['id'],
        order_analysis_id=source_order['id'],checkpoint_sha256=protocol['checkpoint_sha256'],
        encoder_cache_sha256=file_hash(standard/'analysis_inference_cache.npz'),
        inspection_sha256=file_hash(root/'data/inspection.npz'),
        inputs=dict(encoder='Retained native 128-D embeddings; no inference or refitting',predictor=[],
            diagnostic='Full source geometry; no time, temperature or labels as model inputs'),
        interpretation='Descriptive relaxed static observations. Cluster numbers are specific to this fit. No independent-sample confidence intervals or claims about future crystallization.',
        sampling='Spatial views: uniform without replacement within frame. Scatter/atlas/radial: original equal-count frame×cluster stratification. Retrieval: all saved centers of the query frame, excluding distance <20 Å.',
        radial='Physical shells, central atom excluded. Equal neighborhood weighting across stratified samples; bands are neighborhood interquartile ranges, not confidence intervals. q6 alignment is focal-to-shell, signed.',
        controls='Uniform draws among stratified same-frame samples within coherence ±0.05, qbar6 ±0.04, density ±10%; separation >=20 Å; exclude retrieved neighbors. No adaptive calipers.',
        atlas='Four samples without replacement per stable rank third of neighbor q6 coherence within each cluster. Cluster population is the retained stratified sample.'))
    write_json(root/'technical/progress.json',dict(state='complete',metrics=metrics))
    print(json.dumps(metrics,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=['compute','render'],required=True);a=p.parse_args();c=load_json(a.config)
    if a.stage=='compute':compute(c)
    else:
        from .cluster_explorer_vis import render
        render(c)

if __name__=='__main__':main()
