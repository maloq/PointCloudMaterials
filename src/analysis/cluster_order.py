"""Full-source PTM and stratified local-order diagnostics over saved clusters."""
import argparse
import json
import os
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.experiment_runner.artifacts import write_json
from src.experiment_runner.metric_docs import check_metric_docs, snapshot_metric_docs, write_metric_table
from src.experiment_runner.result_records import file_hash, location
from src.project_runtime.paths import load_json, resolve_path
from .liquid_structure import bond_order, ORDER_NAMES
from .representative_structures import _build_ovito_data_collection

TYPE_NAMES = {0:'Other', 1:'FCC', 2:'HCP', 3:'BCC', 4:'ICO'}


def classify(types, rmsd, cutoff):
    return np.where((types != 0) & (rmsd <= cutoff), types, 0).astype(np.int8)


def export_rows(root, name, rows):
    contract = snapshot_metric_docs(root, 'cluster_order')
    path = root/'tables'/f'{name}.csv'
    pd.DataFrame(rows).to_csv(path,index=False)
    write_json(root/'technical/table-contracts'/f'{name}.json', dict(
        table=f'tables/{name}.csv',sha256=file_hash(path),family='cluster_order',
        contract=str(contract.relative_to(root)),definitions='tables/metric-definitions/cluster_order.md'))


def compute(config, config_path):
    check_metric_docs(family='cluster_order')
    run = Path(config['source_run'])
    source = run/'analyses'/config['source_analysis']/'data'
    root = run/'analyses'/config['analysis_name']
    if (root/'data/metrics.json').exists():
        raise FileExistsError(f'Completed order analysis exists: {root}; use render or a new analysis name')
    (root/'data').mkdir(parents=True,exist_ok=True)
    (root/'technical').mkdir(exist_ok=True)
    os.environ['OVITO_THREAD_COUNT'] = str(config['cpu_threads'])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='.*OVITO.*PyPI')
        import ovito
        from ovito.modifiers import PolyhedralTemplateMatchingModifier
    metadata = json.loads((source/'structural-inference-protocol.json').read_text())
    summary = json.loads((source/'real_md/summary.json').read_text())
    reps = json.loads((source/'real_md/representatives/10_cluster_representatives_structure_analysis_k7.json').read_text())['representatives']
    source_record = json.loads((run/'run.json').read_text())
    source_analysis = next(a for a in source_record['analyses'] if a['source']==location(source))
    assignment_hashes = {a['relative']:a['sha256'] for a in source_analysis['artifacts']}
    rng = np.random.default_rng(config['seed'])
    source_receipts, frame_rows, sample_rows, population_parts, representatives = [], [], [], [], []
    offset = 0
    for frame_index,(frame,frame_summary) in enumerate(zip(metadata['frames'],summary['frames'],strict=True)):
        name = Path(frame['file']).stem
        if name != frame_summary['output_name']:
            raise ValueError('Source protocol and analysis frame order differ')
        path = resolve_path(frame['file'])
        if file_hash(path) != frame['source_sha256']:
            raise ValueError(f'Source coordinates changed: {path}')
        rel = f'snapshots/{name}/md_space/local_structure_coords_clusters.npz'
        if file_hash(source/rel) != assignment_hashes[rel]:
            raise ValueError(f'Saved assignments changed: {source/rel}')
        with np.load(source/rel) as assigned:
            centers, labels = assigned['coords'], assigned['clusters']
        if len(centers) != frame['centers'] or len(centers) != frame_summary['num_samples']:
            raise ValueError(f'Frame population mismatch: {name}')
        points = np.load(path).astype(np.float64)
        tree = cKDTree(points)
        distance, atom_rows = tree.query(centers,workers=config['cpu_threads'])
        if np.any(distance != 0):
            raise ValueError(f'Centers are not exact source atoms: {name}, maximum error {distance.max()}')
        print(f'{name}: PTM on all {len(points):,} source atoms; {len(centers):,} saved centers',flush=True)
        data = _build_ovito_data_collection(points)
        modifier = PolyhedralTemplateMatchingModifier(rmsd_cutoff=0,output_rmsd=True)
        for structure in modifier.structures:
            structure.enabled = int(structure.id) in config['ptm_types']
        data.apply(modifier)
        raw_type = np.asarray(data.particles['Structure Type'],dtype=np.int8).copy()
        rmsd = np.asarray(data.particles['RMSD'],dtype=np.float32).copy()
        if not np.isfinite(rmsd).all():
            raise ValueError(f'Nonfinite PTM residuals: {name}')
        classified = classify(raw_type,rmsd,config['ptm_rmsd_cutoff'])
        local_type,local_rmsd,local_raw = classified[atom_rows],rmsd[atom_rows],raw_type[atom_rows]
        np.savez_compressed(root/'data'/f'population-{name}.npz',
            source_atom_rows=atom_rows,coords=centers,clusters=labels,
            ptm_type=local_type,ptm_best_type=local_raw,ptm_rmsd=local_rmsd)
        population_parts.append((labels,local_raw,local_rmsd))
        chosen=[]
        for cluster in range(summary['selected_k']):
            members = np.flatnonzero(labels==cluster)
            selected = np.sort(rng.choice(members,min(len(members),config['samples_per_cluster_frame']),replace=False))
            chosen.extend(selected.tolist())
            for cutoff in config['ptm_sensitivity_cutoffs']:
                types = classify(local_raw[members],local_rmsd[members],cutoff)
                for code,motif in TYPE_NAMES.items():
                    count = int(np.sum(types==code))
                    frame_rows.append(dict(frame=name,cluster=cluster,ptm_rmsd_cutoff=cutoff,
                        motif=motif,count=count,total=len(members),fraction=count/len(members) if len(members) else None))
        chosen=np.array(chosen,dtype=int)
        nearest = tree.query(centers[chosen],k=13,workers=config['cpu_threads'])[1]
        near2 = tree.query(points[nearest].reshape(-1,3),k=13,workers=config['cpu_threads'])[1][:,1:]
        vectors=(points[near2]-points[nearest].reshape(-1,1,3)).reshape(-1,13,12,3)
        observables,coherence_counts = bond_order(vectors,config['coordination_radius_A'])
        for i,row in enumerate(chosen):
            sample_rows.append(dict(frame=name,frame_index=frame_index,cluster=int(labels[row]),
                sample_index=offset+int(row),frame_row=int(row),source_atom_row=int(atom_rows[row]),
                ptm_type=TYPE_NAMES[int(local_type[row])],ptm_best_type=TYPE_NAMES[int(local_raw[row])],
                ptm_rmsd=float(local_rmsd[row]) if local_raw[row] else None,
                **{key:float(observables[i,j]) for j,key in enumerate(ORDER_NAMES)},
                coherent_bonds_065=int(coherence_counts[i,0]),coherent_bonds_070=int(coherence_counts[i,1]),
                coherent_bonds_075=int(coherence_counts[i,2])))
        for rep in reps:
            index = rep['sample_index']-offset
            if not 0 <= index < len(centers):
                continue
            if labels[index] != rep['cluster_id']:
                raise ValueError(f'Representative cluster differs from saved assignment: {rep["sample_index"]}')
            ids = tree.query(centers[index],k=config['display_atoms'])[1]
            local = points[ids]-centers[index]
            physical_shell = np.linalg.norm(local,axis=1)[rep['cna']['shell_indices']]
            saved_shell = np.array(rep['cna']['shell_distances'])
            scale = float(np.median(physical_shell/saved_shell))
            if not np.allclose(physical_shell,scale*saved_shell,rtol=2e-5,atol=1e-5):
                raise ValueError('Source neighborhood does not reproduce the retained representative shell')
            representatives.append(dict(cluster_id=rep['cluster_id'],sample_index=rep['sample_index'],frame=name,
                source_atom_rows=ids.tolist(),points_A=local.tolist(),ptm_type=classified[ids].tolist(),
                ptm_best_type=raw_type[ids].tolist(),ptm_rmsd=rmsd[ids].tolist(),
                center_cutoff_A=scale*rep['cna']['cutoff'],
                historical_shell_indices=rep['cna']['shell_indices'],historical_display_scale_A=scale,
                ptm_support='full source snapshot before display cropping'))
        source_receipts.append(dict(frame=name,file=location(path),sha256=frame['source_sha256'],
            assignments=location(source/rel),assignments_sha256=file_hash(source/rel),
            centers=len(centers),unique_center_atoms=len(np.unique(atom_rows)),order_sample_count=len(chosen)))
        offset += len(centers)
        write_json(root/'technical/progress.json',dict(state='running',completed_frames=len(source_receipts),centers=offset))
        print(f'{name}: complete; {len(chosen)} local-order samples',flush=True)
        del data,tree,points,raw_type,rmsd,classified
    if len(representatives)!=summary['selected_k']:
        raise ValueError('Not all original representatives were recovered')
    representatives.sort(key=lambda row:row['cluster_id'])
    pooled=[]
    labels,raw,rmsd=(np.concatenate(items) for items in zip(*population_parts,strict=True))
    metrics=dict(total_centers=len(labels),local_order_samples=len(sample_rows),clusters={})
    for cluster in range(summary['selected_k']):
        members=labels==cluster
        types=classify(raw[members],rmsd[members],config['ptm_rmsd_cutoff'])
        metrics['clusters'][f'C{cluster+1}']=dict(centers=int(members.sum()),
            crystalline_fraction=float(np.mean(np.isin(types,[1,2,3]))),
            ico_fraction=float(np.mean(types==4)),other_fraction=float(np.mean(types==0)))
        for code,motif in TYPE_NAMES.items():
            count=int(np.sum(types==code))
            pooled.append(dict(cluster=cluster,motif=motif,count=count,total=int(members.sum()),fraction=count/int(members.sum())))
    write_json(root/'data/representatives.json',representatives)
    write_json(root/'data/metrics.json',metrics)
    write_json(root/'technical/protocol.json',dict(config=config,source_analysis_id=source_analysis['id'],
        source_metrics_sha256=file_hash(source/'analysis_metrics.json'),frames=source_receipts,
        checkpoint_sha256=source_analysis['checkpoint_sha256'],ovito_version=ovito.version_string,
        inputs=dict(encoder='No new encoding; reuse saved cluster assignments',predictor=[],
            diagnostic='Physical source positions with full spatial support; no labels or context covariates'),
        sampling='Uniform without replacement within each frame and cluster, fixed seed; tiny strata use all members.',
        limitations='Static relaxed snapshots, overlapping neighborhoods and related frames. Descriptive population, no iid confidence intervals. PTM Other is not a liquid label. ICO is local fivefold order, not counted as crystalline. No encoder inference, fitting or cluster refitting.'))
    write_metric_table(metrics,root,family='cluster_order',name='scores')
    export_rows(root,'ptm_by_frame',frame_rows)
    export_rows(root,'ptm_by_cluster',pooled)
    export_rows(root,'local_order_samples',sample_rows)
    write_json(root/'technical/progress.json',dict(state='complete',centers=len(labels),order_samples=len(sample_rows)))
    print(json.dumps(metrics,indent=2),flush=True)
    return root


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--stage',choices=['compute','render'],required=True)
    args=parser.parse_args();config=load_json(args.config)
    if args.stage=='compute':
        compute(config,args.config)
    else:
        from .cluster_order_vis import render
        render(config)


if __name__=='__main__':main()
