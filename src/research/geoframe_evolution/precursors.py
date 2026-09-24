"""Dense, reviewable structured-liquid regions, explicitly without future-fate labels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.special import sph_harm_y
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from src.data.static_sources import load_points
from src.experiment_runner.metric_docs import write_metric_table
from .reference import bond_descriptors, contexts, write_json


def run(reference_output, output):
    root=Path(output); reference=Path(reference_output)/'technical/reference'
    for part in ('plots','tables','technical'):
        (root/part).mkdir(parents=True,exist_ok=True)
    manifest=json.loads((reference/'manifest.json').read_text())
    destination=root/'technical/structured-liquid-regions';destination.mkdir(exist_ok=True)
    records={}
    for frame_index in (4,7):
        record=manifest['frames'][frame_index]
        material=record['material']; radius=record['radius_A']
        points=load_points(record['file']);tree=cKDTree(points,balanced_tree=False)
        full=np.load(reference/f'full-reference-{frame_index:02d}.npz')
        ptm=np.where(full['rmsd']<=.1,full['best_ptm'],0)
        solid=np.isin(ptm,[1,2,3])
        if not solid.any():
            raise ValueError(f'No crystalline atoms in chosen mixed reference {record["file"]}')
        crystal_tree=cKDTree(points[solid],balanced_tree=False)
        a=np.load(reference/f'frame-{frame_index:02d}.npz');n=record['anchor_count']
        distance,_=crystal_tree.query(a['coords'][:n],workers=2)
        possible=np.flatnonzero((a['context'][:n,1]==6)&(distance>2*radius))
        ranking=possible[np.argsort(-a['order'][possible,1,4],kind='stable')]
        chosen=[]
        for j in ranking:
            if all(np.linalg.norm(a['coords'][j]-a['coords'][old])>4*radius for old in chosen):
                chosen.append(int(j))
            if len(chosen)==6:break
        if not chosen:
            raise ValueError(f'No isolated ordered-liquid candidates in {record["file"]}; selection must be reported empty.')
        frames=[]
        fig,axes=plt.subplots(2,3,figsize=(14,9),constrained_layout=True)
        for region,(seed,ax) in enumerate(zip(chosen,axes.flat)):
            center=a['coords'][seed];rows=np.array(tree.query_ball_point(center,2*radius),dtype=np.int64)
            _,near=tree.query(points[rows],k=15,workers=2)
            values=[];harmonics=[]
            for start in range(0,len(rows),256):
                core=near[start:start+256]
                _,neighbors=tree.query(points[core],k=15,workers=2)
                vectors=points[neighbors[:,:,1:]].astype(float)-points[core][:,:,None]
                values.append(bond_descriptors(vectors)[0])
                v=vectors[:,0];r=np.linalg.norm(v,axis=-1)
                theta=np.arccos(np.clip(v[...,2]/r,-1,1));phi=np.arctan2(v[...,1],v[...,0])
                q=np.stack([sph_harm_y(6,m,theta,phi).mean(-1) for m in range(-6,7)],-1)
                norm=np.linalg.norm(q,axis=-1)
                harmonics.append(np.divide(q,norm[:,None],out=np.zeros_like(q),where=norm[:,None]>1e-14))
            order=np.concatenate(values);q=np.concatenate(harmonics)
            fraction=solid[near[:,1:]].mean(1)
            labels=contexts(ptm[rows],fraction,np.zeros(len(rows),np.int8),order,
                            record['order_threshold_qbar6'],material,-.08)
            candidate=labels==6
            local={int(row):i for i,row in enumerate(rows)}
            edge_i=[];edge_j=[]
            for i in np.flatnonzero(candidate):
                for other in near[i,1:]:
                    j=local.get(int(other))
                    if j is not None and candidate[j] and np.vdot(q[i],q[j]).real>.7:
                        edge_i.append(i);edge_j.append(j)
            graph=csr_matrix((np.ones(len(edge_i)),(edge_i,edge_j)),shape=(len(rows),len(rows)))
            graph=graph.minimum(graph.T)
            _,components=connected_components(graph,directed=False)
            ids,counts=np.unique(components[candidate],return_counts=True)
            ordered=sorted(zip(ids.tolist(),counts.tolist()),key=lambda p:-p[1])
            sizes=[count for _,count in ordered]
            distance_to_crystal,_=crystal_tree.query(points[rows],workers=2)
            boundary=~np.isin(near[:,1:],rows).all(1)
            component_records=[]
            for cid,count in ordered:
                mask=candidate&(components==cid)
                component_records.append(dict(id=cid,atoms=count,mean_qbar6=float(order[mask,4].mean()),
                    mean_hat_w6=float(order[mask,3].mean()),
                    minimum_distance_to_crystal_A=float(distance_to_crystal[mask].min()),
                    touches_region_boundary=bool(boundary[mask].any()),
                    best_template_counts={str(t):int((full['best_ptm'][rows[mask]]==t).sum()) for t in (1,2,3,4)}))
            name=f'{material}-region-{region:02d}'
            np.savez(destination/f'{name}.npz',positions=points[rows],rows=rows,context=labels,
                order=order,components=components,ptm=ptm[rows],distance_to_crystal_A=distance_to_crystal)
            frames.append(dict(region=region,seed_atom_row=int(a['rows'][seed]),
                seed_distance_to_crystal_A=float(distance[seed]),atoms=len(rows),candidate_atoms=int(candidate.sum()),
                largest_component=max(sizes,default=0),components_ge5=sum(s>=5 for s in sizes),
                components=component_records))
            # Central slab, with projected coherent-component bonds.
            slab=np.abs(points[rows,2]-center[2])<radius*.35
            palette=np.array(['#bbbbbb','#e7b800','#00a39b','#8a55be','#735240','#cc5cba','#187bc4'])
            ax.scatter(points[rows[slab],0]-center[0],points[rows[slab],1]-center[1],c=palette[labels[slab]],s=16)
            for i,j in zip(edge_i,edge_j):
                if i<j and slab[i] and slab[j] and graph[i,j]>0:
                    ax.plot(points[rows[[i,j]],0]-center[0],points[rows[[i,j]],1]-center[1],color='#145a8b',lw=.7,alpha=.5)
            ax.set(title=f'{name}: largest component {max(sizes,default=0)} atoms',xlabel='Δx (Å)',ylabel='Δy (Å)',aspect='equal')
        for ax in list(axes.flat)[len(chosen):]:ax.set_visible(False)
        fig.suptitle(f'{material}: dense structured-liquid candidates, separated from existing crystal\nBlue: ordered liquid; pink: five-fold proxy; grey: other; lines: mutual q6-coherent neighbors\nHigh-order regions selected for inspection; no future-fate assertion')
        fig.savefig(root/'plots'/f'{material}-structured-liquid-regions.png',dpi=180);plt.close(fig)
        records[material]=dict(file=record['file'],regions=frames,selected_regions=len(frames),
            maximum_component=max(r['largest_component'] for r in frames),
            total_candidates=sum(r['candidate_atoms'] for r in frames))
        print(material,'dense candidate regions complete',flush=True)
    write_json(destination/'regions.json',records)
    # Nested component arrays remain in JSON; this CSV is intentionally summary-only.
    write_metric_table({m:{k:v for k,v in r.items() if k!='regions'} for m,r in records.items()},root,
                       family='geoframe_precursors',name='structured-liquid-regions')
    (root/'README.md').write_text('# Structured-liquid candidates in Ta and Zr\n\n'
        'Dense neighborhoods around high-order candidate atoms more than two encoder radii from existing crystal. '
        'These are targeted structural examples, not verified nuclei or an unbiased prevalence estimate. '
        'Components touching the sampled sphere boundary are marked as potentially truncated.\n\n'
        '[Ta regions](plots/Ta-structured-liquid-regions.png) · [Zr regions](plots/Zr-structured-liquid-regions.png) · '
        '[Definitions](tables/METRICS.md) · [Summary](tables/structured-liquid-regions.csv)\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True);p.add_argument('--reference-output',required=True)
    a=p.parse_args();run(a.reference_output,a.output)
