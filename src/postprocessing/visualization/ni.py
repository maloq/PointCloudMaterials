"""Descriptive Ni crystallization monitor and retrospective local-order quicklook."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import numpy as np
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.artifacts import file_hash as sha,write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.postprocessing.descriptors.ptm import full_labels
from src.postprocessing.io import frame_geometry
from src.postprocessing.descriptors.structure import components


def order(b,frame,c):
    import freud
    freud.parallel.set_num_threads(1)
    points,box=frame_geometry(b,frame)
    ptm=full_labels(points,box,c['ptm_rmsd_cutoff']);crystal=np.isin(ptm,[1,2,3])
    query=freud.locality.AABBQuery(freud.box.Box(*box),np.asarray(points-box/2,np.float32))
    neighbors=query.query(query.points,dict(r_max=c['neighbor_cutoff_A'],exclude_ii=True)).toNeighborList()
    sl=freud.order.SolidLiquid(l=6,q_threshold=.7,solid_threshold=7,normalize_q=True).compute(query,neighbors=neighbors)
    pair=np.column_stack((neighbors.query_point_indices,neighbors.point_indices))
    correlation=np.asarray(sl.ql_ij)
    if not np.isfinite(correlation).all():raise FloatingPointError('Nonfinite bond correlations')
    accepted=correlation>.7
    degree=np.bincount(pair[accepted,0],minlength=b.atom_count)
    return ptm,degree,components(crystal,pair,4)


def run(c,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=resolve_path(destination);root.mkdir(parents=True,exist_ok=False)
    archive=resolve_path(c['archive']);launch=resolve_path(c['launch'])
    records=json.loads((launch/'plan.json').read_text())['records'];rows=[];curves={};inputs={}
    for r in records:
        path=archive/r['id']/'structure.csv'
        if not path.exists():continue
        content=path.read_bytes();copy=root/'technical/monitors'/path.parent.name/path.name
        copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(content)
        observations=list(csv.DictReader(content.decode().splitlines()));curves[r['id']]=observations
        last=observations[-1]
        rows.append(dict(source=r['id'],temperature_K=r['temperature_K'],replicate=r['replicate'],
            md_complete=(path.parent/'complete.json').exists(),hold_duration_ps=float(last['time_ps'])-c['cooling_ps'],
            final_crystal_fraction=float(last['crystal_fraction']),largest_cluster_atoms=int(last['largest_cluster_atoms']),
            max_cluster_atoms=max(int(x['largest_cluster_atoms']) for x in observations),
            first_64_atom_crossing_hold_ps=next((float(x['time_ps'])-c['cooling_ps'] for x in observations if int(x['largest_cluster_atoms'])>=64),None)))
        inputs[str(copy.relative_to(root))]=sha(copy)
    write_metric_rows(rows,root,family='ni_births_preview',name='campaign-status')
    (root/'plots').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    colors={1100:'#c43b3b',1200:'#d38720',1300:'#2479b1',1400:'#4e986c'}
    for r in rows:
        data=curves[r['source']];t=np.array([float(x['time_ps'])-c['cooling_ps'] for x in data]);sel=t>=0
        style=['-','--',':'][r['replicate']];label=f"{r['temperature_K']} K / melt {r['replicate']}"
        axes[0].plot(t[sel],np.array([float(x['crystal_fraction'])*100 for x in data])[sel],style,color=colors[r['temperature_K']],label=label)
        axes[1].plot(t[sel],np.maximum(1,np.array([int(x['largest_cluster_atoms']) for x in data]))[sel],style,color=colors[r['temperature_K']])
    axes[0].set(ylabel='PTM crystalline atoms (%)',title='Ni: crystallization at 1100 K');axes[0].legend(fontsize=8,ncol=2)
    axes[1].set(ylabel='Largest PTM cluster (atoms; 0 shown as 1)',yscale='log',title='Native-coordinate monitor, every 1 ps')
    for ax in axes:ax.set_xlabel('Time at target temperature (ps)');ax.grid(alpha=.2)
    fig.savefig(root/'plots/crystallization.png',dpi=180);plt.close(fig)
    evidence=[];anchors=[]
    selected=[r for r in records if r['temperature_K']==1100 and r['replicate'] in [0,1]]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,r in zip(axes,selected):
        folder=archive/r['id'];complete=json.loads((folder/'complete.json').read_text())
        b=ShootingBinaryTrajectory.load(folder/'trajectory_binary_float16');b.verify_checksums()
        if sha(b.root/'manifest.json')!=complete['trajectory_manifest_sha256']:raise ValueError('Ni input manifest changed')
        crossing=next(x for x in curves[r['id']] if int(x['largest_cluster_atoms'])>=128)
        anchor_frame=int(np.searchsorted(b.timesteps,int(crossing['step'])))
        _,_,labels=order(b,anchor_frame,c);sizes=np.bincount(labels);sizes[0]=0
        members=np.flatnonzero(labels==np.argmax(sizes))
        if len(members)<4:raise ValueError('No crystalline component at native crossing')
        cohort=np.zeros(b.atom_count,bool);cohort[members]=True
        anchor=dict(source=r['id'],anchor_time_ps=float(crossing['time_ps']),atom_ids=b.atom_ids[members].tolist(),
            anchor_cluster_size=len(members),manifest_sha256=complete['trajectory_manifest_sha256'])
        anchors.append(anchor);samples=[];saved=[]
        for lag in range(-30,1):
            target=int(crossing['step'])+round(lag/c['timestep_ps']);frame=int(np.searchsorted(b.timesteps,target))
            if b.timesteps[frame]!=target:raise ValueError('Requested preview frame absent')
            ptm,degree,clusters=order(b,frame,c);noncrystal=~np.isin(ptm,[1,2,3]);counts=np.bincount(clusters);counts[0]=0
            saved.append((target,ptm,degree))
            for name,group in [('future_cluster',cohort),('other_atoms',~cohort)]:
                eligible=group & noncrystal;n=int(eligible.sum());ordered=int((eligible & (degree>=7)).sum())
                row=dict(source=r['id'],lag_to_anchor_ps=lag,hold_time_ps=target*c['timestep_ps']-c['cooling_ps'],group=name,
                    cohort_atoms=int(group.sum()),noncrystalline_atoms=n,coherent_noncrystalline_atoms=ordered,
                    coherent_fraction_within_noncrystalline=ordered/n if n else None,
                    ptm_crystal_fraction_in_group=float((group & ~noncrystal).sum()/group.sum()),
                    global_largest_ptm_cluster=int(counts.max()))
                samples.append(row);evidence.append(row)
        arrays=root/'technical'/f"{r['id']}-atom-order.npz"
        np.savez_compressed(arrays,timesteps=np.array([x[0] for x in saved]),ptm=np.stack([x[1] for x in saved]),coherent_degree=np.stack([x[2] for x in saved]))
        inputs[str(arrays.relative_to(root))]=sha(arrays)
        for name,label in [('future_cluster','Future cluster atoms'),('other_atoms','Other atoms')]:
            data=[x for x in samples if x['group']==name]
            ax.plot([x['lag_to_anchor_ps'] for x in data],[100*x['coherent_fraction_within_noncrystalline'] if x['coherent_fraction_within_noncrystalline'] is not None else np.nan for x in data],label=label)
        first8=next(float(x['time_ps']) for x in curves[r['id']] if int(x['largest_cluster_atoms'])>=8)
        ax.axvline(first8-float(crossing['time_ps']),color='gray',ls=':',label='First 8-atom PTM cluster anywhere')
        ax.set(title=r['id'],xlabel='Time relative to first 128-atom monitor crossing (ps)',ylabel='Coherent atoms among noncrystalline atoms (%)')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
        print(json.dumps(dict(source=r['id'],anchor_size=len(members),state='preview_complete')),flush=True)
    fig.savefig(root/'plots/retrospective-order.png',dpi=180);plt.close(fig)
    write_metric_rows(evidence,root,family='ni_births_preview',name='retrospective-order')
    write_json(root/'technical/anchors.json',anchors)
    write_json(root/'technical/complete.json',dict(checked_at=datetime.now(timezone.utc).isoformat(),job=os.environ.get('SLURM_JOB_ID'),
        files=inputs,anchors_sha256=sha(root/'technical/anchors.json'),interpretation='Retrospective descriptive quicklook; no lineage confirmation or independent predictive test.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(json.loads(resolve_path(a.config).read_text()),a.output)
