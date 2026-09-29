"""Paired physical comparison of original and exact-0.1-ps Al trajectories."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.metric_docs import snapshot_metric_docs, write_metric_table, fingerprint
from src.project_runtime.paths import load_json
from src.project_runtime.transfer import write_json
from src.simulation.campaigns.common import _read_thermodynamic_log

FAMILY = 'al_replay'


def structure(positions, lengths):
    from ovito.data import DataCollection, Particles, SimulationCell
    from ovito.modifiers import PolyhedralTemplateMatchingModifier, ClusterAnalysisModifier
    data = DataCollection()
    cell = SimulationCell(pbc=(True, True, True))
    cell[:, :3] = np.diag(lengths)
    data.objects.append(cell)
    particles = Particles()
    particles.create_property('Position', data=np.mod(positions, lengths))
    data.objects.append(particles)
    data.apply(PolyhedralTemplateMatchingModifier(rmsd_cutoff=.1))
    labels = np.asarray(data.particles['Structure Type'], dtype=np.int32).copy()
    crystal = np.isin(labels, [1, 2, 3])
    data.particles_.create_property('Selection', data=crystal.astype(np.int32))
    data.apply(ClusterAnalysisModifier(cutoff=3.5, only_selected=True, sort_by_size=True))
    fractions = np.bincount(labels, minlength=5)/len(labels)
    values = {name+'_fraction':float(fractions[i]) for i,name in enumerate(('other','fcc','hcp','bcc','ico'))}
    values.update(crystal_fraction=float(crystal.mean()),
                  largest_crystal_cluster=int(data.attributes['ClusterAnalysis.largest_size']))
    return labels, values


def neighbors(positions, lengths, centers):
    points = np.mod(positions, lengths)
    indices = cKDTree(points, boxsize=lengths).query(points[centers], k=14, workers=1)[1]
    return np.array([row[row != center][:12] for row,center in zip(indices,centers,strict=True)])


def pair_task(record, config):
    os.environ['OVITO_THREAD_COUNT'] = '1'
    start = time.monotonic()
    old_dir, new_dir = Path(record['parent_directory']), Path(record['new_directory'])
    old = ShootingBinaryTrajectory.load(old_dir/'trajectory_binary_float16')
    new = ShootingBinaryTrajectory.load(new_dir/'trajectory_binary_float16')
    if old.atom_count != 70304 or new.atom_count != old.atom_count:
        raise ValueError('Unexpected paired atom counts')
    for name in ('atom_ids','atom_types'):
        if not np.array_equal(getattr(old,name),getattr(new,name)):
            raise ValueError(f'Paired atom identity mismatch: {record["source_id"]}, {name}')
    if fingerprint(old.root/'manifest.json') != record['parent_manifest_sha256']:
        raise ValueError('Original manifest changed')
    for name in ('prepared_liquid.lammps.data','melt_final.restart.bin'):
        expected = record['parent_input_sha256'][name]
        if fingerprint(old_dir/name) != expected or fingerprint(new_dir/name) != expected:
            raise ValueError(f'Preparation mismatch: {name}, source {record["source_id"]}')
    old_outcome = json.loads((old_dir/'outcome.json').read_text())
    if old_outcome['velocity_seed'] != record['velocity_seed']:
        raise ValueError('Velocity seeds differ')
    old_fs, new_fs = old.timesteps*3, new.timesteps*2
    common, oi, ni = np.intersect1d(old_fs, new_fs, return_indices=True)
    if not np.array_equal(common, np.arange(0,600001,1500)):
        raise ValueError('Shared timeline is not exactly 0:1.5:600 ps')
    ot, nt = (_read_thermodynamic_log(d/'measurement.lammps.log') for d in (old_dir,new_dir))
    grid = set(config['structure_times_ps'])
    centers = np.sort(np.random.default_rng(config['seed']+record['source_id']).choice(old.atom_count,config['neighbor_centers'],replace=False))
    rows, structural = [], []
    for fs,i,j in zip(common,oi,ni,strict=True):
        t = float(fs/1000)
        lo = (old.box_high[i]-old.box_low[i]).astype(float)
        ln = (new.box_high[j]-new.box_low[j]).astype(float)
        po, pn = old.positions[i].astype(float), new.positions[j].astype(float)
        delta = pn/ln-po/lo
        delta -= np.rint(delta)
        delta *= (lo+ln)/2
        distances = np.linalg.norm(delta,axis=1)
        vo, vn = old.velocities[i].astype(float), new.velocities[j].astype(float)
        vo -= vo.mean(axis=0); vn -= vn.mean(axis=0)
        thermo_old, thermo_new = ot[int(old.timesteps[i])], nt[int(new.timesteps[j])]
        row = dict(source_id=record['source_id'],role=record['split'],temperature_K=record['temperature_K'],time_ps=t,
            same_atom_rms_A=float(np.sqrt(np.mean(distances**2))),same_atom_median_A=float(np.median(distances)),
            same_atom_within_1A=float(np.mean(distances<1)),
            velocity_correlation=float(np.sum(vo*vn)/np.sqrt(np.sum(vo**2)*np.sum(vn**2))),
            random_position_rms_A=float(np.sqrt(np.sum(((lo+ln)/2)**2)/12)))
        for suffix,values in (('old',thermo_old),('new',thermo_new)):
            row.update({f'temperature_{suffix}_K':values[0],f'pressure_{suffix}_bar':values[1],
                f'volume_{suffix}_A3_per_atom':values[2]/old.atom_count,f'energy_{suffix}_eV_per_atom':values[3]/old.atom_count})
        rows.append(row)
        if t in grid:
            labels_o, stats_o = structure(po,lo)
            labels_n, stats_n = structure(pn,ln)
            co,cn = np.isin(labels_o,[1,2,3]),np.isin(labels_n,[1,2,3])
            union = np.count_nonzero(co|cn)
            no, nn = neighbors(po,lo,centers), neighbors(pn,ln,centers)
            retention = np.mean([np.intersect1d(a,b).size/12 for a,b in zip(no,nn,strict=True)])
            same = float(np.mean(labels_o==labels_n))
            f_o = np.bincount(labels_o,minlength=5)/len(labels_o)
            f_n = np.bincount(labels_n,minlength=5)/len(labels_n)
            chance = float(f_o@f_n)
            structural.append(dict(source_id=record['source_id'],time_ps=t,neighbor_retention=float(retention),
                ptm_agreement=same,ptm_kappa=(same-chance)/(1-chance) if chance<1 else None,
                crystal_atom_jaccard=float(np.count_nonzero(co&cn)/union) if union else None,
                **{k+'_old':v for k,v in stats_o.items()},**{k+'_new':v for k,v in stats_n.items()}))
    return dict(source_id=record['source_id'],record=record,matched_frames=len(rows),
        old_manifest_sha256=fingerprint(old.root/'manifest.json'),new_manifest_sha256=fingerprint(new.root/'manifest.json'),
        rows=rows,structure=structural,elapsed_seconds=time.monotonic()-start)


def table(root,name,rows):
    contract = snapshot_metric_docs(root,FAMILY)
    p = root/'tables'/f'{name}.csv'
    pd.DataFrame(rows).to_csv(p,index=False)
    write_json(root/'technical/table-contracts'/f'{name}.json',dict(table=str(p.relative_to(root)),
        sha256=fingerprint(p),family=FAMILY,contract=str(contract.relative_to(root)),definitions=f'tables/metric-definitions/{FAMILY}.md'))


def crossing(frame,column,threshold):
    values = frame[column].to_numpy() >= threshold
    for i in range(len(values)-1):
        if values[i] and values[i+1]:
            return float(frame.iloc[i]['time_ps'])
    return None


def paired_interval(values,seed):
    values = np.asarray(values,dtype=float)
    draws = np.random.default_rng(seed).choice(values,(10000,len(values)),replace=True).mean(axis=1)
    return dict(mean=float(values.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())


def report(root,config):
    saved = [json.loads(p.read_text()) for p in sorted((root/'technical/pairs').glob('*.json'))]
    a = pd.DataFrame([r for p in saved for r in p['rows']])
    b = pd.DataFrame([r for p in saved for r in p['structure']])
    summary = []
    for p in saved:
        sid = p['source_id']; x=a[a.source_id==sid];y=b[b.source_id==sid].sort_values('time_ps')
        r=dict(source_id=sid,role=p['record']['split'],temperature_K=p['record']['temperature_K'],
               initial_same_atom_rms_A=float(x.iloc[0].same_atom_rms_A),
               final_same_atom_rms_A=float(x.iloc[-1].same_atom_rms_A),
               initial_neighbor_retention=float(y.iloc[0].neighbor_retention),
               final_neighbor_retention=float(y.iloc[-1].neighbor_retention),
               final_crystal_old=float(y.iloc[-1].crystal_fraction_old),final_crystal_new=float(y.iloc[-1].crystal_fraction_new),
               mean_absolute_crystal_fraction_difference=float(np.mean(abs(y.crystal_fraction_new-y.crystal_fraction_old))))
        for label in ('old','new'):
            for threshold in (.1,.5):
                r[f't{int(threshold*100)}_{label}_ps']=crossing(y,f'crystal_fraction_{label}',threshold)
        for field in ('temperature','energy','volume'):
            unit={'temperature':'K','energy':'eV_per_atom','volume':'A3_per_atom'}[field]
            r[f'mean_delta_{field}_{unit}']=float((x[f'{field}_new_{unit}']-x[f'{field}_old_{unit}']).mean())
        summary.append(r)
    s = pd.DataFrame(summary)
    # All comparisons are descriptive; no model training or checkpoint selection.
    curves_o=b.pivot(index='source_id',columns='time_ps',values='crystal_fraction_old').to_numpy()
    curves_n=b.pivot(index='source_id',columns='time_ps',values='crystal_fraction_new').to_numpy()
    costs=abs(curves_o[:,None,:]-curves_n[None,:,:]).mean(axis=2)
    matched=float(np.diag(costs).mean());rng=np.random.default_rng(config['seed'])
    perm=np.array([costs[np.arange(len(s)),rng.permutation(len(s))].mean() for _ in range(10000)])
    aggregate=dict(pairs=len(s),temperature_K=sorted(s.temperature_K.unique().tolist()),
        matched_crystal_curve_mae=matched,permuted_pair_mean_mae=float(perm.mean()),
        pairing_permutation_p_lower=float((1+np.count_nonzero(perm<=matched))/(len(perm)+1)),
        mean_initial_same_atom_rms_A=float(s.initial_same_atom_rms_A.mean()),
        mean_final_same_atom_rms_A=float(s.final_same_atom_rms_A.mean()),
        mean_initial_neighbor_retention=float(s.initial_neighbor_retention.mean()),
        mean_final_neighbor_retention=float(s.final_neighbor_retention.mean()),
        final_crystal_old_mean=float(s.final_crystal_old.mean()),final_crystal_new_mean=float(s.final_crystal_new.mean()),
        final_crystal_delta=paired_interval(s.final_crystal_new-s.final_crystal_old,config['seed']),
        t50_reached_old=int(s.t50_old_ps.notna().sum()),t50_reached_new=int(s.t50_new_ps.notna().sum()),
        mean_temperature_delta_K=paired_interval(s.mean_delta_temperature_K,config['seed']),
        mean_energy_delta_eV_per_atom=paired_interval(s.mean_delta_energy_eV_per_atom,config['seed']),
        mean_volume_delta_A3_per_atom=paired_interval(s.mean_delta_volume_A3_per_atom,config['seed']))
    write_metric_table(aggregate,root,family=FAMILY)
    table(root,'matched_times',a);table(root,'structure',b);table(root,'paired_sources',s)
    write_json(root/'technical/summary.json',aggregate)
    render(root,a,b,s)
    return aggregate


def render(root,a,b,s):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for _,x in a.groupby('source_id'):
        ax[0,0].plot(x.time_ps,x.same_atom_rms_A,alpha=.22,color='#4361ee')
    ax[0,0].set(title='Same-atom position divergence',xlabel='Measurement time (ps)',ylabel='Periodic, box-corrected RMS (Å)')
    for _,x in b.groupby('source_id'):
        ax[0,1].plot(x.time_ps,x.neighbor_retention,alpha=.22,color='#00876c')
    ax[0,1].set(title='Retention of 12 nearest neighbors',xlabel='Measurement time (ps)',ylabel='Shared neighbor fraction',ylim=(0,1))
    g=a.groupby('time_ps').velocity_correlation
    ax[0,2].plot(g.mean().index,g.mean().values);ax[0,2].axhline(0,color='gray',ls='--')
    ax[0,2].set(title='Velocity correlation, same atom IDs',xlabel='Measurement time (ps)',ylabel='Mean correlation')
    for field,label,color in [('crystal_fraction_old','Original: 3 fs / 0.75 ps','#247ba0'),('crystal_fraction_new','Rerun: 2 fs / 0.1 ps','#f25f5c')]:
        g=b.groupby('time_ps')[field];ax[1,0].plot(g.mean().index,g.mean().values,label=label,color=color)
        ax[1,0].fill_between(g.mean().index,g.quantile(.25),g.quantile(.75),alpha=.12,color=color)
    ax[1,0].set(title='Crystal fraction: mean and interquartile range',xlabel='Measurement time (ps)',ylabel='FCC + HCP + BCC fraction',ylim=(0,1));ax[1,0].legend(fontsize=8)
    ax[1,1].scatter(s.final_crystal_old,s.final_crystal_new);ax[1,1].plot([0,1],[0,1],'--',color='gray')
    for _,row in s.iterrows():ax[1,1].annotate(str(int(row.source_id)),(row.final_crystal_old,row.final_crystal_new),fontsize=7)
    ax[1,1].set(title='Final crystal fraction, paired parents',xlabel='Original',ylabel='Rerun',xlim=(0,1),ylim=(0,1))
    for name,color in [('old','#247ba0'),('new','#f25f5c')]:
        g=a.groupby('time_ps')[f'energy_{name}_eV_per_atom'].mean();ax[1,2].plot(g.index,g.values,label=name,color=color)
    ax[1,2].set(title='Mean potential energy per atom',xlabel='Measurement time (ps)',ylabel='eV / atom');ax[1,2].legend()
    fig.suptitle('20 completed Al pairs at 520 K • exact shared times • identical melt ancestry',fontsize=14)
    (root/'plots').mkdir(exist_ok=True);fig.savefig(root/'plots/overview.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(4,5,figsize=(17,11),sharex=True,sharey=True,layout='constrained')
    for ax,(sid,x) in zip(axes.flat,b.groupby('source_id'),strict=True):
        ax.plot(x.time_ps,x.crystal_fraction_old,color='#247ba0',label='Original')
        ax.plot(x.time_ps,x.crystal_fraction_new,color='#f25f5c',label='Rerun')
        ax.set(title=f'Parent {sid}',ylim=(0,1),xlim=(0,600))
    axes[0,0].legend();fig.supxlabel('Measurement time (ps)');fig.supylabel('FCC + HCP + BCC fraction (PTM RMSD ≤ 0.1)')
    fig.suptitle('Same-parent crystallization paths • recomputed identically from both float16 exports')
    fig.savefig(root/'plots/paired_crystallization.png',dpi=180);plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True);parser.add_argument('--stage',choices=['compute','report'],default='compute')
    args=parser.parse_args(argv);config=load_json(args.config);root=Path(config['output'])
    if args.stage=='compute':
        if (root/'technical/summary.json').exists():raise FileExistsError('Completed comparison exists; use report')
        manifest_path=Path(config['campaign'])/'manifest.json';m=json.loads(manifest_path.read_text());records=[]
        for r in m['runs']:
            d=Path(m['root'])/r['run_dir'];p=d/'outcome.json'
            if p.exists() and json.loads(p.read_text())['state']=='complete':records.append(dict(r,new_directory=str(d.resolve())))
        if len(records)!=config['expected_pairs']:raise ValueError('Completed pair count changed; use a new frozen comparison config')
        if {r['temperature_K'] for r in records}!={520.}:raise ValueError('This declared comparison contains only 520 K sources')
        (root/'technical/pairs').mkdir(parents=True,exist_ok=True)
        write_json(root/'technical/protocol.json',dict(config=config,campaign_manifest_sha256=fingerprint(manifest_path),
            records=records,exclusions='Three interrupted and 127 unstarted descendants; no replacement or outcome-based pair selection.',
            interpretation='Descriptive simulation audit including held-out ancestry; no fitting, model selection or benchmark promotion.'))
        with ProcessPoolExecutor(max_workers=config['workers'],mp_context=multiprocessing.get_context('spawn')) as executor:
            futures={}
            for record in records:
                target=root/'technical/pairs'/f'{record["source_id"]}.json'
                if target.exists():raise FileExistsError(f'Partial comparison already exists: {target}; inspect before continuing')
                futures[executor.submit(pair_task,record,config)]=target
            for future in as_completed(futures):
                result=future.result();write_json(futures[future],result)
                print(f'PAIR {result["source_id"]}: {result["elapsed_seconds"]:.1f} seconds',flush=True)
    print(json.dumps(report(root,config),indent=2))


if __name__=='__main__':main()
