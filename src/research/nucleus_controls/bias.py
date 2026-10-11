"""Paired restart-versus-saved-future audit, including common storage precision."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time

import numpy as np
from ase import units

from src.data.fixed_cohort.protocol import sha,write_json
from src.data.trajectories.lammps import TemporalLAMMPSBinaryTrajectory
from src.data.trajectories.shooting import ShootingBinaryTrajectory
from src.experiment_runner.metric_docs import check_metric_docs,write_metric_rows
from src.project_runtime.paths import resolve_path
from src.postprocessing.protocols.ancestry import components
from src.postprocessing.descriptors.ptm import full_labels
from src.research.nucleus_recovery import read,roots

FAMILY='nucleus_continuation_bias'


def thermo(path):
    result={};columns=None
    for line in path.read_text().splitlines():
        fields=line.split()
        if fields[:2]==['Step','Temp']:
            columns=fields;continue
        if columns is None or len(fields)!=len(columns):continue
        try:values=list(map(float,fields))
        except ValueError:continue
        result[int(values[0])]=dict(zip(columns,values))
    if not result:raise ValueError('No LAMMPS thermo records: '+str(path))
    return result


def graph_values(path,indices,atoms):
    with np.load(path) as g:
        maximum=np.zeros(len(g['start']),np.int32)
        np.maximum.at(maximum,g['frame'][1:],g['size'][1:])
        return maximum[indices],g['crystalline_atoms'][indices]/atoms


def branch(arguments):
    c,label,recipe,plan,case,task=arguments
    out=resolve_path(c['output']);_,scratch,archive=roots(recipe)
    target=out/'technical/branches'/label/f'{task["name"]}.json'
    if target.exists():return read(target)
    root=archive/'branches'/task['name'];complete=read(root/'complete.json')
    result=read(root/'result.json')
    if complete['plan_identity']!=plan['identity'] or sha(root/'result.json')!=complete['result_sha256']:
        raise ValueError('Changed completed branch')
    n=case['atom_count'];cadence=case['cadence_ps'];start=case['start_frame']
    indices=np.rint(np.asarray(c['times_ps'])/cadence).astype(int)
    if not np.array_equal(indices*cadence,c['times_ps']):raise ValueError('Non-native audit times')
    original_graph=Path(case['audit_folder'])/'graph.npz'
    if sha(original_graph)!=case['graph_sha256']:raise ValueError('Changed source graph')
    ref_max,ref_fraction=graph_values(original_graph,start+indices,n)
    native_max,native_fraction=graph_values(root/'graph.npz',start+indices,n)
    raw=TemporalLAMMPSBinaryTrajectory.load(root/'trajectory_binary_float16')
    raw.verify_checksums()
    original=ShootingBinaryTrajectory.load(case['raw'])
    if sha(original.root/'manifest.json')!=case['raw_manifest_sha256']:raise ValueError('Changed original trajectory')
    if not np.array_equal(raw.atom_ids,original.atom_ids):raise ValueError('Changed atom identities')
    state=case['state'];dt=recipe['timestep_fs'] if task['integrator']=='baoab' else recipe['source_npt']['timestep_fs']
    if not np.allclose(raw.timesteps*dt/1000-state['start_ps'],np.arange(len(raw.positions))*cadence,rtol=0,atol=1e-8):
        raise ValueError('Changed actual continuation timeline')
    common=[]
    for j in indices:
        q=raw.positions[j].astype(float);box=raw.box_high[j].astype(float)-raw.box_low[j].astype(float);q=np.mod(q,box)
        labels=full_labels(q,box,recipe['ptm']['rmsd_cutoff'])
        _,sizes=components(q,box,labels,recipe['lineage'])
        common.append((int(sizes.max()),float(np.isin(labels,[1,2,3]).mean())))
    source_dir=Path(case['raw']).parent
    old_log=source_dir/'measurement.lammps.log';old_thermo=thermo(old_log)
    source_input=source_dir/'source.in.lammps';text=source_input.read_text();settings=recipe['source_npt']
    commands=[line.split() for line in text.splitlines()]
    expected=[['timestep',f'{settings["timestep_fs"]/1000:g}'],
        ['fix','remove_drift','all','momentum',str(settings['momentum_every_steps']),'linear','1','1','1']]
    for value in expected:
        if value not in commands:raise ValueError('Different original NPT input: '+str(source_input)+' '+str(value))
    nh=next(fields for fields in commands if fields[:4]==['fix','ensemble','all','npt'])
    if nh[4]!='temp' or nh[8]!='iso' or list(map(float,nh[5:8]+nh[9:12]))!=[
        case['temperature_K'],case['temperature_K'],settings['thermostat_ps'],settings['pressure_bar'],settings['pressure_bar'],settings['barostat_ps']]:
        raise ValueError('Different source bath settings: '+str(source_input))
    pair=next(fields for fields in commands if fields[:3]==['pair_coeff','*','*'])
    for key,index in [('library',3),('parameters',5)]:
        if sha(source_dir/pair[index])!=plan['potential_sha256'][key]:raise ValueError('Different source potential')
    new_thermo=thermo(scratch/'branches'/task['name']/'lammps.log') if task['integrator']=='npt' else None
    rows=[];thermal=[];missing_thermo=[]
    for i,(h,j) in enumerate(zip(c['times_ps'],indices)):
        base=dict(campaign=label,case=case['key'],ancestry=case['ancestry'],cohort=case['cohort'],arm=task['arms'][0],
            temperature_K=case['temperature_K'],task=task['name'],time_ps=h,replicate=task['replicate'])
        for measurement,m,f in [('native',native_max[i],native_fraction[i]),('common_float16',*common[i])]:
            rows.append(dict(**base,measurement=measurement,reference_largest_cluster_atoms=int(ref_max[i]),
                restart_largest_cluster_atoms=int(m),reference_crystal_fraction=float(ref_fraction[i]),restart_crystal_fraction=float(f)))
        old=old_thermo[int(original.timesteps[start+j])]
        new=None if new_thermo is None else new_thermo.get(int(raw.timesteps[j]))
        if new_thermo is not None and new is None:missing_thermo.append(int(raw.timesteps[j]))
        new_volume=float(np.prod(raw.box_high[j].astype(float)-raw.box_low[j].astype(float)))
        thermal.append(dict(**base,reference_temperature_K=old['Temp'],restart_temperature_K=None if new is None else new['Temp'],
            reference_pressure_bar=old['Press'],restart_pressure_bar=None if new is None else new['Press'],
            reference_volume_A3=old['Volume'],restart_volume_A3=new_volume,
            reference_energy_eV_atom=old['PotEng']/n,restart_energy_eV_atom=None if new is None else new['PotEng']/n))
    if task['integrator']=='baoab':
        with np.load(root/'state.npz') as values:
            if int(values['index'])!=16:raise ValueError('Incomplete final state')
            p=values['p'].astype(float)
            thermal[-1]['restart_temperature_K']=float(np.sum(p*p)/recipe['mass_amu']/(3*n*units.kB))
    payload=dict(rows=rows,thermal=thermal,source_input_sha256=sha(source_input),source_log_sha256=sha(old_log),
        result_sha256=sha(root/'result.json'),archive_manifest_sha256=sha(raw.root/'manifest.json'),plan_identity=plan['identity'],
        missing_restart_thermo_steps=missing_thermo)
    write_json(target,payload)
    return payload


def paired_summary(c,records,metrics,axes):
    groups=defaultdict(list)
    for r in records:
        for stratum in ['all',f'T{r["temperature_K"]:g}']:
            groups[tuple(r[k] for k in axes)+(stratum,)].append(r)
    summaries=[]
    for key,rows in sorted(groups.items()):
        by_source=defaultdict(list)
        for r in rows:by_source[r['ancestry']].append(r)
        for metric in metrics:
            values=[]
            for rs in by_source.values():
                a=[r['restart_'+metric] for r in rs if r['restart_'+metric] is not None]
                b=[r['reference_'+metric] for r in rs if r['restart_'+metric] is not None]
                if a:values.append((np.mean(b),np.mean(a)))
            if not values:continue
            a=np.asarray(values);delta=a[:,1]-a[:,0]
            rng=np.random.default_rng(c['seed'])
            ids=rng.integers(len(a),size=(c['bootstrap_draws'],len(a)))
            lo,hi=np.quantile(delta[ids].mean(1),[.025,.975]) if len(a)>1 else (None,None)
            summaries.append(dict(zip(axes+['stratum'],key),metric=metric,ancestors=len(a),
                reference_mean=float(a[:,0].mean()),restart_mean=float(a[:,1].mean()),difference=float(delta.mean()),
                difference_ci95_low=None if lo is None else float(lo),difference_ci95_high=None if hi is None else float(hi),
                relative_change_percent=float(100*(a[:,1].mean()/a[:,0].mean()-1)) if a[:,0].mean()!=0 else None))
    return summaries


def run(c):
    check_metric_docs(family=FAMILY);out=resolve_path(c['output']);work=[];inputs=[]
    for label,path in c['campaigns'].items():
        recipe=read(resolve_path(path));source,_,archive=roots(recipe);plan=read(source/'technical/plan.json')
        for task in plan['tasks']:
            case=next(v for v in plan['cases'] if v['key']==task['case'])
            if not (archive/'branches'/task['name']/'complete.json').exists():raise ValueError('Unfinished input: '+task['name'])
            work.append((c,label,recipe,plan,case,task))
        inputs.append(dict(campaign=label,plan=str(source/'technical/plan.json'),sha256=sha(source/'technical/plan.json'),
            identity=plan['identity'],cases=len(plan['cases']),branches=len(plan['tasks'])))
    write_json(out/'technical/inputs.json',inputs);write_json(out/'technical/config.json',c)
    values=[]
    with ProcessPoolExecutor(max_workers=c['workers']) as pool:
        for i,value in enumerate(pool.map(branch,work),1):
            values.append(value)
            if i%20==0:print('audited',i,'/',len(work),flush=True)
    structural=[r for v in values for r in v['rows']];thermal=[r for v in values for r in v['thermal']]
    paired=paired_summary(c,structural,['crystal_fraction','largest_cluster_atoms'],['campaign','cohort','arm','measurement','time_ps'])
    physical=paired_summary(c,thermal,['temperature_K','pressure_bar','volume_A3','energy_eV_atom'],['campaign','cohort','arm','time_ps'])
    for name,rows in [('paired-branches',structural),('paired-structure',paired),('thermodynamic-branches',thermal),('paired-thermodynamics',physical)]:
        write_metric_rows(rows,out,family=FAMILY,name=name)
    plot(out,paired)
    write_json(out/'technical/complete.json',dict(branches=len(values),completed_unix=time.time()))
    print('Complete',len(values),'branch audits',flush=True)


def plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    labels={'sourceNPT':'Original-settings NPT','response450K':'450 K stochastic'}
    for ax,metric,scale,title in zip(axes,['crystal_fraction','largest_cluster_atoms'],[100,1],['PTM crystalline atoms (%)','Largest cluster (atoms)']):
        for measurement,style in [('native',':'),('common_float16','-')]:
            for i,arm in enumerate(labels):
                selected=sorted([r for r in rows if r['campaign']=='extra' and r['cohort']=='random_liquid' and r['stratum']=='all'
                    and r['arm']==arm and r['metric']==metric and r['measurement']==measurement],key=lambda r:r['time_ps'])
                x=[r['time_ps'] for r in selected];y=[r['restart_mean']*scale for r in selected]
                ax.plot(x,y,style,color=f'C{i}',label=labels[arm]+(' (native labels)' if measurement=='native' else ' (matched storage)'))
                if i==0 and measurement=='native':ax.plot(x,[r['reference_mean']*scale for r in selected],color='black',label='Saved original future')
        ax.set_xlabel('Time after restart (ps)');ax.set_ylabel(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('64 paired liquid starting states — identical observation precision audit')
    fig.tight_layout();(out/'plots').mkdir(exist_ok=True);fig.savefig(out/'plots/paired-liquid.png',dpi=170);fig.savefig(out/'plots/paired-liquid.svg');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();run(read(args.config))


if __name__=='__main__':main()
