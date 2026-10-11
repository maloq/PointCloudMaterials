"""Publish saved numerical gates, complete-response timings and partial MD evidence."""
import argparse
import csv
import re
import numpy as np
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json
from src.experiment_runner.metric_docs import write_metric_rows


def table(path):
    if not path.exists():return []
    with path.open() as stream:return [{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)]


def report(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=resolve_path(c['output']);archive=resolve_path(c['simulation_archive'])
    out=root/'analyses/comparison-v4';(out/'plots').mkdir(parents=True,exist_ok=True)
    rows=[];response=[];gates=[]
    path=root/'analyses/numerical-v1/technical/periodic.json'
    if path.exists():
        rows=[dict(configurational_pressure_bar=None,**{k:v for k,v in r.items() if k not in ('samples','configurational_pressure_bar')}) | ({'configurational_pressure_bar':r['configurational_pressure_bar']} if 'configurational_pressure_bar' in r else {}) for r in read(path)]
        write_metric_rows(rows,out,family='meam_gpu',name='force_timing')
    path=root/'analyses/response-v2/technical/timings.json'
    if path.exists():
        timing=read(path)
        for r in timing:
            old=next(x for x in timing if x['parent']==r['parent'] and x['batch']==r['batch'] and x['engine']=='old-meam-f64')
            response.append(dict(parent=r['parent'],atoms=r['atoms'],batch=r['batch'],engine=r['engine'],
                seconds_per_shot=r['seconds']/r['shots'],speedup=old['seconds']/r['seconds'],shots=r['shots']))
        write_metric_rows(response,out,family='meam_gpu',name='response_timing')
    for name,relative in [('nve','analyses/numerical-v1/technical/nve.json'),('response','analyses/response-v2/technical/validation.json')]:
        path=root/relative
        if path.exists():
            value=read(path);values=value if isinstance(value,list) else [value]
            write_metric_rows(values,out,family='meam_gpu',name=name+'_validation');gates.extend(values)
    paths={'original CPU':resolve_path(c['ancestor']).parent/'cpu-mpi8',
           'original GPU float64':resolve_path(c['ancestor']).parent/'gpu-compiled',
           'new GPU mixed':archive/c['production_name'],
           'seeded mixed':archive/'verified-interface-mixed-50ps','seeded float64':archive/'verified-interface-f64-50ps'}
    curves={};statuses=[];thermo=[]
    for label,path in paths.items():
        t=table(path/'thermodynamics.csv');s=table(path/'structure.csv')
        curves[label]=(t,s)
        status=dict(arm=label,state='complete' if (path/'complete.json').exists() else 'running' if t else 'not started',
            observed_ps=t[-1]['time_ps'] if t else 0.,crystal_fraction=s[-1]['crystal_fraction'] if s else None,
            largest_cluster_atoms=s[-1]['largest_cluster_atoms'] if s else None)
        if (path/'complete.json').exists():status['steps_per_second']=read(path/'complete.json')['steps_per_second']
        else:status['steps_per_second']=None
        statuses.append(status)
    write_json(out/'technical/status.json',statuses)
    write_metric_rows(statuses,out,family='meam_gpu',name='trajectory_status')
    for label in ['original CPU','original GPU float64']:
        ta,sa=curves[label];tb,sb=curves['new GPU mixed']
        a={r['time_ps']:r for r in ta};b={r['time_ps']:r for r in tb}
        liquid_a={r['time_ps'] for r in sa if r['crystal_fraction']<.01};liquid_b={r['time_ps'] for r in sb if r['crystal_fraction']<.01}
        times=sorted(t for t in a.keys()&b.keys()&liquid_a&liquid_b if t>=c['thermodynamics_burnin_ps'])
        if times:
            pe=float(np.mean([b[t]['potential_eV_per_atom']-a[t]['potential_eV_per_atom'] for t in times]))
            temperature=float(np.mean([b[t]['temperature_K']-a[t]['temperature_K'] for t in times]))
            thermo.append(dict(reference=label,exact_phase_matched_observations=len(times),first_ps=times[0],last_ps=times[-1],
                potential_mean_difference_eV_per_atom=pe,temperature_mean_difference_K=temperature,
                descriptive_tolerance_pass=abs(pe)<c['gates']['energy_mean_eV_per_atom'] and abs(temperature)<c['gates']['temperature_mean_K']))
    if thermo:write_metric_rows(thermo,out,family='meam_gpu',name='liquid_thermodynamics')
    native=archive/'native-reference/run.log';native_row=[]
    if native.exists():
        matches=re.findall(r'Loop time of ([0-9.eE+-]+) on \d+ procs for (\d+) steps',native.read_text())
        times=[float(t)/int(n) for t,n in matches if int(n)==1000]
        if times:
            native_row=[dict(engine='native Kokkos float64',seconds_per_step=float(np.median(times)),blocks=len(times))]
            write_metric_rows(native_row,out,family='meam_gpu',name='native_timing')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    ax=axes[0,0]
    if rows:
        ax.bar([r['engine'] for r in rows],[r['seconds']*1000 for r in rows],color=['#718096','#3182ce','#159a82']);ax.tick_params(axis='x',rotation=20)
    ax.set(title='16k liquid: force + topology',ylabel='milliseconds / call')
    ax=axes[0,1]
    for j,name in enumerate(['old-meam-f64','cuda-float64','cuda-float32']):
        values=[[r['seconds_per_shot'] for r in response if r['engine']==name and r['batch']==batch] for batch in [1,2]]
        if any(values):
            ax.bar(np.arange(2)+(j-1)*.24,[np.mean(v) if v else np.nan for v in values],width=.24,label=name)
    ax.set(xticks=[0,1],xticklabels=['batch 1','batch 2'],title='Full 100 fs response, 768 features',ylabel='seconds / shot')
    if response:ax.legend(fontsize=8)
    ax=axes[0,2]
    for label,(t,s) in curves.items():
        if not label.startswith('seeded') and s:ax.plot([r['time_ps'] for r in s],[100*r['crystal_fraction'] for r in s],label=label)
    ax.set(title='Unseeded liquid: PTM crystalline fraction',xlabel='time (ps)',ylabel='% FCC + HCP + BCC')
    if any(s for _,s in curves.values()):ax.legend(fontsize=8)
    for col,key,title,ylabel in [(0,'potential_eV_per_atom','Unseeded: potential energy','eV / atom'),(1,'temperature_K','Unseeded: temperature','K')]:
        ax=axes[1,col]
        for label,(t,s) in curves.items():
            if not label.startswith('seeded') and t:ax.plot([r['time_ps'] for r in t],[r[key] for r in t],label=label,alpha=.8)
        ax.set(title=title,xlabel='time (ps)',ylabel=ylabel)
    ax=axes[1,2]
    for label,(t,s) in curves.items():
        if label.startswith('seeded') and s:ax.plot([r['time_ps'] for r in s],[100*r['crystal_fraction'] for r in s],label=label)
    ax.set(title='Seeded interface: growth control',xlabel='time (ps)',ylabel='% FCC + HCP + BCC',ylim=(0,100))
    if any(curves[n][1] for n in ['seeded mixed','seeded float64']):ax.legend(fontsize=8)
    fig.suptitle('Differentiable CUDA MEAM — measured progress; shared ancestors are not independent replicas',fontsize=13)
    fig.savefig(out/'plots/overview.png',dpi=170);fig.savefig(out/'plots/overview.pdf');plt.close(fig)
    lines=['# Differentiable CUDA MEAM: measured results','',
        '[Scientific figure](analyses/comparison-v4/plots/overview.png) · [Metric definitions](analyses/comparison-v4/tables/METRICS.md)','',
        'Same pinned Lee2003 Al energy. CUDA topology and compiled analytic force/Hessian-vector products; no finite-difference replacement. Float64 integration/observables with declared force precision.','',
        '## Full response queries','', '| Force engine | Batch | Mean seconds/shot | Mean paired speedup | Parents measured |','|---|---:|---:|---:|---:|']
    for name in ['old-meam-f64','cuda-float64','cuda-float32']:
        for batch in [1,2]:
            selected=[r for r in response if r['engine']==name and r['batch']==batch]
            if selected:lines.append(f"| {name} | {batch} | {np.mean([r['seconds_per_shot'] for r in selected]):.4f} | {np.mean([r['speedup'] for r in selected]):.2f}x | {len(selected)}/4 |")
    lines+=['','All measurements include both original noise seeds. Each shot is100 steps with768 target features and two response directions. Historical MACE timings are a separate earlier benchmark, not included in these same-run speedup ratios.','','## Trajectories','', '| Arm | Status | Observed ps | Last crystal fraction |','|---|---|---:|---:|']
    for s in statuses:lines.append(f"| {s['arm']} | {s['state']} | {s['observed_ps']:g} | {s['crystal_fraction'] if s['crystal_fraction'] is not None else 'pending'} |")
    lines+=['','The1325K interface candidate failed a diffusion audit and is retained as a hot-crystal cooling control, excluded here. The verified2500K preparation has mobile MSD101.75A² and0% mobile crystal before quenching. The seeded assay starts with a prepared crystalline slab; its growth is not spontaneous nucleation. This study does not establish equal melting points, NPT densities or nucleation rates. Completed liquid observations are compared only at exact common times with both structures below1% crystalline.','',
        'Native float64 checkpoints and RNG states are archived independently of verified float16 position observations. Remaining reference arms and broader Al campaigns retain their separate progress records.']
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    print(dict(response_rows=len(response),trajectories=statuses,liquid_thermodynamics=thermo,native=native_row),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    a=parser.parse_args();report(read(a.config))


if __name__=='__main__':main()
