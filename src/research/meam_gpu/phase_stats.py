"""Phase-restricted descriptive thermodynamics from exact common observations."""
import argparse
import csv
import time
import numpy as np
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json
from src.experiment_runner.metric_docs import write_metric_rows


def load(path):
    with path.open() as stream:return {float(r['time_ps']):{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)}


def run(c,wait=False):
    root=resolve_path(c['output']);archive=resolve_path(c['simulation_archive']);out=root/'analyses/phase-thermodynamics-v1'
    reference=archive/'verified-interface-f64-50ps';candidate=archive/'verified-interface-mixed-50ps'
    deadline=time.monotonic()+2400
    while wait and not (reference/'complete.json').exists():
        if read(root/'technical/worker-status.json')['state']=='failed':raise RuntimeError('MD worker failed; cannot publish completed solid comparison')
        if time.monotonic()>deadline:raise TimeoutError('Float64 growth control did not finish within40minutes')
        time.sleep(20)
    a,b=load(reference/'thermodynamics.csv'),load(candidate/'thermodynamics.csv')
    sa,sb=load(reference/'structure.csv'),load(candidate/'structure.csv')
    times=sorted(t for t in a.keys()&b.keys()&sa.keys()&sb.keys() if t>=10 and sa[t]['crystal_fraction']>.9 and sb[t]['crystal_fraction']>.9)
    if not times:raise ValueError('No common grown-solid observations after10ps; no solid thermodynamic claim')
    rows=[dict(time_ps=t,reference_potential_eV_per_atom=a[t]['potential_eV_per_atom'],mixed_potential_eV_per_atom=b[t]['potential_eV_per_atom'],
        reference_temperature_K=a[t]['temperature_K'],mixed_temperature_K=b[t]['temperature_K'],
        reference_crystal_fraction=sa[t]['crystal_fraction'],mixed_crystal_fraction=sb[t]['crystal_fraction']) for t in times]
    metrics=dict(phase='grown interface, both >90% PTM crystalline',observations=len(times),first_ps=times[0],last_ps=times[-1],
        potential_mean_difference_eV_per_atom=float(np.mean([b[t]['potential_eV_per_atom']-a[t]['potential_eV_per_atom'] for t in times])),
        temperature_mean_difference_K=float(np.mean([b[t]['temperature_K']-a[t]['temperature_K'] for t in times])),
        final_reference_crystal_fraction=sa[max(sa)]['crystal_fraction'],final_mixed_crystal_fraction=sb[max(sb)]['crystal_fraction'])
    metrics['descriptive_tolerance_pass']=abs(metrics['potential_mean_difference_eV_per_atom'])<c['gates']['energy_mean_eV_per_atom'] and abs(metrics['temperature_mean_difference_K'])<c['gates']['temperature_mean_K']
    write_metric_rows(rows,out,family='meam_gpu_phase',name='observations');write_metric_rows([metrics],out,family='meam_gpu_phase',name='summary')
    write_json(out/'technical/complete.json',metrics)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for label,data,structure in [('mixed',b,sb),('float64',a,sa)]:
        ts=sorted(data);ss=sorted(structure)
        axes[0].plot(ss,[100*structure[t]['crystal_fraction'] for t in ss],label=label)
        axes[1].plot(ts,[data[t]['potential_eV_per_atom'] for t in ts],label=label)
        axes[2].plot(ts,[data[t]['temperature_K'] for t in ts],label=label)
    for ax,title,y in zip(axes,['Verified seeded growth','Potential energy','Temperature'],['% crystalline','eV / atom','K']):
        ax.set(title=title,xlabel='time (ps)',ylabel=y);ax.spines[['top','right']].set_visible(False);ax.legend()
    (out/'plots').mkdir(parents=True,exist_ok=True);fig.savefig(out/'plots/solid-comparison.png',dpi=170);fig.savefig(out/'plots/solid-comparison.pdf');plt.close(fig)
    print(metrics,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--wait',action='store_true')
    a=parser.parse_args();run(read(a.config),a.wait)


if __name__=='__main__':main()
