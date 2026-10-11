"""Native-coordinate divergence and RDF comparisons at exact common checkpoints."""
import argparse
import numpy as np
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read
from src.research.meam_md.report import rdf
from src.experiment_runner.metric_docs import write_metric_rows


def analyze(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=resolve_path(c['output']);archive=resolve_path(c['simulation_archive'])
    reference=resolve_path(c['ancestor']).parent
    out=root/'analyses/native-diagnostics-v1';(out/'plots').mkdir(parents=True,exist_ok=True)
    with np.load(resolve_path(c['ancestor'])/'initial.npz') as data:box=data['box']
    paths={'CPU':reference/'cpu-mpi8','old GPU float64':reference/'gpu-compiled','new GPU mixed':archive/c['production_name']}
    checkpoints={name:{int(p.stem.split('-')[1]):p for p in path.joinpath('checkpoints').glob('state-*.npz')} for name,path in paths.items()}
    rows=[];radial=[];fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for name in ['old GPU float64','new GPU mixed']:
        common=sorted(checkpoints['CPU'].keys()&checkpoints[name].keys())
        local=[]
        for step in common:
            a=np.load(checkpoints['CPU'][step])['q'];b=np.load(checkpoints[name][step])['q']
            row=dict(engine=name,time_ps=step*c['timestep_fs']/1000,coordinate_rms_A=float(np.sqrt(np.mean((a-b)**2))))
            rows.append(row);local.append(row)
        if local:axes[0].semilogy([r['time_ps'] for r in local],[max(r['coordinate_rms_A'],1e-16) for r in local],label=name)
    common=sorted(set.intersection(*(set(v) for v in checkpoints.values())))
    if common:
        last=common[-1]
        for name in paths:
            q=np.load(checkpoints[name][last])['q'];r,g=rdf(q,box)
            axes[1].plot(r,g,label=name)
            radial.extend(dict(engine=name,time_ps=last*c['timestep_fs']/1000,r_A=float(x),g=float(y)) for x,y in zip(r,g))
    axes[0].set(xlabel='time (ps)',ylabel='native-coordinate RMS vs CPU (Å)',title='Chaotic trajectory divergence');axes[0].legend()
    axes[1].set(xlabel='r (Å)',ylabel='g(r)',title=f'RDF at latest common checkpoint: {common[-1]/1000:g} ps' if common else 'No common checkpoint yet');axes[1].legend()
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(out/'plots/divergence-rdf.png',dpi=170);fig.savefig(out/'plots/divergence-rdf.pdf');plt.close(fig)
    if rows:write_metric_rows(rows,out,family='meam_gpu_diagnostics',name='divergence')
    if radial:write_metric_rows(radial,out,family='meam_gpu_diagnostics',name='rdf')
    print(dict(paired_checkpoints=len(rows),latest_common_ps=common[-1]/1000 if common else None),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    a=parser.parse_args();analyze(read(a.config))


if __name__=='__main__':main()
