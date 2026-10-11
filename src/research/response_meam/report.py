"""Publish the saved fidelity and same-device response timings."""
import argparse
import csv
from collections import defaultdict
import statistics

from src.experiment_runner.metric_docs import write_metric_rows
from src.research.local_response.common import read, write_json
from src.project_runtime.paths import resolve_path


def report(c):
    root=resolve_path(c['output'])
    with (root/'analyses/timing-v1/tables/timings.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    with (root/'analyses/validation-v1/tables/validation.csv').open() as stream:
        gates=list(csv.DictReader(stream))
    groups=defaultdict(list)
    for row in rows:
        groups[row['engine'],row['dtype'],int(row['batch'])].append(float(row['seconds_per_shot']))
    summaries=[]
    for (engine,dtype,batch),values in sorted(groups.items()):
        baseline=statistics.mean(groups['mace','float32',batch])
        summaries.append(dict(engine=engine,dtype=dtype,batch=batch,timed_batches=len(values),
            mean_seconds_per_shot=statistics.mean(values),median_seconds_per_shot=statistics.median(values),
            minimum_seconds_per_shot=min(values),maximum_seconds_per_shot=max(values),
            speedup_vs_mace=baseline/statistics.mean(values)))
    analysis=root/'analyses/timing-v1'
    write_metric_rows(summaries,analysis,family='response_meam_torch',name='summary')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(c['batch_sizes']),figsize=(10,4),squeeze=False)
    for axis,batch in zip(axes[0],c['batch_sizes']):
        subset=sorted([x for x in summaries if x['batch']==batch],key=lambda x:({'mace':0,'meam':1}[x['engine']],x['dtype']))
        labels=[f"{x['engine'].upper()}\n{x['dtype']}" for x in subset]
        bars=axis.bar(labels,[x['mean_seconds_per_shot'] for x in subset],color=['#65758b','#168b91','#72bcb6'])
        for bar,row in zip(bars,subset):
            axis.annotate(f"{row['mean_seconds_per_shot']:.2f} s\n{row['speedup_vs_mace']:.2f}×",(bar.get_x()+bar.get_width()/2,bar.get_height()),ha='center',va='bottom',fontsize=10)
        axis.set_title(f'{batch} simultaneous trajectory'+('' if batch==1 else 's'))
        axis.set_ylabel('Seconds per trajectory + two responses')
        axis.set_ylim(0,max(x['mean_seconds_per_shot'] for x in subset)*1.3)
        axis.spines[['top','right']].set_visible(False)
    execution=read(root/'technical/execution.json')
    fig.suptitle('Complete 100-step response calculation on the same GPU')
    fig.text(.5,.01,'24 Å moving environment · 768 features · 4 parents × 2 noise seeds · 3 timing repeats',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.93))
    plot=analysis/'plots/response-speed.png';plot.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(plot,dpi=170);plt.close(fig)
    maxima={key:max(float(x[key]) for x in gates) for key in gates[0]
            if key not in ['parent','source','atoms','passed']}
    write_json(root/'analyses/validation-v1/technical/maxima.json',maxima)
    table='\n'.join(f"| {x['engine'].upper()} {x['dtype']} | {x['batch']} | {x['mean_seconds_per_shot']:.3f} | {x['speedup_vs_mace']:.2f}× |" for x in summaries)
    text=f'''# Differentiable MEAM response benchmark

Implemented the pinned Lee2003 pure-Al 2NN MEAM energy in PyTorch, including
angular screening, reference corrections and LAMMPS pair interpolation. Forces
and two directional force derivatives use automatic differentiation.

Completed: four original unrelaxed training parents, two original noise seeds,
24 Å moving open environments, all768 current future features,100 steps of1fs.
This is a numerical/hardware benchmark. No encoder was trained, and no running
MACE training recipe or historical target bank was replaced.

GPU: {execution['gpu']}, host {execution['host']}.
Three timing repeats, complete-query warmup and synchronization; engine order
rotated. Includes noise generation, integration, neighbors, forces, tangents and
observable derivatives. Excludes startup and disk IO. Repeats are not new samples.

| Engine | Batch | Mean seconds per shot | Speedup vs matched MACE |
| --- | ---: | ---: | ---: |
{table}

Batch1 matches current local24Å production. Batch2 is an explicit throughput
comparison above the old4096 batch-atom budget. Speeds compare different
potentials on identical hardware/protocol, not interchangeable physical laws.

![Response calculation timings](analyses/timing-v1/plots/response-speed.png)

Maximum errors across all four numerical gates:

- Energy: {maxima['energy_per_atom_error_eV']:.3g} eV/atom against LAMMPS.
- Forces: {maxima['force_rms_error_eV_A']:.3g} eV/Å RMS against LAMMPS.
- Force derivative: {100*maxima['force_tangent_relative_error']:.4g}% relative to force finite differences.
- Complete-path values: {maxima['cpu_value_rms']:.3g} RMS against float64 LAMMPS dynamics.
- Complete-path responses: {100*maxima['cpu_response_relative']:.4g}% relative to paired LAMMPS trajectories.
- Float32 value RMS against float64: {maxima['float32_value_rms']:.3g}.
- Float32 response discrepancy against float64: {100*maxima['float32_response_relative']:.4g}%.

[Numerical checks](analyses/validation-v1/tables/validation.csv) ·
[Raw timings](analyses/timing-v1/tables/timings.csv) ·
[Timing definitions](analyses/timing-v1/tables/METRICS.md)

Scope: pure Al with exact checked parameter files; open boundaries. Arbitrary
liquid/crystal coordinates are supported, but other MEAM parameterizations,
alloys and periodic cells are outside this specialization. Piecewise spline and
screening derivatives follow the original local branch; this does not establish
smooth second derivatives at every boundary or long-horizon response stability.
Crystal clustering and nuclei-birth prediction with this teacher remain unmeasured.
'''
    (root/'README.md').write_text(text)
    print(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args();report(read(args.config))


if __name__=='__main__':main()
