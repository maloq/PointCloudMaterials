"""Render saved benchmark tables and refresh campaign figures after both workers."""
import argparse
import csv
import time
from pathlib import Path
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read,write_json,sha
from .report import report
from .diagnostics import analyze


def publish(c,wait=False):
    root=resolve_path(c['output']);tech=root/'technical'
    complete=[tech/'campaign-complete.json',root/'analyses/fast-response-comparison-v1/technical/complete.json']
    deadline=time.monotonic()+2700
    while wait and not all(p.exists() for p in complete):
        status=read(tech/'worker-status.json')
        if status['state']=='failed':raise RuntimeError(f'MD worker failed: {status}')
        if time.monotonic()>deadline:raise TimeoutError('Final publication waited45minutes for both workers')
        write_json(tech/'publication-status.json',dict(state='waiting',missing=[str(p) for p in complete if not p.exists()]))
        time.sleep(20)
    report(c);analyze(c)
    source=root/'analyses/fast-response-comparison-v1/tables/summary.csv'
    if not source.exists():return
    with source.open() as stream:rows=list(csv.DictReader(stream))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained')
    names=['mace-f32','old-meam-f32','new-meam-mixed'];colors=['#718096','#3182ce','#159a82']
    for j,(name,color) in enumerate(zip(names,colors)):
        values=[float(next(r['seconds_per_shot'] for r in rows if r['engine']==name and int(r['batch'])==b)) for b in [1,2]]
        bars=ax.bar(np.arange(2)+(j-1)*.24,values,.24,label=name,color=color)
        ax.bar_label(bars,labels=[f'{v:.3f} s' for v in values],padding=3)
    ax.set(xticks=[0,1],xticklabels=['one shot at a time','two shots at a time'],yscale='log',ylabel='seconds per complete response shot (log scale)',title='100 fs / 768 features / 2 response directions — same GPU')
    ax.spines[['top','right']].set_visible(False);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3)
    plot=root/'analyses/comparison-v4/plots/response-fast-baselines.png';fig.savefig(plot,dpi=170);fig.savefig(plot.with_suffix('.pdf'));plt.close(fig)
    lines=['','## Fastest earlier precision and MACE','',
        '[Same-GPU speed figure](analyses/comparison-v4/plots/response-fast-baselines.png) · [Saved timing definitions](analyses/fast-response-comparison-v1/tables/METRICS.md)','',
        '| Engine | Batch | Seconds/shot |','|---|---:|---:|']
    for r in rows:lines.append(f"| {r['engine']} | {r['batch']} | {float(r['seconds_per_shot']):.4f} |")
    lines+=['','Each number is the mean of four per-parent medians, with both original noise seeds in every timing bundle. Older engines retain float32 integration/features; the new engine retains float64 states/features with float32 forces. MACE uses a different potential and is a runtime reference.','',
        '[Native-coordinate divergence and RDF](analyses/native-diagnostics-v1/plots/divergence-rdf.png). Long-time path divergence is not itself a thermodynamic discrepancy.','']
    phase=root/'analyses/phase-thermodynamics-v1/technical/complete.json'
    if phase.exists():
        metrics=read(phase)
        lines+=['## Verified growth and thermodynamics','',
            '[Growth, energy and temperature figure](analyses/phase-thermodynamics-v1/plots/solid-comparison.png) · [Saved phase comparison](analyses/phase-thermodynamics-v1/tables/summary.csv)','',
            f"Over {metrics['observations']} exact common solid-rich observations ({metrics['first_ps']:g}–{metrics['last_ps']:g} ps), mean mixed-minus-float64 energy is {metrics['potential_mean_difference_eV_per_atom']:.6f} eV/atom and temperature is {metrics['temperature_mean_difference_K']:.3f} K. These satisfy the declared descriptive tolerances; they are not an equivalence test for melting points or nucleation rates.",'']
    with (root/'README.md').open('a') as stream:stream.write('\n'.join(lines))
    write_json(tech/'publication-status.json',dict(state='complete',source_table=str(source),source_sha256=sha(source),renderer_sha256=sha(Path(__file__))))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--wait',action='store_true')
    a=parser.parse_args();publish(read(a.config),a.wait)


if __name__=='__main__':main()
