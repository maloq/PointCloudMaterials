"""Render saved timings and numerical gates, optionally after detached completion."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
import time
import numpy as np

from src.research.local_response.common import read, sha, write_json
from .benchmark import paths, summarize


def table(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def render(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out, _, archive = paths(c)
    analysis = out/'analyses/comparison-v1'
    source = analysis/'tables/engine-comparison.csv'
    rows = table(source)
    matched = [r for r in rows if r['parent']=='0' and r['radius_A']=='24'
               and r['epsilon_A']=='0.01' and r['horizon']=='100fs_joint']
    samples, labels, colors = [], [], []
    reference = [float(r['reference_mace_ad_seconds_per_shot']) for r in matched if r['engine']=='meam']
    if reference:
        samples.append(reference); labels.append('MACE AD\n1 GPU'); colors.append('#5579a7')
    for name, color in [('meam', '#22826c'), ('unep', '#bc6f42')]:
        values = [float(r['bundle_seconds']) for r in matched if r['engine']==name]
        if values:
            samples.append(values); labels.append(f'{name.upper()} FD\n1 CPU core'); colors.append(color)
        parallel = out/'analyses/parallel-v1/tables'/f'{name}-3workers.csv'
        if parallel.exists():
            samples.append([float(r['seconds']) for r in table(parallel)])
            labels.append(f'{name.upper()} FD\n3 CPU cores'); colors.append(color)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios':[1.65,1]})
    for i, (values, color) in enumerate(zip(samples, colors)):
        mean = np.mean(values)
        axes[0].bar(i, mean, color=color, alpha=.8)
        axes[0].scatter(np.linspace(i-.10,i+.10,len(values)),values,s=16,c='#202020',zorder=3)
        axes[0].text(i, max(values)+1, f'{mean:.1f}s',ha='center')
    axes[0].set(xticks=range(len(labels)), xticklabels=labels, ylabel='Seconds per complete response bundle',
                title='24 Å environment: one parent, two noise seeds')
    axes[0].set_ylim(0,max(max(v) for v in samples)*1.16)
    for i,(name,color) in enumerate([('meam','#22826c'),('unep','#bc6f42')]):
        for j,horizon in enumerate(['20fs','100fs_joint']):
            values=[float(r['response_relative_to_mace'])*100 for r in rows
                    if r['engine']==name and r['parent']=='0' and r['radius_A']=='24'
                    and r['epsilon_A']=='0.01' and r['horizon']==horizon]
            if values:
                x=j+(i-.5)*.30
                axes[1].bar(x,np.mean(values),width=.28,color=color,label=name.upper() if j==0 else None)
                axes[1].scatter([x]*len(values),values,s=16,c='#202020',zorder=3)
    axes[1].set(xticks=[0,1],xticklabels=['20 fs','20 + 100 fs'],ylabel='Signed response relative difference (%)',
                title='Changing potential changes the target')
    axes[1].legend(frameon=False)
    fig.suptitle('Force-only local Al response experiments',fontsize=15)
    fig.text(.5,.015,'Preliminary matched-parent pilot. CPU/GPU resources differ. Parallel dots include three timing repeats per seed.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,.94))
    plots=analysis/'plots';plots.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf']:
        fig.savefig(plots/f'force-engines.{suffix}',dpi=180)
    plt.close(fig)
    boundary_path=analysis/'tables/boundary-convergence.csv'
    boundary=table(boundary_path) if boundary_path.exists() else []
    decisions={}
    for name in c['engines']:
        engine_rows=[r for r in rows if r['engine']==name and r['epsilon_A']=='0.01']
        completed=len(list((archive/name).glob('parent*-radius*-shot*.pt')))
        expected=len(c['parents'])*len(c['radii_A'])*2
        fd_ok=all(float(r['fd_relative_to_smallest_epsilon'])<=c['fd_relative_tolerance'] for r in engine_rows)
        accepted=[]
        for radius in c['radii_A'][:-1]:
            candidate=[r for r in boundary if r['engine']==name and r['radius_A']==str(radius)
                       and r['epsilon_A']=='0.01']
            n=len(c['parents'])*2*2*sum(r>radius for r in c['radii_A'])
            if len(candidate)==n and all(r['passed']=='True' for r in candidate):
                accepted.append(radius)
        decisions[name]=dict(completed_shots=completed,expected_shots=expected,
            status='complete' if completed==expected else 'running',
            epsilon_A=.01,fd_convergence_passed_so_far=fd_ok,
            eligible_radius_A=min(accepted) if accepted and fd_ok and completed==expected else None)
    review=analysis/'technical/review.json'
    write_json(review,dict(decisions=decisions,created_utc=datetime.now(timezone.utc).isoformat(),
        table_sha256=sha(source),boundary_sha256=sha(boundary_path) if boundary else None,
        publisher_sha256=sha(Path(__file__)),
        rule='At epsilon0.01 require every source/seed/prefix boundary comparison and FD relative≤0.02 against0.003. No automatic potential replacement or encoder training.'))
    lines=['# Force-only local response experiments','',
        'UNEP and parent MEAM teachers use persistent CPU engines, shared-noise finite differences and the original local Fourier observable.',
        '', '![Matched-parent timing and target change](analyses/comparison-v1/plots/force-engines.png)', '',
        '| Engine | Completed queries | Numerical gate |','| --- | ---: | --- |']
    for name,d in decisions.items():
        verdict=f'Eligible radius: {d["eligible_radius_A"]} Å' if d['eligible_radius_A'] else ('Incomplete' if d['status']=='running' else 'No eligible radius')
        lines.append(f'| {name.upper()} | {d["completed_shots"]}/{d["expected_shots"]} | {verdict} |')
    lines.extend(['','The figure uses the same first training parent and two seeds at24Å. Parallel timings have three hardware repeats per seed; they are not new scientific samples. MACE timings are cached GPU AD measurements. Different resources and different potentials prevent a same-hardware or same-physics speed claim.',
        '', 'All four training parents contribute to the numerical gate. No encoder has been trained with these alternative labels. The original MACE experiment remains separately frozen.',
        '', '[Complete metric table](analyses/comparison-v1/tables/engine-comparison.csv) · [Definitions](analyses/comparison-v1/tables/METRICS.md) · [Numerical gate receipt](analyses/comparison-v1/technical/review.json)',
        '', f'Updated {datetime.now(timezone.utc).isoformat()}.'])
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(decisions,flush=True)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--config',required=True);p.add_argument('--watch',action='store_true')
    args=p.parse_args();c=read(args.config)
    out,_,_=paths(c)
    if args.watch:
        # Wait for stage receipts, which are written after each worker's export.
        # This avoids concurrent table writers during final publication.
        while True:
            stages=[read(out/'technical'/f'{name}-stage.json') for name in c['engines']]
            if any(s['state']=='failed' for s in stages):
                raise RuntimeError('Force-engine worker failed; inspect stage receipt')
            if all(s['state']=='complete' for s in stages):
                break
            time.sleep(30)
        summarize(c)
    render(c)


if __name__=='__main__':
    main()
