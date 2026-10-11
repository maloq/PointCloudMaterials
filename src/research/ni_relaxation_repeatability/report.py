"""Publish completed numerical rows and a scientific repeatability figure."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.experiment_runner.artifacts import file_hash,write_json
from src.experiment_runner.metric_docs import write_metric_rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    root=a.output;plan=json.loads((root/'technical/plan.json').read_text());rows=[]
    for lane in [0,1]:
        receipt=json.loads((root/f'technical/complete-{lane}.json').read_text())
        if receipt['identity']!=plan['identity']:raise ValueError('Changed diagnostic lane identity')
        rows+=json.loads((root/f'technical/rows-{lane}.json').read_text())
    out=root/'analyses/overview';columns=list(dict.fromkeys(k for r in rows for k in r))
    write_metric_rows(rows,out,family='ni_quench_repeatability',name='quenches',columns=columns)
    sources=[j['source'] for j in plan['jobs']];x=np.arange(len(sources))
    by_comparison={(r['source'],r['profile'],r['comparison']):r for r in rows}
    def values(profile,comparison,key):
        return [by_comparison[s,profile,comparison][key] for s in sources]
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
    for mode,color,label in [('raw32','#aa493d','Original order'),('canonical32','#167d8d','Sorted neighbors')]:
        axes[0].plot(x,values(mode,'independent_repeat','embedding_rms'),'o-',color=color,label=label)
    axes[0].set_yscale('log');axes[0].set_ylabel('Embedding difference RMS (log scale)')
    axes[0].set_title('Repeat the identical input')
    for comparison,color,label in [('input_jitter_1e-07A','#7567a2','Input change: 1e−7 Å'),('input_jitter_0.0001A','#ca9131','Input change: 1e−4 Å')]:
        axes[1].plot(x,values('canonical32',comparison,'coordinate_rms_A'),'o-',color=color,label=label)
    axes[1].set_ylabel('Local coordinate difference RMS (Å)')
    axes[1].set_title('Change the input; use sorted neighbors')
    for ax in axes:
        ax.set_xticks(x,[s.replace('ni-','').replace('-melt',' / melt ') for s in sources],rotation=30,ha='right',fontsize=8)
        ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Ni full-cell FIRE: repeatability is fixed; input sensitivity remains',fontsize=12)
    plots=out/'plots';plots.mkdir(parents=True,exist_ok=True)
    fig.savefig(plots/'repeatability.png',dpi=180);fig.savefig(plots/'repeatability.pdf');plt.close(fig)
    native=[]
    for index,job in enumerate(plan['jobs']):
        # collect() runs the float64-force ablation only on the first endpoint.
        profiles=['canonical32','canonical32_tight']+(['canonical64'] if index==0 else [])
        for profile in profiles:
            folder=Path(plan['config']['archive'])/job['key']/profile
            with np.load(folder/'base-r0/native-quench-state.npz') as first,np.load(folder/'base-r1/native-quench-state.npz') as second:
                native.append(dict(source=job['source'],profile=profile,atoms=len(first['positions']),
                    exact_native_float64_coordinates=bool(np.array_equal(first['positions'],second['positions']))))
    write_json(out/'technical/native-repeat-check.json',dict(checks=native,report_sha256=file_hash(Path(__file__))))
    write_json(out/'technical/complete.json',dict(state='complete',rows=len(rows),identity=plan['identity'],report_sha256=file_hash(Path(__file__))))
    for profile in ['raw32','canonical32','canonical32_tight']:
        subset=[r for r in rows if r['profile']==profile and r['comparison']=='independent_repeat']
        print(profile,{k:[min(r[k] for r in subset),max(r[k] for r in subset)] for k in ['coordinate_rms_A','embedding_rms']})
    print('native exact repeat checks',native)


if __name__=='__main__':main()
