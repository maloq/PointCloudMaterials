"""Read-only comparison of completed transfer queues on paired held-out sources."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from src.data.structural_pretraining.prepare import save_json,file_hash
from src.experiment_runner.metric_docs import write_metric_table
from src.research.local_predictability.metrics import hazard_loss,stratified_bootstrap


def collect(root):
    plan=json.loads((root/'technical/plan.json').read_text());records={}
    temperatures={s['id']:s['temperature_K'] for s in plan['sources']}
    for spec in json.loads((root/'technical/queue.json').read_text()):
        folder=root/'technical/runs'/spec['name'];status=json.loads((folder/'status.json').read_text())
        if status['state']!='complete':raise ValueError(f'Unfinished comparison: {folder}')
        metrics=json.loads((folder/'metrics.json').read_text());indices=None;per_source=None
        if spec.get('baseline')!='persistence':
            with np.load(folder/'test-index.npz') as f:indices={k:f[k].copy() for k in f.files}
            logits=np.load(folder/'test-logits.npy')
            losses=hazard_loss(torch.from_numpy(logits),torch.from_numpy(indices['event'])).numpy()
            sources=np.unique(indices['source'])
            per_source=np.array([losses[indices['source']==s].astype(float).mean() for s in sources])
            np.testing.assert_allclose(per_source.mean(),metrics['test_event_nll'],rtol=1e-6)
        records[spec['name']]=dict(spec=spec,status=status,metrics=metrics,indices=indices,per_source=per_source,
            temperatures=temperatures,folder=folder)
    return records


def compare(a,b):
    """A minus B; paired whole-source, temperature-stratified NLL intervals."""
    for key in ('indices','source','event','rows'):np.testing.assert_array_equal(a['indices'][key],b['indices'][key])
    sources=np.unique(a['indices']['source'])
    delta=a['per_source']-b['per_source']
    temperatures=[a['temperatures'][s] for s in sources]
    np.testing.assert_array_equal(temperatures,[b['temperatures'][s] for s in sources])
    ci=stratified_bootstrap(delta,temperatures,draws=5000)
    return dict(a=a['spec']['name'],b=b['spec']['name'],delta_nll=float(delta.mean()),
        relative_change_percent=float(100*delta.mean()/b['per_source'].mean()),ci95=ci.tolist(),sources=len(sources))


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main(initial,scaling,output):
    torch.set_num_threads(1);initial=Path(initial);scaling=Path(scaling);output=Path(output)
    for name in ('technical','tables','plots'):(output/name).mkdir(parents=True,exist_ok=True)
    first=collect(initial);later=collect(scaling)
    def f(name):return first[name]
    def s(name):return later[name]
    base='frozen-H12-R25-attention-E3-S90-W1'
    pairs={
        'radius25_vs0':(s(base),s('frozen-H12-R0-attention-E3-S90-W1')),
        'radius25_vs12':(s(base),s('frozen-H12-R12-attention-E3-S90-W1')),
        'attention_vs_mean':(s(base),s('frozen-H12-R25-mean-E3-S90-W1')),
        'epochs6_vs3':(s('frozen-H12-R25-attention-E6-S90-W1'),s(base)),
        'epochs6_vs1':(s('frozen-H12-R25-attention-E6-S90-W1'),s('frozen-H12-R25-attention-E1-S90-W1')),
        'epochs3_vs1':(s(base),s('frozen-H12-R25-attention-E1-S90-W1')),
        'sources90_vs30':(s(base),s('frozen-H12-R25-attention-E3-S30-W1-matched-updates')),
        'sources90_vs60':(s(base),s('frozen-H12-R25-attention-E3-S60-W1-matched-updates')),
        'windows100_vs25':(s(base),s('frozen-H12-R25-attention-E3-S90-W0.25-matched-updates')),
        'tensor_vs_scalar_E3':(s('frozen-tensor-H12-R25-attention-E3-S90-W1'),s(base)),
        'tensor_vs_scalar_E6':(s('frozen-tensor-H12-R25-attention-E6-S90-W1'),s('frozen-H12-R25-attention-E6-S90-W1')),
        'history12_vs_snapshot':(f('frozen-H12-R25-attention'),f('frozen-H0-R25-attention')),
        'history48_vs12':(f('frozen-H48-R25-attention'),f('frozen-H12-R25-attention')),
        'real48_vs_repeated48':(f('frozen-H48-R25-attention'),f('frozen-H48-R25-attention-repeat')),
    }
    comparisons={name:compare(a,b) for name,(a,b) in pairs.items()}
    write_metric_table(dict(comparisons=comparisons,completed_evaluations=len(first)+len(later)),output,
        family='crystallization_transfer_summary',name='paired_metrics')
    fits=[];horizons=[];hashes={}
    for campaign,records in [('initial',first),('scaling',later)]:
        for record in records.values():
            spec=record['spec'];m=record['metrics'];r9=m['classification'][2];r96=m['classification'][-1]
            fits.append(dict(campaign=campaign,**{k:spec[k] for k in ('name','mode','history_ps','radius_A','aggregation','equivariant')},
                **m.get('training',{}),selection_nll=record['status'].get('best_selection_nll'),test_nll=m['test_event_nll'],
                ap9=r9['average_precision'],auroc9=r9['auroc'],ap96=r96['average_precision'],brier96=r96['brier']))
            for i,row in enumerate(m['classification']):
                horizons.append(dict(campaign=campaign,name=spec['name'],**row,
                    **{'timing_'+k:v for k,v in m['timing'][i].items() if k!='horizon_ps'},
                    **{'spatial_'+k:v for k,v in m['spatial'][i].items() if k!='horizon_ps'}))
            folder=record['folder']
            for name in ('metrics.json','status.json','test-index.npz','test-logits.npy'):
                path=folder/name
                if path.exists():hashes[str(path)]=file_hash(path)
    write_csv(output/'tables/fits.csv',fits);write_csv(output/'tables/horizons.csv',horizons)
    write_csv(output/'tables/comparisons.csv',[dict(comparison=name,**{k:v for k,v in c.items() if k!='ci95'},lower=c['ci95'][0],upper=c['ci95'][1]) for name,c in comparisons.items()])
    save_json(output/'technical/comparisons.json',comparisons);save_json(output/'technical/input-hashes.json',hashes)
    plot(later,output)
    print(json.dumps(comparisons,indent=2))


def plot(records,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for prefix,label,color in [('frozen','Scalar context','#2563eb'),('frozen-tensor','Tensor context','#d97706')]:
        for ax,xs,names,xlabel in [
            (axes[0],[0,6,12,18,25],[f'{prefix}-H12-R{r}-attention-E3-S90-W1' for r in [0,6,12,18,25]],'Context-center radius (Å)'),
            (axes[1],[1,3,6],[f'{prefix}-H12-R25-attention-E{e}-S90-W1' for e in [1,3,6]],'Training budget (full epochs)'),
            (axes[2],[30,60,90],[f'{prefix}-H12-R25-attention-E3-S{n}-W1'+('-matched-updates' if n<90 else '') for n in [30,60,90]],'Independent training sources')]:
            ax.plot(xs,[records[n]['metrics']['test_event_nll'] for n in names],'o-',color=color,label=label)
            ax.set_xlabel(xlabel);ax.set_xticks(xs);ax.grid(alpha=.2);ax.set_ylabel('Test event NLL ↓')
    axes[0].legend(frameon=False)
    fig.suptitle('Frozen MACE, 12 ps history · one seed · same 30 test trajectories')
    fig.savefig(output/'plots/scaling.png',dpi=180);fig.savefig(output/'plots/scaling.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--initial',required=True);p.add_argument('--scaling',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();main(args.initial,args.scaling,args.output)
