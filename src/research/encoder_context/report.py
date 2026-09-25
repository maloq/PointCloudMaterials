"""Fixed-order model comparisons; no test-driven selection or ensembles."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from src.research.local_predictability.metrics import source_weights
from src.research.structural_state.common import write_json,sha
from src.experiment_runner.metric_docs import write_metric_table


def paired(base,candidate,draws,seed):
    for key in ('sample_id','source','role','event'):np.testing.assert_array_equal(base[key],candidate[key])
    ids=np.flatnonzero(base['role']=='test');src=base['source'][ids];w=source_weights(src)
    roots,inverse=np.unique(src,return_inverse=True);event=base['event'][ids]
    delta=base['logp'][ids,event]-candidate['logp'][ids,event]
    def score(weight):
        result=[float(weight@delta)];keep=weight>0
        for col in (1,2):
            truth=event<=col
            result.append(float(average_precision_score(truth[keep],candidate['risks'][ids,col][keep],sample_weight=weight[keep])-
                average_precision_score(truth[keep],base['risks'][ids,col][keep],sample_weight=weight[keep])) if weight@truth>0 else np.nan)
        return result
    rng=np.random.default_rng(seed);estimates=[]
    for _ in range(draws):
        m=np.bincount(rng.integers(len(roots),size=len(roots)),minlength=len(roots))
        estimates.append(score(w*m[inverse]))
    a=np.array(estimates);point=score(w)
    return {name:dict(difference=point[k],ci95=np.nanquantile(a[:,k],[.025,.975]).tolist(),
        valid_draws=int(np.isfinite(a[:,k]).sum())) for k,name in enumerate(('event_nll','AP3','AP6'))}


def collect(study):
    rows={};differences={};evidence={}
    for method in study.config['methods']:
        for domain in ('hot','cold'):
            for variant in study.config['variants']:
                key=f'{method}/{domain}/{variant}';folder=study.root/method/f'{domain}-{variant}'/'technical'
                if not (folder/'complete.json').exists():continue
                record=json.loads((folder/'complete.json').read_text())
                if sha(folder/'predictions.npz')!=record['predictions_sha256']:raise ValueError(f'Prediction checksum changed: {folder}')
                rows[key]=json.loads((folder/'metrics.json').read_text());evidence[key]=dict(
                    completion_sha256=sha(folder/'complete.json'),predictions_sha256=record['predictions_sha256'])
                ref=study.root/'scratch'/f'{domain}-{variant}'/'technical'
                if method!='scratch' and (ref/'complete.json').exists():
                    with np.load(ref/'predictions.npz') as a,np.load(folder/'predictions.npz') as b:
                        differences[key]=paired(a,b,study.config['bootstrap'],study.config['seed'])
    metrics=dict(models=rows,paired_against_scratch=differences)
    write_json(study.technical/'comparison.json',metrics)
    write_json(study.technical/'comparison-evidence.json',dict(identity=study.identity,files=evidence))
    write_metric_table(metrics,study.root,family='encoder_context',name='comparison')
    lines=['# Encoder training and directional context','',
        f'{len(rows)} / 16 context fits available. Fixed Al64 all64 population; one seed; no test-based selection.',
        'Likelihood training and validation selection after at least 12 full epochs. AP is a diagnostic only.','',
        '| Encoder initialization | Input | Predictor | Test event NLL ↓ | AP3 ↑ | AP6 ↑ |',
        '| --- | --- | --- | ---: | ---: | ---: |']
    for key,m in rows.items():
        method,domain,variant=key.split('/')
        lines.append(f"| {method} | {'Observed' if domain=='hot' else 'Relaxed'} | {variant} | {m['event_nll']['test']:.5f} | {m['horizons']['3']['test']['average_precision']:.4f} | {m['horizons']['6']['test']['average_precision']:.4f} |")
    lines+=['','Comparisons are paired within the same coordinate domain and predictor. Bootstrap intervals cover sources, not training seeds. These are historically examined test sources.',
        'Structural pretraining treatments use different training populations/budgets; this tests practical pipelines, not an isolated regularizer effect.',
        'The legacy16 subset is exported per model separately and is not interchangeable with all64 scores.','',
        '[Full scores and paired intervals](tables/comparison.csv) · [Definitions](tables/METRICS.md).',
        'Each base encoder contains a full-evaluation table with physical readouts, noise, embedding spectra and (observed inputs only) exact 0.75 ps dynamics.']
    (study.root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plots=study.root/'plots';plots.mkdir(exist_ok=True)
        fig,axes=plt.subplots(1,3,figsize=(15,6),layout='constrained')
        names=list(rows);y=np.arange(len(names))
        for ax,metric,title in zip(axes,('nll','3','6'),('Event NLL (lower better)','3 ps average precision','6 ps average precision'),strict=True):
            values=[m['event_nll']['test'] if metric=='nll' else m['horizons'][metric]['test']['average_precision'] for m in rows.values()]
            ax.scatter(values,y);ax.set_yticks(y,names if ax is axes[0] else []);ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.3)
        fig.savefig(plots/'predictive-comparison.png',dpi=160);plt.close(fig)
    return dict(completed=len(rows),expected=16)
