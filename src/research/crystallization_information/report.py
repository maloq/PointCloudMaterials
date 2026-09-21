"""Paired source-level conditional-information gains; exploratory, one seed."""
import json
from pathlib import Path
import numpy as np
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json
from src.research.local_predictability.metrics import stratified_bootstrap
from src.experiment_runner.metric_docs import write_metric_table


def compare(base,augmented):
    if set(base['per_source'])!=set(augmented['per_source']):raise ValueError('Cannot pair different test sources')
    sources=sorted(base['per_source']);temperature=[base['per_source'][s]['temperature_K'] for s in sources]
    differences=np.array([base['per_source'][s]['event_nll']-augmented['per_source'][s]['event_nll'] for s in sources])
    result={'event_nll_gain':float(differences.mean()),'event_nll_gain_ci95':dict(zip(('lower','upper'),stratified_bootstrap(differences[:,None],temperature,draws=1000)[:,0].tolist()))}
    for h in base['classification']:
        delta=np.array([[base['per_source'][s][h][k]-augmented['per_source'][s][h][k] for k in ('log_loss','brier')] for s in sources])
        ci=stratified_bootstrap(delta,temperature,draws=1000)
        result[h]={k:dict(gain=float(delta[:,j].mean()),ci95=ci[:,j].tolist()) for j,k in enumerate(('log_loss','brier'))}
        result[h]['ap_gain']=augmented['classification'][h]['average_precision']-base['classification'][h]['average_precision']
    return result


def report(config):
    root=resolve_path(config['output']);fits={}
    for p in (root/'technical/fits').glob('*/*/*/metrics.json'):
        m=json.loads(p.read_text());t=m['task'];fits[(t['encoder'],t['variant'],t['readout'])]=m
    gains={}
    lines=['# What information is missing for crystallization within 12 ps?','',
        'Frozen encoders; same natural at-risk windows and historical source splits. '
        'One seed. All predictors have matched input slots and hidden width within a readout family. '
        'No encoder training or test-based checkpoint selection. Positive NLL gain means an added block helps.', '',
        '| Encoder | Input | Readout | Event NLL | 3 ps AP | 6 ps AP | 9 ps AP | 12 ps AP | Paired NLL gain [95% source interval] |',
        '|---|---|---|---:|---:|---:|---:|---:|---|']
    for key,m in sorted(fits.items()):
        if m['task']['task']!='hazard':continue
        name,variant,head=key;base=fits.get((name,'z',head));gain='—'
        if base and variant!='z':
            g=compare(base,m);gains['--'.join(key)]=g;ci=g['event_nll_gain_ci95']
            gain=f"{g['event_nll_gain']:.5f} [{ci['lower']:.5f}, {ci['upper']:.5f}]"
        ap=[m['classification'][h]['average_precision'] for h in ('3.0','6.0','9.0','12.0')]
        lines.append(f'| {name} | {variant} | {head} | {m["event_nll"]:.5f} | '+ ' | '.join('—' if v is None else f'{v:.4f}' for v in ap)+f' | {gain} |')
    lines.extend(['','## Physical information retained in the export','',
        '| Encoder | Decoder | Feature block | Test standardized MSE | Test R² |','|---|---|---|---:|---:|'])
    for (name,variant,head),m in sorted(fits.items()):
        if m['task']['task']!='decoder':continue
        for block,score in m['groups'].items():
            r='—' if score['r2'] is None else f'{score["r2"]:.4f}'
            lines.append(f'| {name} | {head} | {block} | {score["standardized_mse"]:.4f} | {r} |')
    lines.extend(['','## Interpretation','',
        '- An add-back gain measures information inaccessible to the tested frozen readout; it is not proof of information-theoretic absence.',
        '- Compare linear, MLP and stronger embedding-only readouts: recovery with a stronger head suggests accessibility/optimization rather than missing input.',
        '- All additive comparisons use the same input dimensionality, initial weights, update count and source-balanced batches; absent groups are zeroed. The shuffled full-block negative control preserves role and temperature, but is deliberately unpaired with the target observation.',
        '- Weak physical decoding plus a positive corresponding add-back gain identifies a candidate representation deficiency. Good decoding plus a gain points toward readout organization/optimization.',
        '- Velocity, observed history and 7–25 Å outer shells are additional inputs beyond a position-only local snapshot. Their gains identify missing observation context, not necessarily compression failure.',
        '- Current packet support and raw bond-order neighborhoods differ from the encoder crop; even geometry gains can reflect observation support.',
        '- Confidence intervals use paired temperature-stratified whole-source resampling. They exclude training-seed uncertainty, are exploratory and are not corrected for the many comparisons. AP gains are point estimates.',
        '- Timing metrics include missed event windows; MAE alone excludes misses. The 3 ps origin spacing limits timing resolution.',
        '- No future observations, PTM labels or sustained-event confirmation enter predictors. Those frames define labels only. No new simulations or TDA targets were generated.',
        '',f'Completed readouts: {len(fits)}. Missing rows remain pending.'])
    save_json(root/'technical/paired-gains.json',gains)
    write_metric_table(gains,root,family='crystallization_information',name='paired-gains')
    (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
