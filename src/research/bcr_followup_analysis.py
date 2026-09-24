"""Analyze completed frozen-BCR assays; no model fitting or GPU inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path


def paired_change(reference_error, candidate_error, roots, temperatures, draws, seed):
    """Paired observation-weighted RMSE, bootstrap whole roots within temperature."""
    a, b = np.asarray(reference_error), np.asarray(candidate_error)
    roots, temperatures = np.asarray(roots), np.asarray(temperatures)
    if a.shape != b.shape or a.ndim != 2 or len(a) != len(roots) or len(a) != len(temperatures):
        raise ValueError('Paired error arrays and root/temperature labels must align')
    if len(a) == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Empty or nonfinite paired observations')
    unique = np.unique(roots)
    counts, sums, strata = [], [], []
    for root in unique:
        mask = roots == root
        if len(np.unique(temperatures[mask])) != 1:
            raise ValueError(f'Root spans multiple temperatures: {root}')
        counts.append(mask.sum())
        sums.append([np.square(a[mask]).mean(1).sum(), np.square(b[mask]).mean(1).sum()])
        strata.append(temperatures[mask][0])
    counts, sums, strata = np.asarray(counts), np.asarray(sums), np.asarray(strata)
    if sums[:, 0].sum() <= 0:
        raise ValueError('Relative error requires a positive reference error')
    rng = np.random.default_rng(seed)
    sampled = np.concatenate([rng.choice(ids, size=(draws, len(ids)), replace=True)
                              for temp in np.unique(strata)
                              for ids in [np.flatnonzero(strata == temp)]], axis=1)
    score = np.sqrt(sums.sum(0)/counts.sum())
    boot = np.sqrt(sums[sampled].sum(1)/counts[sampled].sum(1)[:, None])
    if np.any(boot[:, 0] == 0):
        raise ValueError('A root-bootstrap reference has zero error')
    changes = 100*(boot[:, 1]/boot[:, 0]-1)
    lo, hi = np.quantile(changes, [.025, .975])
    return dict(reference_rmse=float(score[0]), candidate_rmse=float(score[1]),
                change_percent=float(100*(score[1]/score[0]-1)),
                ci95_lower_percent=float(lo), ci95_upper_percent=float(hi),
                roots=len(unique), observations=len(a), roots_improved=int((sums[:, 1] < sums[:, 0]).sum()))


def run(config_path):
    config_path = Path(config_path)
    cfg = json.loads(config_path.read_text()); source = resolve_path(cfg['input']); root = resolve_path(cfg['output'])
    status = json.loads((source/'technical/queue-status.json').read_text())
    if status['state'] != 'complete': raise ValueError('Analysis requires the completed matched queue')
    snapshot_metric_docs(root, 'bcr_followup_analysis')
    fingerprints = {}
    def read(path):
        fingerprints[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    selection = read(source/'technical/relaxed-selection.json')
    temperature = {s['lineage']: s['temperature_K'] for s in selection['sources']}
    identity = read(source/'technical/preflight.json')['identity']
    probes = {}
    for domain, folder in [('pilot', 'probes'), ('paired_relaxed', 'relaxed/probes')]:
        for path in sorted((source/'technical'/folder).glob('**/complete.json')):
            meta = read(path)
            if meta['identity'] != identity: raise ValueError(f'Mismatched run identity: {path}')
            if meta['selected_tuning_mse'] > meta['ridge_tuning_mse']+1e-12:
                raise ValueError(f'Residual selection worse than eligible ridge: {path}')
            step = meta['step'] if domain == 'pilot' else meta['encoder_step']
            task = 'melt_to_melt' if domain == 'pilot' else f'{meta["input_domain"]}_to_{meta["target_domain"]}'
            p = path.parent/'predictions.npz'
            fingerprints[str(p.relative_to(source))] = hashlib.sha256(p.read_bytes()).hexdigest()
            with np.load(p) as archive: data = {k: archive[k] for k in archive.files}
            for probe in ('ridge', 'residual'):
                actual = float(np.sqrt(np.square(data[probe]-data['target']).mean()))
                expected = meta['metrics'][probe]['all']['all']['standardized_rmse']
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
            probes[domain, task, meta['representation'], meta['family'], step] = (data, meta)
    if len(probes) != 60: raise ValueError(f'Expected 60 probe pairs, found {len(probes)}')
    rows = []
    def compare(contrast, key_a, key_b, probe_a, probe_b):
        a, _ = probes[key_a]; b, _ = probes[key_b]
        for field in ('indices', 'roots', 'liquid', 'target'):
            np.testing.assert_array_equal(a[field], b[field], err_msg=f'Unpaired {field}: {key_a} vs {key_b}')
        domain, task, representation, family, step = key_b
        temps = np.array([1325. if domain == 'pilot' else temperature[r] for r in a['roots']])
        result = paired_change(a[probe_a]-a['target'], b[probe_b]-b['target'], a['roots'], temps,
                               cfg['bootstrap_draws'], cfg['bootstrap_seed'])
        rows.append(dict(contrast=contrast, domain=domain, task=task, representation=representation,
            family=family, step=step, reference_step=key_a[-1], reference_task=key_a[1],
            reference_representation=key_a[2], probe=probe_b, reference_probe=probe_a, **result))
    for key in probes:
        domain, task, representation, family, step = key
        for probe in ('ridge', 'residual'):
            if step != 0: compare('encoder_training', (*key[:-1], 0), key, probe, probe)
            if representation == 'exported': compare('export_vs_pooled', (domain, task, 'pooled', family, step), key, probe, probe)
            if task == 'relaxed_to_relaxed':
                compare('relaxed_vs_observed_input', (domain, 'observed_to_relaxed', representation, family, step), key, probe, probe)
        compare('residual_vs_ridge', key, key, 'ridge', 'residual')
    comparisons = pd.DataFrame(rows)
    comparisons.to_csv(root/'tables/paired_comparisons.csv', index=False)
    # Matched 1,000->10,000 frozen-encoder decoder difference, beyond existing 0->step tables.
    decoder = {}
    for step in (0, 1000, 10000):
        p = source/f'technical/fresh_decoders/{step}/errors.npz'
        fingerprints[str(p.relative_to(source))] = hashlib.sha256(p.read_bytes()).hexdigest()
        with np.load(p) as d: decoder[step] = {k: d[k] for k in d.files}
        meta = read(p.parent/'complete.json')
        if meta['decoder_updates'] != 10000 or not meta['encoder_unchanged']: raise ValueError('Unmatched decoder fit')
    decoder_rows = []
    for baseline, step in [(0, 1000), (0, 10000), (1000, 10000)]:
        for field in ('indices', 'roots'):
            np.testing.assert_array_equal(decoder[baseline][field], decoder[step][field])
        # Errors already are squared-error means. Square root makes paired_change's
        # internal squaring recover NMSE before converting the ratio back to MSE gain.
        for k, level in enumerate((.01, .02, .04, .08, .12)):
            a, b = [decoder[s]['errors'][k].mean(0) for s in (baseline, step)]
            score = paired_change(np.sqrt(a)[:, None], np.sqrt(b)[:, None], decoder[step]['roots'],
                np.full(len(a), 1325.), cfg['bootstrap_draws'], cfg['bootstrap_seed'])
            gain = lambda percent: 100*(1-(1+percent/100)**2)
            decoder_rows.append(dict(reference_encoder=baseline, encoder=step, noise_over_d0=level,
                reference_nmse=score['reference_rmse']**2, candidate_nmse=score['candidate_rmse']**2,
                gain_percent=gain(score['change_percent']), ci95_lower_percent=gain(score['ci95_upper_percent']),
                ci95_upper_percent=gain(score['ci95_lower_percent']), roots_improved=score['roots_improved']))
    pd.DataFrame(decoder_rows).to_csv(root/'tables/decoder_comparisons.csv', index=False)
    groups = pd.read_csv(source/'tables/probe_groups.csv')
    columns = ['domain', 'input_domain', 'target_domain', 'representation', 'family', 'probe', 'target_group']
    endpoints = groups[groups.step.isin([0, 10000])].pivot(index=columns, columns='step', values='standardized_rmse')
    endpoints['rmse_change_percent'] = 100*(endpoints[10000]/endpoints[0]-1)
    endpoints.reset_index().rename(columns={0:'initial_rmse',10000:'final_rmse'}).to_csv(root/'tables/target_changes.csv',index=False)
    populations = []
    for key, (data, meta) in probes.items():
        if key[2:] != ('exported', 'radial', 0): continue
        populations.append(dict(domain=key[0], task=key[1], observations=len(data['roots']),
            roots=len(np.unique(data['roots'])), liquid=int(data['liquid'].sum())))
    traces = [m for _, m in probes.values()]
    summary = dict(completed=True, fitted_models=0, probe_pairs=len(probes),
        elapsed_training_minutes=(status['finished']-status['started'])/60,
        populations=populations, selected_step_zero=sum(m['selected_step']==0 for m in traces),
        selected_step_5000=sum(m['selected_step']==5000 for m in traces),
        median_tuning_improvement_percent=float(np.median([100*(1-m['selected_tuning_mse']/m['ridge_tuning_mse']) for m in traces])))
    (root/'technical/summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name in ('interventions', 'fresh_decoders', 'probes', 'probe_groups', 'probe_temperatures'):
        p=source/f'tables/{name}.csv';fingerprints[str(p.relative_to(source))]=hashlib.sha256(p.read_bytes()).hexdigest()
    original_reconstruction = resolve_path(cfg['pilot'])/'technical/evaluations/010000/bcr-reconstruction.json'
    (root/'technical/inputs.json').write_text(json.dumps(dict(config=cfg, input_identity=identity,
        original_reconstruction_sha256=hashlib.sha256(original_reconstruction.read_bytes()).hexdigest(),
        hashes=fingerprints, analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    plot(source, root, comparisons, original_reconstruction)
    print(json.dumps(summary,indent=2))


def plot(source, root, comparisons, original_reconstruction):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    p = pd.read_csv(source/'tables/probes.csv')
    p = p[(p.population=='all') & (p.probe=='residual')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), layout='constrained')
    colors = {'pooled':'#2877a6', 'exported':'#c95b3e'}
    ax=axes[0,0]
    for representation, color in colors.items():
        d=p[(p.domain=='pilot')&(p.family=='radial')&(p.representation==representation)].sort_values('step')
        ax.plot(d.step,d.standardized_rmse,'o-',label=f'{representation} ({64 if representation=="pooled" else 128}D)',color=color)
    ax.set(xlabel='BCR encoder updates',ylabel='Standardized radial RMSE ↓',title='A  Structural readout deteriorates'); ax.legend(frameon=False)
    ax=axes[0,1]
    interventions=pd.read_csv(source/'tables/interventions.csv')
    d=interventions[(interventions.step==10000)&(interventions.population=='all')&(interventions.noise_over_d0==.12)]
    codes=['alpha_1','alpha_0','optimized_constant','unrestricted_donor']
    labels=['True code','Training mean','Fitted constant','Wrong code']
    values=[float(d[d.intervention==k].other_nmse.iloc[0]) for k in codes]
    pilot=json.loads(original_reconstruction.read_text())
    labels.append('Separate unconditional');values.append(pilot['levels']['0.12']['unconditional'])
    ax.scatter(values,np.arange(len(values)),color=['#2877a6']*4+['#666666'],s=45)
    for i,value in enumerate(values):ax.annotate(f'{value:.5f}',(value,i),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
    ax.set(yticks=np.arange(len(values)),yticklabels=labels,xlabel='Noise MSE ↓  (σ/d₀ = 0.12)',title='B  One constant retains most performance',xlim=(.622,.657),ylim=(-.5,4.7));ax.invert_yaxis()
    ax=axes[1,0];d=pd.read_csv(root/'tables/decoder_comparisons.csv')
    for step,color in [(1000,'#2877a6'),(10000,'#c95b3e')]:
        q=d[(d.reference_encoder==0)&(d.encoder==step)&(d.noise_over_d0>=.04)]
        ax.errorbar(q.noise_over_d0+(step==10000)*.001,q.gain_percent,
            yerr=np.stack([q.gain_percent-q.ci95_lower_percent,q.ci95_upper_percent-q.gain_percent]),fmt='o-',capsize=3,color=color,label=f'Encoder step {step:,}')
    ax.axhline(0,color='#888888',linewidth=.7);ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set(xlabel='Noise σ/d₀',ylabel='Noise MSE reduction vs initial encoder ↑',title='C  Benefit survives a fresh decoder');ax.legend(frameon=False)
    ax=axes[1,1]
    tasks=['observed_to_observed','relaxed_to_relaxed','observed_to_relaxed']
    for representation,color in colors.items():
        q=comparisons[(comparisons.contrast=='encoder_training')&(comparisons.domain=='paired_relaxed')&(comparisons.family=='radial')&(comparisons.probe=='residual')&(comparisons.representation==representation)].set_index('task').loc[tasks]
        positions=np.arange(3)+(-.08 if representation=='pooled' else .08)
        ax.errorbar(positions,q.change_percent,yerr=np.stack([q.change_percent-q.ci95_lower_percent,q.ci95_upper_percent-q.change_percent]),fmt='o',capsize=3,color=color,label=representation)
    ax.axhline(0,color='#888888',linewidth=.7);ax.yaxis.set_major_formatter(PercentFormatter())
    ax.set(xticks=np.arange(3),xticklabels=['Observed → observed','Relaxed → relaxed','Observed → relaxed'],ylabel='Radial RMSE change after BCR training\npositive = worse',title='D  Transfer audit: 15 development roots')
    ax.tick_params(axis='x',labelsize=9);ax.legend(frameon=False)
    for ax in axes.flat:
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='y',alpha=.15)
    fig.suptitle('BCR follow-up: useful denoising code, weaker structural readout',fontsize=15)
    fig.savefig(root/'plots/overview.png',dpi=180);fig.savefig(root/'plots/overview.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)


if __name__=='__main__':main()
