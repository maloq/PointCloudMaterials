"""Summarize completed spatial-size and history cohorts without refitting models."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from src.research.forecast_crystallization.local_analyze import export_rows
from .evaluate import directory

PROBABILITY = 'frame_crystal_probability'
CONTEXTS = (8, 32, 128, 512)
HISTORIES = (3, 12)
SEEDS = (20260913, 20260914)


def spatial_name(history, neighbors):
    suffix = '' if neighbors == 8 else str(neighbors)
    return f'history{history}_spatial{suffix}_mixture4'


def values(report, table, method, metric, **selection):
    rows = [r for r in report[table] if r['method'] == method
            and all(r[k] == v for k, v in selection.items())]
    if tuple(sorted(r['seed'] for r in rows)) != SEEDS:
        raise ValueError(f'Expected exactly the two paired seeds: {table}, {method}, {selection}')
    return np.array([r[metric] for r in sorted(rows, key=lambda r: r['seed'])], dtype=float)


def event_values(report, method, metric, horizon=9, persistence=3, table='onset', score=PROBABILITY):
    return values(report, table, method, metric, score_type=score,
                  horizon_ps=horizon, persistence_frames=persistence)


def mean_rows(report, table, metrics):
    groups = defaultdict(list)
    fields = ['method'] if table == 'embedding' else ['method', 'score_type', 'horizon_ps']
    if table in ('onset', 'fixed_lead'):
        fields.append('persistence_frames')
    for row in report[table]:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for key, rows in groups.items():
        if tuple(sorted(r['seed'] for r in rows)) != SEEDS:
            raise ValueError(f'Incomplete or duplicate seed group: {table}, {key}')
        result.append(dict(zip(fields, key), seeds=len(rows),
            **{metric: float(np.mean([r[metric] for r in rows])) for metric in metrics}))
    return result


def summarize(config):
    root = Path(config['output'])
    if root.exists():
        raise FileExistsError(f'Choose a fresh completed-cohort summary output: {root}')
    reports = {key: json.loads(Path(path).read_text()) for key, path in config['comparisons'].items()}
    spatial, history = reports['spatial'], reports['history']
    root = result_folders(root)
    provenance = {str(path): sha256(Path(path)) for path in config['comparisons'].values()}

    def read(path):
        path = Path(path)
        provenance[str(path)] = sha256(path)
        return json.loads(path.read_text())

    # The prior collector already verified paired identities, normalization and assay hashes.
    runs = spatial['protocol']['runs'] + spatial['protocol']['reference_runs']
    details = {}
    for run in runs:
        fit = Path(run['fit'])/'technical'
        local = directory(spatial['protocol'], run)
        for path in (fit/'status.json', local/'status.json'):
            if read(path)['state'] != 'complete':
                raise ValueError(f'Incomplete retained run: {path}')
        metrics = read(fit/'test_metrics.json')
        for row in [r for r in spatial['embedding'] if r['method'] == run['name'] and r['seed'] == run['seed']]:
            if row['mse'] != metrics['source_mean']['mse']:
                raise ValueError(f'Collected score differs from its producer: {fit}')
        training_path = fit/'training.jsonl'
        provenance[str(training_path)] = sha256(training_path)
        details[(run['name'], run['seed'])] = dict(metrics=metrics,
            training=[json.loads(line) for line in training_path.read_text().splitlines()],
            interventions=read(fit/'history_interventions.json'))

    radii = []
    for k in CONTEXTS:
        cache = (Path(spatial['protocol']['spatial_cache']) if k == 8 else
                 Path(next(r for r in runs if r['name'] == spatial_name(3, k))['spatial_cache']))
        records = [r for r in read(cache/'manifest.json')['records'] if r['split'] == 'test']
        radii.append(dict(neighbors=k, test_sources=len(records),
            median_source_outer_radius_A=float(np.median([r['median_outer_distance_A'] for r in records])),
            mean_source_neighbor_distance_A=float(np.mean([r['mean_neighbor_distance_A'] for r in records]))))

    event_metrics = ['precision', 'recall', 'f1', 'accuracy', 'balanced_accuracy', 'average_precision',
        'timing_mae_ps', 'timing_bias_ps', 'timing_p90_absolute_error_ps', 'timed_within_1_5_ps_recall']
    summaries = dict(onset=mean_rows(spatial, 'onset', event_metrics),
        fixed_lead=mean_rows(spatial, 'fixed_lead', ['recall', 'timed_within_1_5_ps_recall', 'timing_mae_ps']),
        state=mean_rows(spatial, 'state', ['accuracy', 'precision', 'recall', 'f1']),
        embedding=mean_rows(spatial, 'embedding', ['mse']), radii=radii)

    observations = read(Path(spatial['protocol']['local_assay'])/'local_observations.json')
    sources = [s for s in observations['sources'] if s['split'] == 'test']
    assert [s['source_index'] for s in sources] == sorted(s['source_index'] for s in sources)
    temperatures = np.array([s['temperature_K'] for s in sources])
    strata = []
    for h in HISTORIES:
        for k in CONTEXTS:
            name = spatial_name(h, k)
            counts = []
            for run in sorted([r for r in runs if r['name'] == name], key=lambda r: r['seed']):
                path = directory(spatial['protocol'], run)/'source-statistics.npz'
                provenance[str(path)] = sha256(path)
                with np.load(path) as arrays:
                    rows = arrays[f'{PROBABILITY}_9ps_p3']
                assert rows.shape == (len(sources), 7), path
                # Verify the saved row interpretation against its original pooled counts.
                reference = next(r for r in spatial['onset'] if r['method'] == name and
                    r['seed'] == run['seed'] and r['score_type'] == PROBABILITY and
                    r['horizon_ps'] == 9 and r['persistence_frames'] == 3)
                np.testing.assert_array_equal(rows[:, :4].sum(0), [reference[x] for x in ('tp','fp','fn','tn')])
                counts.append(rows)
            counts = np.mean(counts, axis=0)
            for temperature in sorted(set(temperatures)):
                mask = temperatures == temperature
                tp, fp, fn, tn, absolute, bias, timed = counts[mask].sum(0)
                strata.append(dict(method=name, history_ps=h, neighbors=k, temperature_K=float(temperature),
                    sources=int(mask.sum()), f1=float(2*tp/(2*tp+fp+fn)),
                    recall=float(tp/(tp+fn)), timed_within_1_5_ps_recall=float(timed/(tp+fn))))
    summaries['temperature'] = strata
    snapshot_metric_docs(root, 'forecast_spatial_mixture')
    for name, rows in summaries.items():
        export_rows(root/'tables'/f'{name}.csv', rows)
    write_json(root/'technical/summary.json', dict(config=config, summaries=summaries,
        source_sha256=provenance, original_comparisons=reports))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titlepad': 12, 'savefig.facecolor': 'white'})
    colors = {3: '#2676a5', 12: '#c75536'}
    figures = []

    def save(fig, name, caption):
        fig.savefig(root/'plots'/f'{name}.png', dpi=180, bbox_inches='tight')
        fig.savefig(root/'plots'/f'{name}.pdf', bbox_inches='tight')
        plt.close(fig)
        figures.append(dict(name=name, caption=caption))

    def spatial_curve(ax, metric, table='onset', factor=1):
        for h in HISTORIES:
            samples = np.stack([values(spatial, table, spatial_name(h,k), metric) if table == 'embedding'
                               else event_values(spatial, spatial_name(h,k), metric) for k in CONTEXTS])*factor
            ax.plot(range(4), samples.mean(1), 'o-', color=colors[h], label=f'{h} ps history', lw=2)
            ax.scatter(np.repeat(np.arange(4), 2)+np.tile([-.045,.045], 4), samples.ravel(),
                       color=colors[h], s=18, alpha=.6)
        ax.set_xticks(range(4), CONTEXTS)
        ax.set_xlabel('Nearby cached centers')
        ax.grid(alpha=.18)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for ax, metric, table, factor, title, label in [
        (axes[0,0], 'f1', 'onset', 100, 'Transition detection improves up to 32 neighbors', '9 ps transition F1 (%)'),
        (axes[0,1], 'timed_within_1_5_ps_recall', 'onset', 100, 'Timing remains the main limitation', 'All-positive-window recall within 1.5 ps (%)'),
        (axes[1,0], 'timing_mae_ps', 'onset', 1, 'Errors among correctly detected transitions', 'Timing MAE (ps; lower is better)'),
        (axes[1,1], 'mse', 'embedding', 1, 'Full-trajectory embedding error', 'Standardized MSE (lower is better)')]:
        spatial_curve(ax, metric, table, factor)
        ax.set(title=title, ylabel=label)
    axes[0,0].legend()
    save(fig, 'spatial-context', '9 ps local-onset assay, three-frame persistence. Lines are two-seed means; small dots are individual seeds. Embedding MSE averages all twelve forecast frames.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained', sharex=True)
    for ax, h in zip(axes, HISTORIES):
        for i, k in enumerate(CONTEXTS[1:]):
            row = next(r for r in spatial['paired_onset'] if r['method'] == spatial_name(h,k)
                and r['baseline'] == spatial_name(h,8) and r['score_type'] == PROBABILITY
                and r['horizon_ps'] == 9 and r['persistence_frames'] == 3)
            mid = row['event_f1_difference']*100; low, high = np.array(row['ci95'])*100
            ax.errorbar(mid, i, xerr=[[mid-low],[high-mid]], fmt='o', color=colors[h], capsize=5, lw=2)
            ax.text(high+.06, i, f'{mid:+.2f}', va='center', fontsize=10)
        ax.axvline(0, color='#555555', lw=1)
        ax.set(yticks=range(3), yticklabels=['32 vs 8','128 vs 8','512 vs 8'],
               title=f'{h} ps observed history', xlabel='Change in transition F1 (percentage points)', xlim=(-2.3,2.7), ylim=(2.5,-.5))
        ax.grid(axis='x', alpha=.18)
    save(fig, 'paired-evidence', 'Saved paired 95% source-bootstrap intervals at 9 ps; seed-averaged confusion counts, 27 test sources. Intervals are exploratory and unadjusted for multiple comparisons.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained', sharey=True)
    best = spatial_name(12,32)
    for ax, table, title in zip(axes, ['onset','fixed_lead'],
            ['Repeated eligible forecast origins', 'One origin per event at a fixed lead']):
        for metric, label, color, style in [('recall','Detected','#2676a5','o-'),
                ('timed_within_1_5_ps_recall','Detected and timed within 1.5 ps','#c75536','s-')]:
            data = np.stack([event_values(spatial,best,metric,horizon=h,table=table) for h in (3,6,9)])*100
            ax.plot([3,6,9], data.mean(1), style, color=color, lw=2, label=label)
            for x, y in zip([3,6,9], data.mean(1)):
                ax.annotate(f'{y:.1f}%', (x,y), xytext=(0,8), textcoords='offset points', ha='center', fontsize=10)
        ax.set(title=title, xlabel='Forecast horizon / exact lead (ps)', xticks=[3,6,9], ylim=(0,80))
        ax.grid(alpha=.18)
    axes[0].set_ylabel('Recall (%)'); axes[0].legend(loc='lower left', fontsize=9)
    save(fig, 'timing-and-lead', '12 ps history / 32 neighbors. The panels have different denominators. At fixed lead, truth occurs at the last predicted frame, so the assay measures early or missed warnings, not late errors.')

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for ax, metric, title in zip(axes, ['f1','timed_within_1_5_ps_recall'],
                                ['9 ps transition F1 (%)','Recall within 1.5 ps (%)']):
        for family, color, label, score in [('deterministic','#7b7d7d','No spatial input, point prediction','mean_margin'),
                ('spatial_mixture4','#2676a5','8 neighbors, four-component mixture',PROBABILITY)]:
            histories = [0,1.5,3,6,12]
            names = ['history'+str(h).replace('.','p')+'_'+family for h in histories]
            samples = np.stack([event_values(history,n,metric,score=score) for n in names])*100
            ax.plot(histories,samples.mean(1),'o-',label=label,color=color,lw=2)
        ax.plot([3,12],[event_values(spatial,spatial_name(h,32),metric).mean()*100 for h in HISTORIES],
                's--',color='#c75536',label='32 neighbors, four-component mixture',lw=2)
        ax.set(title=title, xlabel='Observed history (ps)', xticks=[0,1.5,3,6,12])
        ax.grid(alpha=.18)
    axes[0].legend(fontsize=8,loc='lower right')
    save(fig, 'observed-history', 'Completed matched two-seed history sweep. All conditions use the same eligible origins and 9 ps target. The 32-neighbor condition was measured only at 3 and 12 ps.')

    fig, axes = plt.subplots(2,2,figsize=(11,8),layout='constrained')
    palette = {8:'#7b7d7d',32:'#c75536',128:'#34855e',512:'#2676a5'}
    for k in (8,32,512):
        name=spatial_name(12,k); data=[details[(name,s)] for s in SEEDS]
        axes[0,0].plot(np.arange(1,13)*.75,np.mean([d['metrics']['curves']['mse_by_step'] for d in data],0),
                       color=palette[k],lw=2,label=f'{k} neighbors')
        axes[1,1].plot(range(1,13),np.mean([[e['validation']['nll'] for e in d['training']] for d in data],0),
                       color=palette[k],lw=2,label=f'{k} neighbors')
    axes[0,0].set(title='12 ps history: error along the future path',xlabel='Future time (ps)',ylabel='Standardized MSE')
    axes[0,0].legend(fontsize=9)
    spatial_curve(axes[0,1],'coverage90',table='embedding',factor=100)
    axes[0,1].axhline(90,color='#555555',ls='--',lw=1)
    axes[0,1].set(title='Marginal 90% embedding intervals',ylabel='Empirical coverage (%)',ylim=(89.8,90.4))
    for i,seed in enumerate(SEEDS):
        weights=details[(best,seed)]['metrics']['curves']['component_weight']
        # Component identities cannot be aligned between independently fitted seeds.
        axes[1,0].bar(np.arange(4)+(i-.5)*.34,sorted(weights,reverse=True),width=.32,label=f'Seed {seed}')
    axes[1,0].set(title='32 neighbors: no unused mixture component',xticks=range(4),
                  xticklabels=['1','2','3','4'],xlabel='Within-seed rank by average gate weight',ylabel='Mean gate weight',ylim=(0,.4))
    axes[1,0].legend(fontsize=9)
    axes[1,1].set(title='Validation still improves at epoch 12',xlabel='Epoch',ylabel='Joint path NLL per embedding element')
    for ax in axes.flat: ax.grid(alpha=.15,axis='y')
    save(fig,'trajectory-quality','All curves use completed selected checkpoints or saved validation logs. Marginal embedding coverage is not calibration of physical transition probabilities. Nonzero gate weights do not prove distinct physical modes.')

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',sharey=True)
    for ax,h in zip(axes,HISTORIES):
        for k in (8,32,512):
            rows=[r for r in strata if r['history_ps']==h and r['neighbors']==k]
            ax.plot([r['temperature_K'] for r in rows],[r['f1']*100 for r in rows], 'o-',
                    color=palette[k],label=f'{k} neighbors',lw=2)
        ax.set(title=f'{h} ps history',xlabel='Simulation temperature (K)',xticks=sorted(set(temperatures)))
        ax.grid(alpha=.18)
    axes[0].set_ylabel('9 ps transition F1 (%)');axes[0].legend(fontsize=9)
    save(fig,'temperature','F1 from seed-averaged counts within each temperature, keeping the original globally validation-selected thresholds. Test-source counts: 6 each at 400/450/500/510 K; 3 at 520 K. Descriptive strata; no new threshold tuning.')

    write_json(root/'technical/figures.json',figures)
    body=['# Completed spatial-context and observed-history experiments\n',
          'All scores reuse completed physical assays and selected forecast checkpoints. '
          'See [metric definitions](tables/METRICS.md) and [source provenance](technical/summary.json).\n']
    for f in figures:
        body.extend([f"![{f['name']}](plots/{f['name']}.png)\n",f['caption']+'\n',f"[PDF](plots/{f['name']}.pdf)\n"])
    (root/'README.md').write_text('\n'.join(body))
    html='<!doctype html><meta charset="utf-8"><title>Spatial context results</title><style>body{font:17px system-ui;max-width:1150px;margin:40px auto;padding:0 20px;line-height:1.6;color:#24333d}img{width:100%;height:auto}figure{margin:30px 0 65px}figcaption{max-width:1000px}a{color:#2676a5}</style><h1>Completed spatial-context experiments</h1><p>Matched local-crystallization forecasts: two fitted seeds and 27 independent test sources.</p>'
    for f in figures:
        html+=f'<figure><a href="plots/{f["name"]}.png"><img src="plots/{f["name"]}.png"></a><figcaption>{f["caption"]} <a href="plots/{f["name"]}.pdf">PDF</a></figcaption></figure>'
    (root/'index.html').write_text(html)
    print(f'Completed cohort summary: {root}',flush=True)
