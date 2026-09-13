"""Report local state forecasts, prospective transitions, timing and source uncertainty."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import write_json
from src.research.forecast_crystallization.local_metrics import (
    classification, counts, first_sustained_onset, onset_metrics, risk_windows,
    select_threshold, source_bootstrap,
)


METHODS = ('ar', 'direct', 'persistence', 'history_mean', 'linear_trend', 'observed_future')


def method_scores(root, selected, observed, anchors, past, future):
    history = observed[..., anchors[:, None]+np.arange(-past, 1)]
    time = np.arange(-past, 1, dtype=np.float64)
    centered = time-time.mean()
    slope = (history*centered).sum(axis=-1)/np.square(centered).sum()
    for method in METHODS:
        if method in ('ar', 'direct'):
            values = np.stack([np.load(root / 'scores' / method / f'source_{s["source_index"]:03d}.npy')
                               for s in selected])
        elif method == 'persistence':
            values = np.broadcast_to(history[..., -1, None], (*history.shape[:-1], future))
        elif method == 'history_mean':
            values = np.broadcast_to(history.mean(axis=-1)[..., None], (*history.shape[:-1], future))
        elif method == 'linear_trend':
            values = history[..., -1, None] + slope[..., None]*np.arange(1, future+1)
        elif method == 'observed_future':
            values = observed[..., anchors[:, None]+np.arange(1, future+1)]
        yield method, values


def export_rows(path, rows):
    flat = [{k: v for k, v in row.items() if not isinstance(v, (dict, list))} for row in rows]
    for exported, row in zip(flat, rows):
        if 'ci95' in row:
            exported['ci95_lower'], exported['ci95_upper'] = row['ci95']
        if 'source_bootstrap' in row:
            for metric, interval in row['source_bootstrap'].items():
                bounds = interval['ci95']
                exported[metric+'_ci95_lower'] = None if bounds is None else bounds[0]
                exported[metric+'_ci95_upper'] = None if bounds is None else bounds[1]
                exported[metric+'_bootstrap_valid'] = interval['valid_repetitions']
    keys = list(dict.fromkeys(k for row in flat for k in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader(); writer.writerows(flat)


def paired_intervals(per_source, repetitions, seed):
    rng = np.random.default_rng(seed)
    output = []
    for horizon in (3, 6, 9):
        baseline = per_source[('history_mean', horizon)]
        indices = rng.integers(len(baseline), size=(repetitions, len(baseline)))
        b = baseline[indices].sum(axis=1)
        bf1 = 2*b[:, 0]/(2*b[:, 0]+b[:, 1]+b[:, 2])
        for method in ('ar', 'direct'):
            m = per_source[(method, horizon)][indices].sum(axis=1)
            f1 = 2*m[:, 0]/(2*m[:, 0]+m[:, 1]+m[:, 2])
            total_m, total_b = per_source[(method, horizon)].sum(0), baseline.sum(0)
            difference = 2*total_m[0]/(2*total_m[0]+total_m[1]+total_m[2])-2*total_b[0]/(2*total_b[0]+total_b[1]+total_b[2])
            output.append(dict(method=method, baseline='history_mean', horizon_ps=horizon,
                event_f1_difference=float(difference), ci95=np.quantile(f1-bf1, [.025, .975]).tolist()))
    return output


def plots(root, states, events, fixed, examples):
    colors = dict(ar='#1967a3', direct='#db6c15', persistence='#777777', history_mean='#389654',
                  linear_trend='#af598d', observed_future='#161616')
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for method in METHODS:
        for ax, rows, metric, label in (
                (axes[0], states, 'balanced_accuracy', 'Future state balanced accuracy'),
                (axes[1], events, 'f1', 'Upcoming local transition F1'),
                (axes[2], fixed, 'recall', 'Transition recall at fixed lead')):
            chosen = [r for r in rows if r['method'] == method and r.get('persistence_frames', 3) == 3]
            ax.plot([r['horizon_ps'] for r in chosen], [r[metric] for r in chosen], marker='o',
                    color=colors[method], label=method, linestyle='--' if method == 'observed_future' else '-')
            ax.set(title=label, xlabel='Horizon / lead (ps)', xticks=[3, 6, 9], ylim=(0, 1))
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    figure.tight_layout(); figure.savefig(root / 'plots/prediction-quality.png', dpi=180); plt.close(figure)
    if examples:
        figure, axes = plt.subplots(len(examples), 1, figsize=(10, 3*len(examples)), squeeze=False)
        for ax, ex in zip(axes[:, 0], examples):
            ax.plot(ex['time'], ex['observed'], color='#888888', label='Observed embedding readout')
            ax.fill_between(ex['time'], -1.3, 1.3, where=ex['crystal'], color='#b3dfc3', alpha=.4, label='Actual PTM crystal')
            for method in ('ar', 'direct'):
                ax.plot(ex['future_time'], ex[method], color=colors[method], marker='.', label=method)
                ax.axhline(ex[method+'_threshold'], color=colors[method], linestyle=':', alpha=.6)
            ax.axvline(ex['onset_ps'], color='black', linestyle='--', label='True sustained onset')
            ax.axvline(ex['origin_ps'], color='#555555', linestyle='-.', label='Forecast origin')
            ax.set(title=ex['title'], xlabel='Time (ps)', ylabel='Crystal readout margin', ylim=(-1.3, 1.3))
            ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=8, ncol=3)
        figure.tight_layout(); figure.savefig(root / 'plots/local-trajectories.png', dpi=180); plt.close(figure)


def analyze(config):
    output = Path(config['output']); root = output / 'technical'
    status = json.loads((root / 'local_prediction_status.json').read_text())
    if status['state'] != 'complete':
        raise RuntimeError('Frozen local forecasts must finish before analysis.')
    observations = json.loads((root / 'local_observations.json').read_text())
    all_sources = observations['sources']
    selected_indices = [i for i, s in enumerate(all_sources) if s['split'] in ('val', 'test')]
    selected = [all_sources[i] for i in selected_indices]
    observed = np.load(root / 'observed_scores.npy')[selected_indices]
    crystal = np.load(root / 'physical_crystal_labels.npy')[selected_indices]
    anchors = np.load(root / 'anchors.npy')
    val_source = np.array([s['split'] == 'val' for s in selected])
    test_source = ~val_source
    test_source_ids = np.array([s['source_index'] for s in selected])[test_source]
    source_grid = np.broadcast_to(np.array([s['source_index'] for s in selected])[:, None, None],
                                  (*crystal.shape[:2], len(anchors)))
    temperatures = np.array([s['temperature_K'] for s in selected])
    cadence = config['cadence_ps']; past = round(config['anchor_history_ps']/cadence)
    future = round(max(config['horizons_ps'])/cadence)
    state_rows, event_rows, fixed_rows, temperature_rows, per_source = [], [], [], [], {}
    thresholds, prediction_arrays = {}, {}
    onsets = {p: first_sustained_onset(crystal, p) for p in config['sensitivity_persistence_frames']}
    risks = {p: risk_windows(crystal, onset, anchors, config['negative_history_frames']) for p, onset in onsets.items()}
    census = []
    for p, onset in onsets.items():
        for split, mask in (('val', val_source), ('test', test_source)):
            for temperature in sorted(set(temperatures[mask])):
                selected_mask = mask & (temperatures == temperature)
                times = onset[selected_mask]
                local_labels = crystal[selected_mask]
                si, ci = np.nonzero(times+8 < crystal.shape[-1])
                continues = local_labels[si[:, None], ci[:, None], times[si, ci, None]+np.arange(9)].all(axis=-1)
                lengths = []
                for source, center in zip(*np.nonzero(times < crystal.shape[-1])):
                    episode = local_labels[source, center, times[source, center]:]
                    lengths.append(len(episode) if episode.all() else int((~episode).argmax()))
                census.append(dict(split=split, temperature_K=int(temperature), persistence_frames=p,
                    trajectories=int(times.size), any_sustained_episode=int(np.sum(times < crystal.shape[-1])),
                    crystal_at_start=int(np.sum(times == 0)), right_censored=int(np.sum(times == crystal.shape[-1])),
                    eligible_origins=int(risks[p][selected_mask].sum()),
                    onset_with_9_frame_followup=len(continues), onset_continues_9_frames=int(continues.sum()),
                    median_initial_episode_frames=float(np.median(lengths)) if lengths else None))
    for method, scores in method_scores(root, selected, observed, anchors, past, future):
        thresholds[method] = {}
        if method in ('ar', 'direct'):
            prediction_arrays[method] = scores[test_source].copy()
        for horizon in config['horizons_ps']:
            steps = round(horizon/cadence)
            truth_state = crystal[..., anchors+steps]
            state_score = scores[..., steps-1]
            threshold, val_f1 = select_threshold(truth_state[val_source].ravel(), state_score[val_source].ravel())
            result = classification(truth_state[test_source].ravel(), state_score[test_source].ravel(), threshold)
            c = [counts(y.ravel(), x.ravel() >= threshold) for y, x in zip(truth_state[test_source], state_score[test_source])]
            result['source_bootstrap'] = source_bootstrap(c, config['bootstrap_repetitions'], config['seed'])
            default = classification(truth_state[test_source].ravel(), state_score[test_source].ravel(), 0.)
            state_rows.append(dict(method=method, horizon_ps=horizon, threshold=threshold, validation_f1=val_f1,
                                   default_zero_threshold_accuracy=default['accuracy'], **result))
            thresholds[method][str(horizon)] = {'state': threshold, 'onset': {}}
            event_score = scores[..., :steps].max(axis=-1)
            for persistence, onset in onsets.items():
                risk = risks[persistence]
                actual = (onset[..., None] <= anchors+steps)
                val_mask = risk & val_source[:, None, None]
                test_mask = risk & test_source[:, None, None]
                threshold, val_f1 = select_threshold(actual[val_mask], event_score[val_mask])
                thresholds[method][str(horizon)]['onset'][str(persistence)] = threshold
                delay = (onset[..., None]-anchors)*cadence
                result, clustered = onset_metrics(actual[test_mask], scores[..., :steps][test_mask], threshold,
                    delay[test_mask], cadence, source_grid[test_mask], test_source_ids, config['bootstrap_repetitions'], config['seed'])
                event_rows.append(dict(method=method, horizon_ps=horizon, persistence_frames=persistence,
                    threshold=threshold, validation_f1=val_f1, **result))
                if persistence == config['persistence_frames']:
                    per_source[(method, horizon)] = clustered
                    for temperature in sorted(set(temperatures[test_source])):
                        mask = test_mask & (temperatures == temperature)[:, None, None]
                        metric, _ = onset_metrics(actual[mask], scores[..., :steps][mask], threshold,
                            delay[mask], cadence, source_grid[mask], test_source_ids[temperatures[test_source] == temperature],
                            config['bootstrap_repetitions'], config['seed'])
                        temperature_rows.append(dict(method=method, horizon_ps=horizon,
                            temperature_K=int(temperature), **metric))
                # One prediction per eligible local event, exactly H ps before onset.
                source, center = np.nonzero(test_source[:, None] & (onset-steps >= anchors[0]) & (onset-steps <= anchors[-1]))
                origin = onset[source, center]-steps-anchors[0]
                eligible = risk[source, center, origin]
                source, center, origin = source[eligible], center[eligible], origin[eligible]
                lead_scores = scores[source, center, origin, :steps]
                metric, _ = onset_metrics(np.ones(len(source), dtype=bool), lead_scores, threshold,
                    np.full(len(source), horizon), cadence, np.array([selected[s]['source_index'] for s in source]),
                    test_source_ids, config['bootstrap_repetitions'], config['seed'])
                fixed_rows.append(dict(method=method, horizon_ps=horizon, persistence_frames=persistence, **metric))
        print(f'Analyzed {method}', flush=True)
    # Select qualitative examples by AR timing outcome, including a miss when present.
    onset = onsets[config['persistence_frames']][test_source]
    actual_scores = observed[test_source]; labels = crystal[test_source]
    test_sources = [s for s in selected if s['split'] == 'test']
    examples, candidates = [], []
    h = str(config['horizons_ps'][-1]); threshold = thresholds['ar'][h]['onset'][str(config['persistence_frames'])]
    for si, ci in zip(*np.nonzero((onset-future >= anchors[0]) & (onset-future <= anchors[-1]))):
        origin = onset[si, ci]-future
        if labels[si, ci, origin-config['negative_history_frames']+1:origin+1].any():
            continue
        forecast = prediction_arrays['ar'][si, ci, origin-anchors[0]]
        crossed = forecast >= threshold
        error = (crossed.argmax()+1-future)*cadence if crossed.any() else None
        candidates.append((si, ci, int(origin), error))
    detected = sorted([c for c in candidates if c[-1] is not None], key=lambda c: abs(c[-1]))
    missed = [c for c in candidates if c[-1] is None]
    chosen = ([detected[0], detected[len(detected)//2]] if detected else []) + missed[:1]
    for si, ci, origin, error in chosen:
        begin, end = max(0, origin-past), min(crystal.shape[-1], int(onset[si, ci])+13)
        frames = np.arange(begin, end)
        ex = dict(time=(frames*cadence).tolist(), observed=actual_scores[si, ci, frames].tolist(),
            crystal=labels[si, ci, frames].tolist(), future_time=((origin+np.arange(1, future+1))*cadence).tolist(),
            onset_ps=float(onset[si, ci]*cadence), origin_ps=float(origin*cadence),
            title=f'Source {test_sources[si]["source_index"]}, center sample {ci}: AR '+('missed' if error is None else f'timing error {error:+.2f} ps'))
        for method in ('ar', 'direct'):
            ex[method] = prediction_arrays[method][si, ci, origin-anchors[0]].tolist()
            ex[method+'_threshold'] = thresholds[method][h]['onset'][str(config['persistence_frames'])]
        examples.append(ex)
    paired = paired_intervals(per_source, config['bootstrap_repetitions'], config['seed'])
    report = dict(protocol=config, test_sources=int(test_source.sum()),
        test_local_trajectories=int(crystal[test_source].shape[0]*crystal.shape[1]),
        state=state_rows, onset=event_rows, fixed_lead=fixed_rows, by_temperature=temperature_rows,
        census=census, paired_event_f1=paired, thresholds=thresholds)
    write_json(root / 'local_results.json', report)
    write_json(root / 'local_examples.json', examples)
    np.savez(root / 'onset_source_statistics.npz', **{f'{k[0]}_{k[1]}ps':v for k, v in per_source.items()})
    snapshot_metric_docs(output, 'forecast_crystallization')
    for name, rows in (('future-state', state_rows), ('local-onset', event_rows), ('fixed-lead', fixed_rows),
                       ('by-temperature', temperature_rows), ('local-events', census), ('paired-event-f1', paired)):
        export_rows(output / 'tables' / (name+'.csv'), rows)
    plots(output, state_rows, event_rows, fixed_rows, examples)
    print('Local analysis complete:', root / 'local_results.json', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    analyze(json.loads(args.config.read_text()))


if __name__ == '__main__':
    main()
