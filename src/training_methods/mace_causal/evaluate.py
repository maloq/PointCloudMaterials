"""Physical errors and declared-lag jumps, with whole-source uncertainty."""
import csv
from pathlib import Path

import numpy as np
import torch

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import write_json
from .objective import GROUPS, cumulative_risk, hazard_nll, targets


def encode(model, samples, device):
    return torch.cat([model(s['history'].to(device)) for s in samples])


@torch.no_grad()
def extract(model, heads, samples, norm, device):
    model.eval()
    heads.eval()
    collected = {k: [] for k in ('z', 'present', 'future', 'scale', 'hazard')}
    for sample in samples:
        z = encode(model, [sample], device)
        temperature = torch.tensor([sample['temperature_K']], device=device)
        pred = heads(z, temperature)
        collected['z'].append(z.cpu().numpy())
        for key in ('present', 'future', 'scale', 'hazard'):
            if pred[key] is not None:
                collected[key].append(pred[key].cpu().numpy())
    return {k: np.concatenate(v) if v else None for k, v in collected.items()}


def physical_metrics(result, samples, norm, lags):
    """Per-source physical metrics and strict task errors for model selection."""
    target = {k: v.numpy() for k, v in targets(samples, norm, 'cpu').items()}
    source = np.array([s['source_id'] for s in samples])
    low_order = np.array([s['present'][4] < .30 for s in samples])
    rows, constraint = [], {}
    present_error = (result['present']-target['present'])**2
    future_error = (result['future']-target['future'])**2
    for population, mask in [('all', np.ones(len(samples), bool)), ('low_order', low_order)]:
        for name, section in GROUPS.items():
            tasks = [(f'present/{name}', present_error[:, section].mean(-1), 'encoder')]
            for h, lag in enumerate(lags):
                tasks.extend([(f'future/{lag:g}ps/{name}', future_error[:, h, section].mean(-1), 'encoder'),
                              (f'future/{lag:g}ps/{name}',
                               ((target['present'][:, section]-target['future'][:, h, section])**2).mean(-1),
                               'persistence'),
                              (f'delta/{lag:g}ps/{name}', future_error[:, h, 169+section.start:169+section.stop].mean(-1), 'encoder')])
            for task, error, method in tasks:
                values = []
                for sid in np.unique(source[mask]):
                    chosen = mask & (source == sid)
                    val = float(error[chosen].mean())
                    rows.append(dict(source_id=int(sid), population=population, method=method, metric=task,
                                     value=val, samples=int(chosen.sum())))
                    values.append(val)
                if population == 'all' and method == 'encoder':
                    constraint[task] = float(np.mean(values))
                if population == 'low_order' and method == 'encoder' and values:
                    constraint[f'low_order/{task}'] = float(np.mean(values))
        for h, lag in enumerate(lags):
            for sid in np.unique(source[mask]):
                chosen = mask & (source == sid)
                rows.append(dict(source_id=int(sid), population=population, method='encoder',
                    metric=f'path/{lag:g}ps/mean_std_qbar6', value=float(future_error[chosen, h, 338:].mean()),
                    samples=int(chosen.sum())))
    if result['scale'] is not None:
        residual = result['future']-target['future']
        nll = .5*(residual/result['scale'])**2+np.log(result['scale'])+.5*np.log(2*np.pi)
        coverage = np.abs(residual) <= result['scale']
        for h, lag in enumerate(lags):
            for sid in np.unique(source):
                mask = source == sid
                for metric, value in [('nll', nll[mask, h, :169].mean()), ('coverage_1sigma', coverage[mask, h, :169].mean())]:
                    rows.append(dict(source_id=int(sid), population='all', method='encoder',
                        metric=f'future/{lag:g}ps/{metric}', value=float(value), samples=int(mask.sum())))
    if result['hazard'] is not None:
        logits = torch.from_numpy(result['hazard'])
        risk = cumulative_risk(logits).numpy()
        likelihood = hazard_nll(logits, torch.from_numpy(target['event_bin']),
                                torch.from_numpy(target['observed_bins'])).numpy()
        eligible = target['at_risk'] & (target['observed_bins'] > 0)
        for sid in np.unique(source[eligible]):
            mask = eligible & (source == sid)
            rows.append(dict(source_id=int(sid), population='at_risk', method='encoder', metric='hazard/nll',
                             value=float(likelihood[mask].mean()), samples=int(mask.sum())))
        for k in range(risk.shape[-1]):
            happened = (target['event_bin'] >= 0) & (target['event_bin'] <= k)
            known = target['at_risk'] & (happened | (target['observed_bins'] > k))
            for sid in np.unique(source[known]):
                mask = known & (source == sid)
                rows.append(dict(source_id=int(sid), population='at_risk', method='encoder',
                    metric=f'hazard/bin{k}/brier', value=float(((risk[mask, k]-happened[mask])**2).mean()),
                    samples=int(mask.sum())))
    # Path and local-event skill are also information constraints for E.
    for metric in {r['metric'] for r in rows if r['metric'].startswith(('path/', 'hazard/'))}:
        values = [r['value'] for r in rows if r['metric'] == metric and r['population'] in ('all', 'at_risk')]
        constraint[metric] = float(np.mean(values))
    return rows, constraint


def jump_metrics(z, samples, lag_ps):
    """J = sqrt(E||delta||^2 / (2 tr Cov(z))); within-source populations only."""
    rows = []
    source = np.array([s['source_id'] for s in samples])
    for sid in np.unique(source):
        ids = np.flatnonzero(source == sid)
        for population in ('all', 'low_order'):
            selected = [i for i in ids if population == 'all' or samples[i]['present'][4] < .30]
            pairs = [(i, j) for i in selected for j in selected
                     if samples[i]['center_atom_id'] == samples[j]['center_atom_id']
                     and abs(samples[j]['anchor_ps']-samples[i]['anchor_ps']-lag_ps) < 1e-8]
            if not pairs:
                continue
            trace = float(np.var(z[selected], axis=0).sum())
            square = np.array([np.sum((z[j]-z[i])**2) for i, j in pairs])
            value = float(np.sqrt(square.mean()/(2*trace))) if trace > 0 else None
            for metric, val in [('normalized_rms_jump', value), ('increment_rms', float(np.sqrt(square.mean()))),
                                ('increment_p95', float(np.quantile(np.sqrt(square), .95))), ('covariance_trace', trace)]:
                rows.append(dict(source_id=int(sid), population=population, method='encoder',
                                 metric=f'temporal/{lag_ps:g}ps/{metric}', value=val, samples=len(pairs)))
    return rows


def aggregate(rows, draws, seed):
    rng = np.random.default_rng(seed)
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ('population', 'method', 'metric'))
        if row['value'] is not None:
            groups.setdefault(key, []).append(row['value'])
    result = []
    for (population, method, metric), values in sorted(groups.items()):
        a = np.array(values)
        boot = a[rng.integers(len(a), size=(draws, len(a)))].mean(-1)
        low, high = np.quantile(boot, [.025, .975]) if len(a) > 1 else (None, None)
        result.append(dict(population=population, method=method, metric=metric, value=float(a.mean()),
                           sources=len(a), ci95_low=low, ci95_high=high))
    return result


def jump_criterion(summary, lag_ps, threshold):
    """Report the declared low-order criterion without overriding information gates."""
    if threshold <= 0:
        raise ValueError('Normalized RMS jump threshold must be positive')
    rows = [r for r in summary if r['population'] == 'low_order'
            and r['metric'] == f'temporal/{lag_ps:g}ps/normalized_rms_jump']
    value = rows[0]['value'] if rows else None
    return dict(lag_ps=lag_ps, population='low_order', aggregation='equal-source mean',
                threshold=threshold, value=value, met=None if value is None else bool(value <= threshold))


def export(root, result, samples, norm, config, split, event_thresholds=None):
    root = Path(root)
    rows, constraint = physical_metrics(result, samples, norm, config['future_lags_ps'])
    rows += jump_metrics(result['z'], samples, config['smoothness']['lag_ps'])
    summary = aggregate(rows, config['bootstrap_draws'], config['seed'])
    snapshot_metric_docs(root, 'mace_causal')
    for name, records in [(f'{split}-sources', rows), (split, summary)]:
        with (root/'tables'/f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    np.savez(root/'technical'/f'{split}-predictions.npz',
             **{k: v for k, v in result.items() if v is not None},
             source_id=[s['source_id'] for s in samples], center_atom_id=[s['center_atom_id'] for s in samples],
             anchor_ps=[s['anchor_ps'] for s in samples],
             present_target=np.stack([s['present'] for s in samples]), future_target=np.stack([s['future'] for s in samples]))
    write_json(root/'technical'/f'{split}-constraints.json', constraint)
    write_json(root/'technical'/f'{split}-jump-criterion.json',
               jump_criterion(summary, config['smoothness']['lag_ps'], config['smoothness']['jump_threshold']))
    if result['hazard'] is not None and event_thresholds is not None:
        from .events import event_report
        events = event_report(result['hazard'], samples, config['events']['bin_edges_ps'], event_thresholds)
        with (root/'tables'/f'{split}-events.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(events[0]))
            writer.writeheader(); writer.writerows(events)
    return constraint
