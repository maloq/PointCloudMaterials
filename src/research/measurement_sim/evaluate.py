"""Held-out path scores, baselines and physical sensitivity; no fitted probes."""
from collections import defaultdict
import time
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from .data import root, write_json, atomic_save, sha
from .train import inputs, response_prediction, FAMILY
from .model import DirectProcess, feature_weights, predictive_scores


def aggregate(values, ancestors, seed=20261015):
    grouped = defaultdict(list)
    for value, ancestor in zip(values, ancestors): grouped[ancestor].append(value)
    means = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(seed)
    bootstrap = means[rng.integers(len(means), size=(1000, len(means)))].mean(1)
    return float(means.mean()), *np.quantile(bootstrap, [.025, .975]).tolist()


def ou(history, rate, steps, particles, seed):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    value = history[:, None, -1].expand(-1, particles, -1)
    decay = torch.exp(-.1*rate); noise = (1-decay.square()).sqrt()
    result = []
    for _ in range(steps):
        value = value*decay+noise*torch.randn(value.shape, device='cuda', generator=generator)
        result.append(value)
    return torch.stack(result, 2)


def run(c, p, arm, deadline=float('inf')):
    out = root(c)/'analyses'/arm['name']; folder = out/'technical'
    if (folder/'evaluation-complete.json').exists(): return True
    data, x, bank, physical, _ = inputs(c)
    saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
    model = DirectProcess(c, data['rate'].cuda()).cuda().eval(); model.load_state_dict(saved['model'])
    weights = feature_weights(data['scale'].cuda()); inc = data['increment_scale'].cuda()
    ids = np.flatnonzero([r['role'] == 'test' for r in data['rows']])
    batch = c['evaluation']['batch_size']; particles = c['evaluation']['particles']
    metrics = defaultdict(list); cases = []; example_ids = set()
    offset_start = 0; checkpoint_sha256 = sha(folder/'best.pt')
    partial = folder/'evaluation-partial.pt'
    if partial.exists():
        old = torch.load(partial, map_location='cpu', weights_only=False)
        if old['checkpoint_sha256'] != checkpoint_sha256: raise ValueError('Changed evaluated checkpoint')
        metrics.update(old['metrics']); cases = old['cases']; offset_start = old['next_offset']
    def save_partial(offset):
        atomic_save(partial, dict(checkpoint_sha256=checkpoint_sha256, next_offset=offset,
                                 metrics=dict(metrics), cases=cases))
    for regime in ['original', 'liquid_liquid', 'growing_front']:
        available = [int(i) for i in ids if data['rows'][i]['regime'] == regime]
        example_ids.update(np.random.default_rng(c['seed']).choice(available, 4, replace=False).tolist())
    with torch.no_grad():
        for offset in range(offset_start, len(ids), batch):
            if time.time() > deadline-300:
                save_partial(offset); return False
            chunk = ids[offset:offset+batch]; observed = x[chunk]
            seed = c['seed']+offset
            predicted = model(observed[:, :4], 60, particles, seed)
            baseline = ou(observed[:, :4], data['rate'].cuda(), 60, particles, seed)
            for method, paths in [('model', predicted), ('ou', baseline)]:
                for steps in [10, 30, 60]:
                    target = observed[:, 4:4+steps]
                    values = paths[:, :, :steps]
                    scores = predictive_scores(values, target, observed[:, 3], weights, inc,
                                                c['training']['likelihood_bandwidth'])
                    for key, value in scores.items(): metrics[(method, steps, key)].append(value.cpu().numpy())
                    mse = ((values.mean(1)[:, -1]-target[:, -1]).square()*weights).sum(-1)
                    metrics[(method, steps, 'endpoint_mse')].append(mse.cpu().numpy())
                    truth_var = target.var(1, unbiased=False)
                    path_var = values.var(2, unbiased=False).mean(1)
                    metrics[(method, steps, 'temporal_power')].append((path_var*weights).sum(-1).cpu().numpy())
                    metrics[(method, steps, 'MD_temporal_power')].append((truth_var*weights).sum(-1).cpu().numpy())
                    metrics[(method, steps, 'mean_temporal_power')].append((values.mean(1).var(1, unbiased=False)*weights).sum(-1).cpu().numpy())
                    covered = (target >= values.quantile(.05, dim=1)) & (target <= values.quantile(.95, dim=1))
                    metrics[(method, steps, 'latent_path_90coverage')].append((covered.float()*weights).sum(-1).mean(-1).cpu().numpy())
            for steps in [10, 30, 60]:
                target = observed[:, 3+steps]
                for method, value in [('mean', torch.zeros_like(target)), ('persistence', observed[:, 3])]:
                    metrics[(method, steps, 'endpoint_mse')].append(((target-value).square()*weights).sum(-1).cpu().numpy())
            for k, index in enumerate(chunk):
                if int(index) in example_ids:
                    cases.append(dict(row=data['rows'][index], truth=data['x'][index],
                        particles=predicted[k].cpu()*data['scale']+data['mean']))
            write_json(folder/'evaluation-progress.json', dict(rows=offset+len(chunk), total=len(ids)))
            if (offset//batch+1) % 10 == 0 or offset+len(chunk) == len(ids): save_partial(offset+len(chunk))
    ancestors = [data['rows'][i]['ancestor'] for i in ids]
    rows = []
    for (method, steps, metric), pieces in metrics.items():
        values = np.concatenate(pieces)
        for regime in ['all', 'original', 'liquid_liquid', 'growing_front']:
            select = np.ones(len(ids), bool) if regime == 'all' else np.array([data['rows'][i]['regime'] == regime for i in ids])
            mean, lo, hi = aggregate(values[select], np.array(ancestors)[select])
            rows.append(dict(model=arm['name'], method=method, horizon_ps=steps*.1, metric=metric,
                regime=regime, value=mean, lo=lo, hi=hi, rows=int(select.sum()), ancestors=len(set(np.array(ancestors)[select]))))
    write_metric_rows(rows, out, family=FAMILY, name='trajectory')
    trajectory_summary = {f'trajectory/{r["method"]}/{r["regime"]}/{r["horizon_ps"]}ps/{r["metric"]}':r['value'] for r in rows}
    atomic_save(folder/'trajectory-arrays.pt', dict(values={key:np.concatenate(v) for key,v in metrics.items()},
                ids=ids, ancestors=ancestors, regimes=[data['rows'][i]['regime'] for i in ids]))
    atomic_save(folder/'example-trajectories.pt', dict(cases=cases, columns=p['columns'],
                checkpoint_update=saved['update'], particles=particles, observation_noise_added=False))

    held = np.flatnonzero([r['role'] == 'test' for r in bank]); measured = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(held), 16):
            chunk = held[start:start+16]
            predicted = response_prediction(model, physical['history'][chunk], physical['pairs'][chunk],
                                    c['training']['response_epsilon'], c['seed']+start, particles=particles)
            half = response_prediction(model, physical['history'][chunk], physical['half_pairs'][chunk],
                                    c['training']['response_epsilon']/2, c['seed']+start, particles=particles)
            target = physical['response'][chunk]
            for g, (a, b) in enumerate([(0, 74), (74, 83), (83, 87), (87, 91)]):
                actual = target[..., a:b, :]; pred = predicted[..., a:b, :]
                measured[(g, 'error')].append((pred-actual).square().mean(2).cpu().numpy())
                measured[(g, 'target_power')].append(actual.square().mean(2).cpu().numpy())
                measured[(g, 'predicted_power')].append(pred.square().mean(2).cpu().numpy())
                measured[(g, 'cross')].append((pred*actual).mean(2).cpu().numpy())
                initial = physical['initial_response'][chunk, None, a:b, :]
                measured[(g, 'persistence_error')].append((initial-actual).square().mean(2).cpu().numpy())
                measured[(g, 'half_error')].append((half[..., a:b, :]-actual).square().mean(2).cpu().numpy())
                measured[(g, 'half_difference')].append((half[..., a:b, :]-pred).square().mean(2).cpu().numpy())
    rows = []; ancestors = [bank[i]['ancestor'] for i in held]
    arrays = {key:np.concatenate(values) for key, values in measured.items()}
    for g, group in enumerate(['density', 'velocity_shells', 'bond_rates', 'flow']):
        for step in range(20):
            for direction, name in enumerate(['q', 'v']):
                values = {key:aggregate(arrays[(g, key)][:, step, direction], ancestors)[0]
                          for key in ['error', 'target_power', 'predicted_power', 'cross', 'persistence_error', 'half_error', 'half_difference']}
                denominator = max(values['target_power'], 1e-5)
                reported = dict(relative_mse=values['error']/denominator,
                    zero_relative_mse=values['target_power']/denominator,
                    persistence_relative_mse=values['persistence_error']/denominator,
                    half_step_relative_mse=values['half_error']/denominator,
                    half_step_difference=values['half_difference']/denominator,
                    response_rms_ratio=np.sqrt(values['predicted_power']/max(values['target_power'], 1e-20)),
                    response_cosine=values['cross']/np.sqrt(max(values['target_power']*values['predicted_power'], 1e-20)),
                    target_power=values['target_power'])
                for metric, value in reported.items():
                    rows.append(dict(model=arm['name'], group=group, horizon_fs=5*(step+1), direction=name,
                        metric=metric, value=float(value), rows=len(held), ancestors=len(set(ancestors))))
    write_metric_rows(rows, out, family=FAMILY, name='physical_sensitivity')
    atomic_save(folder/'sensitivity-arrays.pt', dict(values=arrays, ancestors=ancestors))
    from .plot import plot
    plot(out)
    update_recorded_summary(folder/'wandb.json',
        dict(trajectory_summary, **{f'physical/{r["group"]}/{r["direction"]}/{r["horizon_fs"]}fs/{r["metric"]}':r['value'] for r in rows}),
        evaluation='fixed-measurement-sensitivity-v1', expected=dict(mode='online'))
    write_json(folder/'evaluation-complete.json', dict(state='complete', rows=len(ids), physical_rows=len(held)))
    return True
