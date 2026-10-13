"""Paired ancestor-bootstrap contrasts; negative error differences favor response."""
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from .data import root


def run(c):
    rows = []
    for seed in sorted({a['seed'] for a in c['arms']}):
        arms = {a['treatment']:root(c)/'analyses'/a['name']/'technical' for a in c['arms'] if a['seed'] == seed}
        paths = {k:v/'sensitivity-arrays.pt' for k,v in arms.items()}
        if not all(paths[k].exists() for k in ['connected', 'connected_response']): continue
        response = torch.load(paths['connected_response'], map_location='cpu', weights_only=False)
        baseline = torch.load(paths['connected'], map_location='cpu', weights_only=False)
        if response['ancestors'] != baseline['ancestors']: raise ValueError('Unpaired sensitivity cohort')
        ancestors = np.array(response['ancestors']); unique = sorted(set(ancestors))
        rng = np.random.default_rng(seed)
        draws = rng.integers(len(unique), size=(1000, len(unique)))
        def grouped(x): return np.stack([x[ancestors == a].mean(0) for a in unique])
        for g, group in enumerate(['density', 'velocity_shells', 'bond_rates', 'flow']):
            target = grouped(response['values'][(g, 'target_power')])
            error = grouped(response['values'][(g, 'error')])
            for control, raw in [('connected', baseline['values'][(g, 'error')]),
                                 ('zero', response['values'][(g, 'target_power')]),
                                 ('persistence', response['values'][(g, 'persistence_error')])]:
                delta = error-grouped(raw)
                value = delta.mean(0)/target.mean(0).clip(1e-5)
                boot = delta[draws].mean(1)/target[draws].mean(1).clip(1e-5)
                lo, hi = np.quantile(boot, [.025, .975], axis=0)
                for time in range(20):
                    for direction, name in enumerate(['q', 'v']):
                        rows.append(dict(seed=seed, group=group, control=control, horizon_fs=5*(time+1),
                            direction=name, metric='response_minus_control_relative_mse',
                            value=float(value[time, direction]), lo=float(lo[time, direction]),
                            hi=float(hi[time, direction]), ancestors=len(unique)))
    if rows:
        write_metric_rows(rows, root(c)/'analyses'/'comparison', family='measurement_sim', name='paired_sensitivity')
