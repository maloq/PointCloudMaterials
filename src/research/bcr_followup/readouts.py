"""Ridge-initialized residual probes; only training roots choose duration."""
import copy

import numpy as np
import torch
from torch import nn

from src.training_methods.bcr.probes import error_metrics
from .common import remaining, write_json


def fit_residual_probe(features, targets, fit, tune, config, seed, device='cpu', deadline=None):
    """No development indices/labels enter optimization, scaling or checkpoint selection."""
    x, y = features[fit].astype(np.float64), targets[fit].astype(np.float64)
    mx, sx = x.mean(0), x.std(0).clip(1e-6)
    my, sy = y.mean(0), y.std(0).clip(1e-6)
    x = np.c_[(x-mx)/sx, np.ones(len(x))]
    d = np.c_[(features[tune]-mx)/sx, np.ones(len(tune))]
    y, dy = (y-my)/sy, (targets[tune]-my)/sy
    best = None
    for alpha in np.logspace(-6, 4, 11):
        penalty = np.eye(x.shape[1])*alpha; penalty[-1, -1] = 0
        weight = np.linalg.solve(x.T@x+penalty, x.T@y)
        score = float(np.square(d@weight-dy).mean())
        if best is None or score < best[0]:
            best = (score, float(alpha), weight)
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(x.shape[1], config['width']), nn.SiLU(),
                          nn.Linear(config['width'], config['width']), nn.SiLU(),
                          nn.Linear(config['width'], y.shape[1])).to(device)
    nn.init.zeros_(model[-1].weight); nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    xt, yt, dt = [torch.as_tensor(v, dtype=torch.float32, device=device) for v in (x, y-x@best[2], d)]
    generator = torch.Generator().manual_seed(seed+1)
    chosen_state, chosen_step, chosen_score = copy.deepcopy(model.state_dict()), 0, best[0]
    trace = [dict(step=0, tuning_mse=best[0])]
    for step in range(1, config['updates']+1):
        if step % config['evaluate_every'] == 0:
            remaining(deadline)
        index = torch.randint(len(x), (min(config['batch_size'], len(x)),), generator=generator).to(device)
        loss = (model(xt[index])-yt[index]).square().mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite residual probe at update {step}')
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % config['evaluate_every'] == 0 or step == config['updates']:
            with torch.no_grad():
                score = float(np.square(d@best[2] + model(dt).cpu().numpy() - dy).mean())
            trace.append(dict(step=step, tuning_mse=score))
            if score < chosen_score:
                chosen_score, chosen_step, chosen_state = score, step, copy.deepcopy(model.state_dict())
    model.load_state_dict(chosen_state)
    return dict(feature_mean=mx, feature_scale=sx, target_mean=my, target_scale=sy, ridge=best[2],
                ridge_alpha=best[1], ridge_tuning_mse=best[0], selected_step=chosen_step,
                selected_tuning_mse=chosen_score, trace=trace,
                residual_state={k: v.cpu() for k, v in chosen_state.items()}, model=model)


def predict_probe(fitted, features, device):
    x = np.c_[(features-fitted['feature_mean'])/fitted['feature_scale'], np.ones(len(features))]
    ridge = x@fitted['ridge']
    with torch.no_grad():
        correction = fitted['model'](torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
    return ridge, ridge+correction


def target_groups(family):
    if family == 'radial':
        names = [f'radial_bin_{i:02d}' for i in range(12)] + ['nearest_distance', 'weighted_radius',
                   'weighted_radius_squared', 'weighted_count', 'weighted_density']
        return {name: [i] for i, name in enumerate(names)}
    if family == 'angular':
        return {name: [i] for i, name in enumerate(('q4', 'w4', 'q6', 'w6'))}
    if family == 'rich':
        return {f'l{degree}': list(range(i*36, (i+1)*36)) for i, degree in enumerate((0, 2, 4, 6))}
    raise ValueError(f'Unknown descriptor family: {family}')


def run(study, device='cpu', deadline=None):
    targets, covariates = study.descriptors()
    chosen = np.array(study.chosen)
    records = [study.records[i] for i in chosen]
    liquid = covariates[chosen, 2] < .35
    for step in study.config['feature_steps']:
        features = study.features(step)
        for representation in ('pooled', 'exported'):
            for family, y in targets.items():
                root = study.technical / 'probes' / f'{step:06d}' / representation / family
                if (root / 'complete.json').exists():
                    continue
                remaining(deadline)
                print(f'Probe: {step}/{representation}/{family}', flush=True)
                fitted = fit_residual_probe(features[representation], y, study.split['fit'], study.split['tune'],
                                             study.config['probes'], study.config['seed'], device, deadline)
                predictions = predict_probe(fitted, features[representation][chosen], device)
                actual = (y[chosen]-fitted['target_mean'])/fitted['target_scale']
                metrics = {}
                for name, prediction in zip(('ridge', 'residual'), predictions, strict=True):
                    metrics[name] = dict(all=error_metrics(actual, prediction, records),
                        liquid=error_metrics(actual[liquid], prediction[liquid], [r for r, v in zip(records, liquid) if v]),
                        groups={k: error_metrics(actual[:, ids], prediction[:, ids], records)
                                for k, ids in target_groups(family).items()})
                root.mkdir(parents=True, exist_ok=True)
                np.savez(root/'predictions.npz', target=actual, ridge=predictions[0], residual=predictions[1],
                         indices=chosen, roots=np.array([r['root'] for r in records]), liquid=liquid)
                torch.save({k: v for k, v in fitted.items() if k != 'model'}, root/'probe.pt')
                write_json(root/'complete.json', dict(identity=study.identity, step=step, representation=representation,
                    family=family, selected_step=fitted['selected_step'], ridge_alpha=fitted['ridge_alpha'],
                    ridge_tuning_mse=fitted['ridge_tuning_mse'], selected_tuning_mse=fitted['selected_tuning_mse'],
                    metrics=metrics))
