"""Train-only scaling, selection-only readout fitting, source-balanced scoring."""
import copy

import numpy as np
import torch
from torch import nn

from src.data_utils.topology_targets import BLOCKS, topology_loss
from src.research.mace_tda_ridge_audit.math import balanced_errors, ridge_path
from .data import check_deadline


def select_ridge(features, targets, train, selection, alphas, block_scales):
    predictions = ridge_path(features[train], targets[train], features, alphas)
    scores = {alpha: float(balanced_errors(value[selection], targets[selection], block_scales)[0].mean())
              for alpha, value in predictions.items()}
    alpha = min(alphas, key=lambda a: scores[a])
    return predictions[alpha].astype(np.float32), dict(alpha=alpha, selection_scores=scores)


def residual_probe(features, scaled_target, ridge, train, selection, config, *, device='cuda'):
    """A common-capacity nonlinear residual; zero updates is the ridge baseline.

    Only training labels affect gradients and scaling; selection labels choose
    the checkpoint. Test/calibration labels are never accessed in fitting.
    """
    torch.manual_seed(config['seed'])
    mean = features[train].mean(0, dtype=np.float64)
    scale = features[train].std(0, dtype=np.float64)
    scale[scale == 0] = 1.
    x = torch.as_tensor((features-mean)/scale, dtype=torch.float32, device=device)
    y = torch.as_tensor(scaled_target, dtype=torch.float32, device=device)
    base = torch.as_tensor(ridge, dtype=torch.float32, device=device)
    width = config['probe_width']
    model = nn.Sequential(nn.Linear(features.shape[1], width), nn.SiLU(),
                          nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 144)).to(device)
    nn.init.zeros_(model[-1].weight); nn.init.zeros_(model[-1].bias)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['probe_learning_rate'],
                                 weight_decay=config['probe_weight_decay'])
    train = torch.as_tensor(train, device=device)
    selection = torch.as_tensor(selection, device=device)
    generator = torch.Generator(device=device).manual_seed(config['seed'])
    with torch.no_grad():
        best = float(topology_loss(base[selection], y[selection], 'blocks'))
    best_state = copy.deepcopy(model.state_dict()); best_step = 0; stale = 0
    trace = [dict(step=0, selection_loss=best)]
    for step in range(1, config['probe_updates']+1):
        check_deadline(config)
        indices = train[torch.randint(len(train), (config['probe_batch_size'],),
                                     generator=generator, device=device)]
        optimizer.zero_grad(set_to_none=True)
        loss = topology_loss(base[indices]+model(x[indices]), y[indices], 'blocks')
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite TDA probe loss at update {step}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
        optimizer.step()
        if step % config['probe_evaluate_every'] == 0 or step == config['probe_updates']:
            with torch.no_grad():
                value = float(topology_loss(base[selection]+model(x[selection]), y[selection], 'blocks'))
            trace.append(dict(step=step, selection_loss=value))
            if value < best:
                best, best_step, stale = value, step, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= config['probe_patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        prediction = np.concatenate([(base[start:start+1024]+model(x[start:start+1024])).cpu().numpy()
                                     for start in range(0, len(x), 1024)])
    artifact = dict(model={k: v.cpu() for k, v in best_state.items()}, feature_mean=mean,
                    feature_scale=scale, best_step=best_step, updates=step, trace=trace)
    return prediction, artifact


def scores(prediction, target, sources, contexts, block_scales):
    """Equal source weight, then equal rows within source and equal H0/H1/H2."""
    if len(target) == 0:
        return dict(rows=0, sources=0, balanced_mse=None, blocks={})
    names, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
    weights = 1./(len(names)*counts[inverse])
    errors, block_errors = balanced_errors(prediction, target, block_scales)
    y = target.astype(np.float64)
    context_mean = np.empty_like(y)
    for context in np.unique(contexts):
        selected = contexts == context
        context_mean[selected] = y[selected].mean(0)
    blocks = {}
    for d, block in enumerate(BLOCKS):
        mse = float(weights @ np.square(prediction[:, block].astype(np.float64)-y[:, block]).mean(1))
        mean = weights @ y[:, block]
        variance = float(weights @ np.square(y[:, block]-mean).mean(1))
        local_variance = float(weights @ np.square(y[:, block]-context_mean[:, block]).mean(1))
        blocks[f'H{d}'] = dict(mse=mse, scaled_mse=float(weights @ block_errors[:, d]),
            r2=1-mse/variance if variance > 0 else None,
            within_frame_r2=1-mse/local_variance if local_variance > 0 else None)
    return dict(rows=len(y), sources=len(names), balanced_mse=float(weights @ errors), blocks=blocks)
