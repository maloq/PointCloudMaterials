"""Forecast scores, causal controls and uncertainty over whole source lineages."""

import math

import numpy as np
import torch

from .model import bin_means, joint_distribution


def relative_gain(candidate, reference):
    """Undefined skill against a perfectly constant reference is reported explicitly as null."""
    denominator = float(np.mean(reference))
    return 1 - float(np.mean(candidate)) / denominator if denominator > 0 else None


def source_bootstrap(candidate, reference, sources, seed, repetitions=2000):
    unique = np.unique(sources)
    a = np.array([candidate[sources == s].mean() for s in unique])
    b = np.array([reference[sources == s].mean() for s in unique])
    gain = relative_gain(a, b)
    if len(unique) < 2 or gain is None:
        return dict(gain=gain, ci95=None, sources=len(unique),
                    interval_status='undefined: fewer than two sources or zero reference error')
    draws = np.random.default_rng(seed).integers(len(unique), size=(repetitions, len(unique)))
    denominators = b[draws].mean(1)
    if np.any(denominators == 0):
        return dict(gain=gain, ci95=None, sources=len(unique),
                    interval_status='undefined: bootstrap includes zero reference error')
    gains = 1 - a[draws].mean(1) / denominators
    return dict(gain=gain, ci95=np.quantile(gains, [0.025, 0.975]).tolist(),
                sources=len(unique), interval_status='exploratory source bootstrap, conditional on fitted seeds')


@torch.inference_mode()
def evaluate(model, loader, mean, scale, device, intervention='real', sample_paths=False, retain_rows=True):
    model.eval()
    rows = {}
    examples = []
    source_sums, source_counts, source_temperatures = {}, {}, {}
    for batch in loader:
        original = (batch['history'].to(device) - mean) / scale
        history = original
        if intervention == 'repeat_anchor':
            history = original[:, -1:].expand_as(original)
        elif intervention == 'reverse_past':
            history = torch.cat((original[:, :-1].flip(1), original[:, -1:]), 1)
        elif intervention != 'real':
            raise ValueError(f'Unknown intervention: {intervention}')
        future = (batch['future'].to(device) - mean) / scale
        target = model.target_values(future)
        output = model(history)
        prediction = output['mean']
        anchor = original[:, -1:]
        times = torch.arange(1, future.shape[1] + 1, device=device, dtype=future.dtype)
        past = torch.arange(original.shape[1], device=device, dtype=future.dtype)
        past = past - past.mean()
        if len(past) > 1:
            slope = (original * past[None, :, None]).sum(1) / past.square().sum()
        else:
            slope = torch.zeros_like(original[:, 0])  # Defined single-observation trend is persistence.
        baselines = dict(persistence=anchor.expand_as(future),
                         history_mean=original.mean(1, keepdim=True).expand_as(future),
                         linear_trend=anchor + times[None, :, None] * slope[:, None],
                         train_mean=torch.zeros_like(future))
        error = (prediction - target).square()
        measures = dict(mse=error.mean((1, 2)), mse_by_step=error.mean(2),
                        raw_mse=(error * scale.square()).mean((1, 2)),
                        delta_energy=(target - anchor).square().mean((1, 2)),
                        predicted_delta_energy=(prediction - anchor).square().mean((1, 2)))
        true_delta, pred_delta = (target - anchor).flatten(1), (prediction - anchor).flatten(1)
        measures['delta_dot'] = (true_delta * pred_delta).sum(1)
        measures['delta_norm_product'] = true_delta.norm(dim=1) * pred_delta.norm(dim=1)
        for name, path in baselines.items():
            values = (model.target_values(path) - target).square()
            measures[f'{name}_mse'] = values.mean((1, 2))
            measures[f'{name}_mse_by_step'] = values.mean(2)
        if model.target == 'trajectory':
            measures['bin_mse'] = (bin_means(prediction, model.edges) - bin_means(future, model.edges)).square().mean(2)
            measures['increment_mse'] = (torch.diff(torch.cat((anchor, prediction), 1), dim=1) -
                                        torch.diff(torch.cat((anchor, future), 1), dim=1)).square().mean((1, 2))
        else:
            measures['bin_mse'] = error.mean(2)
        if model.distribution == 'low_rank_gaussian':
            distribution = joint_distribution(output)
            measures['nll'] = -distribution.log_prob(target.flatten(1)) / target[0].numel()
            variance = output['std'].square() + output['factor'].square().sum(-1).reshape_as(target)
            marginal_std = variance.sqrt()
            residual = (target - prediction) / marginal_std
            measures['coverage90'] = (residual.abs() <= 1.6448536269514722).float().mean((1, 2))
            measures['interval90_width'] = (2 * 1.6448536269514722 * marginal_std).mean((1, 2))
            cdf = 0.5 * (1 + torch.erf(residual / math.sqrt(2)))
            pdf = torch.exp(-residual.square()/2) / math.sqrt(2*math.pi)
            measures['marginal_crps'] = (marginal_std * (residual * (2*cdf-1) + 2*pdf - 1/math.sqrt(math.pi))).mean((1, 2))
            if sample_paths:
                samples = distribution.sample((16,))
                truth_distance = (samples - target.flatten(1)[None]).norm(dim=-1).mean(0)
                paired_distance = (samples[:8] - samples[8:]).norm(dim=-1).mean(0)
                measures['energy_score'] = (truth_distance - 0.5*paired_distance) / math.sqrt(target[0].numel())
        batch_values = {}
        for key, values in measures.items():
            if not torch.isfinite(values).all():
                raise FloatingPointError(f'Nonfinite forecast metric {key}, intervention={intervention}')
            batch_values[key] = values.cpu().numpy()
            if retain_rows:
                rows.setdefault(key, []).append(batch_values[key])
        source_ids = batch['source'].numpy()
        for source in np.unique(source_ids):
            mask = source_ids == source
            source_counts[source] = source_counts.get(source, 0) + int(mask.sum())
            source_temperatures[source] = float(batch['temperature_K'][np.flatnonzero(mask)[0]])
            sums = source_sums.setdefault(source, {})
            for key, values in batch_values.items():
                sums[key] = sums.get(key, 0.0) + values[mask].sum(axis=0, dtype=np.float64)
        if retain_rows:
            for key in ('source', 'atom_id', 'anchor_frame', 'temperature_K'):
                rows.setdefault(key, []).append(batch[key].numpy())
        if not examples:
            examples = [dict(history=batch['history'][:16].numpy(), future=batch['future'][:16].numpy(),
                             prediction=(prediction[:16]*scale+mean).cpu().numpy())]
            if sample_paths and model.distribution == 'low_rank_gaussian':
                examples[0]['sample_paths'] = (samples[:, :16].reshape(16, -1, *target.shape[1:]) * scale + mean).cpu().numpy()
    arrays = {key: np.concatenate(values) for key, values in rows.items()}
    source_ids = np.array(sorted(source_sums))
    count = sum(source_counts.values())
    total = {key: sum(s[key] for s in source_sums.values()) for key in batch_values}
    scalar_keys = [key for key, value in total.items() if value.ndim == 0]
    metrics = dict(samples=count, sources=len(source_ids), intervention=intervention)
    metrics['sample_mean'] = {key: float(total[key] / count) for key in scalar_keys}
    metrics['per_source'] = {str(s): {key: float(source_sums[s][key] / source_counts[s]) for key in scalar_keys}
                             for s in source_ids}
    metrics['source_mean'] = {key: float(np.mean([v[key] for v in metrics['per_source'].values()])) for key in scalar_keys}
    metrics['per_temperature'] = {}
    for temperature in sorted(set(source_temperatures.values())):
        ids = [s for s in source_ids if source_temperatures[s] == temperature]
        metrics['per_temperature'][str(temperature)] = {
            key: float(sum(source_sums[s][key] for s in ids) / sum(source_counts[s] for s in ids)) for key in scalar_keys}
    metrics['curves'] = {key: (value / count).tolist() for key, value in total.items() if value.ndim == 1}
    source_values = {key: np.array([metrics['per_source'][str(s)][key] for s in source_ids]) for key in scalar_keys}
    metrics['skill'] = {name: source_bootstrap(source_values['mse'], source_values[f'{name}_mse'], source_ids, 20260911)
                        for name in baselines}
    energy = total['delta_energy']
    metrics['change_amplitude_ratio'] = float(np.sqrt(total['predicted_delta_energy']/energy)) if energy > 0 else None
    norm = total['delta_norm_product']
    metrics['change_cosine_weighted'] = float(total['delta_dot']/norm) if norm > 0 else None
    metrics['undefined_metrics'] = 'null denotes zero physical-change/reference variance or insufficient source count; no epsilon denominator'
    return metrics, arrays, examples[0]
