"""Explicit particle streams and joint-future scores for frozen Ni simulators."""
import math
import numpy as np
import torch
from .model import nll
from .data import digest


def stream_seed(key, row, particle, stream):
    return int(digest([key, row, particle, stream])[:15], 16)


def noise_tape(model, rows, key, particles, intervals, device):
    # One CPU stream per physical row and particle. Neither batching nor K
    # changes any retained draw; transition and observation noise are separate.
    values = []
    for row in rows:
        values.append(torch.stack([
            torch.randn(intervals, model.substeps, model.d,
                        generator=torch.Generator().manual_seed(
                            stream_seed(key, row, k, 'transition')))
            for k in range(particles)]))
    return torch.stack(values).to(device)


def emission_noise(rows, key, particles, intervals, features, device):
    return torch.stack([torch.stack([
        torch.randn(intervals, features, generator=torch.Generator().manual_seed(
            stream_seed(key, row, k, 'emission')))
        for k in range(particles)]) for row in rows]).to(device)


def rollout(model, x, tape, *, dense=False):
    history = model.encode(x[:, -model.memory:])
    history = history[:, None].expand(-1, tape.shape[1], -1, -1)
    outputs = []
    for t in range(tape.shape[2]):
        current = history[..., -1, :]
        for j in range(model.substeps):
            current = model.microstep(history[..., :-1, :], current, tape[:, :, t, j])
            if dense or j == model.substeps-1:
                outputs.append(model.emit(current))
        history = torch.cat((history[..., 1:, :], current.unsqueeze(-2)), -2)
    return tuple(torch.stack(parts, 2) for parts in zip(*outputs))


def energy(samples, truth):
    """All-distinct-pair U energy score, scaled by sqrt(feature count)."""
    k, d = samples.shape[1], samples.shape[-1]
    distance = torch.cdist(samples, samples, compute_mode='donot_use_mm_for_euclid_dist')
    return ((samples-truth[:, None]).norm(dim=-1).mean(1)
            - distance.sum((1, 2))/(2*k*(k-1))) / math.sqrt(d)


def scramble(values, rows, key):
    """Same time-specific particle permutation for mean, variance and samples."""
    b, k, t, _ = values[0].shape
    index = torch.stack([torch.stack([
        torch.randperm(k, generator=torch.Generator().manual_seed(
            stream_seed(key, row, j, 'scramble')))
        for j in range(t)], 1) for row in rows]).to(values[0].device)
    return tuple(v.gather(1, index[..., None].expand_as(v)) for v in values)


def occupancy(path, initial):
    active = path > initial[..., None, :]+1
    starts = active.clone()
    starts[..., 1:, :] &= ~active[..., :-1, :]
    fraction = active.float().mean(-2)
    duration = .1*active.sum(-2)/starts.sum(-2).clamp_min(1)
    return torch.cat((fraction, duration), -1)


def joint_scores(samples, truth, initial):
    b, k, t, d = samples.shape
    result = {'path_energy': energy(samples.flatten(2), truth.flatten(1))}
    for left, right in [(0, 1), (4, 9), (29, 59)]:
        pair = [left, right]
        result[f'joint_{(left+1)/10:g}_{(right+1)/10:g}ps_energy'] = energy(
            samples[:, :, pair].flatten(2), truth[:, pair].flatten(1))
        result[f'increment_{(left+1)/10:g}_{(right+1)/10:g}ps_energy'] = energy(
            samples[:, :, right]-samples[:, :, left], truth[:, right]-truth[:, left])
    # Both endpoints are uncertain; do not insert the deterministic t0 here.
    result['future_increments_energy'] = energy(
        samples.diff(dim=2).flatten(2), truth.diff(dim=1).flatten(1))
    sx = samples-initial[:, None, None]
    ty = truth-initial[:, None]
    for lag in [1, 5, 10]:
        result[f'lag{lag/10:g}ps_product_energy'] = energy(
            (sx[:, :, :-lag]*sx[:, :, lag:]).mean(2),
            (ty[:, :-lag]*ty[:, lag:]).mean(1))
    result['excursion_energy'] = energy(sx.abs().amax(2), ty.abs().amax(1))
    result['occupancy_dwell_energy'] = energy(
        occupancy(samples, initial[:, None]), occupancy(truth, initial))
    return result


def endpoint_scores(mu, lv, samples, truth):
    return dict(nll=nll(mu[:, :, None], lv[:, :, None], truth[:, None])[:, 0],
                energy=energy(samples, truth),
                mean_mse=(mu.mean(1)-truth).square().mean(-1),
                coverage90=((truth >= samples.quantile(.05, dim=1)) &
                            (truth <= samples.quantile(.95, dim=1))).float().mean(-1),
                process_variance=mu.var(1).mean(-1),
                emission_variance=lv.exp().mean((1, 2)))


def weighted_summary(values, records, seed):
    """Fixed-design weights; ancestry bootstrap keeps every draw's multiplicity."""
    values = np.asarray(values, dtype=float)
    ancestors = np.array([r['ancestor'] for r in records])
    families = np.array([r['regime'] for r in records])
    keys, slot = np.unique(ancestors, return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(keys), size=(500, len(keys)))
    multiplicity = np.stack([np.bincount(d, minlength=len(keys)) for d in draws])
    output = []
    for weighting, group in [('row', ['all']*len(records)),
                             ('family', families), ('ancestor', ancestors),
                             ('ancestor_family', [a+'|'+f for a, f in zip(ancestors, families)])]:
        _, inverse, count = np.unique(group, return_inverse=True, return_counts=True)
        weights = 1/count[inverse]
        weights /= weights.sum()
        boot_weights = multiplicity[:, slot]*weights
        boot = (boot_weights @ values)/boot_weights.sum(1)
        lo, hi = np.quantile(boot, [.025, .975])
        output.append(dict(weighting=weighting, value=float(weights@values),
                           lo=float(lo), hi=float(hi), rows=len(records), ancestors=len(keys)))
    return output


def check():
    """Local end-to-end numerical assay, not a training run or unit-test suite."""
    from .model import make_model
    torch.manual_seed(81)
    model = make_model(dict(dimension=16, history=4, neural_step_fs=5)).eval().requires_grad_(False)
    x = torch.randn(2, 4, 280)
    generator = torch.Generator().manual_seed(19)
    legacy = model.rollout(x, 2, 8, generator)
    generator.manual_seed(19)
    tape = torch.stack([torch.randn(20, 2, 8, 16, generator=generator)
                        for _ in range(2)]).permute(2, 3, 0, 1, 4)
    with torch.no_grad():
        replay = rollout(model, x, tape)
    error = max(float((a-b).abs().max()) for a, b in zip(legacy, replay))
    a = noise_tape(model, ['a', 'b'], 71, 8, 2, 'cpu')
    b = noise_tape(model, ['b'], 71, 4, 2, 'cpu')
    nested = torch.equal(a[1, :4], b[0])
    # Independent, endpoint-correct but temporally false ensemble: perfect
    # persistent +/-1 trajectories become temporally incoherent after scrambling.
    s = torch.tensor([-1., 1.]).repeat(16)[None, :, None, None].expand(1, 32, 60, 24)
    y = s[:, 0]; mu, lv, shuffled = scramble((s, torch.zeros_like(s), s), ['control'], 23)
    marginal_error = float((energy(s[:, :, -1], y[:, -1])-
                            energy(shuffled[:, :, -1], y[:, -1])).abs().max())
    before = joint_scores(s, y, y[:, 0])['future_increments_energy'].item()
    after = joint_scores(shuffled, y, y[:, 0])['future_increments_energy'].item()
    if error > 1e-6 or not nested or marginal_error > 1e-6 or after <= before+.1:
        raise ValueError(dict(rollout_error=error, nested=nested, marginal_error=marginal_error,
                              coherent_increment_score=before, scrambled_increment_score=after))
    return dict(rollout_max_error=error, nested_batch_independent=nested,
                endpoint_scramble_error=marginal_error,
                synthetic_coherent_increment_energy=before,
                synthetic_scrambled_increment_energy=after)
