"""Frozen, method-native diagnostics of temporal representations.

These are prospective heldout-pair tests on the enriched native Ni population,
not equilibrium timescales, a Markov-state-model validation, or nucleus rates.
Train-only linear VAMP bases are scored with the VAMP-E functional of Wu/Noe,
https://arxiv.org/pdf/1707.04659, Eq. 30 and Appendix H. Reported scores omit the
constant singular function. Fixed ranks are diagnostics, never model selection.

SelfTICA heads are evaluated using their published off-diagonal objective;
FMRC heads use their actual noisy Gaussian path regression and an independent
forward-head ODE sampler. The latter is a conditional future marginal, not a
joint trajectory. Neither flow-matching loss nor latent density is called NLL.
The common Gaussian readouts instead score exactly the SAME train-normalized
244 measurement targets. Their history advantage is evidence of information
available to this fixed linear probe, not a proof of non-Markovianity.

All test pairs start at native frame 3. All valid TRAIN origins >=3 fit probes
and VAMP bases. Source/regime shuffling breaks pairing while retaining that
mixture; shuffled test values are a diagnostic counterfactual, not an available
forecast. The present-liquid slice is only an evaluation annotation. No labels
fit the encoders, probes, covariance bases, or shuffled-pair permutations.
"""
from collections import defaultdict
import fcntl
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

from src.experiment_runner.artifacts import file_hash, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.research.continuous_motion.data import atomic_save, digest, read, root
from src.research.ni_kinetic_latent import assessment as native
from src.research.ni_kinetic_latent.data import GROUPS, GROUP_NAMES, inverse
from src.temporal_vamp.linear_vamp import LinearVAMP


DEFAULTS = dict(lags=[1, 5, 10, 30, 60], operator_lags=[1, 5, 10, 30],
    native_lags=[5, 10], ranks=[2, 4, 8, 16, 32], ridge=.001,
    covariance_cutoff=1e-8, variance_floor=1e-4, field_draws=4,
    particles=32, flow_steps=64, flow_path_sigma=.001, batch_size=64,
    encode_batch_size=4096, fit_batch_size=8192, bootstrap=1000,
    per_source_regime=4)


def _time_left(deadline):
    return time.time() < deadline-30


def _packet(path, callback, deadline):
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    if not _time_left(deadline):
        return None
    result = callback()
    atomic_save(path, result)
    return result


def _ancestors(bank, ids):
    return np.asarray([bank['rows'][i]['ancestor'] for i in ids])


def _order_thresholds(bank):
    columns = [bank['columns'].index(key) for key in native.ORDER[:2]]
    values = bank['x'][bank['train_rows'], 3:][:, :, columns].flatten(0, 1)
    return columns, values.median(0).values


def _permutation(bank, ids, seed, matched=False, frame=3):
    """Nonidentity cyclic shifts; optional TRAIN-median current-order bins.

    The matched control adds the two binary bins qbar6/coherence. Singleton
    strata keep their own condition and are counted, never globally reassigned.
    """
    columns, thresholds = _order_thresholds(bank)
    bins = (bank['x'][ids, frame][:, columns] > thresholds).numpy()
    groups = defaultdict(list)
    for j, i in enumerate(ids):
        row = bank['rows'][i]
        key = (row['ancestor'], row['regime'])
        if matched:
            key += tuple(bins[j])
        groups[key].append(j)
    rng = np.random.default_rng(seed)
    result = np.arange(len(ids))
    singletons = 0
    for key in sorted(groups):
        members = np.asarray(groups[key])
        if len(members) == 1:
            singletons += 1
            continue
        members = rng.permutation(members)
        result[members] = np.roll(members, rng.integers(1, len(members)))
    return result, singletons


def _scopes(bank, ids, labels):
    yield 'all', np.ones(len(ids), bool)
    yield 'present_liquid', (~labels['crystal'][ids, 3] &
                            (labels['fraction'][ids, 3] <= .05)).numpy()


def _aggregate(values, bank, ids, labels, settings, seed, metadata):
    """Equal ancestor means; undefined ancestors are never silently discarded."""
    rows, sources = [], []
    ancestors = _ancestors(bank, ids)
    for scope, mask in _scopes(bank, ids, labels):
        if not mask.any():
            continue
        for metric, value in values.items():
            chosen = np.asarray(value)[mask]
            info = dict(metadata, scope=scope, metric=metric)
            rows.append(dict(info, rows=int(mask.sum()),
                **native.estimate(chosen, ancestors[mask], seed, settings['bootstrap'])))
            keys, means = native.source_values(chosen, ancestors[mask])
            sources.extend(dict(info, ancestor=str(key), value=float(mean))
                           for key, mean in zip(keys, means))
    return dict(rows=rows, sources=sources)


@torch.no_grad()
def _encode(model, values, settings):
    flat = values.flatten(0, 1)
    device = next(model.parameters()).device
    parts = [model.encode(flat[start:start+settings['encode_batch_size']].to(device)).cpu()
             for start in range(0, len(flat), settings['encode_batch_size'])]
    result = torch.cat(parts).reshape(*values.shape[:2], -1)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite encoded observations in kinetics diagnostics')
    return result


def _train_pairs(values, train, lag):
    return (values[train, 3:-lag].flatten(0, 1).numpy(),
            values[train, 3+lag:].flatten(0, 1).numpy())


def _vamp_score(moment, singular):
    ff, gg, fg = moment
    return 2*np.dot(singular, np.diag(fg))-np.sum(
        singular[:, None]*singular[None, :]*ff*gg.T)


def _operator(values, bank, labels, settings, seed, name, lag):
    train, ids = bank['train_rows'], bank['test_rows']
    with threadpool_limits(limits=2):
        estimator = LinearVAMP(settings['ridge'], settings['covariance_cutoff'])
        estimator.fit(*_train_pairs(values, train, lag))
        rank = min(max(settings['ranks']), estimator.rank)
        f = estimator.left_singular_functions(values[ids, 3].numpy(), rank)
        g = estimator.right_singular_functions(values[ids, 3+lag].numpy(), rank)
    permutation, singletons = _permutation(bank, ids, seed+lag)
    matched, matched_singletons = _permutation(bank, ids, seed+lag, matched=True)
    ancestors = _ancestors(bank, ids)
    rows, source_rows, moments = [], [], {}
    for scope, mask in _scopes(bank, ids, labels):
        if not mask.any():
            continue
        keys = np.unique(ancestors[mask])
        packets = {}
        for pairing, right in (('paired', g), ('shuffled', g[permutation]),
                               ('order_matched_shuffled', g[matched])):
            packets[pairing] = np.stack([np.stack((
                f[m].T@f[m], right[m].T@right[m], f[m].T@right[m]))/m.sum()
                for key in keys if (m := mask & (ancestors == key)).any()])
        moments[scope] = dict(ancestors=keys, **packets)
        draws = np.random.default_rng(seed).integers(len(keys), size=(settings['bootstrap'], len(keys)))
        for k in settings['ranks']:
            for pairing in ('paired', 'shuffled', 'order_matched_shuffled',
                            'paired_minus_shuffled', 'paired_minus_order_matched_shuffled'):
                info = dict(model=name, horizon_ps=lag*.1, rank=k, scope=scope,
                    pairing=pairing, metric='centered_vamp_e', rows=int(mask.sum()),
                    ancestors=len(keys), available_rank=estimator.rank,
                    shuffle_singletons=singletons,
                    matched_shuffle_self_donor_fraction=matched_singletons/len(ids))
                if k > estimator.rank:
                    rows.append(dict(info, value=None, lo=None, hi=None))
                    continue
                s = estimator.singular_values_[:k]
                pair = {key: value[:, :, :k, :k] for key, value in packets.items()}
                def score(mean):
                    if pairing.startswith('paired_minus_'):
                        return _vamp_score(mean['paired'], s)-_vamp_score(mean[pairing[13:]], s)
                    return _vamp_score(mean[pairing], s)
                value = score({key: item.mean(0) for key, item in pair.items()})
                boot = np.array([score({key: item[draw].mean(0) for key, item in pair.items()})
                                 for draw in draws])
                lo, hi = np.quantile(boot, [.025, .975])
                rows.append(dict(info, value=float(value), lo=float(lo), hi=float(hi)))
                # Source-local values aid inspection but are NOT an additive
                # decomposition of the pooled nonlinear VAMP-E functional.
                source_rows.extend(dict(info, ancestor=str(key),
                    value=float(score({label: item[j] for label, item in pair.items()})))
                    for j, key in enumerate(keys))
    return dict(rows=rows, sources=source_rows, estimator=estimator, moments=moments)


def _feature_batches(features, targets, lag, history, batch):
    origins = targets.shape[1]-3-lag
    for start in range(0, len(targets)*origins, batch):
        index = torch.arange(start, min(start+batch, len(targets)*origins), device=features.device)
        row, frame = index//origins, index % origins+3
        if history == 4:
            x = features[row[:, None], frame[:, None]+torch.arange(-3, 1, device=features.device)]
            x = x.flatten(1)
        else:
            x = features[row, frame]
        yield x, targets[row, frame+lag]


@torch.no_grad()
def _ridge_fit(features, targets, lag, history, settings, device):
    """All training pairs, float32 matrix products and float64 accumulation.

    Coordinate scales, coefficients and diagonal residual variances are fitted
    only on TRAIN. This is a frozen diagnostic, never likelihood calibration on
    test. Ridge is applied to standardized input covariance with fixed strength.
    """
    features, targets = features.to(device), targets.to(device)
    d = features.shape[-1]*history
    n = len(features)*(features.shape[1]-3-lag)
    sx = torch.zeros(d, dtype=torch.float64, device=device)
    sx2 = torch.zeros_like(sx)
    sy = torch.zeros(targets.shape[-1], dtype=torch.float64, device=device)
    for x, y in _feature_batches(features, targets, lag, history, settings['fit_batch_size']):
        sx += x.double().sum(0); sx2 += x.double().square().sum(0)
        sy += y.double().sum(0)
    mean = sx/n
    scale = (sx2/n-mean.square()).clamp_min(0).sqrt()
    scale = torch.where(scale > 1e-6, scale, 1.)
    target_mean = sy/n
    xx = torch.zeros((d, d), dtype=torch.float64, device=device)
    xy = torch.zeros((d, targets.shape[-1]), dtype=torch.float64, device=device)
    for x, y in _feature_batches(features, targets, lag, history, settings['fit_batch_size']):
        a, b = ((x-mean)/scale).float(), (y-target_mean).float()
        xx += (a.T@a).double(); xy += (a.T@b).double()
    covariance = xx/n
    coefficient = torch.linalg.solve(covariance+settings['ridge']*torch.eye(d, device=device), xy/n)
    residual = torch.zeros_like(target_mean)
    for x, y in _feature_batches(features, targets, lag, history, settings['fit_batch_size']):
        error = y.double()-target_mean-((x.double()-mean)/scale)@coefficient
        residual += error.square().sum(0)
    variance = (residual/n).clamp_min(settings['variance_floor'])
    return {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in
        dict(mean=mean, scale=scale, target_mean=target_mean, coefficient=coefficient,
             variance=variance, train_pairs=n, history=history).items()}


def _measurement_weights(bank):
    yield 'all244', bank['weights'].double()
    for name, (a, b) in zip(GROUP_NAMES, GROUPS):
        weights = torch.zeros_like(bank['weights'], dtype=torch.float64)
        weights[a:b] = bank['weights'][a:b]
        yield name, weights/weights.sum()
    for name in native.ORDER:
        weights = torch.zeros_like(bank['weights'], dtype=torch.float64)
        weights[bank['columns'].index(name)] = 1
        yield name, weights


def _gaussian_values(prediction, variance, actual, bank):
    error = (prediction.double()-actual.double()).square()
    nll = .5*(np.log(2*np.pi)+variance.log()+error/variance)
    result = {}
    for name, weights in _measurement_weights(bank):
        result[name+'/mse'] = (error*weights).sum(-1).numpy()
        result[name+'/gaussian_nll'] = (nll*weights).sum(-1).numpy()
        result[name+'/coverage90'] = ((error.sqrt() <= 1.6448536269514722*variance.sqrt())*
                                     weights).sum(-1).numpy()
    return result


def _probe(features, bank, labels, settings, seed, name, lag, history, device):
    fit = _ridge_fit(features[bank['train_rows']], bank['x'][bank['train_rows']],
                     lag, history, settings, device)
    ids = bank['test_rows']
    x = features[ids, :4].flatten(1) if history == 4 else features[ids, 3]
    prediction = ((x.double()-fit['mean'])/fit['scale'])@fit['coefficient']+fit['target_mean']
    values = _gaussian_values(prediction, fit['variance'], bank['x'][ids, 3+lag], bank)
    result = _aggregate(values, bank, ids, labels, settings, seed,
                        dict(model=name, horizon_ps=lag*.1, history_frames=history))
    return dict(result, fit=fit, values=values)


def _simple_probes(bank, labels, settings, seed, lag):
    train, ids = bank['train_rows'], bank['test_rows']
    before, after = _train_pairs(bank['x'], train, lag)
    mean = torch.from_numpy(after.mean(0, dtype=np.float64))
    variance = torch.from_numpy(after.var(0, dtype=np.float64)).clamp_min(settings['variance_floor'])
    residual = torch.from_numpy(np.square(after.astype(np.float64)-before).mean(0))
    models = [('train_prior', mean.expand(len(ids), -1), variance),
              ('persistence', bank['x'][ids, 3], residual.clamp_min(settings['variance_floor']))]
    return {name: _aggregate(_gaussian_values(prediction, var, bank['x'][ids, 3+lag], bank),
        bank, ids, labels, settings, seed,
        dict(model=name, horizon_ps=lag*.1, history_frames=1)) for name, prediction, var in models}


def _selftica(model, latent, bank, labels, settings, seed, name, lag):
    ids = bank['test_rows']
    device = next(model.parameters()).device
    head = model.representation.representation_heads['predictor'][f'lag{lag}']
    with torch.no_grad():
        a = head(latent[ids, 3].to(device)).double().cpu().numpy()
    b = latent[ids, 3+lag].double().numpy()
    permutation, singletons = _permutation(bank, ids, seed+lag)
    matched, matched_singletons = _permutation(bank, ids, seed+lag, matched=True)
    # Source-local all-pairs U-statistic equals the published minibatch
    # expression for a batch containing this source. It is not the mixed-source
    # training loss; both populations and the no-self-pair denominator are kept.
    rows, sources = [], []
    ancestors = _ancestors(bank, ids)
    for scope, mask in _scopes(bank, ids, labels):
        if not mask.any():
            continue
        keys = np.unique(ancestors[mask])
        packets = defaultdict(list)
        for key in keys:
            m = mask & (ancestors == key)
            left = a[m]
            n = len(left)
            for pairing, right in (('paired', b[m]), ('shuffled', b[permutation][m]),
                                   ('order_matched_shuffled', b[matched][m])):
                diagonal = np.einsum('nd,nd->n', left, right)
                negative = ((np.sum((left.T@left)*(right.T@right))-np.square(diagonal).sum())/
                            (n*(n-1))) if n > 1 else np.nan
                packets[pairing].append(negative-2*diagonal.mean())
        packets['shuffled_minus_paired'] = np.asarray(packets['shuffled'])-packets['paired']
        packets['order_matched_shuffled_minus_paired'] = (np.asarray(
            packets['order_matched_shuffled'])-packets['paired'])
        for pairing, values in packets.items():
            info = dict(model=name, horizon_ps=lag*.1, scope=scope, pairing=pairing,
                metric='source_local_selftica_eq4', shuffle_singletons=singletons,
                matched_shuffle_self_donor_fraction=matched_singletons/len(ids))
            rows.append(dict(info, rows=int(mask.sum()), **native.estimate(
                values, keys, seed, settings['bootstrap'])))
            sources.extend(dict(info, ancestor=str(key), value=float(value))
                           for key, value in zip(keys, values))
    return dict(rows=rows, sources=sources)


@torch.no_grad()
def _field(model, latent, bank, labels, settings, seed, name, lag):
    rep = model.representation
    ids = bank['test_rows']; device = next(model.parameters()).device
    permutation, singletons = _permutation(bank, ids, seed+lag)
    matched, matched_singletons = _permutation(bank, ids, seed+lag, matched=True)
    reverse_matched, reverse_singletons = _permutation(bank, ids, seed+lag,
                                                       matched=True, frame=3+lag)
    generator = torch.Generator(device=device).manual_seed(seed+lag)
    values = defaultdict(list)
    directions = ('future', 'past') if rep.kind == 'fmrc_bidirectional' else ('future',)
    for start in range(0, len(ids), settings['batch_size']):
        stop = start+settings['batch_size']; chosen = ids[start:stop]
        for direction in directions:
            frame, target_frame = (3, 3+lag) if direction == 'future' else (3+lag, 3)
            target = bank['x'][chosen, target_frame].to(device)
            condition = latent[chosen, frame].to(device)
            shuffled = latent[ids[permutation[start:stop]], frame].to(device)
            matched_indices = matched if direction == 'future' else reverse_matched
            order_matched = latent[ids[matched_indices[start:stop]], frame].to(device)
            head = rep.representation_heads[direction][f'lag{lag}']
            accum = defaultdict(list)
            for _ in range(settings['field_draws']):
                noise = rep.flow_prior_mean+rep.flow_prior_scale*torch.randn(
                    target.shape, generator=generator, device=device)
                s = torch.rand((len(target), 1), generator=generator, device=device)
                path = (1-s)*noise+s*target+settings['flow_path_sigma']*torch.randn(
                    target.shape, generator=generator, device=device)
                velocity = target-noise
                for pairing, z in (('paired', condition), ('shuffled', shuffled),
                                   ('order_matched_shuffled', order_matched)):
                    accum[pairing].append((head(path, s, z)-velocity).square().sum(-1))
                accum['zero'].append(velocity.square().sum(-1))
            for key, draws in accum.items():
                values[f'{direction}/{key}/squared_error'].append(torch.stack(draws).mean(0).cpu().numpy())
    values = {key: np.concatenate(value) for key, value in values.items()}
    for direction in directions:
        values[f'{direction}/conditioning_gain'] = (values[f'{direction}/shuffled/squared_error']-
                                                   values[f'{direction}/paired/squared_error'])
        values[f'{direction}/order_matched_conditioning_gain'] = (
            values[f'{direction}/order_matched_shuffled/squared_error']-
            values[f'{direction}/paired/squared_error'])
    return _aggregate(values, bank, ids, labels, settings, seed,
                      dict(model=name, horizon_ps=lag*.1, shuffle_singletons=singletons,
                           matched_shuffle_self_donor_fraction=matched_singletons/len(ids),
                           reverse_matched_shuffle_self_donor_fraction=reverse_singletons/len(ids)))


@torch.no_grad()
def _flow_sample(model, condition, lag, settings, seed):
    """Fixed Heun integration of the ORIGINAL forward conditional flow head.

    Constant path smoothing makes its starting marginal N(mu, sd^2+sigma^2),
    and the terminal marginal is observation space convolved with sigma=.001.
    No clipping or physical projection is applied.
    """
    rep = model.representation
    b, p = len(condition), settings['particles']
    condition = condition[:, None].expand(-1, p, -1).flatten(0, 1)
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    state = rep.flow_prior_mean+(rep.flow_prior_scale.square()+settings['flow_path_sigma']**2).sqrt()*torch.randn(
        (b*p, 244), generator=generator, device=condition.device)
    head = rep.representation_heads['future'][f'lag{lag}']
    dt = 1/settings['flow_steps']
    for step in range(settings['flow_steps']):
        s = state.new_full((len(state), 1), step*dt)
        first = head(state, s, condition)
        second = head(state+dt*first, s+dt, condition)
        state = state+.5*dt*(first+second)
    return state.reshape(b, p, 244)


def _sample_scores(samples, actual, bank):
    from .assessment import _bounds
    values = {}
    raw = inverse(samples, bank)
    lower, upper = _bounds(bank['columns'])
    violates = (raw < lower.to(raw)-1e-5) | (raw > upper.to(raw)+1e-5)
    values['normalized_nonfinite_sample_fraction'] = (~torch.isfinite(samples).all(-1)).float().mean(1)
    values['raw_nonfinite_sample_fraction'] = (~torch.isfinite(raw).all(-1)).float().mean(1)
    values['constraint_violation_sample_fraction'] = violates.any(-1).float().mean(1)
    values['raw_invalid_or_constraint_sample_fraction'] = (
        ~torch.isfinite(raw).all(-1) | violates.any(-1)).float().mean(1)
    # One invalid sample makes this origin's score undefined; no survivor-only
    # score is exported. Aggregate then preserves undefined source estimates.
    valid = torch.isfinite(samples).all(-1).all(-1)
    for group, weight in _measurement_weights(bank):
        weight = weight.to(samples)
        metrics = dict(energy=native.energy(samples[:, :, None], actual[:, None], weight),
            mean_mse=((samples.mean(1)-actual).square()*weight).sum(-1),
            ensemble_variance=(samples.var(1, unbiased=False)*weight).sum(-1),
            coverage90=(((actual >= samples.quantile(.05, dim=1)) &
                         (actual <= samples.quantile(.95, dim=1))).float()*weight).sum(-1))
        values.update({group+'/'+key: torch.where(valid, value, torch.nan)
                       for key, value in metrics.items()})
    return {key: value.cpu().numpy() for key, value in values.items()}


def _samples(model, latent, bank, labels, settings, seed, name, lag, folder, deadline):
    ids = native.selected_rows(bank, settings['per_source_regime'], seed)
    # Draw condition donors from the complete heldout population, avoiding
    # artificial singleton strata caused by the small generative audit panel.
    all_ids = bank['test_rows']
    locations = {int(i): j for j, i in enumerate(all_ids)}
    positions = np.array([locations[int(i)] for i in ids])
    permutation, _ = _permutation(bank, all_ids, seed+lag)
    matched, _ = _permutation(bank, all_ids, seed+lag, matched=True)
    donors, matched_donors = all_ids[permutation[positions]], all_ids[matched[positions]]
    singletons = int((donors == ids).sum())
    matched_singletons = int((matched_donors == ids).sum())
    device = next(model.parameters()).device
    train_future = bank['x'][bank['train_rows'], 3+lag:].flatten(0, 1)
    chunks = []
    for start in range(0, len(ids), settings['batch_size']):
        path = folder/f'samples-lag{lag}-batch{start}.pt'
        def generate():
            selected = ids[start:start+settings['batch_size']]
            actual = bank['x'][selected, 3+lag].to(device)
            predictions = {}
            for pairing, chosen in (('paired', selected),
                    ('shuffled', donors[start:start+len(selected)]),
                    ('order_matched_shuffled', matched_donors[start:start+len(selected)])):
                predictions[pairing] = _flow_sample(model, latent[chosen, 3].to(device),
                    lag, settings, seed+10000*lag+start)
            generator = torch.Generator().manual_seed(seed+10000*lag+start)
            choice = torch.randint(len(train_future), (len(selected), settings['particles']), generator=generator)
            predictions['train_empirical'] = train_future[choice].to(device)
            predictions['persistence'] = bank['x'][selected, 3:4].to(device)
            return dict(indices=selected,
                scores={method: _sample_scores(values, actual, bank) for method, values in predictions.items()},
                samples={method: values.cpu() for method, values in predictions.items()})
        packet = _packet(path, generate, deadline)
        if packet is None:
            return None
        chunks.append(packet)
    rows, sources = [], []
    for method in ('paired', 'shuffled', 'order_matched_shuffled', 'train_empirical', 'persistence'):
        values = {key: np.concatenate([chunk['scores'][method][key] for chunk in chunks])
                  for key in chunks[0]['scores'][method]}
        result = _aggregate(values, bank, ids, labels, settings, seed,
            dict(model=name, horizon_ps=lag*.1, method=method, shuffle_singletons=singletons,
                 matched_shuffle_self_donor_fraction=matched_singletons/len(ids)))
        rows.extend(result['rows']); sources.extend(result['sources'])
    return dict(rows=rows, sources=sources, indices=ids)


def _publish(packets, out, family, prefix, cohort):
    for category, parts in packets.items():
        for key, suffix in (('rows', ''), ('sources', '_sources')):
            rows = [dict(row, **cohort) for part in parts for row in part[key]]
            if rows:
                write_metric_rows(rows, out, family=family, name=f'{prefix}_{category}{suffix}')


def _controls(c, bank, labels, settings, seed, identity, device, deadline, family, prefix,
              cohort, support='central'):
    """Single shared, resumable raw-measurement control fit across all arms."""
    suffix = '' if support == 'central' else '-r20'
    out = root(c)/'analyses'/f'{prefix}-controls-{cohort["cohort_key"]}{suffix}'
    folder = out/'technical'/prefix
    folder.mkdir(parents=True, exist_ok=True)
    marker = out/'technical'/f'{prefix}-complete.json'
    if marker.exists():
        if read(marker)['identity'] != identity:
            raise ValueError('Changed shared kinetics control identity')
        return True
    with (folder/'fit.lock').open('w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        inputs = folder/'inputs.json'
        if inputs.exists() and read(inputs)['identity'] != identity:
            raise ValueError('Changed in-progress kinetics controls')
        write_json(inputs, dict(identity=identity, settings=settings, **cohort,
            train_rows=np.asarray(bank['train_rows']).tolist(),
            test_rows=np.asarray(bank['test_rows']).tolist()))
        packets = defaultdict(list)
        for lag in settings['operator_lags'] if support == 'central' else []:
            packet = _packet(folder/f'operator-{lag}.pt', lambda: _operator(
                bank['x'], bank, labels, settings, seed, 'current244', lag), deadline)
            if packet is None:
                return False
            packets['operator'].append(packet)
        for lag in settings['lags']:
            controls = ((1, 'current244'), (4, 'history976')) if support == 'central' else (
                (1, 'current331_r20'), (4, 'history1324_r20'))
            features = bank['x'] if support == 'central' else bank['input']
            for history, name in controls:
                packet = _packet(folder/f'probe-{lag}-{history}.pt', lambda: _probe(
                    features, bank, labels, settings, seed, name, lag, history, device), deadline)
                if packet is None:
                    return False
                packets['probe'].append(packet)
            if support == 'central':
                simple = _packet(folder/f'simple-{lag}.pt', lambda: _simple_probes(
                    bank, labels, settings, seed, lag), deadline)
                if simple is None:
                    return False
                packets['probe'].extend(simple.values())
        _publish(packets, out, family, prefix, cohort)
        write_json(marker, dict(identity=identity, state='complete', **cohort))
    return True


def evaluate(c, arm, deadline, *, load_provider, fitted_provider, family,
             output_suffix='kinetics'):
    """Evaluate a reference checkpoint into NEW c['output'], without training.

    Providers accept (c, arm). The data plan must carry the original
    ``files.labels.path`` and ``data_identity``. Each independent calculation
    has an atomic resume packet; native FMRC sampling also checkpoints batches.
    The queue should retry False, not treat it as scientific failure. Numerical
    failures remain explicit and do not silently turn into finite-only scores.
    """
    out = root(c)/'analyses'/arm['name']; folder = out/'technical'/output_suffix
    folder.mkdir(parents=True, exist_ok=True)
    marker = out/'technical'/f'{output_suffix}-complete.json'
    settings = dict(DEFAULTS, **c.get('kinetics', {}))
    plan, bank = load_provider(c, arm)
    model, saved = fitted_provider(c, arm)
    seed = c['seed']
    split = dict(train_rows=np.asarray(bank['train_rows']).tolist(),
                 test_rows=np.asarray(bank['test_rows']).tolist())
    cohort = dict(cohort_key=digest([plan['data_identity'], split['train_rows']])[:12],
        training_row_count=len(bank['train_rows']),
        training_ancestor_count=len(set(_ancestors(bank, bank['train_rows']))),
        test_row_count=len(bank['test_rows']))
    identity = digest([plan['data_identity'], saved['identity'], settings,
                       file_hash(__file__), seed, split])
    if marker.exists():
        if read(marker)['identity'] != identity:
            raise ValueError('Changed completed kinetics evaluation identity')
        return True
    receipt = folder/'inputs.json'
    if receipt.exists() and read(receipt)['identity'] != identity:
        raise ValueError('Changed checkpoint or settings during kinetics evaluation')
    write_json(receipt, dict(identity=identity, settings=settings,
        checkpoint_identity=saved['identity'], data_identity=plan['data_identity'],
        **split, **cohort,
        test_origin=3, cadence_ps=.1, train_origins='all valid origins >=3',
        temperature_or_absolute_time_inputs=False, test_selection=False,
        covariance_scope='Nonreversible finite-lag observational pair distribution',
        sampler='Original FMRC forward head, fixed64-step Heun unless explicitly configured',
        liquid_annotation='PTM central not FCC/HCP/BCC and current neighborhood fraction <=.05; diagnostic only',
        order_shuffle_columns=list(native.ORDER[:2]),
        order_shuffle_train_medians_normalized=_order_thresholds(bank)[1].tolist()))
    annotation = plan['files']['labels']
    if file_hash(annotation['path']) != annotation['sha256']:
        raise ValueError('Changed native PTM annotations in kinetics diagnostics')
    labels = torch.load(annotation['path'], map_location='cpu', weights_only=False)
    device = next(model.parameters()).device
    latent = _encode(model, bank['input'], settings)
    packets = defaultdict(list)
    for lag in settings['operator_lags']:
        packet = _packet(folder/f'operator-{lag}.pt', lambda: _operator(
            latent, bank, labels, settings, seed, arm['name'], lag), deadline)
        if packet is None:
            return False
        packets['operator'].append(packet)
    for lag in settings['lags']:
        probes = []
        for history in (1, 4):
            packet = _packet(folder/f'probe-{lag}-{history}.pt', lambda: _probe(
                latent, bank, labels, settings, seed, arm['name'], lag, history, device), deadline)
            if packet is None:
                return False
            packets['probe'].append(packet); probes.append(packet)
        gain = {key: probes[0]['values'][key]-probes[1]['values'][key]
                for key in probes[0]['values'] if not key.endswith('coverage90')}
        packets['history_gain'].append(_aggregate(gain, bank, bank['test_rows'], labels,
            settings, seed, dict(model=arm['name'], horizon_ps=lag*.1,
                                comparison='current_minus_history')))
    kind = arm['representation']
    for lag in settings['native_lags']:
        if kind == 'selftica':
            packet = _packet(folder/f'selftica-{lag}.pt', lambda: _selftica(
                model, latent, bank, labels, settings, seed, arm['name'], lag), deadline)
            if packet is None:
                return False
            packets['selftica'].append(packet)
        elif kind.startswith('fmrc'):
            packet = _packet(folder/f'field-{lag}.pt', lambda: _field(
                model, latent, bank, labels, settings, seed, arm['name'], lag), deadline)
            if packet is None:
                return False
            packets['field'].append(packet)
            path = folder/f'samples-{lag}.pt'
            packet = torch.load(path, map_location='cpu', weights_only=False) if path.exists() else _samples(
                model, latent, bank, labels, settings, seed, arm['name'], lag, folder, deadline)
            if packet is None:
                return False
            if not path.exists():
                atomic_save(path, packet)
            packets['samples'].append(packet)
    _publish(packets, out, family, output_suffix, cohort)
    controls_identity = digest([plan['data_identity'], settings, seed, file_hash(__file__), split])
    if not _controls(c, bank, labels, settings, seed, controls_identity,
                     device, deadline, family, output_suffix, cohort):
        return False
    if arm['scale'] == 'r20' and not _controls(c, bank, labels, settings, seed,
            digest([controls_identity, 'r20']), device, deadline, family, output_suffix,
            cohort, support='r20'):
        return False
    write_json(marker, dict(identity=identity, state='complete', settings=settings,
        **cohort,
        controls=[f'{output_suffix}-controls-{cohort["cohort_key"]}'+suffix
                  for suffix in (('', '-r20') if arm['scale'] == 'r20' else ('',))],
        checkpoint_identity=saved['identity'], native_head=kind if kind in
        ('selftica', 'fmrc_forward', 'fmrc_bidirectional') else 'not_applicable'))
    return True
