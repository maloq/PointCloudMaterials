"""Frozen liquid-order diagnostics and native-cadence local trajectory tests.

Phase labels annotate evaluation and a frozen likelihood readout only. They do
not train the structural encoder or define a nucleus in independently generated
local neighborhoods. The enriched sequence population is not natural incidence.
"""
from collections import defaultdict
import hashlib
import time

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits
import torch

from src.experiment_runner.artifacts import file_hash, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.research.continuous_motion.data import atomic_save
from .data import root, cache, load, inverse, read
from .learning import fitted


FAMILY = 'ni_kinetic_latent'
ORDER = ('bond_order/l6_qbar', 'bond_order/l6_coherence_mean',
         'bond_order/l4_qbar', 'bond_order/l6_wbar')
REGIMES = ('original', 'liquid_liquid', 'growing_front')


def selected_rows(bank, count, seed):
    """Outcome-independent fixed panel, at most count per source/regime."""
    groups = defaultdict(list)
    for i in bank['test_rows']:
        row = bank['rows'][i]
        groups[row['ancestor'], row['regime']].append(int(i))
    chosen = []
    for key in sorted(groups):
        ranked = sorted(groups[key], key=lambda i: hashlib.sha256(
            f"{seed}/{bank['rows'][i]['id']}".encode()).hexdigest())
        chosen.extend(ranked[:count])
    return np.array(chosen, dtype=np.int64)


def source_values(values, ancestors):
    values = np.asarray(values)
    unique = np.unique(ancestors)
    return unique, np.array([values[ancestors == key].mean(axis=0) for key in unique])


def estimate(values, ancestors, seed, bootstrap, allow_undefined=False):
    keys, means = source_values(values, np.asarray(ancestors))
    valid = np.isfinite(means)
    if not valid.all() and not allow_undefined:
        return dict(value=None, lo=None, hi=None, ancestors=len(keys))
    keys, means = keys[valid], means[valid]
    if not len(means):
        return dict(value=None, lo=None, hi=None, ancestors=0)
    draw = np.random.default_rng(seed).integers(len(means), size=(bootstrap, len(means)))
    lo, hi = np.quantile(means[draw].mean(1), [.025, .975])
    return dict(value=float(means.mean()), lo=float(lo), hi=float(hi), ancestors=len(keys))


def probability(z, probe):
    return expit(((np.asarray(z)-probe['mean'])/probe['scale']) @ probe['coefficient']
                 + probe['intercept'])


def first_entry(states, width=3):
    """Completion of the first run of width future frames; censor at T+1."""
    states = np.asarray(states, dtype=bool)
    found = np.zeros(states.shape[:-1], bool)
    first = np.full(states.shape[:-1], states.shape[-1]+1, dtype=np.int16)
    for index in range(width-1, states.shape[-1]):
        enters = states[..., index-width+1:index+1].all(-1) & ~found
        first[enters] = index+1
        found |= enters
    return first


def longest_run(states):
    states = np.asarray(states, dtype=bool)
    current = np.zeros(states.shape[:-1], np.int16)
    maximum = current.copy()
    for index in range(states.shape[-1]):
        current = np.where(states[..., index], current+1, 0)
        maximum = np.maximum(current, maximum)
    return maximum*.1


def energy(samples, target, weights):
    """Joint-path energy score with unbiased ensemble pair term, also for P=1."""
    distance = (((samples-target[:, None]).square()*weights).sum(-1).mean(-1)).sqrt()
    score = distance.mean(1)
    if samples.shape[1] > 1:
        flat = (samples*weights.sqrt()).flatten(2)/(samples.shape[2]**.5)
        pairs = torch.cdist(flat, flat, compute_mode='donot_use_mm_for_euclid_dist')
        i, j = torch.tril_indices(samples.shape[1], samples.shape[1], offset=-1,
                                 device=samples.device)
        score = score-.5*pairs[:, i, j].mean(1)
    return score


def scores(samples, target, initial, weights, increment_scale):
    true_delta = torch.diff(torch.cat((initial[:, None], target), 1), dim=1)/increment_scale
    anchor = initial[:, None, None].expand(-1, samples.shape[1], -1, -1)
    delta = torch.diff(torch.cat((anchor, samples), 2), dim=2)/increment_scale
    true_corr, predicted_corr = [], []
    for lag in (1, 5):
        true_corr.append((true_delta[:, lag:]*true_delta[:, :-lag]).mean(1))
        predicted_corr.append((delta[:, :, lag:]*delta[:, :, :-lag]).mean(2))
    return dict(path_energy=energy(samples, target, weights),
        increment_energy=energy(delta, true_delta, weights),
        increment_correlation_energy=energy(torch.stack(predicted_corr, 2),
                                           torch.stack(true_corr, 1), weights),
        endpoint_mse=((samples.mean(1)[:, -1]-target[:, -1]).square()*weights).sum(-1),
        temporal_power=(samples.var(2, unbiased=False).mean(1)*weights).sum(-1),
        truth_temporal_power=(target.var(1, unbiased=False)*weights).sum(-1),
        ensemble_power=(samples.var(1, unbiased=False)*weights).sum(-1).mean(-1),
        coverage90=(((target >= samples.quantile(.05, dim=1)) &
                     (target <= samples.quantile(.95, dim=1))).float()*weights).sum(-1).mean(-1))


def _encode(model, x, batch=2048):
    device = next(model.parameters()).device
    values, decoded = [], []
    with torch.no_grad():
        for part in x.split(batch):
            z = model.encode(part.to(device))
            values.append(z.cpu()); decoded.append(model.decode(z).cpu())
    return torch.cat(values), torch.cat(decoded)


def _prepare_diagnostics(c, arm, model, bank, labels, folder, *, input_key='x'):
    path = folder/'frozen-diagnostics.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    rng = np.random.default_rng(c['seed'])
    measured_phase = c['evaluation'].get('phase_probe_space') == 'measurements'
    train = np.asarray(bank['reference_train_rows'] if measured_phase else bank['train_rows'])
    # Equal source contribution to the fixed-size readout sample. No phase
    # oversampling and no test transformation or coefficient fitting.
    by_source = defaultdict(list)
    for row in train:
        by_source[bank['rows'][row]['ancestor']].append(int(row))
    flat_indices = []
    for source in sorted(by_source):
        available = (np.array(by_source[source])[:, None]*64+np.arange(64)).reshape(-1)
        flat_indices.extend(rng.choice(available, min(len(available), 60000//len(by_source)), replace=False))
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    encoder_input = bank[input_key]
    train_x = encoder_input.reshape(-1, encoder_input.shape[-1])[flat_indices]
    train_z, _ = _encode(model, train_x)
    z = train_z.numpy(); mean = z.mean(0); scale = z.std(0).clip(1e-6)
    phase_features = bank['x'].reshape(-1, 244)[flat_indices].numpy() if measured_phase else z
    phase_mean, phase_scale = phase_features.mean(0), phase_features.std(0).clip(1e-6)
    y = np.asarray(labels['crystal']).reshape(-1)[flat_indices]
    with threadpool_limits(limits=2):
        classifier = LogisticRegression(C=1., max_iter=1000, solver='lbfgs').fit((phase_features-phase_mean)/phase_scale, y)
    probe = dict(mean=phase_mean, scale=phase_scale, coefficient=classifier.coef_[0],
                 intercept=float(classifier.intercept_[0]), threshold=.5,
                 regularization_C=1., sample_indices=flat_indices,
                 iterations=int(classifier.n_iter_[0]))
    # Train-only diagnostic ordering thresholds use all available present-liquid
    # training states, not the observed future of a held-out sequence.
    crystal = np.asarray(labels['crystal'])[train]
    fraction = np.asarray(labels['fraction'])[train]
    liquid = (~crystal) & (fraction <= .05)
    names = bank['columns']; order_columns = [names.index(name) for name in ORDER]
    raw_train = inverse(bank['x'][train], bank).numpy()
    threshold = np.quantile(raw_train[..., order_columns[:2]][liquid], .8, axis=0)
    covariance = np.cov(z, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    positive = eigenvalues.clip(0); proportions = positive/max(positive.sum(), 1e-30)
    rank = float(np.exp(-(proportions[proportions > 0]*np.log(proportions[proportions > 0])).sum()))
    # Increment units are fitted on train native measurements only.
    delta = torch.diff(bank['x'][train], dim=1)
    increment_scale = delta.square().mean((0, 1)).sqrt().clamp_min(.05)
    result = dict(probe=probe, order_columns=order_columns, high_order_threshold=threshold,
        train_projection_mean=mean, train_projection=eigenvectors[:, -2:],
        latent_variance=eigenvalues, effective_rank=rank,
        increment_scale=increment_scale, train_probe_rows=len(flat_indices),
        train_liquid_states=int(liquid.sum()), phase_probe_space='measurements' if measured_phase else 'latent',
        ordering_definition='joint train-liquid 80th percentile of qbar6 and coherence; not a phase')
    atomic_save(path, result)
    return result


def _retention(c, arm, model, bank, labels, diagnostic, out, *, input_key='x', family=FAMILY):
    folder = out/'technical'; path = folder/'retention.pt'
    if path.exists():
        return torch.load(path, map_location='cpu', weights_only=False)
    ids = np.asarray(bank['test_rows']); x = bank['x'][ids, 3]
    z, decoded = _encode(model, bank[input_key][ids, 3])
    raw = inverse(x, bank).numpy(); predicted = inverse(decoded, bank).numpy()
    if not torch.isfinite(decoded).all() or not torch.isfinite(z).all():
        raise FloatingPointError(f"Nonfinite transformed present-state decoding: {arm['name']}")
    invalid_raw = ~np.isfinite(predicted).all(1)
    if invalid_raw.any():
        write_json(folder/'retention-raw-inverse-failure.json', dict(
            model=arm['name'], affected_indices=ids[invalid_raw].tolist(),
            reason='raw inverse overflow; normalized retention and phase scores retained'))
    ancestors = np.array([bank['rows'][i]['ancestor'] for i in ids])
    regimes = np.array([bank['rows'][i]['regime'] for i in ids])
    crystal = np.asarray(labels['crystal'])[ids, 3]
    fraction = np.asarray(labels['fraction'])[ids, 3]
    liquid = ~crystal & (fraction <= .05)
    phase_input = decoded if c['evaluation'].get('phase_probe_space') == 'measurements' else z
    phase = probability(phase_input.numpy(), diagnostic['probe'])
    records = []; source_rows = []
    seed, boots = c['seed'], c['evaluation']['bootstrap']
    def add(values, mask, metric, feature='', subset='all', regime='all'):
        if not mask.any():
            return
        aa = ancestors[mask]; vv = np.asarray(values)[mask]
        summary = estimate(vv, aa, seed, boots)
        records.append(dict(model=arm['name'], method='decoder', regime=regime,
            subset=subset, metric=metric, feature=feature, rows=int(mask.sum()), **summary))
        keys, means = source_values(vv, aa)
        source_rows.extend(dict(model=arm['name'], method='decoder', regime=regime,
            subset=subset, metric=metric, feature=feature, ancestor=str(a), value=float(v) if np.isfinite(v) else None)
            for a, v in zip(keys, means))
    standardized_error = (decoded-x).square().numpy()
    groups = [('bond_order', 0, 40), ('tda', 40, 153), ('density', 153, 227), ('velocity', 227, 244)]
    for regime in ('all', *REGIMES):
        group_mask = np.ones(len(ids), bool) if regime == 'all' else regimes == regime
        add(invalid_raw.astype(float), group_mask, 'invalid_raw_row_fraction', regime=regime)
        for subset, risk in [('all', np.ones(len(ids), bool)), ('present_liquid', liquid)]:
            mask = group_mask & risk
            for group, lo, hi in groups:
                active = np.asarray(bank['active'])[lo:hi]
                add(standardized_error[:, lo:hi][:, active].mean(1), mask,
                    'normalized_mse', group, subset, regime)
            for name, column in zip(ORDER, diagnostic['order_columns']):
                add((raw[:, column]-predicted[:, column])**2, mask, 'raw_mse', name, subset, regime)
                # Within-source correlations and R2 cannot be replaced by a
                # pooled correlation driven by differences between sources.
                keys = np.unique(ancestors[mask]); r2 = []; rank = []
                for ancestor in keys:
                    chosen = mask & (ancestors == ancestor)
                    target, value = raw[chosen, column], predicted[chosen, column]
                    variance = np.var(target)
                    finite = np.isfinite(value).all()
                    r2.append(1-np.mean((target-value)**2)/variance if variance > 1e-20 and finite else np.nan)
                    rank.append(float(spearmanr(target, value).statistic)
                                if finite and len(target) > 1 and np.ptp(target) > 0 and np.ptp(value) > 0 else np.nan)
                for metric, values in [('within_source_r2', r2), ('within_source_spearman', rank)]:
                    if not len(keys):
                        continue
                    records.append(dict(model=arm['name'], method='decoder', regime=regime,
                        subset=subset, metric=metric, feature=name, rows=int(mask.sum()),
                        **estimate(np.array(values), keys, seed, boots,
                                   allow_undefined=not invalid_raw[mask].any())))
                    source_rows.extend(dict(model=arm['name'], method='decoder', regime=regime,
                        subset=subset, metric=metric, feature=name, ancestor=str(a),
                        value=float(v) if np.isfinite(v) else None) for a, v in zip(keys, values))
        add((phase-crystal)**2, group_mask, 'phase_probe_brier', regime=regime)
        add(-(crystal*np.log(phase.clip(1e-7, 1-1e-7))+
              ~crystal*np.log1p(-phase.clip(1e-7, 1-1e-7))), group_mask, 'phase_probe_nll', regime=regime)
        add(((phase >= .5) == crystal).astype(float), group_mask, 'phase_probe_accuracy', regime=regime)
    write_metric_rows(records, out, family=family, name='retention')
    write_metric_rows(source_rows, out, family=family, name='retention_sources')
    result = dict(ids=ids, latent=z.numpy(), decoded=decoded, raw_order=raw[:, diagnostic['order_columns']],
                  crystal=crystal, fraction=fraction, phase_probability=phase, ancestors=ancestors,
                  regimes=regimes, records=records)
    atomic_save(path, result)
    return result


def _plot(out, diagnostic, retention, examples, forecast):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = out/'plots'; plots.mkdir(parents=True, exist_ok=True)
    projected = (retention['latent']-diagnostic['train_projection_mean']) @ diagnostic['train_projection']
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    rng = np.random.default_rng(20261015)
    show = rng.choice(len(projected), min(8000, len(projected)), replace=False)
    for axis, color, title in zip(axes, [retention['crystal'], retention['raw_order'][:, 0]],
                                 ['Observed crystalline center', 'Observed averaged bond order q̄₆']):
        scatter = axis.scatter(projected[show, 0], projected[show, 1], c=color[show], s=3, alpha=.5,
                               cmap='viridis', rasterized=True)
        figure.colorbar(scatter, ax=axis); axis.set_title(title)
        axis.set_xlabel('Latent PC 1, fitted on train'); axis.set_ylabel('Latent PC 2')
    figure.savefig(plots/'latent_order.png', dpi=180); plt.close(figure)
    figure, axes = plt.subplots(len(examples), 1, figsize=(10, max(3, 2.5*len(examples))),
                                squeeze=False, constrained_layout=True)
    for axis, example in zip(axes[:, 0], examples):
        truth = example['truth'][:, 0]; samples = example['samples'][:, :, 0]
        times = np.arange(1, 61)*.1
        axis.plot(np.arange(64)*.1-.3, truth, color='black', linewidth=1.4, label='Saved MD')
        if example['invalid_raw_paths']:
            axis.text(.02, .9, 'Raw decoding overflow; forecast omitted', transform=axis.transAxes)
        else:
            for sample in samples[:8]: axis.plot(times, sample, color='#5d9fbd', alpha=.2, linewidth=.6)
            axis.fill_between(times, *np.quantile(samples, [.05, .95], axis=0), color='#5d9fbd', alpha=.2)
            axis.plot(times, samples.mean(0), color='#267596', label='Neural ensemble mean')
        axis.axvline(0, color='gray', linestyle=':'); axis.set_ylabel('q̄₆')
        axis.set_title(f"{example['regime']}: fixed atom {example['center']}")
    axes[0, 0].legend(); axes[-1, 0].set_xlabel('Time from observed snapshot (ps)')
    figure.savefig(plots/'order_trajectories.png', dpi=180); plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for regime in REGIMES:
        chosen = (forecast['regimes'] == regime) & forecast['risk']
        if not chosen.any(): continue
        for column, (label, entry) in enumerate([('Saved MD', forecast['truth_entry']),
                                                ('Generated', forecast['sample_entry'])]):
            probability_curve = np.array([(entry[chosen] <= k).reshape(chosen.sum(), -1).mean(1)
                                          for k in range(1, 61)]).T
            _, grouped = source_values(probability_curve, forecast['ancestors'][chosen])
            axes[column].plot(np.arange(1, 61)*.1, grouped.mean(0), label=regime)
            axes[column].set_title(label+' local persistent crystal entry')
    for axis in axes:
        axis.set_ylim(0, 1); axis.set_xlabel('Time from snapshot (ps)')
        axis.set_ylabel('Conditional cohort first-entry fraction'); axis.legend()
    figure.savefig(plots/'local_crystallization.png', dpi=180); plt.close(figure)


def evaluate(c, arm, deadline, *, load_provider=None, fitted_provider=None,
             family=FAMILY, input_key='x', labels_path=None,
             deterministic_method='conditional_mean'):
    out = root(c)/'analyses'/arm['name']; folder = out/'technical'
    if (folder/'evaluation-complete.json').exists():
        return True
    plan, bank = load(c, device='cpu') if load_provider is None else load_provider(c, arm)
    label_file = cache(c)/'labels.pt' if labels_path is None else labels_path
    labels = torch.load(label_file, map_location='cpu', weights_only=False)
    model, saved = (fitted if fitted_provider is None else fitted_provider)(c, arm)
    inputs = dict(checkpoint_sha256=file_hash(folder/'final.pt'),
                  data_identity=plan['data_identity'], labels_sha256=file_hash(label_file))
    receipt = folder/'evaluation-inputs.json'
    if receipt.exists() and read(receipt) != inputs:
        raise ValueError('Changed inputs for cached kinetic evaluation: '+arm['name'])
    write_json(receipt, inputs)
    diagnostic = _prepare_diagnostics(c, arm, model, bank, labels, folder, input_key=input_key)
    write_metric_rows([dict(model=arm['name'], metric='effective_rank', component=None,
        value=diagnostic['effective_rank'], train_rows=diagnostic['train_probe_rows'])] +
        [dict(model=arm['name'], metric='latent_variance', component=i, value=float(value),
              train_rows=diagnostic['train_probe_rows'])
         for i, value in enumerate(diagnostic['latent_variance'])], out, family=family, name='latent_spectrum')
    retention = _retention(c, arm, model, bank, labels, diagnostic, out, input_key=input_key, family=family)
    ids = selected_rows(bank, c['evaluation']['per_source_regime'], c['seed'])
    rows = [bank['rows'][i] for i in ids]
    ancestors = np.array([row['ancestor'] for row in rows]); regimes = np.array([row['regime'] for row in rows])
    write_json(folder/'evaluation-panel.json', dict(indices=ids.tolist(), identities=[row['id'] for row in rows],
        seed=c['seed'], selection=f"hash-ranked, at most {c['evaluation']['per_source_regime']} per ancestor/regime, independent of outcomes"))
    device = next(model.parameters()).device
    weights = bank['weights'].to(device); increment_scale = diagnostic['increment_scale'].to(device)
    batch = c['evaluation']['batch_size']; particles = c['evaluation']['particles']
    values = defaultdict(list); examples = []; sample_entries = []; truth_entries = []
    start = 0; partial = folder/'forecast-partial.pt'
    if partial.exists():
        previous = torch.load(partial, map_location='cpu', weights_only=False)
        if previous['identity'] != saved['identity'] or not np.array_equal(previous['ids'], ids):
            raise ValueError('Changed forecast checkpoint or evaluation panel')
        start = previous['next']; values.update(previous['values']); examples = previous['examples']
        sample_entries = previous['sample_entries']; truth_entries = previous['truth_entries']
    example_ids = {next(int(i) for i in ids if bank['rows'][i]['regime'] == regime) for regime in REGIMES}
    order_columns = diagnostic['order_columns']; threshold = diagnostic['high_order_threshold']
    def save(next_index):
        atomic_save(partial, dict(identity=saved['identity'], ids=ids, next=next_index, values=dict(values),
            examples=examples, sample_entries=sample_entries, truth_entries=truth_entries))
    with torch.no_grad():
        for offset in range(start, len(ids), batch):
            if time.time() > deadline-180:
                save(offset); return False
            chunk = ids[offset:offset+batch]; observed = bank['x'][chunk].to(device)
            encoder_input = bank[input_key][chunk].to(device)
            latent, prediction = model.rollout(encoder_input[:, :4], 60, particles, c['seed']+offset)
            deterministic_z, deterministic = model.rollout(encoder_input[:, :4], 60, 1,
                                                           c['seed']+offset, deterministic=True)
            encoded_md = model.encode(encoder_input[:, 4:])
            reconstructed_md = model.decode(encoded_md)[:, None]
            if not all(torch.isfinite(value).all() for value in
                       (latent, prediction, deterministic_z, deterministic)):
                failure = dict(state='generation_failed', model=arm['name'], panel_offset=offset,
                    ids=chunk.tolist(), retained_current_rows=len(retention['ids']),
                    completed_forecast_rows=offset, reason='nonfinite autonomous latent or transformed measurement')
                write_json(folder/'generation-failure.json', failure)
                write_json(folder/'evaluation-complete.json', dict(failure, **inputs))
                update_recorded_summary(folder/'wandb.json', dict(generation_failed=True,
                    **{'latent/effective_rank': diagnostic['effective_rank']}),
                    evaluation='kinetic-latent-generation-failure', expected=dict(identity=saved['identity']))
                return True
            persistence = observed[:, None, 3:4].expand(-1, 1, 60, -1)
            for method, samples in [('model', prediction), (deterministic_method, deterministic),
                                     ('persistence', persistence), ('encoded_MD', reconstructed_md)]:
                for steps in (10, 30, 60):
                    result = scores(samples[:, :, :steps], observed[:, 4:4+steps], observed[:, 3],
                                    weights, increment_scale)
                    for metric, value in result.items():
                        values[(method, steps, metric, '')].append(value.cpu().numpy())
            actual_raw = inverse(bank['x'][chunk], bank).numpy()
            generated_raw = inverse(prediction.cpu(), bank).numpy()
            reconstructed_raw = inverse(reconstructed_md.cpu(), bank).numpy()
            true_crystal = np.asarray(labels['crystal'])[chunk, 4:]
            measured_phase = c['evaluation'].get('phase_probe_space') == 'measurements'
            prob = probability((prediction if measured_phase else latent).cpu().numpy(), diagnostic['probe'])
            predicted_crystal = prob >= .5
            true_entry = first_entry(true_crystal); sample_entry = first_entry(predicted_crystal)
            truth_entries.append(true_entry); sample_entries.append(sample_entry)
            true_high = (actual_raw[:, 4:, order_columns[0]] > threshold[0]) & (
                         actual_raw[:, 4:, order_columns[1]] > threshold[1]) & ~true_crystal
            initial_z = model.encode(encoder_input[:, 3]).cpu().numpy()
            persistence_z = np.broadcast_to(initial_z[:, None, None], (len(chunk), 1, 60, initial_z.shape[-1]))
            phase_inputs = ([('model', prediction.cpu().numpy()),
                             (deterministic_method, deterministic.cpu().numpy()),
                             ('persistence', persistence.cpu().numpy()),
                             ('encoded_MD', reconstructed_md.cpu().numpy())] if measured_phase else
                            [('model', latent.cpu().numpy()),
                             (deterministic_method, deterministic_z.cpu().numpy()),
                             ('persistence', persistence_z), ('encoded_MD', encoded_md.cpu().numpy()[:, None])])
            for method, future_z in phase_inputs:
                phase_probability = probability(future_z, diagnostic['probe'])
                entry = first_entry(phase_probability >= .5)
                for steps in (10, 30, 60):
                    event = true_entry <= steps; predicted_event = (entry <= steps).mean(1)
                    for metric, value in dict(local_entry_brier=(predicted_event-event)**2,
                        local_entry_observed=event.astype(float), local_entry_predicted=predicted_event,
                        phase_probability_brier=((phase_probability[:, :, :steps].mean(1)-true_crystal[:, :steps])**2).mean(1)).items():
                        values[(method, steps, metric, '')].append(value)
            # This oracle control encodes each actually observed future. It
            # separates representation/decoder losses from rollout failure and
            # must never be interpreted as a forecast.
            reconstructed_phase_input = reconstructed_md[:, 0] if measured_phase else encoded_md
            reconstructed_phase = probability(reconstructed_phase_input.cpu().numpy(), diagnostic['probe'])[:, None] >= .5
            for method, measured, phase in [('model', generated_raw, predicted_crystal),
                                            ('encoded_MD', reconstructed_raw, reconstructed_phase)]:
                valid = np.isfinite(measured).all(axis=-1)
                invalid_path = ~valid.all((1, 2))
                values[(method, 60, 'invalid_raw_coordinate_fraction', '')].append(
                    (~np.isfinite(measured)).mean((1, 2, 3)))
                values[(method, 60, 'invalid_raw_path_fraction', '')].append((~valid.all(2)).mean(1))
                if invalid_path.any():
                    write_json(folder/f'raw-inverse-failure-{method}-{offset:05d}.json', dict(
                        model=arm['name'], method=method, panel_offset=offset,
                        affected_indices=chunk[invalid_path].tolist(),
                        reason='inverse asinh overflow; raw metrics for affected ensemble paths are undefined; no clipping'))
                high = (measured[..., order_columns[0]] > threshold[0]) & (
                        measured[..., order_columns[1]] > threshold[1]) & ~phase
                for steps in (10, 30, 60):
                    occupancy = high[:, :, :steps].mean((1, 2))
                    actual_occupancy = true_high[:, :steps].mean(1)
                    for metric, value in dict(high_order_occupancy_error=(occupancy-actual_occupancy)**2,
                        high_order_occupancy=occupancy, high_order_observed_occupancy=actual_occupancy,
                        high_order_longest_dwell_error=(longest_run(high[:, :, :steps]).mean(1)-longest_run(true_high[:, :steps]))**2).items():
                        value[~valid[:, :, :steps].all((1, 2))] = np.nan
                        values[(method, steps, metric, '')].append(value)
                for column, name in zip(order_columns, ORDER):
                    truth = actual_raw[:, 4:, column]
                    generated = measured[..., column]
                    # Train-normalized fluctuations retain amplitude as well as
                    # temporal ordering; no per-path normalization hides collapse.
                    mean = float(bank['raw_mean'][column]); scale = float(bank['raw_scale'][column])
                    true_order = (truth-mean)/max(scale, 1e-12)
                    predicted_order = (generated-mean)/max(scale, 1e-12)
                    for lag in (1, 3, 5, 10, 30):
                        true_product = (true_order[:, lag:]*true_order[:, :-lag]).mean(1)
                        with np.errstate(over='ignore', invalid='ignore'):
                            pred_product = (predicted_order[:, :, lag:]*predicted_order[:, :, :-lag]).mean((1, 2))
                        pred_product[invalid_path] = np.nan
                        values[(method, lag, 'order_lag_product_error', name)].append((pred_product-true_product)**2)
                        values[(method, lag, 'order_lag_product', name)].append(pred_product)
                        values[(method, lag, 'order_observed_lag_product', name)].append(true_product)
            for k, i in enumerate(chunk):
                if int(i) in example_ids:
                    examples.append(dict(id=bank['rows'][i]['id'], regime=bank['rows'][i]['regime'],
                        center=bank['rows'][i]['center'], truth=actual_raw[k][:, order_columns],
                        samples=generated_raw[k][..., order_columns],
                        invalid_raw_paths=int((~np.isfinite(generated_raw[k]).all((1, 2))).sum())))
            save(offset+len(chunk))
            write_json(folder/'evaluation-progress.json', dict(rows=offset+len(chunk), total=len(ids), stage='forecast'))
    arrays = {key: np.concatenate(parts) for key, parts in values.items()}
    risk = ~np.asarray(labels['crystal'])[ids, 3]
    liquid_risk = risk & (np.asarray(labels['fraction'])[ids, 3] <= .05)
    records = []; source_records = []
    for (method, steps, metric, feature), value in arrays.items():
        for regime in ('all', *REGIMES):
            select = np.ones(len(ids), bool) if regime == 'all' else regimes == regime
            if metric.startswith('local_entry'):
                subsets = [('initial_noncrystal', risk), ('present_liquid', liquid_risk)]
            else:
                subsets = [('all', np.ones(len(ids), bool)), ('present_liquid', liquid_risk)]
            for subset, mask in subsets:
                chosen = select & mask
                if not chosen.any(): continue
                lagged = metric.startswith('order_')
                base = dict(model=arm['name'], method=method, regime=regime, subset=subset,
                            metric=metric, feature=feature,
                            horizon_ps=None if lagged else steps*.1, lag_ps=steps*.1 if lagged else None)
                records.append(dict(base, rows=int(chosen.sum()), **estimate(value[chosen], ancestors[chosen],
                    c['seed'], c['evaluation']['bootstrap'])))
                keys, means = source_values(value[chosen], ancestors[chosen])
                source_records.extend(dict(base, ancestor=str(a), value=float(v) if np.isfinite(v) else None)
                                      for a, v in zip(keys, means))
    write_metric_rows(records, out, family=family, name='trajectory')
    write_metric_rows(source_records, out, family=family, name='trajectory_sources')
    forecast = dict(ids=ids, values=arrays, ancestors=ancestors, regimes=regimes, risk=risk,
                    sample_entry=np.concatenate(sample_entries), truth_entry=np.concatenate(truth_entries))
    atomic_save(folder/'trajectory-arrays.pt', forecast)
    atomic_save(folder/'examples.pt', examples)
    _plot(out, diagnostic, retention, examples, forecast)
    summary = {f"trajectory/{r['method']}/{r['regime']}/{r['subset']}/{r['metric']}/{r['horizon_ps'] if r['horizon_ps'] is not None else r['lag_ps']}": r['value']
               for r in records if r['feature'] == '' and r['value'] is not None}
    summary['latent/effective_rank'] = diagnostic['effective_rank']
    update_recorded_summary(folder/'wandb.json', summary, evaluation='kinetic-latent-native-paths',
                            expected=dict(identity=saved['identity']))
    write_json(folder/'evaluation-complete.json', dict(state='complete', rows=len(ids),
        retention_rows=len(retention['ids']), train_phase_probe_rows=diagnostic['train_probe_rows'],
        sampled_paths=particles, source_count=len(set(ancestors)), **inputs,
        local_transition='completion of three consecutive future PTM-like frames at 0.1 ps spacing (0.2 ps first-to-last span); frozen train-only logistic probe',
        scope='conditional enriched local paths; not whole-cell nucleus formation, natural rates or free energy'))
    return True
