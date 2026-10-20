"""Matched native-path scores, complete validity accounting and nucleus readouts.

The shared kinetic assessor owns the scientific path/ordering definitions. This
adapter adds an audit of every requested path even when that assessor records a
generation failure, and names zero-noise flow trajectories honestly.
"""
from collections import defaultdict
import time

import numpy as np
import torch

from src.experiment_runner.artifacts import file_hash, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.research.continuous_motion.data import atomic_save
from src.research.ni_kinetic_latent import assessment as native
from src.research.ni_kinetic_latent import births
from .data import load, load_births, root, read, labels_path, FAMILY


def _deterministic_method(arm):
    return 'conditional_mean' if arm['predictor'] == 'mdn' else 'zero_noise'


def _bounds(columns):
    """Necessary inequalities from the actual BO/density/velocity producers.

    These are not sufficient for existence of a corresponding atomistic state.
    Cross-radial angular powers and signed bond-power rates are unconstrained.
    """
    lower = torch.full((len(columns),), -torch.inf)
    upper = torch.full((len(columns),), torch.inf)
    for i, name in enumerate(columns):
        if name.startswith('bond_order/'):
            suffix = name.split('/', 1)[1].split('_', 1)[1]
            if suffix in ('q', 'qbar', 'neighbor_q_mean', 'neighbor_q_std', 'coherence_std'):
                lower[i], upper[i] = 0., 1.
            elif suffix in ('coherence_mean', 'coherence_q10', 'coherence_q50', 'coherence_q90'):
                lower[i], upper[i] = -1., 1.
        elif name.startswith('density/radial'):
            lower[i] = 0.
        elif name.startswith('density/l'):
            _, a, b = name.split('_')
            if a == b:
                lower[i] = 0.
        elif name.startswith('velocity/') and name.endswith(('second', 'squared')):
            lower[i] = 0.
    return lower, upper


class _Validity(torch.nn.Module):
    """Transparent model adapter; inspect generated values without changing them."""
    def __init__(self, model, c, arm, bank, ids, folder, inputs):
        super().__init__()
        self.model, self.c, self.bank = model, c, bank
        self.deterministic_method = _deterministic_method(arm)
        self.ids, self.path = ids, folder/'validity-partial.pt'
        self.lower, self.upper = _bounds(bank['columns'])
        if c.get('workflow') == 'validation':
            from .support import MeasurementSupport
            self.measurement_audit = MeasurementSupport(bank, epsilon=c['support']['epsilon'])
            self.lower = self.measurement_audit.lower
            self.upper = self.measurement_audit.upper
        train = bank['x'][bank['train_rows']]
        self.train_min = train.amin((0, 1))
        self.train_max = train.amax((0, 1))
        self.identity = dict(inputs, panel=ids.tolist(), seed=c['seed'],
                             particles=c['evaluation']['particles'])
        self.records = {}
        if self.path.exists():
            previous = torch.load(self.path, map_location='cpu', weights_only=False)
            if previous['identity'] != self.identity:
                raise ValueError('Changed inputs for the complete path-validity audit')
            self.records = previous['records']

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)

    def _record(self, z, samples, method, offset):
        raw = native.inverse(samples, self.bank)
        lower, upper = self.lower.to(raw), self.upper.to(raw)
        finite = torch.isfinite(samples)
        raw_finite = torch.isfinite(raw)
        violation = (raw < lower-1e-5) | (raw > upper+1e-5)
        joint = None
        if self.c.get('workflow') == 'validation':
            from .support import validity_metrics
            joint = validity_metrics(raw, self.bank['columns'], self.bank['raw_scale'])
            violation = joint['scalar_violation']
        outside = ((samples < self.train_min.to(samples)) |
                   (samples > self.train_max.to(samples)))
        result = {}
        for steps in (10, 30, 60):
            nf = finite[:, :, :steps]
            rf = raw_finite[:, :, :steps]
            bad = violation[:, :, :steps]
            excursion = outside[:, :, :steps]
            latent_finite = torch.isfinite(z[:, :, :steps])
            values = dict(
                latent_nonfinite_path_fraction=(~latent_finite.all(-1).all(-1)).float().mean(1),
                normalized_nonfinite_path_fraction=(~nf.all(-1).all(-1)).float().mean(1),
                raw_nonfinite_path_fraction=(~rf.all(-1).all(-1)).float().mean(1),
                raw_nonfinite_coordinate_fraction=(~rf).float().mean((1, 2, 3)),
                constraint_violation_path_fraction=bad.any(-1).any(-1).float().mean(1),
                constraint_violation_state_fraction=bad.any(-1).float().mean((1, 2)),
                raw_invalid_or_constraint_path_fraction=((~rf).any(-1) | bad.any(-1)).any(-1).float().mean(1),
                outside_train_range_coordinate_fraction=excursion.float().mean((1, 2, 3)),
                outside_train_range_path_fraction=excursion.any(-1).any(-1).float().mean(1))
            if joint is not None:
                combined = bad.any(-1)
                for kind in ('density_psd', 'density_l0_identity', 'velocity_moment'):
                    failed = joint[kind+'_violation'][:, :, :steps]
                    if failed.ndim == 4:
                        failed = failed.any(-1)
                    combined = combined | failed
                    values[kind+'_state_fraction'] = failed.float().mean((1, 2))
                    values[kind+'_path_fraction'] = failed.any(-1).float().mean(1)
                values['any_necessary_constraint_state_fraction'] = combined.float().mean((1, 2))
                values['any_necessary_constraint_path_fraction'] = combined.any(-1).float().mean(1)
            result.update({(steps, metric): value.detach().cpu().numpy()
                           for metric, value in values.items()})
        self.records[method, offset] = dict(indices=self.ids[offset:offset+len(samples)],
            particles=samples.shape[1], metrics=result)
        if joint is not None:
            self.records[method, offset]['channels'] = dict(
                states=int(np.prod(raw.shape[:-1])),
                violations=joint['scalar_violation'].sum((0, 1, 2)).cpu().numpy(),
                maximum_scaled_excess=joint['scalar_excess'].amax((0, 1, 2)).cpu().numpy())
        atomic_save(self.path, dict(identity=self.identity, records=self.records))

    @torch.no_grad()
    def rollout(self, historyx, steps, particles, seed, deterministic=False):
        z, samples = self.model.rollout(historyx, steps, particles, seed, deterministic=deterministic)
        method = self.deterministic_method if deterministic else 'model'
        offset = seed-self.c['seed']
        self._record(z, samples, method, offset)
        if ('saved_MD', offset) not in self.records:
            actual = self.bank['x'][self.ids[offset:offset+len(samples)], 4:].to(samples)[:, None]
            # The direct observed measurement is its own audit state; no encoder
            # is applied to a raw MD reference in this validity comparison.
            self._record(actual, actual, 'saved_MD', offset)
        if self.c.get('workflow') == 'validation' and ('encoded_MD', offset) not in self.records:
            chosen = self.ids[offset:offset+len(samples)]
            encoded = self.model.encode(self.bank['input'][chosen, 4:].to(samples))[:, None]
            self._record(encoded, self.model.decode(encoded), 'encoded_MD', offset)
        return z, samples


def _publish_validity(c, arm, bank, ids, audit, out, *, family=FAMILY):
    ancestors = np.asarray([bank['rows'][i]['ancestor'] for i in ids])
    regimes = np.asarray([bank['rows'][i]['regime'] for i in ids])
    records, sources = [], []
    methods = ('model', audit.deterministic_method, 'saved_MD')
    if c.get('workflow') == 'validation':
        methods += ('encoded_MD',)
    for method in methods:
        pieces = [audit.records[method, offset]
                  for offset in range(0, len(ids), c['evaluation']['batch_size'])]
        np.testing.assert_array_equal(np.concatenate([part['indices'] for part in pieces]), ids)
        particles = pieces[0]['particles']
        for key in pieces[0]['metrics']:
            steps, metric = key
            values = np.concatenate([part['metrics'][key] for part in pieces])
            for regime in ('all', *native.REGIMES):
                mask = np.ones(len(ids), bool) if regime == 'all' else regimes == regime
                if not mask.any():
                    continue
                scope = dict(model=arm['name'], method=method, regime=regime,
                             horizon_ps=steps*.1, metric=metric)
                records.append(dict(scope, rows=int(mask.sum()), paths=int(mask.sum())*particles,
                    **native.estimate(values[mask], ancestors[mask], c['seed'], c['evaluation']['bootstrap'])))
                keys, means = native.source_values(values[mask], ancestors[mask])
                sources.extend(dict(scope, ancestor=str(key), value=float(value))
                               for key, value in zip(keys, means))
    write_metric_rows(records, out, family=family, name='validity')
    write_metric_rows(sources, out, family=family, name='validity_sources')
    if c.get('workflow') == 'validation':
        channels = []
        for method in methods:
            pieces = [value['channels'] for (record_method, _), value in audit.records.items()
                      if record_method == method]
            states = sum(piece['states'] for piece in pieces)
            count = np.sum([piece['violations'] for piece in pieces], axis=0)
            excess = np.max([piece['maximum_scaled_excess'] for piece in pieces], axis=0)
            channels.extend(dict(model=arm['name'], method=method, column=column,
                horizon_ps=6., states=states, violation_count=int(count[i]),
                violation_fraction=float(count[i]/states), maximum_scaled_excess=float(excess[i]))
                for i, column in enumerate(bank['columns']))
        write_metric_rows(channels, out, family=family, name='validity_channels')
    return records


def evaluate(c, arm, deadline, *, family=FAMILY):
    """Native trajectories and all-path validity; independent of birth evaluation."""
    from .learning import fitted
    out = root(c)/'analyses'/arm['name']; folder = out/'technical'
    if (folder/'flow-evaluation-complete.json').exists():
        return True
    plan, bank = load(c, arm)
    model, saved = fitted(c, arm)
    ids = native.selected_rows(bank, c['evaluation']['per_source_regime'], c['seed'])
    inputs = dict(checkpoint_sha256=file_hash(folder/'final.pt'), data_identity=plan['data_identity'])
    audit = _Validity(model, c, arm, bank, ids, folder, inputs)
    write_json(folder/'likelihood-availability.json', dict(
        teacher_forced_log_likelihood='not_reported',
        reason='Common comparison uses sample-based proper scores. A flow-matching objective is not a tractable density evaluation and is never relabeled NLL.',
        deterministic_rollout=_deterministic_method(arm),
        deterministic_interpretation='Recursive MDN conditional means or a zero-noise flow representative; the latter is not the conditional expectation.'))
    complete = native.evaluate(c, arm, deadline,
        load_provider=lambda _c, _arm: (plan, bank), fitted_provider=lambda _c, _arm: (audit, saved),
        family=family, input_key='input', labels_path=labels_path(c),
        deterministic_method=_deterministic_method(arm))
    if not complete:
        return False
    # Shared scoring records a numerical generation failure without exporting
    # misleading proper scores. Finish counting every requested path anyway.
    device = next(model.parameters()).device
    with torch.no_grad():
        for offset in range(0, len(ids), c['evaluation']['batch_size']):
            for method, particles, deterministic in (
                    ('model', c['evaluation']['particles'], False), (_deterministic_method(arm), 1, True)):
                if (method, offset) in audit.records:
                    continue
                if time.time() > deadline-180:
                    return False
                chunk = ids[offset:offset+c['evaluation']['batch_size']]
                audit.rollout(bank['input'][chunk, :4].to(device), 60, particles,
                              c['seed']+offset, deterministic=deterministic)
                write_json(folder/'validity-progress.json', dict(
                    completed_batches=len(audit.records), total_rows=len(ids), stage='all-path audit'))
    records = _publish_validity(c, arm, bank, ids, audit, out, family=family)
    update_recorded_summary(folder/'wandb.json', {
        f"validity/{r['method']}/{r['metric']}/{r['horizon_ps']}ps": r['value']
        for r in records if r['regime'] == 'all'}, evaluation='flow-all-path-validity',
        expected=dict(identity=saved['identity']))
    write_json(folder/'flow-evaluation-complete.json', dict(
        state='complete', native_status=read(folder/'evaluation-complete.json')['state'],
        rows=len(ids), sampled_paths=len(ids)*c['evaluation']['particles'],
        deterministic_paths=len(ids), **inputs,
        validity='All requested paths counted; no clipping, retries, replacement or survivor-only averaging',
        physical_scope='Necessary measurement bounds only; passing does not prove atomistic realizability'))
    return True


def controls(c, deadline):
    def observed_history(config, bank):
        cohort, central, _inputs = load_births(config, bank)
        return cohort, central
    return births.controls(c, deadline, load_provider=load,
                           birth_provider=observed_history, family=FAMILY)


def evaluate_births(c, arm, deadline):
    """Unchanged full birth assay; no generated fixed-membership history feature."""
    if not arm['birth_evaluate']:
        write_json(root(c)/'analyses'/arm['name']/'technical/birth-evaluation-complete.json', dict(
            state='not_repeated_present_features', arm=arm,
            reason='Present features are unchanged across predictor variants. Representative frozen encoders are evaluated once per seed; direct244 is covered by the shared measurement control. Changing the trajectory predictor cannot change this present-state readout.',
            generated_future_features=False))
        return True
    from .learning import fitted
    return births.evaluate(c, arm, deadline, load_provider=load, birth_provider=load_births,
        fitted_provider=fitted, family=FAMILY,
        encoder_input_description='Present observed measurement input only; independent full 32/64-atom birth assay')


def compare(c, *, family=FAMILY):
    """Shared source-paired native/birth contrasts plus all-path validity contrasts."""
    births.compare(c, family=family, treatment_key='treatment')
    out = root(c)/'analyses/comparison'
    scopes = ('method', 'regime', 'horizon_ps', 'metric')
    bundles = defaultdict(lambda: defaultdict(dict))
    individual = []
    for arm in c['arms']:
        path = root(c)/'analyses'/arm['name']/'tables/validity_sources.csv'
        if not path.exists():
            continue
        for row in births._rows(path):
            key = tuple(row[field] for field in scopes)
            bundles[arm['treatment'], key][arm['seed']][row['ancestor']] = float(row['value'])
        individual.extend(dict(row, treatment=arm['treatment'], seed=arm['seed'])
                          for row in births._rows(path.with_name('validity.csv')))
    paired = []
    for left, right in c['comparisons']:
        keys = sorted(key for treatment, key in bundles if treatment == left and (right, key) in bundles)
        for key in keys:
            result = births._source_pair(bundles[left, key], bundles[right, key], c['model_seeds'],
                                        c['evaluation']['bootstrap'], c['seed'])
            if result is not None:
                paired.append(dict(left=left, right=right, **dict(zip(scopes, key)), **result))
    if individual:
        write_metric_rows(individual, out, family=family, name='validity_individual')
    if paired:
        write_metric_rows(paired, out, family=family, name='validity_paired')
