"""Frozen nucleus readouts of present kinetic features; no future labels in encoders."""
import csv
from collections import defaultdict
from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.research.ni_phase_sim.births import fit_spaces
from src.research.birth_prediction.fit import scores
from src.research.birth_prediction.analysis import source_draws, bootstrap_scores
from .data import load, load_births, root, cache, read, write_json, sha, digest

FAMILY = 'ni_kinetic_latent'
SPACES = ['latent', 'reconstructed', 'measurements_latent', 'history_measurements_latent']


def _rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def settings(c):
    return dict(c['birth_probe'], output=str(root(c)/'births'), cache=str(cache(c)/'birth-features'))


def controls(c, deadline, *, load_provider=None, birth_provider=None, family=FAMILY):
    p, bank = (load if load_provider is None else load_provider)(c)
    cohort, history = (load_births if birth_provider is None else birth_provider)(c, bank)
    values = history.numpy()
    failures = fit_spaces(settings(c), p, cohort, 'controls', dict(
        prior=None, structural=values[:, -1, :227], measurements=values[:, -1],
        history_measurements=values.reshape(len(values), -1)), family=family)
    if failures:
        raise RuntimeError('One or more fixed measurement controls failed')
    write_json(root(c)/'analyses/controls/technical/controls-complete.json', dict(state='complete'))
    return True


def evaluate(c, arm, deadline, *, load_provider=None, birth_provider=None,
             fitted_provider=None, family=FAMILY, extra_spaces_provider=None,
             encoder_input_description='present244 only; no future labels or fixed-membership past'):
    from .learning import fitted
    folder = root(c)/'analyses'/arm['name']/'technical'
    if (folder/'birth-evaluation-complete.json').exists():
        return True
    model, saved = (fitted if fitted_provider is None else fitted_provider)(c, arm)
    p, bank = load(c) if load_provider is None else load_provider(c, arm)
    if birth_provider is None:
        cohort, history = load_births(c, bank)
        input_history = history
    else:
        cohort, history, input_history = birth_provider(c, bank, arm)
    latent, decoded = [], []
    with torch.no_grad():
        for current in input_history[:, -1].split(c['birth_probe']['batch_size']):
            if time.time() > deadline-180:
                return False
            z = model.encode(current.cuda())
            latent.append(z.cpu()); decoded.append(model.decode(z).cpu())
    z = torch.cat(latent).numpy(); decoded = torch.cat(decoded).numpy(); observed = history.numpy()
    if not np.isfinite(z).all() or not np.isfinite(decoded).all():
        raise FloatingPointError('Nonfinite current-observation birth features: '+arm['name'])
    spaces = dict(latent=z, reconstructed=decoded,
        measurements_latent=np.concatenate((observed[:, -1], z), -1),
        history_measurements_latent=np.concatenate((observed.reshape(len(z), -1), z), -1))
    if extra_spaces_provider is not None:
        spaces.update(extra_spaces_provider(observed, input_history))
    failures = fit_spaces(settings(c), p, cohort, arm['name'], spaces, family=family)
    if failures:
        raise RuntimeError('One or more frozen nucleus readouts failed')
    fields = {}
    for path in (root(c)/'births/analyses'/arm['name']).glob('*/*/lead-*/tables/scores.csv'):
        for row in _rows(path):
            fields[f'birth/{row["space"]}/{row["task"]}/{row["lead_ps"]}ps/{row["metric"]}'] = float(row['value'])
    update_recorded_summary(folder/'wandb.json', fields, evaluation='nucleus-readouts',
                            expected=dict(identity=saved['identity']))
    write_json(folder/'birth-evaluation-complete.json', dict(state='complete', rows=len(cohort['rows']),
        unique_states=len(z), checkpoint_sha256=sha(folder/'final.pt'),
        encoder_input=encoder_input_description,
        history_control='existing original four-frame fixed-membership measurements'))
    return True


def _contrasts(c, *, treatment_key='kind', spaces=SPACES):
    contrasts = [(a, space, b, space) for a, b in c['comparisons'] for space in spaces]
    for kind in sorted({a[treatment_key] for a in c['arms']}):
        contrasts += [(kind, 'measurements_latent', 'controls', 'measurements'),
                      (kind, 'history_measurements_latent', 'controls', 'history_measurements')]
    return contrasts


def _birth_bundle(c, ec, path, model, space, task, lead, out):
    """Cache expensive bootstrap curves against sealed producer identities."""
    receipt = read(path.parent/'complete.json')
    plan = read(root(c)/'technical/plan.json')
    expected = digest([plan['identity'], model, space, task, lead])
    if receipt['identity'] != expected or sha(path) != receipt['sha256']:
        raise ValueError(f'Changed frozen birth predictions: {path}')
    with np.load(path) as saved:
        take = saved['role'] == 'test'
        data = {key: saved[key][take].copy() for key in
                ('id', 'ancestor', 'event', 'label', 'weight', 'probability')}
    identity = digest([receipt['sha256'], ec['bootstrap'], ec['seed'],
                       sha(Path(bootstrap_scores.__code__.co_filename)),
                       sha(Path(scores.__code__.co_filename))])
    target = out/'technical/bootstrap'/f'{identity}.npz'
    marker = target.with_suffix('.json')
    if marker.exists():
        done = read(marker)
        if done['identity'] != identity or sha(target) != done['sha256']:
            raise ValueError('Changed cached paired bootstrap: '+str(target))
        with np.load(target) as saved:
            boot = {metric: saved[metric].copy() for metric in ('ap', 'auroc', 'nll', 'brier')}
            valid = int(saved['valid'])
    else:
        _, slots, draws = source_draws(data['ancestor'], ec['bootstrap'], ec['seed'])
        boot, valid = bootstrap_scores(data['label'], data['probability'], data['weight'], slots, draws, None)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.building.npz')
        np.savez_compressed(temporary, **boot, valid=valid)
        temporary.replace(target)
        write_json(marker, dict(identity=identity, sha256=sha(target),
                               predictions=str(path), predictions_sha256=receipt['sha256']))
    return dict(data=data, scores=scores(data['label'], data['probability'], data['weight']),
                bootstrap=boot, valid=valid)


def _check_paired_births(left, right):
    # Equal IDs alone would not establish equal bootstrap populations or weights.
    for key in ('id', 'ancestor', 'event', 'label', 'weight'):
        np.testing.assert_array_equal(left['data'][key], right['data'][key],
                                      err_msg='Unmatched paired birth field: '+key)
    if left['valid'] != right['valid']:
        raise ValueError('Paired birth bootstrap retained different source draws')


def _compare_births(c, out, *, family=FAMILY, treatment_key='kind', spaces=SPACES,
                    extra_contrasts=()):
    ec = settings(c); individual = []
    for path in root(ec).glob('analyses/*/*/*/lead-*/tables/scores.csv'):
        if (path.parent.parent/'technical/complete.json').exists():
            individual.extend(_rows(path))
    if individual:
        write_metric_rows(individual, out, family=family, name='birth_individual')
    grouped = defaultdict(list)
    for row in individual:
        key = (row['model'].split('-s')[0], row['space'], row['task'], row['lead_ps'], row['metric'])
        grouped[key].append(float(row['value']))
    summary = [dict(treatment=k[0], space=k[1], task=k[2], lead_ps=k[3], metric=k[4],
                    value=float(np.mean(v)), seed_sd=float(np.std(v)), seeds=len(v)) for k, v in grouped.items()]
    if summary:
        write_metric_rows(summary, out, family=family, name='birth_summary')
    paired = []
    contrasts = _contrasts(c, treatment_key=treatment_key, spaces=spaces)+list(extra_contrasts)
    needed = set()
    for left, ls, right, rs in contrasts:
        for seed in c['model_seeds']:
            needed.add((f'{left}-s{seed}', ls))
            needed.add(('controls' if right == 'controls' else f'{right}-s{seed}', rs))
    for task in ec['tasks']:
        for lead in ec['leads_ps']:
            bundles = {}
            for path in root(ec).glob(f'analyses/*/*/{task}/lead-{lead:g}ps/technical/predictions.npz'):
                model, space = path.relative_to(root(ec)/'analyses').parts[:2]
                if (model, space) in needed and (path.parent/'complete.json').exists():
                    bundles[model, space] = _birth_bundle(c, ec, path, model, space, task, lead, out)
            for left, ls, right, rs in contrasts:
                changes, boots = defaultdict(list), defaultdict(list); matched = []
                population = None
                for seed in c['model_seeds']:
                    a = bundles.get((f'{left}-s{seed}', ls))
                    b = bundles.get(('controls' if right == 'controls' else f'{right}-s{seed}', rs))
                    if a is None or b is None:
                        continue
                    _check_paired_births(a, b)
                    if population is not None:
                        _check_paired_births(a, population)
                    population = a
                    matched.append(seed)
                    for metric in ['ap', 'auroc', 'nll', 'brier']:
                        changes[metric].append(a['scores'][metric]-b['scores'][metric])
                        boots[metric].append(a['bootstrap'][metric]-b['bootstrap'][metric])
                for metric, values in changes.items():
                    lo, hi = np.quantile(np.mean(boots[metric], axis=0), [.025, .975])
                    paired.append(dict(left=left, left_space=ls, right=right, right_space=rs,
                        task=task, lead_ps=lead, metric=metric, difference=float(np.mean(values)),
                        lower95=float(lo), upper95=float(hi), seeds=len(values),
                        configured_seeds=len(c['model_seeds']),
                        matched_seeds=','.join(map(str, matched)),
                        status='complete' if len(matched) == len(c['model_seeds']) else 'partial_seeds'))
    if paired:
        write_metric_rows(paired, out, family=family, name='birth_paired')


def _source_pair(left, right, seeds, bootstrap, seed):
    """Mean matched-seed difference, with whole-source uncertainty only."""
    available = [s for s in seeds if s in left and s in right]
    if not available:
        return None
    sources = sorted(left[available[0]])
    values = []
    for s in available:
        if sorted(left[s]) != sources or sorted(right[s]) != sources:
            raise ValueError('Native paired comparison has mismatched source identities')
        values.append([left[s][source]-right[s][source] for source in sources])
    values = np.asarray(values, dtype=float)
    missing = int((~np.isfinite(values)).any(0).sum())
    base = dict(seeds=len(available), configured_seeds=len(seeds),
                matched_seeds=','.join(map(str, available)), sources=len(sources),
                nonfinite_sources=missing)
    if missing:
        return dict(base, difference=None, lower95=None, upper95=None, seed_sd=None,
                    status='undefined_source_values')
    # Average the same fitted seeds first, then resample complete independent sources.
    source_mean = values.mean(0)
    draw = np.random.default_rng(seed).integers(len(sources), size=(bootstrap, len(sources)))
    lo, hi = np.quantile(source_mean[draw].mean(1), [.025, .975])
    return dict(base, difference=float(source_mean.mean()), lower95=float(lo), upper95=float(hi),
                seed_sd=float(values.mean(1).std()),
                status='complete' if len(available) == len(seeds) else 'partial_seeds')


def _compare_native(c, out, *, family=FAMILY, treatment_key='kind'):
    # Native populations and protocol controls remain separate from the birth assay.
    for table in ['retention', 'trajectory', 'latent_spectrum']:
        rows = []
        for arm in c['arms']:
            path = root(c)/'analyses'/arm['name']/'tables'/f'{table}.csv'
            if path.exists():
                rows.extend(dict(row, model=arm['name'], treatment=arm[treatment_key], seed=arm['seed'])
                            for row in _rows(path))
        if rows:
            write_metric_rows(rows, out, family=family, name=table+'_individual')
    fields = ('method', 'regime', 'subset', 'metric', 'feature', 'horizon_ps', 'lag_ps')
    for table in ['retention', 'trajectory']:
        bundles = defaultdict(lambda: defaultdict(dict))
        for arm in c['arms']:
            path = root(c)/'analyses'/arm['name']/'tables'/f'{table}_sources.csv'
            if not path.exists():
                continue
            for row in _rows(path):
                key = tuple(row.get(field, '') for field in fields)
                source = row['ancestor']
                slot = bundles[arm[treatment_key], key][arm['seed']]
                if source in slot:
                    raise ValueError('Duplicate native source summary: '+str(path))
                slot[source] = float(row['value']) if row['value'] else np.nan
        paired = []
        for left, right in c['comparisons']:
            scopes = sorted(key for kind, key in bundles if kind == left and (right, key) in bundles)
            for key in scopes:
                result = _source_pair(bundles[left, key], bundles[right, key], c['model_seeds'],
                                      c['evaluation']['bootstrap'], c['seed'])
                if result is None:
                    continue
                failed = []
                if table == 'trajectory':
                    for seed in c['model_seeds']:
                        if any((root(c)/'analyses'/f'{kind}-s{seed}'/'technical/generation-failure.json').exists()
                               for kind in (left, right)):
                            failed.append(seed)
                if failed:
                    result.update(difference=None, lower95=None, upper95=None, seed_sd=None,
                                  status='generation_failure')
                paired.append(dict(left=left, right=right, **dict(zip(fields, key)), **result,
                    failed_seeds=','.join(map(str, failed)),
                    uncertainty='whole-source bootstrap of matched-seed mean; fitted seeds fixed'))
        if paired:
            write_metric_rows(paired, out, family=family, name=table+'_paired')


def compare(c, *, family=FAMILY, treatment_key='kind', spaces=SPACES, extra_contrasts=()):
    """Publish every available native and birth analysis independently."""
    out = root(c)/'analyses/comparison'
    _compare_native(c, out, family=family, treatment_key=treatment_key)
    _compare_births(c, out, family=family, treatment_key=treatment_key,
                    spaces=spaces, extra_contrasts=extra_contrasts)
