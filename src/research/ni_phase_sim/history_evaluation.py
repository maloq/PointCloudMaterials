"""Matched physical and nucleus readouts for history-conditioned dynamics."""
import csv
from collections import defaultdict
from pathlib import Path
import time
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.research.birth_prediction.fit import scores
from src.research.birth_prediction.analysis import source_draws, bootstrap_scores
from src.research.equivariant_context.cache import RetainedCache
from src.research.measurement_sim.features import structural
from .data import read, root, cache, sha, digest, write_json, atomic_save
from .history_data import load, FAMILY
from .history_model import HistoryProcess
from .assessment import measure_errors, aggregate, measurements, figures
from .model import ballistic
from .births import invariant, fit_spaces, measured


def fitted(c, arm):
    folder = root(c)/'analyses'/arm['name']/'technical'
    if sha(folder/'final.pt') != read(folder/'complete.json')['checkpoint_sha256']:
        raise ValueError('Changed final history checkpoint')
    saved = torch.load(folder/'final.pt', map_location='cuda', weights_only=False)
    model = HistoryProcess(c, arm).cuda().eval(); model.load_state_dict(saved['model'])
    return model, saved


def evaluate(c, arm, deadline):
    folder = root(c)/'analyses'/arm['name']/'technical'
    if (folder/'evaluation-complete.json').exists():
        return True
    model, saved = fitted(c, arm)
    p, records, state, tangent, history = load(c, 'test')
    norm = {k: v.cuda() for k, v in saved['normalization'].items()}
    predictions = []; responses = []; latent = []
    for offset in range(0, len(state), c['evaluation']['batch_size']):
        if time.time() > deadline-180:
            return False
        stop = offset+c['evaluation']['batch_size']; x = state[offset:stop, 0]
        with torch.no_grad():
            bound = model.condition(history[offset:stop])
            predictions.append(bound(x, 20).cpu()); derivatives = []
            for k in range(2):
                _, value = torch.func.jvp(lambda y: bound(y, 20), (x,), (tangent[offset:stop, 0, ..., k],))
                derivatives.append(value.cpu())
            responses.append(torch.stack(derivatives, -1))
            h = state[offset:stop, :-1].flatten(0, 1); future = state[offset:stop, 1:].flatten(0, 1)
            _, left = torch.func.jvp(lambda y: bound.step(bound.encode(y)),
                (h,), (tangent[offset:stop, :-1, ..., 0].flatten(0, 1),))
            _, right = torch.func.jvp(bound.encode, (future,), (tangent[offset:stop, 1:, ..., 0].flatten(0, 1),))
            dims = tuple(range(1, left.ndim))
            latent.extend([dict(job=records[offset+i//20]['job']['key'], horizon_fs=(i%20+1)*5,
                response_mse=float(a), target_power=float(b)) for i, (a, b) in enumerate(zip(
                    (left-right).square().mean(dims), right.square().mean(dims)))])
    prediction = torch.cat(predictions).cuda(); response = torch.cat(responses).cuda()
    actual = state[:, 1:]; true_response = tangent[:, 1:]; base = ballistic(state[:, 0], 20)
    times = torch.arange(1, 21, device='cuda')*.01
    base_response = tangent[:, None, 0].expand(-1, 20, -1, -1, -1).clone()
    base_response[..., :3, :] += times[None, :, None, None, None]*tangent[:, None, 0, :, 3:, :]
    rows = []; all_metrics = {}
    for name, value, derivative in [('neural', prediction, response), ('ballistic', base, base_response),
            ('persistence', state[:, :1].expand(-1, 20, -1, -1), tangent[:, :1].expand(-1, 20, -1, -1, -1))]:
        result = measure_errors(value, actual, derivative, true_response, norm)
        all_metrics[name] = {k: v.cpu() for k, v in result.items()}; rows.extend(aggregate(result, records, name))
    zero = {k: v for k, v in measure_errors(actual, actual, torch.zeros_like(response), true_response, norm).items() if '/response_' in k}
    all_metrics['zero_response'] = {k: v.cpu() for k, v in zero.items()}; rows.extend(aggregate(zero, records, 'zero_response'))
    _, _, train, _, _ = load(c, 'train'); scale = measurements(train[:, 0]).std(0).clamp_min(1e-5)
    ancestors = sorted({r['ancestor'] for r in records})
    def source_mean(value):
        return float(np.mean([float(value[[i for i, r in enumerate(records) if r['ancestor'] == a]].mean()) for a in ancestors]))
    measurement_rows = []
    for index in [0, 3, 9, 19]:
        truth = measurements(actual[:, index])
        for name, values in [('neural', prediction), ('ballistic', base)]:
            error = ((measurements(values[:, index])-truth)/scale).square()
            for group, lo, hi in [('density', 0, 74), ('velocity_shell', 74, 83), ('bond_rates', 83, 87), ('flow', 87, 91)]:
                measurement_rows.append(dict(model=name, horizon_fs=(index+1)*5, group=group,
                    normalized_mse=source_mean(error[:, lo:hi].mean(-1)), states=len(records), ancestors=len(ancestors)))
    names = p['columns'][:153]; factor = c['length_factor']
    reference = structural((train[:, 0, :, :3]*5).cpu().numpy(), names, factor)
    truth = structural((actual[:, -1, :, :3]*5).cpu().numpy(), names, factor)
    scale = reference.std(0); active = scale > 1e-8
    for name, values in [('neural', prediction), ('ballistic', base)]:
        estimate = structural((values[:, -1, :, :3]*5).cpu().numpy(), names, factor)
        for group, lo, hi in [('bond_order', 0, 40), ('tda', 40, 153)]:
            columns = np.arange(lo, hi)[active[lo:hi]]
            per = (((estimate[:, columns]-truth[:, columns])/scale[columns])**2).mean(-1)
            measurement_rows.append(dict(model=name, horizon_fs=100, group=group,
                normalized_mse=source_mean(per), states=len(records), ancestors=len(ancestors)))
    with torch.no_grad():
        x = state[:4, 0]; u = tangent[:4, 0, ..., 0]; bound = model.condition(history[:4])
        _, jvp = torch.func.jvp(lambda y: bound(y, 20), (x,), (u,))
        secant = (bound(x+.1*u, 20)-bound(x-.1*u, 20))/.2
        numerical = float((jvp-secant).norm()/jvp.norm().clamp_min(1e-8))
    atomic_save(folder/'predictions.pt', dict(prediction=prediction.cpu(), response=response.cpu(),
        state=state.cpu(), tangent=tangent.cpu(), metrics=all_metrics,
        rows=[dict(job=r['job']['key'], source=r['source'], ancestor=r['ancestor'], atom_ids=r['atom_ids']) for r in records]))
    for name, values in [('prediction', rows), ('measurements', measurement_rows), ('latent_transport', latent)]:
        write_metric_rows(values, folder.parent, family=FAMILY, name=name)
    figures(folder, prediction, state, records)
    summary = {f'evaluation/{r["metric"]}/{r["horizon_fs"]}fs': r['value'] for r in rows if r['model'] == 'neural'}
    summary['evaluation/jvp_secant_error'] = numerical
    write_json(folder/'evaluation-complete.json', dict(states=len(records), jvp_secant_error=numerical))
    update_recorded_summary(folder/'wandb.json', summary, evaluation='history-physical', expected=dict(identity=saved['identity']))
    return True


def birth_config(c):
    return dict(c['birth_probe'], output=str(root(c)/'births'), cache=str(cache(c)/'birth-measurements'))


def controls(c, p):
    ec = birth_config(c); data = torch.load(cache(c)/'births.pt', map_location='cpu', weights_only=False)
    # Fixed measurements for all four observed frames, plus unchanged present controls.
    history_data = dict(data, state=data['history'].flatten(0, 1))
    packet = dict(p, phase_config=c)
    values = measured(ec, packet, history_data).reshape(len(data['state']), 4, 244)
    failures = fit_spaces(ec, p, data, 'controls', dict(prior=None, structural=values[:, -1, :227],
        measurements=values[:, -1], history_measurements=values.reshape(len(values), -1)), family=FAMILY)
    if failures:
        raise RuntimeError(f'History measurement controls failed: {failures}')
    write_json(root(c)/'technical/birth-controls-complete.json', dict(state='complete'))


def births(c, p, arm, deadline):
    folder = root(c)/'analyses'/arm['name']/'technical'
    if (folder/'birth-evaluation-complete.json').exists():
        return True
    model, saved = fitted(c, arm); ec = birth_config(c)
    data = torch.load(cache(c)/'births.pt', map_location='cpu', weights_only=False)
    measurement = torch.load(cache(ec)/'measurements.pt', map_location='cpu', weights_only=False).numpy().reshape(len(data['state']), 4, 244)
    metadata = dict(plan=p['identity'], checkpoint=sha(folder/'final.pt'))
    pool = RetainedCache(cache(c)/'features', 6)
    with pool.lease(digest(metadata), deadline=deadline, metadata=metadata) as bank:
        if not (bank/'complete.json').exists():
            first = []; path = []
            with torch.no_grad():
                for h in data['history'].split(ec['batch_size']):
                    h = h.cuda()/h.new_tensor([5., 5., 5., 10., 10., 10.]).cuda()
                    x = h[:, -1]; bound = model.condition(h); z = bound.encode(x)
                    value = invariant(z, x); values = [value]
                    for step in range(1, 21):
                        z = bound.step(z)
                        if step in [4, 10, 20]:
                            values.append(invariant(z, x)-value)
                    first.append(value.cpu()); path.append(torch.cat(values, -1).cpu())
            value = dict(phi=torch.cat(first).numpy(), path=torch.cat(path).numpy())
            if not all(np.isfinite(v).all() for v in value.values()):
                raise FloatingPointError('Nonfinite history birth features')
            atomic_save(bank/'features.pt', value)
            write_json(bank/'complete.json', dict(sha256=sha(bank/'features.pt')))
        if sha(bank/'features.pt') != read(bank/'complete.json')['sha256']:
            raise ValueError('Changed history feature cache')
        value = torch.load(bank/'features.pt', map_location='cpu', weights_only=False)
        spaces = dict(phi=value['phi'], path=value['path'],
            measurements_path=np.concatenate((measurement[:, -1], value['path']), -1),
            history_measurements_path=np.concatenate((measurement.reshape(len(measurement), -1), value['path']), -1))
        failures = fit_spaces(ec, p, data, arm['name'], spaces, family=FAMILY)
    if failures:
        raise RuntimeError(f'Independent birth fits failed: {failures}')
    fields = {}
    for f in root(ec).glob(f'analyses/{arm["name"]}/*/*/lead-*/tables/scores.csv'):
        with f.open() as stream:
            for r in csv.DictReader(stream):
                fields[f"birth90/{r['space']}/{r['task']}/{r['lead_ps']}ps/{r['metric']}"] = float(r['value'])
    write_json(folder/'birth-evaluation-complete.json', dict(state='complete'))
    update_recorded_summary(folder/'wandb.json', fields, evaluation='history-births', expected=dict(identity=saved['identity']))
    return True


def compare(c):
    individual = []; paired = []
    complete = {}
    for arm in c['arms']:
        f = root(c)/'analyses'/arm['name']/'technical'
        if (f/'evaluation-complete.json').exists():
            complete[arm['name']] = torch.load(f/'predictions.pt', map_location='cpu', weights_only=False)
    for seed in c['model_seeds']:
        for left, right in c['comparisons']:
            a = complete.get(f'{left}-s{seed}'); b = complete.get(f'{right}-s{seed}')
            if a is None or b is None:
                continue
            if [r['job'] for r in a['rows']] != [r['job'] for r in b['rows']]:
                raise ValueError('Unpaired physical histories')
            ancestors = np.asarray([r['ancestor'] for r in a['rows']]); groups = np.unique(ancestors)
            draw = np.random.default_rng(c['seed']).integers(len(groups), size=(1000, len(groups)))
            for metric, value in a['metrics']['neural'].items():
                delta = (value-b['metrics']['neural'][metric]).numpy()
                means = np.stack([delta[ancestors == group].mean(0) for group in groups]); boot = means[draw].mean(1)
                for index in [0, 3, 9, 19]:
                    lo, hi = np.quantile(boot[:, index], [.025, .975])
                    paired.append(dict(left=left, right=right, seed=seed, metric=metric, horizon_fs=(index+1)*5,
                        difference=float(means[:, index].mean()), lower95=lo, upper95=hi, ancestors=len(groups)))
    out = root(c)/'analyses/comparison'
    if paired:
        write_metric_rows(paired, out, family=FAMILY, name='physical_paired')
    ec = birth_config(c)
    for f in root(ec).glob('analyses/*/*/*/lead-*/tables/scores.csv'):
        with f.open() as stream:
            individual.extend(csv.DictReader(stream))
    if individual:
        write_metric_rows(individual, out, family=FAMILY, name='birth_individual')
        groups = defaultdict(list)
        for r in individual:
            groups[r['model'].split('-s')[0], r['space'], r['task'], r['lead_ps'], r['metric']].append(float(r['value']))
        summary = [dict(treatment=k[0], space=k[1], task=k[2], lead_ps=k[3], metric=k[4],
            value=float(np.mean(v)), seed_sd=float(np.std(v)), seeds=len(v)) for k, v in groups.items()]
        write_metric_rows(summary, out, family=FAMILY, name='birth_summary')
    pairs = []
    for task in ec['tasks']:
        for lead in ec['leads_ps']:
            bundles = {}
            for f in root(ec).glob(f'analyses/*/*/{task}/lead-{lead:g}ps/technical/predictions.npz'):
                with np.load(f) as z:
                    te = z['role'] == 'test'; _, slots, draws = source_draws(z['ancestor'][te], ec['bootstrap'], ec['seed'])
                    boot, _ = bootstrap_scores(z['label'][te], z['probability'][te], z['weight'][te], slots, draws, None)
                    name, space = f.parts[-6:-4]
                    bundles[name, space] = (z['id'][te].copy(), scores(z['label'][te], z['probability'][te], z['weight'][te]), boot)
            for space in ['phi', 'path', 'measurements_path', 'history_measurements_path']:
                contrasts = [(a, space, b, space) for a, b in c['comparisons']]
                if space.endswith('measurements_path'):
                    baseline = 'history_measurements' if space.startswith('history_') else 'measurements'
                    contrasts += [(a['name'].split('-s')[0], space, 'controls', baseline) for a in c['arms'] if a['seed'] == c['seed']]
                for left, ls, right, rs in contrasts:
                    differences = defaultdict(list); boots = defaultdict(list)
                    for seed in c['model_seeds']:
                        lk = (f'{left}-s{seed}', ls); rk = ('controls' if right == 'controls' else f'{right}-s{seed}', rs)
                        if lk not in bundles or rk not in bundles:
                            continue
                        a, b = bundles[lk], bundles[rk]; np.testing.assert_array_equal(a[0], b[0])
                        for metric in ['ap', 'auroc', 'nll', 'brier']:
                            differences[metric].append(a[1][metric]-b[1][metric]); boots[metric].append(a[2][metric]-b[2][metric])
                    for metric, v in differences.items():
                        lo, hi = np.quantile(np.mean(boots[metric], axis=0), [.025, .975])
                        pairs.append(dict(left=left, left_space=ls, right=right, right_space=rs, task=task,
                            lead_ps=lead, metric=metric, difference=float(np.mean(v)), lo=lo, hi=hi, seeds=len(v)))
    if pairs:
        write_metric_rows(pairs, out, family=FAMILY, name='birth_paired')
