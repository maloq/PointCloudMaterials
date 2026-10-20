"""Physical, decoded-measurement, and frozen nucleus evaluation."""
from pathlib import Path
import time
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from .data import root, read, atomic_save, write_json
from .history_data import load
from . import history_evaluation as shared
from .births import invariant
from .measurement_data import targets, GROUPS, FAMILY
from .measurement_model import MeasuredHistoryProcess


def evaluate(c, arm, deadline):
    if not shared.evaluate(c, arm, deadline, model_factory=MeasuredHistoryProcess, family=FAMILY):
        return False
    folder = root(c)/'analyses'/arm['name']/'technical'
    if (folder/'measurement-evaluation-complete.json').exists():
        return True
    model, saved = shared.fitted(c, arm, model_factory=MeasuredHistoryProcess)
    _, records, state, tangent, history = load(c, 'test'); data = targets(c, 'test')
    predictions = []; responses = []
    with torch.no_grad():
        for start in range(0, len(state), c['evaluation']['batch_size']):
            if time.time() > deadline-180:
                return False
            stop = start+c['evaluation']['batch_size']; x = state[start:stop, 0]
            bound = model.condition(history[start:stop])
            present = model.measurement(bound.encode(x))
            future = model.measured_path(bound, x, 20)
            predictions.append(torch.cat((present[:, None], future), 1).cpu())
            queries = []
            for k in range(2):
                _, value = torch.func.jvp(lambda h: model.measured_path(bound, h, 20), (x,), (tangent[start:stop, 0, ..., k],))
                queries.append(value[..., 153:].cpu())
            responses.append(torch.stack(queries, -1))
    prediction = torch.cat(predictions); response = torch.cat(responses)
    truth = data['value'].cpu(); true_response = data['response'][:, 1:].cpu(); active = data['active'].cpu()
    ancestors = np.asarray([r['ancestor'] for r in records]); unique = np.unique(ancestors)
    def source_mean(value):
        return float(np.mean([float(value[ancestors == ancestor].mean()) for ancestor in unique]))
    rows = []; errors = {}
    for name, lo, hi in GROUPS:
        mask = active[lo:hi]
        error = (prediction[..., lo:hi]-truth[..., lo:hi])[..., mask].square().mean(-1)
        persistence = (truth[:, :1, lo:hi]-truth[..., lo:hi])[..., mask].square().mean(-1)
        increments = (torch.diff(prediction[..., lo:hi], dim=1)-torch.diff(truth[..., lo:hi], dim=1))[..., mask].square().mean(-1)
        errors['measurement/'+name] = error[:, 1:]
        errors['increment/'+name] = increments
        for index in [0, 1, 4, 10, 20]:
            for model_name, value in [('decoder', error), ('present_persistence', persistence)]:
                rows.append(dict(model=model_name, group=name, metric='normalized_mse', horizon_fs=index*5,
                                 value=source_mean(value[:, index]), states=len(records), ancestors=len(unique)))
        for index in [0, 3, 9, 19]:
            rows.append(dict(model='decoder', group=name, metric='increment_mse', horizon_fs=(index+1)*5,
                             value=source_mean(increments[:, index]), states=len(records), ancestors=len(unique)))
    for name, lo, hi in [('density', 0, 74), ('velocity', 74, 91)]:
        mask = active[153+lo:153+hi]
        for k, query in enumerate(['Q', 'V']):
            error = (response[..., lo:hi, k]-true_response[..., lo:hi, k])[..., mask].square().mean(-1)
            power = true_response[..., lo:hi, k][..., mask].square().mean(-1)
            errors[f'measurement_response/{query}/{name}'] = error
            for index in [0, 3, 9, 19]:
                for metric, value in [('response_mse', source_mean(error[:, index])),
                    ('target_power', source_mean(power[:, index])),
                    ('response_zero_ratio', source_mean(error[:, index])/max(source_mean(power[:, index]), 1e-12))]:
                    rows.append(dict(model='decoder', group=f'{name}/{query}', metric=metric, horizon_fs=(index+1)*5,
                                     value=value, states=len(records), ancestors=len(unique)))
    atomic_save(folder/'measurement-predictions.pt', dict(prediction=prediction, response=response, target=truth,
        target_response=true_response, errors=errors, ancestors=ancestors, keys=data['keys']))
    write_metric_rows(rows, folder.parent, family=FAMILY, name='measurement_decoder')
    update_recorded_summary(folder/'wandb.json',
        {f'measurement_decoder/{r["group"]}/{r["metric"]}/{r["horizon_fs"]}fs': r['value'] for r in rows if r['model'] == 'decoder'},
        evaluation='measurement-decoder', expected=dict(identity=saved['identity']))
    write_json(folder/'measurement-evaluation-complete.json', dict(state='complete', states=len(records)))
    return True


def extract(model, data, batch_size):
    blocks = {k: [] for k in ['phi', 'path', 'decoded_present', 'decoded_path']}
    with torch.no_grad():
        for history in data['history'].split(batch_size):
            history = history.cuda()/history.new_tensor([5., 5., 5., 10., 10., 10.]).cuda()
            x = history[:, -1]; bound = model.condition(history); z = bound.encode(x)
            first = invariant(z, x); first_y = model.measurement(z)
            path = [first]; measured_path = [first_y]
            for step in range(1, 21):
                z = bound.step(z)
                if step in [4, 10, 20]:
                    path.append(invariant(z, x)-first); measured_path.append(model.measurement(z)-first_y)
            for key, value in dict(phi=first, path=torch.cat(path, -1), decoded_present=first_y,
                                   decoded_path=torch.cat(measured_path, -1)).items():
                blocks[key].append(value.cpu())
    return {k: torch.cat(v).numpy() for k, v in blocks.items()}


def births(c, p, arm, deadline):
    def extra(value, measurement):
        return dict(decoded_present=value['decoded_present'], decoded_path=value['decoded_path'],
            history_measurements_decoded_path=np.concatenate((measurement.reshape(len(measurement), -1), value['decoded_path']), -1))
    return shared.births(c, p, arm, deadline, model_factory=MeasuredHistoryProcess,
                         family=FAMILY, extract=extract, extra_spaces=extra)


def compare(c):
    shared.compare(c, family=FAMILY)
    packets = {}
    for arm in c['arms']:
        f = root(c)/'analyses'/arm['name']/'technical/measurement-predictions.pt'
        if f.exists():
            packets[arm['name']] = torch.load(f, map_location='cpu', weights_only=False)
    rows = []
    for seed in c['model_seeds']:
        for left, right in c['comparisons']:
            a = packets.get(f'{left}-s{seed}'); b = packets.get(f'{right}-s{seed}')
            if a is None or b is None:
                continue
            if a['keys'] != b['keys']:
                raise ValueError('Unpaired measurement comparison')
            groups = np.unique(a['ancestors']); draw = np.random.default_rng(c['seed']).integers(len(groups), size=(1000, len(groups)))
            for metric, value in a['errors'].items():
                difference = (value-b['errors'][metric]).numpy()
                means = np.stack([difference[a['ancestors'] == group].mean(0) for group in groups])
                boot = means[draw].mean(1)
                for index in [0, 3, 9, 19]:
                    lo, hi = np.quantile(boot[:, index], [.025, .975])
                    rows.append(dict(left=left, right=right, seed=seed, metric=metric, horizon_fs=(index+1)*5,
                                     difference=float(means[:, index].mean()), lower95=lo, upper95=hi))
    if rows:
        write_metric_rows(rows, root(c)/'analyses/comparison', family=FAMILY, name='measurement_paired')
