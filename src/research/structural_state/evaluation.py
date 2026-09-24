"""Frozen physical probes, native-space neighbors and sustained-onset likelihood."""
import copy
import json

import numpy as np
import torch
from torch import nn

from src.research.bcr_followup.readouts import fit_residual_probe, predict_probe
from src.research.local_predictability.metrics import source_weights, hazard_loss, cumulative_risk, weighted_scores, threshold_at_fpr
from .common import sha, write_json, save_checkpoint, remaining
from .data import Corpus, BLOCKS
from .dynamics import PROTOCOL, targets as dynamics_targets
from src.research.liquid_geometry.metrics import participation

HORIZONS = np.array([.75, 3., 6., 9., 12.])


def native_head_scores(study, name, corpus, features):
    """Actual saved training heads versus their calibrated initial and mean controls."""
    ids = corpus.split['development']
    domain = study.arm(name)['input']
    source = np.array([r['source'] for r in corpus.records])[ids]
    temperature = np.array([r['temperature_K'] for r in corpus.records])[ids]
    phase = corpus.targets['phase'][ids]
    result = {}
    for checkpoint, representation in [('initial', 'initial_exported'), ('last', 'exported')]:
        saved = torch.load(study.technical/'fits'/name/(checkpoint+'.pt'), weights_only=False, map_location='cpu')
        scaler = saved['target_scalers'][domain]
        target = (corpus.geometry[domain][ids]-scaler['mean'])/scaler['scale']
        weight = saved['model'][f'heads.{domain}.weight'].numpy()
        bias = saved['model'][f'heads.{domain}.bias'].numpy()
        prediction = features[representation][ids] @ weight.T + bias
        values = {k: regression_score(target[:, sl], prediction[:, sl], source, temperature, phase)
                  for k, sl in BLOCKS.items()}
        values['block_mean'] = regression_score(
            np.concatenate([target[:, sl]/np.sqrt(3*(sl.stop-sl.start)) for sl in BLOCKS.values()], 1)*np.sqrt(89),
            np.concatenate([prediction[:, sl]/np.sqrt(3*(sl.stop-sl.start)) for sl in BLOCKS.values()], 1)*np.sqrt(89),
            source, temperature, phase)
        result[checkpoint] = values
    result['constant'] = {k: regression_score(target[:, sl], np.zeros_like(target[:, sl]), source, temperature, phase)
                          for k, sl in BLOCKS.items()}
    return dict(domain=domain, scores=result)


def standardized(values, fit):
    x = values[fit].astype(np.float64)
    mean, scale = x.mean(0), x.std(0).clip(1e-6)
    return ((values-mean)/scale).astype(np.float32), mean, scale


def pad(values, width=128):
    if values.shape[1] > width:
        raise ValueError('Control exceeds the fixed readout input dimension')
    return np.pad(values, ((0, 0), (0, width-values.shape[1])))


def controls(corpus, domain):
    fit = corpus.split['fit']
    values, _, _ = standardized(corpus.geometry[domain], fit)
    # Equal target-block contribution also defines the declared teacher metric.
    for sl in BLOCKS.values():
        values[:, sl] /= np.sqrt(3*(sl.stop-sl.start))
    _, singular, vectors = np.linalg.svd(values[fit], full_matrices=False)
    coordinates = values @ vectors[:64].T
    return {'descriptor': pad(values), 'descriptor_pca64': pad(coordinates),
            'conditions': np.zeros((len(values), 128), np.float32)}, dict(
                pca_variance_retained=float(np.square(singular[:64]).sum()/np.square(singular).sum()),
                components=vectors[:64])


def target_families(corpus, config):
    result = {}
    for domain in ('observed', 'relaxed'):
        result[domain + '_radial'] = corpus.targets[domain + '_radial']
        result[domain + '_angular'] = corpus.targets[domain + '_angular']
        result[domain + '_l6'] = corpus.targets[domain + '_rich'][:, 108:144]
    result['current_order'] = corpus.targets['current_order']
    result.update({f'future_order_{lag:g}': corpus.targets[f'future_order_{lag:g}'] for lag in config['future_ps']})
    if config.get('protocol') == PROTOCOL:
        auxiliary, _, _ = dynamics_targets(corpus, config)
        result['future_increment_9'] = auxiliary['future_residual']
    return result


def auxiliary_head_scores(study, name, corpus, features):
    auxiliary, receipt, _ = dynamics_targets(corpus, study.config)
    ids=corpus.split['development']
    source=np.array([r['source'] for r in corpus.records])[ids]
    temperature=np.array([r['temperature_K'] for r in corpus.records])[ids]
    phase=corpus.targets['phase'][ids]
    results={}
    for checkpoint, representation in [('initial','initial_exported'),('last','exported')]:
        saved=torch.load(study.technical/'fits'/name/(checkpoint+'.pt'),weights_only=False,map_location='cpu')
        if saved['auxiliary_targets'] != receipt:
            raise ValueError('Future target baseline/normalization differs from training')
        results[checkpoint]={}
        for target in ('current_order','future_residual'):
            weight=saved['model'][f'heads.{target}.weight'].numpy()
            bias=saved['model'][f'heads.{target}.bias'].numpy()
            predicted=features[representation][ids]@weight.T+bias
            results[checkpoint][target]=regression_score(auxiliary[target][ids],predicted,source,temperature,phase)
    results['constant']={target:regression_score(auxiliary[target][ids],np.zeros_like(auxiliary[target][ids]),source,temperature,phase)
                         for target in ('current_order','future_residual')}
    return results


def regression_score(actual, predicted, source, temperature, phase):
    error = np.square(actual-predicted).mean(-1)
    masks = {'all': np.ones(len(error), bool), 'PTM_other': phase == 0, 'PTM_crystalline': phase == 1}
    masks.update({f'T{t:g}': temperature == t for t in np.unique(temperature)})
    groups = {}
    for name, mask in masks.items():
        groups[name] = dict(rows=int(mask.sum()), sources=len(np.unique(source[mask])),
                           mse=float(source_weights(source[mask]) @ error[mask]) if mask.any() else None)
    return dict(groups=groups, per_source={str(s): float(error[source == s].mean()) for s in np.unique(source)})


def nearest_neighbors(features, fit, development, temperature, phase, k=5):
    """Original embedding Euclidean distance; never per-channel whitening of z."""
    result = np.empty((len(development), k), np.int64)
    for j, index in enumerate(development):
        candidates = fit[(temperature[fit] == temperature[index]) & (phase[fit] == phase[index])]
        if len(candidates) < k:
            raise ValueError(f'Insufficient matching training neighbors for development row {index}')
        distance = np.square(features[candidates].astype(float)-features[index]).sum(-1)
        result[j] = candidates[np.argsort(distance, kind='stable')[:k]]
    return result


def hazard_probe(features, conditions, corpus, config, kind, device, deadline):
    selected = {k: v[corpus.targets['at_risk'][v]] for k, v in corpus.split.items()}
    fit, tune, development = (selected[k] for k in ('fit', 'tune', 'development'))
    if any(not len(v) for v in selected.values()):
        raise ValueError('Empty source-held-out at-risk onset population')
    if any(np.any(corpus.targets['event_bin'][v] < 0) for v in selected.values()):
        raise ValueError('Onset readout contains an ineligible event label')
    source = np.array([r['source'] for r in corpus.records])
    features, mean, scale = standardized(features, fit)
    x = torch.as_tensor(np.c_[features, conditions], device=device)
    target = torch.as_tensor(corpus.targets['event_bin'], device=device)
    probe_seed=config.get('probe_seed',config['seed'])
    torch.manual_seed(probe_seed)
    pc = config['probes']
    model = (nn.Linear(x.shape[1], 5) if kind == 'linear' else nn.Sequential(
        nn.Linear(x.shape[1], pc['width']), nn.SiLU(), nn.Linear(pc['width'], 5))).to(device)
    final = model if kind == 'linear' else model[-1]
    w = source_weights(source[fit])
    event = corpus.targets['event_bin'][fit]
    frequency = np.array([(w@(event == k)+1e-6)/(w@(event >= k)+2e-6) for k in range(5)]).clip(1e-5, 1-1e-5)
    with torch.no_grad():
        final.bias.copy_(torch.as_tensor(np.log(frequency/(1-frequency)), dtype=torch.float32, device=device))
        # Step zero must be a true constant-risk control: tiny random slopes
        # otherwise produce arbitrary AP/alarms despite identical prior NLL.
        final.weight.zero_()
    optimizer = torch.optim.AdamW(model.parameters(), lr=pc['learning_rate'], weight_decay=pc['weight_decay'])
    rng = np.random.default_rng(probe_seed+15)
    tuning_weight = torch.as_tensor(source_weights(source[tune]), device=device)
    with torch.no_grad():
        best = float(tuning_weight @ hazard_loss(model(x[tune]), target[tune]).double())
    best_state, best_step = copy.deepcopy(model.state_dict()), 0
    for step in range(1, pc['updates']+1):
        if step % pc['evaluate_every'] == 0:
            remaining(deadline)
        ix = rng.choice(fit, size=pc['batch_size'], p=w)
        value = hazard_loss(model(x[ix]), target[ix]).mean()
        if not torch.isfinite(value):
            raise FloatingPointError(f'Nonfinite frozen hazard probe: {kind}, update={step}')
        optimizer.zero_grad(set_to_none=True)
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if step % pc['evaluate_every'] == 0 or step == pc['updates']:
            with torch.no_grad():
                score = float(tuning_weight @ hazard_loss(model(x[tune]), target[tune]).double())
            if score < best:
                best, best_step, best_state = score, step, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    with torch.no_grad():
        logits = model(x[development]).cpu()
        tune_logits = model(x[tune]).cpu()
    risks = cumulative_risk(logits).numpy().astype(float)
    tune_risks = cumulative_risk(tune_logits).numpy().astype(float)
    y = corpus.targets['event_bin'][development]
    nll = hazard_loss(logits, torch.as_tensor(y)).numpy()
    sources = source[development]
    metrics = dict(rows=len(y), events_12ps=int((y < 5).sum()),
        nll=float(source_weights(sources)@nll), best_step=best_step, tuning_nll=best,
        per_source={str(s): float(nll[sources == s].mean()) for s in np.unique(sources)}, horizons={})
    for k, horizon in enumerate(HORIZONS):
        threshold = threshold_at_fpr(corpus.targets['event_bin'][tune] <= k, tune_risks[:, k], source[tune])
        actual = y <= k
        row = weighted_scores(actual, risks[:, k], sources, threshold)
        probability = np.diff(np.c_[np.zeros(len(y)), risks[:, :k+1]], axis=1)
        midpoint = (np.r_[0, HORIZONS[:k]] + HORIZONS[:k+1])/2
        estimate = probability @ midpoint / risks[:, k].clip(1e-12)
        hit = actual & (risks[:, k] >= threshold)
        error = np.abs(estimate-corpus.targets['delay_ps'][development])
        row.update(events=int(actual.sum()), missed=int((actual & ~hit).sum()),
            detected_timing_mae_ps=float(error[hit].mean()) if hit.any() else None,
            timed_within_3ps_recall=float((hit & (error <= 3)).sum()/actual.sum()) if actual.any() else None)
        wdev = source_weights(sources)
        row['calibration'] = []
        for lo, hi in zip(np.linspace(0, 1, 6)[:-1], np.linspace(0, 1, 6)[1:], strict=True):
            mask = (risks[:, k] >= lo) & (risks[:, k] < hi if hi < 1 else risks[:, k] <= hi)
            if mask.any():
                ww = wdev[mask]/wdev[mask].sum()
                row['calibration'].append(dict(lower=float(lo), upper=float(hi), rows=int(mask.sum()),
                    predicted=float(ww@risks[mask, k]), observed=float(ww@actual[mask])))
        metrics['horizons'][str(horizon)] = row
    fitted = dict(model={k: v.cpu() for k, v in best_state.items()}, mean=mean, scale=scale, selected_step=best_step)
    predictions = dict(indices=development, logits=logits.numpy(), event=y, risks=risks, source=sources,
                       tuning_indices=tune, tuning_logits=tune_logits.numpy())
    return metrics, fitted, predictions


def run(study, name, device='cuda', deadline=None):
    folder = study.technical / 'evaluation' / name
    folder.mkdir(parents=True, exist_ok=True)
    completion = folder / 'complete.json'
    if completion.exists():
        if json.loads(completion.read_text())['identity'] != study.identity:
            raise ValueError('Evaluation identity changed')
        return
    corpus = Corpus(study)
    arm = study.arm(name)
    fit, tune, development = (corpus.split[k] for k in ('fit', 'tune', 'development'))
    fit_receipt = json.loads((study.technical / 'fits' / name / 'complete.json').read_text())
    feature_path = study.technical / 'fits' / name / 'features.npz'
    if fit_receipt['identity'] != study.identity or sha(feature_path) != fit_receipt['feature_sha256']:
        raise ValueError('Completed encoder export identity changed')
    with np.load(feature_path) as values:
        features = {k: pad(values[k]) for k in values.files}
    write_json(folder / 'native-heads.json', native_head_scores(study, name, corpus, features))
    if study.config['protocol'] == PROTOCOL:
        write_json(folder/'auxiliary-heads.json',auxiliary_head_scores(study,name,corpus,features))
    if name in ('A-observed', 'B-relaxed'):
        extra, pca = controls(corpus, arm['input'])
        features.update(extra)
        np.savez(folder / 'pca.npz', **pca)
    source = np.array([r['source'] for r in corpus.records])
    temperature = np.array([r['temperature_K'] for r in corpus.records])
    phase = corpus.targets['phase']
    conditions = (temperature[:, None] == np.unique(temperature[fit])[None]).astype(np.float32)
    targets = target_families(corpus, study.config)
    for representation, z in features.items():
        remaining(deadline)
        # All frozen readouts have 128 feature slots plus the same temperature
        # conditions. Training reconstruction heads receive z only.
        x = np.c_[z, conditions].astype(np.float32)
        if study.config['protocol'] == PROTOCOL:
            diagnostic={}
            for population, mask in [('all',np.ones(len(development),bool)),('noncrystalline',phase[development]==0)]:
                rows=development[mask]
                diagnostic[population]=dict(n=len(rows),rank=participation(z[rows]) if len(rows)>1 else None,
                    per_source={str(s):participation(z[rows[source[rows]==s]])
                                for s in np.unique(source[rows]) if np.sum(source[rows]==s)>1})
            write_json(folder/representation/'embedding-geometry.json',diagnostic)
        for family, y in targets.items():
            destination = folder / representation / family
            if (destination / 'metrics.json').exists():
                continue
            remaining(deadline)
            fitted = fit_residual_probe(x,y,fit,tune,study.config['probes'],study.config.get('probe_seed',study.config['seed']),device,deadline)
            predictions = predict_probe(fitted, x[development], device)
            actual = (y[development]-fitted['target_mean'])/fitted['target_scale']
            scores = {kind: regression_score(actual, prediction, source[development], temperature[development], phase[development])
                      for kind, prediction in zip(('ridge', 'residual'), predictions, strict=True)}
            destination.mkdir(parents=True, exist_ok=True)
            np.savez(destination / 'predictions.npz', indices=development, target=actual,
                     ridge=predictions[0], residual=predictions[1], source=source[development])
            save_checkpoint(destination / 'probe.pt', {k: v for k, v in fitted.items() if k != 'model'})
            write_json(destination / 'metrics.json', scores)
        neighbor_path = folder / representation / 'neighbors.json'
        if not neighbor_path.exists():
            neighbors = nearest_neighbors(z, fit, development, temperature, phase)
            scores = {}
            families=['observed_angular','relaxed_angular','observed_l6','relaxed_l6','future_order_9']
            if study.config['protocol'] == PROTOCOL:families+=['future_order_12','future_increment_9']
            for family in families:
                y, _, _ = standardized(targets[family], fit)
                # Mean pairwise discrepancy, not discrepancy to the neighbor mean.
                error = np.square(y[development, None]-y[neighbors]).mean((1, 2))
                scores[family] = dict(mse=float(source_weights(source[development])@error),
                    per_source={str(s): float(error[source[development] == s].mean()) for s in np.unique(source[development])})
                if study.config['protocol'] == PROTOCOL:
                    liquid = phase[development] == 0
                    scores[family]['noncrystalline'] = dict(
                        mse=float(source_weights(source[development][liquid]) @ error[liquid]),
                        rows=int(liquid.sum()),
                        per_source={str(s): float(error[liquid & (source[development] == s)].mean())
                                    for s in np.unique(source[development][liquid])})
            np.save(folder / representation / 'neighbors.npy', neighbors)
            write_json(neighbor_path, scores)
        # Onset probes use final/initial exported states and physical controls.
        if representation not in ('pooled', 'initial_pooled'):
            for kind in ('linear', 'mlp'):
                destination = folder / representation / ('hazard_' + kind)
                if (destination / 'metrics.json').exists():
                    continue
                scores, fitted, predictions = hazard_probe(z, conditions, corpus, study.config, kind, device, deadline)
                destination.mkdir(parents=True, exist_ok=True)
                np.savez(destination / 'predictions.npz', **predictions)
                save_checkpoint(destination / 'probe.pt', fitted)
                write_json(destination / 'metrics.json', scores)
        print(f'Evaluated {name}/{representation}', flush=True)
    persistence = {}
    for lag in study.config['future_ps']:
        family = f'future_order_{lag:g}'
        y, mean, scale = standardized(targets[family], fit)
        prediction = (corpus.targets['current_order'][development]-mean)/scale
        persistence[family] = regression_score(y[development], prediction, source[development], temperature[development], phase[development])
    write_json(folder / 'persistence.json', persistence)
    write_json(completion, dict(state='complete', identity=study.identity, feature_sha256=sha(feature_path),
        representations=list(features), targets=list(targets), source_split='25 fit, 5 tuning, 15 development roots',
        uncertainty='One seed; all cohorts are development/reused, not a new final test',
        threshold='5% false-alarm threshold fitted on tuning roots; no separate calibration split'))
