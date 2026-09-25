"""Calibration/test isolation, frozen readouts, controls and auditable AP exports."""
import csv
import json
import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import average_precision_score

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.local_predictability.metrics import (
    source_weights, cumulative_risk, hazard_loss, weighted_scores, threshold_at_fpr)
from src.research.trajectory_stability.spectrum import spectrum, lag_pairs, tracks
from src.research.robust_onset.metrics import perturb_patch
from src.research.structural_state.data import graph_arrays
from src.models.encoders.graph_bank import GraphBank
from .common import write_json, save_checkpoint, sha
from .data import sampling_distribution
from .train import make_model, encode, selection_key, configure_runtime


def calibrate_risks(corpus, risks):
    """One increasing map shared across horizons: preserves AP and cumulative order."""
    ids = corpus.split['calibration']
    logits = torch.logit(torch.as_tensor(risks.astype(np.float64)).clamp(1e-15, 1-1e-15))
    target = torch.as_tensor(np.c_[corpus.pop['event'][ids] <= 1, corpus.pop['event'][ids] <= 2], dtype=torch.double)
    weights = torch.as_tensor(source_weights(corpus.pop['source'][ids]), dtype=torch.double)
    log_slope = torch.zeros((), dtype=torch.double, requires_grad=True)
    intercept = torch.zeros((), dtype=torch.double, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_slope, intercept], max_iter=64, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        prediction = logits[ids][:, [1, 2]] * log_slope.clamp(-5, 5).exp() + intercept
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(prediction, target, reduction='none').mean(1)*weights).sum()
        loss = loss + 1e-4*(log_slope.square()+intercept.square())
        loss.backward()
        return loss
    optimizer.step(closure)
    with torch.no_grad():
        slope = float(log_slope.clamp(-5, 5).exp())
        offset = float(intercept)
        result = (logits*slope+offset).sigmoid().numpy()
    if not np.isfinite(result).all():
        raise FloatingPointError('Nonfinite probability calibration')
    return result, dict(slope=slope, intercept=offset, fit_role='calibration', horizons_ps=[3, 6],
                        penalty=1e-4, shared_increasing_map=True)


def bootstrap_ap(actual, score, source, draws, seed):
    """Whole-source resampling; report excluded zero-event draws explicitly."""
    roots, inverse = np.unique(source, return_inverse=True)
    weight = source_weights(source)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        multiplicity = np.bincount(rng.integers(len(roots), size=len(roots)), minlength=len(roots))
        w = weight * multiplicity[inverse]
        if not (w @ actual):
            continue
        keep = w > 0
        values.append(average_precision_score(actual[keep], score[keep], sample_weight=w[keep]))
    if not values:
        raise ValueError('Every source-bootstrap draw lacks an event')
    return dict(ci95=np.quantile(values, [.025, .975]).tolist(), valid_draws=len(values),
                zero_event_draws=draws-len(values), draws=draws)


def score_predictions(corpus, risks, draws, seed, calibrated=None):
    pop = corpus.pop
    if risks.shape != (len(pop['event']), 5) or not np.isfinite(risks).all():
        raise ValueError('Require aligned finite five-horizon cumulative predictions')
    if np.any(np.diff(risks, axis=1) < -1e-7) or np.any((risks < 0) | (risks > 1)):
        raise ValueError('Cumulative onset risk must be monotone and in [0,1]')
    calibrated = risks if calibrated is None else calibrated
    result = {}
    for horizon, column in ((3, 1), (6, 2), (12, 4)):
        cal = corpus.split['calibration']
        threshold = threshold_at_fpr(pop['event'][cal] <= column, calibrated[cal, column], pop['source'][cal])
        block = {}
        for role in ('selection', 'calibration', 'test'):
            ids = corpus.split[role]
            block[role] = weighted_scores(pop['event'][ids] <= column, calibrated[ids, column], pop['source'][ids], threshold)
            raw = weighted_scores(pop['event'][ids] <= column, risks[ids, column], pop['source'][ids])
            # Primary ranking is always measured on raw scores: calibrator
            # saturation must never turn finite but distinct risks into ties.
            block[role].update(average_precision=raw['average_precision'], raw_brier=raw['brier'],
                               raw_log_loss=raw['log_loss'])
        ids = corpus.split['test']
        block['test']['bootstrap'] = bootstrap_ap(pop['event'][ids] <= column, risks[ids, column],
            pop['source'][ids], draws, seed)
        result[str(horizon)] = block
    return result


def readout(study, corpus, features, name, kind, device):
    """A fresh likelihood-trained readout; frozen encoder, NLL-only selection."""
    from .tracking import tracked_run
    is_encoder = name in {a['name'] for a in study.config['arms']}
    with tracked_run(study, name, job_type='encoder' if is_encoder else 'control') as tracking:
        axis = f'readout/{kind}/update'
        tracking.define_metric(axis, hidden=True)
        tracking.define_metric(f'readout/{kind}/*', step_metric=axis, summary='last')
        return _readout(study, corpus, features, name, kind, device, tracking)


def _readout(study, corpus, features, name, kind, device, tracking):
    root = study.technical / 'readouts' / name / kind
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'complete.json').exists():
        return np.load(root / 'risks.npy')
    torch.manual_seed(study.config['seed'])
    rng = np.random.default_rng(study.config['seed'])
    fit, selection = corpus.split['train'], corpus.split['selection']
    w = source_weights(corpus.pop['source'][fit])
    mean = w @ features[fit].astype(float)
    scale = np.sqrt(w @ (features[fit]-mean)**2).clip(1e-5)
    normalized = ((features-mean)/scale).astype(np.float32)
    x = torch.as_tensor(normalized, device=device)
    y = torch.as_tensor(corpus.pop['event'], device=device)
    model = nn.Linear(x.shape[1], 5) if kind == 'linear' else nn.Sequential(
        nn.Linear(x.shape[1], 128), nn.SiLU(), nn.Linear(128, 5))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=study.config['baselines']['learning_rate'], weight_decay=1e-4)
    q, iw = sampling_distribution(corpus.pop['source'][fit], corpus.pop['event'][fit], .5)
    selection_weights = torch.as_tensor(source_weights(corpus.pop['source'][selection]), dtype=torch.float32, device=device)
    best, state = None, None
    updates = study.config['baselines']['updates']
    epoch_stream=None
    if 'epochs' in study.config['baselines']:
        import math
        from src.research.encoder_context.epochs import batches,epoch_weights
        settings=study.config['baselines'];steps_per_epoch=math.ceil(len(fit)/settings['batch_size'])
        if updates!=steps_per_epoch*settings['epochs']:raise ValueError('Readout epoch budget differs')
        epoch_stream=iter(batches(np.arange(len(fit)),settings['batch_size'],settings['epochs'],study.config['seed']))
        full_weights=epoch_weights(corpus.pop['source'][fit])
    for update in range(updates):
        if epoch_stream is None:
            drawn=rng.choice(len(fit),size=study.config['baselines']['batch_size'],p=q);importance=iw[drawn]
        else:
            step,drawn=next(epoch_stream)
            if step!=update:raise ValueError('Readout epoch cursor differs')
            importance=full_weights[drawn]
        ids = fit[drawn]
        weight = torch.as_tensor(importance, dtype=torch.float32, device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = (hazard_loss(model(x[ids]), y[ids]) * weight).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        if (update+1) % 32 == 0 or update+1 == updates:
            tracking.log({f'readout/{kind}/update':update+1,
                          f'readout/{kind}/sampled_objective':float(loss.detach())})
        if (update+1) % 128 == 0 or update+1 == updates:
            with torch.no_grad():
                selection_logits = model(x[selection])
                nll = float(selection_weights @ hazard_loss(selection_logits, y[selection]))
                prediction = cumulative_risk(selection_logits).cpu().numpy()
            metrics = {f'AP{h}': weighted_scores(corpus.pop['event'][selection] <= k,
                prediction[:, k], corpus.pop['source'][selection])['average_precision'] for h, k in ((3, 1), (6, 2))}
            metrics['nll'] = nll
            tracking.log({f'readout/{kind}/update':update+1,
                          **{f'readout/{kind}/selection_{key}':value for key,value in metrics.items()}})
            minimum=study.config['baselines'].get('minimum_selection_epoch',0)*(steps_per_epoch if epoch_stream is not None else 1)
            if update+1>=minimum and (best is None or selection_key(metrics) < selection_key(best)):
                best = dict(metrics, update=update+1)
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(state)
    with torch.no_grad():
        risks = cumulative_risk(model(x)).cpu().numpy()
    np.save(root / 'risks.npy', risks)
    save_checkpoint(root / 'best.pt', dict(model=state, selection=best, mean=mean, scale=scale,
        encoder_frozen=True, input_name=name, kind=kind, identity=study.identity,
        input_dim=x.shape[1], external_inputs=[]))
    write_json(root / 'complete.json', dict(identity=study.identity, selection=best, risks_sha256=sha(root/'risks.npy')))
    tracking.summary[f'readout/{kind}/best'] = best
    return risks


def save_result(study, corpus, name, kind, risks, **metadata):
    root = study.technical / 'evaluation' / name / kind
    root.mkdir(parents=True, exist_ok=True)
    calibrated, calibration = calibrate_risks(corpus, risks)
    sample_ids={'sample_id':corpus.pop['sample_id']} if 'sample_id' in corpus.pop else {}
    np.savez(root / 'predictions.npz', risks=risks, calibrated_risks=calibrated, source=corpus.pop['source'],
        event=corpus.pop['event'], role=corpus.pop['role'], frame=corpus.pop['frame'], atom=corpus.pop['atom'],**sample_ids)
    scores = score_predictions(corpus, risks, study.config['bootstrap'], study.config['seed'], calibrated)
    # Reconstruct the first-event distribution, including survival beyond 12 ps.
    probabilities = np.diff(np.c_[np.zeros(len(risks)), risks.astype(np.float64), np.ones(len(risks))], axis=1)
    observed_probability = probabilities[np.arange(len(risks)), corpus.pop['event']]
    event_nll = {}
    for role in ('selection', 'calibration', 'test'):
        ids = corpus.split[role]
        event_nll[role] = float(source_weights(corpus.pop['source'][ids]) @ -np.log(observed_probability[ids].clip(1e-12)))
    write_json(root / 'metrics.json', dict(name=name, kind=kind, branch=study.config['branch'],
        identity=study.identity, scores=scores, calibration=calibration, event_nll=event_nll,
        prediction_context='technical/prediction-context.json', external_inputs=[], **metadata))
    from .tracking import evaluation_record
    evaluation_record(study, name, kind, scores, dict(metadata, calibration=calibration))
    return scores


def descriptor_baselines(study, corpus, device):
    for name, x in (
        ('constant', np.zeros((len(corpus.pop['source']), 0), np.float32)),
        ('observed-descriptors', corpus.descriptors['hot']),
        ('relaxed-descriptors', corpus.descriptors['cold']),
        ('paired-descriptors', np.c_[corpus.descriptors['hot'], corpus.descriptors['cold']])):
        risks = readout(study, corpus, x, name, 'mlp', device)
        save_result(study, corpus, name, 'mlp', risks, representation='fixed descriptors; supervised predictor only')
        print(json.dumps(dict(stage='descriptor_control_complete', name=name)), flush=True)


@torch.no_grad()
def noise_diagnostics(study, corpus, model, banks, features, device):
    # Identical predetermined rows/seeds in every arm; no test-based selection.
    test = corpus.split['test']
    ids = np.concatenate([np.flatnonzero((corpus.pop['source'] == s) & (corpus.pop['role'] == 'test'))[:4]
                          for s in np.unique(corpus.pop['source'][test])])
    trace = spectrum(features[corpus.split['train']], source_weights(corpus.pop['source'][corpus.split['train']]))['total_energy']
    if trace <= 0:
        raise ValueError('Collapsed fitted state: normalized noise response is undefined')
    domains = ('hot', 'cold') if model.input == 'paired' else (model.input,)
    result = []
    for domain in domains:
        a = corpus.arrays[domain]
        original = [a['positions'][a['offsets'][i]:a['offsets'][i+1]] for i in ids]
        for fraction in study.config['noise_rms_fractions']:
            rng = np.random.default_rng(study.config['seed'])
            perturbed = [perturb_patch(p, fraction, rng) for p in original]
            noisy_bank = GraphBank(graph_arrays([p for p, _ in perturbed], study.config['encoder']['cutoff']),
                model.encoder, device, node_capacity=80*study.config['training']['microbatch'])
            # Map the unchanged paired domain to the same compact row IDs.
            view_banks = {domain: noisy_bank}
            if model.input == 'paired':
                other = 'cold' if domain == 'hot' else 'hot'
                b = corpus.arrays[other]
                other_patches = [b['positions'][b['offsets'][i]:b['offsets'][i+1]] for i in ids]
                view_banks[other] = GraphBank(graph_arrays(other_patches, study.config['encoder']['cutoff']),
                    model.encoder, device, node_capacity=80*study.config['training']['microbatch'])
            z = encode(model, view_banks, np.arange(len(ids)), study.config['training']['microbatch']).cpu().numpy()
            energy = np.square(z-features[ids]).sum(1)
            weights = source_weights(corpus.pop['source'][ids])
            result.append(dict(domain=domain, requested_rms_fraction=fraction,
                realized_rms_fraction=float(np.sqrt(weights @ np.array([r['input_relative_mse'] for _, r in perturbed]))),
                input_rms_A=float(np.sqrt(weights @ np.array([r['input_mse_A2'] for _, r in perturbed]))),
                normalized_embedding_rms=float(np.sqrt(weights @ energy/(2*trace))), rows=len(ids),
                operation='Perturb the supplied view and rebuild edges; no re-quench after noise'))
    return result


def evaluate_arm(study, corpus, banks, name, device, deadline):
    root = study.technical / 'runs' / name
    arm = study.arm(name)
    all_ids = np.arange(len(corpus.pop['event']))
    model = make_model(study, corpus, arm, device)
    selector = 'best'
    if time.time() >= deadline-60:
        raise TimeoutError(f'Evaluation budget exhausted before {name}/{selector}')
    path = root / f'{selector}.pt'
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved['identity'] != study.identity:
        raise ValueError('Encoder checkpoint identity mismatch')
    model.load_state_dict(saved['model'])
    configure_runtime(study, model, banks, corpus)
    z = encode(model, banks, all_ids, study.config['training']['microbatch'])
    with torch.no_grad():
        risks = cumulative_risk(model.logits(z)).cpu().numpy()
    values = z.cpu().numpy()
    dest = study.technical / 'evaluation' / name / selector
    dest.mkdir(parents=True, exist_ok=True)
    np.save(dest / 'features.npy', values)
    save_result(study, corpus, name, selector, risks, input=arm['input'], updates=saved['update'],
        selected_by='minimum selection hazard NLL', checkpoint_sha256=sha(path))
    best_features = values
    fit, test = corpus.split['train'], corpus.split['test']
    pairs = lag_pairs(tracks(corpus.pop['source'][test], corpus.pop['atom'][test], corpus.pop['frame'][test]*.75),
                      corpus.pop['frame'][test]*.75, .75)
    if len(pairs):
        raise ValueError('Unexpected 0.75 ps pairs in the fixed 12 ps evaluation cadence; revise diagnostics explicitly')
    write_json(dest / 'diagnostics.json', dict(
        dataset_spectrum=spectrum(values, source_weights(corpus.pop['source'])),
        fitting_spectrum=spectrum(values[fit], source_weights(corpus.pop['source'][fit])),
        test_spectrum=spectrum(values[test], source_weights(corpus.pop['source'][test])),
        movement_lag_ps=.75, movement=None,
        movement_unavailable='Existing paired held-out cache uses 12 ps anchors; no 0.75 ps pairs. No interpolation or substitute lag.',
        noise=noise_diagnostics(study, corpus, model, banks, values, device)))
    for kind in ('linear', 'mlp'):
        if time.time() >= deadline-90:
            raise TimeoutError(f'Evaluation budget exhausted before frozen {name}/{kind}')
        risks = readout(study, corpus, best_features, name, kind, device)
        save_result(study, corpus, name, kind, risks, input=arm['input'],
            selected_by='encoder and independent frozen readout: minimum selection hazard NLL')
    write_json(root / 'evaluation-complete.json', dict(identity=study.identity))
    print(json.dumps(dict(stage='encoder_evaluation_complete', name=name)), flush=True)


def ensemble(study, corpus, finalists):
    if len(finalists) != 2:
        raise ValueError('Predetermined ensemble uses exactly two selection finalists')
    risks = []
    for name in finalists:
        path = study.technical / 'evaluation' / name / 'best/predictions.npz'
        with np.load(path) as a:
            np.testing.assert_array_equal(a['source'], corpus.pop['source'])
            risks.append(a['risks'])
    prediction = .5*risks[0] + .5*risks[1]
    save_result(study, corpus, 'uniform-ensemble', 'risk-mixture', prediction,
        representation='two-model predictive ensemble, not a single encoder', members=finalists,
        mixture_weights=[.5, .5], selected_by='NLL-selected members; fixed equal mixture, no fitted weights')


def collect(study):
    rows = []
    for path in sorted((study.technical / 'evaluation').glob('*/*/metrics.json')):
        r = json.loads(path.read_text())
        row = dict(branch=r['branch'], model=r['name'], readout=r['kind'], input=r.get('input', ''),
                   updates=r.get('updates'), identity=r['identity'],
                   selection_event_nll=r['event_nll']['selection'], test_event_nll=r['event_nll']['test'],
                   prediction_context=r['prediction_context'], external_inputs=json.dumps(r['external_inputs']))
        for h in ('3', '6', '12'):
            for role in ('selection', 'test'):
                row[f'{role}_AP{h}'] = r['scores'][h][role]['average_precision']
            test = r['scores'][h]['test']
            row.update({f'test_AP{h}_lower': test['bootstrap']['ci95'][0],
                        f'test_AP{h}_upper': test['bootstrap']['ci95'][1],
                        f'test_prevalence{h}': test['prevalence'],
                        f'test_raw_log_loss{h}': test['raw_log_loss'],
                        f'test_calibrated_log_loss{h}': test['log_loss'],
                        f'test_Brier{h}': test['brier'],
                        f'test_recall{h}_at_calibration_FPR5': test['recall'],
                        f'test_FPR{h}': test['false_positive_rate']})
        rows.append(row)
    if rows:
        snapshot_metric_docs(study.root, 'supervised_onset')
        with (study.root / 'tables/comparison.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(sorted(rows, key=lambda r: (r['model'], r['readout'])))
    write_json(study.technical / 'collection.json', dict(identity=study.identity, rows=len(rows)))
    return rows
