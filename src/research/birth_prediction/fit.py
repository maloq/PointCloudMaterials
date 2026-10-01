"""Likelihood-selected frozen readouts, paired endpoint assays and source uncertainty."""
import json
from pathlib import Path
import warnings

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .data import ROLES, load, read
from .features import history_packet, load_bank


def scores(y, p, w):
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    w = w / w.sum()
    result = dict(nll=float(w @ (-y * np.log(p) - (1 - y) * np.log1p(-p))),
                  brier=float(w @ (p - y) ** 2), prevalence=float(w @ y))
    if len(np.unique(y)) == 2:
        result.update(auroc=float(roc_auc_score(y, p, sample_weight=w)),
                      ap=float(average_precision_score(y, p, sample_weight=w)))
    else:
        result.update(auroc=None, ap=None)
    return result


def calibrate(y, p, w):
    if len(np.unique(y)) != 2:
        raise ValueError('Calibration requires held-out cases and liquid controls')
    x = logit(np.clip(p, 1e-6, 1 - 1e-6))
    w = w / w.sum()
    def loss(theta):
        value = theta[0] * x + theta[1]
        return w @ (np.logaddexp(0, value) - y * value) + 1e-4 * ((theta[0] - 1) ** 2 + theta[1] ** 2)
    result = minimize(loss, np.array([1., 0.]), method='L-BFGS-B', bounds=[(.01, 20.), (-20., 20.)])
    if not result.success:
        raise RuntimeError(f'Calibration failed: {result.message}')
    return result.x


def fit(c, arm_name, remove, *, split=None, destination=None,
        family='birth_prediction', evaluation=None, observation=None):
    arm = next(a for a in c['arms'] if a['name'] == arm_name)
    root = Path(destination) if destination is not None else resolve_path(c['output']) / 'analyses/classification-v1' / arm_name / f'minus-{remove}'
    tech = root / 'technical'
    tech.mkdir(parents=True, exist_ok=True)
    _, rows, manifest = load(c)
    binding = dict(dataset_identity=manifest['identity'], config=c, arm=arm, remove=remove,
                   fit_sha256=sha(Path(__file__)), feature_sha256=sha(Path(__file__).with_name('features.py')),
                   evaluation=evaluation, metric_family=family)
    if observation is not None:
        from .temporal_inputs import context, packet
        binding.update(observation=observation,
                       temporal_input_sha256=sha(Path(__file__).with_name('temporal_inputs.py')))
    if split is None:
        split = {r: np.flatnonzero(rows['role'] == r) for r in ROLES}
    joined = np.concatenate(list(split.values()))
    if len(np.unique(joined)) != len(joined):
        raise ValueError('Readout fitting, selection, calibration and evaluation rows overlap')
    source_sets = [set(rows['source'][ids]) for ids in split.values()]
    if any(a & b for i, a in enumerate(source_sets) for b in source_sets[i + 1:]):
        raise ValueError('A source crosses readout fitting/selection/calibration/evaluation roles')
    binding['partition_sha256'] = digest({k: rows['id'][v].tolist() for k, v in split.items()})
    identity = digest(binding)
    if (tech / 'complete.json').exists():
        record = read(tech / 'complete.json')
        if record['identity'] != identity or sha(tech / 'predictions.npz') != record['predictions_sha256']:
            raise ValueError('Completed birth fit identity changed')
        return
    y, weights = rows['label'], rows['weight']
    train, val = split['train'], split['selection']
    write_json(tech / 'binding.json', binding)
    encoder = next((m for m in c['encoders'] if m['name'] == arm['bank']), None)
    if encoder is not None:
        encoder = dict(checkpoint=encoder, input=dict(
            coordinates='centered original nearest-80 atom coordinates; radius 8 A',
            computational_context='same nearest-80 patch; no extra halo or surrounding patches',
            history_frames=1, velocity=False, conditions=[], species='one constant atom channel',
            material_scale='fixed preprocessing normalization; Al factor 1; no scale tensor',
            relaxation=False, training_only_teachers='recorded frozen pretraining; none during this readout fit'))
    prediction_context = dict(
        encoder=encoder,
        predictor=dict(model=arm['model'], bank=arm['bank'], families=arm.get('families', []),
            spatial_support='none' if arm['model'] == 'prior' else 'nearest 80 atoms, radius 8 A; every patch shared across models',
            history_frames=0 if arm['model'] == 'prior' else c['history_frames'] - remove,
            cadence_ps=c['cadence_ps'],
            history_span_ps=0 if arm['model'] == 'prior' else (c['history_frames'] - remove - 1) * c['cadence_ps'],
            endpoint_rule='same start; remove newest frames', conditions=[], velocity=False,
            relaxation=False, temporal_features='none' if arm['model'] == 'prior' else 'chronological states, mean, std, last minus first'),
        labels_as_inputs=False, future_centroid_as_input=False, selection='selection-source binary NLL',
        calibration='separate calibration sources; event-enriched classification probabilities',
        tracking='local frozen diagnostic readout / descriptor control', seed=c['seed'], evaluation=evaluation)
    if observation is not None:
        predictor = prediction_context['predictor']
        predictor.update(context(observation, c['cadence_ps']))
        predictor.update(history_frames=len(observation['frames']),
            history_span_ps=predictor['observed_span_ps'],
            endpoint_rule='explicit observed-frame subset from the frozen eight-frame history',
            temporal_features=observation['kind'])
    if 'input_domain' in manifest:
        domain = manifest['input_domain']
        if domain != c['input_domain'] or domain['relaxation'] is not True:
            raise ValueError('Derived input producer and declared relaxation contract differ')
        prediction_context['input_domain'] = domain
        if encoder is not None:
            encoder['input'].update(coordinates=domain['coordinates'],
                computational_context=domain['computational_context'], relaxation=True)
        if arm['model'] != 'prior':
            prediction_context['predictor'].update(relaxation=True,
                spatial_support='same original nearest-80 atom identities after full periodic-cell quench',
                computational_context=domain['computational_context'])
    write_json(tech / 'prediction-context.json', prediction_context)
    history = []
    if arm['model'] == 'prior':
        prior = float(np.average(y[train], weights=weights[train]))
        probability = np.full(len(y), prior)
        selection = dict(training_prevalence=prior)
    else:
        bank, columns = load_bank(c, arm['bank'])
        selected = np.array([i for i, col in enumerate(columns) if
            arm['bank'] != 'descriptors' or col.split('/')[0] in arm['families']], int)
        if not len(selected):
            raise ValueError(f'No input features for {arm_name}')
        x = (history_packet(bank, rows, remove, selected) if observation is None else
             packet(bank, rows, selected, observation, c['seed']))
        del bank
        if arm['model'] == 'linear':
            w = weights[train] / weights[train].sum()
            mean = w @ x[train]
            scale = np.sqrt(w @ (x[train] - mean) ** 2).clip(1e-4)
            xx = (x - mean) / scale
            best = np.inf
            with threadpool_limits(limits=c['fit_threads']), warnings.catch_warnings():
                warnings.simplefilter('error', ConvergenceWarning)
                for strength in c['linear_C']:
                    candidate = LogisticRegression(C=strength, solver='lbfgs', max_iter=3000,
                                                   tol=1e-5, random_state=c['seed'])
                    candidate.fit(xx[train], y[train], sample_weight=weights[train] / weights[train].mean())
                    value = scores(y[val], candidate.predict_proba(xx[val])[:, 1], weights[val])['nll']
                    history.append(dict(C=strength, selection_nll=value))
                    if value < best:
                        best, model = value, candidate
                        selection = dict(C=strength, nll=value)
                probability = model.predict_proba(xx)[:, 1]
            joblib.dump(dict(model=model, mean=mean, scale=scale, selected_columns=selected), tech / 'model.joblib')
        elif arm['model'] == 'catboost':
            from catboost import CatBoostClassifier, Pool
            model = CatBoostClassifier(**c['catboost'], loss_function='Logloss', eval_metric='Logloss',
                random_seed=c['seed'], thread_count=c['fit_threads'], train_dir=str(tech / 'catboost'))
            def pool(ids):
                return Pool(x[ids], y[ids], weight=weights[ids] / weights[ids].mean())
            model.fit(pool(train), eval_set=pool(val), early_stopping_rounds=c['patience'],
                      use_best_model=True, verbose=50, save_snapshot=True,
                      snapshot_file=str(tech / 'snapshot.cbs'), snapshot_interval=60)
            probability = model.predict_proba(x)[:, 1]
            model.save_model(str(tech / 'model.cbm'))
            selection = dict(iterations=int(model.get_best_iteration()) + 1,
                             nll=scores(y[val], probability[val], weights[val])['nll'])
            history = model.get_evals_result()
            write_json(tech / 'catboost-parameters.json', model.get_all_params())
        else:
            raise ValueError(arm['model'])
        del x
    cal = split['calibration']
    theta = calibrate(y[cal], probability[cal], weights[cal])
    calibrated = expit(theta[0] * logit(np.clip(probability, 1e-6, 1 - 1e-6)) + theta[1])
    # Lock a false-alarm threshold using calibration liquid controls only.
    negatives = cal[y[cal] == 0]
    order = negatives[np.argsort(calibrated[negatives])]
    cumulative = np.cumsum(weights[order]) / weights[order].sum()
    threshold = float(calibrated[order[min(np.searchsorted(cumulative, .95), len(order) - 1)]])
    included = np.sort(joined)
    roles = np.full(len(y), '', dtype='U32')
    for role, ids in split.items():
        roles[ids] = role
    np.savez_compressed(tech / 'predictions.npz', probability=probability[included],
                        calibrated=calibrated[included], role=roles[included],
                        original_role=rows['role'][included],
                        **{k: rows[k][included] for k in ('id', 'source', 'event', 'pair', 'label', 'weight')})
    records = []
    for role, ids in split.items():
        for kind, pred in (('raw', probability), ('calibrated', calibrated)):
            record = dict(arm=arm_name, remove_frames=remove, role=role, score=kind,
                          rows=len(ids), **scores(y[ids], pred[ids], weights[ids]))
            if kind == 'calibrated':
                pos, neg = ids[y[ids] == 1], ids[y[ids] == 0]
                record.update(recall_at_locked_fpr05=float(np.average(pred[pos] > threshold, weights=weights[pos])),
                              observed_fpr=float(np.average(pred[neg] > threshold, weights=weights[neg])))
            else:
                record.update(recall_at_locked_fpr05=None, observed_fpr=None)
            records.append(record)
    write_metric_rows(records, root, family=family, name='scores')
    write_json(tech / 'selection.json', dict(selector='selection binary NLL', selected=selection,
               calibration_parameters=theta.tolist(), threshold_fpr05=threshold, history=history))
    write_json(tech / 'complete.json', dict(identity=identity,
               predictions_sha256=sha(tech / 'predictions.npz'), scores=records))
    print(json.dumps(records[-2:]), flush=True)


def collect(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _, rows, manifest = load(c)
    root = resolve_path(c['output']) / 'analyses/comparison-v1'
    root.mkdir(parents=True, exist_ok=True)
    summary, paired, predictions = [], [], {}
    test = np.flatnonzero(rows['role'] == 'test')
    source_ids = np.unique(rows['source'][test])
    rng = np.random.default_rng(c['seed'])
    draw_counts = np.stack([np.bincount(rng.integers(len(source_ids), size=len(source_ids)),
                           minlength=len(source_ids)) for _ in range(c['bootstrap_draws'])])
    source_slot = np.searchsorted(source_ids, rows['source'][test])
    w, y = rows['weight'][test], rows['label'][test]
    for remove in c['remove_frames']:
        for arm in c['arms']:
            tech = resolve_path(c['output']) / 'analyses/classification-v1' / arm['name'] / f'minus-{remove}' / 'technical'
            complete = read(tech / 'complete.json')
            if sha(tech / 'predictions.npz') != complete['predictions_sha256']:
                raise ValueError('Changed retained birth predictions')
            with np.load(tech / 'predictions.npz') as a:
                if not np.array_equal(a['id'], rows['id']):
                    raise ValueError('Models evaluated different birth histories')
                p = a['calibrated'][test].copy()
                raw = a['probability'][test].copy()
            predictions[(arm['name'], remove)] = p
            for score_kind, prob in (('raw', raw), ('calibrated', p)):
                value = scores(y, prob, w)
                interval = {}
                boot = []
                for draw in draw_counts:
                    ww = w * draw[source_slot]
                    if len(np.unique(y[ww > 0])) == 2:
                        boot.append(scores(y, prob, ww))
                for metric in ('nll', 'brier', 'auroc', 'ap'):
                    lo, hi = np.quantile([v[metric] for v in boot], [.025, .975])
                    interval[metric + '_lo'], interval[metric + '_hi'] = float(lo), float(hi)
                summary.append(dict(arm=arm['name'], remove_frames=remove, score=score_kind,
                    history_frames=c['history_frames'] - remove,
                    appearance_lead_ps=(remove + 1) * c['cadence_ps'], bootstrap_valid_draws=len(boot), **value, **interval))
    for remove in c['remove_frames']:
        reference = predictions[('rich_gbdt', remove)]
        ref_loss = -y * np.log(np.clip(reference, 1e-7, 1)) - (1-y) * np.log(np.clip(1-reference, 1e-7, 1))
        for arm in c['arms']:
            prob = predictions[(arm['name'], remove)]
            loss = -y * np.log(np.clip(prob, 1e-7, 1)) - (1-y) * np.log(np.clip(1-prob, 1e-7, 1))
            delta = loss - ref_loss
            values = [float(np.average(delta, weights=w * draw[source_slot])) for draw in draw_counts]
            lo, hi = np.quantile(values, [.025, .975])
            paired.append(dict(arm=arm['name'], reference='rich_gbdt', remove_frames=remove,
                              calibrated_nll_difference=float(np.average(delta, weights=w)),
                              difference_lo=float(lo), difference_hi=float(hi)))
    write_metric_rows(summary, root, family='birth_prediction', name='test-comparison')
    write_metric_rows(paired, root, family='birth_prediction', name='paired-vs-rich')
    endpoints = []
    for arm in c['arms']:
        base = np.clip(predictions[(arm['name'], 0)], 1e-7, 1 - 1e-7)
        base_loss = -y * np.log(base) - (1-y) * np.log1p(-base)
        for remove in c['remove_frames'][1:]:
            prob = np.clip(predictions[(arm['name'], remove)], 1e-7, 1 - 1e-7)
            delta = -y * np.log(prob) - (1-y) * np.log1p(-prob) - base_loss
            values = [float(np.average(delta, weights=w * draw[source_slot])) for draw in draw_counts]
            lo, hi = np.quantile(values, [.025, .975])
            endpoints.append(dict(arm=arm['name'], remove_frames=remove, reference_removed_frames=0,
                calibrated_nll_difference=float(np.average(delta, weights=w)), difference_lo=float(lo), difference_hi=float(hi)))
    write_metric_rows(endpoints, root, family='birth_prediction', name='paired-endpoint-changes')
    # Report case/control strata by physical time to establishment, without
    # relabeling later births or presenting case-control AP as natural-risk AP.
    horizon = []
    for remove in c['remove_frames']:
        lag = (rows['birth_frame'] - rows['end_frame'] + remove) * c['cadence_ps']
        for tau in (3., 6.):
            subset = lag[test] <= tau
            for arm in c['arms']:
                value = scores(y[subset], predictions[(arm['name'], remove)][subset], w[subset]) if subset.any() else dict(nll=None, brier=None, prevalence=None, auroc=None, ap=None)
                horizon.append(dict(arm=arm['name'], remove_frames=remove, establishment_within_ps=tau,
                                    rows=int(subset.sum()), **value))
    write_metric_rows(horizon, root, family='birth_prediction', name='establishment-lead-strata')
    plots = root / 'plots'
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm in c['arms']:
        values = [r for r in summary if r['arm'] == arm['name'] and r['score'] == 'calibrated']
        for ax, metric in zip(axes, ('nll', 'ap')):
            ax.plot([r['appearance_lead_ps'] for r in values], [r[metric] for r in values], marker='o', label=arm['name'])
            ax.set(xlabel='Time before observed crystal appearance (ps)', ylabel=metric)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(plots / 'history-endpoints.png', dpi=170)
    plt.close(fig)
    lines = ['# Birth prediction before observed crystal appearance', '',
             'Completed frozen-feature case/control classification. Entire input histories are PTM-clear in an 8 A sphere.',
             'Probabilities and AP refer to event-enriched, matched birth/liquid examples, not natural nucleation risk.', '',
             '| Predictor | Removed frames | Lead to appearance (ps) | Test NLL | Test AP |',
             '| --- | ---: | ---: | ---: | ---: |']
    for r in summary:
        if r['score'] == 'calibrated':
            lines.append(f'| {r["arm"]} | {r["remove_frames"]} | {r["appearance_lead_ps"]:.2f} | {r["nll"]:.4f} | {r["ap"]:.4f} |')
    lines += ['', 'Intervals resample whole test sources, conditional on one fitted seed; anchors within a birth are correlated.',
              'Appearance and establishment are separate times. Coverage exclusions and all source roles are retained.',
              '[Tables](tables/test-comparison.csv) · [Endpoint plot](plots/history-endpoints.png)']
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    write_json(root / 'technical/complete.json', dict(dataset_identity=manifest['identity'],
               fits=len(c['arms']) * len(c['remove_frames']), test_sources=source_ids.tolist()))
