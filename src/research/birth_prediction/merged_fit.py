"""Training-source CV selection, full-train refit and an uncalibrated merged test."""
from pathlib import Path
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from .data import read
from .fit import scores

FAMILY = 'birth_merged_test'


def standardized(x, weights):
    w = weights / weights.sum()
    mean = w @ x
    scale = np.sqrt(w @ (x-mean)**2).clip(1e-4)
    return mean, scale


def fit(c, b, rows, x, spec, root, folds, context, *, family):
    """Only training indices reach model fitting, hyperparameter selection or thresholds."""
    root = Path(root)
    tech = root / 'technical'
    tech.mkdir(parents=True, exist_ok=True)
    train = np.flatnonzero(rows['role'] == 'train')
    test = np.flatnonzero(rows['role'] != 'train')
    y, w = rows['label'], rows['weight']
    binding = dict(config=c, treatment=spec, prediction_context=context, metric_family=family,
        train_ids=rows['id'][train].tolist(), test_ids=rows['id'][test].tolist(),
        training_folds=folds.tolist(), implementation_sha256=sha(Path(__file__)))
    identity = digest(binding)
    done = tech / 'complete.json'
    if done.exists():
        saved = read(done)
        if saved['identity'] != identity or sha(tech/'predictions.npz') != saved['predictions_sha256']:
            raise ValueError(f'Changed completed merged-test fit: {root}')
        return
    write_json(tech/'binding.json', binding)
    write_json(tech/'prediction-context.json', context)
    trace = []
    selected = {}
    cv = np.empty(len(train))
    model_kind = spec['model']
    with threadpool_limits(limits=c['fit_threads']), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        if model_kind == 'linear':
            candidates = b['linear_C']
            predictions = np.empty((len(candidates), len(train)))
            for fold in range(c['folds']):
                holdout = np.flatnonzero(folds == fold)
                fitting = train[folds != fold]
                mean, scale = standardized(x[fitting], w[fitting])
                xx = (x-mean)/scale
                for k, strength in enumerate(candidates):
                    model = LogisticRegression(C=strength, solver='lbfgs', max_iter=3000,
                        tol=1e-5, random_state=c['seed'])
                    model.fit(xx[fitting], y[fitting], sample_weight=w[fitting]/w[fitting].mean())
                    predictions[k, holdout] = model.predict_proba(xx[train[holdout]])[:, 1]
            errors = [scores(y[train], p, w[train])['nll'] for p in predictions]
            winner = int(np.argmin(errors))
            cv = predictions[winner]
            selected = dict(C=candidates[winner], training_cv_nll=errors[winner])
            trace = [dict(C=v, training_cv_nll=errors[i]) for i, v in enumerate(candidates)]
            mean, scale = standardized(x[train], w[train])
            xx = (x-mean)/scale
            model = LogisticRegression(C=selected['C'], solver='lbfgs', max_iter=3000,
                                       tol=1e-5, random_state=c['seed'])
            model.fit(xx[train], y[train], sample_weight=w[train]/w[train].mean())
            p = model.predict_proba(xx)[:, 1]
            joblib.dump(dict(model=model, mean=mean, scale=scale), tech/'model.joblib')
        elif model_kind == 'catboost':
            from catboost import CatBoostClassifier, Pool
            params = dict(b['catboost'], loss_function='Logloss', eval_metric='Logloss',
                          random_seed=c['seed'], thread_count=c['fit_threads'], allow_writing_files=False)
            histories = []
            for fold in range(c['folds']):
                val = train[folds == fold]
                fitting = train[folds != fold]
                def pool(ids):
                    return Pool(x[ids], y[ids], weight=w[ids]/w[ids].mean())
                model = CatBoostClassifier(**params)
                model.fit(pool(fitting), eval_set=pool(val), use_best_model=False, verbose=False)
                model.save_model(str(tech/f'fold-{fold}.cbm'))
                histories.append(model.get_evals_result()['validation']['Logloss'])
            curves = np.asarray(histories)
            if curves.shape != (c['folds'], params['iterations']):
                raise ValueError(f'Incomplete training-only selection curves: {curves.shape}')
            mass = np.array([w[train[folds == k]].sum() for k in range(c['folds'])])
            mean_curve = mass @ curves / mass.sum()
            trees = int(np.argmin(mean_curve))+1
            selected = dict(iterations=trees, training_cv_nll=float(mean_curve[trees-1]))
            trace = [dict(iterations=i+1, training_cv_nll=float(value)) for i, value in enumerate(mean_curve)]
            for fold in range(c['folds']):
                local = np.flatnonzero(folds == fold)
                model = CatBoostClassifier()
                model.load_model(str(tech/f'fold-{fold}.cbm'))
                cv[local] = model.predict_proba(x[train[local]], ntree_end=trees,
                                                thread_count=c['fit_threads'])[:, 1]
            # No eval_set or early stopping touches the merged test.
            params['iterations'] = trees
            model = CatBoostClassifier(**params)
            model.fit(Pool(x[train], y[train], weight=w[train]/w[train].mean()), verbose=False)
            p = model.predict_proba(x, thread_count=c['fit_threads'])[:, 1]
            model.save_model(str(tech/'model.cbm'))
            write_json(tech/'catboost-parameters.json', model.get_all_params())
            selected['replayed_training_cv_nll'] = scores(y[train], cv, w[train])['nll']
        elif model_kind == 'prior':
            p = np.full(len(y), np.average(y[train], weights=w[train]))
            for fold in range(c['folds']):
                fitting = train[folds != fold]
                cv[folds == fold] = np.average(y[fitting], weights=w[fitting])
        else:
            raise ValueError(model_kind)
    if not np.isfinite(p).all() or not np.isfinite(cv).all():
        raise FloatingPointError(f'Nonfinite probabilities: {spec}')
    negatives = np.flatnonzero(y[train] == 0)
    order = negatives[np.argsort(cv[negatives])]
    cumulative = np.cumsum(w[train[order]])/w[train[order]].sum()
    threshold = float(cv[order[min(np.searchsorted(cumulative, .95), len(order)-1)]])
    np.savez_compressed(tech/'predictions.npz', probability=p, training_oof=cv,
        train_indices=train, role=np.where(rows['role']=='train', 'train', 'merged_test'),
        original_role=rows['role'], **{k: rows[k] for k in ('id','source','event','pair','label','weight')})
    records = []
    for role, ids, values in [('train', train, p[train]), ('training_selection_cv', train, cv),
                              ('merged_test', test, p[test])]:
        pos = y[ids] == 1
        records.append(dict(arm=spec['arm'], observation=spec['observation']['name'],
            role=role, score='raw', rows=len(ids), **scores(y[ids], values, w[ids]),
            recall_at_training_oof_fpr05=float(np.average(values[pos]>threshold, weights=w[ids][pos])),
            observed_fpr=float(np.average(values[~pos]>threshold, weights=w[ids][~pos]))))
    write_metric_rows(records, root, family=family, name='scores')
    write_json(tech/'selection.json', dict(selector='training-source pooled five-fold CV binary NLL',
        selected=selected, history=trace, calibration='none', threshold_training_oof_fpr05=threshold,
        cv_interpretation='Used for hyperparameter selection; not an unbiased nested-CV performance estimate'))
    write_json(done, dict(identity=identity, predictions_sha256=sha(tech/'predictions.npz'), scores=records))
    print(dict(arm=spec['arm'], observation=spec['observation']['name'], selected=selected,
               train_nll=records[0]['nll'], merged_test_nll=records[-1]['nll']), flush=True)
