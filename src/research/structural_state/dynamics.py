"""Fixed, fit-source-only labels for the geometry/future factorial experiment."""
import numpy as np
from sklearn.linear_model import Ridge

PROTOCOL = 'fixed_geometry_future_relation_v3'


def targets(corpus, config):
    """Current order plus future residual beyond a declared present-state model.

    The baseline and every normalization use fitting sources only. Its inputs
    are observed current order8, relaxed radial/l2/l4 descriptors, temperature,
    and elapsed time. These are training-label construction inputs, never extra
    encoder inputs. Onset labels and development rows cannot fit this baseline.
    """
    settings = config['dynamics']
    fit = corpus.split['fit']
    now = corpus.targets['current_order'].astype(np.float64)
    future = corpus.targets[f'future_order_{settings["lag_ps"]:g}'].astype(np.float64)
    temp = np.array([r['temperature_K'] for r in corpus.records])
    time = np.array([r['frame']*.75/600 for r in corpus.records])
    levels = np.unique(temp[fit])
    if not np.isin(temp, levels).all():
        raise ValueError('Future residual has an unseen evaluation temperature')
    x = np.c_[now, corpus.geometry['relaxed'], temp[:, None] == levels, time, time**2]
    mean = x[fit].mean(0); scale = x[fit].std(0).clip(1e-6)
    x = (x-mean)/scale
    model = Ridge(alpha=settings['baseline_ridge'], solver='svd').fit(x[fit], future[fit])
    baseline = model.predict(x)
    values = dict(current_order=now, future_residual=future-baseline)
    scalers = {}
    for name, value in values.items():
        scalers[name] = dict(mean=value[fit].mean(0).tolist(),
                             scale=value[fit].std(0).clip(1e-6).tolist())
        values[name] = ((value-scalers[name]['mean'])/scalers[name]['scale']).astype(np.float32)
    receipt = dict(lag_ps=settings['lag_ps'], baseline_ridge=settings['baseline_ridge'],
                   x_mean=mean.tolist(), x_scale=scale.tolist(),
                   coefficient=model.coef_.tolist(), intercept=model.intercept_.tolist(),
                   temperatures=levels.tolist(), scalers=scalers,
                   fit_rows=fit.tolist(), target='original-MD order8 at same atom',
                   baseline_inputs='current order8 + relaxed geometry89 + temperature + time/time²')
    return values, receipt, baseline.astype(np.float32)
