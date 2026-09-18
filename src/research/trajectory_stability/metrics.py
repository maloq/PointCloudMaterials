"""Float64 temporal metrics with a frozen training-source distance reference."""
import numpy as np


def reference_statistics(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError(f'Expected finite reference matrix, got {x.shape}')
    mean = x.mean(0)
    centered = x-mean
    variance = np.mean(centered**2, axis=0)
    trace = variance.sum()
    if trace <= 0:
        raise ValueError('Collapsed reference representation: zero total variance')
    active = variance > variance.max()*1e-12
    eigenvalues = np.linalg.svd(centered, compute_uv=False)**2/len(x)
    return dict(mean=mean, variance=variance, trace=trace, active=active,
        effective_rank=trace**2/np.sum(eigenvalues**2),
        scale=np.sqrt(2*trace), dimensions=x.shape[1])


def trajectory_metrics(values, reference, lags):
    """Input order: time, tracked atom, feature; never join different atoms."""
    z = np.asarray(values, dtype=np.float64)
    if z.ndim != 3 or len(z) <= max(lags) or not np.isfinite(z).all():
        raise ValueError(f'Expected finite consecutive trajectory longer than lags: {z.shape}')
    delta = np.diff(z, axis=0)
    jump2 = np.sum(delta**2, axis=-1)/(2*reference['trace'])
    second = np.diff(z, n=2, axis=0)
    second_energy = np.mean(np.sum(second**2, axis=-1))
    paired_increment_energy = np.mean(np.sum(delta[:-1]**2+delta[1:]**2, axis=-1))
    norm = np.linalg.norm(delta, axis=-1)
    product = norm[:-1]*norm[1:]
    valid = product > 0
    cosine = np.sum(delta[:-1]*delta[1:], axis=-1)[valid]/product[valid]
    active = reference['active']
    standardized = delta[..., active]/np.sqrt(reference['variance'][active])
    lag2 = np.array([np.mean(np.sum((z[lag:]-z[:-lag])**2, axis=-1))/(2*reference['trace']) for lag in lags])
    return dict(jump2=jump2, lag2=lag2, jump2_mean=float(jump2.mean()),
        jump_p95=float(np.quantile(np.sqrt(jump2), .95)),
        raw_increment_mse=float(np.mean(delta**2)),
        standardized_jump2=float(np.mean(standardized**2)/2),
        acceleration2=second_energy/(6*reference['trace']),
        roughness=second_energy/paired_increment_energy if paired_increment_energy > 0 else None,
        reversal_fraction=float(np.mean(cosine < 0)) if len(cosine) else None,
        increment_cosine=float(cosine.mean()) if len(cosine) else None,
        zero_increment_fraction=float(np.mean(norm == 0)))


def stratified_draws(temperatures, repeats, seed):
    temperature = np.asarray(temperatures)
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.choice(indices, (repeats, len(indices)), replace=True)
        for value in np.unique(temperature) for indices in [np.flatnonzero(temperature == value)]], axis=1)


def rms_interval(squared_source_values, draws):
    values = np.asarray(squared_source_values, dtype=np.float64)
    return float(np.sqrt(values.mean())), np.quantile(np.sqrt(values[draws].mean(axis=1)), [.025, .975])
