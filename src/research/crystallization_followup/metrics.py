"""Paired natural-population short-horizon diagnostics; one fitted seed."""
import numpy as np


def window_scores(cdf, event):
    p = np.asarray(cdf, dtype=np.float64)[:, :16]
    event = np.asarray(event)
    if p.shape != (len(event), 16) or not np.isfinite(p).all():
        raise ValueError('Expected finite 0.75 ps CDF through 12 ps')
    if (p < 0).any() or (p > 1).any() or (np.diff(p, axis=1) < -1e-6).any():
        raise ValueError('Invalid cumulative onset distribution')
    occurred = np.arange(16)[None] >= event[:, None]
    # Discrete upper-endpoint time: E[min(T,12)] = .75 sum_{j=0}^{15} S(.75 j).
    restricted_mean = .75 * (1 + (1 - p[:, :15]).sum(1))
    return dict(brier12=((p - occurred)**2).mean(1),
                restricted_time_mae12=np.abs(restricted_mean - .75*np.minimum(event+1, 16)),
                positive12=event < 16)


def paired_source_gain(reference, candidate, sources, draws=1000, seed=20260922):
    """Positive gain means lower candidate error; identical source mass in every draw."""
    sources = np.asarray(sources)
    gains = np.array([(np.asarray(reference)-np.asarray(candidate))[sources == s].mean()
                      for s in np.unique(sources)])
    rng = np.random.default_rng(seed)
    boot = gains[rng.integers(len(gains), size=(draws, len(gains)))].mean(1)
    return dict(gain=float(gains.mean()), ci95=np.quantile(boot, [.025, .975]).tolist(),
                sources=len(gains), bootstrap_draws=draws)
