"""Source-balanced dimensions and exact-lag movement, without interpolation.

These are linear spectral dimensions, not nonlinear intrinsic dimensions.
All inputs refer to rows of one exported representation; atom IDs are local to
their source. No edge or difference may connect separate tracked atoms.
"""
import numpy as np


def source_weights(source):
    _, inverse, counts = np.unique(source, return_inverse=True, return_counts=True)
    return 1. / (len(counts) * counts[inverse])


def spectrum(values, weights=None, *, centered=True):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or not np.isfinite(x).all():
        raise ValueError(f'Expected nonempty finite embedding matrix, got {x.shape}')
    w = np.ones(len(x))/len(x) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(x),) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Spectrum weights must be finite, positive and aligned to rows')
    w = w/w.sum()
    if centered:
        # Subtract an anchor first: native states can have large constant means.
        x = x-x[0]
        x = x-w@x
    gram = (x.T*w)@x
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    if eigenvalues[-1] < -max(eigenvalues[0], np.finfo(float).tiny)*1e-10:
        raise FloatingPointError('Embedding second moment is numerically non-PSD')
    eigenvalues = eigenvalues.clip(0)
    total = float(eigenvalues.sum())
    p = eigenvalues/total if total > 0 else eigenvalues
    positive = p > 0
    result = dict(rows=len(x), dimensions=x.shape[1], centered=centered,
                  rank_ceiling=min(x.shape[1], len(x)-int(centered)),
                  total_energy=total, collapsed=total == 0,
                  numerical_rank=int(np.count_nonzero(eigenvalues > eigenvalues[0]*1e-10)),
                  participation_rank=float(1/np.square(p).sum()) if total > 0 else None,
                  entropy_rank=float(np.exp(-np.sum(p[positive]*np.log(p[positive])))) if total > 0 else None,
                  eigenvalues=eigenvalues.tolist())
    for fraction in (90, 95, 99):
        result[f'd{fraction}'] = int(np.searchsorted(np.cumsum(p), fraction/100)+1) if total > 0 else None
    return result


def tracks(source, atom, time_ps):
    """Return chronological row indices, rejecting ambiguous source/atom/time keys."""
    source, atom, time_ps = np.asarray(source), np.asarray(atom), np.asarray(time_ps, dtype=float)
    if source.ndim != 1 or atom.shape != source.shape or time_ps.shape != source.shape or not np.isfinite(time_ps).all():
        raise ValueError('Source, atom and finite physical times must be aligned vectors')
    groups = {}
    for i, key in enumerate(zip(source.tolist(), atom.tolist(), strict=True)):
        groups.setdefault(key, []).append(i)
    result = []
    for key, rows in groups.items():
        ix = np.array(rows)[np.argsort(time_ps[rows], kind='stable')]
        if np.any(np.diff(time_ps[ix]) <= 0):
            raise ValueError(f'Duplicate source/atom/time observation: {key}')
        result.append(ix)
    return result


def lag_pairs(track_rows, time_ps, lag_ps):
    if not np.isfinite(lag_ps) or lag_ps <= 0:
        raise ValueError(f'Physical lag must be positive: {lag_ps}')
    pairs = []
    for ix in track_rows:
        times = time_ps[ix]
        right = np.searchsorted(times, times+lag_ps-1e-8)
        left = np.flatnonzero(right < len(ix))
        left = left[np.isclose(times[right[left]]-times[left], lag_ps, atol=1e-8, rtol=0)]
        pairs.extend(zip(ix[left], ix[right[left]], strict=True))
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _quantile(x, w, q):
    order = np.argsort(x, kind='stable')
    return float(x[order[np.searchsorted(np.cumsum(w[order])/w.sum(), q)]])


def _compact(spec):
    return {k: v for k, v in spec.items() if k != 'eigenvalues'}


def _population(z, source, rows):
    if not len(rows):
        return dict(rows=0, sources=0, spectrum=None)
    return dict(rows=len(rows), sources=len(np.unique(source[rows])),
                spectrum=spectrum(z[rows], source_weights(source[rows])))


def analyze(values, source, atom, time_ps, reference_indices, evaluation_indices,
            *, lags_ps, domains=None):
    """Descriptive full-dataset ranks; held-out dynamics normalized on fit rows.

    Domains are named Boolean masks in the full row space, e.g. noncrystalline
    or temperature. Both endpoints must belong to the same domain. Lags never
    pool across different physical durations. Sources carry equal total weight.
    """
    z = np.asarray(values, dtype=np.float64)
    source, atom, time_ps = np.asarray(source), np.asarray(atom), np.asarray(time_ps, dtype=float)
    if z.ndim != 2 or len(z) != len(source) or not np.isfinite(z).all():
        raise ValueError('Finite exported features must align with source metadata')
    all_tracks = tracks(source, atom, time_ps)
    n = len(z)
    reference_indices, evaluation_indices = np.asarray(reference_indices), np.asarray(evaluation_indices)
    for label, ix in [('reference', reference_indices), ('evaluation', evaluation_indices)]:
        if ix.ndim != 1 or ix.dtype.kind not in 'iu' or not len(ix) or np.any((ix < 0) | (ix >= n)) or len(np.unique(ix)) != len(ix):
            raise ValueError(f'Invalid {label} row indices')
    if np.intersect1d(source[reference_indices], source[evaluation_indices]).size:
        raise ValueError('Reference and evaluation sources must be disjoint')
    if not lags_ps or len(set(lags_ps)) != len(lags_ps):
        raise ValueError('Require distinct positive physical lags')
    domains = {} if domains is None else domains
    masks = {'all': np.ones(n, dtype=bool)}
    for name, mask in domains.items():
        mask = np.asarray(mask)
        if name == 'all' or mask.shape != (n,) or mask.dtype != bool:
            raise ValueError(f'Invalid named domain mask: {name}')
        masks[name] = mask
    ref = spectrum(z[reference_indices], source_weights(source[reference_indices]))
    if ref['collapsed']:
        raise ValueError('Collapsed training reference: normalized stability is undefined')
    reference_trace = ref['total_energy']
    included = np.zeros(n, dtype=bool); included[evaluation_indices] = True
    eval_tracks = [ix[included[ix]] for ix in all_tracks if included[ix].any()]
    result = dict(protocol='embedding_dynamics_v1', reference=ref,
                  evaluation_sources=len(np.unique(source[evaluation_indices])),
                  evaluation_tracks=len(eval_tracks), domains={}, per_track=[])
    for name, mask in masks.items():
        ev = evaluation_indices[mask[evaluation_indices]]
        tr = reference_indices[mask[reference_indices]]
        residual = np.empty_like(z)
        for track in eval_tracks:
            ix = track[mask[track]]
            if len(ix):
                x = z[ix]-z[ix[0]]
                residual[ix] = x-x.mean(0)
        local_ref = _population(z, source, tr)
        variance = local_ref['spectrum']['total_energy'] if len(tr) else 0.
        block = dict(dataset=_population(z, source, np.flatnonzero(mask)),
                     evaluation=_population(z, source, ev), reference=local_ref,
                     within_track=_population(residual, source, ev), lags={})
        for lag in lags_ps:
            pairs = lag_pairs(eval_tracks, time_ps, lag)
            pairs = pairs[mask[pairs].all(1)]
            count = len(pairs)
            row = dict(pairs=count, sources=len(np.unique(source[pairs[:, 0]])) if count else 0)
            if count:
                delta = z[pairs[:, 1]]-z[pairs[:, 0]]
                weights = source_weights(source[pairs[:, 0]])
                energy = np.square(delta).sum(1)
                velocity = delta/lag
                mean_velocity = weights@velocity
                total_velocity = float(weights@np.square(velocity).sum(1))
                jump = np.sqrt(energy/(2*reference_trace))
                row.update(rms_jump=float(np.sqrt(weights@np.square(jump))),
                           p50_jump=_quantile(jump, weights, .5), p95_jump=_quantile(jump, weights, .95),
                           p99_jump=_quantile(jump, weights, .99),
                           domain_reference_rms_jump=float(np.sqrt(weights@energy/(2*variance))) if variance > 0 else None,
                           zero_increment_fraction=float(weights@(energy == 0)),
                           velocity_rms=float(np.sqrt(total_velocity)),
                           drift_energy_fraction=float(np.square(mean_velocity).sum()/total_velocity) if total_velocity > 0 else None,
                           movement=spectrum(velocity, weights, centered=False),
                           fluctuation=spectrum(velocity, weights, centered=True))
            block['lags'][str(float(lag))] = row
        result['domains'][name] = block
    for ix in eval_tracks:
        state = spectrum(z[ix])
        row = dict(source=str(source[ix[0]]), atom=str(atom[ix[0]]), observations=len(ix),
                   first_ps=float(time_ps[ix[0]]), last_ps=float(time_ps[ix[-1]]),
                   state=_compact(state), movement=None, fluctuation=None,
                   velocity_roughness=None, increment_cosine=None, reversal_fraction=None,
                   turn_pairs=0, adjacent_lags_ps=np.unique(np.diff(time_ps[ix])).tolist())
        if len(ix) > 1:
            dt = np.diff(time_ps[ix])
            velocity = np.diff(z[ix], axis=0)/dt[:, None]
            row['movement'] = _compact(spectrum(velocity, centered=False))
            row['fluctuation'] = _compact(spectrum(velocity))
            # Unequal steps do not enter equal-lag turn/roughness measurements.
            equal = np.isclose(dt[:-1], dt[1:], rtol=0, atol=1e-8)
            a, b = velocity[:-1][equal], velocity[1:][equal]
            denominator = np.square(a).sum()+np.square(b).sum()
            row['turn_pairs'] = len(a)
            row['velocity_roughness'] = float(np.square(b-a).sum()/denominator) if denominator > 0 else None
            product = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)
            nonzero = product > 0
            cosine = np.sum(a*b, axis=1)[nonzero]/product[nonzero]
            row['increment_cosine'] = float(cosine.mean()) if len(cosine) else None
            row['reversal_fraction'] = float(np.mean(cosine < 0)) if len(cosine) else None
        result['per_track'].append(row)
    return result
