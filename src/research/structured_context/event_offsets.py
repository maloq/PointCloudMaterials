"""Fixed-event, same-source matched-control analysis of archived onset forecasts."""
import numpy as np
from sklearn.metrics import average_precision_score


def cohort(corpus, test_indices, offsets_ps, seed):
    """One local first onset and one control, both available at every offset.

    Origins round down to the archived 3 ps grid: actual leads are in
    [requested, requested + 3) ps. Controls are at risk at every origin and
    have no first onset by the case's onset time. They may crystallize later.
    Pair selection uses labels/availability only, never predictor scores.
    """
    steps = np.asarray(offsets_ps) / .75
    if np.any(steps % 4) or np.any(steps <= 0) or np.any(steps > 124):
        raise ValueError('Offsets must be positive multiples of 3 ps, at most 93 ps')
    steps = steps.astype(int)
    lookup = {(s, corpus.plan['anchors'][a], c): row
              for row, i in enumerate(test_indices) for s, a, c, _ in [corpus.rows[i]]}
    sources = sorted({key[0] for key in lookup})
    records = []; rows = []; leads = []; excluded = dict(unavailable_offsets=0, no_control=0)
    for sid in sources:
        a = corpus.arrays[sid]
        onsets = np.asarray(a['onset']); last = a['labels'].shape[-1]
        for center, onset in enumerate(onsets):
            if onset >= last:
                continue
            origins = ((int(onset)-steps)//4)*4
            case_keys = [(sid, int(t), center) for t in origins]
            if any(key not in lookup for key in case_keys):
                excluded['unavailable_offsets'] += 1
                continue
            controls = [c for c, future in enumerate(onsets) if future > onset and
                        all((sid, int(t), c) in lookup for t in origins)]
            if not controls:
                excluded['no_control'] += 1
                continue
            rng = np.random.default_rng(np.random.SeedSequence([seed, int(sid), center]))
            control = int(rng.choice(controls))
            rows.append([[lookup[(sid, int(t), center)], lookup[(sid, int(t), control)]] for t in origins])
            leads.append((onset-origins)*.75)
            records.append(dict(source=int(sid), center=center, control_center=control,
                onset_frame=int(onset), onset_ps=float(onset*.75), origins_frames=origins.tolist(),
                control_onset_frame=int(onsets[control])))
    if not records:
        raise ValueError(f'No complete event/control pairs: {excluded}')
    return dict(records=records, rows=np.asarray(rows, dtype=int), leads=np.asarray(leads),
                sources=np.asarray([r['source'] for r in records]), excluded=excluded)


def timing(cdf):
    """96 ps restricted mean, plus mean conditional on an event within 96 ps.

    End-of-bin masses match the original 0.75 ps discrete label producer.
    The restricted mean puts all surviving probability at 96 ps; no known
    event time is used to truncate or renormalize the predictive distribution.
    """
    cdf = np.asarray(cdf, dtype=np.float64)
    if cdf.shape[-1] != 128 or not np.isfinite(cdf).all() or np.any(np.diff(cdf, axis=-1) < -1e-6):
        raise ValueError('Expected finite monotone 128-bin onset CDF')
    if cdf.min() < 0 or cdf.max() > 1:
        raise ValueError('Onset probabilities outside [0,1]')
    mass = np.diff(np.concatenate((np.zeros_like(cdf[..., :1]), cdf), axis=-1), axis=-1)
    first_moment = mass @ (.75*np.arange(1, 129))
    restricted = first_moment + (1-cdf[..., -1])*96
    conditional = np.divide(first_moment, cdf[..., -1],
        out=np.full_like(first_moment, np.nan), where=cdf[..., -1] > 0)
    return restricted, conditional


def evaluate(predictions, matched, offsets_ps, *, draws=1000, seed=20260921):
    rows = matched['rows']; sources = matched['sources']; lead = matched['leads']
    unique, inverse, counts = np.unique(sources, return_inverse=True, return_counts=True)
    rng = np.random.default_rng(seed)
    boot = rng.integers(len(unique), size=(draws, len(unique)))
    multiplicity = np.stack([np.bincount(b, minlength=len(unique)) for b in boot])
    event_weights = 1/(len(unique)*counts[inverse])
    boot_weights = multiplicity[:, inverse]*event_weights[None]
    result = {}; arrays = {}
    for name, p in predictions.items():
        cdf = p['test_cdf'][rows]
        bins = np.rint(lead/.75).astype(int)-1
        risk = np.take_along_axis(cdf, bins[..., None, None].repeat(2, axis=2), axis=-1)[..., 0]
        restricted, conditional = timing(cdf[:, :, 0])
        error = np.abs(restricted-lead); conditional_error = np.abs(conditional-lead)
        if not np.isfinite(conditional_error).all():
            raise ValueError(f'{name}: zero total event probability; conditional timing undefined')
        arrays[name+'_risk'] = risk; arrays[name+'_restricted_time'] = restricted
        arrays[name+'_conditional_time'] = conditional
        rows_metrics = {}
        for j, offset in enumerate(offsets_ps):
            labels = np.tile([1, 0], len(rows)); scores = risk[:, j].reshape(-1)
            def stats(weights):
                return [average_precision_score(labels, scores, sample_weight=np.repeat(weights*.5, 2)),
                    weights @ error[:, j], weights @ conditional_error[:, j],
                    weights @ risk[:, j, 0], weights @ risk[:, j, 1],
                    weights @ (cdf[:, j, 0, -1])]
            point = stats(event_weights)
            replicate = np.array([stats(w) for w in boot_weights])
            low, high = np.quantile(replicate, [.025, .975], axis=0)
            names = ('matched_ap', 'restricted_timing_mae_ps', 'conditional_timing_mae_ps',
                     'case_probability', 'control_probability', 'event_mass_96ps')
            rows_metrics[str(offset)] = {key:dict(value=float(v), ci95=[float(l), float(h)])
                for key, v, l, h in zip(names, point, low, high, strict=True)}
            rows_metrics[str(offset)].update(events=len(rows), sources=len(unique),
                actual_lead_min_ps=float(lead[:, j].min()), actual_lead_max_ps=float(lead[:, j].max()))
        result[name] = rows_metrics
    return result, arrays
