"""Paired source uncertainty, locked-predictor replay and within-site structural changes."""
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .analysis import bootstrap_scores, predictions as read_prediction, source_draws
from .data import load, read
from .extension import base
from .features import load_bank
from .fit import scores
from .temporal import FAMILY, fit_root, partition
from .temporal_inputs import packet


def interval(value):
    value = np.asarray(value)
    value = value[np.isfinite(value)]
    return tuple(map(float, np.quantile(value, [.025, .975]))) if len(value) else (None, None)


def matched(rows, ids):
    y, pairs = rows['label'][ids], rows['pair'][ids]
    cases, controls, members = [], [], []
    for pair in np.unique(pairs):
        group = np.flatnonzero(pairs == pair)
        pos, neg = group[y[group] == 1], group[y[group] == 0]
        if len(pos) != 1 or len(neg) != 4:
            raise ValueError(f'Broken retained 1:4 match {pair}')
        cases.append(pos[0]); controls.append(neg); members.append(group)
    return np.asarray(cases), np.asarray(controls), np.asarray(members)


def weights_boot(rows, ids, c):
    _, slot, counts = source_draws(rows['source'][ids], c['bootstrap_draws'], c['seed'])
    w = rows['weight'][ids]
    ww = counts[:, slot] * w[None]
    ww /= ww.sum(1, keepdims=True)
    return w / w.sum(), ww, slot, counts


def metric_bundle(rows, ids, p, c):
    y = rows['label'][ids]
    w, ww, slot, counts = weights_boot(rows, ids, c)
    value = scores(y, p, w)
    boot, valid = bootstrap_scores(y, p, w, slot, counts, alarm=None)
    if valid != c['bootstrap_draws']:
        raise ValueError('A whole-source bootstrap lost a class despite complete matched sets')
    value['bootstrap_valid_draws'] = valid
    case, control, members = matched(rows, ids)
    wins = ((p[case, None] > p[control]) + .5 * (p[case, None] == p[control])).mean(1)
    margins = p[case] - p[control].mean(1)
    for key, data in [('matched_auc', wins), ('matched_probability_gap', margins)]:
        per_row = np.empty(len(ids))
        per_row[members] = data[:, None]
        value[key] = float(w @ per_row)
        boot[key] = ww @ per_row
    for key in ('nll', 'brier', 'ap', 'auroc', 'matched_auc', 'matched_probability_gap'):
        value[key + '_lo'], value[key + '_hi'] = interval(boot[key])
    return value, boot


def predictions(c, rows, arm, observation, scope):
    ids = np.flatnonzero(rows['role'] == ('test' if scope == 'fixed_test' else 'train'))
    output = {k: np.empty(len(ids)) for k in ['raw', 'calibrated']}
    covered = np.zeros(len(ids), int)
    provenance = []
    for lane in ([0] if scope == 'fixed_test' else range(1, c['folds'] + 1)):
        fit_scope, split, role = partition(c, rows, lane)
        local = split[role]
        loc = np.searchsorted(ids, local)
        root = fit_root(c, fit_scope, arm, observation)
        data, receipt = read_prediction(root, rows['id'][local], role)
        provenance.append(receipt)
        for key in output:
            output[key][loc] = data[key]
        covered[loc] += 1
    if not np.all(covered == 1):
        raise ValueError(f'Not exactly one held-out prediction per row: {scope}/{arm}/{observation}')
    return ids, output, provenance


def replay(c, b, rows, scope, progress):
    """One selected/calibrated snapshot model per fold, evaluated at every past frame."""
    ids = np.flatnonzero(rows['role'] == ('test' if scope == 'fixed_test' else 'train'))
    dest = resolve_path(c['output']) / f'analyses/{scope}-replay-v1'
    (dest / 'technical').mkdir(parents=True, exist_ok=True)
    summary, changes, provenance = [], [], []
    metric_cache = {}
    for name in c['arms']:
        arm = next(a for a in b['arms'] if a['name'] == name)
        bank, columns = load_bank(b, arm['bank'])
        selected = np.array([i for i, col in enumerate(columns) if
            arm['bank'] != 'descriptors' or col.split('/')[0] in arm['families']])
        for anchor in ['early', 'current']:
            progress.update(replay_scope=scope, arm=name, anchor=anchor)
            outputs = {key: np.empty((8, len(ids))) for key in ['raw', 'calibrated']}
            covered = np.zeros(len(ids), int)
            for lane in ([0] if scope == 'fixed_test' else range(1, c['folds'] + 1)):
                fit_scope, split, role = partition(c, rows, lane)
                local = split[role]
                loc = np.searchsorted(ids, local)
                tech = fit_root(c, fit_scope, name, anchor) / 'technical'
                complete = read(tech / 'complete.json')
                if sha(tech / 'predictions.npz') != complete['predictions_sha256']:
                    raise ValueError('Changed snapshot predictions')
                theta = read(tech / 'selection.json')['calibration_parameters']
                if arm['model'] == 'linear':
                    model_file = tech / 'model.joblib'
                    saved = joblib.load(model_file)
                    if not np.array_equal(saved['selected_columns'], selected):
                        raise ValueError('Snapshot readout descriptor columns changed')
                    def predict(x):
                        return saved['model'].predict_proba((x - saved['mean']) / saved['scale'])[:, 1]
                else:
                    from catboost import CatBoostClassifier
                    model_file = tech / 'model.cbm'
                    model = CatBoostClassifier()
                    model.load_model(str(model_file))
                    def predict(x):
                        return model.predict_proba(x, thread_count=4)[:, 1]
                provenance.append(dict(arm=name, anchor=anchor, scope=fit_scope,
                    model=str(model_file), model_sha256=sha(model_file), fitting_identity=complete['identity']))
                for frame in range(8):
                    x = packet(bank, rows, selected, dict(kind='snapshot', frames=[frame]), c['seed'])
                    p = predict(x[local])
                    outputs['raw'][frame, loc] = p
                    outputs['calibrated'][frame, loc] = expit(theta[0] * logit(np.clip(p, 1e-6, 1-1e-6)) + theta[1])
                original, _ = read_prediction(tech.parent, rows['id'][local], role)
                anchor_frame = 0 if anchor == 'early' else 7
                for key in outputs:
                    if not np.allclose(outputs[key][anchor_frame, loc], original[key], rtol=1e-6, atol=1e-7):
                        raise ValueError(f'Frozen snapshot replay failed: {name}/{fit_scope}/{anchor}/{key}')
                covered[loc] += 1
            if not np.all(covered == 1):
                raise ValueError('Missing held-out rows in temporal replay')
            np.savez_compressed(dest / f'technical/{name}-{anchor}.npz', **outputs,
                **{k: rows[k][ids] for k in ['id', 'source', 'event', 'pair', 'label', 'weight']})
            for key, matrix in outputs.items():
                endpoints = []
                for frame in range(8):
                    values, boot = metric_bundle(rows, ids, matrix[frame], c)
                    summary.append(dict(scope=scope, arm=name, anchor=anchor, score=key,
                        frame=frame, appearance_lead_ps=(8-frame)*.75, **values))
                    if frame in [0, 7]:
                        endpoints.append((values, boot))
                    metric_cache[(name, anchor, key, frame)] = values
                early, late = endpoints
                for metric in ['nll', 'ap', 'auroc', 'matched_auc', 'matched_probability_gap']:
                    lo, hi = interval(late[1][metric] - early[1][metric])
                    changes.append(dict(scope=scope, arm=name, anchor=anchor, score=key,
                        metric=metric, late_minus_early=late[0][metric]-early[0][metric],
                        difference_lo=lo, difference_hi=hi))
        del bank
    write_metric_rows(summary, dest, family=FAMILY, name='time-course')
    write_metric_rows(changes, dest, family=FAMILY, name='late-minus-early')
    write_json(dest / 'technical/replay-provenance.json', provenance)
    plot_replay(summary, dest)
    return summary, changes


def site_descriptors(c, progress):
    b = base(c)
    _, rows, _ = load(b)
    train = np.flatnonzero(rows['role'] == 'train')
    result = []
    root = resolve_path(c['output']) / 'analyses/site-structure-v1'
    for name in ['descriptors', 'mace_rich', 'mace_vicreg']:
        bank, columns = load_bank(b, name)
        values = bank[rows['indices']].astype(np.float64)
        tw = rows['weight'][train]; tw = tw / tw.sum()
        mean = np.einsum('i,ijd->d', tw, values[train]) / 8
        scale = np.sqrt(np.einsum('i,ijd->d', tw, (values[train]-mean)**2) / 8).clip(1e-4)
        z = (values - mean) / scale
        for scope, role in [('training_descriptive', 'train'), ('fixed_test', 'test')]:
            progress.update(structure_bank=name, scope=scope)
            ids = np.flatnonzero(rows['role'] == role)
            zz = z[ids]
            w, _, slot, counts = weights_boot(rows, ids, c)
            case, control, _ = matched(rows, ids)
            cw = rows['weight'][ids][case]
            ww = counts[:, slot[case]] * cw[None]
            ww /= ww.sum(1, keepdims=True)
            cw /= cw.sum()
            means = zz.mean(1)
            center = w @ means
            between = w @ (means-center)**2
            within = w @ zz.var(1)
            fraction = np.divide(between, between+within, out=np.full_like(between, np.nan), where=between+within>1e-12)
            early = zz[case, 0] - zz[control, 0].mean(1)
            late = zz[case, 7] - zz[control, 7].mean(1)
            estimates = {}
            for label, gaps in [('early_gap', early), ('late_gap', late), ('change_gap', late-early)]:
                boot = ww @ gaps
                estimates[label] = cw @ gaps
                estimates[label+'_lo'], estimates[label+'_hi'] = np.quantile(boot, [.025, .975], axis=0)
            for i, column in enumerate(columns):
                result.append(dict(scope=scope, bank=name, feature=column,
                    between_site_variation_fraction=float(fraction[i]) if np.isfinite(fraction[i]) else None,
                    **{k: float(v[i]) for k, v in estimates.items()}))
        del bank, values, z
    write_metric_rows(result, root, family=FAMILY, name='site-structure')
    write_json(root / 'technical/interpretation.json', dict(
        note='Between-site variation fraction is descriptive, not a noise-corrected intraclass correlation.',
        changes='Late-minus-early within-site change, contrasted between the case and its four source/time-matched controls.',
        scaling='Original training rows only; equal event weighting across all eight frames.',
        multiple_comparisons='Exploratory featurewise intervals; no multiplicity correction or test-driven feature selection.',
        window_ps=5.25, limitation='Does not establish persistence beyond the observed window or prospective incidence.'))


def collect(c, progress):
    b = base(c)
    _, rows, manifest = load(b)
    root = resolve_path(c['output'])
    reports = []
    for scope in ['fixed_test', 'readout_cv']:
        summary, contrasts, provenance = [], [], []
        boot_cache, point_cache = {}, {}
        for arm in ['prior'] + c['arms']:
            observations = [dict(name='prior', frames=[])] if arm == 'prior' else c['observations']
            for observation in observations:
                name = observation['name']
                progress.update(comparison_scope=scope, arm=arm, observation=name)
                ids, data, receipts = predictions(c, rows, arm, name, scope)
                provenance.extend(receipts)
                for key, p in data.items():
                    value, boot = metric_bundle(rows, ids, p, c)
                    boot_cache[(arm, name, key)] = boot
                    point_cache[(arm, name, key)] = value
                    frames = observation['frames']
                    summary.append(dict(scope=scope, arm=arm, observation=name, score=key,
                        rows=len(ids), sources=len(np.unique(rows['source'][ids])),
                        observed_frames=len(frames), appearance_lead_ps=(8-frames[-1])*.75 if frames else None,
                        history_span_ps=(frames[-1]-frames[0])*.75 if frames else None, **value))
        for arm in c['arms']:
            comparisons = [(arm, a, arm, z) for a, z in c['contrasts']]
            comparisons += [(arm, o['name'], 'prior', 'prior') for o in c['observations']]
            for a, left, z, right in comparisons:
                for key in ['raw', 'calibrated']:
                    for metric in ['nll', 'brier', 'ap', 'auroc', 'matched_auc']:
                        l, r = (a, left, key), (z, right, key)
                        lo, hi = interval(boot_cache[l][metric] - boot_cache[r][metric])
                        contrasts.append(dict(scope=scope, arm=arm, observation=left, reference_arm=z,
                            reference=right, score=key, metric=metric,
                            difference=point_cache[l][metric]-point_cache[r][metric], difference_lo=lo, difference_hi=hi))
        dest = root / f'analyses/{scope}-comparison-v1'
        write_metric_rows(summary, dest, family=FAMILY, name='summary')
        write_metric_rows(contrasts, dest, family=FAMILY, name='paired-contrasts')
        write_json(dest / 'technical/prediction-provenance.json', provenance)
        plot_comparison(summary, dest)
        _, shifts = replay(c, b, rows, scope, progress)
        reports.extend(summary)
    text = ['# Fixed-endpoint history and persistent-site comparison', '',
        'All fits use the existing cohort, fixed original source roles, and one fit seed. No new encoder training.',
        'Selection is validation NLL, with separate calibration sources. AP is diagnostic.',
        'Fixed test is independent of encoder pretraining; readout CV is conditional on pretraining-exposed sources.',
        'This is exploratory follow-up on previously inspected held-out sources, not a new confirmatory holdout.', '',
        '| Predictor | Input | Fixed-test NLL | Fixed-test AP | Readout-CV NLL | Readout-CV AP |',
        '| --- | --- | ---: | ---: | ---: | ---: |']
    for arm in c['arms']:
        for obs in c['observations']:
            rec = [next(r for r in reports if r['scope']==s and r['arm']==arm and
                        r['observation']==obs['name'] and r['score']=='calibrated') for s in ['fixed_test','readout_cv']]
            text.append(f'| {arm} | {obs["name"]} | {rec[0]["nll"]:.5f} | {rec[0]["ap"]:.4f} | {rec[1]["nll"]:.5f} | {rec[1]["ap"]:.4f} |')
    text += ['', 'Paired intervals and raw scores: `analyses/*-comparison-v1/tables/`.',
        'Locked early/current predictors across all frames: `analyses/*-replay-v1/`.',
        'Within-site descriptor changes and stable variation: `analyses/site-structure-v1/`.', '',
        'Interpretation: a real-history advantage over repeated current frames supports information beyond the current descriptor.',
        'An advantage over shuffled past frames supports use of past ordering, with the current frame held fixed.',
        'Persistent early separation plus little incremental/change information is consistent with site propensity; weak/null models cannot establish that mechanism.',
        'Within-site changes and rising same-model matched separation support evolving precursors within this selected population.',
        'These controls cannot recover natural nucleation incidence or remove future-based site selection.',
        'History spans at most 5.25 ps. Appearance is the first local PTM detection, not critical-nucleus or 64-atom establishment time.',
        'Intervals resample whole sources conditional on fitted models; they exclude fit-seed uncertainty and are not multiplicity adjusted.']
    (root / 'RESULTS.md').write_text('\n'.join(text)+'\n')
    write_json(root / 'technical/comparison-complete.json', dict(dataset_identity=manifest['identity'],
        scopes=['fixed_test','readout_cv'], observations=len(c['observations']), all_rows_preserved=True))


def plot_comparison(summary, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = root / 'plots'; plots.mkdir(parents=True, exist_ok=True)
    groups = dict(fixed_endpoint=['current','repeated_current','history2','history4','history8','shuffled_past'],
                  fixed_length=['early4','middle4','history4'],
                  site_controls=['early','current','mean8','changes8','early_capacity_control','early_plus_change'])
    arms = sorted({r['arm'] for r in summary} - {'prior'})
    for group, names in groups.items():
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        for a, arm in enumerate(arms):
            values = [next(r for r in summary if r['arm']==arm and r['observation']==n and r['score']=='calibrated') for n in names]
            x = np.arange(len(names)) + (a-(len(arms)-1)/2)*.09
            for ax, metric in zip(axes, ['nll','ap','matched_auc']):
                y = np.array([r[metric] for r in values])
                lo = np.array([r[metric+'_lo'] for r in values]); hi = np.array([r[metric+'_hi'] for r in values])
                ax.vlines(x, lo, hi, alpha=.35, color=f'C{a}')
                ax.plot(x, y, 'o-', markersize=3, linewidth=.8, label=arm, color=f'C{a}')
                ax.set_xticks(np.arange(len(names)), names, rotation=30, ha='right')
                ax.set_ylabel(metric); ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7)
        fig.suptitle(root.name + ' · source-bootstrap 95% intervals')
        fig.tight_layout()
        for suffix in ['png','pdf']:
            fig.savefig(plots/f'{group}.{suffix}', dpi=180)
        plt.close(fig)


def plot_replay(summary, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = root / 'plots'; plots.mkdir(parents=True, exist_ok=True)
    arms = sorted({r['arm'] for r in summary})
    for anchor in ['early','current']:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for arm in arms:
            values = sorted([r for r in summary if r['arm']==arm and r['anchor']==anchor and r['score']=='calibrated'], key=lambda r:r['frame'])
            x = [-r['appearance_lead_ps'] for r in values]
            for ax, metric in zip(axes, ['nll','matched_auc','matched_probability_gap']):
                line, = ax.plot(x, [r[metric] for r in values], 'o-', markersize=3, label=arm)
                ax.fill_between(x, [r[metric+'_lo'] for r in values], [r[metric+'_hi'] for r in values], alpha=.08, color=line.get_color())
                ax.set_xlabel('Time relative to first local PTM appearance (ps)')
                ax.set_ylabel(metric); ax.grid(alpha=.2)
        axes[-1].legend(fontsize=7)
        fig.suptitle(f'{root.name}: frozen {anchor} snapshot model, same calibration at every frame')
        fig.tight_layout()
        for suffix in ['png','pdf']:
            fig.savefig(plots/f'locked-{anchor}.{suffix}', dpi=180)
        plt.close(fig)
