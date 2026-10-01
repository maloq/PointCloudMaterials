"""Source bootstrap, cross-fitted readout diagnostics and physical-lead figures."""
from pathlib import Path
import hashlib

import numpy as np

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .data import load, read
from .fit import scores

METRICS = ('nll', 'brier', 'ap', 'auroc', 'recall_at_locked_fpr05', 'observed_fpr')
LABELS = dict(nll='Binary log loss (lower is better)', brier='Brier score (lower is better)',
              ap='Average precision', auroc='AUROC', recall_at_locked_fpr05='Recall at calibration-locked threshold',
              observed_fpr='Achieved false-positive rate')
NAMES = dict(prior='Constant prior', rich_linear='Rich descriptors · linear', rich_gbdt='Rich descriptors · trees',
             mace_rich_linear='Rich/TDA MACE · linear', mace_rich_gbdt='Rich/TDA MACE · trees',
             mace_vicreg_linear='VICReg MACE · linear', mace_vicreg_gbdt='VICReg MACE · trees',
             tda_gbdt='TDA · trees', bond_order_gbdt='Bond order · trees',
             geometry_linear='Geometry · linear', geometry_cna_gbdt='Geometry/CNA · trees')
GROUPS = dict(primary=['prior', 'rich_linear', 'rich_gbdt', 'mace_rich_linear', 'mace_rich_gbdt', 'mace_vicreg_linear', 'mace_vicreg_gbdt'],
              descriptor_controls=['prior', 'geometry_linear', 'geometry_cna_gbdt', 'bond_order_gbdt', 'tda_gbdt', 'rich_linear', 'rich_gbdt'])


def source_draws(source, draws, seed):
    ids = np.unique(source)
    slot = np.searchsorted(ids, source)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(len(ids), size=len(ids)), minlength=len(ids)) for _ in range(draws)])
    return ids, slot, counts


def bootstrap_scores(y, p, w, slot, counts, alarm):
    """Weighted sklearn-equivalent curves, sorting only once for every source draw.

    Aggregate tied prediction thresholds before integration. Zero-weight rows
    contribute no curve increments. Every resample contains whole sources.
    """
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    weight = counts[:, slot] * w[None]
    total = weight.sum(1)
    positive = weight @ y
    negative = total - positive
    valid = (positive > 0) & (negative > 0)
    weight, total, positive, negative = weight[valid], total[valid], positive[valid], negative[valid]
    loss = -y * np.log(p) - (1 - y) * np.log1p(-p)
    result = dict(nll=weight @ loss / total, brier=weight @ (p - y) ** 2 / total)
    order = np.argsort(p, kind='stable')[::-1]
    thresholds = np.r_[np.flatnonzero(np.diff(p[order])), len(p) - 1]
    tp = np.cumsum(weight[:, order] * y[order], axis=1)[:, thresholds]
    fp = np.cumsum(weight[:, order] * (1 - y[order]), axis=1)[:, thresholds]
    recall = tp / positive[:, None]
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    previous_recall = np.column_stack((np.zeros(len(recall)), recall[:, :-1]))
    result['ap'] = (precision * (recall - previous_recall)).sum(1)
    fpr = fp / negative[:, None]
    previous_fpr = np.column_stack((np.zeros(len(fpr)), fpr[:, :-1]))
    result['auroc'] = (.5 * (recall + previous_recall) * (fpr - previous_fpr)).sum(1)
    if alarm is not None:
        result['recall_at_locked_fpr05'] = weight @ (y * alarm) / positive
        result['observed_fpr'] = weight @ ((1 - y) * alarm) / negative
    return result, int(valid.sum())


def measured(y, p, w, alarm):
    result = scores(y, p, w)
    if alarm is not None:
        result.update(recall_at_locked_fpr05=float(np.average(alarm[y == 1], weights=w[y == 1])),
                      observed_fpr=float(np.average(alarm[y == 0], weights=w[y == 0])))
    else:
        result.update(recall_at_locked_fpr05=None, observed_fpr=None)
    return result


def predictions(folder, expected_ids, role):
    tech = folder / 'technical'
    record = read(tech / 'complete.json')
    path = tech / 'predictions.npz'
    if sha(path) != record['predictions_sha256']:
        raise ValueError(f'Changed retained predictions: {path}')
    selection = read(tech / 'selection.json')
    with np.load(path) as data:
        ids = np.flatnonzero(data['role'] == role)
        if not np.array_equal(data['id'][ids], expected_ids):
            raise ValueError(f'Mismatched evaluation identities: {path}/{role}')
        p = data['calibrated'][ids].copy()
        raw = data['probability'][ids].copy()
    return dict(raw=raw, calibrated=p, alarm=p > selection['threshold_fpr05']), dict(
        path=str(path), sha256=record['predictions_sha256'], role=role,
        threshold=selection['threshold_fpr05'], fitting_identity=record['identity'])


def render(summary, root, title, horizons):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = root / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    curves = [r for r in summary if r['score'] == 'calibrated']
    for group, names in GROUPS.items():
        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        for color, name in zip(plt.get_cmap('tab10').colors, names):
            values = sorted([r for r in curves if r['arm'] == name], key=lambda r: r['remove_frames'])
            x = [r['appearance_lead_ps'] for r in values]
            for metric, ax in zip(METRICS, axes.flat):
                ax.plot(x, [r[metric] for r in values], 'o-', color=color, linewidth=1.4, markersize=3, label=NAMES[name])
                ax.fill_between(x, [r[metric + '_lo'] for r in values], [r[metric + '_hi'] for r in values], color=color, alpha=.09)
                ax.set_ylabel(LABELS[metric])
                ax.grid(alpha=.18)
                if metric in ('ap', 'auroc', 'recall_at_locked_fpr05', 'observed_fpr'):
                    ax.set_ylim(0, 1)
        for ax in axes[0]:
            top = ax.secondary_xaxis('top')
            top.set_xticks(horizons, [str(9 - int(round(h / .75))) for h in horizons])
            top.set_xlabel('Retained frames (common starting frame)')
        for ax in axes[1]:
            ax.set_xlabel('Lead before first observed crystal atom (ps)')
            ax.set_xticks(horizons)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, fontsize=9)
        fig.suptitle(title + '\nBands: 95% whole-source bootstrap intervals, conditional on fitted readouts', fontsize=12)
        fig.tight_layout(rect=(0, .12, 1, .95))
        for suffix in ('png', 'pdf'):
            fig.savefig(plots / f'{group}-metrics.{suffix}', dpi=180)
        plt.close(fig)


def reliability(packets, y, w, source, root, family, c):
    import matplotlib.pyplot as plt
    names = ['rich_linear', 'rich_gbdt', 'mace_rich_linear', 'mace_vicreg_linear']
    removed = sorted({r for _, r in packets})
    chosen = [removed[0], removed[-1]]
    ids, slot, counts = source_draws(source, c['bootstrap_draws'], c['fold_seed'])
    bootw = counts[:, slot] * w
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    records = []
    for rem, ax in zip(chosen, axes):
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
        for name in names:
            p = packets[(name, rem)]['calibrated']
            bins = np.minimum((p * 10).astype(int), 9)
            points = []
            for index in range(10):
                take = bins == index
                if not take.any():
                    continue
                mass = bootw[:, take].sum(1)
                available = mass > 0
                rates = (bootw[:, take] @ y[take])[available] / mass[available]
                low, high = np.quantile(rates, [.025, .975])
                value = dict(arm=name, remove_frames=rem, bin=index, rows=int(take.sum()),
                    sources=int(len(np.unique(source[take]))), predicted_mean=float(np.average(p[take], weights=w[take])),
                    positive_fraction=float(np.average(y[take], weights=w[take])), rate_lo=float(low), rate_hi=float(high),
                    supported_source_draws=int(available.sum()))
                records.append(value)
                points.append(value)
            ax.plot([r['predicted_mean'] for r in points], [r['positive_fraction'] for r in points], 'o-', label=NAMES[name])
        ax.set(title=f'{(rem + 1) * .75:g} ps before appearance', xlabel='Mean predicted probability', xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Observed positive fraction')
    axes[1].legend(fontsize=8)
    fig.suptitle('Calibration in the enriched 20% positive population')
    fig.tight_layout()
    fig.savefig(root / 'plots/reliability.png', dpi=180)
    fig.savefig(root / 'plots/reliability.pdf')
    plt.close(fig)
    write_metric_rows(records, root, family=family, name='reliability')


def summarize(scope, packets, y, w, source, event, fold_records, c, root, family):
    ids, slot, counts = source_draws(source, c['bootstrap_draws'], c['fold_seed'])
    summary, contrasts = [], []
    for (name, removed), packet in packets.items():
        for kind in ('raw', 'calibrated'):
            alarm = packet['alarm'] if kind == 'calibrated' else None
            result = measured(y, packet[kind], w, alarm)
            boot, valid = bootstrap_scores(y, packet[kind], w, slot, counts, alarm)
            ci = {metric + ending: float(np.quantile(values, quantile)) for metric, values in boot.items()
                  for ending, quantile in (('_lo', .025), ('_hi', .975))}
            for metric in METRICS:
                ci.setdefault(metric + '_lo', None)
                ci.setdefault(metric + '_hi', None)
            spread = {metric + '_fold_sd': None for metric in METRICS}
            if fold_records:
                values = [r for r in fold_records if r['arm'] == name and r['remove_frames'] == removed and r['score'] == kind]
                spread.update({metric + '_fold_sd': float(np.std([r[metric] for r in values], ddof=1)) for metric in METRICS if result[metric] is not None})
            summary.append(dict(scope=scope, arm=name, remove_frames=removed, history_frames=8 - removed,
                history_span_ps=(7 - removed) * .75, appearance_lead_ps=(removed + 1) * .75,
                score=kind, rows=len(y), positives=int(y.sum()), births=len(set(zip(source.tolist(), event.tolist()))),
                sources=len(ids), bootstrap_valid_draws=valid, **result, **ci, **spread))
    # Source-paired proper-score differences to the prior and to the full history.
    mass = np.bincount(slot, weights=w, minlength=len(ids))
    denominator = counts @ mass
    def loss(packet):
        p = np.clip(packet['calibrated'], 1e-7, 1 - 1e-7)
        return -y * np.log(p) - (1 - y) * np.log1p(-p)
    for (name, removed), packet in packets.items():
        for reference, r0, contrast in [('prior', removed, 'versus_prior'), (name, 0, 'versus_full_history')]:
            if (reference, r0) == (name, removed):
                continue
            delta = loss(packet) - loss(packets[(reference, r0)])
            numerator = np.bincount(slot, weights=w * delta, minlength=len(ids))
            low, high = np.quantile(counts @ numerator / denominator, [.025, .975])
            contrasts.append(dict(scope=scope, arm=name, remove_frames=removed, reference=reference,
                reference_remove_frames=r0, contrast=contrast, nll_difference=float(np.average(delta, weights=w)),
                difference_lo=float(low), difference_hi=float(high)))
    write_metric_rows(summary, root, family=family, name='summary')
    write_metric_rows(contrasts, root, family=family, name='paired-nll')
    if fold_records:
        write_metric_rows(fold_records, root, family=family, name='fold-scores')
    render(summary, root, 'Fixed held-out test' if scope == 'fixed_test' else 'Readout CV on original training sources; encoders fixed',
           sorted({r['appearance_lead_ps'] for r in summary}))
    reliability(packets, y, w, source, root, family, c)
    write_json(root / 'technical/source-bootstrap.json', dict(source_ids=ids.tolist(), seed=c['fold_seed'],
               draws=c['bootstrap_draws'], draws_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
               conditional_on_fitted_models=True, refits_in_each_bootstrap=False, fold_spread_is_not_an_independent_standard_error=True))
    return summary


def collect(c, existing_only=False):
    from .extension import FAMILY, base
    b = base(c)
    _, rows, manifest = load(b)
    root = resolve_path(c['output'])
    parent = resolve_path(c['parent_run'])
    removals = b['remove_frames'] if existing_only else c['remove_frames']
    test = np.flatnonzero(rows['role'] == 'test')
    packets, provenance = {}, []
    for removed in removals:
        for arm in b['arms']:
            folder = (parent / 'analyses/classification-v1' if removed in b['remove_frames'] else root / 'analyses/fixed-test-v1') / arm['name'] / f'minus-{removed}'
            packet, record = predictions(folder, rows['id'][test], 'test')
            packets[(arm['name'], removed)] = packet
            provenance.append(record)
    dest = root / 'analyses' / ('existing-visualization-v1' if existing_only else 'fixed-test-comparison-v1')
    fixed = summarize('fixed_test', packets, rows['label'][test], rows['weight'][test], rows['source'][test],
                      rows['event'][test], [], c, dest, FAMILY)
    write_json(dest / 'technical/prediction-provenance.json', provenance)
    if existing_only:
        return
    design = read(root / 'technical/folds.json')
    train = np.flatnonzero(rows['role'] == 'train')
    slots = {identity: i for i, identity in enumerate(rows['id'][train])}
    folds = np.array([design['binding']['source_folds'][str(s)] for s in rows['source'][train]])
    packets, fold_records, provenance = {}, [], []
    for removed in c['remove_frames']:
        for arm in b['arms']:
            combined = dict(raw=np.empty(len(train)), calibrated=np.empty(len(train)), alarm=np.empty(len(train), bool))
            filled = np.zeros(len(train), bool)
            for fold in range(c['folds']):
                ids = train[folds == fold]
                packet, record = predictions(root / f'analyses/cv-v1/fold-{fold}' / arm['name'] / f'minus-{removed}', rows['id'][ids], 'cv_evaluation')
                where = np.array([slots[i] for i in rows['id'][ids]])
                if filled[where].any():
                    raise ValueError('An OOF row was predicted by multiple outer folds')
                filled[where] = True
                for key in combined:
                    combined[key][where] = packet[key]
                for kind in ('raw', 'calibrated'):
                    alarm = packet['alarm'] if kind == 'calibrated' else None
                    fold_records.append(dict(fold=fold, arm=arm['name'], remove_frames=removed, score=kind,
                        rows=len(ids), sources=len(np.unique(rows['source'][ids])),
                        **measured(rows['label'][ids], packet[kind], rows['weight'][ids], alarm)))
                provenance.append(record)
            if not filled.all():
                raise ValueError('Incomplete out-of-fold prediction coverage')
            packets[(arm['name'], removed)] = combined
    dest = root / 'analyses/readout-cv-comparison-v1'
    cv = summarize('readout_cv', packets, rows['label'][train], rows['weight'][train], rows['source'][train],
                   rows['event'][train], fold_records, c, dest, FAMILY)
    write_json(dest / 'technical/prediction-provenance.json', provenance)
    write_json(dest / 'technical/oof-coverage.json', dict(rows=len(train), folds=c['folds'], each_row_predicted_once=True,
        original_test_included=False, original_selection_calibration_fixed=True, encoder_pretraining_exposed=True))
    lines = ['# Birth-history truncation down to one frame', '',
        'Fixed test: original 15 births / 11 observed sources. Cross-validation: source/ancestry-grouped readouts within original train only.',
        'Encoders are frozen. Their pretraining saw the CV sources; CV is not an independent end-to-end encoder evaluation.',
        'Each fit uses predictive likelihood selection. AP is diagnostic; the event-enriched population has 20% positives.', '',
        '| Predictor | Frames | Lead (ps) | Fixed-test NLL | Fixed-test AP | OOF NLL | OOF AP |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm in b['arms']:
        for removed in c['remove_frames']:
            f = next(r for r in fixed if r['arm'] == arm['name'] and r['remove_frames'] == removed and r['score'] == 'calibrated')
            v = next(r for r in cv if r['arm'] == arm['name'] and r['remove_frames'] == removed and r['score'] == 'calibrated')
            lines.append(f'| {NAMES[arm["name"]]} | {8-removed} | {(removed+1)*.75:.2f} | {f["nll"]:.4f} | {f["ap"]:.4f} | {v["nll"]:.4f} | {v["ap"]:.4f} |')
    lines.extend(['', '95% source-bootstrap intervals are in the summary tables and figure bands. They condition on fitted models and one split/fit seed.',
                  'Fold standard deviations describe five overlapping refits and evaluation populations; they are not independent standard errors.',
                  'Truncation changes both lead and available history. Appearance is distinct from establishment; this is not natural-risk calibration.', '',
                  '[Fixed test](analyses/fixed-test-comparison-v1/plots/primary-metrics.png) · [Readout CV](analyses/readout-cv-comparison-v1/plots/primary-metrics.png)'])
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    write_json(root / 'technical/comparison-complete.json', dict(dataset_identity=manifest['identity'], fixed_test_treatments=88,
        reused_original_treatments=44, cv_fits=c['folds'] * 88, remaining_frames=list(range(8, 0, -1))))
