"""Pooled ancestry-held-out appearance and fate scores with matched comparisons."""
import numpy as np
from threadpoolctl import threadpool_limits

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .analysis import source_draws, bootstrap_scores
from .data import read
from .fit import scores
from .temporal_analysis import interval, metric_bundle
from . import shooting_appearance_data as data
from .shooting_appearance import FAMILY, tasks, cohort, fit_root


def measured(c, rows, ids, probability, task):
    if task == 'appearance':
        return metric_bundle(dict(rows, source=rows['ancestor']), ids, probability, c)
    y, w = rows['label'][ids], rows['weight'][ids]
    value = scores(y, probability, w)
    _, slot, counts = source_draws(rows['ancestor'][ids], c['bootstrap_draws'], c['seed'])
    boot, valid = bootstrap_scores(y, probability, w, slot, counts, alarm=None)
    value['bootstrap_valid_draws'] = valid
    for key in ('nll', 'brier', 'ap', 'auroc'):
        value[key+'_lo'], value[key+'_hi'] = interval(boot[key])
    return value, boot


def collect(c, design, progress):
    all_rows, features, manifest = data.dataset(c)
    if manifest['identity'] != design['dataset']: raise ValueError('Changed evaluation dataset')
    root = resolve_path(c['output']) / 'analyses/transfer-and-fate-v1'
    (root / 'technical').mkdir(parents=True, exist_ok=True)
    summary, folds, calibration, paired, evidence = [], [], [], [], []
    cached = {}
    with threadpool_limits(limits=c['analysis_threads']):
        for spec in tasks():
            task, arm, obs, pool = spec['task'], spec['arm'], spec['observation']['name'], spec['pool']
            rows, _ = cohort(all_rows, features, task)
            n = len(rows['id'])
            probability, prior = np.full(n, np.nan), np.full(n, np.nan)
            covered = np.zeros(n, int)
            train_scores = []
            for fold in range(5):
                tech = fit_root(c, spec, fold) / 'technical'
                done, split = read(tech / 'complete.json'), read(tech / 'outer-split.json')
                if split['design'] != design['identity'] or sha(tech / 'predictions.npz') != done['predictions_sha256']:
                    raise ValueError('Changed outer predictions')
                with np.load(tech / 'predictions.npz') as saved:
                    global_ids = np.asarray(split['global_indices'])
                    if not np.array_equal(saved['id'], rows['id'][global_ids]):
                        raise ValueError('Changed evaluation row order')
                    held = saved['role'] == 'merged_test'; ids = global_ids[held]
                    if any(design['outer'][g] != fold for g in rows['ancestor'][ids]):
                        raise ValueError('Predictions include training ancestors')
                    probability[ids] = saved['probability'][held]
                    prior[ids] = split['train_prior']; covered[ids] += 1
                evidence.append(dict(task=task, arm=arm, observation=obs, training_pool=pool, fold=fold,
                    predictions_sha256=done['predictions_sha256'], fit_identity=done['identity']))
                for score in done['scores']:
                    role = {'train':'outer_training', 'training_selection_cv':'inner_selection',
                            'merged_test':'outer_test'}[score['role']]
                    folds.append(dict(task=task, training_pool=pool, fold=fold, **dict(score, role=role)))
                    if role == 'outer_training': train_scores.append(score)
            if not np.all(covered == 1) or not np.isfinite(probability).all():
                raise ValueError('Every retained row must have exactly one held-out prediction')
            name = f'{task}-{arm}-{obs}-{pool}'
            np.savez_compressed(root / 'technical' / f'{name}.npz', probability=probability,
                prior=prior, outer_fold=np.asarray([design['outer'][g] for g in rows['ancestor']]), **rows)
            progress.update(analysis=name)
            for evaluation_pool in ('earlier', 'shooting', 'combined'):
                # Fate strata are outcome classes, not case-control cohorts: evaluate them jointly.
                strata = ('combined', 'established', 'failed_all') if task == 'appearance' else ('resolved',)
                for stratum in strata:
                    mask = np.ones(n, bool)
                    if evaluation_pool != 'combined': mask &= rows['pool'] == evaluation_pool
                    if stratum == 'established': mask &= rows['stratum'] == 'established'
                    if stratum == 'failed_all': mask &= rows['stratum'] != 'established'
                    ids = np.flatnonzero(mask)
                    if not len(ids): continue
                    value, boot = measured(c, rows, ids, probability[ids], task)
                    base, base_boot = measured(c, rows, ids, prior[ids], task)
                    w, y = rows['weight'][ids], rows['label'][ids]
                    summary.append(dict(task=task, arm=arm, observation=obs, training_pool=pool,
                        evaluation_pool=evaluation_pool, stratum=stratum, rows=len(ids),
                        events=len(set(zip(rows['source'][ids], rows['event'][ids]))),
                        ancestors=len(set(rows['ancestor'][ids])),
                        mean_train_nll=float(np.mean([s['nll'] for s in train_scores])),
                        mean_train_brier=float(np.mean([s['brier'] for s in train_scores])),
                        prior_nll=base['nll'], prior_brier=base['brier'], prior_ap=base['ap'], **value))
                    cached[(task, arm, obs, pool, evaluation_pool, stratum)] = value, boot
                    for metric in ('nll', 'brier'):
                        lo, hi = interval(boot[metric]-base_boot[metric])
                        paired.append(dict(task=task, contrast='model_minus_training_prior',
                            left=name, right='outer-training-prior', evaluation_pool=evaluation_pool,
                            stratum=stratum, metric=metric, difference=value[metric]-base[metric], lo=lo, hi=hi))
                    bins = np.minimum((probability[ids]*10).astype(int), 9)
                    for k in range(10):
                        chosen = bins == k
                        if chosen.any():
                            calibration.append(dict(task=task, arm=arm, observation=obs, training_pool=pool,
                                evaluation_pool=evaluation_pool, stratum=stratum, bin=k, rows=int(chosen.sum()),
                                predicted_mean=float(np.average(probability[ids][chosen], weights=w[chosen])),
                                positive_fraction=float(np.average(y[chosen], weights=w[chosen]))))
        for left in cached:
            task, arm, obs, pool, evaluation_pool, stratum = left
            comparisons = []
            if obs == 'history4':
                comparisons.append(('history_minus_snapshot', (task, arm, 'snapshot', pool, evaluation_pool, stratum)))
            if task == 'appearance' and pool == 'combined':
                for old_pool in ('earlier', 'shooting'):
                    comparisons.append((f'combined_minus_{old_pool}', (task, arm, obs, old_pool, evaluation_pool, stratum)))
            for contrast, right in comparisons:
                a, aa = cached[left]; b, bb = cached[right]
                for metric in ('nll', 'brier', 'ap', 'auroc'):
                    if a[metric] is None or b[metric] is None: continue
                    lo, hi = interval(aa[metric]-bb[metric])
                    paired.append(dict(task=task, contrast=contrast, left='/'.join(left[:4]),
                        right='/'.join(right[:4]), evaluation_pool=evaluation_pool, stratum=stratum,
                        metric=metric, difference=a[metric]-b[metric], lo=lo, hi=hi))
    for name, table in [('out-of-fold',summary), ('fold-train-and-test',folds),
                        ('calibration',calibration), ('paired-comparisons',paired)]:
        write_metric_rows(table, root, family=FAMILY, name=name)
    lines = ['# Al appearance transfer and nucleus fate prediction', '',
        'Exploratory five-fold ancestry-held-out CV; five inner NLL-selection folds; one seed.',
        'Original coordinates and 442 physical descriptors; four observations at −6, −4.5, −3, −1.5 ps before first local crystal, or the last observation alone.',
        'Earlier and shooting sources use identical input offsets. Label detection retains its native 0.75/0.30 ps cadence and collection-specific follow-up.',
        'Appearance models train on the earlier pool, shooting pool or their union; all predict the same outer holdout.',
        'Fate models use resolved nuclei only: positive means verified dissolution, negative means documented continued growth. Liquid controls and uncertain endings are excluded.',
        'Continued growth does not establish that a nucleus will never dissolve. These retrospective, unequal-follow-up labels are not fixed-horizon survival probabilities.',
        'Train scores are five-fold mean resubstitution errors; outer scores pool exactly-once held-out predictions. Selection scores are not unbiased test estimates.',
        'Intervals resample entire ancestors conditional on fitted outer predictions; they omit seed/refitting uncertainty. AP remains diagnostic.', '',
        '| Task | Model | Input | Train pool | Test pool | Train NLL | Outer NLL | Prior NLL | Outer AP |',
        '|---|---|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['stratum'] not in ('combined','resolved'): continue
        ap = 'undefined' if r['ap'] is None else f'{r["ap"]:.4f}'
        lines.append(f'| {r["task"]} | {r["arm"]} | {r["observation"]} | {r["training_pool"]} | '
            f'{r["evaluation_pool"]} | {r["mean_train_nll"]:.4f} | {r["nll"]:.4f} | {r["prior_nll"]:.4f} | {ap} |')
    lines += ['', '[Full scores](tables/out-of-fold.csv) · [Train and test errors](tables/fold-train-and-test.csv) · '
              '[Paired differences](tables/paired-comparisons.csv) · [Definitions](tables/METRICS.md)', '']
    (root / 'README.md').write_text('\n'.join(lines))
    write_json(root / 'technical/complete.json', dict(design=design['identity'], fits=evidence,
        predictions={p.name:sha(p) for p in (root / 'technical').glob('*.npz')}))
