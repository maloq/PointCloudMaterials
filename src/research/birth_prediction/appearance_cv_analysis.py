"""Exactly-once outer predictions and paired ancestry uncertainty for appearance CV."""
import numpy as np
from threadpoolctl import threadpool_limits

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .appearance import inputs, specs
from .appearance_cv import FAMILY, STRATA, feature_identity, fit_root, row_identity
from .data import read
from .temporal_analysis import interval, metric_bundle


def collect(c, design, domain, progress):
    root = resolve_path(c['output']) / f'analyses/{domain}-cv-v1'
    (root / 'technical').mkdir(parents=True, exist_ok=True)
    summary, folds, paired, calibration, evidence = [], [], [], [], []
    cached = {}
    with threadpool_limits(limits=c['analysis_threads']):
        for spec in specs(design['appearance']):
            if spec['domain'] != domain:
                continue
            _, rows, x, _, _, _ = inputs(design['appearance'], spec)
            if row_identity(rows) != design['rows_identity']:
                raise ValueError('Changed outer evaluation cohort')
            features = read(resolve_path(c['output']) / f'technical/inputs-{domain}-{spec["observation"]["name"]}.json')
            if feature_identity(x) != features:
                raise ValueError('Changed outer evaluation features')
            del x
            ancestors = np.asarray([design['source_ancestry'][str(s)] for s in rows['source']])
            # Resample complete ancestors; retain original source IDs in saved predictions.
            bootstrap_rows = dict(rows, source=ancestors)
            for pool in ('original', 'combined'):
                progress.update(analysis=domain, arm=spec['arm'], pool=pool,
                                observation=spec['observation']['name'])
                probability = np.full(len(rows['id']), np.nan)
                thresholds = np.full(len(probability), np.nan)
                covered = np.zeros(len(probability), int)
                train_scores = []
                for fold in range(c['outer_folds']):
                    tech = fit_root(c, domain, spec, pool, fold) / 'technical'
                    done, split = read(tech / 'complete.json'), read(tech / 'outer-split.json')
                    if split['design'] != design['identity'] or sha(tech / 'predictions.npz') != done['predictions_sha256']:
                        raise ValueError(f'Changed outer fit: {tech}')
                    selection = read(tech / 'selection.json')
                    with np.load(tech / 'predictions.npz') as saved:
                        global_ids = np.asarray(split['global_indices'])
                        if not np.array_equal(saved['id'], rows['id'][global_ids]):
                            raise ValueError(f'Outer prediction row ordering changed: {tech}')
                        held = saved['role'] == 'merged_test'
                        ids = global_ids[held]
                        if any(design['outer'][str(g)] != fold for g in ancestors[ids]):
                            raise ValueError('Prediction is not on its declared outer holdout')
                        probability[ids] = saved['probability'][held]
                        thresholds[ids] = selection['threshold_training_oof_fpr05']
                        covered[ids] += 1
                    evidence.append(dict(arm=spec['arm'], observation=spec['observation']['name'],
                        training_pool=pool, outer_fold=fold, path=str(tech),
                        predictions_sha256=done['predictions_sha256'], fit_identity=done['identity']))
                    for score in done['scores']:
                        role = split['role_mapping'][score['role']]
                        folds.append(dict(domain=domain, training_pool=pool, outer_fold=fold,
                            **dict(score, role=role)))
                        if role == 'outer_training':
                            train_scores.append(score)
                if not np.all(covered == 1) or not np.isfinite(probability).all():
                    raise ValueError('Require exactly one outer-held-out prediction per retained row')
                name = f'{spec["arm"]}-{spec["observation"]["name"]}-{pool}'
                np.savez_compressed(root / 'technical' / f'{name}.npz', probability=probability,
                    threshold=thresholds, ancestor=ancestors,
                    outer_fold=np.asarray([design['outer'][str(g)] for g in ancestors]),
                    **{k: rows[k] for k in ('id', 'source', 'event', 'pair', 'role', 'label', 'weight', 'stratum')})
                for stratum in (*STRATA, 'failed_all', 'combined'):
                    mask = (np.ones(len(probability), bool) if stratum == 'combined' else
                            rows['stratum'] != 'established' if stratum == 'failed_all' else rows['stratum'] == stratum)
                    ids = np.flatnonzero(mask)
                    value, boot = metric_bundle(bootstrap_rows, ids, probability[ids], c)
                    positive = rows['label'][ids] == 1
                    negative = ~positive
                    weights = rows['weight'][ids]
                    summary.append(dict(domain=domain, arm=spec['arm'], observation=spec['observation']['name'],
                        training_pool=pool, stratum=stratum, rows=len(ids),
                        events=len(set(zip(rows['source'][ids][positive], rows['event'][ids][positive]))),
                        sources=len(set(rows['source'][ids])), ancestors=len(set(ancestors[ids])),
                        mean_outer_train_nll=float(np.mean([s['nll'] for s in train_scores])),
                        mean_outer_train_ap=float(np.mean([s['ap'] for s in train_scores])),
                        recall_at_inner_oof_fpr05=float(np.average(probability[ids][positive] > thresholds[ids][positive], weights=weights[positive])),
                        observed_fpr=float(np.average(probability[ids][negative] > thresholds[ids][negative], weights=weights[negative])),
                        **value))
                    cached[(spec['arm'], spec['observation']['name'], pool, stratum)] = (value, boot)
                    bins = np.minimum((probability[ids] * 10).astype(int), 9)
                    for k in range(10):
                        chosen = bins == k
                        if chosen.any():
                            calibration.append(dict(domain=domain, arm=spec['arm'],
                                observation=spec['observation']['name'], training_pool=pool, stratum=stratum,
                                bin=k, rows=int(chosen.sum()),
                                predicted_mean=float(np.average(probability[ids][chosen], weights=weights[chosen])),
                                positive_fraction=float(np.average(positive[chosen], weights=weights[chosen]))))
        contrasts = []
        for arm in ('rich_linear', 'rich_gbdt'):
            for stratum in (*STRATA, 'failed_all', 'combined'):
                for obs in ('current', 'history8'):
                    contrasts.append(('combined_minus_original', (arm, obs, 'combined', stratum),
                                      (arm, obs, 'original', stratum)))
                for pool in ('original', 'combined'):
                    contrasts.append(('history_minus_snapshot', (arm, 'history8', pool, stratum),
                                      (arm, 'current', pool, stratum)))
        for contrast, left, right in contrasts:
            a, aa = cached[left]; b, bb = cached[right]
            for metric in ('nll', 'brier', 'ap', 'auroc', 'matched_auc'):
                lo, hi = interval(aa[metric] - bb[metric])
                paired.append(dict(domain=domain, contrast=contrast, left='/'.join(left), right='/'.join(right),
                    metric=metric, difference=a[metric] - b[metric], lo=lo, hi=hi))
    for name, table in [('out-of-fold', summary), ('fold-train-and-test', folds),
                        ('paired-comparisons', paired), ('calibration', calibration)]:
        write_metric_rows(table, root, family=FAMILY, name=name)
    write_json(root / 'technical/provenance.json', dict(design=design['identity'], fits=evidence))
    lines = [f'# Appearance prediction: {domain} nested CV', '',
        'Five outer ancestry folds; five inner NLL-selection folds; one seed. All original roles enter this explicitly separate exploratory CV protocol.',
        'Every retained row has exactly one outer-held-out prediction. The older fixed-test comparison remains unchanged.',
        'Original-pool models are refitted on established births only within each outer training fold; combined models also fit failed appearances.',
        'Intervals resample whole ancestors from the fixed outer predictions. They do not include refitting/seed uncertainty or make previously inspected data a fresh confirmation set.',
        'All 18 eligible stronger failed events are evaluated; 28 detected strong events did not all pass the crystal-free history screen.',
        'Case/control prevalence is 20%, not natural nucleation incidence. Inner CV scores are selection diagnostics.', '',
        '| Model | Input | Training pool | Population | Events | Train NLL (fold mean) | Outer NLL | Outer AP |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for r in summary:
        if r['stratum'] in ('failed_strong', 'combined'):
            lines.append(f'| {r["arm"]} | {r["observation"]} | {r["training_pool"]} | {r["stratum"]} | '
                f'{r["events"]} | {r["mean_outer_train_nll"]:.4f} | {r["nll"]:.4f} | {r["ap"]:.4f} |')
    lines += ['', '[Full scores and intervals](tables/out-of-fold.csv) · [Per-fold errors](tables/fold-train-and-test.csv) · '
              '[Paired changes](tables/paired-comparisons.csv) · [Definitions](tables/METRICS.md)', '']
    (root / 'README.md').write_text('\n'.join(lines))
    write_json(root / 'technical/complete.json', dict(design=design['identity'], domain=domain, fits=len(evidence),
        predictions={p.name: sha(p) for p in (root / 'technical').glob('*.npz')}))
