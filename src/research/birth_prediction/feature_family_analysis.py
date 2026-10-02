"""Held-out family reliance, paired ablation scores and scientific figures."""
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .analysis import predictions
from .data import load, read
from .feature_families import FAMILY, FAMILIES, MODELS, domain_config, fit_path, variants
from .features import load_bank
from .leakage import feature_groups, loss
from .temporal import partition
from .temporal_analysis import interval, matched, metric_bundle, weights_boot


def pooled(c, domain, model, variant, rows, scope):
    ids = np.flatnonzero(rows['role'] == ('test' if scope == 'fixed_test' else 'train'))
    output = {k: np.empty(len(ids)) for k in ('raw', 'calibrated')}
    covered = np.zeros(len(ids), int)
    receipts = []
    for index in ([0] if scope == 'fixed_test' else range(1, c['folds']+1)):
        _, split, role = partition(c, rows, index)
        local = split[role]
        slot = np.searchsorted(ids, local)
        data, receipt = predictions(fit_path(c, domain, model, variant, index), rows['id'][local], role)
        for key in output:
            output[key][slot] = data[key]
        receipts.append(receipt)
        covered[slot] += 1
    if not np.all(covered == 1):
        raise ValueError(f'Not one held-out prediction per row: {domain}/{model}/{variant}/{scope}')
    return ids, output, receipts


def explain(c, progress):
    from catboost import CatBoostClassifier
    root = resolve_path(c['output']) / 'analyses/permutation-v1'
    (root / 'technical').mkdir(parents=True, exist_ok=True)
    summary, receipts = [], []
    with threadpool_limits(limits=c['analysis_threads']):
        for domain in c['domains']:
            b = domain_config(c, domain)
            _, rows, _ = load(b)
            bank, columns = load_bank(b, 'descriptors')
            x = bank[rows['indices'][:, 7]]
            groups = feature_groups(columns)
            groups = {name: groups[name] for name in c['permutation_groups']}
            for model in MODELS:
                for scope in ('fixed_test', 'readout_cv'):
                    ids, expected, _ = pooled(c, domain, model, 'full', rows, scope)
                    delta = {key: np.empty((len(groups), len(ids))) for key in expected}
                    for index in ([0] if scope == 'fixed_test' else range(1, c['folds']+1)):
                        progress.update(domain=domain['name'], model=model, scope=scope, permutation_fold=index-1)
                        _, split, role = partition(c, rows, index)
                        local = split[role]
                        slot = np.searchsorted(ids, local)
                        tech = fit_path(c, domain, model, 'full', index) / 'technical'
                        theta = read(tech / 'selection.json')['calibration_parameters']
                        if model == 'linear':
                            file = tech / 'model.joblib'
                            saved = joblib.load(file)
                            if not np.array_equal(saved['selected_columns'], np.arange(len(columns))):
                                raise ValueError(f'Changed full-model columns: {file}')
                            def predict(values):
                                return saved['model'].predict_proba((values-saved['mean'])/saved['scale'])[:, 1]
                        else:
                            file = tech / 'model.cbm'
                            predictor = CatBoostClassifier()
                            predictor.load_model(str(file))
                            def predict(values):
                                return predictor.predict_proba(values, thread_count=c['analysis_threads'])[:, 1]
                        def probabilities(values):
                            raw = predict(values)
                            return dict(raw=raw, calibrated=expit(theta[0]*logit(np.clip(raw, 1e-6, 1-1e-6))+theta[1]))
                        xx, y = x[local], rows['label'][local]
                        baseline = probabilities(xx)
                        error = max(float(np.max(np.abs(baseline[k]-expected[k][slot]))) for k in expected)
                        if error > 2e-6:
                            raise ValueError(f'Frozen full-feature replay differs: {file}: {error}')
                        _, _, members = matched(rows, local)
                        for member in members:
                            if len(np.unique(rows['source'][local[member]])) != 1:
                                raise ValueError('Permutation match crosses source')
                        rng = np.random.default_rng(np.random.SeedSequence([c['seed'], index]))
                        donors = []
                        for _ in range(c['permutation_repeats']):
                            donor = np.arange(len(local))
                            for member in members:
                                donor[member] = rng.permutation(member)
                            donors.append(donor)
                        replay = dict(domain=domain['name'], model=model, scope=scope, fold=index-1,
                            path=str(file), model_sha256=sha(file), replay_error=error)
                        receipts.append(replay)
                        detail = dict(id=rows['id'][local], donors=np.asarray(donors),
                                      group_names=np.asarray(list(groups)))
                        for group_index, (group, selected) in enumerate(groups.items()):
                            progress.update(group=group)
                            effects = {key: [] for key in expected}
                            for donor in donors:
                                changed = xx.copy()
                                changed[:, selected] = xx[donor[:, None], selected]
                                pp = probabilities(changed)
                                for key in effects:
                                    effects[key].append(loss(y, pp[key])-loss(y, baseline[key]))
                            for key, effects_list in effects.items():
                                effects_array = np.asarray(effects_list)
                                delta[key][group_index, slot] = effects_array.mean(0)
                                detail[f'{key}/{group}'] = effects_array
                        np.savez_compressed(root / f'technical/{domain["name"]}-{model}-{scope}-{index}.npz', **detail)
                    w, ww, _, _ = weights_boot(rows, ids, c)
                    for key in delta:
                        for group_index, (group, selected) in enumerate(groups.items()):
                            values = delta[key][group_index]
                            lo, hi = interval(ww @ values)
                            summary.append(dict(domain=domain['name'], model=model, scope=scope, score=key,
                                group=group, features=len(selected), rows=len(ids), repeats=c['permutation_repeats'],
                                nll_increase=float(w @ values), difference_lo=lo, difference_hi=hi))
                    write_json(root / 'technical/progress.json', dict(models=receipts, completed_summaries=len(summary)))
    write_metric_rows(summary, root, family=FAMILY, name='matched-family-permutation')
    write_json(root / 'technical/provenance.json', dict(models=receipts,
        groups={name: [columns[i] for i in selected] for name, selected in groups.items()},
        seed=c['seed'], repeats=c['permutation_repeats'], conditioning='source/time matched sets, not other descriptors'))
    write_json(root / 'technical/complete.json', dict(rows=len(summary), config=c))


def collect(c, progress):
    root = resolve_path(c['output']) / 'analyses/family-comparison-v1'
    summary, all_scores, contrasts, provenance = [], [], [], []
    measured = {}
    with threadpool_limits(limits=c['analysis_threads']):
        for domain in c['domains']:
            _, rows, _ = load(domain_config(c, domain))
            for model in MODELS:
                for variant in variants():
                    name = variant['name']
                    progress.update(domain=domain['name'], model=model, variant=name)
                    for index in range(c['folds']+1):
                        path = fit_path(c, domain, model, name, index)
                        for record in read(path / 'technical/complete.json')['scores']:
                            all_scores.append(dict(domain=domain['name'], model=model, variant=name,
                                scope='fixed-test' if index == 0 else f'cv/fold-{index-1}',
                                reused_full_reference=name == 'full', **record))
                    for scope in ('fixed_test', 'readout_cv'):
                        ids, pp, receipts = pooled(c, domain, model, name, rows, scope)
                        provenance.extend(receipts)
                        for score, probabilities in pp.items():
                            values, boot = metric_bundle(rows, ids, probabilities, c)
                            measured[(domain['name'], model, name, scope, score)] = (values, boot)
                            summary.append(dict(domain=domain['name'], model=model, variant=name, scope=scope,
                                score=score, rows=len(ids), sources=len(np.unique(rows['source'][ids])),
                                appearance_lead_ps=.75, **values))
                for scope in ('fixed_test', 'readout_cv'):
                    for score in ('raw', 'calibrated'):
                        baseline, baseline_boot = measured[(domain['name'], model, 'full', scope, score)]
                        for variant in variants()[1:]:
                            value, boot = measured[(domain['name'], model, variant['name'], scope, score)]
                            for metric in ('nll', 'brier', 'ap', 'auroc', 'matched_auc'):
                                lo, hi = interval(boot[metric]-baseline_boot[metric])
                                contrasts.append(dict(domain=domain['name'], model=model, scope=scope, score=score,
                                    variant=variant['name'], reference='full', metric=metric,
                                    difference=value[metric]-baseline[metric], difference_lo=lo, difference_hi=hi))
    domain_effect = []
    original, relaxed = [d['name'] for d in c['domains']]
    for model in MODELS:
        for variant in variants():
            for scope in ('fixed_test', 'readout_cv'):
                for score in ('raw', 'calibrated'):
                    a, aa = measured[(original, model, variant['name'], scope, score)]
                    b, bb = measured[(relaxed, model, variant['name'], scope, score)]
                    for metric in ('nll', 'brier', 'ap', 'auroc', 'matched_auc'):
                        lo, hi = interval(bb[metric]-aa[metric])
                        domain_effect.append(dict(model=model, variant=variant['name'], scope=scope, score=score,
                            metric=metric, reference=original, domain=relaxed,
                            difference=b[metric]-a[metric], difference_lo=lo, difference_hi=hi))
    for name, records in [('heldout-scores', summary), ('train-selection-calibration-test', all_scores),
                          ('paired-vs-full', contrasts), ('relaxed-minus-original', domain_effect)]:
        write_metric_rows(records, root, family=FAMILY, name=name)
    write_json(root / 'technical/provenance.json', dict(predictions=provenance, config=c))
    import csv
    permutation_root = resolve_path(c['output']) / 'analyses/permutation-v1'
    done = read(permutation_root / 'technical/complete.json')
    if done['config'] != c:
        raise ValueError('Permutation and refitting configuration differ')
    with (permutation_root / 'tables/matched-family-permutation.csv').open() as handle:
        permutations = list(csv.DictReader(handle))
    render(c, root, summary, contrasts, permutations)
    write_json(root / 'technical/complete.json', dict(config=c, summary_rows=len(summary),
        fit_scores=len(all_scores), reused_fits=24, new_fits=192))


def render(c, root, summary, contrasts, permutations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots = root / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    for scope in ('fixed_test', 'readout_cv'):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
        for row, domain in enumerate(c['domains']):
            for col, model in enumerate(MODELS):
                ax = axes[row, col]
                for offset, mode, label in [(-.18, 'without', 'Refit without family'), (.0, 'only', 'Refit only family'),
                                            (.18, 'shuffle', 'Shuffle in full model')]:
                    points, low, high = [], [], []
                    for family in FAMILIES:
                        if mode == 'shuffle':
                            r = next(r for r in permutations if r['domain'] == domain['name'] and r['model'] == model
                                and r['scope'] == scope and r['score'] == 'raw' and r['group'] == family)
                            value = float(r['nll_increase'])
                        else:
                            r = next(r for r in contrasts if r['domain'] == domain['name'] and r['model'] == model
                                and r['scope'] == scope and r['score'] == 'raw' and r['variant'] == f'{mode}_{family}'
                                and r['metric'] == 'nll')
                            value = r['difference']
                        points.append(value); low.append(float(r['difference_lo'])); high.append(float(r['difference_hi']))
                    yy = np.arange(len(FAMILIES))+offset
                    ax.hlines(yy, low, high, linewidth=1)
                    ax.plot(points, yy, 'o', label=label, markersize=4)
                ax.axvline(0, color='gray', linewidth=.8)
                ax.set(yticks=np.arange(len(FAMILIES)), yticklabels=FAMILIES,
                       title=f'{domain["name"]} · {model}', xlabel='NLL difference from full model (higher = worse)')
                ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(scope+' · paired 95% source intervals; raw probability scores')
        fig.tight_layout()
        for suffix in ('png', 'pdf'):
            fig.savefig(plots / f'{scope}-family-information.{suffix}', dpi=180)
        plt.close(fig)
    lines = ['# Physical descriptor families in birth-site prediction', '',
        'Original and full-cell-relaxed snapshots, 0.75 ps before first local PTM appearance.',
        '192 refits and 24 reused full-feature models; one seed, fixed source roles and five source folds.',
        'Selection uses validation NLL. AP is diagnostic. Probabilities concern the matched 20% case population.', '',
        '| Domain | Model | Features | Test NLL | CV NLL | Test AP | CV AP |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: |']
    for domain in c['domains']:
        for model in MODELS:
            for variant in variants():
                rr = [r for r in summary if r['domain'] == domain['name'] and r['model'] == model
                      and r['variant'] == variant['name'] and r['score'] == 'raw']
                test = next(r for r in rr if r['scope'] == 'fixed_test')
                cv = next(r for r in rr if r['scope'] == 'readout_cv')
                lines.append(f'| {domain["name"]} | {model} | {variant["name"]} | {test["nll"]:.5f} | '
                             f'{cv["nll"]:.5f} | {test["ap"]:.3f} | {cv["ap"]:.3f} |')
    lines += ['', 'Both raw/calibrated scores and all training/selection/calibration errors are in tables/.',
        'Permutation conditions on matched source/time sets, not all other descriptors. Effects are not additive or causal.',
        'Intervals condition on fitted models and fixed permutations; no seed uncertainty or multiplicity correction.',
        'The inspected fixed test has 15 births in 11 sources and is exploratory.',
        'Full-cell relaxation supplies broader computational context than the 80-atom descriptor input.',
        'The descriptor producer retains its original radial sorting, hard 8 A cutoff and neighbor definitions.', '',
        '[Test figure](plots/fixed_test-family-information.png) · [CV figure](plots/readout_cv-family-information.png)']
    (root / 'README.md').write_text('\n'.join(lines)+'\n')
    (resolve_path(c['output']) / 'RESULTS.md').write_text('\n'.join(lines[:6])+'\n\n'
        '[Full report](analyses/family-comparison-v1/README.md)\n')
