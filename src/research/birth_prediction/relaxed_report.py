"""Paired relaxed/raw comparison and explicit resubstitution-versus-test errors."""
from pathlib import Path

import numpy as np

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .data import read, load
from .temporal import base, fit_root
from .temporal_analysis import predictions, metric_bundle, interval

FAMILY = 'birth_prediction_relaxed'


def compare(c, sc, progress):
    old = resolve_path(c['unrelaxed_comparison'])
    oc = read(old / 'technical/code/config.json')
    _, original_rows, om = load(base(oc))
    _, rows, rm = load(base(sc))
    for name in rows:
        np.testing.assert_array_equal(rows[name], original_rows[name],
            err_msg=f'Relaxation altered matched row field {name}')
    if oc['observations'] != sc['observations'] or oc['arms'] != sc['arms'] or oc['fold_identity'] != sc['fold_identity']:
        raise ValueError('Raw and relaxed experiments have different treatments, arms or folds')
    root = resolve_path(c['output']) / 'analyses/relaxed-versus-md-v1'
    errors, contrasts, provenance = [], [], []
    for arm in ['prior'] + c['arms']:
        observations = ['prior'] if arm == 'prior' else [o['name'] for o in c['observations']]
        for obs in observations:
            progress.update(paired_domain_arm=arm, paired_domain_observation=obs)
            for domain, recipe in [('unrelaxed_md', oc), ('full_cell_relaxed', sc)]:
                folder = fit_root(recipe, 'fixed-test', arm, obs)
                receipt_path = folder / 'technical/complete.json'
                receipt = read(receipt_path)
                if sha(folder / 'technical/predictions.npz') != receipt['predictions_sha256']:
                    raise ValueError(f'Changed retained fit predictions: {folder}')
                provenance.append(dict(path=str(receipt_path), sha256=sha(receipt_path)))
                for score in ['raw', 'calibrated']:
                    train = next(r for r in receipt['scores'] if r['role'] == 'train' and r['score'] == score)
                    test = next(r for r in receipt['scores'] if r['role'] == 'test' and r['score'] == score)
                    errors.append(dict(domain=domain, arm=arm, observation=obs, score=score,
                        train_rows=train['rows'], test_rows=test['rows'],
                        **{f'{role}_{m}': rec[m] for role, rec in [('train', train), ('test', test)]
                           for m in ['nll', 'brier', 'ap', 'auroc']},
                        nll_generalization_gap=test['nll']-train['nll'],
                        brier_generalization_gap=test['brier']-train['brier']))
            for scope in ['fixed_test', 'readout_cv']:
                oi, op, pr = predictions(oc, original_rows, arm, obs, scope)
                ri, rp, rr = predictions(sc, rows, arm, obs, scope)
                np.testing.assert_array_equal(oi, ri)
                provenance.extend(pr + rr)
                for score in ['raw', 'calibrated']:
                    ov, ob = metric_bundle(rows, ri, op[score], c)
                    rv, rb = metric_bundle(rows, ri, rp[score], c)
                    for m in ['nll', 'brier', 'ap', 'auroc', 'matched_auc']:
                        lo, hi = interval(rb[m]-ob[m])
                        contrasts.append(dict(scope=scope, arm=arm, observation=obs, score=score, metric=m,
                            unrelaxed=ov[m], relaxed=rv[m], difference=rv[m]-ov[m],
                            difference_lo=lo, difference_hi=hi))
    write_metric_rows(errors, root, family=FAMILY, name='train-test-errors')
    write_metric_rows(contrasts, root, family=FAMILY, name='paired-domain-differences')
    write_json(root / 'technical/provenance.json', dict(original_identity=om['identity'],
        relaxed_identity=rm['identity'], input_domain=rm['input_domain'], predictions=provenance))
    render(errors, root)
    text = ['# Full-cell relaxation versus original MD inputs', '',
        'All original rows, labels, event weights, source roles, source folds and atom identities are preserved.',
        'Each observed full periodic cell is minimized independently with the generating MEAM potential.',
        'Frozen encoders are unchanged; readouts are refit and selected by validation NLL.', '',
        '**Train is resubstitution error. Test is the fixed source-held-out population.**',
        'These inspected test sources provide exploratory follow-up, not a fresh confirmatory holdout.',
        'Readout CV remains conditional on encoder pretraining exposure. AP is a diagnostic, not a selector.', '',
        '| Input domain | Readout | Observation | Train NLL | Test NLL | Train Brier | Test Brier | Test AP |',
        '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in errors:
        if r['score'] == 'raw' and r['observation'] in ['current', 'history8', 'early', 'mean8', 'prior']:
            text.append(f'| {r["domain"]} | {r["arm"]} | {r["observation"]} | {r["train_nll"]:.5f} | {r["test_nll"]:.5f} | {r["train_brier"]:.5f} | {r["test_brier"]:.5f} | {r["test_ap"]:.4f} |')
    text += ['', 'The CSV contains every treatment with both raw and separately calibrated scores.',
        'Paired source-bootstrap differences use 2,000 shared source draws. Negative NLL/Brier differences favor relaxation.',
        'Relaxation uses the full observed cell, so its computational context is broader than the exported 80-atom patch.',
        'Eligibility and first-crystal labels are from original MD; relaxed configurations are not relabeled or excluded.',
        'Frozen encoders were trained on their original input distributions. This experiment measures inference transfer to quenched geometry; it does not retrain encoders on relaxed data.',
        'Quenches are static inherent configurations, not future MD frames or additional physical elapsed time.']
    (root / 'README.md').write_text('\n'.join(text)+'\n')
    # The queue's top-level report must include the requested train/test errors.
    (resolve_path(c['output']) / 'RESULTS.md').write_text('\n'.join(text)+'\n\nTemporal controls and replay: `analyses/*-comparison-v1/` and `analyses/*-replay-v1/`.\n')


def render(errors, root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plot = root / 'plots'
    plot.mkdir(parents=True, exist_ok=True)
    arms = list(dict.fromkeys(r['arm'] for r in errors if r['arm'] != 'prior'))
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharey=True)
    for arm, ax in zip(arms, axes.flat):
        for domain, color in [('unrelaxed_md', '#636363'), ('full_cell_relaxed', '#1976b9')]:
            for role, style in [('train', '--'), ('test', '-')]:
                values = [next(r for r in errors if r['domain']==domain and r['arm']==arm and
                               r['observation']==o and r['score']=='raw') for o in ['early', 'current', 'mean8', 'history8']]
                ax.plot(range(4), [v[f'{role}_nll'] for v in values], style, marker='o', color=color,
                        label=f'{domain} {role}')
        ax.set_title(arm)
        ax.set_xticks(range(4), ['early', 'current', 'mean8', 'history8'])
        ax.set_ylabel('Binary log loss')
        ax.grid(alpha=.15)
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plot / 'train-test-log-loss.png', dpi=180)
    plt.close(fig)
