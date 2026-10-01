"""Audit retained birth inputs and explain frozen readouts without fitting models."""
import argparse
from collections import defaultdict
import hashlib
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit, logit

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import recorded_stage
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from .analysis import source_draws
from .data import SourceFrames, load, read, ROLES
from .extension import base
from .features import history_packet, load_bank
from .fit import scores

FAMILY = 'birth_prediction_leakage'
FAMILIES = ('geometry', 'bond_order', 'cna', 'tda')


def setup(c):
    ext = read(resolve_path(c['extension_config']))
    b = base(ext)
    positions, rows, manifest = load(b)
    bank, columns = load_bank(b, 'descriptors')
    root = resolve_path(c['output'])
    (root / 'technical').mkdir(parents=True, exist_ok=True)
    return ext, b, positions, rows, manifest, bank, columns, root


def audit(c):
    ext, b, positions, rows, manifest, bank, columns, root = setup(c)
    plan = read(resolve_path(b['cache']) / 'plan.json')
    checks = []

    def check(name, value, evidence):
        checks.append(dict(check=name, passed=bool(value), evidence=evidence))
        write_json(root / 'technical/audit-checks.json', checks)
        if not value:
            raise ValueError(f'Birth leakage audit failed: {name}: {evidence}')

    lineage_roles = defaultdict(set)
    for s in plan['sources']:
        lineage_roles[s['lineage']].add(s['role'])
    check('source_ancestry_roles', all(len(v) == 1 for v in lineage_roles.values()),
          dict(registered_sources=len(plan['sources']), distinct_lineages=len(lineage_roles)))
    pairs = defaultdict(list)
    for i, pair in enumerate(rows['pair']):
        pairs[pair].append(i)
    mismatch = []
    for pair, ids in pairs.items():
        ok = len(ids) == 5 and rows['label'][ids].sum() == 1
        ok &= all(len(set(rows[k][ids])) == 1 for k in
                  ('source', 'role', 'event', 'start_frame', 'end_frame', 'appearance_frame',
                   'birth_frame', 'confirmation_frame', 'weight'))
        if not ok:
            mismatch.append(pair)
    check('matched_case_controls', not mismatch, dict(pairs=len(pairs), mismatches=mismatch,
          implication='source, endpoint, future-event metadata and history length are identical inside each 1:4 set'))
    check('history_precedes_appearance', np.all(rows['end_frame'] + 1 == rows['appearance_frame'])
          and np.all(rows['start_frame'] == rows['end_frame'] - 7), dict(cadence_ps=b['cadence_ps'],
          appearance_leads_ps=[(r + 1) * b['cadence_ps'] for r in c['fixed_remove_frames']]))

    # This is a scientific data-flow intervention: alter audit/label metadata,
    # leaving the consumed observation indices untouched.
    altered = {k: v.copy() for k, v in rows.items()}
    changed = []
    for k in altered:
        if k == 'indices':
            continue
        altered[k] = altered[k][::-1].copy()
        changed.append(k)
    for remove in c['fixed_remove_frames']:
        x = history_packet(bank, rows, remove, np.arange(len(columns)))
        xx = history_packet(bank, altered, remove, np.arange(len(columns)))
        n = 8 - remove
        values = bank[rows['indices'][:, :n]]
        manual = np.concatenate((values.reshape(len(values), -1), values.mean(1), values.std(1),
                                 values[:, -1] - values[:, 0]), axis=1).astype(np.float32)
        check(f'packet_metadata_and_causal_offsets_minus_{remove}',
              np.array_equal(x, xx) and np.array_equal(x[:, :n * len(columns)], manual[:, :n * len(columns)])
              and np.allclose(x, manual, rtol=1e-6, atol=1e-6),
              dict(changed_metadata=changed, frames=n, columns=x.shape[1], latest_offset_before_appearance=remove + 1,
                   summary_rounding_max_absolute_error=float(np.abs(x - manual).max()),
                   numerical_note='Independent contiguous versus indexed float32 reductions; raw observations and metadata intervention are exact.'))

    # Exact geometry/descriptor duplicates crossing source roles are a direct
    # contamination route. Near duplicates are not certified by this check.
    patch_roles = defaultdict(set)
    for role in ROLES:
        for patch in np.unique(rows['indices'][rows['role'] == role]):
            patch_roles[int(patch)].add(role)
    for name, array in [('coordinates', positions), ('descriptors', bank)]:
        hashes = defaultdict(set)
        for i in range(len(array)):
            hashes[hashlib.sha256(np.asarray(array[i]).tobytes()).hexdigest()].update(patch_roles[i])
        overlap = sum(len(v) > 1 for v in hashes.values())
        check(f'exact_{name}_duplicates_across_roles', overlap == 0,
              dict(observed_patches=len(array), unique_hashes=len(hashes), cross_role_duplicates=overlap))

    # Reconstruct every observation from its raw source, tracked atom and frame.
    # PTM is inspected over the full 8-A sphere, not just the 80 consumed atoms.
    raw_counts = dict(sources=0, timelines=0, patches=0, crystalline_input_patches=0,
                      coordinate_mismatches=0, index_mismatches=0, max_coordinate_error_A=0.)
    patch_offset = 0
    for item, receipt in zip(plan['sources'], manifest['sources']):
        folder = resolve_path(b['cache']) / 'sources' / str(item['id'])
        with np.load(folder / 'rows.npz') as a:
            local = {k: a[k] for k in a.files}
        if receipt['source'] != item['id']:
            raise ValueError('Manifest and plan source order changed')
        if not len(local['label']):
            patch_offset += receipt['patches']
            continue
        access = SourceFrames(plan, item)  # checks manifest, shapes, IDs and actual timelines
        raw_counts['sources'] += 1
        raw_counts['timelines'] += 1
        keys = {}
        for i in range(len(local['label'])):
            atom = int(np.searchsorted(access.raw.atom_ids, local['atom'][i]))
            if access.raw.atom_ids[atom] != local['atom'][i]:
                raise ValueError(f'Missing tracked atom {item["id"]}/{local["atom"][i]}')
            for frame, index in zip(range(local['start_frame'][i], local['end_frame'][i] + 1), local['indices'][i]):
                key = (int(frame), atom)
                if int(index) in keys and keys[int(index)] != key:
                    raw_counts['index_mismatches'] += 1
                keys[int(index)] = key
        if set(keys) != set(range(receipt['patches'])):
            raise ValueError(f'Source observation coverage changed: {item["id"]}')
        by_frame = defaultdict(list)
        for index, (frame, atom) in keys.items():
            by_frame[frame].append((index, atom))
        for frame, entries in sorted(by_frame.items()):
            indices = np.array([v[0] for v in entries])
            atoms = np.array([v[1] for v in entries])
            clear = access.clear(frame, atoms, b['radius_A'])
            patch, _ = access.patch(frame, atoms)
            saved = positions[patch_offset + indices]
            error = float(np.abs(patch - saved).max())
            raw_counts['max_coordinate_error_A'] = max(raw_counts['max_coordinate_error_A'], error)
            raw_counts['coordinate_mismatches'] += int(np.count_nonzero(np.any(patch != saved, axis=(1, 2))))
            raw_counts['crystalline_input_patches'] += int(np.count_nonzero(~clear))
            raw_counts['patches'] += len(entries)
        patch_offset += receipt['patches']
        write_json(root / 'technical/raw-audit-progress.json', raw_counts)
        print('raw source', item['id'], raw_counts, flush=True)
    check('raw_timeline_and_geometry', raw_counts['coordinate_mismatches'] == 0 and raw_counts['index_mismatches'] == 0
          and raw_counts['patches'] == len(positions), raw_counts)
    check('no_PTM_crystal_in_observed_sphere', raw_counts['crystalline_input_patches'] == 0, raw_counts)

    from src.research.liquid_predictability.descriptors import patch_descriptors
    errors = []
    for role in ROLES:
        patches = np.unique(rows['indices'][rows['role'] == role])
        chosen = patches[np.linspace(0, len(patches) - 1, c['descriptor_recomputations_per_role'], dtype=int)]
        for patch in chosen:
            actual, names = patch_descriptors(positions[patch])
            error = float(np.abs(actual - bank[patch]).max())
            errors.append(dict(role=role, patch=int(patch), max_absolute_error=error))
            if names != columns or not np.allclose(actual, bank[patch], rtol=1e-6, atol=1e-6):
                raise ValueError(f'Frozen descriptor replay changed: {role}/{patch}/{error}')
    check('descriptor_recomputation', True, dict(patches=len(errors), max_absolute_error=max(e['max_absolute_error'] for e in errors)))
    write_json(root / 'technical/descriptor-replay.json', errors)

    result = dict(dataset_identity=manifest['identity'], checks=checks,
        direct_leakage='No direct label, future-frame, forbidden-covariate or exact cross-role duplicate route found in the audited arrays.',
        limitations=[
            'Case centers were chosen within 8 A of the future established-cluster centroid; this is retrospective site localization.',
            'All endpoints are aligned to future first local PTM appearance; this is not prospective observation-time sampling.',
            'Liquid controls are selected using future survival through case confirmation, at least 6 ps; not all currently liquid sites are represented.',
            'Only sites with an eligible, subsequently confirmed isolated establishment enter positive sampling; unestablished PTM flickers are excluded.',
            'Frozen encoder pretraining includes outer readout-CV sources; fixed original test is the independent encoder-source evaluation.',
            'Feature explanations, permutation results and bootstrap intervals are conditional and exploratory, not causal or corrected for multiple comparisons.',
            'The same original test has been inspected across multiple research choices; new conclusions need independent future confirmation.'])
    write_json(root / 'technical/audit.json', result)
    return result


def feature_groups(columns):
    groups = {f: np.array([i for i, col in enumerate(columns) if col.startswith(f + '/')]) for f in FAMILIES}
    for family, prefix in [('geometry', 'angle'), ('geometry', 'pair_hist'), ('geometry', 'shape'),
                           ('cna', 'fixed32'), ('cna', 'fixed36'), ('cna', 'adaptive12')]:
        groups[f'{family}/{prefix}'] = np.array([i for i, col in enumerate(columns) if col.startswith(f'{family}/{prefix}')])
    groups['geometry/radial'] = np.array([i for i, col in enumerate(columns) if col.startswith('geometry/')
                                        and not any(f'geometry/{p}' in col for p in ('angle', 'pair_hist', 'shape'))])
    for ell in (2, 4, 6, 8):
        groups[f'bond_order/l{ell}'] = np.array([i for i, col in enumerate(columns) if col.startswith(f'bond_order/l{ell}_')])
    for n in (32, 80):
        for h in (0, 1, 2):
            groups[f'tda/n{n}_h{h}'] = np.array([i for i, col in enumerate(columns) if col.startswith(f'tda/n{n}_h{h}_')])
    if any(not len(v) for v in groups.values()):
        raise ValueError('An attribution group has no actual producer columns')
    return groups


def fit_path(ext, b, arm, remove, fold):
    if fold is not None:
        return resolve_path(ext['output']) / f'analyses/cv-v1/fold-{fold}/{arm}/minus-{remove}'
    if remove in b['remove_frames']:
        return resolve_path(b['output']) / f'analyses/classification-v1/{arm}/minus-{remove}'
    return resolve_path(ext['output']) / f'analyses/fixed-test-v1/{arm}/minus-{remove}'


def loss(y, p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return -y * np.log(p) - (1 - y) * np.log1p(-p)


def explain(c):
    from catboost import CatBoostClassifier, Pool
    ext, b, _, rows, manifest, bank, columns, root = setup(c)
    folds = read(resolve_path(ext['output']) / 'technical/folds.json')['binding']['source_folds']
    groups = feature_groups(columns)
    feature_rows, part_rows, replay, provenance = [], [], [], []
    packets = {}
    for scope, removed in [('fixed_test', c['fixed_remove_frames']), ('readout_cv', c['cv_remove_frames'])]:
        eval_ids = np.flatnonzero(rows['role'] == ('test' if scope == 'fixed_test' else 'train'))
        for arm in c['arms']:
            for remove in removed:
                x = history_packet(bank, rows, remove, np.arange(len(columns)))
                n = 8 - remove
                parts = [f'frame_Aminus{8 - k}' for k in range(n)] + ['mean', 'std', 'last_minus_first']
                packet = dict(ids=eval_ids, base_loss=np.empty(len(eval_ids)),
                              probability=np.empty(len(eval_ids)), contribution=np.empty((len(eval_ids), n + 3, len(columns))),
                              deltas={key: np.empty(len(eval_ids)) for key in groups},
                              unrestricted={key: np.empty(len(eval_ids)) for key in FAMILIES})
                for fold in ([None] if scope == 'fixed_test' else range(ext['folds'])):
                    ids = eval_ids if fold is None else eval_ids[np.array([folds[str(int(s))] == fold for s in rows['source'][eval_ids]])]
                    slot = np.searchsorted(eval_ids, ids)
                    path = fit_path(ext, b, arm, remove, fold)
                    tech = path / 'technical'
                    selected = read(tech / 'selection.json')
                    theta = np.array(selected['calibration_parameters'])
                    saved = read(tech / 'complete.json')
                    if sha(tech / 'predictions.npz') != saved['predictions_sha256']:
                        raise ValueError(f'Saved predictions changed: {tech}')
                    with np.load(tech / 'predictions.npz') as a:
                        ii = np.flatnonzero(a['role'] == ('test' if fold is None else 'cv_evaluation'))
                        if not np.array_equal(a['id'][ii], rows['id'][ids]):
                            raise ValueError(f'Frozen readout evaluation identity mismatch: {tech}')
                        expected_raw, expected_cal = a['probability'][ii], a['calibrated'][ii]
                    xx = x[ids]
                    if arm == 'rich_gbdt':
                        model = CatBoostClassifier()
                        model.load_model(str(tech / 'model.cbm'))
                        model_file = tech / 'model.cbm'
                        def predict(values):
                            return model.predict_proba(values, thread_count=c['threads'])[:, 1]
                        shap = model.get_feature_importance(Pool(xx), type='ShapValues', thread_count=c['threads'])
                        raw_margin = model.predict(xx, prediction_type='RawFormulaVal', thread_count=c['threads'])
                        if not np.allclose(shap.sum(1), raw_margin, rtol=1e-7, atol=1e-7):
                            raise ValueError('SHAP does not reproduce the retained model logit')
                        contribution = shap[:, :-1]
                    else:
                        saved_model = joblib.load(tech / 'model.joblib')
                        model_file = tech / 'model.joblib'
                        if not np.array_equal(saved_model['selected_columns'], np.arange(len(columns))):
                            raise ValueError('Rich linear model uses another feature schema')
                        model = saved_model['model']
                        mean, scale = saved_model['mean'], saved_model['scale']
                        def predict(values):
                            return model.predict_proba((values - mean) / scale)[:, 1]
                        contribution = ((xx - mean) / scale) * model.coef_[0]
                    def calibrated(values):
                        return expit(theta[0] * logit(np.clip(predict(values), 1e-6, 1 - 1e-6)) + theta[1])
                    raw, p = predict(xx), calibrated(xx)
                    error = max(float(np.max(np.abs(raw - expected_raw))), float(np.max(np.abs(p - expected_cal))))
                    if error > 2e-6:
                        raise ValueError(f'Frozen readout replay failed: {tech}/{error}')
                    replay.append(dict(scope=scope, arm=arm, remove_frames=remove, fold=fold, max_probability_error=error))
                    provenance.append(dict(path=str(tech), model_sha256=sha(model_file), fitting_identity=saved['identity'],
                                           prediction_sha256=saved['predictions_sha256']))
                    packet['probability'][slot] = p
                    packet['base_loss'][slot] = loss(rows['label'][ids], p)
                    packet['contribution'][slot] = contribution.reshape(len(ids), n + 3, len(columns)) * theta[0]

                    rng = np.random.default_rng(np.random.SeedSequence([c['seed'], remove, 0 if fold is None else fold + 1]))
                    local_pairs = defaultdict(list)
                    for i, key in enumerate(rows['pair'][ids]):
                        local_pairs[key].append(i)
                    permutations = []
                    unrestricted = []
                    for _ in range(c['permutation_repeats']):
                        donor = np.arange(len(ids))
                        for pair_ids in local_pairs.values():
                            donor[pair_ids] = rng.permutation(pair_ids)
                        permutations.append(donor)
                        unrestricted.append(rng.permutation(len(ids)))
                    for key, base_columns in groups.items():
                        selected_columns = np.concatenate([base_columns + block * len(columns) for block in range(n + 3)])
                        changed = np.zeros(len(ids))
                        for donor in permutations:
                            permuted = xx.copy()
                            permuted[:, selected_columns] = xx[donor[:, None], selected_columns]
                            changed += loss(rows['label'][ids], calibrated(permuted)) - packet['base_loss'][slot]
                        packet['deltas'][key][slot] = changed / c['permutation_repeats']
                        if key in FAMILIES:
                            changed = np.zeros(len(ids))
                            for donor in unrestricted:
                                permuted = xx.copy()
                                permuted[:, selected_columns] = xx[donor[:, None], selected_columns]
                                changed += loss(rows['label'][ids], calibrated(permuted)) - packet['base_loss'][slot]
                            packet['unrestricted'][key][slot] = changed / c['permutation_repeats']
                    print('explained', scope, arm, remove, fold, 'replay', error, flush=True)
                    write_json(root / 'technical/explanation-progress.json', replay)
                packets[(scope, arm, remove)] = packet
                weights = rows['weight'][eval_ids]
                weights = weights / weights.sum()
                absolute = np.abs(packet['contribution'])
                per_feature = np.einsum('n,npf->f', weights, absolute)
                per_part = np.einsum('n,npf->p', weights, absolute)
                for i, col in enumerate(columns):
                    feature_rows.append(dict(scope=scope, arm=arm, remove_frames=remove, feature=col,
                        attribution='TreeSHAP absolute calibrated logit' if arm == 'rich_gbdt' else 'absolute calibrated linear contribution relative to fitting mean',
                        mean_absolute_contribution=float(per_feature[i]), share=float(per_feature[i] / per_feature.sum())))
                for label, value in zip(parts, per_part):
                    part_rows.append(dict(scope=scope, arm=arm, remove_frames=remove, part=label,
                                          mean_absolute_contribution=float(value), share=float(value / per_part.sum())))
                np.savez_compressed(root / f'technical/{scope}-{arm}-minus-{remove}-interventions.npz',
                    id=rows['id'][eval_ids], base_loss=packet['base_loss'], probability=packet['probability'],
                    contribution=packet['contribution'],
                    **{f'within_pair/{k}': v for k, v in packet['deltas'].items()},
                    **{f'unrestricted/{k}': v for k, v in packet['unrestricted'].items()})

    summaries, discrimination = [], []
    for (scope, arm, remove), p in packets.items():
        ids = p['ids']
        y, w, source = rows['label'][ids], rows['weight'][ids], rows['source'][ids]
        _, slot, counts = source_draws(source, c['bootstrap_draws'], c['seed'])
        bootw = counts[:, slot] * w[None]
        denom = bootw.sum(1)
        for mode, deltas in [('within_pair', p['deltas']), ('unrestricted', p['unrestricted'])]:
            for key, delta in deltas.items():
                values = bootw @ delta / denom
                low, high = np.quantile(values, [.025, .975])
                summaries.append(dict(scope=scope, arm=arm, remove_frames=remove, permutation=mode, group=key,
                    nll_increase=float(np.average(delta, weights=w)), nll_increase_lo=float(low), nll_increase_hi=float(high),
                    base_features=len(groups[key]), packet_columns=len(groups[key]) * (11 - remove),
                    repeats=c['permutation_repeats'], rows=len(ids), sources=len(np.unique(source))))
        local_pairs = defaultdict(list)
        for i, key in enumerate(rows['pair'][ids]):
            local_pairs[key].append(i)
        wins, top, pair_weights = [], [], []
        for pp in local_pairs.values():
            pos = next(i for i in pp if y[i] == 1)
            negatives = [i for i in pp if y[i] == 0]
            scores_p = p['probability']
            wins.append(np.mean((scores_p[pos] > scores_p[negatives]) + .5 * (scores_p[pos] == scores_p[negatives])))
            maximum = scores_p[pp].max()
            top.append(float(scores_p[pos] == maximum) / np.count_nonzero(scores_p[pp] == maximum))
            pair_weights.append(w[pp].sum())
        discrimination.append(dict(scope=scope, arm=arm, remove_frames=remove,
            **scores(y, p['probability'], w), within_pair_auc=float(np.average(wins, weights=pair_weights)),
            within_pair_top1=float(np.average(top, weights=pair_weights)), pairs=len(local_pairs)))
    for name, values in [('feature-attribution', feature_rows), ('temporal-attribution', part_rows),
                         ('group-permutation', summaries), ('matched-discrimination', discrimination)]:
        write_metric_rows(values, root, family=FAMILY, name=name)
    write_json(root / 'technical/explanation-provenance.json', dict(dataset_identity=manifest['identity'], models=provenance, replay=replay,
        groups={k: [columns[i] for i in v] for k, v in groups.items()}, permutation_repeats=c['permutation_repeats'],
        bootstrap_draws=c['bootstrap_draws'], seed=c['seed'], no_refitting=True))
    return summaries


def plot(c):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = resolve_path(c['output'])
    def table(name):
        return list(csv.DictReader((root / f'tables/{name}.csv').open()))
    permutations = table('group-permutation')
    attribution = table('feature-attribution')
    parts = table('temporal-attribution')
    plots = root / 'plots'
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = dict(geometry='#2878b5', bond_order='#39a269', cna='#e59c23', tda='#8b61a8')
    for scope, ax in zip(['fixed_test', 'readout_cv'], axes):
        selected = {r['group']: r for r in permutations if r['scope'] == scope and r['arm'] == 'rich_gbdt'
                    and r['remove_frames'] == '0' and r['permutation'] == 'within_pair' and r['group'] in FAMILIES}
        values = np.array([float(selected[k]['nll_increase']) for k in FAMILIES])
        low = np.array([float(selected[k]['nll_increase_lo']) for k in FAMILIES])
        high = np.array([float(selected[k]['nll_increase_hi']) for k in FAMILIES])
        ax.bar(FAMILIES, values, color=[colors[k] for k in FAMILIES], alpha=.85)
        ax.errorbar(range(4), values, yerr=np.stack((values - low, high - values)), fmt='none', color='black', capsize=4)
        ax.axhline(0, color='black', linewidth=.7)
        ax.set(title=scope.replace('_', ' ').title() + ' · rich-descriptor boosting', ylabel='NLL increase after shuffling within matched sets')
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('Past geometry only · eight frames · source-bootstrap 95% intervals')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(plots / f'feature-family-reliance.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for scope, ax in zip(['fixed_test', 'readout_cv'], axes):
        selected = sorted([r for r in attribution if r['scope'] == scope and r['arm'] == 'rich_gbdt'
                           and r['remove_frames'] == '0'], key=lambda r: float(r['mean_absolute_contribution']), reverse=True)[:12][::-1]
        ax.barh([r['feature'] for r in selected], [float(r['mean_absolute_contribution']) for r in selected],
                color=[colors[r['feature'].split('/')[0]] for r in selected])
        ax.set(title=scope.replace('_', ' ').title(), xlabel='Mean absolute calibrated-logit TreeSHAP, summed over temporal fields')
    fig.suptitle('Model attribution is descriptive; correlated features can substitute')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(plots / f'top-descriptor-attribution.{suffix}', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for scope, ax in zip(['fixed_test', 'readout_cv'], axes):
        for arm, color in [('rich_gbdt', '#2878b5'), ('rich_linear', '#39a269')]:
            selected = [r for r in parts if r['scope'] == scope and r['arm'] == arm and r['remove_frames'] == '0']
            ax.plot([r['part'] for r in selected], [float(r['share']) for r in selected], 'o-', label=arm, color=color)
        ax.tick_params(axis='x', rotation=60)
        ax.set(title=scope.replace('_', ' ').title(), ylabel='Share of absolute logit attribution')
        ax.legend()
    fig.suptitle('Frame and summary contributions in the retained eight-frame models')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(plots / f'temporal-attribution.{suffix}', dpi=180)
    plt.close(fig)


def supplement(c):
    """Source uncertainty for matched rankings and recorded calibration spread."""
    ext, b, _, rows, _, _, _, root = setup(c)
    results = []
    for scope in ('fixed_test', 'readout_cv'):
        ids = np.flatnonzero(rows['role'] == ('test' if scope == 'fixed_test' else 'train'))
        y, w, source = rows['label'][ids], rows['weight'][ids], rows['source'][ids]
        _, slot, counts = source_draws(source, c['bootstrap_draws'], c['seed'])
        bootw = counts[:, slot] * w[None]
        bootw /= bootw.sum(1, keepdims=True)
        pairs = defaultdict(list)
        for i, pair in enumerate(rows['pair'][ids]):
            pairs[pair].append(i)
        for arm in c['arms']:
            endpoint_auc = []
            for remove in (0, 7):
                path = root / f'technical/{scope}-{arm}-minus-{remove}-interventions.npz'
                with np.load(path) as arrays:
                    if not np.array_equal(arrays['id'], rows['id'][ids]):
                        raise ValueError(f'Matched uncertainty row identity changed: {path}')
                    p = arrays['probability']
                auc, top = np.empty(len(ids)), np.empty(len(ids))
                for pair in pairs.values():
                    case = next(i for i in pair if y[i] == 1)
                    negative = [i for i in pair if y[i] == 0]
                    auc[pair] = np.mean((p[case] > p[negative]) + .5 * (p[case] == p[negative]))
                    top[pair] = float(p[case] == p[pair].max()) / np.count_nonzero(p[pair] == p[pair].max())
                endpoint_auc.append(auc)
                for metric, value in [('within_pair_auc', auc), ('within_pair_top1', top)]:
                    lo, hi = np.quantile(bootw @ value, [.025, .975])
                    results.append(dict(scope=scope, arm=arm, remove_frames=remove, metric=metric,
                        value=float(np.average(value, weights=w)), lo=float(lo), hi=float(hi)))
            delta = endpoint_auc[0] - endpoint_auc[1]
            lo, hi = np.quantile(bootw @ delta, [.025, .975])
            results.append(dict(scope=scope, arm=arm, metric='within_pair_auc_full_minus_one',
                value=float(np.average(delta, weights=w)), lo=float(lo), hi=float(hi)))
    write_json(root / 'technical/matched-uncertainty.json', dict(
        bootstrap=f'{c["bootstrap_draws"]} whole source resamples of fixed row assignments; same draw for each endpoint',
        results=results))
    spread = []
    for arm in ('rich_gbdt', 'rich_linear', 'mace_vicreg_linear'):
        for remove in (0, 7):
            tech = fit_path(ext, b, arm, remove, None) / 'technical'
            selected = read(tech / 'selection.json')
            complete = read(tech / 'complete.json')
            if sha(tech / 'predictions.npz') != complete['predictions_sha256']:
                raise ValueError(f'Changed calibration-spread predictions: {tech}')
            with np.load(tech / 'predictions.npz') as arrays:
                take = arrays['role'] == 'test'
                p, raw, w = arrays['calibrated'][take], arrays['probability'][take], arrays['weight'][take]
                w = w / w.sum()
                spread.append(dict(arm=arm, remove_frames=remove, slope=selected['calibration_parameters'][0],
                    intercept=selected['calibration_parameters'][1], calibrated_sd=float(np.sqrt(w @ (p - (w @ p)) ** 2)),
                    raw_sd=float(np.sqrt(w @ (raw - (w @ raw)) ** 2)),
                    calibrated_quantiles=np.quantile(p, [0, .1, .5, .9, 1]).tolist()))
    write_json(root / 'technical/probability-spread.json', spread)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['audit', 'explain', 'supplement', 'plot', 'all'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    c = read(resolve_path(args.config))
    root = resolve_path(c['output'])
    (root / 'technical').mkdir(parents=True, exist_ok=True)
    write_json(root / 'technical/config.json', c)
    with recorded_stage(root / f'technical/{args.stage}-stage.json') as progress:
        if args.stage in ('audit', 'all'):
            progress.update(audit=audit(c))
        if args.stage in ('explain', 'all'):
            progress.update(group_results=len(explain(c)))
        if args.stage in ('supplement', 'all'):
            supplement(c)
        if args.stage in ('plot', 'all'):
            plot(c)


if __name__ == '__main__':
    main()
