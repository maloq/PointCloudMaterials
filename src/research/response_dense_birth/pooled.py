"""Birth-label holdout pooling with readout selection confined to training ancestors."""
import argparse
from collections import defaultdict
from datetime import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, resolve_path
from src.research.birth_prediction.fit import scores
from src.research.birth_prediction.analysis import source_draws, bootstrap_scores
from src.research.response_birth import matched
from src.research.equivariant_context.cache import RetainedCache
from .data import read, dataset, cache

FAMILY = 'response_birth_pooled'
METRICS = ['nll', 'brier', 'auroc', 'ap', 'within_pair_auc', 'within_pair_top1']


def root(c):
    return resolve_path(c['output'])


def load(c):
    parent = resolve_path(c['parent'])
    p = read(parent / 'technical/plan.json')
    if p['identity'] != c['parent_identity']:
        raise ValueError('Changed parent cohort')
    rows = dataset(p['config'])
    return p, rows


def prepare(c):
    p, rows = load(c)
    ancestors = sorted(set(rows['ancestor'][rows['role'] == 'train']),
                       key=lambda a: digest([c['seed'], str(a), 'inner-selection']))
    validation = ancestors[:c['inner_selection_ancestors']]
    if not validation or len(validation) >= len(ancestors):
        raise ValueError('Invalid training-only internal split')
    roles = {str(a): str(rows['role'][np.flatnonzero(rows['ancestor'] == a)[0]])
             for a in set(rows['ancestor'])}
    for a, role in roles.items():
        if set(rows['role'][rows['ancestor'] == a]) != {role}:
            raise ValueError('Ancestor crosses original roles')
    partition = dict(inner_selection=validation, inner_fit=ancestors[len(validation):],
                     final_train=ancestors, test=sorted(a for a, r in roles.items() if r != 'train'))
    plan = dict(config=c, parent_identity=p['identity'], parent_plan_sha256=sha(resolve_path(c['parent'])/'technical/plan.json'),
                parent_data=read(cache(p['config'])/'complete.json'), partition=partition,
                implementations=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    plan['identity'] = digest(plan)
    path = root(c)/'technical/plan.json'
    if path.exists() and read(path) != plan:
        raise ValueError('Changed pooled plan; use a new revision')
    write_json(path, plan)
    context = read(resolve_path(c['parent'])/'technical/prediction-context.json')
    context['predictor'].update(selection='NLL on four original training ancestors, followed by refit on all original training ancestors',
                                calibration='none; raw probabilities only', evaluation='original selection + calibration + test; birth labels unused by refits')
    context['historical_exposure'] = 'Frozen encoders retain their label-free response checkpoint selection. Earlier birth results have been inspected; this is exploratory.'
    write_json(root(c)/'technical/prediction-context.json', context)
    return plan


def fit(c, p, rows, x, model, space, lead, kind, *, plan=None):
    if plan is None:
        plan = read(root(c)/'technical/plan.json')
    folder = root(c)/'analyses/fits-v1'/model['name']/space/f'lead-{lead:g}ps'/kind/'technical'
    folder.mkdir(parents=True, exist_ok=True)
    spec = dict(plan=plan['identity'], model=model, space=space, lead_ps=lead, readout=kind)
    identity = digest(spec)
    if (folder/'complete.json').exists():
        done = read(folder/'complete.json')
        if done['identity'] != identity or sha(folder/'predictions.npz') != done['predictions_sha256']:
            raise ValueError('Changed pooled readout')
        return
    ix = {k: np.flatnonzero(np.isin(rows['ancestor'], v)) for k, v in plan['partition'].items()}
    tr, va, final = [ix[k] for k in ['inner_fit', 'inner_selection', 'final_train']]
    if np.intersect1d(final, ix['test']).size or set(np.r_[tr, va]) != set(final):
        raise ValueError('Training/test partition violation')
    y, w = rows['label'], rows['weight']
    recipe = p['config']['readout']
    if kind == 'prior':
        prior = float(np.average(y[final], weights=w[final]))
        probability = np.full(len(y), prior)
        selection = dict(prior=prior)
    elif kind == 'linear':
        def standardize(indices):
            weights = w[indices]/w[indices].sum()
            mean = weights @ x[indices]
            scale = np.sqrt(weights @ (x[indices]-mean)**2).clip(1e-4)
            return (x-mean)/scale, mean, scale
        def train(xx, indices, strength):
            m = LogisticRegression(C=strength, solver='lbfgs', max_iter=3000, tol=1e-5, random_state=c['seed'])
            m.fit(xx[indices], y[indices], sample_weight=w[indices]/w[indices].mean())
            if np.any(m.n_iter_ >= m.max_iter):
                raise RuntimeError('Logistic readout did not converge')
            return m
        history = []
        with threadpool_limits(p['config']['fit_threads']):
            xx, _, _ = standardize(tr)
            for strength in recipe['linear_C']:
                m = train(xx, tr, strength)
                history.append(dict(C=strength, nll=scores(y[va], m.predict_proba(xx[va])[:, 1], w[va])['nll']))
            choice = min(history, key=lambda r: r['nll'])
            xx, mean, scale = standardize(final)
            m = train(xx, final, choice['C'])
            probability = m.predict_proba(xx)[:, 1]
        joblib.dump(dict(model=m, mean=mean, scale=scale), folder/'model.joblib')
        selection = dict(selected=choice, history=history)
    elif kind == 'catboost':
        from catboost import CatBoostClassifier, Pool
        params = dict(recipe['catboost'], loss_function='Logloss', eval_metric='Logloss',
                      random_seed=c['seed'], thread_count=p['config']['fit_threads'], allow_writing_files=False)
        def pool(indices):
            return Pool(x[indices], y[indices], weight=w[indices]/w[indices].mean())
        m = CatBoostClassifier(**params)
        m.fit(pool(tr), eval_set=pool(va), early_stopping_rounds=recipe['patience'], use_best_model=True, verbose=False)
        iterations = m.get_best_iteration()+1
        selection = dict(iterations=iterations, inner_selection_nll=scores(y[va], m.predict_proba(x[va])[:, 1], w[va])['nll'], history=m.get_evals_result())
        m = CatBoostClassifier(**dict(params, iterations=iterations))
        m.fit(pool(final), verbose=False)
        probability = m.predict_proba(x)[:, 1]
        m.save_model(str(folder/'model.cbm'))
    else:
        raise ValueError(kind)
    if not np.isfinite(probability).all():
        raise FloatingPointError('Nonfinite pooled prediction')
    np.savez_compressed(folder/'predictions.npz', probability=probability,
                        **{k: rows[k] for k in ['id', 'role', 'ancestor', 'event', 'pair', 'label', 'weight']})
    write_json(folder/'selection.json', dict(selection, objective='inner-training-ancestor NLL', calibration=None, partition=plan['partition']))
    write_json(folder/'complete.json', dict(identity=identity, spec=spec, predictions_sha256=sha(folder/'predictions.npz')))
    print('fit', model['name'], space, lead, kind, flush=True)


def lane(c, index):
    p, rows = load(c)
    pc = p['config']
    pool = RetainedCache(resolve_path(pc['feature_cache']), 6)
    deadline = read(root(c)/'technical/launch.json')['deadline_unix']
    with recorded_stage(root(c)/'technical'/f'lane-{index}.json', lane=index) as progress:
        for model in p['models'][index::2]:
            metadata = dict(plan=p['identity'], model=model, inference=sha(Path(__file__).parents[1]/'response_birth_infer.py'))
            with pool.lease(digest(metadata), deadline=deadline, metadata=metadata) as bank:
                request = read(resolve_path(c['parent'])/'technical'/('inference-'+model['name']+'.json'))
                if Path(request['destination']) != bank:
                    raise ValueError('Feature producer identity differs from parent')
                if not (bank/'complete.json').exists():
                    req = root(c)/'technical'/('inference-'+model['name']+'.json')
                    write_json(req, request)
                    subprocess.run([sys.executable, '-u', str(Path(__file__).parents[1]/'response_birth_infer.py'), str(req)], check=True)
                done = read(bank/'complete.json')
                if any(sha(bank/n) != h for n, h in done['files'].items()):
                    raise ValueError('Changed inferred features')
                write_json(root(c)/'technical'/('features-'+model['name']+'.json'), dict(path=str(bank), **done))
                for space in pc['spaces']:
                    features = np.load(bank/(space+'.npy'))
                    for j, lead in enumerate(pc['leads_ps']):
                        for kind in pc['readouts']:
                            progress.update(model=model['name'], space=space, lead_ps=lead, readout=kind)
                            fit(c, p, rows, features[rows['indices'][:, j]], model, space, lead, kind)
        if index == 0:
            desc = np.load(cache(pc)/'descriptors.npy')
            for j, lead in enumerate(pc['leads_ps']):
                model = dict(name='rich', arm='rich', seed=c['seed'], velocities=False)
                for kind in pc['readouts']:
                    fit(c, p, rows, desc[rows['indices'][:, j]], model, 'control', lead, kind)
                prior = dict(name='prior', arm='prior', seed=c['seed'], velocities=False)
                fit(c, p, rows, None, prior, 'control', lead, 'prior')


def analyze(c):
    p, all_rows = load(c)
    mask = np.isin(all_rows['role'], ['selection', 'calibration', 'test'])
    rows = {k: v[mask] for k, v in all_rows.items()}
    y, w = rows['label'], rows['weight']
    ancestors, slot, counts = source_draws(rows['ancestor'], c['bootstrap_draws'], c['seed'])
    bw = counts[:, slot]*w[None]
    groups = defaultdict(list)
    individual, receipts = [], []
    for f in sorted((root(c)/'analyses/fits-v1').glob('*/*/*/*/technical/complete.json')):
        done = read(f); spec = done['spec']; path = f.parent/'predictions.npz'
        if sha(path) != done['predictions_sha256']:
            raise ValueError('Changed pooled predictions')
        z = np.load(path)
        np.testing.assert_array_equal(z['id'], all_rows['id'])
        probability = z['probability'][mask]
        vals = scores(y, probability, w)
        boot, _ = bootstrap_scores(y, probability, w, slot, counts, None)
        for name, values in zip(['within_pair_auc', 'within_pair_top1'], matched(rows, probability)):
            vals[name] = float(np.average(values, weights=w)); boot[name] = bw@values/bw.sum(1)
        model = spec['model']; key = (model['arm'], spec['space'], spec['lead_ps'], spec['readout'])
        groups[key].append((vals, boot))
        receipts.append(dict(path=str(path), sha256=done['predictions_sha256']))
        for metric in METRICS:
            lo, hi = np.quantile(boot[metric], [.025, .975])
            individual.append(dict(model=model['name'], arm=model['arm'], space=spec['space'], lead_ps=spec['lead_ps'],
                                   readout=spec['readout'], metric=metric, value=vals[metric], lo=float(lo), hi=float(hi)))
    if len(receipts) != 153:
        raise ValueError(f'Incomplete pooled fits: {len(receipts)}/153')
    summary, means = [], {}
    for key, records in groups.items():
        for metric in METRICS:
            values = np.array([r[0][metric] for r in records]); bs = np.mean([r[1][metric] for r in records], axis=0)
            lo, hi = np.quantile(bs, [.025, .975]); means[key+(metric,)] = (float(values.mean()), bs)
            summary.append(dict(arm=key[0], space=key[1], lead_ps=key[2], readout=key[3], score='raw', metric=metric,
                                value=float(values.mean()), lo=float(lo), hi=float(hi), seed_sd=float(values.std(ddof=1)) if len(values)>1 else None,
                                fitted_seeds=len(values), test_events=len(set(rows['event'])), test_ancestors=len(ancestors), test_rows=len(y)))
    contrasts = []
    for space in p['config']['spaces']:
        for lead in p['config']['leads_ps']:
            for kind in p['config']['readouts']:
                for left, right in [('q_response', 'q_values'), ('qv_response', 'qv_values'), ('qv_response', 'q_response'), ('qv_values', 'q_values')]:
                    for metric in METRICS:
                        a, b = [means[(arm, space, lead, kind, metric)] for arm in [left, right]]
                        lo, hi = np.quantile(a[1]-b[1], [.025, .975])
                        contrasts.append(dict(left=left, right=right, space=space, lead_ps=lead, readout=kind, metric=metric,
                                              difference=a[0]-b[0], lo=float(lo), hi=float(hi)))
    out = root(c)/'analyses/comparison-v1'
    for name, values in [('models', individual), ('summary', summary), ('paired-comparisons', contrasts)]:
        write_metric_rows(values, out, family=FAMILY, name=name)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for ax, metric in zip(axes, ['ap', 'nll', 'brier']):
        for arm in ['q_values', 'q_response', 'qv_values', 'qv_response', 'rich']:
            rr = sorted([r for r in summary if r['arm']==arm and r['space']==('control' if arm=='rich' else 'embedding') and r['readout']=='catboost' and r['metric']==metric], key=lambda r:r['lead_ps'])
            ax.plot([r['lead_ps'] for r in rr], [r['value'] for r in rr], 'o-', label=arm)
        ax.set(xlabel='Lead before 8-atom emergence (ps)', ylabel=metric)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'Pooled birth-label holdout: {len(set(rows["event"]))} births, {len(ancestors)} ancestors')
    (out/'plots').mkdir(exist_ok=True)
    fig.savefig(out/'plots/pooled-scores.png', dpi=180); plt.close(fig)
    lines = ['# Pooled birth-label holdout', '', f'{len(set(rows["event"]))} events, {len(ancestors)} melt ancestors, {len(y)} matched rows.',
             'Original validation, calibration and test are pooled by user request. Readouts are newly selected within original training ancestors and refitted on all original training rows. Raw probabilities; no pooled labels used for selection or calibration.',
             'Frozen encoder checkpoints retain historical label-free selection. Earlier analyses were inspected: exploratory evidence. Neighborhood eligibility and matched controls are unchanged.', '',
             '| Encoder | Space | Lead ps | AP | NLL |', '|---|---|---:|---:|---:|']
    for key in sorted(groups):
        if key[3] == 'catboost':
            lines.append(f'| {key[0]} | {key[1]} | {key[2]:g} | {means[key+("ap",)][0]:.4f} | {means[key+("nll",)][0]:.4f} |')
    lines += ['', 'Mean of three seed scores for neural arms; no prediction ensemble. Descriptor control also sees full-cell component statistics.',
              '[All scores](tables/summary.csv) · [Paired differences](tables/paired-comparisons.csv) · [Definitions](tables/METRICS.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'technical/complete.json', dict(predictions=receipts, test_events=len(set(rows['event'])), test_ancestors=ancestors.tolist()))
    from src.experiment_runner.wandb_tracking import update_recorded_summary
    for model in p['models']:
        values = {f'all010_pooled_v3/{r["space"]}/{r["lead_ps"]}ps/{r["readout"]}/raw/{r["metric"]}':r['value'] for r in individual if r['model']==model['name']}
        update_recorded_summary(resolve_path(p['config']['campaign'])/'analyses'/model['name']/'technical/wandb.json', values,
                                evaluation='all010-pooled-birth-v3', expected=dict(entity='teshbek', project='PointCloudMaterials'))


def run(c):
    children = []
    with recorded_stage(root(c)/'technical/coordinator.json', stage='refitting') as progress:
        try:
            for stage, index in [('audit', 0), ('lane', 0), ('lane', 1)]:
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
                command = [sys.executable, '-u', '-m', __spec__.name, stage, '--config', str(Path('config.json').resolve()), '--lane', str(index)]
                log = (root(c)/'technical'/f'{stage}-{index}.log').open('a')
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                children.append((child, log))
                write_json(root(c)/'technical'/f'{stage}-{index}-launch.json', dict(pid=child.pid, command=command, owner='current response/nucleus-recovery chat'))
            while any(child.poll() is None for child, _ in children):
                if any(child.poll() not in (None, 0) for child, _ in children):
                    raise RuntimeError('Pooled readout or neighborhood audit failed; inspect recorded logs')
                time.sleep(10)
            if any(child.returncode != 0 for child, _ in children):
                raise RuntimeError('Pooled child failed')
            progress.update(stage='analysis'); analyze(c)
        finally:
            for child, log in children:
                if child.poll() is None:
                    child.terminate()
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    child.kill(); child.wait()
                log.close()


def launch(c):
    prepare(c)
    tech = root(c)/'technical'
    if (tech/'launch.json').exists():
        raise FileExistsError(tech/'launch.json')
    bundle = ExecutionBundle.freeze(REPO, tech/'code', c, directories=('src', 'docs/metrics'))
    started = time.time()
    deadline = min(started+c['wall_seconds'], read(resolve_path(c['parent'])/'technical/launch.json')['deadline_unix'])
    command = [sys.executable, '-u', '-m', __spec__.name, 'run', '--config', str(bundle.config_path)]
    receipt = dict(owner='current response/nucleus-recovery chat', host=os.uname().nodename,
                   allocation=os.environ.get('SLURM_JOB_ID'), started_unix=started, deadline_unix=deadline,
                   deadline_Paris=datetime.fromtimestamp(deadline, ZoneInfo('Europe/Paris')).isoformat(), command=command)
    write_json(tech/'launch.json', receipt)
    env = dict(os.environ, PCM_PROJECT_ROOT=str(REPO), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with (tech/'coordinator.log').open('a') as log:
        child = subprocess.Popen(command, cwd=bundle.root, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    receipt.update(pid=child.pid, process_group=child.pid); write_json(tech/'launch.json', receipt)
    print(receipt, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'launch', 'run', 'lane', 'audit', 'analyze'])
    parser.add_argument('--config', required=True); parser.add_argument('--lane', type=int, default=0)
    args = parser.parse_args(); c = read(args.config)
    check_metric_docs(family=FAMILY)
    def stop(signum, frame):
        raise TimeoutError(f'Pooled diagnostic deadline/signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    if args.stage in ['run', 'lane', 'audit']:
        signal.signal(signal.SIGALRM, stop)
        signal.setitimer(signal.ITIMER_REAL, max(.1, read(root(c)/'technical/launch.json')['deadline_unix']-time.time()))
    if args.stage == 'lane':
        lane(c, args.lane)
    elif args.stage == 'audit':
        from .interference import audit
        audit(c)
    else:
        globals()[args.stage](c)


if __name__ == '__main__':
    main()
