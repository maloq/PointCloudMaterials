"""Frozen nonlinear birth probes; fixed budgets and train/test ancestry only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from catboost import CatBoostClassifier, Pool
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.research.birth_prediction.fit import scores
from src.research.birth_prediction.analysis import source_draws, bootstrap_scores
from src.research.equivariant_context.cache import RetainedCache
from src.research.liquid_predictability.descriptors import patch_descriptor_batch
from src.research.measurement_sim.features import SmoothMeasurements
from .data import read, root, cache, sha, digest, write_json, atomic_save
from .model import PhaseProcess
from .birth_data import prepare, collect, FAMILY


def invariant(z, state):
    """Central, global, and three radial-shell means of channel Gram matrices."""
    a, b = torch.triu_indices(z.shape[-2], z.shape[-2], device=z.device)
    gram = (z @ z.transpose(-1, -2))[..., a, b]
    radius = state[:, 1:, :3].norm(dim=-1)*5
    weights = torch.exp(-.5*((radius[..., None]-radius.new_tensor([2.5, 3.5, 4.5]))/.45).square())
    weights *= (1-radius[..., None]/5.5).clamp_min(0).square()
    shells = weights.transpose(1, 2) @ gram[:, 1:]
    shells /= weights.sum(1)[..., None]+1e-12
    return torch.cat((gram[:, 0], gram[:, 1:].mean(1), shells.flatten(1)), -1)


def measured(c, p, data):
    path = cache(c)/'measurements.pt'
    if path.exists() and (cache(c)/'measurements.json').exists():
        if sha(path) != read(cache(c)/'measurements.json')['sha256']:
            raise ValueError('Changed measurement controls')
        return torch.load(path, weights_only=False, map_location='cpu').numpy()
    smooth = SmoothMeasurements().cuda(); values = []
    for offset in range(0, len(data['state']), c['batch_size']):
        x = data['state'][offset:offset+c['batch_size']]
        with torch.no_grad():
            other = smooth(x.cuda()).cpu().numpy()
        geometry, names = patch_descriptor_batch(
            x[:, :, :3].double().numpy()*p['phase_config']['length_factor'],
            families=('bond_order', 'tda'), alpha_backend='native', alpha_precision='safe',
            threads=c['threads'], selected_columns=p['columns'][:153])
        if names != p['columns'][:153]:
            raise ValueError('Changed BO/TDA feature order')
        values.append(np.concatenate((geometry, other), -1).astype('float32'))
        write_json(root(c)/'technical/progress.json', dict(stage='measurements', completed=offset+len(x), total=len(data['state'])))
    value = np.concatenate(values)
    if not np.isfinite(value).all():
        raise FloatingPointError('Nonfinite structural/velocity controls')
    atomic_save(path, torch.from_numpy(value))
    write_json(cache(c)/'measurements.json', dict(sha256=sha(path), columns=p['columns']))
    return value


def features(c, p, data, spec):
    model = PhaseProcess(p['phase_config'], spec['arm']).cuda().eval()
    if sha(spec['path']) != spec['sha256']:
        raise ValueError('Changed frozen phase checkpoint')
    saved = torch.load(spec['path'], map_location='cuda', weights_only=False)
    model.load_state_dict(saved['model'])
    scale = torch.tensor([5., 5., 5., 10., 10., 10.], device='cuda')
    initial = []; path = []
    with torch.no_grad():
        for block in data['state'].split(c['batch_size']):
            x = block.cuda()/scale; z = model.encode(x)
            first = invariant(z, x); values = [first]
            for step in range(1, 21):
                z = model.step(z)
                if step in [4, 10, 20]:
                    values.append(invariant(z, x)-first)
            initial.append(first.cpu()); path.append(torch.cat(values, -1).cpu())
    result = dict(phi=torch.cat(initial).numpy(), path=torch.cat(path).numpy())
    if not all(np.isfinite(v).all() for v in result.values()):
        raise FloatingPointError('Nonfinite frozen neural birth features')
    return result


def fit_one(c, p, data, feature, name, space, task, lead, *, family=FAMILY):
    out = root(c)/'analyses'/name/space/task/f'lead-{lead:g}ps'; tech = out/'technical'
    if (tech/'complete.json').exists():
        done = read(tech/'complete.json')
        if sha(tech/'predictions.npz') != done['sha256']:
            raise ValueError('Changed completed probe predictions')
        return
    tech.mkdir(parents=True, exist_ok=True)
    selected = [i for i, r in enumerate(data['rows']) if r['task'] == task and r['lead'] == lead]
    records = [data['rows'][i] for i in selected]
    rows = {k: np.asarray([r[k] for r in records]) for k in ['id', 'role', 'original_role', 'catalogue_role', 'ancestor', 'event', 'label', 'weight']}
    tr = rows['role'] == 'train'; te = ~tr; y = rows['label']; w = rows['weight']
    if set(rows['ancestor'][tr]) & set(rows['ancestor'][te]):
        raise ValueError('Cross-role ancestry')
    if len(np.unique(y[tr])) != 2 or len(np.unique(y[te])) != 2:
        raise ValueError('Both classes are required for this declared assay')
    started = time.monotonic()
    if space == 'prior':
        probability = np.full(len(y), np.average(y[tr], weights=w[tr]))
        dimension = 0
    else:
        x = feature[data['indices'][selected]]; dimension = x.shape[1]
        m = CatBoostClassifier(**c['probe'], loss_function='Logloss', random_seed=c['seed'],
                               thread_count=c['fit_threads'], allow_writing_files=False)
        # No eval_set, inner split, early stopping, class reweighting or AP selector.
        m.fit(Pool(x[tr], y[tr], weight=w[tr]/w[tr].mean()), verbose=False)
        m.save_model(str(tech/'model.cbm'))
        probability = m.predict_proba(x)[:, 1]
    np.savez_compressed(tech/'predictions.npz', probability=probability, **rows)
    ancestors, slots, draws = source_draws(rows['ancestor'][te], c['bootstrap'], c['seed'])
    values = scores(y[te], probability[te], w[te])
    boot, valid = bootstrap_scores(y[te], probability[te], w[te], slots, draws, None)
    report = []
    for metric in ['ap', 'auroc', 'nll', 'brier']:
        lo, hi = np.quantile(boot[metric], [.025, .975])
        report.append(dict(model=name, space=space, task=task, lead_ps=lead, metric=metric,
            value=values[metric], lo=lo, hi=hi, dimension=dimension, test_rows=int(te.sum()),
            positive_rows=int(y[te].sum()), events=len(set(rows['event'][te])), ancestors=len(ancestors),
            prevalence=values['prevalence'], bootstrap_valid=valid))
    write_metric_rows(report, out, family=family, name='scores')
    # Descriptive reliability bins only; never fit a calibration transform.
    calibration = []
    bins = np.minimum((probability[te]*10).astype(int), 9)
    for k in range(10):
        take = bins == k
        if take.any():
            calibration.append(dict(bin=k, rows=int(take.sum()), weight=float(w[te][take].sum()),
                mean_probability=float(np.average(probability[te][take], weights=w[te][take])),
                observed_fraction=float(np.average(y[te][take], weights=w[te][take]))))
    write_metric_rows(calibration, out, family=family, name='calibration')
    write_json(tech/'complete.json', dict(identity=digest([p['identity'], name, space, task, lead]),
        sha256=sha(tech/'predictions.npz'), seconds=time.monotonic()-started,
        budget=c['probe'], training_rows=int(tr.sum()), selection=None, calibration=None))
    print('probe complete', name, space, task, lead, round(time.monotonic()-started, 1), flush=True)


def fit_spaces(c, p, data, name, spaces, *, family=FAMILY):
    jobs = [(space, task, lead) for space in spaces for task in c['tasks'] for lead in c['leads_ps']]
    failures = []
    def attempt(job):
        space, task, lead = job
        for trial in range(2):
            try:
                fit_one(c, p, data, spaces[space], name, space, task, lead, family=family)
                return
            except Exception:
                record = dict(model=name, space=space, task=task, lead=lead, attempt=trial+1, traceback=traceback.format_exc())
                write_json(root(c)/'technical/failures'/f'{name}-{space}-{task}-{lead}-{trial}.json', record)
        failures.append(record)
    with ThreadPoolExecutor(max_workers=c['fit_workers']) as pool:
        list(pool.map(attempt, jobs))
    return failures


def summarize(c, p):
    import csv
    grouped = {}; rows = []; pairs = []
    for f in sorted((root(c)/'analyses').glob('*/*/*/lead-*/tables/scores.csv')):
        for r in csv.DictReader(f.open()):
            rows.append(r)
    write_metric_rows(rows, root(c)/'analyses/comparison', family=FAMILY, name='individual')
    for r in rows:
        treatment = r['model'].split('-s')[0]
        key = (treatment, r['space'], r['task'], r['lead_ps'], r['metric'])
        grouped.setdefault(key, []).append(float(r['value']))
    summary = [dict(treatment=k[0], space=k[1], task=k[2], lead_ps=k[3], metric=k[4],
                    value=float(np.mean(v)), seed_sd=float(np.std(v)), seeds=len(v)) for k, v in grouped.items()]
    write_metric_rows(summary, root(c)/'analyses/comparison', family=FAMILY, name='summary')
    # Reuse exactly the same ancestor draws across models and checkpoint seeds.
    for task in c['tasks']:
        for lead in c['leads_ps']:
            bundles = {}
            for f in (root(c)/'analyses').glob(f'*/*/{task}/lead-{lead:g}ps/technical/predictions.npz'):
                z = np.load(f); te = z['role'] == 'test'
                name, space = f.parts[-6:-4]
                _, slots, draws = source_draws(z['ancestor'][te], c['bootstrap'], c['seed'])
                boot, _ = bootstrap_scores(z['label'][te], z['probability'][te], z['weight'][te], slots, draws, None)
                bundles[name, space] = (z['id'][te], scores(z['label'][te], z['probability'][te], z['weight'][te]), boot)
            comparisons = []
            for space in ['phi', 'path', 'measurements_path']:
                for left, right in [('latent-physical', 'latent'), ('latent-transport', 'latent'), ('latent-transport', 'latent-physical')]:
                    comparisons.append((left, space, right, space))
            for treatment in ['latent', 'latent-physical', 'latent-transport', 'direct']:
                comparisons.append((treatment, 'measurements_path', 'controls', 'measurements'))
            for left, left_space, right, right_space in comparisons:
                changes = {m: [] for m in ['ap', 'auroc', 'nll', 'brier']}; resampled = {m: [] for m in changes}
                for seed in c['model_seeds']:
                    lk = (f'{left}-s{seed}', left_space)
                    rk = (right if right == 'controls' else f'{right}-s{seed}', right_space)
                    if lk not in bundles or rk not in bundles:
                        continue  # Missing independent fit is retained as failure, never silently averaged as three seeds.
                    a, b = bundles[lk], bundles[rk]
                    np.testing.assert_array_equal(a[0], b[0])
                    for m in changes:
                        changes[m].append(a[1][m]-b[1][m]); resampled[m].append(a[2][m]-b[2][m])
                for m, v in changes.items():
                    if v:
                        lo, hi = np.quantile(np.mean(resampled[m], axis=0), [.025, .975])
                        pairs.append(dict(left=left, left_space=left_space, right=right, right_space=right_space,
                            task=task, lead_ps=lead, metric=m, difference=float(np.mean(v)), lo=lo, hi=hi, seeds=len(v)))
    write_metric_rows(pairs, root(c)/'analyses/comparison', family=FAMILY, name='paired')
    return summary


def run(c, p):
    started = time.time(); torch.set_num_threads(c['threads'])
    data = collect(c, p); measurement = measured(c, p, data)
    failures = fit_spaces(c, p, data, 'controls', dict(prior=None, structural=measurement[:, :227], measurements=measurement))
    pool = RetainedCache(cache(c)/'features', 6)
    for spec in p['checkpoints']:
        name = spec['arm']['name']; metadata = dict(plan=p['identity'], checkpoint=spec['sha256'])
        write_json(root(c)/'technical/progress.json', dict(stage='model', model=name, started=started))
        try:
            with pool.lease(digest(metadata), deadline=time.time()+3600, metadata=metadata) as folder:
                if not (folder/'complete.json').exists():
                    value = features(c, p, data, spec)
                    atomic_save(folder/'features.pt', value)
                    write_json(folder/'complete.json', dict(sha256=sha(folder/'features.pt')))
                if sha(folder/'features.pt') != read(folder/'complete.json')['sha256']:
                    raise ValueError('Changed frozen-model feature cache')
                value = torch.load(folder/'features.pt', map_location='cpu', weights_only=False)
                spaces = dict(path=value['path'], measurements_path=np.concatenate((measurement, value['path']), -1))
                if spec['arm']['model'] == 'latent':
                    spaces['phi'] = value['phi']
                failures.extend(fit_spaces(c, p, data, name, spaces))
            # Existing scientific run receives local evaluation summaries only.
            import csv
            fields = {}
            for f in (root(c)/'analyses'/name).glob('*/*/lead-*/tables/scores.csv'):
                for r in csv.DictReader(f.open()):
                    fields[f"birth90/{r['space']}/{r['task']}/{r['lead_ps']}ps/{r['metric']}"] = float(r['value'])
            try:
                update_recorded_summary(Path(c['phase_run'])/'analyses'/name/'technical/wandb.json',
                    fields, evaluation='birth90-fixed-budget-v1', expected=dict(mode='online'))
            except Exception:
                write_json(root(c)/'technical'/f'wandb-failed-{name}.json', dict(traceback=traceback.format_exc()))
        except Exception:
            error = dict(model=name, traceback=traceback.format_exc())
            failures.append(error); write_json(root(c)/'technical'/f'failed-{name}.json', error)
    summary = summarize(c, p)
    write_json(root(c)/'technical/complete.json', dict(state='complete' if not failures else 'partial',
        failures=failures, seconds=time.time()-started, finished=time.time(), summary_rows=len(summary)))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'launch', 'run', 'summarize'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args(); c = read(args.config)
    if args.stage == 'launch':
        import os
        import subprocess
        import sys
        from src.experiment_runner.execution import ExecutionBundle
        from .campaign import environment
        prepare(c)
        repository = Path(__file__).resolve().parents[3]
        technical = root(c)/'technical'
        bundle = ExecutionBundle.freeze(repository, technical/'source', c,
            directories=['src', 'configs', 'docs/metrics'],
            files=[(repository/'machine.local.yaml', 'machine.local.yaml')])
        write_json(technical/'source-files.json', {str(f.relative_to(bundle.root)): sha(f)
            for f in bundle.root.rglob('*') if f.is_file()})
        command = [sys.executable, '-u', '-m', 'src.research.ni_phase_sim.births',
                   'run', '--config', str(bundle.config_path)]
        log = technical/'worker.log'
        with log.open('a') as stream:
            process = subprocess.Popen(command, cwd=bundle.root,
                env=dict(os.environ, **environment(bundle)), stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(technical/'launch.json', dict(pid=process.pid, started=time.time(),
            allocation=os.environ.get('SLURM_JOB_ID'), command=command, log=str(log),
            independent_of_chat=True, source=str(bundle.root)))
        print('Detached birth evaluation PID', process.pid, flush=True)
    elif args.stage == 'prepare':
        prepare(c)
    else:
        p = read(root(c)/'technical/plan.json')
        if args.stage == 'run':
            try:
                run(c, p)
            except Exception:
                write_json(root(c)/'technical/failed.json', dict(traceback=traceback.format_exc()))
                raise
        else:
            summarize(c, p)


if __name__ == '__main__':
    main()
