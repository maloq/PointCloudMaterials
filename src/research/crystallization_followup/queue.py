"""Immutable, resumable two-GPU queue over already prepared paired observations."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import digest, file_hash, save_json
from src.training_methods.shared_pretraining.queue import snapshot, deadline_for_job
from src.research.crystallization_transfer.runtime import setup
from src.research.crystallization_paths.runtime import fit, make_model, initialize
from src.research.local_predictability.metrics import source_weights
from src.experiment_runner.metric_docs import write_metric_table
from .data import FollowupPaths
from .metrics import window_scores, paired_source_gain


def freeze(config):
    root = resolve_path(config['output']) / 'technical'; root.mkdir(parents=True, exist_ok=True)
    path = root / 'plan.json'
    if path.exists():
        plan = json.loads(path.read_text())
        if plan['followup_config'] != config:
            raise ValueError('Follow-up configuration changed after freeze')
        if file_hash(resolve_path(config['source_plan'])) != plan['followup_parent_sha256']:
            raise ValueError('Parent observation plan changed')
        return plan
    parent = resolve_path(config['source_plan']); plan = json.loads(parent.read_text())
    plan['followup_config'] = config; plan['followup_parent_sha256'] = file_hash(parent)
    plan['config'] = dict(plan['config'], output=config['output'], seed=config['seed'],
                          batch_size=config['batch_size'])
    plan['identity'] = digest(plan)
    references = json.loads((parent.parent / 'queue.json').read_text())
    base = next(s for s in references if s['observation_domain'] == 'relaxed' and s['method'] == 'ar_mse')
    specs = []
    for variant in config['variants']:
        spec = copy.deepcopy(base)
        spec.update(context_layout='cuboctahedral_followup_v1', history_rates=False,
                    dense_mode='off', secondary_domain='off', quench_descriptors=False,
                    short_weight=0., absolute_clock=True, selection_horizon_ps=12.,
                    patience=config['patience'], minimum_epochs=6, front_mode='off',
                    front_radius_A=25., front_features='all')
        spec['training']['epochs'] = config['epochs']
        spec.update(variant)
        specs.append(spec)
    if len({s['name'] for s in specs}) != len(specs):
        raise ValueError('Duplicate experiment names')
    save_json(root/'queue.json', specs); save_json(path, plan)
    return plan


def save_population(data, root):
    with (root/'population.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = root/'population.npz'
        if path.exists():
            with np.load(path) as old:
                np.testing.assert_array_equal(old['rows'], np.asarray(data.corpus.rows))
        else:
            np.savez_compressed(path, rows=np.asarray(data.corpus.rows), source=data.corpus.source_ids)


def verify(config):
    """Actual full-data joins, future poisoning, and finite updates for every variant."""
    setup(); plan = freeze(config); root = resolve_path(config['output'])/'technical'
    specs = json.loads((root/'queue.json').read_text())
    started = time.monotonic(); data = FollowupPaths(plan, specs[0]); save_population(data, root)
    ids = data.corpus.splits['train'][:4]
    checks = []
    # All variants must share exactly the same underlying MD labels and future states.
    reference = data.targets(ids)['state'].clone()
    for spec in specs:
        data.set_context(spec); torch.manual_seed(config['seed'])
        model = make_model(copy.deepcopy(spec)).to(data.device)
        initialize(model, data); model.initialize_information(data)
        observed = data.observed(ids); saved = data.states.clone()
        data.states.fill_(12345.)
        for key, value in data.observed(ids).items():
            torch.testing.assert_close(value, observed[key], rtol=0, atol=0)
        data.states.copy_(saved); del saved
        torch.testing.assert_close(data.targets(ids)['state'], reference, rtol=0, atol=0)
        target = data.targets(ids); target['present'] = data.baseline(ids)[:, 0]
        model.train(); loss = model.loss(observed, target, 0.).mean(); loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec['head_lr']); optimizer.step()
        model.eval()
        with torch.no_grad():
            paths, cdf = model.forecast(observed)
        if not torch.isfinite(paths).all() or not torch.isfinite(loss):
            raise FloatingPointError(spec['name'])
        window_scores(cdf.cpu().numpy(), target['event'].cpu().numpy())
        checks.append(dict(name=spec['name'], loss=float(loss.detach()), gradient_norm=float(grad),
                           parameters=sum(p.numel() for p in model.parameters())))
        print(json.dumps(checks[-1]), flush=True)
        del model, optimizer
    # One complete epoch/evaluation/checkpoint cycle on a four-role subset.
    # Inputs were already joined and verified on all 150 sources above.
    data.corpus.splits = {role: group[:8] for role, group in data.corpus.splits.items()}
    train = data.corpus.splits['train']; sid = int(data.corpus.source_ids[train[0]])
    data.full_training_indices = [i for i in train if data.corpus.source_ids[i] == sid]
    data.full_training_groups = {sid: np.array(data.full_training_indices)}
    # configure_training balances the temperatures present in these groups.
    smoke = copy.deepcopy(plan); smoke['config']['output'] = str(root/'smoke')
    smoke['config']['selection_per_source'] = 8
    data.plan = smoke; data.corpus.plan = smoke
    spec = copy.deepcopy(specs[-1]); spec['name'] = 'pipeline-smoke'
    spec['training'] = dict(spec['training'], sources=1, epochs=1)
    spec.update(patience=0, minimum_epochs=1)
    if not fit(smoke, spec, data, time.time()+3600):
        raise RuntimeError('Full checkpoint/evaluation smoke did not finish')
    implementation = implementation_hashes()
    save_json(root/'validation.json', dict(passed=True, checks=checks, full_data_causal_join=True,
              identical_targets=True, pipeline_smoke=True, seconds=time.monotonic()-started,
              implementation=implementation, plan_identity=plan['identity']))


def implementation_hashes():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [Path('src/research/crystallization_paths/runtime.py'),
              Path('src/research/structured_context/reuse_data.py')]
    # Relative paths work in both workspace and immutable executable snapshot.
    base = Path(__file__).resolve().parents[3]
    return {str(p.resolve().relative_to(base)): file_hash(p) for p in paths}


def report(config):
    root = resolve_path(config['output']); technical = root/'technical'
    specs = json.loads((technical/'queue.json').read_text())
    lines = ['# Literature-guided frozen-MACE follow-up', '',
             'One seed; matched archived origins; selection uses integrated Brier through 12 ps. '
             'All models retain 96 ps physical/embedding forecasts. Historical test sources are reused: '
             'this is an exploratory follow-up, not a fresh confirmatory test.', '',
             '| Experiment | State | AP 12 ps ↑ | Brier 0.75–12 ps ↓ | Timing MAE, detected (ps) ↓ | Missed / positive |',
             '|---|---|---:|---:|---:|---:|']
    scores = {}; arrays = {}; sources = None
    for spec in specs:
        folder = technical/'runs'/spec['name']; status_path = folder/'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {'state':'pending'}
        if not (folder/'metrics.json').exists():
            lines.append(f'| {spec["name"]} | {status["state"]} | | | | |'); continue
        metrics = json.loads((folder/'metrics.json').read_text())
        with np.load(folder/'predictions.npz') as p, np.load(technical/'population.npz') as pop:
            source = pop['source'][p['test_indices']]
            if sources is None:
                sources = source; indices = p['test_indices'].copy(); events = p['test_event'].copy()
            else:
                np.testing.assert_array_equal(source, sources)
                np.testing.assert_array_equal(p['test_indices'], indices)
                np.testing.assert_array_equal(p['test_event'], events)
            rows = window_scores(p['test_cdf'], p['test_event'])
        w = source_weights(source); cls = metrics['short_horizon']['classification']['12.0']
        timing = metrics['short_horizon']['timing']['12.0']
        scores[spec['name']] = dict(average_precision12=cls['average_precision'],
            integrated_brier12=float(w@rows['brier12']), restricted_time_mae12=float(w@rows['restricted_time_mae12']),
            recall12=cls['recall'], realized_fpr12=cls['false_positive_rate'], timing12=timing,
            brier96=metrics['dense_integrated_brier'], short_event_nll=metrics['short_horizon']['event_nll'])
        arrays[spec['name']] = rows
        error = timing['detected_timing_mae_ps']
        text_error = f'{error:.3f}' if error is not None else 'undefined'
        lines.append(f'| {spec["name"]} | {status["state"]} | {cls["average_precision"]:.4f} | '
                     f'{scores[spec["name"]]["integrated_brier12"]:.5f} | {text_error} | '
                     f'{timing["missed_windows"]} / {timing["event_windows"]} |')
    pairs = {}
    for candidate, control in config['comparisons']:
        if candidate in arrays and control in arrays:
            pairs[f'{candidate}_vs_{control}'] = paired_source_gain(
                arrays[control]['brier12'], arrays[candidate]['brier12'], sources,
                config['bootstrap_draws'], config['seed'])
    with (technical/'report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        write_metric_table(dict(experiments=scores, paired_brier12=pairs), root,
                           family='crystallization_followup', name='comparison')
        save_json(technical/'comparison.json', dict(experiments=scores, paired_brier12=pairs))
        (root/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def worker(config, lane):
    setup(); plan = freeze(config); root = resolve_path(config['output'])/'technical'
    validation = json.loads((root/'validation.json').read_text())
    if not validation['passed'] or validation['plan_identity'] != plan['identity']:
        raise ValueError('This plan has not passed validation')
    if implementation_hashes() != validation['implementation']:
        raise ValueError('Executable implementation changed after validation')
    specs = json.loads((root/'queue.json').read_text()); deadline = deadline_for_job()
    def state(value, **extra):
        save_json(root/f'lane-{lane}.json', dict(state=value, pid=os.getpid(), updated_at=time.time(), **extra))
    try:
        state('loading_existing_observations')
        data = FollowupPaths(plan, specs[0]); save_population(data, root)
        while time.time() < deadline-1800:
            claimed = False; pending = False
            for spec in specs:
                folder = root/'runs'/spec['name']; folder.mkdir(parents=True, exist_ok=True)
                with (folder/'worker.lock').open('a') as lock:
                    try: fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        pending = True; continue
                    status = json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {}
                    if status.get('state') == 'complete': continue
                    if status.get('state') == 'failed':
                        raise RuntimeError(f'Inspect failed fit before continuation: {folder}')
                    pending = claimed = True; state('training', fit=spec['name'])
                    try:
                        complete = fit(plan, spec, data, deadline)
                    except Exception as error:
                        save_json(folder/'status.json', dict(state='failed', error=repr(error), traceback=traceback.format_exc()))
                        raise
                    report(config)
                    if not complete:
                        state('checkpointed', fit=spec['name']); return
            if not pending:
                state('complete'); return
            if not claimed:
                state('waiting_for_other_lane'); time.sleep(20)
        state('checkpointed')
    except Exception as error:
        state('failed', error=repr(error), traceback=traceback.format_exc()); raise


def submit(config_path):
    config = json.loads(resolve_path(config_path).read_text()); plan = freeze(config)
    root = resolve_path(config['output'])/'technical'
    if (root/'launches.json').exists(): raise FileExistsError('Queue already launched')
    validation = json.loads((root/'validation.json').read_text())
    if not validation['passed'] or validation['implementation'] != implementation_hashes():
        raise ValueError('Current executable has not passed verification')
    job = os.environ['SLURM_JOB_ID']; code = snapshot(root)
    env = dict(os.environ, PCM_PROJECT_ROOT=str(code), OPENBLAS_NUM_THREADS='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTORCH_ALLOC_CONF='expandable_segments:True')
    command = ['srun', f'--jobid={job}', '--overlap', '--exact', '-N1', f'-n{config["workers"]}', '-c6',
               '--gpus-per-task=1', '--gpu-bind=single:1', '--mem=60G',
               f'--output={root}/lane-%t.log', sys.executable, '-u', '-m',
               'src.research.crystallization_followup.queue', 'worker', '--config',
               str(code/config_path), '--lane', 'slurm']
    with (root/'launcher.log').open('a') as log:
        proc = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    launches = [dict(lanes=config['workers'], pid=proc.pid, allocation=job, command=command)]
    save_json(root/'launches.json', launches)
    report(config); print(json.dumps(launches), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'verify', 'submit', 'worker', 'report'))
    parser.add_argument('--config', required=True); parser.add_argument('--lane', default='manual')
    args = parser.parse_args(); config = json.loads(resolve_path(args.config).read_text())
    if args.stage == 'prepare': freeze(config); report(config)
    elif args.stage == 'verify': verify(config)
    elif args.stage == 'submit': submit(args.config)
    elif args.stage == 'worker': worker(config, os.environ['SLURM_LOCALID'] if args.lane == 'slurm' else args.lane)
    else: report(config)


if __name__ == '__main__':
    main()
