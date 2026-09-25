"""One detached, deadline-bounded GPU queue with validation-only promotion."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
import torch

from src.training_methods.shared_pretraining.queue import snapshot, deadline_for_job
from .common import Study, write_json


def configure():
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def preflight(study, device):
    from .data import prepare, Corpus, sampling_distribution
    from .train import make_banks, make_model, initialize, supervised_step, validate, configure_runtime
    prepare(study)
    corpus = Corpus(study)
    banks = make_banks(study, corpus, device)
    # Preflight checks production-sized batches and all gradient paths on a
    # small, explicitly labelled fixture from the actual cohort. Full fitting
    # and validation populations are restored by the independent worker load.
    for role in ('train', 'selection'):
        ids = corpus.split[role]
        coverage = [ids[corpus.pop['event'][ids] == k][:2] for k in range(6)]
        corpus.split[role] = np.unique(np.concatenate([ids[:128], *coverage]))
    c = study.config['training']
    fit = corpus.split['train']
    events = torch.as_tensor(corpus.pop['event'], device=device)
    q, iw = sampling_distribution(corpus.pop['source'][fit], corpus.pop['event'][fit], c['positive_sampling_fraction'])
    rng = np.random.default_rng(study.config['seed'])
    drawn = rng.choice(len(fit), size=c['batch_size'], p=q)
    weights = torch.as_tensor(iw[drawn], dtype=torch.float32, device=device)
    receipts = []
    for arm in study.config['arms']:
        model = make_model(study, corpus, arm, device)
        initialize(model, banks, corpus, c['microbatch'])
        configure_runtime(study, model, banks, corpus)
        teacher = None
        if arm['teacher']:
            # Production-shaped finite synthetic targets exercise the auxiliary
            # path only; actual training refuses absent frozen teacher receipts.
            teacher = dict(features=torch.zeros((len(events), 128), device=device),
                           hazards=torch.full((len(events), 5), .05, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        # The calibrated zero-weight hazard first updates its head; the second
        # step must propagate into the spatial encoder.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            loss = supervised_step(model, banks, fit[drawn], weights, events, c, teacher)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
        grad = model.encoder.center_embedding.weight.grad
        if grad is None or not torch.isfinite(grad).all() or float(grad.norm()) == 0:
            raise ValueError(f'No supervised spatial encoder gradient: {arm["name"]}')
        torch.cuda.synchronize()
        ordinary_seconds = (time.monotonic()-started)/2
        scores = validate(model, banks, corpus, c['microbatch'])
        record = dict(arm=arm['name'], batch_size=c['batch_size'], microbatch=c['microbatch'],
            training_rows=len(fit), ordinary_step_seconds=ordinary_seconds,
            peak_GiB=torch.cuda.max_memory_allocated()/2**30,
            loss=loss, gradient_norm=float(norm), selection=scores,
            purpose='correctness smoke on recorded cohort subset; not a scientific fit',
            fit_rows=fit.tolist(), selection_rows=corpus.split['selection'].tolist())
        receipts.append(record)
        print(json.dumps(dict(stage='preflight', **record)), flush=True)
        del model, optimizer, teacher
    study.bind()
    write_json(study.technical/'preflight.json', dict(passed=True, identity=study.identity,
        gpu=torch.cuda.get_device_name(), arms=receipts))


def launch(study):
    """Start a Slurm step inside the already granted idle GPU allocation."""
    study.bind()
    receipt = json.loads((study.technical/'preflight.json').read_text())
    if not receipt['passed'] or receipt['identity'] != study.identity:
        raise ValueError('Production preflight does not match this exact source/config/data')
    launch_path = study.technical/'launch.json'
    if launch_path.exists():
        raise FileExistsError('Already launched; inspect launch.json and queue-state.json')
    cfg = study.config['queue']
    if cfg['allocation'] is None:
        raise ValueError('Set queue.allocation to a granted Slurm allocation before launch')
    result = subprocess.run(['scontrol', 'show', 'job', cfg['allocation'], '--json'], check=True, capture_output=True, text=True)
    job = json.loads(result.stdout)['jobs'][0]
    end = job['end_time']['number'] if isinstance(job['end_time'], dict) else job['end_time']
    if end-time.time() < cfg['hours']*3600 + 600:
        raise ValueError('Current allocation cannot contain the full queue plus cleanup reserve')
    code = snapshot(study.technical)
    config = code / study.path.relative_to(Path.cwd().resolve())
    env = dict(os.environ, PCM_PROJECT_ROOT=str(code), TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTORCH_ALLOC_CONF='expandable_segments:True')
    command = ['srun', '--jobid', cfg['allocation'], '--overlap', '--exact', '--nodes=1', '--ntasks=1',
        f'--cpus-per-task={cfg["cpus"]}', '--gres=gpu:1', '--time=08:40:00',
        '--job-name=supervised-information', '--unbuffered', sys.executable, '-u', '-m',
        'src.research.supervised_onset.queue', 'worker', '--config', str(config)]
    with (study.technical/'queue.log').open('ab', buffering=0) as log:
        process = subprocess.Popen(command, cwd=code, env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(state='launched', allocation=cfg['allocation'], launcher_pid=process.pid,
        identity=study.identity, code=str(code), command=command, hours=cfg['hours'],
        requested_at=datetime.now(timezone.utc).isoformat(), log=str(study.technical/'queue.log'))
    write_json(launch_path, record)
    print(json.dumps(record, indent=2), flush=True)


def finalists(results, count):
    from .train import selection_key
    if len(results) < count:
        raise ValueError('Insufficient completed screen arms for promotion')
    return [r['arm'] for r in sorted(results, key=lambda r: selection_key(r['best']))[:count]]


def worker(study, device):
    from .data import prepare, Corpus
    from .train import make_banks, fit
    from .evaluate import descriptor_baselines, evaluate_arm, ensemble, collect
    cfg = study.config['queue']
    state_path = study.technical/'queue-state.json'
    # This worker is an srun step, not a process merely inheriting SLURM_JOB_ID.
    step_id = os.environ['SLURM_STEP_ID']
    with (study.technical/'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.time()
        deadline = min(start+cfg['hours']*3600, deadline_for_job())
        train_deadline = deadline-cfg['evaluation_reserve_minutes']*60
        def status(state, **extra):
            write_json(state_path, dict(state=state, allocation=os.environ['SLURM_JOB_ID'],
                step=step_id, pid=os.getpid(),
                deadline_utc=datetime.fromtimestamp(deadline, timezone.utc).isoformat(),
                updated_utc=datetime.now(timezone.utc).isoformat(), **extra))
        try:
            status('running', stage='verified_data_load')
            prepare(study)
            study.bind()
            corpus = Corpus(study)
            banks = make_banks(study, corpus, device)
            status('running', stage='descriptor_controls')
            descriptor_baselines(study, corpus, device)
            results = []
            for arm in study.config['arms']:
                name = arm['name']
                path = study.technical/'runs'/name/'screen.json'
                if path.exists():
                    results.append(json.loads(path.read_text()))
                    continue
                status('running', stage='screen', arm=name)
                end = min(train_deadline, time.time()+cfg['screen_maximum_minutes_per_arm']*60)
                result = fit(study, corpus, banks, name, until=end,
                             max_updates=study.config['training']['screen_updates'], device=device)
                if result['updates'] == 0:
                    raise TimeoutError(f'No training budget left for screen arm {name}')
                shutil.copy2(study.technical/'runs'/name/'best.pt', path.with_name('screen-best.pt'))
                write_json(path, result)
                results.append(result)
            winners = finalists(results, cfg['finalists'])
            write_json(study.technical/'promotion.json', dict(arms=winners, screen=results,
                rule='Smallest source-weighted selection hazard NLL; stable arm order breaks exact ties. AP/calibration/test scores are not selectors.'))
            for position, name in enumerate(winners):
                status('running', stage='extended_training', arm=name, finalists=winners)
                available = train_deadline-time.time()
                if available < 180:
                    break
                end = time.time()+available/(len(winners)-position)
                fit(study, corpus, banks, name, until=end,
                    max_updates=study.config['training']['maximum_updates'], device=device)
            # Finalists first so a deadline cannot prioritize diagnostics over the
            # primary result. All comparisons use the same archived row ordering.
            order = winners+[a['name'] for a in study.config['arms'] if a['name'] not in winners]
            for name in order:
                status('running', stage='evaluation', arm=name, finalists=winners)
                if not (study.technical/'runs'/name/'evaluation-complete.json').exists():
                    evaluate_arm(study, corpus, banks, name, device, deadline)
                collect(study)
            ensemble(study, corpus, winners)
            rows = collect(study)
            status('complete', finalists=winners, comparison_rows=len(rows), elapsed_hours=(time.time()-start)/3600)
        except Exception as error:
            # Preserve a partial export and the actual failure, never claim that
            # an incomplete scientific comparison completed successfully.
            status('checkpointed' if isinstance(error, TimeoutError) else 'failed',
                   error=repr(error), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=('prepare', 'preflight', 'launch', 'worker', 'collect'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    configure()
    study = Study(args.config)
    if args.action == 'prepare':
        from .data import prepare
        prepare(study)
    elif args.action == 'preflight':
        preflight(study, args.device)
    elif args.action == 'launch':
        launch(study)
    elif args.action == 'worker':
        worker(study, args.device)
    else:
        from .evaluate import collect
        study.bind()
        collect(study)


if __name__ == '__main__':
    main()
