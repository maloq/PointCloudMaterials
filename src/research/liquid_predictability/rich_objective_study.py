"""Matched packed-data VCReg/TDA pilots and validation-selected full-data fitting."""
import argparse
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
from src.experiment_runner.metric_docs import check_metric_docs
from src.project_runtime.paths import REPO, resolve_path
from .data import config
from .rich_objectives import descriptor_blocks
from .rich_packed import pack, pack_pilot


def recipes(study):
    base = config(Path(__file__).resolve().parents[3] / study['base_config'])
    base.update(fitting_reference=study['fitting_reference'], protocol='rich_packed_objectives_v1',
                metric_family='rich_tda_objectives')
    base['loader'] = dict(packed_cache=study['packed_full'], max_open_shards=128, prefetch=2)
    base['runtime'].update(gpus=1, allowed_world_sizes=[1], partition=study['partition'],
                           walltime='24:00:00', memory_GB=48, cpu_threads=6)
    base['patch_chunk'] = 512
    base['training']['epochs'] = study['pilot_epochs']
    base['evaluation']['export_test'] = False
    base['regularization'].update(placement='embedding', covariance_normalization='mean_pair')
    base['objective'] = dict(weighting='family', normalization='coordinate',
        topology_distance=dict(weight=0., panel_size=512, temperature=.1))
    base['wandb']['group'] = study['name']
    base['descriptor_head']['block_width'] = 64
    base.pop('restart_reason', None)
    columns = config(resolve_path(base['cache']) / 'manifest.json')['columns']
    base['descriptor_head']['output_blocks'] = descriptor_blocks(columns)
    result = []
    for arm in study['arms']:
        c = copy.deepcopy(base)
        c['name'] = f"MM-TDA-{arm['name']}"
        c['output'] = f"{study['output']}/pilots/{arm['name']}"
        c['loader']['packed_cache'] = study['packed_pilot']
        c['data']['training_rows'] = round(base['data']['training_rows'] * study['pilot_fraction'])
        c['regularization'].update(arm['regularization'])
        if arm.get('structured_head'):
            c['descriptor_head']['kind'] = 'structured_residual_v1'
            c['objective']['weighting'] = 'structured'
        c['objective']['normalization'] = arm.get('normalization', 'coordinate')
        c['objective']['topology_distance']['weight'] = arm.get('topology_weight', 0.)
        c['wandb']['display_name'] = f"{c['name']} | 10% pilot | {study['pilot_epochs']} epochs"
        result.append(c)
    return base, result


def prepare(study):
    base, runs = recipes(study)
    full = pack(base)
    pilot = pack_pilot(base, study['packed_pilot'], study['pilot_fraction'])
    root = resolve_path(study['output']) / 'technical'
    root.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(runs):
        path = root / f'pilot-{i}.json'
        if path.exists() and config(path) != c:
            raise ValueError(f'Pilot recipe changed: {path}')
        write_json(path, c)
    write_json(root / 'prepared.json', dict(full_identity=full['identity'],
        pilot_identity=pilot['identity'], full_rows=full['binding']['rows'],
        pilot_rows=pilot['binding']['rows'], pilot_material_rows=pilot['material_rows'],
        pilot_source_count=pilot['source_count'], validation_rows=192960,
        sampling='one nested uniform draw; identical rows and global epoch permutations in all pilots'))


def configure_run(c, measurement):
    """Bind existing packed rows to a new run; do not redraw or profile in training."""
    import numpy as np
    from .models import RichPatchMACE
    packed = resolve_path(c['loader']['packed_cache'])
    manifest = config(packed / 'manifest.json')
    tech = resolve_path(c['output']) / 'technical'
    tech.mkdir(parents=True, exist_ok=True)
    for name in ('training-pool-row-ids.npy', 'target-standardization.npz'):
        shutil.copy2(packed / name, tech / name)
    # Construction is local and does not create an online run or train weights.
    import torch
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'])
        model = RichPatchMACE(c, len(manifest['columns']))
    steps = math.ceil(c['data']['training_rows'] / c['training']['batch_size'])
    plan = dict(config_sha256=digest(c), dataset_identity=manifest['binding']['dataset'],
        packed_identity=manifest['identity'], global_batch=c['training']['batch_size'],
        packed_manifest_sha256=sha(packed / 'manifest.json'),
        per_gpu_batch=c['training']['batch_size'], world_size=1,
        minimum_device_memory_bytes=measurement['minimum_device_memory_bytes'],
        train_rows=c['data']['training_rows'], updates_per_epoch=steps,
        total_updates=steps * c['training']['epochs'],
        subset_sha256=sha(tech / 'training-pool-row-ids.npy'),
        transform_sha256=sha(tech / 'target-standardization.npz'),
        material_rows=manifest['material_rows'], pool_rows=manifest['pool_rows'],
        source_count=manifest['source_count'],
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),
        total_parameters=sum(p.numel() for p in model.parameters()),
        numerical_verification=measurement)
    if (tech / 'batch-plan.json').exists() and config(tech / 'batch-plan.json') != plan:
        raise ValueError('Existing training plan changed')
    write_json(tech / 'batch-plan.json', plan)
    write_json(tech / 'config.json', c)
    return plan


def execute(c):
    from .rich_multimaterial_train import train
    tech = resolve_path(c['output']) / 'technical'
    os.environ['PCM_RICH_OWNER'] = f'{socket.gethostname()}:{os.getpid()}:{os.environ["SLURM_JOB_ID"]}'
    kernel = resolve_path(c['runtime']['kernel_cache'])
    kernel.mkdir(parents=True, exist_ok=True)
    os.environ['CUEQUIVARIANCE_OPS_NVRTC_CACHE_DIR'] = str(kernel)
    with (tech / 'fit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (tech / 'complete.json').exists() and not c['evaluation']['export_test']:
            return
        train(c)
        if not (tech / 'complete.json').exists():
            raise RuntimeError('Allocation ended before requested epochs; resume this unchanged recipe')
        if not c['evaluation']['export_test']:
            write_json(tech / 'state.json', dict(state='pilot_complete', epochs=c['training']['epochs'],
                                                test_evaluated=False))


def promote(study):
    """Fixed validation-only rule; test labels/predictions are never read."""
    from .control_train import table
    root = resolve_path(study['output'])
    tech = root / 'technical'
    _, runs = recipes(study)
    scores = []
    for i, c in enumerate(runs):
        t = resolve_path(c['output']) / 'technical'
        complete = config(t / 'complete.json')
        if complete['epochs'] != study['pilot_epochs']:
            raise ValueError(f'Incomplete pilot {c["name"]}')
        history = [json.loads(line) for line in (t / 'validation.jsonl').read_text().splitlines()]
        best = min(history, key=lambda row: row['validation/descriptor_nll'])
        scores.append(dict(arm=study['arms'][i]['name'], index=i,
            selection_nll=best['validation/descriptor_nll'], selected_epoch=best['epoch'],
            selection_relative_mse=best['validation/relative_mse_to_training_mean'],
            selection_tda_relative_mse=best['validation/tda_relative_mse_to_training_mean'],
            embedding_participation_rank=best['validation/embedding_participation_rank'],
            embedding_d95=best['validation/embedding_d95'],
            checkpoint_sha256=complete['best_sha256']))
    winner = min(scores, key=lambda row: (row['selection_nll'], row['index']))
    c = runs[winner['index']]
    base, _ = recipes(study)
    c['data']['training_rows'] = base['data']['training_rows']
    c['loader']['packed_cache'] = study['packed_full']
    c['training']['epochs'] = study['full_epochs']
    c['evaluation']['export_test'] = True
    c['name'] += '-FULL'
    c['output'] = f"{study['output']}/full"
    c['wandb']['display_name'] = f"{c['name']} | full fitting subset | {study['full_epochs']} epochs"
    record = dict(rule='minimum Al selection family-balanced Gaussian NLL; tie: declared arm index',
                  winner=winner, pilots=scores, initialization='scratch; no pilot weights reused',
                  test_used_for_selection=False, full_config=c)
    if (tech / 'promotion.json').exists() and config(tech / 'promotion.json') != record:
        raise ValueError('Frozen full-data selection changed')
    write_json(tech / 'promotion.json', record)
    table(root / 'analyses/pilot-selection', 'pilots', scores, family='rich_tda_objectives')
    configure_run(c, config(tech / 'numerical-check.json'))
    execute(c)


def submit(path, first_external=False):
    study = config(path)
    root = resolve_path(study['output']) / 'technical'
    if (root / 'launch.json').exists():
        raise ValueError('Study already submitted')
    measurement = config(root / 'numerical-check.json')
    if not measurement['finite'] or measurement['study_sha256'] != digest(study):
        raise ValueError('Matching local numerical verification required')
    check_metric_docs(family='rich_tda_objectives')
    base, runs = recipes(study)
    for c in runs:
        configure_run(c, measurement)
    bundle = ExecutionBundle.freeze(REPO, root / 'code', study,
        directories=('src', 'docs/metrics', 'configs/liquid_predictability'))
    env = dict(PCM_PROJECT_ROOT=str(REPO), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    receipt = dict(submitted_at=time.time(), jobs={}, first_pilot_external=first_external,
                   code=str(bundle.root), study_sha256=digest(study))
    queue = SlurmQueue(root, bundle, 'src.research.liquid_predictability.rich_objective_study',
                       env, root / 'launch.json', receipt, 'MM-TDA')
    with queue.submission():
        array = f'{1 if first_external else 0}-{len(runs)-1}%{1 if first_external else 2}'
        pilots = queue.submit('pilot', [f'--array={array}', '--gpus=1', '--cpus-per-task=6',
            '--mem=48G', '--time=08:00:00'], partition=study['partition'])
        queue.submit('promote', ['--gpus=1', '--cpus-per-task=6', '--mem=48G', '--time=24:00:00'],
                     dependency=f'afterok:{pilots}', partition=study['partition'])
    return receipt


def submit_run(path):
    """Submit an explicit full-data recipe without rerunning pilot selection."""
    c = config(path)
    tech = resolve_path(c['output']) / 'technical'
    if (tech / 'launch.json').exists():
        raise ValueError('Full-data run already submitted')
    verification = config(tech / 'numerical-check.json')
    if not verification['finite'] or verification['config_sha256'] != digest(c):
        raise ValueError('Matching numerical verification required before submission')
    check_metric_docs(family=c['metric_family'])
    configure_run(c, verification)
    bundle = ExecutionBundle.freeze(REPO, tech / 'code', c,
        directories=('src', 'docs/metrics', 'configs/liquid_predictability'))
    env = dict(PCM_PROJECT_ROOT=str(REPO), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1',
               PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    receipt = dict(submitted_at=time.time(), jobs={}, code=str(bundle.root), config_sha256=digest(c))
    queue = SlurmQueue(tech, bundle, 'src.research.liquid_predictability.rich_objective_study',
                       env, tech / 'launch.json', receipt, 'MM-TDA-direct')
    r = c['runtime']
    with queue.submission():
        queue.submit('train-run', [f'--gpus={r["gpus"]}', f'--cpus-per-task={r["cpu_threads"]}',
            f'--mem={r["memory_GB"]}G', f'--time={r["walltime"]}'], partition=r['partition'])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'submit', 'pilot', 'promote', 'submit-run', 'train-run'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', type=int)
    parser.add_argument('--first-external', action='store_true')
    args = parser.parse_args()
    study = config(args.config)
    if args.stage == 'submit-run':
        print(json.dumps(submit_run(args.config), indent=2))
    elif args.stage == 'train-run':
        execute(study)
    elif args.stage == 'prepare':
        prepare(study)
    elif args.stage == 'submit':
        print(json.dumps(submit(args.config, args.first_external), indent=2))
    elif args.stage == 'pilot':
        index = args.arm if args.arm is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
        _, runs = recipes(study)
        execute(runs[index])
    else:
        promote(study)


if __name__ == '__main__':
    main()
