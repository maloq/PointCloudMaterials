"""Disjoint workers for the frozen native onset stages; no scientific changes."""
import argparse
import fcntl
import json
from pathlib import Path
import resource

import torch

from src.data.predictive_memory.prepare import write_json
from src.project_runtime.paths import load_json, resolve_path
from .supervised import prepare_rows, fit_stage, score_stage


def configure_file_limit(required=8192):
    # The maintained trajectory loader retains seven mmap files per source.
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < required:
        raise RuntimeError(f'Native 150-source loader requires NOFILE >= {required}; hard limit is {hard}')
    if soft != resource.RLIM_INFINITY and soft < required:
        resource.setrlimit(resource.RLIMIT_NOFILE, (required, hard))
    return dict(previous_soft=soft, soft=resource.getrlimit(resource.RLIMIT_NOFILE)[0], hard=hard)


def run(config, stages, worker, resume=False):
    if len(set(stages)) != len(stages) or not set(stages) <= {'snapshot', 'history12', 'repeat12'}:
        raise ValueError(f'Expected distinct native continuation stages: {stages}')
    limits = configure_file_limit()
    torch.set_num_threads(config['torch_threads'])
    root = resolve_path(config['output']) / 'technical'
    root.mkdir(parents=True, exist_ok=True)
    status = root / f'worker-{worker}.json'
    completed = []
    print(json.dumps(dict(worker=worker, stages=stages, file_limit=limits)), flush=True)
    try:
        budget = json.loads(resolve_path(config['budget_receipt']).read_text())
        if budget['state'] != 'agreed' or budget['updates_per_stage'] != config['updates_per_stage']:
            raise ValueError('Worker must preserve the agreed matched stage budget')
        windows, labels, cond, events, splits, selection, identity = prepare_rows(config)
        parent = root / 'parent/best.pt'
        if not (root / 'parent/complete.json').exists():
            raise RuntimeError('Disjoint workers require a completed common parent')
        for stage in stages:
            with (root / f'{stage}.worker.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                write_json(status, dict(state='running', current=stage, completed=completed, file_limit=limits))
                fit_stage(config, windows, labels, cond, events, splits, selection, identity,
                          stage, parent=parent, resume=resume)
                score_stage(config, windows, labels, cond, events, splits, identity, stage)
                completed.append(stage)
        write_json(status, dict(state='complete', completed=completed, file_limit=limits))
        with (root / 'completion.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            variants = ['snapshot', 'history12', 'repeat12']
            if all((root / stage / 'result.json').exists() for stage in variants):
                for stage in variants:
                    if json.loads((root / stage / 'complete.json').read_text())['identity'] != identity:
                        raise ValueError(f'Matched completion identity differs: {stage}')
                write_json(root / 'status.json', dict(state='complete', seed=config['seed'],
                           identity=identity, variants=variants, execution='disjoint H100/RTX6000 workers'))
    except Exception as exc:
        write_json(status, dict(state='failed', completed=completed, error=repr(exc), file_limit=limits))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stages', nargs='+', required=True, choices=['snapshot', 'history12', 'repeat12'])
    parser.add_argument('--worker', required=True, choices=['h100', 'rtx6000'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(load_json(args.config), args.stages, args.worker, args.resume)


if __name__ == '__main__':
    main()
