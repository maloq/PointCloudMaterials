"""Restart unconverged pilot cells from archived full-precision coordinates."""

import argparse
import json
import os
from pathlib import Path
import time
import traceback
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import save_json, file_hash
from src.data.relaxed_targets.worker import verify_archive
from src.training_methods.shared_pretraining.queue import deadline_for_job
from .prepare import freeze, produce
from .queue import claim, build


def retry_cells(plan, root, lane, ranks, accelerator=None):
    recipe = json.loads((root / 'plan.json').read_text())
    technical = resolve_path(plan['config']['output']) / 'technical'
    deadline = deadline_for_job()
    failed = False
    for item in recipe['cells']:
        task = item['task']
        status = root / 'cells' / f'{task["id"]}.json'
        if status.exists() and json.loads(status.read_text())['state'] == 'complete':
            continue
        if time.time() > deadline - recipe['limits']['frame_timeout_seconds'] - 300:
            raise SystemExit(75)
        with claim(technical / 'locks' / f'cell-{task["id"]}') as acquired:
            if not acquired:
                continue
            if status.exists() and json.loads(status.read_text())['state'] == 'complete':
                continue
            save_json(status, dict(state='running', lane=lane, task=task))
            try:
                verify_archive(Path(item['archive']))
                recovery = dict(
                    name=recipe['name'],
                    limits=recipe['limits'],
                    restart_dump=item['restart_dump'],
                    restart_sha256=item['restart_sha256'],
                )
                result = produce(plan, task, ranks, recovery=recovery, accelerator=accelerator)
                previous = technical / 'failures' / f'{task["id"]}.json'
                if previous.exists():
                    if file_hash(previous) != item['failure_sha256']:
                        raise ValueError('Original failure record changed')
                    dest = root / 'resolved-failures' / previous.name
                    dest.parent.mkdir(exist_ok=True)
                    previous.rename(dest)
                save_json(
                    status,
                    dict(
                        state='complete',
                        lane=lane,
                        force=result['relaxation']['fmax_eV_per_A'],
                        seconds=result['relaxation']['seconds'],
                    ),
                )
                print(
                    json.dumps(
                        dict(
                            cell=task['id'],
                            state='complete',
                            force=result['relaxation']['fmax_eV_per_A'],
                        )
                    ),
                    flush=True,
                )
            except Exception as exc:
                failed = True
                save_json(
                    status, dict(state='failed', error=repr(exc), traceback=traceback.format_exc())
                )
                traceback.print_exc()
    if failed:
        raise RuntimeError('Some restarted quenches failed; inspect recovery cell receipts')


def rebuild(plan, root):
    technical = resolve_path(plan['config']['output']) / 'technical'
    try:
        from .availability import fatal_failures

        if fatal_failures(plan):
            raise RuntimeError('Unresolved cell failures')
        old = technical / 'build-failed.json'
        if old.exists():
            old.rename(root / 'original-build-failed.json')
        save_json(root / 'build-status.json', dict(state='running'))
        build(plan)
        if (
            not (technical / 'training-ready.json').exists()
            or not (technical / 'assay/ready.json').exists()
        ):
            raise RuntimeError('Cache preparation did not finish before allocation deadline')
        save_json(root / 'build-status.json', dict(state='complete'))
    except Exception as exc:
        save_json(
            root / 'build-status.json',
            dict(state='failed', error=repr(exc), traceback=traceback.format_exc()),
        )
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['cpu', 'cuda', 'build'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--accelerator-config')
    parser.add_argument('--backend', choices=['h100', 'a100', 'v100'])
    args = parser.parse_args()
    plan = freeze(json.loads(Path(args.config).read_text()))
    root = resolve_path(plan['config']['output']) / 'technical/restarts' / args.name
    if args.stage == 'cpu':
        retry_cells(plan, root, os.environ['SLURM_ARRAY_TASK_ID'], 32)
    elif args.stage == 'cuda':
        from .accelerated import wait_for_benchmark

        accelerator_config = json.loads(resolve_path(args.accelerator_config).read_text())
        profile = wait_for_benchmark(
            accelerator_config, args.backend, deadline_for_job(), root / 'accelerator.json'
        )
        retry_cells(plan, root, args.backend, 1, accelerator=profile)
    else:
        rebuild(plan, root)


if __name__ == '__main__':
    main()
