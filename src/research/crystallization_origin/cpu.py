"""CPU orchestration; worker counts do not change frozen scientific contracts."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import digest, write_json
from src.project_runtime.paths import resolve_path
from . import extract


def extract_fixed(config, workers):
    _, plan = read_release(config['release'])
    root = resolve_path(config['output']) / 'technical'
    contract = extract.extraction_contract(config, plan)
    if json.loads((root / 'extraction-contract.json').read_text()) != contract:
        raise ValueError('CPU migration must use the exact existing extraction producer')
    completed = []
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            futures = [pool.submit(extract.source, config, plan, item, digest(contract)) for item in plan['sources']]
            for future in as_completed(futures):
                result = future.result()
                completed.append(result['source'])
                write_json(root / 'extraction-state.json', dict(state='running', completed=completed,
                           total_sources=len(futures), workers=workers, allocation=os.environ.get('SLURM_JOB_ID')))
                print(json.dumps(dict(source=result['source'], completed=len(completed), seconds=result['seconds'])), flush=True)
        write_json(root / 'extraction-state.json', dict(state='complete', completed=completed,
                   total_sources=len(plan['sources']), allocation=os.environ.get('SLURM_JOB_ID')))
    except Exception:
        write_json(root / 'extraction-state.json', dict(state='failed', completed=completed, traceback=traceback.format_exc()))
        raise


def pipeline(commands, root):
    """Cancel sibling processes on failure; Slurm owns all child process groups."""
    root.mkdir(parents=True, exist_ok=True)
    processes = []
    job = os.environ['SLURM_JOB_ID']
    try:
        for name, args in commands:
            with (root / f'{name}-{job}.log').open('a') as log:
                child = subprocess.Popen([sys.executable, '-u', '-m', *args], stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            processes.append((name, child))
        while True:
            states = {name: child.poll() for name, child in processes}
            write_json(root / f'cpu-job-{job}.json', dict(allocation=job, node=os.uname().nodename,
                       state='running', returncodes=states, updated_at=time.time()))
            if any(code is not None and code != 0 for code in states.values()):
                raise RuntimeError(f'CPU stage failed: {states}; inspect per-stage logs')
            if all(code == 0 for code in states.values()):
                break
            time.sleep(10)
        write_json(root / f'cpu-job-{job}.json', dict(allocation=job, node=os.uname().nodename,
                   state='complete', returncodes=states, updated_at=time.time()))
    except BaseException:
        for _, child in processes:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        write_json(root / f'cpu-job-{job}.json', dict(allocation=job, state='failed', traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fixed', 'extract-fixed'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--analysis-workers', type=int, default=4)
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if args.stage == 'extract-fixed':
        extract_fixed(config, args.workers)
    else:
        pipeline([
            ('cpu-extraction', ['src.research.crystallization_origin.cpu', 'extract-fixed', '--config', args.config,
                                '--workers', str(args.workers)]),
            ('cpu-ancestry', ['src.research.crystallization_origin.audit', '--config', args.config,
                              '--watch', '--workers', str(args.analysis_workers)])],
            resolve_path(config['output']) / 'technical')


if __name__ == '__main__':
    main()
