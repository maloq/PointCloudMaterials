"""Detached per-case benchmarks, with bounded failures and independent cases."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch

from src.experiment_runner.artifacts import file_hash, json_digest, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
from src.project_runtime.paths import REPO, resolve_path
from src.research.ni_relaxation_repeatability.experiment import read

FAMILY = 'ni_relaxation_speed'
MODULE = 'src.research.ni_relaxation_speed.experiment'


def prepare(c):
    out = Path(c['output'])
    parent = Path(c['parent'])
    previous = read(Path(c['repeatability'])/'technical/plan.json')
    plan = read(parent/'technical/plan.json')
    config = read(Path(read(parent/'technical/launch.json')['bundle'])/'config.json')
    keys = [j['key'] for j in previous['jobs']] + ['41743c27c0e6084b062f']
    inputs = out/'technical/inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    cases = []
    for key in keys:
        job = next(j for j in plan['response_jobs'] if j['key'] == key)
        path = parent/'technical/physical-bank'/key/'restart.pt'
        md = torch.load(path, map_location='cpu', weights_only=False)
        for kind in ['initial', 'endpoint']:
            variant = 4 if key == keys[-1] else 0
            q = md['initial'][0, :, :3] if kind == 'initial' else md['future'][0, variant, :, :3]
            name = f'{len(cases):02d}-{key}-{kind}'
            target = inputs/(name+'.npz')
            np.savez(target, positions=q.numpy(), box=md['box'],
                atom_ids=md['atom_ids'], neighbors=md['neighbors'])
            cases.append(dict(name=name, input=str(target), input_sha256=file_hash(target),
                source=job['source_key'], source_job=key, original_checkpoint=str(path),
                original_checkpoint_sha256=file_hash(path), kind=kind,
                selection='previous five-source repeatability cohort plus the recorded iteration-limit case',
                endpoint_variant=variant if kind == 'endpoint' else 0))
    payload = dict(config=c, cases=cases, parent_plan_sha256=file_hash(parent/'technical/plan.json'),
        potential=str(resolve_path(config['potential_file'])), potential_sha256=config['potential_sha256'],
        encoder_config=config, cell_atoms=32000, force_dtype='float32', state_dtype='float64')
    payload['identity'] = json_digest(payload)
    write_json(out/'technical/plan.json', payload)
    return payload


def analyze_endpoint(q, box, neighbors, p):
    from src.simulation.eam_gpu.benchmark import reference
    from src.research.ni_response.model import Motion
    from src.research.ni_response.data import full_labels
    # Same complete periodic PTM analysis used by the Ni label producer.
    labels = full_labels(q % box, box, .1)
    e, f, _, _ = reference(q % box, box, p['potential'])
    patch = q[neighbors]-q[neighbors[0]]
    patch -= box*np.rint(patch/box)
    model = Motion(p['encoder_config']).cuda().eval().requires_grad_(False)
    with torch.no_grad():
        code = model.geometry(torch.from_numpy(patch[None]).float().cuda()
                              * p['encoder_config']['length_factor']).cpu().numpy()[0]
    return dict(cpu_energy_eV=e, cpu_fmax_eV_A=float(abs(f).max()),
                crystalline_fraction=float(np.isin(labels, [1, 2, 3]).mean())), labels, patch, code


def one(c, p, case_index, profile_index):
    from .engine import warmup, native_fire, ase_minimize
    from src.data.relaxed_targets.worker import publish
    torch.set_num_threads(2)
    np.random.seed(c['seed'])
    case = p['cases'][case_index]
    profile = c['profiles'][profile_index]
    work = Path(c['scratch'])/case['name']/profile['name']
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    if file_hash(case['input']) != case['input_sha256']:
        raise ValueError('Benchmark input changed')
    data = np.load(case['input'])
    q, box = data['positions'], data['box']
    began = time.perf_counter()
    warmup(q, box, p['potential'], work)
    warmup_seconds = time.perf_counter()-began
    if profile['engine'] == 'native':
        xyz, stats = native_fire(q, box, p['potential'], profile, work)
    else:
        xyz, stats = ase_minimize(q, box, p['potential'], profile, work, c['ase_case_limit_seconds'])
    # This is a full-precision minimizer checkpoint for endpoint comparisons,
    # not a newly sampled MD trajectory or a replacement for the source data.
    np.savez_compressed(work/'minimizer-checkpoint.npz', positions=xyz, box=box,
                        atom_ids=data['atom_ids'])
    analysis, labels, patch, code = analyze_endpoint(xyz, box, data['neighbors'], p)
    np.savez_compressed(work/'observables.npz', labels=labels, patch=patch, embedding=code)
    row = dict(case=case['name'], source=case['source'], kind=case['kind'], profile=profile['name'],
        tolerance_eV_A=profile['tolerance'], atoms=len(q), **stats, **analysis,
        warmup_seconds=warmup_seconds, gpu=torch.cuda.get_device_name(),
        cpu_reference_tolerance_pass=analysis['cpu_fmax_eV_A'] <= profile['tolerance']*1.002,
        identity=p['identity'])
    write_json(work/'result.json', row)
    begin = time.perf_counter()
    publish(work, Path(c['archive'])/case['name']/profile['name'])
    row['publication_seconds'] = time.perf_counter()-begin
    write_json(Path(c['output'])/'technical/cases'/case['name']/(profile['name']+'.json'), row)
    print(json.dumps(row), flush=True)


def collect(c, p, config_path):
    out = Path(c['output'])
    tasks = [(i, j) for i in range(len(p['cases'])) for j, profile in enumerate(c['profiles'])
             if i in profile.get('case_indices', range(len(p['cases'])))]
    started = time.time()
    for number, (i, j) in enumerate(tasks):
        case, profile = p['cases'][i], c['profiles'][j]
        receipt = out/'technical/cases'/case['name']/(profile['name']+'.json')
        if receipt.exists():
            continue
        write_json(out/'technical/progress.json', dict(state='running', done=number,
            total=len(tasks), case=case['name'], profile=profile['name'], started=started))
        log = out/'technical/logs'/case['name']/(profile['name']+'.log')
        log.parent.mkdir(parents=True, exist_ok=True)
        begin = time.perf_counter()
        try:
            with log.open('w') as handle:
                subprocess.run([sys.executable, '-u', '-m', MODULE, 'case', '--config', config_path,
                    '--case-index', str(i), '--profile-index', str(j)], stdout=handle,
                    stderr=subprocess.STDOUT, check=True, timeout=c['hard_process_limit_seconds'])
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            failure = dict(case=case['name'], source=case['source'], kind=case['kind'],
                profile=profile['name'], converged=False, failure=traceback.format_exc(),
                identity=p['identity'], process_seconds=time.perf_counter()-begin)
            write_json(receipt, failure)
            from src.data.relaxed_targets.worker import publish
            work = Path(c['scratch'])/case['name']/profile['name']
            if work.exists():
                write_json(work/'failure.json', failure)
                publish(work, Path(c['archive'])/'failures'/case['name']/profile['name'])
        # Every later case remains independent of an earlier optimizer failure.
    from .report import report
    report(c, p)
    write_json(out/'technical/complete.json', dict(state='complete', identity=p['identity'], cases=len(tasks)))


def submit(c):
    from src.experiment_runner.metric_docs import check_metric_docs
    check_metric_docs(family=FAMILY)
    out = Path(c['output'])/'technical'
    if (out/'launch.json').exists():
        raise ValueError('Already submitted; use existing launch receipt')
    p = prepare(c)
    bundle = ExecutionBundle.freeze(REPO, out/'code', c,
        directories=['src', 'configs', 'docs/metrics'], files=[(REPO/'machine.local.yaml', 'machine.local.yaml')])
    dependencies = out/'dependencies'
    env = dict(PYTHONPATH=str(dependencies)+':'+str(bundle.root), PCM_PROJECT_ROOT=str(REPO),
        TORCH_EXTENSIONS_DIR=str(out/'cuda-extensions'), PYTHONFAULTHANDLER='1',
        UCX_TLS='self,sm,tcp', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        OVITO_THREAD_COUNT='1', TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', CUDA_HOME=c['cuda_home'],
        PATH=str(Path(sys.executable).parent)+':'+c['cuda_home']+'/bin:/usr/local/bin:/usr/bin:/bin')
    receipt = dict(identity=p['identity'], jobs={}, owner='current chat: requested relaxation speedup benchmarks',
        bundle=str(bundle.root), started=time.time(), dependency='isolated pyamg5.3.0, no shared environment mutation')
    queue = SlurmQueue(out, bundle, MODULE, env, out/'launch.json', receipt, 'ni-relax-speed')
    with queue.submission():
        queue.submit('collect', ['--gres=gpu:1', '--cpus-per-task=4', '--mem=32G', '--time=03:00:00'],
            partition=c['partitions'])
    return receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['submit', 'collect', 'case', 'report'])
    ap.add_argument('--config', required=True)
    ap.add_argument('--case-index', type=int)
    ap.add_argument('--profile-index', type=int)
    a = ap.parse_args()
    c = read(a.config)
    for key in ['output', 'scratch', 'archive', 'parent', 'repeatability', 'cuda_home']:
        c[key] = str(resolve_path(c[key]))
    if a.stage == 'submit':
        print(submit(c)); return
    p = read(Path(c['output'])/'technical/plan.json')
    if a.stage == 'case': one(c, p, a.case_index, a.profile_index)
    elif a.stage == 'collect': collect(c, p, str(Path(a.config).resolve()))
    else:
        from .report import report
        report(c, p)


if __name__ == '__main__':
    main()
