"""Matched train-only force-engine experiments; resumable, locally tracked."""
import argparse
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
import torch

from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read, save, sha, write_json, digest
from src.research.local_response.oracle import LocalFeatures
from .engines import make_engine, checked_files
from .oracle import FrozenFeatures, ForceOracle

FAMILY = 'response_force_engines'


def paths(c):
    return resolve_path(c['output']), resolve_path(c['simulation_scratch']), resolve_path(c['simulation_archive'])


def error(a, b):
    a, b = torch.as_tensor(a).double().flatten(), torch.as_tensor(b).double().flatten()
    delta = a - b
    denominator = b.norm().clamp_min(1e-15)
    return dict(rms=float(delta.square().mean().sqrt()), relative=float(delta.norm() / denominator),
                cosine=float(a @ b / (a.norm() * denominator).clamp_min(1e-30)))


def publish_payload(c, name, payload):
    _, scratch, archive = paths(c)
    target, destination = scratch / name, archive / name
    save(target, payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.copying.pt')
    shutil.copy2(target, temporary)
    if sha(temporary) != sha(target):
        raise IOError(f'Archive verification failed: {target}')
    temporary.replace(destination)
    return destination


def prepare(c):
    out, _, archive = paths(c)
    reference = resolve_path(c['reference'])
    binding = read(reference / 'technical/binding.json')
    if binding['identity'] != c['reference_identity']:
        raise ValueError('Unexpected reference experiment identity')
    oracle_path = Path(LocalFeatures.__module__.replace('.', '/') + '.py')
    if sha(oracle_path) != binding['implementation'][str(oracle_path)]:
        raise ValueError('LocalFeatures producer differs from frozen reference')
    parents_path = reference / 'technical/parents.pt'
    if sha(parents_path) != read(reference / 'technical/parents.json')['sha256']:
        raise ValueError('Reference parents changed')
    for spec in c['engines'].values():
        checked_files(spec)
    all_states = torch.load(parents_path, weights_only=False)
    states = [all_states[i] for i in c['parents']]
    if any(s['role'] != 'train' for s in states):
        raise ValueError('Engineering/numerical selection must only use train sources')
    oracle_config = binding['config']['oracle']
    rff = LocalFeatures(oracle_config, torch.float32)
    feature_payload = dict(cutoff=rff.cutoff, scale=rff.scale, centers=rff.centers.cpu(),
                           omega=[x.cpu() for x in rff.omega], phase=[x.cpu() for x in rff.phase])
    cpu_features = FrozenFeatures(feature_payload)
    sanity_q = states[0]['q'][:80].float().cuda().flatten()[None]
    feature_error = error(cpu_features.observed(sanity_q.cpu().double()), rff.observed(sanity_q).cpu())
    if feature_error['rms'] > 1e-6:
        raise ValueError(f'CPU observable disagrees with original: {feature_error}')
    reference_hashes = {}
    for state in states:
        index = state['index']
        for radius in c['radii_A']:
            name = f'parent{index}-radius{radius}.pt'
            source = reference / 'technical/gates' / name
            target = out / 'technical/reference' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            reference_hashes[name] = sha(target)
        reference_data = torch.load(out / 'technical/reference' / f'parent{index}-radius{c["radii_A"][0]}.pt', weights_only=False)
        for seed in reference_data['seeds']:
            name = f'noise/parent{index}-seed{seed}.pt'
            if (archive / name).exists():
                continue
            rng = torch.Generator(device='cuda').manual_seed(seed)
            rows = state['atom_rows'].cuda()
            # Exactly the original full-cell CUDA float64 stream; save only the
            # maximum nested environment. CPU RNGs are not interchangeable.
            noise = torch.stack([torch.randn(state['original_atoms'], 3, device='cuda',
                dtype=torch.float64, generator=rng)[rows].cpu()
                for _ in range(max(oracle_config['horizons_steps']) + 1)])
            publish_payload(c, name, noise)
            print(dict(stage='noise-prepared', parent=index, seed=seed), flush=True)
    payload = dict(states=states, features=feature_payload, oracle=oracle_config)
    publish_payload(c, 'inputs.pt', payload)
    source_files = sorted(Path(__file__).parent.glob('*.py')) + [oracle_path]
    implementation = {str(p): sha(p) for p in source_files}
    record = dict(config=c, implementation=implementation, parents_sha256=sha(parents_path),
                  inputs_sha256=sha(archive / 'inputs.pt'), reference_hashes=reference_hashes,
                  noise_hashes={str(p.relative_to(archive)): sha(p) for p in (archive / 'noise').glob('*.pt')},
                  feature_cpu_float64_vs_gpu_float32=feature_error,
                  versions={p: importlib.metadata.version(p) for p in ['calorine', 'lammps', 'ase', 'torch', 'numpy']})
    record['identity'] = digest(record)
    write_json(out / 'technical/binding.json', record)
    write_json(out / 'technical/prediction-context.json', dict(
        encoder_inputs='No encoder trained by this benchmark. Proposed fits retain nearest80 geometry-only inputs.',
        predictor_inputs='No predictor fitted. No temperature, simulation age, time, motion, material ID or species input.',
        teacher=dict(support='moving open spheres; 18/24/36Å; actual atom counts recorded',
                     history=0, initial_momenta='resampled shared CUDA noise', conditions=oracle_config,
                     precision='float64 CPU BAOAB and forces; same frozen float32 RFF coefficients promoted to float64',
                     targets='same256 Fourier coordinates and two local80 derivative directions',
                     potential_change='UNEP differs from MACE and source MEAM; MEAM restores parent potential, but open BAOAB is not the original full-cell trajectory'),
        ancestry='four training sources from fixed Al64 release; no held-out numerical selection'))
    print(dict(stage='prepared', identity=record['identity']), flush=True)


def audit_engine(spec, q, basis, half_box):
    engine = make_engine(spec, q, half_box)
    larger = make_engine(spec, q, half_box + 12)
    try:
        energy, force = engine(q)
        energy2, force2 = larger(q)
        vacuum = error(force2, force)
        eps = 1e-4
        ep, _ = engine(q + eps * basis[..., 0])
        em, _ = engine(q - eps * basis[..., 0])
        fd = -(ep - em) / (2 * eps)
        projected_force = float((force * basis[..., 0]).sum())
        directional_error = abs(fd - projected_force) / max(abs(projected_force), 1e-8)
        if vacuum['relative'] > 1e-9 or abs(energy2 - energy) / len(q) > 1e-9 or directional_error > 1e-4:
            raise ValueError(f'Force-engine audit failed: vacuum={vacuum}, gradient={directional_error}')
        return dict(vacuum_force=vacuum, energy_box_difference_eV=energy2-energy,
                    directional_force=projected_force, energy_fd_force=fd,
                    energy_gradient_relative=directional_error)
    finally:
        engine.close()
        larger.close()


def run(c, name):
    out, _, archive = paths(c)
    record = read(out / 'technical/binding.json')
    if record['config'] != c or sha(archive / 'inputs.pt') != record['inputs_sha256']:
        raise ValueError('Inputs/config changed after preparation')
    for filename, expected in record['implementation'].items():
        if sha(Path(filename)) != expected:
            raise ValueError(f'Implementation changed: {filename}; prepare a new revision')
    torch.set_num_threads(1)
    inputs = torch.load(archive / 'inputs.pt', weights_only=False)
    features = FrozenFeatures(inputs['features'])
    spec = c['engines'][name]
    write_json(out / 'technical' / f'{name}-execution.json', dict(
        hostname=platform.node(), pid=os.getpid(), job=os.environ.get('SLURM_JOB_ID'),
        threads=torch.get_num_threads(), OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'),
        start_unix=time.time(), identity=record['identity']))
    for state in inputs['states']:
        index = state['index']
        for radius in c['radii_A']:
            ref_path = out / 'technical/reference' / f'parent{index}-radius{radius}.pt'
            if sha(ref_path) != record['reference_hashes'][ref_path.name]:
                raise ValueError(f'Reference changed: {ref_path}')
            reference = torch.load(ref_path, weights_only=False)
            mask = state['q'].norm(dim=-1) < radius
            q = state['q'][mask].numpy().copy()
            basis = np.zeros((*q.shape, 2), dtype=np.float64)
            basis[:80] = state['basis'].numpy()
            half_box = radius + c['vacuum_margin_A']
            audit_path = out / 'technical/audits' / f'{name}-parent{index}-radius{radius}.json'
            if not audit_path.exists():
                write_json(audit_path, audit_engine(spec, q, basis, half_box))
            engine = make_engine(spec, q, half_box)
            try:
                oracle = ForceOracle(engine, features, inputs['oracle'])
                # Warmup is outside timed bundles; engines persist across all shots.
                engine(q)
                for shot, seed in enumerate(reference['seeds']):
                    target = f'{name}/parent{index}-radius{radius}-shot{shot}.pt'
                    if (archive / target).exists():
                        continue
                    noise_name = f'noise/parent{index}-seed{seed}.pt'
                    if sha(archive / noise_name) != record['noise_hashes'][noise_name]:
                        raise ValueError(f'Noise tape changed: {noise_name}')
                    noise = torch.load(archive / noise_name, weights_only=False)[:, mask].numpy()
                    print(dict(stage='query', engine=name, parent=index, radius_A=radius,
                               atoms=len(q), seed=seed), flush=True)
                    result = oracle.query(q, basis, noise, c['epsilons_A'])
                    if index == c['parents'][0] and radius == c['radii_A'][0] and shot == 0:
                        repeat = oracle.path_value(q, noise)
                        result['repeat_value_error'] = error(repeat, result['values'])
                        if result['repeat_value_error']['rms'] > 1e-10:
                            raise ValueError('Repeated common-noise path changed')
                    result.update(engine=name, parent=index, radius_A=radius, atoms=len(q),
                                  seed=seed, shot=shot, identity=record['identity'])
                    destination = publish_payload(c, target, result)
                    print(dict(stage='saved', file=str(destination), seconds=result['bundle_seconds']), flush=True)
            finally:
                engine.close()
    write_json(out / 'technical' / f'{name}-complete.json', dict(identity=record['identity'], end_unix=time.time()))
    summarize(c)


def summarize(c):
    out, _, archive = paths(c)
    rows = []
    for name in c['engines']:
        for index in c['parents']:
            for radius in c['radii_A']:
                ref = torch.load(out / 'technical/reference' / f'parent{index}-radius{radius}.pt', weights_only=False)
                for shot, seed in enumerate(ref['seeds']):
                    file = archive / name / f'parent{index}-radius{radius}-shot{shot}.pt'
                    if not file.exists():
                        continue
                    payload = torch.load(file, weights_only=False)
                    finest = payload['responses'][str(c['epsilons_A'][-1])]
                    for epsilon in c['epsilons_A']:
                        response = payload['responses'][str(epsilon)]
                        fd = error(response, finest)
                        for horizon, lo, hi in [('20fs', 0, 128), ('100fs_joint', 128, 256)]:
                            physics = error(response[lo:hi], ref['responses'][shot, lo:hi])
                            values = error(payload['values'][lo:hi], ref['values'][shot, lo:hi])
                            reference_seconds = ref['seconds'] / len(ref['seeds'])
                            rows.append(dict(engine=name, parent=index, radius_A=radius, atoms=payload['atoms'],
                                shot=shot, seed=seed, epsilon_A=epsilon, horizon=horizon,
                                value_seconds=payload['value_seconds'], bundle_seconds=payload['bundle_seconds'][str(epsilon)],
                                reference_mace_ad_seconds_per_shot=reference_seconds,
                                reference_gpu_to_cpu_wall_ratio=reference_seconds/payload['bundle_seconds'][str(epsilon)],
                                fd_relative_to_smallest_epsilon=fd['relative'],
                                response_relative_to_mace=physics['relative'], response_cosine_to_mace=physics['cosine'],
                                value_rms_to_mace=values['rms']))
    if rows:
        write_metric_rows(rows, out / 'analyses/comparison-v1', family=FAMILY, name='engine-comparison')
    boundary = []
    for name in c['engines']:
        for index in c['parents']:
            for shot in range(2):
                files = [archive/name/f'parent{index}-radius{r}-shot{shot}.pt' for r in c['radii_A']]
                if not all(p.exists() for p in files):
                    continue
                payloads = [torch.load(p, weights_only=False) for p in files]
                for i, a in enumerate(payloads[:-1]):
                    for j in range(i + 1, len(payloads)):
                        b = payloads[j]
                        for eps in c['epsilons_A']:
                            for horizon, lo, hi in [('20fs', 0, 128), ('100fs_joint', 128, 256)]:
                                ve = error(a['values'][lo:hi], b['values'][lo:hi])
                                re = error(a['responses'][str(eps)][lo:hi], b['responses'][str(eps)][lo:hi])
                                boundary.append(dict(engine=name, parent=index, shot=shot, radius_A=c['radii_A'][i],
                                    reference_A=c['radii_A'][j], epsilon_A=eps, horizon=horizon,
                                    value_rms=ve['rms'], response_relative=re['relative'],
                                    passed=ve['rms'] <= c['value_rms_tolerance'] and re['relative'] <= c['response_relative_tolerance']))
    if boundary:
        write_metric_rows(boundary, out / 'analyses/comparison-v1', family=FAMILY, name='boundary-convergence')
    write_json(out / 'technical/progress.json', dict(completed_shots=len(rows)//(len(c['epsilons_A'])*2),
        expected_shots=len(c['engines'])*len(c['parents'])*len(c['radii_A'])*2,
        interpretation='Wall-time ratio compares one CPU thread with cached GPU AD; changed potential and float64 dynamics, not MACE numerical equivalence.'))


def launch(c):
    from src.experiment_runner.execution import ExecutionBundle
    out, _, _ = paths(c)
    receipt = out / 'technical/launch.json'
    if receipt.exists():
        raise ValueError('Already launched; use the frozen run command for explicit resumption')
    if 'SLURM_JOB_ID' not in os.environ:
        raise RuntimeError('Detach only inside an allocated compute job')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, out / 'technical/code', c,
                                    directories=('src', 'docs/metrics'))
    env = dict(os.environ, PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1', CUDA_VISIBLE_DEVICES='')
    processes = {}
    for name in c['engines']:
        logfile = out / 'technical' / f'{name}.log'
        command = [sys.executable, '-u', '-m', 'src.research.response_force_engines.benchmark',
                   'run', '--config', str(bundle.config_path), '--engine', name]
        with logfile.open('a') as log:
            process = subprocess.Popen(command, cwd=bundle.root, env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        processes[name] = dict(pid=process.pid, command=command, log=str(logfile))
    write_json(receipt, dict(processes=processes, job=os.environ['SLURM_JOB_ID'],
                            hostname=platform.node(), code=str(bundle.root), start_unix=time.time()))
    print(read(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'launch', 'run', 'summarize'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--engine')
    args = parser.parse_args()
    c = read(args.config)
    torch.set_num_threads(1)
    if args.stage == 'prepare':
        prepare(c)
    elif args.stage == 'launch':
        launch(c)
    elif args.stage == 'run':
        if args.engine not in c['engines']:
            parser.error('--engine must name a configured engine')
        from src.experiment_runner.execution import recorded_stage
        with recorded_stage(paths(c)[0] / 'technical' / f'{args.engine}-stage.json',
                            job=os.environ.get('SLURM_JOB_ID')):
            run(c, args.engine)
    else:
        summarize(c)


if __name__ == '__main__':
    main()
