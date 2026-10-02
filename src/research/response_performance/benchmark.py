"""Local, frozen, full-horizon throughput and equivalence measurements; no fitting."""
import argparse
import gc
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch

from src.experiment_runner.artifacts import file_hash as sha, write_json, json_digest
from src.experiment_runner.execution import ExecutionBundle
from src.experiment_runner.metric_docs import write_metric_rows, check_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.response_atlas.atomistic import configuration, MACEPotential, PathFeatures, oracle_for
from src.research.response_atlas.reference import ShotBundle
from .oracle import ConditionalOracle, GPUPotential, BatchedFeatures, BatchedOracle, force_and_tangent

FAMILY = 'response_performance'


def read(path):
    return json.loads(Path(path).read_text())


def fixture(c, parent):
    base = resolve_path(c['reference_run'])
    binding = read(base/'technical/binding.json')
    if binding['identity'] != c['reference_identity']:
        raise ValueError('Historical reference identity changed')
    repo = Path(__file__).resolve().parents[3]
    for relative in ('src/research/response_atlas/reference.py', 'src/research/response_atlas/atomistic.py'):
        if sha(repo/relative) != binding['binding']['sources'][relative]:
            raise ValueError(f'Historical numerical source changed: {relative}')
    folder = resolve_path(c['reference_archive'])/f'parent-{parent:03d}'
    receipt = read(folder/'complete.json')
    if receipt['identity'] != c['reference_identity'] or sha(folder/'query.pt') != receipt['sha256']:
        raise ValueError('Historical query changed')
    return torch.load(folder/'query.pt', map_location='cpu', weights_only=False)


def errors(value, reference):
    delta = value.double()-reference.to(value.device).double()
    return dict(relative=float(delta.norm()/reference.double().norm().clamp_min(1e-12)),
                maximum_absolute=float(delta.abs().max()))


def make(c, variant, parent=0):
    o = read(resolve_path(c['oracle_recipe']))
    o['horizons_steps'] = [20, 100]
    dtype = getattr(torch, variant['dtype'])
    saved = fixture(c, parent)
    q = saved['q'].flatten().to(device='cuda', dtype=dtype)
    basis = saved['basis'].reshape(len(q), 2).to(device='cuda', dtype=dtype)
    atoms = configuration(o, parent)
    if variant['graph'] == 'ase':
        potential = MACEPotential(o, atoms, 'cuda')
        features = PathFeatures(atoms.cell.lengths(), o['horizons_steps'], o, 'cuda')
        oracle = oracle_for(o, potential, features, q, o['horizons_steps'])
        if variant['conditional']:
            oracle.__class__ = ConditionalOracle
    else:
        potential = GPUPotential(o, atoms, backend=variant['backend'], dtype=dtype, skin=variant['skin_A'])
        features = BatchedFeatures(atoms.cell.lengths(), o, dtype)
        oracle = BatchedOracle(potential, features, o)
    return oracle, potential, q, basis, saved


def static_gate(c, potential, q, basis, parent):
    o = read(resolve_path(c['oracle_recipe']))
    reference = MACEPotential(o, configuration(o, parent), 'cuda')
    q64, b64 = q.double(), basis.double()
    energy64 = reference(q64.detach().requires_grad_(True)).detach()
    force64, hvp64 = force_and_tangent(reference, q64, b64)
    energy = potential(q.detach().requires_grad_(True)).detach()
    force, hvp = force_and_tangent(potential, q, basis)
    record = dict(energy=errors(energy, energy64), force=errors(force, force64), hvp=errors(hvp, hvp64))
    moved = q.reshape(-1, 3).clone()
    moved[17, 0] += o['lattice_A']*4
    moved_energy = potential(moved.flatten().requires_grad_(True)).detach()
    moved_force, moved_hvp = force_and_tangent(potential, moved.flatten(), basis)
    record['periodic_energy'] = errors(moved_energy, energy)
    record['periodic_force'] = errors(moved_force, force)
    record['periodic_hvp'] = errors(moved_hvp, hvp)
    del reference
    gc.collect(); torch.cuda.empty_cache()
    return record


def variant_run(c, variant):
    out = resolve_path(c['output'])/'analyses/benchmark-v1'
    name = variant['name']
    destination = out/'technical/variants'/f'{name}.json'
    if destination.exists():
        print(f'Already measured: {name}', flush=True)
        return
    record = dict(variant=variant, state='running', started_at=time.time(),
        device=torch.cuda.get_device_name(), measurements=[], gates=[])
    write_json(destination, record)
    try:
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(.90)
        oracle, potential, q, basis, saved = make(c, variant)
        gate = static_gate(c, potential, q, basis, 0)
        record['gates'].append(dict(parent=0, static=gate))
        tolerance = c['tolerance'][variant['dtype']]
        for field in ('force', 'hvp', 'periodic_force', 'periodic_hvp'):
            if gate[field]['relative'] > tolerance['static_relative']:
                raise ValueError(f'{name} static {field} gate failed: {gate[field]}')
        # Both modes receive their own warm-up; setup/conversion is excluded.
        for kind in ('value', 'response'):
            directions = basis if kind == 'response' else q.new_empty(len(q), 0)
            force_and_tangent(potential, q, directions)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            before_calls = potential.calls
            began = time.monotonic()
            seeds = saved['value_seeds'][:variant['batch']]
            method = variant.get('method', 'ad')
            if method == 'fd' and kind == 'response':
                epsilon = variant['epsilon_A']
                initial = torch.stack([q, q+epsilon*basis[:,0], q-epsilon*basis[:,0],
                                       q+epsilon*basis[:,1], q-epsilon*basis[:,1]])
                vals, _ = oracle.simulate(initial, initial.new_empty(5,len(q),0), [seeds[0]]*5)
                derivative = torch.stack(((vals[1]-vals[2])/(2*epsilon),
                                          (vals[3]-vals[4])/(2*epsilon)), -1)[None]
                bundle = ShotBundle(q, basis, vals[:1], derivative, tuple(seeds))
                bundle.validate()
            else:
                bundle = oracle.query(q, directions, seeds)
            torch.cuda.synchronize(); elapsed = time.monotonic()-began
            value_error = errors(bundle.values, saved['values'][:len(seeds)])
            response_error = None
            if kind == 'response':
                # Archive contains8 AD labels; all32 ordinary values are available.
                count = min(len(seeds), len(saved['responses']))
                response_error = errors(bundle.responses[:count], saved['responses'][:count])
            row = dict(variant=name, kind=kind, backend=variant['backend'], dtype=variant['dtype'],
                graph=variant['graph'], skin_A=variant['skin_A'], replicas=len(seeds),
                method=method, epsilon_A=variant.get('epsilon_A'),
                trajectory_executions=5 if method=='fd' and kind=='response' else len(seeds),
                seconds=elapsed, seconds_per_trajectory=elapsed/len(seeds),
                trajectories_per_gpu_hour=3600*len(seeds)/elapsed,
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                model_calls=potential.calls-before_calls,
                value_relative_error=value_error['relative'], value_max_abs_error=value_error['maximum_absolute'],
                response_relative_error=response_error['relative'] if response_error else None,
                response_max_abs_error=response_error['maximum_absolute'] if response_error else None,
                response_reference_replicas=min(len(seeds),8) if kind=='response' else 0)
            row['passed'] = (value_error['relative'] <= tolerance['path_relative'] and
                value_error['maximum_absolute'] <= tolerance['path_absolute'] and
                (response_error is None or (response_error['relative'] <= tolerance['path_relative'] and
                 response_error['maximum_absolute'] <= tolerance['path_absolute'])))
            record['measurements'].append(row)
            torch.save(dict(seeds=seeds, values=bundle.values.cpu(), responses=bundle.responses.cpu()),
                       out/'technical/variants'/f'{name}-{kind}.pt')
            write_json(destination, record)
            print(json.dumps(row), flush=True)
            if not row['passed']:
                raise ValueError(f'{name}/{kind} full100fs path gate failed')
        for parent in variant.get('additional_gate_parents', []):
            reference = fixture(c, parent)
            point = reference['q'].flatten().to(q)
            directions = reference['basis'].reshape(len(q),2).to(q)
            result = oracle.query(point, directions, reference['value_seeds'][:1])
            value_error = errors(result.values, reference['values'][:1])
            response_error = errors(result.responses, reference['responses'][:1])
            passed = all(e['relative'] <= tolerance['path_relative'] and
                         e['maximum_absolute'] <= tolerance['path_absolute']
                         for e in (value_error, response_error))
            record['gates'].append(dict(parent=parent, value=value_error, response=response_error, passed=passed))
            write_json(destination, record)
            if not passed:
                raise ValueError(f'{name} additional parent{parent} failed: {record["gates"][-1]}')
        record['state'] = 'complete'
    except Exception as error:
        record.update(state='failed', error=repr(error), traceback=traceback.format_exc())
        print(json.dumps(dict(variant=name, state='failed', error=repr(error))), flush=True)
    finally:
        record['finished_at'] = time.time()
        write_json(destination, record)


def aggregate(c):
    out = resolve_path(c['output'])/'analyses/benchmark-v1'
    rows, failures = [], []
    for variant in c['variants']:
        p = out/'technical/variants'/f'{variant["name"]}.json'
        if not p.exists():
            continue
        record = read(p)
        rows.extend(record['measurements'])
        if record['state'] != 'complete':
            failures.append(dict(variant=variant['name'], state=record['state'], error=record.get('error')))
    if rows:
        write_metric_rows(rows, out, family=FAMILY, name='throughput')
    write_json(out/'technical/status.json', dict(measured_variants=len({r['variant'] for r in rows}),
        requested_variants=len(c['variants']), failures=failures,
        note='A measured failed variant is explicitly rejected; no numerical fallback'))


def run(config):
    c = read(config)
    check_metric_docs(family=FAMILY)
    root = resolve_path(c['output']); tech = root/'technical'
    if not (tech/'code').exists():
        bundle = ExecutionBundle.freeze(Path(__file__).resolve().parents[3], tech/'code', c,
            directories=('src', 'docs/metrics', 'configs/simulation', 'configs/analysis'))
    else:
        bundle = ExecutionBundle(tech/'code')
        if read(bundle.config_path) != c:
            raise ValueError('Benchmark recipe changed; use a new run revision')
    (root/'analyses/benchmark-v1/technical/variants').mkdir(parents=True, exist_ok=True)
    write_json(tech/'binding.json', dict(identity=json_digest(c), config=c,
        implementation={str(p.relative_to(Path(__file__).parent)):sha(p)
                        for p in Path(__file__).parent.glob('*.py')},
        packages={k:version(k) for k in ('torch','mace-torch','e3nn','cuequivariance-torch')}))
    for variant in c['variants']:
        print(f'Benchmarking {variant["name"]}', flush=True)
        subprocess.run([sys.executable, '-u', '-m', 'src.research.response_performance.benchmark',
            'variant', '--config', str(bundle.config_path), '--name', variant['name']],
            cwd=bundle.root, check=True)
        aggregate(c)
    write_json(tech/'complete.json', dict(state='complete', finished_at=time.time(),
        numerical_failures_are_recorded=True, scientific_training=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=('run', 'variant', 'aggregate'))
    parser.add_argument('--config', required=True)
    parser.add_argument('--name')
    args = parser.parse_args()
    if args.stage == 'run':
        run(args.config)
    elif args.stage == 'aggregate':
        aggregate(read(args.config))
    else:
        config = read(args.config)
        variant_run(config, next(v for v in config['variants'] if v['name']==args.name))
