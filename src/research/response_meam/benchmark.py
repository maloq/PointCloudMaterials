"""Validate an exact MEAM port and time complete 20/100fs response queries.

Numerical gates and hardware measurements remain local, with no model training.
All cases retain the four original training ancestors and shared noise streams.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import torch

from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read, write_json, save, sha, digest
from src.research.local_response.oracle import LocalOracle, OpenPotential, Teacher
from src.research.response_geometry.features import Features
from src.research.response_force_engines.engines import MEAM
from src.research.response_force_engines.oracle import ForceOracle
from src.research.response_performance.oracle import force_and_tangent
from .potential import Lee2003Al, FILE_HASHES

FAMILY = 'response_meam_torch'


def error(a, b):
    a, b = torch.as_tensor(a).detach().double().cpu(), torch.as_tensor(b).detach().double().cpu()
    difference = a-b
    return dict(rms=float(difference.square().mean().sqrt()),
                relative=float(difference.norm()/b.norm().clamp_min(1e-15)),
                maximum=float(difference.abs().max()))


def features(oracle, normalizer, dtype, device='cuda'):
    # Generate the actual float32 campaign maps once, then promote their stored
    # coefficients. Precision comparisons must not also change the target map.
    f = Features(oracle, normalizer, torch.float32)
    move = lambda x: x.to(device=device, dtype=dtype)
    f.old.centers = move(f.old.centers)
    f.old.omega = [move(x) for x in f.old.omega]
    f.old.phase = [move(x) for x in f.old.phase]
    f.mean, f.scale = move(f.mean), move(f.scale)
    f.maps = {k: [(move(w), move(b)) for w, b in value] for k, value in f.maps.items()}
    return f


def inputs(c):
    reference = resolve_path(c['force_reference'])
    original = read(reference/'technical/binding.json')
    if original['identity'] != c['force_reference_identity']:
        raise ValueError('Force-engine reference identity changed')
    archive = resolve_path(c['input_archive'])
    if sha(archive/'inputs.pt') != original['inputs_sha256']:
        raise ValueError('Original parent inputs changed')
    payload = torch.load(archive/'inputs.pt', weights_only=False)
    if [s['index'] for s in payload['states']] != c['parents']:
        raise ValueError('Benchmark parent identities changed')
    if any(s['role'] != 'train' for s in payload['states']):
        raise ValueError('Only training ancestors may enter engineering selection')
    geo = resolve_path(c['geometry_reference'])
    binding = read(geo/'technical/binding.json')
    if binding['identity'] != c['geometry_reference_identity']:
        raise ValueError('Rich-target campaign identity changed')
    norm_file = geo/'technical/feature-normalization.pt'
    if sha(norm_file) != read(geo/'technical/parents.json')['normalizer_sha256']:
        raise ValueError('Training-only observable normalization changed')
    normalizer = torch.load(norm_file, weights_only=False)
    return payload, normalizer, original


def bind(c, original):
    root = Path(__file__).resolve().parents[3]
    files = list(Path(__file__).parent.glob('*.py')) + [root/p for p in (
        'src/research/local_response/oracle.py', 'src/research/response_performance/oracle.py',
        'src/research/response_geometry/features.py', 'src/research/response_force_engines/engines.py',
        'src/research/response_force_engines/oracle.py', 'docs/metrics/response_meam_torch.md')]
    record = dict(config=c, implementation={str(p.relative_to(root)):sha(p) for p in files},
                  input_hash=original['inputs_sha256'], noise_hashes=original['noise_hashes'],
                  potential_hashes=FILE_HASHES)
    record['identity'] = digest(record)
    out = resolve_path(c['output'])
    path = out/'technical/binding.json'
    if path.exists() and read(path) != record:
        raise ValueError('Frozen implementation/config changed; use a new output revision')
    write_json(path, record)
    for source in files:
        target = out/'technical/code'/source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
    write_json(out/'technical/execution.json', dict(
        host=platform.node(), pid=os.getpid(), allocation=os.environ.get('SLURM_JOB_ID'),
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
        threads=torch.get_num_threads(), start_unix=time.time(), identity=record['identity'],
        target='All three current target banks, 768 features, two displacement directions',
        interpretation='Hardware benchmark only; no encoder/predictor fit or online tracking',
        encoder_inputs=None, predictor_inputs=None, relaxation=False,
        ancestry='Four existing unrelaxed fixed-Al64 train ancestors; no new independent sources'))
    return record


def query_data(c, state, original):
    mask = state['q'].norm(dim=-1) < c['radius_A']
    q = state['q'][mask]
    basis = torch.zeros(len(q), 3, 2, dtype=torch.float64)
    basis[:80] = state['basis']
    ref_path = resolve_path(c['force_reference'])/'technical/reference'/f'parent{state["index"]}-radius{c["radius_A"]}.pt'
    if sha(ref_path) != original['reference_hashes'][ref_path.name]:
        raise ValueError('Reference query changed')
    seeds = torch.load(ref_path, weights_only=False)['seeds']
    tapes=[]
    for seed in seeds:
        name=f'noise/parent{state["index"]}-seed{seed}.pt'
        path=resolve_path(c['input_archive'])/name
        if sha(path)!=original['noise_hashes'][name]:
            raise ValueError(f'Noise tape changed: {path}')
        tapes.append(torch.load(path, weights_only=False)[:, mask])
    return q, basis, state['atom_rows'][mask], seeds, tapes


def make(c, state, q, rows, settings, normalizer, name, dtype, teacher=None):
    if name == 'meam':
        potential = Lee2003Al({k:resolve_path(v) for k,v in c['meam_files'].items()}, len(q), dtype=dtype)
    else:
        potential = OpenPotential(teacher, len(q), settings['skin_A'])
    return LocalOracle(potential, features(settings, normalizer, dtype), settings,
                       rows.cuda(), state['original_atoms'])


def published(c, name, payload):
    scratch, archive = resolve_path(c['simulation_scratch']), resolve_path(c['simulation_archive'])
    target, destination = scratch/name, archive/name
    save(target, payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary=destination.with_suffix('.copying.pt')
    shutil.copy2(target, temporary)
    if sha(target)!=sha(temporary):
        raise IOError(f'Archive mismatch: {name}')
    temporary.replace(destination)


def validate(c, payload, normalizer, original):
    out=resolve_path(c['output'])
    rows=[]
    spec=dict(engine='meam',files={k:dict(path=v,sha256=FILE_HASHES[k]) for k,v in c['meam_files'].items()})
    for state in payload['states']:
        q,basis,atom_rows,seeds,tapes=query_data(c,state,original)
        engine=MEAM(spec,q.numpy(),float(q.abs().max())+12)
        try:
            reference_energy, reference_force=engine(q.numpy())
            oracle=make(c,state,q,atom_rows,payload['oracle'],normalizer,'meam',torch.float64)
            x=q.cuda().flatten()[None]
            directions=basis.cuda().reshape(1,-1,2)
            energy=oracle.potential(x)
            force,tangent=force_and_tangent(oracle.potential,x,directions)
            e_error=abs(float(energy)-reference_energy)/len(q)
            force_error=error(force.reshape(-1,3),reference_force)
            fd=[]
            epsilon=c['hvp_epsilon_A']
            for j in range(2):
                _,plus=engine((q+epsilon*basis[...,j]).numpy())
                _,minus=engine((q-epsilon*basis[...,j]).numpy())
                fd.append(torch.from_numpy((plus-minus)/(2*epsilon)))
            tangent_error=error(tangent.reshape(-1,3,2),torch.stack(fd,-1))
            if (e_error>c['energy_per_atom_tolerance_eV'] or force_error['rms']>c['force_rms_tolerance_eV_A']
                    or tangent_error['relative']>c['hvp_relative_tolerance']):
                raise ValueError(f'MEAM force/derivative gate failed: parent{state["index"]}, energy={e_error}, force={force_error}, tangent={tangent_error}')
            cpu=ForceOracle(engine,features(payload['oracle'],normalizer,torch.float64,'cpu'),payload['oracle'])
            # One full CPU paired-trajectory check per ancestor, all 768 outputs.
            cpu_query=cpu.query(q.numpy(),basis.numpy(),tapes[0].numpy(),[c['response_fd_epsilon_A']])
            value,response=oracle.simulate(x,directions,seeds[:1])
            target_error=error(value[0],cpu_query['values'])
            response_error=error(response[0],cpu_query['responses'][str(c['response_fd_epsilon_A'])])
            if target_error['rms']>c['value_rms_tolerance'] or response_error['relative']>c['response_relative_tolerance']:
                raise ValueError(f'Complete trajectory gate failed: {state["index"]}: {target_error}, {response_error}')
            single=make(c,state,q,atom_rows,payload['oracle'],normalizer,'meam',torch.float32)
            v32,r32=single.simulate(x.float(),directions.float(),seeds[:1])
            precision_value=error(v32,value)
            precision_response=error(r32,response)
            if precision_value['rms']>c['value_rms_tolerance'] or precision_response['relative']>c['response_relative_tolerance']:
                raise ValueError(f'Float32 response gate failed: {state["index"]}: {precision_value}, {precision_response}')
            row=dict(parent=state['index'],source=state['source'],atoms=len(q),
                energy_per_atom_error_eV=e_error,force_rms_error_eV_A=force_error['rms'],
                force_tangent_relative_error=tangent_error['relative'],
                cpu_value_rms=target_error['rms'],cpu_response_relative=response_error['relative'],
                float32_value_rms=precision_value['rms'],float32_response_relative=precision_response['relative'],passed=True)
            rows.append(row)
            published(c,f'validation/parent{state["index"]}.pt',dict(cpu=cpu_query,
                value64=value.cpu(),response64=response.cpu(),value32=v32.cpu(),response32=r32.cpu(),metrics=row))
            write_json(out/'technical/validation-progress.json',rows)
            print(dict(stage='validated',**row),flush=True)
        finally:
            engine.close()
    write_metric_rows(rows,out/'analyses/validation-v1',family=FAMILY,name='validation')
    write_json(out/'technical/validation-complete.json',dict(parents=c['parents'],passed=True))


def measure(c,payload,normalizer,original):
    out=resolve_path(c['output'])
    if not read(out/'technical/validation-complete.json')['passed']:
        raise ValueError('Timing requires validated response derivatives')
    teacher=Teacher(payload['oracle'],torch.float32)
    timings=[]
    variants=[('mace',torch.float32),('meam',torch.float64),('meam',torch.float32)]
    for state in payload['states']:
        q,basis,atom_rows,seeds,_=query_data(c,state,original)
        for batch in c['batch_sizes']:
            if len(seeds)%batch:
                raise ValueError('Batch must divide the two declared independent noise seeds')
            oracles={(name,str(dtype)):make(c,state,q,atom_rows,payload['oracle'],normalizer,name,dtype,teacher)
                     for name,dtype in variants}
            # A complete untimed warmup for each shape/variant includes AD kernels,
            # neighbor construction and observable derivative initialization.
            for name,dtype in variants:
                oracle=oracles[name,str(dtype)]
                x=q.cuda().to(dtype).flatten()[None].repeat(batch,1)
                v=basis.cuda().to(dtype).reshape(1,-1,2).repeat(batch,1,1)
                oracle.simulate(x,v,seeds[:batch]);torch.cuda.synchronize()
            for repeat in range(c['repeats']):
                # Rotate order to reduce a fixed GPU-temperature/order advantage.
                ordered=variants[repeat%len(variants):]+variants[:repeat%len(variants)]
                for start in range(0,len(seeds),batch):
                    shot_seeds=seeds[start:start+batch]
                    for name,dtype in ordered:
                        oracle=oracles[name,str(dtype)]
                        x=q.cuda().to(dtype).flatten()[None].repeat(batch,1)
                        v=basis.cuda().to(dtype).reshape(1,-1,2).repeat(batch,1,1)
                        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
                        begin=time.perf_counter()
                        values,responses=oracle.simulate(x,v,shot_seeds)
                        torch.cuda.synchronize();elapsed=time.perf_counter()-begin
                        row=dict(parent=state['index'],source=state['source'],atoms=len(q),
                            radius_A=c['radius_A'],engine=name,dtype=str(dtype).split('.')[-1],
                            batch=batch,repeat=repeat,first_shot=start,features=values.shape[1],directions=responses.shape[-1],
                            seconds=elapsed,seconds_per_shot=elapsed/batch,
                            peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
                        timings.append(row)
                        print(dict(stage='timed',**row),flush=True)
                        write_json(out/'technical/timing-progress.json',timings)
                        if repeat==0:
                            published(c,f'queries/{name}-{row["dtype"]}-parent{state["index"]}-batch{batch}-shot{start}.pt',
                                dict(values=values.cpu(),responses=responses.cpu(),seeds=shot_seeds,timing=row))
    write_metric_rows(timings,out/'analyses/timing-v1',family=FAMILY,name='timings')
    write_json(out/'technical/timing-complete.json',dict(rows=len(timings),end_unix=time.time()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['validate','measure','all'])
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    c=read(args.config)
    payload,normalizer,original=inputs(c)
    bind(c,original)
    if args.stage in ['validate','all']:validate(c,payload,normalizer,original)
    if args.stage in ['measure','all']:measure(c,payload,normalizer,original)


if __name__=='__main__':main()
