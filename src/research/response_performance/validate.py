"""Repeat isolated timings and exercise resumable value reuse on archived streams."""
import argparse
import json
from pathlib import Path
import time

import torch

from src.experiment_runner.artifacts import write_json, file_hash, json_digest
from src.project_runtime.paths import resolve_path
from .benchmark import read, make, errors
from .oracle import force_and_tangent
from .collection import acquire_parent


def run(config):
    c = read(config)
    root = resolve_path(c['output'])
    out = root/'analyses/benchmark-v1'
    destination = out/'technical/validation'
    destination.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.90)
    records = [read(out/'technical/variants'/f'{v["name"]}.json') for v in c['variants']]
    accepted = []
    for record in records:
        if record['state'] != 'complete':
            continue
        tol = c['tolerance'][record['variant']['dtype']]['static_relative']
        # The initial sweep saved energies but gated force/HVPs. Explicitly
        # enforce saved absolute-energy and periodic-image equivalence here.
        gate = record['gates'][0]['static']
        if any(gate[key]['relative'] > tol for key in ('energy', 'periodic_energy')):
            raise ValueError(f'Energy gate failed: {record["variant"]}')
        accepted.append(record)
    candidates = [r for r in accepted if r['variant']['dtype']=='float64'
                  and r['variant'].get('method','ad')=='ad' and r['variant']['graph']=='gpu']
    fastest = {kind:min(candidates, key=lambda r:next(x['seconds_per_trajectory']
        for x in r['measurements'] if x['kind']==kind))['variant'] for kind in ('value','response')}
    variants = [next(v for v in c['variants'] if v['name']==name)
                for name in dict.fromkeys(['reference64','conditional64',*[v['name'] for v in fastest.values()]])]
    measurements=[]
    for variant in variants:
        path=destination/f'repeat-{variant["name"]}.json'
        if path.exists():
            measurements.extend(read(path)['measurements'])
            continue
        oracle,potential,q,basis,saved=make(c,variant)
        rows=[]
        for kind in ('value','response'):
            directions=basis if kind=='response' else q.new_empty(len(q),0)
            seeds=saved['value_seeds'][:variant['batch']]
            force_and_tangent(potential,q,directions)
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.monotonic()
            bundle=oracle.query(q,directions,seeds)
            torch.cuda.synchronize();seconds=time.monotonic()-begin
            value=errors(bundle.values,saved['values'][:len(seeds)])
            response=errors(bundle.responses[:8],saved['responses'][:len(seeds)]) if kind=='response' else None
            tol=c['tolerance']['float64']
            for error in (value,response):
                if error is not None and (error['relative']>tol['path_relative'] or error['maximum_absolute']>tol['path_absolute']):
                    raise ValueError(f'Repeated trajectory agreement failed: {variant}, {kind}, {error}')
            row=dict(variant=variant['name'],kind=kind,seconds=seconds,
                seconds_per_trajectory=seconds/len(seeds),replicas=len(seeds),
                peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
                value_error=value,response_error=response)
            rows.append(row);print(json.dumps(row),flush=True)
        write_json(path,dict(variant=variant,measurements=rows))
        measurements.extend(rows)
        del oracle,potential,q,basis,saved,bundle
        import gc
        gc.collect();torch.cuda.empty_cache()
    selected=fastest['response']
    oracle,potential,q,basis,saved=make(c,selected)
    spec=dict(oracle_recipe=read(resolve_path(c['oracle_recipe'])),variant=selected,
        numerical_sources={str(p.relative_to(Path(__file__).parent)):file_hash(p)
            for p in Path(__file__).parent.glob('*.py')},reference_identity=c['reference_identity'],
        purpose='Replay archived training streams to validate acquisition reuse and restart; no new scientific label bank')
    identity=json_digest(spec)
    write_json(destination/'reuse-binding.json',dict(identity=identity,binding=spec))
    args=dict(role='train',value_seeds=saved['value_seeds'][:4],
        response_seeds=saved['response_seeds'][:2],audit_seeds=saved['response_seeds'][:1],
        destination=destination/'reuse-demo',identity=identity,value_batch=2,response_batch=2)
    result=acquire_parent(oracle,q,basis,**args)
    value=errors(result['values'],saved['values'][:4]);response=errors(result['responses'],saved['responses'][:2])
    for error in (value,response):
        if error['relative']>1e-5 or error['maximum_absolute']>1e-7:
            raise ValueError(f'Reuse collection differs from archive: {error}')
    calls=potential.calls
    acquire_parent(oracle,q,basis,**args)
    if potential.calls != calls:
        raise ValueError('Resuming completed collection unexpectedly recomputed trajectories')
    write_json(destination/'complete.json',dict(state='complete',measurements=measurements,
        fastest_float64_ad={k:v['name'] for k,v in fastest.items()},
        energy_gated_variants=[r['variant']['name'] for r in accepted],
        reuse=dict(value_error=value,response_error=response,resume_new_model_calls=potential.calls-calls,
                   executed_trajectories=5,reference_without_reuse=6),
        timing_note='Initial reference overlapped PaCMAP inference. Use these isolated repeats for reference and conditional comparisons.',
        device=torch.cuda.get_device_name(),finished_at=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
