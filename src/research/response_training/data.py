"""Fresh, branch-resumable response/value collection with durable publication."""
import shutil
import time

import numpy as np
import torch

from src.experiment_runner.execution import allocation_deadline
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.response_atlas.atomistic import (configuration, MACEPotential, PathFeatures,
    admissible, physical_screen, basis_for, oracle_for)
from .common import FAMILY, read, root, oracle_config, parents, sha, write_json, bind


def save_pt(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.building')
    torch.save(value, temporary)
    temporary.replace(path)


def prepare(c):
    record = bind(c)
    o = oracle_config(c)
    states = []
    for p in parents(c):
        atoms = configuration(o, p['index'])
        q = torch.tensor(atoms.positions, dtype=torch.float64)
        box = torch.tensor(atoms.cell.lengths(), dtype=torch.float64)
        admissible(q.flatten(), box, o['minimum_separation_A'])
        states.append(dict(**p, q=q, box=box))
    path = root(c)/'technical/parents.pt'
    if path.exists():
        saved = torch.load(path, weights_only=False)
        if saved['identity'] != record['identity'] or any(
            not torch.equal(a['q'], b['q']) for a,b in zip(states,saved['parents'],strict=True)):
            raise ValueError('Changed prepared query states')
    else:
        save_pt(path, dict(identity=record['identity'], parents=states))
    archive = resolve_path(c['simulation_archive'])
    archive.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, archive/'parents.pt')
    write_json(archive/'plan.json', dict(identity=record['identity'], config=c, parents=parents(c),
        parent_state_sha256=sha(path), ancestor='synthetic-FCC-256-prototype',
        potential_sha256=o['potential_sha256'], coordinate_precision='float64 numerical restart states; no trajectory export'))
    return states


def collect(c):
    identity = bind(c)['identity']
    o = oracle_config(c)
    states = prepare(c)
    archive, scratch = resolve_path(c['simulation_archive']), resolve_path(c['simulation_scratch'])
    atoms = configuration(o, 0)
    potential = MACEPotential(o, atoms, 'cuda')
    features = PathFeatures(atoms.cell.lengths(), o['horizons_steps'], o, 'cuda')
    deadline = allocation_deadline(reserve_seconds=240)
    costs = []
    for state in states:
        index, role = state['index'], state['role']
        dest = archive/f'parent-{index:03d}'
        work = scratch/f'parent-{index:03d}'
        dest.mkdir(parents=True, exist_ok=True); work.mkdir(parents=True, exist_ok=True)
        if (dest/'complete.json').exists():
            receipt = read(dest/'complete.json')
            if receipt['identity'] != identity or sha(dest/'query.pt') != receipt['sha256']:
                raise ValueError(f'Changed query: {dest}')
            costs.append(receipt['cost']); continue
        try:
            q = state['q'].flatten().cuda()
            basis = basis_for(q, c['directions'], o['seed']+100+index)
            oracle = oracle_for(o, potential, features, q, o['horizons_steps'])
            if (dest/'screen.json').exists():
                screen = read(dest/'screen.json')
                if screen['identity'] != identity: raise ValueError('Changed parent screen')
            else:
                start = time.monotonic(); values = physical_screen(potential, q)
                screen = dict(identity=identity, seconds=time.monotonic()-start, **values)
                write_json(dest/'screen.json', screen)
            streams = {}
            for kind in ('value', 'response'):
                count = c['value_shots'] if kind == 'value' else (0 if role == 'selection' else c['response_shots'])
                offset = 100 if role == 'test' and kind == 'response' else 0
                seeds = range(c['branch_seed_base']+1000*index+offset,
                              c['branch_seed_base']+1000*index+offset+count)
                records = []
                for seed in seeds:
                    if time.time() >= deadline:
                        raise TimeoutError('Allocation reserve reached; completed branches remain archived for continuation')
                    filename = f'{kind}-{seed}.pt'
                    path = dest/filename
                    if path.exists():
                        saved = torch.load(path, map_location='cpu', weights_only=False)
                        if saved['identity'] != identity or saved['seed'] != seed or saved['kind'] != kind:
                            raise ValueError(f'Changed cached branch: {path}')
                    else:
                        start = time.monotonic()
                        directions = basis if kind == 'response' else q.new_empty(len(q), 0)
                        bundle = oracle.query(q, directions, [seed])
                        torch.cuda.synchronize()
                        saved = dict(identity=identity, seed=seed, kind=kind, values=bundle.values.cpu(),
                            responses=bundle.responses.cpu(), seconds=time.monotonic()-start,
                            force_calls=bundle.force_calls, hvp_calls=bundle.hvp_calls,
                            device=torch.cuda.get_device_name())
                        save_pt(work/filename, saved)
                        shutil.copy2(work/filename, path)
                    records.append(saved)
                    write_json(root(c)/'technical/collection-progress.json', dict(state='running', parent=index,
                        role=role, kind=kind, completed_branches=len(records), total_branches=count,
                        completed_parents=len(costs), total_parents=len(states)))
                streams[kind] = records
            values = torch.cat([s['values'] for s in streams['value']])
            response = (torch.cat([s['responses'] for s in streams['response']])
                        if streams['response'] else torch.empty(0,256,2,dtype=torch.float64))
            agreement = None
            if role == 'train':
                ad_values = torch.cat([s['values'] for s in streams['response']])
                agreement = float((ad_values-values[:c['response_shots']]).abs().max())
                if agreement > 1e-10:
                    raise ValueError(f'Value-only and AD branches changed the common random trajectories: {agreement}')
            cost = dict(parent=index, role=role, value8_seconds=sum(s['seconds'] for s in streams['value'][:8]),
                value32_seconds=sum(s['seconds'] for s in streams['value']),
                response8_seconds=sum(s['seconds'] for s in streams['response']), screen_seconds=screen['seconds'],
                value_force_calls=sum(s['force_calls'] for s in streams['value']),
                response_force_calls=sum(s['force_calls'] for s in streams['response']),
                response_hvp_calls=sum(s['hvp_calls'] for s in streams['response']))
            payload = dict(identity=identity, **state, basis=basis.cpu().reshape(256,3,2), values=values,
                responses=response, value_seeds=[s['seed'] for s in streams['value']],
                response_seeds=[s['seed'] for s in streams['response']], cost=cost, same_path_error=agreement)
            save_pt(work/'query.pt', payload); shutil.copy2(work/'query.pt', dest/'query.pt')
            write_json(dest/'complete.json', dict(identity=identity, sha256=sha(dest/'query.pt'), cost=cost))
            costs.append(cost)
            print(f'Collected parent {index:02d} ({role}): value32, response{len(response)}', flush=True)
        except BaseException as error:
            write_json(dest/'failure.json', dict(identity=identity, error=repr(error)))
            raise
        finally:
            shutil.copytree(work, dest, dirs_exist_ok=True)
    write_metric_rows(costs, root(c)/'analyses/collection-v1', family=FAMILY, name='oracle-cost')
    seal(c)


def seal(c):
    identity = bind(c)['identity']
    archive = resolve_path(c['simulation_archive'])
    records, hashes = [], {}
    for p in parents(c):
        location = archive/f'parent-{p["index"]:03d}'
        receipt = read(location/'complete.json')
        if receipt['identity'] != identity or sha(location/'query.pt') != receipt['sha256']:
            raise ValueError(f'Incomplete or changed parent: {location}')
        records.append(torch.load(location/'query.pt', weights_only=False))
        hashes[str(p['index'])] = receipt['sha256']
    train = [r for r in records if r['role'] == 'train']
    shots = torch.stack([r['values'][:8] for r in train])
    center = shots.mean((0,1))
    scale = shots.flatten(0,1).std(0,unbiased=False).clamp_min(c['training']['value_scale_floor'])
    h = torch.stack([r['responses'].mean(0) for r in train])/scale[None,:,None]
    response_scale = h.square().mean().sqrt().clamp_min(c['training']['response_scale_floor'])
    path = resolve_path(c['cache'])/'training.pt'
    save_pt(path, dict(identity=identity, parents=records, center=center, scale=scale, response_scale=response_scale))
    write_json(root(c)/'technical/data-complete.json', dict(state='complete', identity=identity,
        cache=str(path), sha256=sha(path), archived_query_sha256=hashes,
        role_counts={role:sum(r['role']==role for r in records) for role in c['parent_roles']}))


def load(c):
    receipt = read(root(c)/'technical/data-complete.json')
    if receipt['identity'] != read(root(c)/'technical/binding.json')['identity'] or sha(receipt['cache']) != receipt['sha256']:
        raise ValueError('Sealed atomistic training data changed')
    return torch.load(receipt['cache'], map_location='cpu', weights_only=False)
