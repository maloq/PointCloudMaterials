"""Measured values and exact smooth-observable tangents from sealed Ni teachers."""
from pathlib import Path
import numpy as np
import torch
from src.experiment_runner.metric_docs import check_metric_docs
from src.research.measurement_sim.features import SmoothMeasurements, structural
from .data import read, root, cache, sha, digest, write_json, atomic_save

FAMILY = 'ni_phase_measurements'
GROUPS = [('bond_order', 0, 40), ('tda', 40, 153), ('density', 153, 227), ('velocity', 227, 244)]


def parent(c):
    path = Path(c['history_run'])/'technical/plan.json'
    if sha(path) != c['history_plan_sha256']:
        raise ValueError('Changed parent history plan')
    return read(path)


def collect(c):
    p = parent(c); out = cache(c); out.mkdir(parents=True, exist_ok=True)
    identity = digest([c['history_plan_sha256'], sha(Path(__file__))])
    marker = out/'measurements-complete.json'
    if marker.exists():
        m = read(marker)
        if m['identity'] != identity or sha(out/'teachers.pt') != m['sha256']:
            raise ValueError('Changed observable teacher cache')
        return
    smooth = SmoothMeasurements().cuda().double(); packets = []
    for index, job in enumerate(p['jobs']):
        dest = out/'targets'/job['key']; dest.mkdir(parents=True, exist_ok=True)
        source = Path(c['phase_archive'])/job['key']
        if sha(source/'path.pt') != read(source/'complete.json')['sha256']:
            raise ValueError('Changed physical source: '+job['key'])
        if not (dest/'complete.json').exists():
            physical = torch.load(source/'path.pt', map_location='cuda', weights_only=False)
            h = physical['state']; u = physical['tangent']
            # Historical potential arithmetic is unchanged; these observables and JVPs are float64.
            with torch.no_grad():
                values = smooth(h); responses = []
                for k in range(2):
                    _, response = torch.func.jvp(smooth, (h,), (u[..., k],))
                    responses.append(response)
            geometry = structural(h[..., :3].cpu().numpy(), p['columns'][:153], c['length_factor'])
            value = torch.cat((torch.from_numpy(geometry), values.cpu()), -1)
            response = torch.stack(responses, -1).cpu()
            if not torch.isfinite(value).all() or not torch.isfinite(response).all():
                raise FloatingPointError('Nonfinite measurement teacher: '+job['key'])
            atomic_save(dest/'target.pt', dict(value=value, response=response))
            write_json(dest/'complete.json', dict(identity=identity, sha256=sha(dest/'target.pt')))
        m = read(dest/'complete.json')
        if m['identity'] != identity or m['sha256'] != sha(dest/'target.pt'):
            raise ValueError('Changed measurement target: '+job['key'])
        packets.append(torch.load(dest/'target.pt', map_location='cpu', weights_only=False))
        write_json(root(c)/'technical/measurement-progress.json', dict(completed=index+1, total=len(p['jobs'])))
    values = torch.stack([x['value'] for x in packets]); responses = torch.stack([x['response'] for x in packets])
    train = torch.tensor([j['role'] == 'train' for j in p['jobs']])
    mean = values[train].flatten(0, 1).mean(0); std = values[train].flatten(0, 1).std(0)
    scale = std.clamp_min(1e-5); active = std > 1e-8
    power = (responses[train]/scale[None, None, 153:, None]).square().mean(0).clamp_min(1e-6)
    atomic_save(out/'teachers.pt', dict(keys=[j['key'] for j in p['jobs']], roles=[j['role'] for j in p['jobs']],
        value=values, response=responses, mean=mean, scale=scale, active=active, response_power=power))
    write_json(marker, dict(identity=identity, sha256=sha(out/'teachers.pt'), states=len(packets),
        precision='Float64 saved states, tangents, measurements and observable JVPs; FP32 neural training'))


def targets(c, role, *, device='cuda'):
    data = torch.load(cache(c)/'teachers.pt', map_location='cpu', weights_only=False)
    take = torch.tensor([r == role for r in data['roles']])
    result = {k: data[k].to(device=device, dtype=torch.float32) for k in ['mean', 'scale', 'response_power']}
    result['active'] = data['active'].to(device)
    result['value'] = (data['value'][take].to(device=device, dtype=torch.float32)-result['mean'])/result['scale']
    result['response'] = data['response'][take].to(device=device, dtype=torch.float32)/result['scale'][None, None, 153:, None]
    result['keys'] = [k for k, use in zip(data['keys'], take) if use]
    return result


def prepare(c):
    p = parent(c); m = read(cache(c)/'measurements-complete.json')
    if sha(cache(c)/'teachers.pt') != m['sha256']:
        raise ValueError('Changed sealed measurement data')
    sources = cache(p['config'])
    original = read(sources/'complete.json')
    for filename, expected in original['files'].items():
        if sha(sources/filename) != expected:
            raise ValueError('Changed native history inputs')
        link = cache(c)/filename
        if not link.exists():
            link.symlink_to(sources/filename)
    link = cache(c)/'birth-measurements'
    if not link.exists():
        link.symlink_to(sources/'birth-measurements', target_is_directory=True)
    controls = root(c)/'births/analyses/controls'; controls.parent.mkdir(parents=True, exist_ok=True)
    if not controls.exists():
        controls.symlink_to(Path(c['history_run'])/'births/analyses/controls', target_is_directory=True)
    frozen = {}
    for seed in c['model_seeds']:
        name = f'attn4-full-s{seed}'; path = Path(c['history_run'])/'analyses'/name/'technical/final.pt'
        frozen[name] = dict(path=str(path), sha256=sha(path))
    result = dict(p, config=c, history_parent_identity=p['identity'], measurement_cache=m,
        frozen_diagnostic_models=frozen, implementations=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    result.pop('identity'); result['identity'] = digest(result)
    path = root(c)/'technical/plan.json'
    if path.exists() and read(path) != result:
        raise ValueError('Changed observable protocol; use a new run revision')
    write_json(path, result)
    return result
