"""Re-measure existing physical continuations, retaining their analytic tangents."""
import time
import numpy as np
import torch
from .data import root, cache, read, sha, write_json, atomic_save, digest, resolve_path, Source
from .features import SmoothMeasurements, structural


def collect(c, p):
    physical_plan = read(resolve_path(c['physical_run'])/'technical/plan.json')
    sources = {s['key']: s for s in physical_plan['sources']}
    measure = SmoothMeasurements().cuda().eval()
    epsilon = c['training']['response_epsilon']
    for index, job in enumerate(physical_plan['jobs']):
        destination = cache(c)/'physical'/job['key']
        if (destination/'complete.json').exists(): continue
        folder = resolve_path(c['physical_archive'])/job['key']
        receipt = read(folder/'complete.json')
        if sha(folder/'response.pt') != receipt['sha256']:
            raise ValueError('Changed analytic Ni physical bank: '+job['key'])
        saved = torch.load(folder/'response.pt', weights_only=False, map_location='cpu')
        item = sources[job['source_key']]; reader = Source(item['source'])
        history = np.stack([reader.patches(f, [saved['center']])[0][0]
                            for f in range(job['frame']-3, job['frame'])])
        history = torch.cat((torch.from_numpy(history), saved['state'][None])).float()
        pairs = torch.stack([saved['state']+sign*epsilon*saved['basis'][..., k]
                             for k in range(2) for sign in [-1, 1]]).float()
        half_pairs = torch.stack([saved['state']+sign*.5*epsilon*saved['basis'][..., k]
                                  for k in range(2) for sign in [-1, 1]]).float()
        future = saved['future'].float().flatten(0, 1)
        all_states = torch.cat((history, pairs, half_pairs, future))
        hard = structural(all_states[..., :3].numpy(), p['columns'][:153], c['length_factor'])
        with torch.no_grad(): smooth = measure(all_states.cuda()).cpu()
        values = torch.cat((torch.from_numpy(hard), smooth), -1)
        tangents = saved['tangent'].float().flatten(0, 1).cuda()
        response, initial = [], []
        for k in range(2):
            _, direction = torch.func.jvp(measure, (future.cuda(),), (tangents[..., k],))
            response.append(direction.cpu().reshape(2, 20, 91).mean(0))
            _, direction0 = torch.func.jvp(measure, (saved['state'][None].float().cuda(),),
                                          (saved['basis'][None, ..., k].float().cuda(),))
            initial.append(direction0[0].cpu())
        initial = torch.stack(initial, -1)
        # Student inputs retain finite, potentially nonsmooth BO/TDA changes.
        # Output responses are analytic on the91 smooth measurement channels.
        full = values[4:8].reshape(2, 2, 244)
        half = values[8:12].reshape(2, 2, 244)
        fd = (full[:, 1, 153:]-full[:, 0, 153:]).T/(2*epsilon)
        fd_half = (half[:, 1, 153:]-half[:, 0, 153:]).T/epsilon
        audit = dict(input_smooth_secant_relative_l2=((fd-initial).norm(dim=0)/initial.norm(dim=0).clamp_min(1e-12)).tolist(),
                     half_step_relative_l2=((fd_half-initial).norm(dim=0)/initial.norm(dim=0).clamp_min(1e-12)).tolist())
        record = dict(history=values[:4], pairs=full, half_pairs=half,
            target=values[12:].reshape(2, 20, 244), response=torch.stack(response, -1),
            initial_response=initial, role=item['role'], ancestor=saved['ancestor'], job=job['key'],
            center=saved['center'], horizons_fs=saved['horizons_fs'], audit=audit)
        if not all(torch.isfinite(v).all() for v in record.values() if isinstance(v, torch.Tensor)):
            raise FloatingPointError('Nonfinite measured physical path '+job['key'])
        atomic_save(destination/'data.pt', record)
        write_json(destination/'complete.json', dict(identity=digest([p['identity'], job]),
            sha256=sha(destination/'data.pt'), physical_sha256=receipt['sha256'], audit=audit))
        write_json(root(c)/'technical/physical-progress.json', dict(groups=index+1, total=300, updated=time.time()))
        if index % 10 == 0: print('measured physical', index+1, '/300', flush=True)
    write_json(root(c)/'technical/physical-complete.json', dict(state='complete', groups=300))


def load(c):
    p = read(resolve_path(c['physical_run'])/'technical/plan.json')
    records = []
    for job in p['jobs']:
        folder = cache(c)/'physical'/job['key']; receipt = read(folder/'complete.json')
        if sha(folder/'data.pt') != receipt['sha256']: raise ValueError('Changed physical measurements')
        records.append(torch.load(folder/'data.pt', weights_only=False, map_location='cpu'))
    return records
