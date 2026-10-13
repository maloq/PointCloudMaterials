"""Exact native 0.1-ps histories attached to unchanged dense response teachers."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from src.experiment_runner.metric_docs import check_metric_docs
from src.research.continuous_motion.data import Source
from src.postprocessing.io import arrays
from .data import read, root, cache, sha, digest, write_json, atomic_save, load as load_phase

FAMILY = 'ni_phase_history'


def prepare(c):
    phase_path = Path(c['phase_run'])/'technical/plan.json'
    birth_path = Path(c['birth_run'])/'technical/plan.json'
    for path, expected in [(phase_path, c['phase_plan_sha256']), (birth_path, c['birth_plan_sha256'])]:
        if sha(path) != expected:
            raise ValueError('Changed history parent plan: '+str(path))
    parent = read(phase_path); birth = read(birth_path)
    phase_jobs = parent['jobs']
    if any(j['frame'] < 3 for j in phase_jobs):
        raise ValueError('Incomplete history in the fixed physical cohort')
    data_path = cache(birth['config'])/'cohort.pt'
    if sha(data_path) != read(cache(birth['config'])/'cohort.json')['sha256']:
        raise ValueError('Changed fixed nucleus input cohort')
    data = torch.load(data_path, map_location='cpu', weights_only=False)
    if any(r['frame'] < 3 for r in data['rows']):
        raise ValueError('Missing native history in fixed birth rows; do not drop observations')
    p = dict(config=c, parent_config=parent['config'], jobs=phase_jobs, sources=parent['sources'],
        partition=birth['partition'], columns=parent['columns'], birth_config=birth['config'],
        birth_cohort_sha256=sha(data_path), history_offsets_ps=[-.3, -.2, -.1, 0.],
        implementations=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    p['data_identity'] = digest([c['phase_plan_sha256'], p['birth_cohort_sha256'],
                                p['history_offsets_ps'], sha(Path(__file__))])
    p['identity'] = digest(p)
    destination = root(c)/'technical/plan.json'
    if destination.exists() and read(destination) != p:
        raise ValueError('Changed history protocol; use a new revision')
    write_json(destination, p)
    return p


def fixed_history(reader, frame, atom_ids):
    """Batched [centers,80] identities; one full-cell read per historical frame."""
    times = np.asarray(reader.source['times_ps'])
    frames = np.arange(frame-3, frame+1)
    if frame < 3 or not np.allclose(times[frames]-times[frame], [-.3, -.2, -.1, 0.], rtol=0, atol=1e-8):
        raise ValueError('History does not have exact native observation offsets')
    index = np.searchsorted(reader.ids, atom_ids)
    if not np.array_equal(reader.ids[index], atom_ids):
        raise ValueError('Fixed atom identity missing from history')
    values = []
    for f in frames:
        q, v, box = reader.frame(int(f))
        q = q[index]-q[index[:, :1]]; q -= box*np.rint(q/box)
        v = v[index]-v[index[:, :1]]
        values.append(np.concatenate((q, v), -1))
    return torch.from_numpy(np.stack(values, axis=1)).float()


def collect(c, p):
    """No MD or tangent recomputation; only recover preceding saved frames."""
    marker = cache(c)/'complete.json'
    if marker.exists():
        done = read(marker)
        if done['identity'] != p['data_identity'] or any(sha(cache(c)/f) != h for f, h in done['files'].items()):
            raise ValueError('Changed history cache')
        return
    original = torch.load(cache(p['birth_config'])/'cohort.pt', map_location='cpu', weights_only=False)
    birth_records = {}
    for i, r in enumerate(original['rows']):
        birth_records.setdefault(int(original['indices'][i]), r)
    physical = {}; birth_history = torch.empty(len(original['state']), 4, 80, 6)
    for item in p['sources']:
        reader = Source(item['source'])
        for part in reader.source['parts']:
            raw = arrays(part['path'], part['manifest_sha256'])
            if not np.allclose(np.diff(raw['timesteps'])*part['timestep_fs']/1000, .1, atol=1e-8, rtol=0):
                raise ValueError('Actual source timestep cadence mismatch')
        for job in p['jobs']:
            if job['source_key'] != item['key']:
                continue
            packet = torch.load(Path(c['phase_archive'])/job['key']/'path.pt', map_location='cpu', weights_only=False)
            history = fixed_history(reader, job['frame'], packet['atom_ids'][None])[0]
            if not torch.allclose(history[-1], packet['state'][0].float(), atol=1e-5, rtol=0):
                raise ValueError('History endpoint differs from reused physical initial state')
            physical[job['key']] = history
        groups = defaultdict(list)
        for index, r in birth_records.items():
            if r['source'] == item['key']:
                groups[r['frame']].append((index, r['center']))
        for frame, pairs in groups.items():
            # Build the nearest-80 selection once per saved observation frame.
            _, selected = reader.patches(frame, [center for _, center in pairs])
            histories = fixed_history(reader, frame, reader.ids[selected])
            for (index, _), h in zip(pairs, histories):
                if not torch.allclose(h[-1], original['state'][index], atol=1e-5, rtol=0):
                    raise ValueError('Fixed birth input changed during history recovery')
                # Preserve the cached present input bit-for-bit for probe comparisons.
                h[-1] = original['state'][index]
                birth_history[index] = h
        write_json(root(c)/'technical/history-progress.json', dict(source=item['key'], physical=len(physical)))
    if not torch.isfinite(birth_history).all():
        raise FloatingPointError('Nonfinite historical observations')
    atomic_save(cache(c)/'physical.pt', physical)
    atomic_save(cache(c)/'births.pt', dict(history=birth_history, **original))
    write_json(marker, dict(identity=p['data_identity'], physical=len(physical), birth_states=len(birth_history),
        files={f: sha(cache(c)/f) for f in ['physical.pt', 'births.pt']}))


def load(c, role):
    p = read(root(c)/'technical/plan.json')
    _, records, state, tangent = load_phase(p['parent_config'], role)
    history = torch.load(cache(c)/'physical.pt', map_location='cpu', weights_only=False)
    scale = state.new_tensor([5., 5., 5., 10., 10., 10.])
    h = torch.stack([history[r['job']['key']] for r in records]).cuda()/scale
    return p, records, state, tangent, h
