"""Exact-cadence, outcome-blind local observations from pinned Al sources."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import json
import numpy as np
import torch
from scipy.spatial import cKDTree
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path as _resolve_path
from src.postprocessing.io import arrays
from src.postprocessing.io import geometry
from src.research.response_geometry.features import observed

FAMILY = 'continuous_motion'


@lru_cache(maxsize=512)
def resolve_path(value):
    """Storage mappings are fixed for a frozen worker's lifetime."""
    return _resolve_path(value)


def read(path):
    return json.loads(Path(path).read_text())


def root(c):
    return resolve_path(c['output'])


def cache(c):
    return resolve_path(c['cache'])


def atomic_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.building')
    torch.save(payload, temporary)
    temporary.replace(path)


class Source:
    def __init__(self, source):
        self.source = source
        self.velocity = []
        self.ids = arrays(source['parts'][0]['path'], source['parts'][0]['manifest_sha256'])['atom_ids']
        for part in source['parts']:
            directory = resolve_path(part['path'])
            if sha(directory/'manifest.json') != part['manifest_sha256']:
                raise ValueError(f'Changed input manifest: {directory}')
            m = read(directory/'manifest.json')
            if m['velocity_units'] != 'angstrom_per_ps':
                raise ValueError(f'Unexpected velocity units: {directory}')
            a = arrays(part['path'], part['manifest_sha256'])
            if not np.array_equal(a['atom_ids'], self.ids):
                raise ValueError('Changed atom order across archive parts')
            spec = m['arrays']['velocities']
            v = np.load(directory/spec['file'], mmap_mode='r')
            if list(v.shape) != spec['shape'] or str(v.dtype) != spec['dtype']:
                raise ValueError('Velocity array schema differs from manifest')
            self.velocity.append(v)

    def frame(self, frame):
        q, box = geometry(self.source, int(frame))
        pi, raw = self.source['frames'][int(frame)]
        return q, np.asarray(self.velocity[pi][raw], dtype=np.float64), box

    def patches(self, frame, atom_ids):
        q, v, box = self.frame(frame)
        centers = np.searchsorted(self.ids, atom_ids)
        if not np.array_equal(self.ids[centers], atom_ids):
            raise ValueError('Fixed Al64 center absent from source')
        _, neighbors = cKDTree(q, boxsize=box).query(q[centers], k=80, workers=1)
        if not np.array_equal(neighbors[:, 0], centers):
            raise ValueError('Center is not its own nearest atom')
        local = q[neighbors]-q[centers, None]
        local -= box*np.rint(local/box)
        return np.concatenate((local, v[neighbors]), -1), neighbors


def prepare(c):
    """Choose sources/centers/times before reading any future structure labels."""
    upstream_path = resolve_path(c['source_plan'])
    upstream = read(upstream_path)
    if upstream['identity'] != c['source_identity']:
        raise ValueError('Pinned dense-source inventory changed')
    split = read(resolve_path(c['fixed_release'])/'splits.json')
    if split['identity'] != c['fixed_identity']:
        raise ValueError('Fixed Al64 source/sample identity changed')
    fixed = {s['lineage']: s for s in split['sources']}
    ancestors = sorted({x['source']['ancestry_group'] for x in upstream['sources']
                        if x['source']['role'] == 'train'}, key=lambda x: digest([c['seed'], x]))
    selection = set(ancestors[:max(1, round(len(ancestors)*c['inner_selection_fraction']))])
    sources, jobs, exclusions = [], [], list(upstream['excluded'])
    for item in upstream['sources']:
        s = item['source']
        contract = fixed[s['ancestry_group']]
        if contract['role'] != s['role']:
            raise ValueError('Ancestor role mismatch')
        if s['potential'] == []:
            exclusions.append(dict(source=s['id'], reason='unknown generating potential in historical archive; no MEAM teacher assigned'))
            continue
        if s['potential'] != ['al-lee2003-meam']:
            raise ValueError(f'Unexpected teacher potential: {s["dataset"]}')
        view = np.asarray(s['views']['010'], dtype=np.int64)
        times = np.asarray(s['times_ps'])[view]
        if not np.allclose(np.diff(times), .1, atol=1e-8, rtol=0):
            raise ValueError('Source does not have exact 0.1 ps observations')
        last = len(view)-1-int(round(max(c['horizons_ps'])/.1))
        if last < len(c['anchor_fractions']):
            exclusions.append(dict(source=s['id'], reason='insufficient complete future window'))
            continue
        role = ('selection' if s['ancestry_group'] in selection else 'train') if s['role']=='train' else 'test'
        record = dict(key=item['key'], source=s, role=role, original_role=s['role'],
                      centers=contract['center_atom_ids'], legacy_centers=contract['legacy_center_atom_ids'])
        sources.append(record)
        anchors = sorted(set(int(round(f*last)) for f in c['anchor_fractions']))
        for slot in anchors:
            frame = int(view[slot])
            frames = [int(view[slot+round(h/.1)]) for h in c['horizons_ps']]
            if not np.allclose(np.asarray(s['times_ps'])[frames]-s['times_ps'][frame], c['horizons_ps'], atol=1e-8, rtol=0):
                raise ValueError('Future observations have wrong physical offsets')
            key = digest([item['key'], frame])[:20]
            teacher_center = int(digest([c['seed'], key])[:8], 16)%64
            jobs.append(dict(key=key, source_key=item['key'], frame=frame, future_frames=frames,
                             teacher_center=teacher_center))
    plan = dict(config=c, source_plan_sha256=sha(upstream_path), sources=sources, jobs=jobs,
                exclusions=exclusions, inner_selection_ancestors=sorted(selection))
    plan['identity'] = digest(plan)
    destination = root(c)/'technical/plan.json'
    if destination.exists() and read(destination) != plan:
        raise ValueError('Changed continuous-motion plan; use a new revision')
    write_json(destination, plan)
    cache(c).mkdir(parents=True, exist_ok=True)
    write_json(root(c)/'technical/prediction-context.json', dict(
        encoder=dict(geometry='unrelaxed nearest80, smooth8A support, 5A edges, no halo',
                     species='one constant channel', motion='native same-atom V in A/ps; m=0.1ps*D E(Q)[V]',
                     history=1, conditions=[], exported_structural_dim=128, exported_motion_dim=128),
        predictor=dict(inputs='structural128 + motion128; autonomous drift/diffusion', conditions=[],
                       duration='solver interval only; never a network covariate',
                       targets='24 future-only radial/angular statistics, no crystal or reconstruction loss'),
        teacher=dict(backend='FastAl', full_periodic_cell=True, force_dtype='float32',
                     integration_state_tangent_noise_observable_dtype='float64',
                     protocol='fixed-box BAOAB, source thermostat T; independent of saved NPT future targets',
                     target_coordinates='prequantization local response observables in float64'),
        split='fixed Al64 ancestry; inner selection from original train; original selection/calibration/test pooled only for evaluation',
        observations='exact0.1ps; no interpolated training labels; native finer observations excluded from training',
        scope='verified-MEAM subset of pinned92 velocity views; unknown potentials explicitly excluded; no relaxed or velocity-free data',
        sampling='all64 fixed centers; outcome-blind times; no PTM/nucleus eligibility filter'))
    return plan


def prepare_job(c, plan, job):
    directory = cache(c)/'observations'/job['key']
    identity = digest([plan['identity'], job])
    if (directory/'complete.json').exists():
        done = read(directory/'complete.json')
        if done['identity'] != identity or sha(directory/'data.pt') != done['sha256']:
            raise ValueError('Changed prepared observations')
        return
    item = next(s for s in plan['sources'] if s['key']==job['source_key'])
    source = Source(item['source'])
    state, neighbors = source.patches(job['frame'], item['centers'])
    futures = []
    for frame in job['future_frames']:
        qv, _ = source.patches(frame, item['centers'])
        futures.append(observed(torch.from_numpy(qv[..., :3].copy())).double())
    target = torch.stack(futures, 1)
    # The same sampled physical observation can occur in a partial/full export.
    # Bind Q,V and future observations to deduplicate without relabeling ancestry.
    import hashlib
    fingerprints = [hashlib.sha256(state[i].tobytes()+target[i].numpy().tobytes()).hexdigest() for i in range(64)]
    dense = None
    if abs(item['source']['native_cadence_ps']-.01)<1e-8:
        times=np.asarray(item['source']['times_ps'])
        dense=[]
        for h in (.01,.02,.05,.09):
            requested=times[job['frame']]+h
            f=int(np.argmin(np.abs(times-requested)))
            if abs(times[f]-requested)>1e-8:
                raise ValueError('Missing actual native0.01ps observation')
            qv,_=source.patches(f,item['centers'])
            dense.append(observed(torch.from_numpy(qv[...,:3].copy())).double())
        dense=torch.stack(dense,1)
    atomic_save(directory/'data.pt', dict(state=torch.from_numpy(state).float(), target=target,
                dense_target=dense,neighbors=torch.from_numpy(neighbors), fingerprints=fingerprints))
    write_json(directory/'complete.json', dict(identity=identity, sha256=sha(directory/'data.pt')))


def seal(c, plan):
    states, targets, rows, seen = [], [], [], {}
    for job in plan['jobs']:
        folder = cache(c)/'observations'/job['key']
        done = read(folder/'complete.json')
        if done['identity'] != digest([plan['identity'],job]) or sha(folder/'data.pt') != done['sha256']:
            raise ValueError('Unsealed observation job')
        data = torch.load(folder/'data.pt', weights_only=False)
        item = next(s for s in plan['sources'] if s['key']==job['source_key'])
        for i, fingerprint in enumerate(data['fingerprints']):
            if fingerprint in seen:
                if seen[fingerprint] != (item['source']['ancestry_group'],item['role']):
                    raise ValueError('Identical observation crosses an ancestry or fitting role')
                continue
            seen[fingerprint]=(item['source']['ancestry_group'],item['role'])
            states.append(data['state'][i]); targets.append(data['target'][i])
            rows.append(dict(job=job['key'], index=i, source=item['source']['id'],
                             ancestor=item['source']['ancestry_group'], atom=item['centers'][i],
                             role=item['role'], original_role=item['original_role'],
                             legacy=item['centers'][i] in item['legacy_centers'], fingerprint=fingerprint))
    states, targets = torch.stack(states), torch.stack(targets)
    fitting = torch.tensor([r['role']=='train' for r in rows])
    mean = targets[fitting].flatten(0, 1).mean(0)
    scale = targets[fitting].flatten(0, 1).std(0, unbiased=False).clamp_min(1e-4)
    if not torch.isfinite(states).all() or not torch.isfinite(targets).all():
        raise FloatingPointError('Nonfinite observations')
    atomic_save(cache(c)/'dataset.pt', dict(state=states, target=targets, mean=mean, scale=scale, rows=rows))
    counts = Counter(r['role'] for r in rows)
    write_json(root(c)/'technical/data-complete.json', dict(identity=plan['identity'], rows=dict(counts),
                sources=len(plan['sources']), ancestors=len({r['ancestor'] for r in rows}),
                duplicates_removed=64*len(plan['jobs'])-len(rows), sha256=sha(cache(c)/'dataset.pt')))


def load(c):
    done = read(root(c)/'technical/data-complete.json')
    if sha(cache(c)/'dataset.pt') != done['sha256']:
        raise ValueError('Dataset cache changed')
    return torch.load(cache(c)/'dataset.pt', weights_only=False)
