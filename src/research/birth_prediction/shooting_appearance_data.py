"""Strict crystal-free appearance histories from the completed Al shooting audit."""
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.postprocessing.protocols import ancestry
from src.postprocessing.protocols import external_data
from src.research.liquid_predictability.descriptors import patch_descriptors
from .data import read, load
from .appearance import original_config
from .appearance_data import cache_config
from .features import load_bank


def earlier(c):
    a = read(resolve_path(c['appearance_config']))
    bound = read(resolve_path(a['output']) / 'technical/prepared.json')
    if bound['identity'] != c['appearance_identity'] or bound['config'] != a:
        raise ValueError('Earlier appearance cohort changed')
    return a


def prepare(c):
    audit_root = resolve_path(c['shooting_audit'])
    p = read(audit_root / 'technical/plan.json')
    done = read(audit_root / 'technical/complete.json')
    if (done['identity'] != c['shooting_identity'] or p['identity'] != c['shooting_identity'] or done['branches'] != len(p['sources'])
            or digest({k: v for k, v in p.items() if k != 'identity'}) != p['identity']):
        raise ValueError('Require the complete, unchanged Al shooting audit')
    candidates = []
    for item in p['sources']:
        folder = audit_root / 'technical/sources' / item['id']
        receipt = read(folder / 'complete.json')
        if receipt['identity'] != digest(dict(plan=p['identity'], source=item['id'])):
            raise ValueError('Shooting result population changed')
        for name in ('events.json', 'established-fates.json', 'transient-episodes.json'):
            if sha(folder / name) != receipt['files'][name]:
                raise ValueError(f'Changed candidate labels: {folder/name}')
        events = {e['event_id']: e for e in read(folder / 'events.json')}
        for e in read(folder / 'established-fates.json'):
            if e['kind'] == 'isolated_establishment':
                origin = events[e['event']]
                candidates.append(dict(source=item['id'], event=f'established:{e["event"]}',
                    stratum='established', first_frame=e['birth_frame'], first_node=origin['birth_node'],
                    centroid_A=origin['centroid_A'], followup_frame=e['confirmation_frame'], fate=e['fate_label']))
        for e in read(folder / 'transient-episodes.json'):
            if e['brief_failed_candidate']:
                candidates.append(dict(source=item['id'], event=e['episode'], stratum='failed_other',
                    first_frame=e['first_frame'], first_node=e['first_node'],
                    centroid_A=e['centroid_A'], followup_frame=e['dissolution_frame'], fate='dissolved'))
    counts = dict(Counter(e['stratum'] for e in candidates))
    if counts != c['expected_candidates']:
        raise ValueError(f'Shooting candidate population changed: {counts}')
    a = earlier(c)
    old = original_config(a, 'original')
    _, columns = load_bank(old, 'descriptors')
    sources = [s for s in p['sources'] if s['id'] in {e['source'] for e in candidates}]
    for s in sources:
        if s['cadence_ps'] != .3 or not np.allclose(np.diff(s['times_ps']), .3, rtol=0, atol=1e-8):
            raise ValueError('Shooting observation timeline is not the declared exact0.30ps grid')
    record = dict(config=c, audit=p, candidates=candidates, sources=sources,
        descriptor_columns=columns, earlier=a, implementation=sha(Path(__file__)))
    record['identity'] = digest(record)
    path = resolve_path(c['output']) / 'technical/data-plan.json'
    if path.exists() and read(path) != record:
        raise ValueError('Candidate preparation changed; choose a new output revision')
    write_json(path, record)
    return record


class Frames:
    def __init__(self, p, item):
        self.item = item
        self.folder = resolve_path(p['audit']['config']['output']) / 'technical/sources' / item['id']
        receipt = read(self.folder / 'complete.json')
        for name in ('ptm-complete.json', 'graph.npz'):
            if sha(self.folder / name) != receipt['files'][name]:
                raise ValueError(f'Changed source audit: {self.folder/name}')
        self.ptm = read(self.folder / 'ptm-complete.json')
        self.chunk = p['audit']['config']['chunk_frames']
        part = item['parts'][0]
        self.atom_ids = external_data.arrays(part['path'], part['manifest_sha256'])['atom_ids']
        self.labels = lru_cache(maxsize=4)(self._labels)
        self.frame = lru_cache(maxsize=48)(self._frame)

    def _labels(self, start):
        stop = min(start+self.chunk, self.item['frame_count'])
        name = f'ptm-{start:04d}-{stop:04d}.npz'
        if sha(self.folder / name) != self.ptm['files'][name]:
            raise ValueError(f'Changed PTM chunk: {self.folder/name}')
        with np.load(self.folder / name) as saved:
            return saved['labels'].copy()

    def _frame(self, frame):
        points, box = external_data.geometry(self.item, frame)
        start = frame//self.chunk*self.chunk
        labels = self.labels(start)[frame-start]
        solid = np.flatnonzero(np.isin(labels, (1, 2, 3)))
        tree = cKDTree(points[solid], boxsize=box) if len(solid) else None
        return points, box, labels, solid, tree

    def clear(self, frame, atoms, radius):
        points, _, _, _, tree = self.frame(int(frame))
        return np.ones(len(atoms), bool) if tree is None else tree.query(points[atoms], workers=1)[0] > radius

    def patch(self, frame, atoms):
        points, box, _, _, _ = self.frame(int(frame))
        _, neighbors = cKDTree(points, boxsize=box).query(points[atoms], k=80, workers=1)
        if not np.array_equal(neighbors[:, 0], atoms):
            raise ValueError('Focal atom mismatch/duplicate coordinates in shooting patch')
        delta = points[neighbors]-points[atoms, None]
        delta -= box*np.rint(delta/box)
        return delta.astype(np.float32)


def source(c, p, item, *, pool='shooting'):
    folder = resolve_path(c['cache']) / 'sources' / item['id']
    folder.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(data_plan=p['identity'], source=item['id']))
    if (folder / 'complete.json').exists():
        done = read(folder / 'complete.json')
        if done['identity'] != identity or any(sha(folder/n) != h for n, h in done['files'].items()):
            raise ValueError('Changed prepared shooting source')
        return done
    start = time.monotonic()
    access = Frames(p, item)
    with np.load(access.folder / 'graph.npz') as graph:
        starts, sizes = graph['start'].copy(), graph['size'].copy()
    rng = np.random.default_rng(np.random.SeedSequence([c['seed'], int(item['id'][5:13], 16)]))
    rows, sequences, coverage, used = [], [], [], set()
    offsets = np.asarray(c['offsets_ps'])
    steps = np.rint(offsets/item['cadence_ps']).astype(int)
    np.testing.assert_allclose(steps*item['cadence_ps'], offsets, rtol=0, atol=1e-8)
    radius = c['radius_A']
    for event in [e for e in p['candidates'] if e['source'] == item['id']]:
        first = event['first_frame']
        points, box, labels, _, _ = access.frame(first)
        local, _ = ancestry.components(points, box, labels, item['lineage'])
        members = np.flatnonzero(np.where(local > 0, local+starts[first], 0) == event['first_node'])
        if len(members) != sizes[event['first_node']]:
            raise ValueError('Candidate component reconstruction changed')
        delta = points-np.asarray(event['centroid_A']); delta -= box*np.rint(delta/box)
        centers = rng.permutation(np.flatnonzero(np.linalg.norm(delta, axis=1) <= radius))
        accepted, reasons = 0, Counter()
        for center in centers[:c['maximum_centers_scanned_per_event']]:
            if accepted == c['centers_per_event']:
                break
            begin = max(0, first-int(round(c['appearance_search_ps']/item['cadence_ps'])))
            appearance = next((f for f in range(begin, first+1)
                if not access.clear(f, np.array([center]), radius)[0]), None)
            if appearance is None or appearance == begin or appearance+steps[0] < 0:
                reasons['insufficient_crystal_free_history'] += 1; continue
            px, _, _, solid, tree = access.frame(appearance)
            near = solid[np.asarray(tree.query_ball_point(px[center], radius), int)]
            if not np.intersect1d(near, members).size:
                reasons['appearance_not_candidate_core'] += 1; continue
            frames = appearance+steps
            np.testing.assert_allclose(np.asarray(item['times_ps'])[frames]-item['times_ps'][appearance],
                                       offsets, rtol=0, atol=1e-8)
            end = int(frames[-1])
            if (int(center), end) in used:
                reasons['duplicate_history'] += 1; continue
            if not all(access.clear(f, np.array([center]), radius)[0] for f in range(int(frames[0]), appearance)):
                reasons['earlier_crystal_in_history'] += 1; continue
            follow = max(event['followup_frame'], end+int(np.ceil(c['control_followup_ps']/item['cadence_ps'])))
            if follow >= item['frame_count']:
                reasons['insufficient_control_followup'] += 1; continue
            draw = rng.choice(item['atom_count'], min(item['atom_count'], c['negative_candidates']), replace=False)
            controls = []
            for block in np.array_split(draw, max(1, len(draw)//128)):
                keep = np.ones(len(block), bool)
                for frame in range(int(frames[0]), follow+1):
                    active = np.flatnonzero(keep)
                    if not len(active): break
                    keep[active] &= access.clear(frame, block[active], radius)
                controls.extend(int(a) for a in block[keep] if (int(a), end) not in used and a != center)
                if len(controls) >= 4: break
            if len(controls) < 4:
                reasons['insufficient_liquid_controls'] += 1; continue
            pair = f'{item["id"]}:{event["event"]}:{int(access.atom_ids[center])}:{end}'
            for atom, label in [(int(center), 1), *[(a, 0) for a in controls[:4]]]:
                used.add((atom, end))
                sequences.append([(int(f), atom) for f in frames])
                rows.append(dict(id=f'{item["id"]}:{int(access.atom_ids[atom])}:{end}:{label}',
                    source=item['id'], event=event['event'], pair=pair, atom=int(access.atom_ids[atom]),
                    role=item['role'], label=label, stratum=event['stratum'], pool=pool,
                    fate=event['fate'] if label else 'liquid_control',
                    ancestor=item['ancestry_group'], parent=item['parent_sha256'], campaign=item['campaign'],
                    end_frame=end, appearance_frame=appearance, control_followup_ps=(follow-end)*item['cadence_ps']))
            accepted += 1
        coverage.append(dict(source=item['id'], event=event['event'], stratum=event['stratum'],
                             histories=accepted, exclusions=dict(reasons)))
        write_json(folder / 'progress.json', dict(stage='eligibility', candidates=len(coverage), rows=len(rows)))
    keys = sorted({key for seq in sequences for key in seq})
    lookup = {key: i for i, key in enumerate(keys)}
    positions = np.empty((len(keys), 80, 3), np.float32)
    by_frame = defaultdict(list)
    for frame, atom in keys: by_frame[frame].append(atom)
    for frame, atoms in by_frame.items():
        clouds = access.patch(frame, np.asarray(atoms))
        for atom, xyz in zip(atoms, clouds): positions[lookup[(frame, atom)]] = xyz
    values = np.empty((len(keys), len(p['descriptor_columns'])), np.float32)
    for i, xyz in enumerate(positions):
        descriptor, names = patch_descriptors(xyz)
        if names != p['descriptor_columns']: raise ValueError('Physical descriptor schema changed')
        values[i] = descriptor
        if i % 64 == 0: write_json(folder / 'progress.json', dict(stage='descriptors', completed=i, total=len(keys)))
    indices = np.asarray([[lookup[key] for key in seq] for seq in sequences], int).reshape(-1, len(offsets))
    np.save(folder / 'positions.npy', positions)
    np.save(folder / 'features.npy', values[indices])
    write_json(folder / 'rows.json', rows); write_json(folder / 'coverage.json', coverage)
    done = dict(identity=identity, rows=len(rows), eligible_events=sum(e['histories'] > 0 for e in coverage),
        seconds=time.monotonic()-start, files={n: sha(folder/n) for n in
            ('positions.npy', 'features.npy', 'rows.json', 'coverage.json')})
    write_json(folder / 'complete.json', done)
    access.frame.cache_clear(); access.labels.cache_clear(); external_data.arrays.cache_clear()
    return done


def seal(c, p):
    a = p['earlier']
    original_plan = read(resolve_path(a['output']) / 'technical/data-plan.json')['original_plan']
    ancestors = {str(s['id']): s['lineage'] for s in original_plan['sources']}
    fate_root = resolve_path(a['fate_output']) / 'technical'
    if read(fate_root / 'complete.json')['identity'] != a['fate_identity']:
        raise ValueError('Earlier fate audit changed')
    fates = {}
    for s in original_plan['sources']:
        directory = fate_root / 'sources' / str(s['id'])
        if sha(directory / 'established-fates.json') != read(directory / 'complete.json')['files']['established-fates.json']:
            raise ValueError('Earlier established fate labels changed')
        fates.update({(str(e['source']), str(e['event'])): e['fate_label'] for e in read(directory / 'established-fates.json')})
    records, features, coverage = [], [], []
    for i, b in enumerate((original_config(a, 'original'), cache_config(a, 'original'))):
        _, rows, _ = load(b); bank, names = load_bank(b, 'descriptors')
        if names != p['descriptor_columns']: raise ValueError('Earlier descriptor schema differs')
        features.append(bank[rows['indices'][:, [0, 2, 4, 6]]])
        for j in range(len(rows['id'])):
            records.append(dict(id='earlier:'+str(rows['id'][j]), source='earlier:'+str(rows['source'][j]),
                event=str(rows['event'][j]), pair='earlier:'+str(rows['pair'][j]), atom=int(rows['atom'][j]),
                role=str(rows['role'][j]), label=int(rows['label'][j]),
                stratum='established' if i == 0 else str(rows['stratum'][j]), pool='earlier',
                fate=('liquid_control' if not rows['label'][j] else 'dissolved' if i else
                      fates[(str(rows['source'][j]), str(rows['event'][j]))]),
                ancestor=ancestors[str(rows['source'][j])], parent='', campaign='fixed_al64',
                end_frame=int(rows['end_frame'][j])-1, appearance_frame=int(rows['appearance_frame'][j]),
                control_followup_ps=float((rows['confirmation_frame'][j]-(rows['end_frame'][j]-1))*.75)))
            if rows['appearance_frame'][j] - rows['end_frame'][j] != 1:
                raise ValueError('Earlier physical lead differs from declared appearance anchor')
    for item in p['sources']:
        folder = resolve_path(c['cache']) / 'sources' / item['id']
        done = read(folder / 'complete.json')
        if (done['identity'] != digest(dict(data_plan=p['identity'], source=item['id']))
                or any(sha(folder/n) != h for n, h in done['files'].items())):
            raise ValueError('Prepared shooting source identity changed')
        records.extend(read(folder / 'rows.json')); coverage.extend(read(folder / 'coverage.json'))
        features.append(np.load(folder / 'features.npy'))
    if len({r['id'] for r in records}) != len(records): raise ValueError('Duplicate sample identity')
    event_rows = Counter((r['source'], r['event']) for r in records)
    rows = {k: np.asarray([r[k] for r in records]) for k in records[0]}
    rows['weight'] = np.asarray([1/event_rows[(r['source'], r['event'])] for r in records])
    root = resolve_path(c['cache'])
    np.savez_compressed(root / 'rows.npz', **rows); np.save(root / 'features.npy', np.concatenate(features))
    manifest = dict(data_plan=p['identity'], rows=len(records), coverage=coverage,
        files={n: sha(root/n) for n in ('rows.npz', 'features.npy')}, offsets_ps=c['offsets_ps'])
    manifest['identity'] = digest(manifest)
    write_json(root / 'manifest.json', manifest)
    return rows, manifest


def dataset(c):
    root = resolve_path(c['cache']); manifest = read(root / 'manifest.json')
    if digest({k: v for k, v in manifest.items() if k != 'identity'}) != manifest['identity']:
        raise ValueError('Dataset manifest changed')
    for name, h in manifest['files'].items():
        if sha(root/name) != h: raise ValueError(f'Changed dataset: {root/name}')
    with np.load(root / 'rows.npz') as saved: rows = {k: saved[k] for k in saved.files}
    return rows, np.load(root / 'features.npy'), manifest
