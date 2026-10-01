"""Matched, source-held-out birth/liquid histories from the frozen full-cell audit."""
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin import ancestry, extract

ROLES = ('train', 'selection', 'calibration', 'test')


def read(path):
    return json.loads(Path(path).read_text())


def plan(c):
    audit = read(resolve_path(c['audit_config']))
    _, release = read_release(audit['release'])
    if release['identity'] != c['fixed_dataset']['identity']:
        raise ValueError('Birth study changed the fixed source release')
    old = resolve_path(audit['output']) / 'technical'
    state, contract = read(old / 'audit-state.json'), read(old / 'audit-contract.json')
    if (state['state'] != 'complete' or contract['ancestry_sha256'] != sha(Path(ancestry.__file__))
            or contract['config'] != audit):
        raise ValueError('Require the complete, unchanged full-cell ancestry audit')
    if release['config']['cadence_ps'] != c['cadence_ps'] or c['cadence_ps'] != .75:
        raise ValueError('This protocol requires exact 0.75 ps original trajectories')
    binding = dict(config=c, audit_identity=state['identity'], release_identity=release['identity'],
                   producer_sha256=sha(Path(__file__)))
    return dict(identity=digest(binding), binding=binding, audit=audit,
                sources=[{k: v for k, v in s.items() if k != 'cells'} for s in release['sources']])


class SourceFrames:
    """Original periodic geometry and verified full-cell PTM, with bounded frame caching."""
    def __init__(self, p, item):
        self.item = item
        self.raw = extract.raw_source(item)  # verifies the actual source timeline
        self.root = resolve_path(p['audit']['output']) / 'technical/sources' / str(item['id'])
        self.receipt = read(self.root / 'ptm-complete.json')
        self.chunk = p['audit']['ptm']['chunk_frames']
        self.verified = set()

    @lru_cache(maxsize=2)
    def labels(self, begin):
        end = min(begin + self.chunk, self.item['frame_count'])
        name = f'ptm-{begin:04d}-{end:04d}.npz'
        if name not in self.verified:
            if sha(self.root / name) != self.receipt['files'][name]:
                raise ValueError(f'Changed full-cell PTM: {self.item["id"]}/{name}')
            self.verified.add(name)
        with np.load(self.root / name) as arrays:
            return arrays['labels'].copy()

    @lru_cache(maxsize=64)
    def frame(self, frame):
        points, box = extract.frame_geometry(self.raw, frame)
        begin = frame // self.chunk * self.chunk
        labels = self.labels(begin)[frame - begin]
        solid = np.flatnonzero(np.isin(labels, [1, 2, 3]))
        tree = cKDTree(points[solid], boxsize=box) if len(solid) else None
        return points, box, labels, solid, tree

    def clear(self, frame, atoms, radius):
        points, _, _, _, tree = self.frame(int(frame))
        if tree is None:
            return np.ones(len(atoms), bool)
        return tree.query(points[atoms], workers=1)[0] > radius

    def patch(self, frame, atoms):
        points, box, _, _, _ = self.frame(int(frame))
        _, ids = cKDTree(points, boxsize=box).query(points[atoms], k=80, workers=1)
        if not np.array_equal(ids[:, 0], atoms):
            raise ValueError(f'Duplicate coordinates or focal-neighbor mismatch: {self.item["id"]}/{frame}')
        delta = points[ids] - points[atoms, None]
        delta -= box * np.rint(delta / box)
        return delta.astype(np.float32), ids


def prepare_source(c, p, item):
    root = resolve_path(c['cache']) / 'sources' / str(item['id'])
    root.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(plan=p['identity'], source=item))
    if (root / 'complete.json').exists():
        done = read(root / 'complete.json')
        if done['identity'] != identity or any(sha(root / f) != h for f, h in done['files'].items()):
            raise ValueError(f'Changed prepared birth histories: {item["id"]}')
        return done
    started = time.time()
    access = SourceFrames(p, item)
    old = read(access.root / 'audit.json')
    if sha(access.root / 'events.json') != old['files']['events.json']:
        raise ValueError(f'Changed birth catalogue: {item["id"]}')
    events = [e for e in read(access.root / 'events.json') if e['threshold'] == 'primary']
    graph_receipt = read(access.root / 'graph-complete.json')
    if sha(access.root / 'graph.npz') != graph_receipt['sha256']:
        raise ValueError('Changed full-cell lineage graph')
    with np.load(access.root / 'graph.npz') as graph:
        frame_starts = graph['start'].copy()
    history = c['history_frames']
    radius = c['radius_A']
    rng = np.random.default_rng(np.random.SeedSequence([c['seed'], item['id']]))
    sequences, metadata, counts, outcomes = [], [], Counter(), []
    used = set()

    def add(atom, end, event, label, appearance, confirm, pair):
        frames = np.arange(end - history + 1, end + 1, dtype=np.int32)
        key = (int(atom), int(end))
        if key in used:
            return False
        used.add(key)
        if not all(access.clear(f, np.array([atom]), radius)[0] for f in frames):
            raise ValueError('Crystal entered a supposedly clear input history')
        sequences.append([(int(f), int(atom)) for f in frames])
        metadata.append(dict(source=item['id'], role=item['role'], event=event['event_id'],
            pair=pair, atom=int(access.raw.atom_ids[atom]), label=label, end_frame=end,
            appearance_frame=appearance, birth_frame=event['birth_frame'],
            confirmation_frame=confirm, start_frame=int(frames[0])))
        return True

    for event in events:
        if event['kind'] != 'isolated_establishment':
            continue
        counts['catalogue_isolated_events'] += 1
        birth = event['birth_frame']
        points, box, labels, _, _ = access.frame(birth)
        dense, _ = ancestry.components(points, box, labels, p['audit']['lineage'])
        # Component IDs become graph node IDs through the recorded frame offset.
        members = np.flatnonzero(np.where(dense > 0, dense + frame_starts[birth], 0) == event['birth_node'])
        if len(members) != event['birth_size']:
            raise ValueError('Birth-member reconstruction changed')
        center = np.array(event['centroid_A'])
        delta = points - center
        delta -= box * np.rint(delta / box)
        candidates = np.flatnonzero(np.linalg.norm(delta, axis=1) <= radius)
        candidates = rng.permutation(candidates)
        accepted = 0
        reasons = Counter()
        for atom in candidates[:c['maximum_centers_scanned_per_event']]:
            if accepted == c['centers_per_event']:
                break
            # First observed crystal in a bounded pre-establishment search. Require
            # an atom from this birth's core at that appearance, not an unrelated front.
            begin = max(0, birth - c['appearance_search_frames'])
            appearance = None
            for f in range(begin, birth + 1):
                if not access.clear(f, np.array([atom]), radius)[0]:
                    appearance = f
                    break
            if appearance is None:
                reasons['no_observed_appearance'] += 1
                continue
            if appearance == begin or appearance < history:
                reasons['left_censored_appearance_or_history'] += 1
                continue
            px, bx, _, solid, tree = access.frame(appearance)
            local_solid = solid[np.asarray(tree.query_ball_point(px[atom], radius), dtype=int)]
            if not np.intersect1d(local_solid, members).size:
                reasons['appearance_not_associated_with_birth_core'] += 1
                continue
            end = appearance - 1
            if not all(access.clear(f, np.array([atom]), radius)[0]
                       for f in range(end - history + 1, end + 1)):
                reasons['crystal_in_input_history'] += 1
                continue
            # Negatives remain liquid through the case's confirmation, with at
            # least six ps follow-up. Time and temperature never become features.
            follow = max(event['confirmation_frame'], end + c['minimum_followup_frames'])
            if follow >= item['frame_count']:
                reasons['incomplete_followup'] += 1
                continue
            eligible = []
            draw = rng.choice(item['atom_count'], min(item['atom_count'], c['negative_candidates']), replace=False)
            for block in np.array_split(draw, max(1, len(draw) // 128)):
                keep = np.ones(len(block), bool)
                for f in range(end - history + 1, follow + 1):
                    active = np.flatnonzero(keep)
                    if not len(active):
                        break
                    keep[active] &= access.clear(f, block[active], radius)
                eligible.extend(int(a) for a in block[keep] if (int(a), end) not in used and a != atom)
                if len(eligible) >= c['negatives_per_positive']:
                    break
            if len(eligible) < c['negatives_per_positive']:
                reasons['insufficient_matched_liquid_controls'] += 1
                continue
            pair = f'{item["id"]}:{event["event_id"]}:{int(access.raw.atom_ids[atom])}:{end}'
            if not add(atom, end, event, 1, appearance, follow, pair):
                reasons['duplicate_case'] += 1
                continue
            for control in eligible[:c['negatives_per_positive']]:
                if not add(control, end, event, 0, appearance, follow, pair):
                    raise ValueError('Unexpected duplicate matched control')
            accepted += 1
        counts['positive_histories'] += accepted
        counts['events_with_histories'] += int(accepted > 0)
        counts.update(reasons)
        outcomes.append(dict(event=event['event_id'], histories=accepted, exclusions=dict(reasons)))
        write_json(root / 'progress.json', dict(identity=identity, events_done=len(outcomes),
                   positive_histories=counts['positive_histories'], seconds=time.time() - started))

    keys = sorted(set(k for s in sequences for k in s))
    index = {k: i for i, k in enumerate(keys)}
    bank = np.empty((len(keys), 80, 3), np.float32)
    by_frame = defaultdict(list)
    for f, atom in keys:
        by_frame[f].append(atom)
    for f, atoms in by_frame.items():
        patches, _ = access.patch(f, np.array(atoms))
        for atom, xyz in zip(atoms, patches):
            bank[index[(f, atom)]] = xyz
    np.save(root / 'positions.npy', bank)
    indices = np.array([[index[k] for k in s] for s in sequences], np.int64).reshape(-1, history)
    fields = ('source', 'role', 'event', 'pair', 'atom', 'label', 'end_frame', 'appearance_frame',
              'birth_frame', 'confirmation_frame', 'start_frame')
    arrays = {k: np.array([m[k] for m in metadata], dtype='U100' if k in ('role', 'pair') else np.int64) for k in fields}
    np.savez_compressed(root / 'rows.npz', indices=indices, **arrays)
    write_json(root / 'event-coverage.json', dict(source=item['id'], role=item['role'], events=outcomes))
    done = dict(identity=identity, source=item['id'], role=item['role'], rows=len(metadata),
                patches=len(keys), counts=dict(counts), seconds=time.time() - started,
                files={n: sha(root / n) for n in ('positions.npy', 'rows.npz', 'event-coverage.json')})
    write_json(root / 'complete.json', done)
    return done


def prepare(c, index, *, source_id=None):
    p = plan(c)
    root = resolve_path(c['cache'])
    root.mkdir(parents=True, exist_ok=True)
    if read(root / 'plan.json') != p:
        raise ValueError('Bind the unchanged study before preparing sources')
    sources = [s for s in p['sources'] if s['id'] == source_id] if source_id is not None else p['sources'][index::c['prepare_tasks']]
    for item in sources:
        done = prepare_source(c, p, item)
        print(json.dumps(dict(source=item['id'], rows=done['rows'], counts=done['counts'])), flush=True)


def seal(c):
    root = resolve_path(c['cache'])
    p = plan(c)
    arrays, banks, offset, records, coverage = [], [], 0, [], []
    for item in p['sources']:
        folder = root / 'sources' / str(item['id'])
        done = read(folder / 'complete.json')
        if done['identity'] != digest(dict(plan=p['identity'], source=item)):
            raise ValueError('Prepared source identity changed')
        for f, h in done['files'].items():
            if sha(folder / f) != h:
                raise ValueError(f'Changed source histories {item["id"]}/{f}')
        with np.load(folder / 'rows.npz') as a:
            rows = {k: a[k].copy() for k in a.files}
        rows['indices'] += offset
        bank = np.load(folder / 'positions.npy', mmap_mode='r')
        arrays.append(rows)
        banks.append(bank)
        offset += len(bank)
        records.append(done)
        coverage.append(dict(source=item['id'], role=item['role'], **done['counts']))
    meta = {k: np.concatenate([a[k] for a in arrays]) for k in arrays[0]}
    meta['id'] = np.array([f'{s}:{a}:{e}:{y}' for s, a, e, y in
                          zip(meta['source'], meta['atom'], meta['end_frame'], meta['label'])])
    if len(set(meta['id'])) != len(meta['id']):
        raise ValueError('Duplicate released sequence identities')
    # Every event contributes equal total mass; each matched case/control set
    # retains the declared 1:4 prevalence. Correlated anchors are not new events.
    group = np.array([f'{s}:{e}' for s, e in zip(meta['source'], meta['event'])])
    totals = Counter(group)
    meta['weight'] = np.array([1 / totals[g] for g in group], float)
    np.savez_compressed(root / 'rows.npz', **meta)
    np.save(root / 'positions.npy', np.concatenate(banks))
    summary = []
    supported = True
    for role in ROLES:
        ids = meta['role'] == role
        pos = ids & (meta['label'] == 1)
        n = len(set(group[pos]))
        supported &= n >= c['minimum_events'][role]
        summary.append(dict(role=role, rows=int(ids.sum()), positives=int(pos.sum()),
                            distinct_births=n, sources=int(len(np.unique(meta['source'][ids]))),
                            required_births=c['minimum_events'][role]))
    write_json(root / 'manifest.json', dict(identity=p['identity'], files={n: sha(root / n) for n in
        ('plan.json', 'positions.npy', 'rows.npz')}, sources=records, summary=summary,
        supported=bool(supported), cohort='event-enriched matched case/control classification; not natural-risk calibration'))
    dest = resolve_path(c['output']) / 'analyses/coverage-v1'
    write_metric_rows(summary, dest, family='birth_prediction', name='split-coverage')
    columns = ['source', 'role'] + sorted(set(k for r in coverage for k in r) - {'source', 'role'})
    write_metric_rows([{k: r.get(k, 0) for k in columns} for r in coverage], dest,
                      family='birth_prediction', name='source-coverage', columns=columns)
    if not supported:
        raise ValueError(f'Insufficient independent birth support; see {dest}: {summary}')
    return summary


def load(c):
    root = resolve_path(c['cache'])
    manifest = read(root / 'manifest.json')
    if not manifest['supported']:
        raise ValueError('Birth classification support gate did not pass')
    for name, h in manifest['files'].items():
        if sha(root / name) != h:
            raise ValueError(f'Changed birth classification release: {name}')
    with np.load(root / 'rows.npz') as arrays:
        rows = {k: arrays[k] for k in arrays.files}
    return np.load(root / 'positions.npy', mmap_mode='r'), rows, manifest
