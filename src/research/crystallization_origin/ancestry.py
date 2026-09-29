"""Periodic cluster graph and conservative establishment/arrival attribution.

This is an operational PTM lineage audit, not a critical-nucleus estimator.
All look-ahead belongs to label construction, never to a model input.
"""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.data.fixed_cohort.protocol import digest, sha, write_json
from .extract import frame_geometry, raw_source


def components(points, box, labels, settings):
    """All crystalline atoms in the periodic cell; 0 means untracked/liquid."""
    atoms = np.flatnonzero(np.isin(labels, [1, 2, 3]))
    dense = np.zeros(len(points), dtype=np.int32)
    if not len(atoms):
        return dense, np.zeros(1, dtype=np.int32)
    pairs = cKDTree(points[atoms], boxsize=box).query_pairs(settings['neighbor_cutoff_A'], output_type='ndarray')
    graph = coo_matrix((np.ones(len(pairs), dtype=np.uint8), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(atoms), len(atoms))).tocsr()
    _, component = connected_components(graph, directed=False)
    sizes = np.bincount(component)
    keep = sizes >= settings['minimum_tracked_size']
    renumber = np.zeros(len(sizes), dtype=np.int32)
    renumber[keep] = np.arange(1, keep.sum() + 1)
    dense[atoms] = renumber[component]
    return dense, np.r_[0, sizes[keep]].astype(np.int32)


def overlap(previous, current, sizes, current_sizes, settings):
    mask = (previous > 0) & (current > 0)
    base = len(current_sizes)
    keys, counts = np.unique(previous[mask].astype(np.int64) * base + current[mask], return_counts=True)
    parent, child = keys // base, keys % base
    parent_sizes = np.asarray([sizes[int(p)] for p in parent])
    strong = ((counts >= settings['minimum_shared_atoms']) &
              (counts >= settings['minimum_overlap_fraction'] * np.minimum(parent_sizes, current_sizes[child])))
    return parent, child, counts, strong


def build_graph(config, item, folder):
    settings = config['lineage']
    extraction = json.loads((folder / 'ptm-complete.json').read_text())
    identity = digest(dict(extraction=extraction['identity'], settings=settings, producer=sha(Path(__file__))))
    path, receipt = folder / 'graph.npz', folder / 'graph-complete.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['identity'] != identity or saved['sha256'] != sha(path):
            raise ValueError(f'Changed ancestry graph: {folder}')
        with np.load(path) as arrays:
            return {k: arrays[k] for k in arrays.files}, saved
    raw = raw_source(item)
    rows = np.searchsorted(raw.atom_ids, item['center_atom_ids'])
    frames, sizes = [-1], [0]
    starts, centers, crystalline = [], [], []
    edges = []
    previous = []
    chunk_size = config['ptm']['chunk_frames']
    for start in range(0, item['frame_count'], chunk_size):
        stop = min(item['frame_count'], start + chunk_size)
        chunk_path = folder / f'ptm-{start:04d}-{stop:04d}.npz'
        if sha(chunk_path) != extraction['files'][chunk_path.name]:
            raise ValueError(f'Changed PTM chunk: {chunk_path}')
        with np.load(chunk_path) as data:
            labels = data['labels']
        for offset, value in enumerate(labels):
            frame = start + offset
            points, box = frame_geometry(raw, frame)
            local, local_sizes = components(points, box, value, settings)
            node_start = len(sizes) - 1
            starts.append(node_start)
            immediate_strong = set()
            for lag, old in enumerate(reversed(previous), 1):
                parent, child, counts, strong = overlap(old, local, sizes, local_sizes, settings)
                if lag == 1:
                    immediate_strong = set(child[strong].tolist())
                else:
                    # A missing-frame link can restore a lost lineage, never add an
                    # extra stale ancestor to a component already matched strongly.
                    retain = np.array([int(c) not in immediate_strong for c in child], dtype=bool)
                    parent, child, counts, strong = (v[retain] for v in (parent, child, counts, strong))
                edges.extend(zip(parent.tolist(), (child + node_start).tolist(), counts.tolist(),
                                 strong.astype(int).tolist(), [lag] * len(parent)))
            global_nodes = np.where(local > 0, local + node_start, 0).astype(np.int32)
            centers.append(global_nodes[rows])
            frames.extend([frame] * (len(local_sizes) - 1))
            sizes.extend(local_sizes[1:].tolist())
            crystalline.append(int(np.isin(value, [1, 2, 3]).sum()))
            previous.append(global_nodes)
            previous = previous[-(settings['maximum_missing_frames'] + 1):]
        write_json(folder / 'graph-progress.json', dict(frames=stop, total_frames=item['frame_count'],
                    nodes=len(sizes) - 1, edges=len(edges)))
    graph = dict(frame=np.asarray(frames, np.int32), size=np.asarray(sizes, np.int32),
                 start=np.asarray(starts, np.int32), center_node=np.stack(centers).T,
                 crystalline_atoms=np.asarray(crystalline, np.int32),
                 edges=np.asarray(edges, np.int64).reshape(-1, 5))
    temporary = path.with_suffix('.building.npz')
    np.savez_compressed(temporary, **graph)
    temporary.replace(path)
    saved = dict(identity=identity, sha256=sha(path), nodes=len(sizes) - 1, edges=len(edges))
    write_json(receipt, saved)
    return graph, saved


def establish(graph, threshold):
    """A lineage is established after K atoms for P adjacent observed frames.

    Strong overlap matches carry streaks. Every established ancestor survives a
    merge/split or a weak overlap; weak-only inheritance is explicitly uncertain.
    Persistence confirmation backfills its actual streak, for outcome labels only.
    """
    n = len(graph['size'])
    parents = [[] for _ in range(n)]
    children = [[] for _ in range(n)]
    for parent, child, count, strong, lag in graph['edges']:
        parents[child].append((int(parent), int(count), bool(strong), int(lag)))
        children[parent].append(int(child))
    roots = [set() for _ in range(n)]
    certain = [set() for _ in range(n)]
    uncertain = np.zeros(n, bool)
    streak = np.zeros(n, np.int32)
    predecessor = np.zeros(n, np.int32)
    events = []
    persistence = threshold['persistence_frames']
    for node in range(1, n):
        all_roots = set().union(*(roots[p] for p, _, _, _ in parents[node]))
        roots[node].update(all_roots)
        if all_roots or graph['size'][node] < threshold['size']:
            continue
        eligible = [(int(streak[p]), count, -p) for p, count, strong, lag in parents[node]
                    if strong and lag == 1 and streak[p] > 0]
        if eligible:
            length, _, negative_parent = max(eligible)
            predecessor[node] = -negative_parent
            streak[node] = length + 1
        else:
            streak[node] = 1
        if streak[node] < persistence:
            continue
        path = [node]
        for _ in range(persistence - 1):
            path.append(int(predecessor[path[-1]]))
        inherited = set().union(*(roots[p] for p in path))
        if inherited:
            roots[node].update(inherited)
            continue
        event_id = len(events) + 1
        birth = path[-1]
        events.append(dict(event_id=event_id, birth_node=birth, confirmation_node=node,
                           birth_frame=int(graph['frame'][birth]), confirmation_frame=int(graph['frame'][node]),
                           birth_size=int(graph['size'][birth]), left_censored=bool(graph['frame'][birth] == 0),
                           establishment_merge_ambiguous=any(sum(strong and lag == 1 and streak[p] > 0
                               for p, _, strong, lag in parents[q]) > 1 for q in path)))
        for p in path:
            certain[p].add(event_id)
        # Also update already visited side branches. Otherwise two branches of
        # the same persistent streak can be counted twice before the final pass.
        pending = list(path)
        while pending:
            p = pending.pop()
            if event_id in roots[p]:
                continue
            roots[p].add(event_id)
            pending.extend(c for c in children[p] if c <= node)
    # Propagate the confirmed backfill to side branches visited before confirmation.
    for node in range(1, n):
        all_roots = set().union(*(roots[p] for p, _, _, _ in parents[node]))
        roots[node].update(all_roots)
        certain[node].update(set().union(*(certain[p] for p, _, strong, _ in parents[node] if strong)))
        # A weak peripheral branch must not erase an independently supported
        # strong ancestry path through a merge. Keep possible extra roots for
        # conservative class comparison, but distinguish weak-only support.
        uncertain[node] = bool(roots[node]) and not certain[node]
    for event in events:
        event.update(peak_single_lineage_size=0, last_single_lineage_frame=event['birth_frame'], ever_merged=False)
    for node in range(1, n):
        for event_id in roots[node]:
            event = events[event_id - 1]
            if len(roots[node]) > 1:
                event['ever_merged'] = True
            else:
                event['peak_single_lineage_size'] = max(event['peak_single_lineage_size'], int(graph['size'][node]))
                event['last_single_lineage_frame'] = max(event['last_single_lineage_frame'], int(graph['frame'][node]))
    return events, roots, uncertain


class GeometryAccess:
    def __init__(self, config, item, folder, graph):
        self.config, self.item, self.folder, self.graph = config, item, folder, graph
        self.raw = raw_source(item)
        self.rows = np.searchsorted(self.raw.atom_ids, item['center_atom_ids'])
        self.chunk_start, self.labels = -1, None
        self.last_frame, self.value = -1, None

    def frame(self, frame):
        if frame == self.last_frame:
            return self.value
        count = self.config['ptm']['chunk_frames']
        start = (frame // count) * count
        if start != self.chunk_start:
            stop = min(self.item['frame_count'], start + count)
            with np.load(self.folder / f'ptm-{start:04d}-{stop:04d}.npz') as data:
                self.labels = data['labels']
            self.chunk_start = start
        points, box = frame_geometry(self.raw, frame)
        local, _ = components(points, box, self.labels[frame - start], self.config['lineage'])
        dense = np.where(local > 0, local + self.graph['start'][frame], 0)
        self.last_frame = frame
        self.value = points, box, dense
        return self.value


def locate_births(config, item, graph, events, roots, geometry):
    by_id = {e['event_id']: e for e in events}
    by_frame = defaultdict(list)
    for e in events:
        by_frame[e['birth_frame']].append(e)
    for frame, births in sorted(by_frame.items()):
        points, box, dense = geometry.frame(frame)
        nodes = np.flatnonzero(graph['frame'] == frame)
        for event in births:
            member = dense == event['birth_node']
            crystal = points[member]
            if len(crystal) != event['birth_size']:
                raise ValueError(f'Cluster reconstruction changed for {item["id"]}/{frame}')
            displacement = crystal - crystal[0]
            displacement -= box * np.rint(displacement / box)
            center = np.mod(crystal[0] + displacement.mean(0), box)
            extent_ambiguous = bool(np.any(np.ptp(displacement, axis=0) > box / 2))
            offsets = points[geometry.rows] - center
            offsets -= box * np.rint(offsets / box)
            distances = np.linalg.norm(offsets, axis=-1)
            existing_nodes = [n for n in nodes if any(r != event['event_id'] and
                              by_id[r]['confirmation_frame'] <= frame for r in roots[n])]
            old = np.isin(dense, existing_nodes) if existing_nodes else np.zeros(len(dense), bool)
            nearest = float(cKDTree(points[old], boxsize=box).query(crystal, workers=1)[0].min()) if old.any() else None
            if event['left_censored']:
                kind = 'left_censored_existing'
            elif event['establishment_merge_ambiguous']:
                kind = 'unresolved_establishment_merge'
            elif extent_ambiguous:
                kind = 'unresolved_periodic_extent'
            elif nearest is not None and nearest <= config['lineage']['interface_distance_A']:
                kind = 'interface_associated'
            else:
                kind = 'isolated_establishment'
            covered = distances <= config['lineage']['local_birth_radius_A']
            legacy = np.isin(item['center_atom_ids'], item['legacy_center_atom_ids'])
            event.update(kind=kind, centroid_A=center.tolist(), center_distances_A=distances.tolist(),
                         nearest_established_crystal_A=nearest, covered_centers_all64=int(covered.sum()),
                         covered_centers_legacy16=int((covered & legacy).sum()), periodic_extent_ambiguous=extent_ambiguous)
    return events


LABELS = ('no_onset_within_horizon', 'existing_crystal_arrival', 'local_nucleus_establishment',
          'external_new_crystal_arrival', 'formation_in_progress', 'interface_associated_new_cluster',
          'unresolved_unestablished', 'unresolved_ancestry', 'unresolved_ptm_disagreement')


def onset_roots(graph, roots, uncertain, onset, center, persistence):
    if onset + persistence > graph['center_node'].shape[1]:
        return set(), 'unresolved_unestablished'
    nodes = graph['center_node'][center, onset:onset + persistence]
    if any(n == 0 or not roots[n] for n in nodes):
        return set(), 'unresolved_unestablished'
    if any(uncertain[n] for n in nodes):
        return set().union(*(roots[n] for n in nodes)), 'unresolved_ancestry'
    return set().union(*(roots[n] for n in nodes)), None


def classify(origin, onset, center, root_ids, reason, events, radius):
    if reason:
        return reason
    labels = set()
    for event_id in root_ids:
        event = events[event_id - 1]
        if event['confirmation_frame'] <= origin:
            label = 'existing_crystal_arrival'
        elif event['birth_frame'] <= origin:
            label = 'formation_in_progress'
        elif event['birth_frame'] > onset:
            label = 'unresolved_unestablished'
        elif event['kind'] == 'isolated_establishment':
            label = ('local_nucleus_establishment' if event['center_distances_A'][center] <= radius
                     else 'external_new_crystal_arrival')
        elif event['kind'] == 'interface_associated':
            label = 'interface_associated_new_cluster'
        else:
            label = 'unresolved_ancestry'
        labels.add(label)
    return next(iter(labels)) if len(labels) == 1 else 'unresolved_ancestry'
