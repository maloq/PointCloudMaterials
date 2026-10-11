"""Shooting-specific fate adapter: physical durations and parent-present embryos."""
from collections import Counter, defaultdict
from functools import lru_cache
import time
import fcntl

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.trajectories.shooting import _array_sha256
from src.postprocessing.protocols import ancestry, fates, external_data as data, external_audit as audit
from src.postprocessing.protocols.shooting_data import read


class Observations(audit.FrameAccess):
    def __init__(self, plan, item, graph):
        super().__init__(plan, item, graph)
        part = item['parts'][0]
        self.atom_ids = data.arrays(part['path'], part['manifest_sha256'])['atom_ids']
        self.cached_frame = lru_cache(maxsize=8)(super().frame)

    def frame(self, frame):
        value = self.cached_frame(int(frame))
        _, _, dense = value
        nodes = np.flatnonzero(self.graph['frame'] == frame)
        ids, counts = np.unique(dense[dense > 0], return_counts=True)
        if not np.array_equal(ids, nodes) or not np.array_equal(counts, self.graph['size'][nodes]):
            raise ValueError(f'Changed shooting component reconstruction: {self.source["id"]}/{frame}')
        return value

    def members(self, nodes):
        by_frame = defaultdict(list)
        for node in nodes:
            by_frame[int(self.graph['frame'][node])].append(node)
        values = [np.flatnonzero(np.isin(self.frame(frame)[2], ids)) for frame, ids in sorted(by_frame.items())]
        return np.unique(np.concatenate(values))


def evaluate(c, item, g, events, roots, uncertain, observations):
    parents, children = fates.graph_links(g)
    common = dict(source=item['id'], campaign=item['campaign'], branch_id=item['branch_id'],
        parent_id=item['parent_id'], parent_sha256=item['parent_sha256'], lineage=item['ancestry_group'],
        role=item['role'], cadence_ps=item['cadence_ps'], material=item['material'], potential=item['potential'],
        dynamics=item['dynamics'], branch_duration_ps=item['branch_observed_duration_ps'])
    n = len(g['size'])
    edges = g['edges']
    _, components = connected_components(coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(n, n)).tocsr(), directed=False)
    strong_edges = edges[edges[:, 3] == 1]
    _, strong_components = connected_components(coo_matrix((np.ones(len(strong_edges)),
        (strong_edges[:, 0], strong_edges[:, 1])), shape=(n, n)).tocsr(), directed=False)
    groups = defaultdict(list)
    for node in range(1, n):
        groups[int(components[node])].append(node)
    established_groups = {int(components[e['birth_node']]) for e in events}
    transient, established = [], []
    counts = Counter(episodes=len(groups), established_connected_episodes=len(established_groups))
    for component, nodes in groups.items():
        if component in established_groups:
            continue  # Even weak contact excludes a putative failed independent nucleus.
        sizes = g['size'][nodes]
        first, last = int(g['frame'][nodes].min()), int(g['frame'][nodes].max())
        start_nodes = [node for node in nodes if not parents[node]]
        persistent = fates.qualified_streak(g, nodes, parents, c['primary_minimum_atoms'], c['primary_persistence_frames'])
        local_edges = edges[components[edges[:, 0]] == component]
        record = dict(**common, episode=f'{item["id"]}:component:{min(nodes)}',
            first_node=min(nodes), first_frame=first, last_frame=last,
            observed_span_ps=float(item['times_ps'][last] - item['times_ps'][first]), observations=len(np.unique(g['frame'][nodes])),
            peak_atoms=int(sizes.max()), root_count=len(start_nodes),
            strong_lineage=len(np.unique(strong_components[nodes])) == 1,
            weak_links=int((local_edges[:, 3] == 0).sum()), gap_links=int((local_edges[:, 4] > 1).sum()),
            branching=any(len(parents[x]) > 1 or len(children[x]) > 1 for x in nodes),
            primary_size_duration=bool(persistent is not None), left_censored=first == 0,
            terminal_status='below_verification_size', dissolution_frame=None,
            terminal_core_atoms=None, residual_crystal_atoms=None,
            origin='not_checked', nearest_established_crystal_A=None, anchor_atom=None,
            centroid_A=None, primary_failed_candidate=False)
        if record['peak_atoms'] >= c['minimum_verified_peak_atoms']:
            record.update(fates.verify_disappearance(c, g, nodes, children, observations, item['frame_count']))
            nearest, ambiguous = [], False
            for node in start_nodes:
                frame = int(g['frame'][node])
                xyz, box, dense = observations.frame(frame)
                crystal = xyz[dense == node]
                displacement = crystal - crystal[0]
                displacement -= box * np.rint(displacement / box)
                ambiguous |= bool(np.any(np.ptp(displacement, axis=0) > box / 2))
                center = np.mod(crystal[0] + displacement.mean(0), box)
                if node == min(nodes):
                    record['centroid_A'] = center.tolist()
                    offsets = xyz - center
                    offsets -= box * np.rint(offsets / box)
                    record['anchor_atom'] = int(observations.atom_ids[np.argmin(np.linalg.norm(offsets, axis=1))])
                old_nodes = [k for k in np.flatnonzero(g['frame'] == frame)
                             if any(events[r-1]['confirmation_frame'] <= frame for r in roots[k])]
                old_atoms = np.isin(dense, old_nodes)
                if old_atoms.any():
                    nearest.append(float(cKDTree(xyz[old_atoms], boxsize=box).query(crystal)[0].min()))
            distance = min(nearest) if nearest else None
            record['nearest_established_crystal_A'] = distance
            record['origin'] = ('left_censored' if first == 0 else
                'unresolved_periodic_extent' if ambiguous else
                'interface_associated' if distance is not None and distance <= item['lineage']['interface_distance_A']
                else 'isolated')
            # Multiple roots are one encounter episode, not multiple independent nuclei.
            record['primary_failed_candidate'] = bool(record['terminal_status'] == 'dissolved'
                and record['origin'] == 'isolated' and persistent is not None and len(start_nodes) == 1
                and record['strong_lineage'])
        transient.append(record)
    for event in events:
        eid = event['event_id']
        all_nodes = [node for node in range(1, n) if eid in roots[node]]
        merged = [node for node in all_nodes if len(roots[node]) > 1]
        merge_frame = int(g['frame'][merged].min()) if merged else None
        exclusive = [node for node in all_nodes if len(roots[node]) == 1
                     and (merge_frame is None or g['frame'][node] < merge_frame)]
        if not exclusive:
            raise ValueError(f'Established event lacks exclusive origin: {item["id"]}/{eid}')
        reference = int(g['size'][event['confirmation_node']])
        growth_size = max(c['growth_minimum_atoms'], int(np.ceil(c['growth_factor'] * reference)))
        growth_nodes = [node for node in exclusive if not uncertain[node]
                        and g['frame'][node] > event['confirmation_frame']]
        growth_frame = fates.qualified_streak(g, growth_nodes, parents, growth_size, c['growth_persistence_frames'])
        record = dict(**common, event=eid, kind=event['kind'], establishment_kind=event['establishment_kind'],
            initial_cluster_ancestry=event['initial_cluster_ancestry'], birth_frame=event['birth_frame'],
            confirmation_frame=event['confirmation_frame'], confirmation_atoms=reference,
            growth_threshold_atoms=growth_size, growth_confirmed=growth_frame is not None,
            growth_confirmation_frame=growth_frame, merge_frame=merge_frame,
            last_exclusive_frame=int(g['frame'][exclusive].max()),
            peak_exclusive_atoms=int(g['size'][exclusive].max()),
            terminal_status='merged' if merged else None, dissolution_frame=None,
            terminal_core_atoms=None, residual_crystal_atoms=None)
        if not merged:
            record.update(fates.verify_disappearance(c, g, exclusive, children, observations, item['frame_count']))
        terminal = record['terminal_status']
        record['fate_label'] = ('grew_then_dissolved' if growth_frame is not None and terminal == 'dissolved'
            else 'dissolved' if terminal == 'dissolved'
            else 'continued_growth' if growth_frame is not None
            else 'unresolved_merge' if terminal == 'merged'
            else 'right_censored' if terminal == 'right_censored'
            else 'unresolved_disappearance')
        established.append(record)
    counts.update(transient_episodes=len(transient), atom_verified_episodes=sum(
        r['terminal_status'] != 'below_verification_size' for r in transient),
        primary_failed_candidates=sum(r['primary_failed_candidate'] for r in transient), established=len(established))
    for record in transient:
        record['brief_failed_candidate'] = bool(record['terminal_status'] == 'dissolved'
            and record['origin'] == 'isolated' and record['root_count'] == 1 and record['strong_lineage'])
        record['first_time_ps'] = item['times_ps'][record['first_frame']]
        record['last_time_ps'] = item['times_ps'][record['last_frame']]
    for record in established:
        record['birth_time_ps'] = item['times_ps'][record['birth_frame']]
        record['confirmation_time_ps'] = item['times_ps'][record['confirmation_frame']]
    counts.update(isolated_new_establishments=sum(r['kind'] == 'isolated_establishment' for r in established),
        parent_present_embryo_growth=sum(r['kind'] == 'parent_present_embryo_growth' for r in established),
        brief_failed_candidates=sum(r['brief_failed_candidate'] for r in transient))
    return transient, established, dict(counts)


def process(plan, item):
    out = data.source_folder(plan, item['id'])
    out.mkdir(parents=True, exist_ok=True)
    identity = digest(dict(plan=plan['identity'], source=item['id']))
    done = out / 'complete.json'
    with (out / 'source.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if done.exists():
            receipt = read(done)
            if receipt['identity'] != identity or any(sha(out / k) != v for k,v in receipt['files'].items()):
                raise ValueError(f'Changed shooting audit: {out}')
            return receipt
        started = time.monotonic()
        # Bind actual consumed arrays, not merely an old manifest checksum.
        for part in item['parts']:
            a = data.arrays(part['path'], part['manifest_sha256'])
            for name, array in a.items():
                if _array_sha256(array) != part['array_sha256'][name]:
                    raise ValueError(f'Changed trajectory array: {part["path"]}/{name}')
        write_json(out / 'input-verified.json', dict(identity=identity, parts=item['parts']))
        tasks = [t for t in plan['tasks'] if t['source'] == item['id']]
        for number, task in enumerate(tasks):
            data.extract_chunk(plan, task)
            write_json(out / 'progress.json', dict(stage='PTM', completed=number+1, total=len(tasks),
                seconds=time.monotonic()-started))
        audit.seal_chunks(plan, item)
        g = audit.graph_for_source(plan, item, identity)
        observations = Observations(plan, item, g)
        threshold = item['lineage']['thresholds'][0]
        events, roots, uncertain = ancestry.establish(g, threshold)
        audit.locate(item, g, events, roots, observations)
        parents, _ = fates.graph_links(g)
        first = g['frame'].copy()
        for node in range(1, len(first)):
            first[node] = min([int(first[node])] + [int(first[p]) for p, _, _ in parents[node]])
        for event in events:
            event['center_distances_A'] = event['center_distances_A'].tolist()
            event['establishment_kind'] = event['kind']
            event['initial_cluster_ancestry'] = bool(first[event['birth_node']] == 0)
            # A subcritical embryo already in the parent is not a new birth,
            # even when it first passes the establishment threshold in the shot.
            if event['initial_cluster_ancestry'] and not event['left_censored']:
                event['kind'] = 'parent_present_embryo_growth'
        c = dict(plan['config']['fates'])
        dt = item['cadence_ps']
        c['primary_persistence_frames'] = 1 + int(np.ceil(c['primary_persistence_ps'] / dt - 1e-9))
        c['growth_persistence_frames'] = 1 + int(np.ceil(c['growth_persistence_ps'] / dt - 1e-9))
        c['dissolution_confirmation_frames'] = int(np.ceil(c['dissolution_followup_ps'] / dt - 1e-9))
        transient, established, counts = evaluate(c, item, g, events, roots, uncertain, observations)
        for name, values in [('events',events), ('transient-episodes',transient), ('established-fates',established)]:
            write_json(out / (name + '.json'), values)
        files = ['events.json','transient-episodes.json','established-fates.json','graph.npz','ptm-complete.json',
                 'input-verified.json']
        receipt = dict(identity=identity, source=item['id'], counts=counts, seconds=time.monotonic()-started,
            actual_durations=dict(cadence_ps=dt, establishment_span_ps=(threshold['persistence_frames']-1)*dt,
                primary_failed_span_ps=(c['primary_persistence_frames']-1)*dt,
                growth_span_ps=(c['growth_persistence_frames']-1)*dt,
                dissolution_followup_ps=c['dissolution_confirmation_frames']*dt),
            files={k: sha(out / k) for k in files})
        write_json(done, receipt)
        observations.cached_frame.cache_clear()
        data.arrays.cache_clear()
        return receipt
