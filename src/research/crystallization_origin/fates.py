"""Finite-observation cluster dissolution and post-establishment growth audit."""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import resolve_path
from src.research.birth_prediction.data import SourceFrames, plan, read
from . import ancestry, extract

FAMILY = 'nucleus_fates'


def binding(c):
    b = read(resolve_path(c['birth_config']))
    p = plan(b)
    cache = resolve_path(b['cache'])
    manifest = read(cache / 'manifest.json')
    for name in ('plan.json', 'rows.npz'):
        if sha(cache / name) != manifest['files'][name]:
            raise ValueError(f'Changed birth cohort: {cache / name}')
    return b, p, dict(config=c, birth_identity=manifest['identity'],
        birth_rows_sha256=manifest['files']['rows.npz'], audit_identity=p['binding']['audit_identity'],
        producer_sha256=sha(Path(__file__)), ancestry_sha256=sha(Path(ancestry.__file__)))


def prepare(c):
    check_metric_docs(family=FAMILY)
    _, p, identity = binding(c)
    record = dict(identity=digest(identity), binding=identity, sources=p['sources'])
    path = resolve_path(c['output']) / 'technical/prepared.json'
    if path.exists() and read(path) != record:
        raise ValueError('Fate definition changed; use a new output revision')
    write_json(path, record)
    return record


class Observations:
    """Reuse verified PTM chunks; reconstruct membership only at requested frames."""
    def __init__(self, p, item, graph):
        self.access = SourceFrames(p, item)
        self.settings = p['audit']['lineage']
        self.graph = graph
        self.last_frame = -1

    def labels(self, frame):
        start = frame // self.access.chunk * self.access.chunk
        return self.access.labels(start)[frame - start]

    def frame(self, frame):
        if frame != self.last_frame:
            points, box = extract.frame_geometry(self.access.raw, frame)
            dense, sizes = ancestry.components(points, box, self.labels(frame), self.settings)
            nodes = np.flatnonzero(self.graph['frame'] == frame)
            if not np.array_equal(sizes[1:], self.graph['size'][nodes]):
                raise ValueError(f'Changed cluster reconstruction: {self.access.item["id"]}/{frame}')
            dense = np.where(dense > 0, dense + self.graph['start'][frame], 0)
            self.value = points, box, dense
            self.last_frame = frame
        return self.value

    def members(self, nodes):
        by_frame = defaultdict(list)
        for node in nodes:
            by_frame[int(self.graph['frame'][node])].append(node)
        members = []
        for frame, ids in sorted(by_frame.items()):
            _, _, dense = self.frame(frame)
            members.append(np.flatnonzero(np.isin(dense, ids)))
        return np.unique(np.concatenate(members))


def graph_links(g):
    n = len(g['size'])
    parents, children = [[] for _ in range(n)], [[] for _ in range(n)]
    for parent, child, count, strong, lag in g['edges']:
        parents[child].append((int(parent), bool(strong), int(lag)))
        children[parent].append(int(child))
    return parents, children


def qualified_streak(g, nodes, parents, minimum, persistence):
    """Adjacent, strong, size-qualified observations within the declared node set."""
    lengths = {}
    for node in sorted(nodes):
        if g['size'][node] >= minimum:
            lengths[node] = 1 + max((lengths.get(parent, 0) for parent, strong, lag in parents[node]
                                     if strong and lag == 1), default=0)
            if lengths[node] >= persistence:
                return int(g['frame'][node])
    return None


def verify_disappearance(c, g, nodes, children, observations, frames):
    """All terminal-branch atom IDs must be PTM liquid at each confirmation frame."""
    last = int(g['frame'][nodes].max())
    count = c['dissolution_confirmation_frames']
    if last + count >= frames:
        return dict(terminal_status='right_censored', dissolution_frame=None,
                    terminal_core_atoms=None, residual_crystal_atoms=None)
    selected = set(nodes)
    leaves = [node for node in nodes if not any(child in selected for child in children[node])]
    # The caller must not turn an outgoing merge into a disappearance.
    if any(children[node] for node in leaves):
        raise ValueError('Dissolution verification attempted on an externally connected lineage')
    members = observations.members(leaves)
    residual = [int(np.isin(observations.labels(frame)[members], [1, 2, 3]).sum())
                for frame in range(last + 1, last + count + 1)]
    dissolved = max(residual) == 0
    return dict(terminal_status='dissolved' if dissolved else 'unresolved_residual_crystal',
                dissolution_frame=last + count if dissolved else None,
                terminal_core_atoms=len(members), residual_crystal_atoms=max(residual))


def source(c, p, item, identity):
    out = resolve_path(c['output']) / 'technical/sources' / str(item['id'])
    out.mkdir(parents=True, exist_ok=True)
    done = out / 'complete.json'
    if done.exists():
        receipt = read(done)
        if receipt['identity'] != identity or any(sha(out / k) != h for k, h in receipt['files'].items()):
            raise ValueError(f'Changed fate results: {out}')
        return receipt
    started = time.monotonic()
    old = resolve_path(p['audit']['output']) / 'technical/sources' / str(item['id'])
    graph_receipt, audit_receipt = read(old / 'graph-complete.json'), read(old / 'audit.json')
    if sha(old / 'graph.npz') != graph_receipt['sha256'] or sha(old / 'events.json') != audit_receipt['files']['events.json']:
        raise ValueError(f'Changed source audit {item["id"]}')
    with np.load(old / 'graph.npz') as arrays:
        g = {k: arrays[k] for k in arrays.files}
    events = [e for e in read(old / 'events.json') if e['threshold'] == 'primary']
    threshold = next(t for t in p['audit']['lineage']['thresholds'] if t['name'] == 'primary')
    rebuilt, roots, uncertain = ancestry.establish(g, threshold)
    if [(e['event_id'], e['birth_node'], e['confirmation_node']) for e in events] != [
            (e['event_id'], e['birth_node'], e['confirmation_node']) for e in rebuilt]:
        raise ValueError(f'Changed primary establishments: {item["id"]}')
    parents, children = graph_links(g)
    observations = Observations(p, item, g)  # Also checks the actual source timeline.
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
    common = dict(source=item['id'], role=item['role'], lineage=item['lineage'],
                  cadence_ps=.75, material='Al', potential='al-lee2003-meam')
    for component, nodes in groups.items():
        if component in established_groups:
            continue  # Even weak contact excludes a putative failed independent nucleus.
        sizes = g['size'][nodes]
        first, last = int(g['frame'][nodes].min()), int(g['frame'][nodes].max())
        start_nodes = [node for node in nodes if not parents[node]]
        persistent = qualified_streak(g, nodes, parents, c['primary_minimum_atoms'], c['primary_persistence_frames'])
        local_edges = edges[components[edges[:, 0]] == component]
        record = dict(**common, episode=f'{item["id"]}:component:{min(nodes)}',
            first_node=min(nodes), first_frame=first, last_frame=last,
            observed_span_ps=(last-first)*.75, observations=len(np.unique(g['frame'][nodes])),
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
            record.update(verify_disappearance(c, g, nodes, children, observations, item['frame_count']))
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
                    record['anchor_atom'] = int(observations.access.raw.atom_ids[np.argmin(np.linalg.norm(offsets, axis=1))])
                old_nodes = [k for k in np.flatnonzero(g['frame'] == frame)
                             if any(events[r-1]['confirmation_frame'] <= frame for r in roots[k])]
                old_atoms = np.isin(dense, old_nodes)
                if old_atoms.any():
                    nearest.append(float(cKDTree(xyz[old_atoms], boxsize=box).query(crystal)[0].min()))
            distance = min(nearest) if nearest else None
            record['nearest_established_crystal_A'] = distance
            record['origin'] = ('left_censored' if first == 0 else
                'unresolved_periodic_extent' if ambiguous else
                'interface_associated' if distance is not None and distance <= p['audit']['lineage']['interface_distance_A']
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
        growth_frame = qualified_streak(g, growth_nodes, parents, growth_size, c['growth_persistence_frames'])
        record = dict(**common, event=eid, kind=event['kind'], birth_frame=event['birth_frame'],
            confirmation_frame=event['confirmation_frame'], confirmation_atoms=reference,
            growth_threshold_atoms=growth_size, growth_confirmed=growth_frame is not None,
            growth_confirmation_frame=growth_frame, merge_frame=merge_frame,
            last_exclusive_frame=int(g['frame'][exclusive].max()),
            peak_exclusive_atoms=int(g['size'][exclusive].max()),
            terminal_status='merged' if merged else None, dissolution_frame=None,
            terminal_core_atoms=None, residual_crystal_atoms=None)
        if not merged:
            record.update(verify_disappearance(c, g, exclusive, children, observations, item['frame_count']))
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
    write_json(out / 'transient-episodes.json', transient)
    write_json(out / 'established-fates.json', established)
    receipt = dict(identity=identity, source=item['id'], role=item['role'], counts=dict(counts),
        seconds=time.monotonic()-started, graph_sha256=graph_receipt['sha256'],
        events_sha256=audit_receipt['files']['events.json'],
        files={k: sha(out / k) for k in ('transient-episodes.json', 'established-fates.json')})
    write_json(done, receipt)
    return receipt


def collect(c):
    b, _, bound = binding(c)
    prepared = read(resolve_path(c['output']) / 'technical/prepared.json')
    if digest(bound) != prepared['identity']:
        raise ValueError('Collector identity changed')
    root = resolve_path(c['output'])
    transient, established = [], []
    for item in prepared['sources']:
        folder = root / 'technical/sources' / str(item['id'])
        done = read(folder / 'complete.json')
        if done['identity'] != prepared['identity'] or any(sha(folder / k) != h for k,h in done['files'].items()):
            raise ValueError(f'Incomplete/changed fate source {item["id"]}')
        transient.extend(read(folder / 'transient-episodes.json'))
        established.extend(read(folder / 'established-fates.json'))
    with np.load(resolve_path(b['cache']) / 'rows.npz') as data:
        rows = {k: data[k] for k in data.files}
    lookup = {(r['source'], r['event']): r for r in established}
    retained = {(int(s), int(e)) for s,e,y in zip(rows['source'],rows['event'],rows['label']) if y == 1}
    sample_labels = []
    for index in range(len(rows['label'])):
        positive = int(rows['label'][index]) == 1
        event = lookup[(int(rows['source'][index]), int(rows['event'][index]))]
        sample_labels.append(dict(row_id=str(rows['id'][index]), source=int(rows['source'][index]),
            event=int(rows['event'][index]), atom=int(rows['atom'][index]),
            role=str(rows['role'][index]), evaluation_role='train' if rows['role'][index] == 'train' else 'merged_test',
            original_label=int(rows['label'][index]),
            fate_label=event['fate_label'] if positive else 'not_applicable_liquid_control',
            growth_confirmed=event['growth_confirmed'] if positive else None,
            terminal_status=event['terminal_status'] if positive else None))
    retained_events = [dict(r, retained_in_birth_cohort=(r['source'], r['event']) in retained) for r in established]
    dest = root / 'analyses/fates-v1'
    for name, values in [('transient-episodes',transient),('established-fates',retained_events),('birth-row-fate-labels',sample_labels)]:
        write_metric_rows(values, dest, family=FAMILY, name=name)
    summaries = []
    populations = [('existing_birth_events',[r for r in retained_events if r['retained_in_birth_cohort']], 'fate_label'),
        ('all_established_events',established,'fate_label'), ('all_unestablished_episodes',transient,'terminal_status'),
        ('primary_failed_candidates',[r for r in transient if r['primary_failed_candidate']], 'terminal_status')]
    for population, values, field in populations:
        for role in ['all', 'train', 'selection', 'calibration', 'test']:
            selected = [r for r in values if role == 'all' or r['role'] == role]
            for label in sorted(set(r[field] for r in selected)):
                match = [r for r in selected if r[field] == label]
                summaries.append(dict(population=population, role=role, label=label,
                    count=len(match), sources=len(set(r['source'] for r in match))))
    write_metric_rows(summaries, dest, family=FAMILY, name='fate-counts')
    sensitivity = []
    for minimum in [8,16,32,64]:
        for persistent in [False,True]:
            selected = [r for r in transient if r['terminal_status']=='dissolved' and r['origin']=='isolated'
                and r['root_count']==1 and r['strong_lineage'] and r['peak_atoms'] >= minimum
                and (not persistent or r['primary_size_duration'])]
            sensitivity.append(dict(minimum_peak_atoms=minimum, requires_two_adjacent_ge8_frames=persistent,
                                    episodes=len(selected), sources=len(set(r['source'] for r in selected))))
    write_metric_rows(sensitivity, dest, family=FAMILY, name='failed-size-duration-sensitivity')
    lines = ['# Nucleus fate audit', '',
        f'Complete: {len(prepared["sources"])} sources; {len(retained)} existing birth events; {len(sample_labels)} unchanged sample rows.', '',
        'Growth means at least doubling the confirmation size (and reaching at least 128 atoms) for three later adjacent observations before merging.',
        'Dissolution requires no remaining tracked descendants and all terminal-branch atom IDs PTM-liquid for three later observations. This is finite-observation evidence.', '',
        '| Population | Label | Count | Sources |', '|---|---|---:|---:|']
    lines += [f'| {r["population"]} | {r["label"]} | {r["count"]} | {r["sources"]} |' for r in summaries if r['role']=='all']
    lines += ['', 'Primary failed candidates are isolated single-origin episodes with ≥8 atoms in two adjacent observations, no graph connection (even weak) to any primary establishment, and confirmed disappearance.',
        'Four-to-seven-atom episodes remain in the inventory without atom-level disappearance verification. Multiple-origin encounters, interface-associated episodes and ambiguous endings remain explicit.',
        'Candidate histories have not yet been screened for the strict pre-appearance crystal-free input rule. These are label-side harvest candidates, not new classifier training samples.',
        'The historical birth labels and active predictor fits are unchanged. Both original and relaxed inputs share these outcome labels; only positive rows inherit their nucleus fate.', '',
        '[Counts](tables/fate-counts.csv) · [Existing events](tables/established-fates.csv) · [Sample labels](tables/birth-row-fate-labels.csv) · [Candidate inventory](tables/transient-episodes.csv) · [Definitions](tables/METRICS.md)', '']
    (dest / 'README.md').write_text('\n'.join(lines))
    write_json(root / 'technical/complete.json', dict(identity=prepared['identity'], sources=len(prepared['sources']),
        retained_events=len(retained), sample_rows=len(sample_labels), counts=summaries))
    (root / 'README.md').write_text('# Nucleus fates\n\n[Completed analysis](analyses/fates-v1/README.md)\n')
    return summaries


def submit(path):
    c = read(path)
    prepared = prepare(c)
    tech = resolve_path(c['output']) / 'technical'
    if (tech / 'launch.json').exists():
        raise ValueError('Already submitted; resume from the frozen bundle')
    repo = Path(__file__).resolve().parents[3]
    bundle = ExecutionBundle.freeze(repo, tech / 'code', c, directories=('src','configs','docs/metrics'))
    record = dict(identity=prepared['identity'], code=str(bundle.root), jobs={}, sources=len(prepared['sources']))
    queue = SlurmQueue(tech,bundle,'src.research.crystallization_origin.fates',
        dict(PCM_PROJECT_ROOT=str(repo), OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'),
        tech / 'launch.json', record, 'NUCLEUS-FATE')
    with queue.submission():
        worker = queue.submit('worker',[f'--array=0-{c["workers"]-1}','--cpus-per-task=1','--mem=6G','--time=04:00:00'])
        queue.submit('collect',['--cpus-per-task=1','--mem=6G','--time=00:30:00'],dependency='afterok:'+worker)
    return record


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['prepare','submit','source','worker','collect'])
    parser.add_argument('--config',required=True)
    parser.add_argument('--source-id',type=int)
    parser.add_argument('--index',type=int,default=int(os.environ.get('SLURM_ARRAY_TASK_ID','0')))
    args = parser.parse_args()
    c = read(args.config)
    if args.stage == 'submit':
        print(json.dumps(submit(args.config),indent=2)); return
    if args.stage == 'prepare':
        print(json.dumps(prepare(c),indent=2)); return
    with recorded_stage(resolve_path(c['output']) / f'technical/{args.stage}-{args.source_id or args.index}.json',
                        job=os.environ.get('SLURM_JOB_ID')) as progress:
        if args.stage == 'collect':
            collect(c); return
        _, p, bound = binding(c)
        prepared = read(resolve_path(c['output']) / 'technical/prepared.json')
        if digest(bound) != prepared['identity']:
            raise ValueError('Fate worker changed after preparation')
        selected = ([s for s in p['sources'] if s['id']==args.source_id] if args.stage=='source'
                    else p['sources'][args.index::c['workers']])
        if not selected:
            raise ValueError('No selected source')
        for number,item in enumerate(selected):
            progress.update(source=item['id'],completed=number,total=len(selected))
            result = source(c,p,item,prepared['identity'])
            progress.update(completed=number+1,last_counts=result['counts'])
            print(json.dumps(dict(source=item['id'],seconds=result['seconds'],counts=result['counts'])),flush=True)


if __name__ == '__main__':
    main()
