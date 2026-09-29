"""Full-cell ancestry on external materials, with a declared physical-time grid."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.forecast_crystallization.local_metrics import first_sustained_onset
from . import ancestry
from .audit import csv_table
from . import external_data as data


def contract(plan):
    return dict(plan=plan['identity'], implementation={p.name: sha(p) for p in
                (Path(__file__), Path(ancestry.__file__), Path(data.__file__))})


def seal_chunks(plan, source):
    folder = data.source_folder(plan, source['id'])
    files = {}
    for task in plan['tasks']:
        if task['source'] != source['id']:
            continue
        name = f'ptm-{task["start"]:04d}-{task["stop"]:04d}'
        entry = json.loads((folder / (name + '.json')).read_text())
        path = folder / (name + '.npz')
        if entry['identity'] != data.chunk_identity(plan, task) or entry['sha256'] != sha(path):
            raise ValueError(f'Changed external PTM chunk: {path}')
        files[path.name] = entry['sha256']
    write_json(folder / 'ptm-complete.json', dict(plan_identity=plan['identity'], source=source['id'], files=files))


class FrameAccess:
    def __init__(self, plan, source, graph=None):
        self.plan, self.source, self.graph = plan, source, graph
        self.folder = data.source_folder(plan, source['id'])
        self.loaded_start, self.values, self.last_frame, self.last_value = -1, None, -1, None
        part = source['parts'][0]
        ids = data.arrays(part['path'], part['manifest_sha256'])['atom_ids']
        self.rows = np.searchsorted(ids, source['center_atom_ids'])
        np.testing.assert_array_equal(ids[self.rows], source['center_atom_ids'])

    def labels(self, frame):
        count = self.plan['config']['chunk_frames']
        start = frame // count * count
        if start != self.loaded_start:
            stop = min(self.source['frame_count'], start + count)
            with np.load(self.folder / f'ptm-{start:04d}-{stop:04d}.npz') as arrays:
                self.values = arrays['labels']
            self.loaded_start = start
        return self.values[frame - start]

    def frame(self, frame):
        if frame != self.last_frame:
            points, box = data.geometry(self.source, frame)
            local, _ = ancestry.components(points, box, self.labels(frame), self.source['lineage'])
            dense = np.where(local > 0, local + self.graph['start'][frame], 0)
            self.last_frame, self.last_value = frame, (points, box, dense)
        return self.last_value


def graph_for_source(plan, source, identity):
    folder = data.source_folder(plan, source['id'])
    path, receipt = folder / 'graph.npz', folder / 'graph-complete.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved['identity'] != identity or saved['sha256'] != sha(path):
            raise ValueError(f'Changed graph: {source["id"]}')
        with np.load(path) as a:
            return {k: a[k] for k in a.files}
    access = FrameAccess(plan, source)
    frames, sizes, starts, edges, center_nodes, center_labels, crystal_counts = [-1], [0], [], [], [], [], []
    previous = []
    for frame in range(source['frame_count']):
        points, box = data.geometry(source, frame)
        labels = access.labels(frame)
        local, local_sizes = ancestry.components(points, box, labels, source['lineage'])
        shift = len(sizes) - 1
        starts.append(shift)
        matched = set()
        for lag, old in enumerate(reversed(previous), 1):
            parent, child, counts, strong = ancestry.overlap(old, local, sizes, local_sizes, source['lineage'])
            if lag > 1:
                keep = np.array([int(c) not in matched for c in child], bool)
                parent, child, counts, strong = (x[keep] for x in (parent, child, counts, strong))
            matched.update(child[strong].tolist())
            edges.extend(zip(parent.tolist(), (child + shift).tolist(), counts.tolist(), strong.astype(int).tolist(), [lag] * len(parent)))
        dense = np.where(local > 0, local + shift, 0).astype(np.int32)
        center_nodes.append(dense[access.rows])
        center_labels.append(labels[access.rows].copy())
        crystal_counts.append(int(np.isin(labels, [1, 2, 3]).sum()))
        sizes.extend(local_sizes[1:].tolist())
        frames.extend([frame] * (len(local_sizes) - 1))
        previous = (previous + [dense])[-(source['lineage']['maximum_missing_frames'] + 1):]
        if frame % 8 == 0 or frame == source['frame_count'] - 1:
            write_json(folder / 'graph-progress.json', dict(frames=frame + 1, total_frames=source['frame_count'], nodes=len(sizes) - 1))
    graph = dict(frame=np.asarray(frames, np.int32), size=np.asarray(sizes, np.int32), start=np.asarray(starts, np.int32),
                 edges=np.asarray(edges, np.int64).reshape(-1, 5), center_node=np.stack(center_nodes).T,
                 center_labels=np.stack(center_labels).T, crystalline_atoms=np.asarray(crystal_counts))
    temporary = path.with_suffix('.building.npz')
    np.savez_compressed(temporary, **graph)
    temporary.replace(path)
    write_json(receipt, dict(identity=identity, sha256=sha(path)))
    return graph


def locate(source, graph, events, roots, access):
    by_frame = defaultdict(list)
    baseline = np.asarray(source['baseline_center_indices'])
    for event in events:
        by_frame[event['birth_frame']].append(event)
    for frame, births in sorted(by_frame.items()):
        points, box, dense = access.frame(frame)
        nodes = np.flatnonzero(graph['frame'] == frame)
        for event in births:
            cluster = points[dense == event['birth_node']]
            if len(cluster) != event['birth_size']:
                raise ValueError('Changed reconstructed birth component')
            delta = cluster - cluster[0]
            delta -= box * np.rint(delta / box)
            center = np.mod(cluster[0] + delta.mean(0), box)
            extent = bool(np.any(np.ptp(delta, axis=0) > box / 2))
            offset = points[access.rows] - center
            offset -= box * np.rint(offset / box)
            distances = np.linalg.norm(offset, axis=-1)
            old_nodes = [n for n in nodes if any(r != event['event_id'] and events[r - 1]['confirmation_frame'] <= frame for r in roots[n])]
            nearest = None
            if old_nodes:
                from scipy.spatial import cKDTree
                mask = np.isin(dense, old_nodes)
                nearest = float(cKDTree(points[mask], boxsize=box).query(cluster, workers=1)[0].min())
            if event['left_censored']:
                kind = 'left_censored_existing'
            elif event['establishment_merge_ambiguous']:
                kind = 'unresolved_establishment_merge'
            elif extent:
                kind = 'unresolved_periodic_extent'
            elif nearest is not None and nearest <= source['lineage']['interface_distance_A']:
                kind = 'interface_associated'
            else:
                kind = 'isolated_establishment'
            covered = np.flatnonzero(distances <= source['lineage']['local_birth_radius_A'])
            event.update(kind=kind, centroid_A=center.tolist(), center_distances_A=distances,
                         covered_center_indices=covered.tolist(), covered_centers=int(len(covered)),
                         covered_baseline64=int(np.isin(covered, baseline).sum()), nearest_established_crystal_A=nearest,
                         periodic_extent_ambiguous=extent, birth_time_ps=float(source['times_ps'][frame]),
                         confirmation_time_ps=float(source['times_ps'][event['confirmation_frame']]))
    return events


def analyze(plan_path, sid):
    plan = json.loads(Path(plan_path).read_text())
    source = next(s for s in plan['sources'] if s['id'] == sid)
    folder = data.source_folder(plan, sid)
    identity = digest(contract(plan))
    with (folder / 'audit.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'audit.json').exists():
            old = json.loads((folder / 'audit.json').read_text())
            if old['identity'] != identity or any(sha(folder / k) != v for k, v in old['files'].items()):
                raise ValueError(f'Changed external result: {sid}')
            return old
        started = time.monotonic()
        seal_chunks(plan, source)
        graph = graph_for_source(plan, source, identity)
        access = FrameAccess(plan, source, graph)
        config = plan['config']
        cadence = source['cadence_ps']
        persistence = round(config['onset_persistence_ps'] / cadence) + 1
        history = round(config['negative_history_ps'] / cadence) + 1
        crystal = np.isin(graph['center_labels'], [1, 2, 3])
        # The deliberately stitched Al melt is preparation context. Its initial
        # solid must not make every atom permanently ineligible for predicting
        # recrystallization in the subsequent measurement trajectory.
        forecast_start = (next(i for i, pair in enumerate(source['frames']) if pair[0] == 1) - 1
                          if source['phase_boundary'] is not None else 0)
        onsets = first_sustained_onset(crystal[:, forecast_start:], persistence) + forecast_start
        max_steps = round(max(config['horizons_ps']) / cadence)
        origins = np.arange(max(history - 1, forecast_start), source['frame_count'] - max_steps - persistence + 1)
        recent = np.lib.stride_tricks.sliding_window_view(crystal, history, axis=1).any(-1)
        risk = (origins[None, :] < onsets[:, None]) & ~recent[:, origins - history + 1]
        positive6 = risk & ((onsets[:, None] - origins[None, :]) <= max_steps)
        center, origin_index = np.nonzero(positive6)
        origin = origins[origin_index]
        delay = (onsets[center] - origin) * cadence
        baseline = np.zeros(len(onsets), bool)
        baseline[source['baseline_center_indices']] = True
        selectors = dict(sampled1pct=np.ones(len(onsets), bool), baseline64=baseline)
        count_rows, catalog, codes_out, regional_out = [], [], [], {}
        boundaries = np.r_[0, np.flatnonzero(np.diff(center)) + 1, len(center)]
        for threshold in source['lineage']['thresholds']:
            events, roots, uncertain = ancestry.establish(graph, threshold)
            locate(source, graph, events, roots, access)
            base_codes = np.empty(len(center), np.int8)
            onset_cache = {}
            for begin, end in zip(boundaries[:-1], boundaries[1:]):
                if begin == end:
                    continue
                ci = int(center[begin])
                onset = int(onsets[ci])
                key = tuple(graph['center_node'][ci, onset:onset + persistence])
                if key not in onset_cache:
                    rids, reason = ancestry.onset_roots(graph, roots, uncertain, onset, ci, persistence)
                    last_confirmation = max((events[r - 1]['confirmation_frame'] for r in rids), default=source['frame_count'])
                    onset_cache[key] = rids, reason, last_confirmation
                rids, reason, last_confirmation = onset_cache[key]
                for row in range(begin, end):
                    if reason is None and origin[row] >= last_confirmation:
                        label = 'existing_crystal_arrival'
                    else:
                        label = ancestry.classify(int(origin[row]), onset, ci, rids, reason, events, source['lineage']['local_birth_radius_A'])
                    base_codes[row] = ancestry.LABELS.index(label)
            horizon_codes = []
            for hi, horizon in enumerate(config['horizons_ps']):
                codes = np.where(delay <= horizon, base_codes, 0).astype(np.int8)
                horizon_codes.append(codes)
                regional = np.zeros_like(risk)
                for event in events:
                    if event['kind'] != 'isolated_establishment' or not event['covered_center_indices']:
                        continue
                    covered = np.asarray(event['covered_center_indices'])
                    d = (event['birth_frame'] - origins) * cadence
                    columns = np.flatnonzero((d > 0) & (d <= horizon))
                    regional[np.ix_(covered, columns)] = risk[np.ix_(covered, columns)]
                rc, ro = np.nonzero(regional)
                regional_out[f'{threshold["name"]}_{hi}_center'] = rc.astype(np.int32)
                regional_out[f'{threshold["name"]}_{hi}_origin'] = origins[ro].astype(np.int32)
                for track, select_centers in selectors.items():
                    selected_rows = select_centers[center]
                    counts = np.bincount(codes[selected_rows], minlength=len(ancestry.LABELS))
                    total = int(risk[select_centers].sum())
                    counts[0] = total - int(counts[1:].sum())
                    for code, label in enumerate(ancestry.LABELS):
                        count_rows.append(dict(threshold=threshold['name'], track=track, unit='onset_windows', horizon_ps=horizon,
                                               label=label, count=int(counts[code])))
                        if code:
                            count_rows.append(dict(threshold=threshold['name'], track=track, unit='distinct_onsets_represented', horizon_ps=horizon,
                                label=label, count=int(len(np.unique(center[selected_rows & (codes == code)])))))
                    count_rows.append(dict(threshold=threshold['name'], track=track, unit='regional_birth_windows', horizon_ps=horizon,
                        label='local_nucleus_establishment', count=int(regional[select_centers].sum())))
            codes_out.append(np.stack(horizon_codes, axis=-1))
            for kind in ('isolated_establishment', 'interface_associated', 'left_censored_existing', 'unresolved_periodic_extent', 'unresolved_establishment_merge'):
                subset = [e for e in events if e['kind'] == kind]
                count_rows.append(dict(threshold=threshold['name'], track='full_cell', unit='distinct_clusters', horizon_ps=0., label=kind, count=len(subset)))
                for track, field in (('sampled1pct', 'covered_centers'), ('baseline64', 'covered_baseline64')):
                    count_rows.append(dict(threshold=threshold['name'], track=track, unit='distinct_clusters_covered', horizon_ps=0., label=kind,
                                           count=sum(e[field] > 0 for e in subset)))
            catalog.extend(dict(threshold=threshold['name'], phase='measurement' if e['birth_frame'] >= forecast_start else 'preparation',
                                **{k: v for k, v in e.items() if k != 'center_distances_A'}) for e in events)
        np.savez_compressed(folder / 'positive_windows.npz', center_index=center.astype(np.int32), origin=origin.astype(np.int32),
                            label_code=np.stack(codes_out, axis=1), center_atom_ids=np.asarray(source['center_atom_ids']),
                            onset_frame=onsets, all_origins=origins, **regional_out)
        write_json(folder / 'events.json', catalog)
        result = dict(identity=identity, source=sid, material=source['material'], potential=source['potential'],
                      ancestry_group=source['ancestry_group'], frame_count=source['frame_count'], centers=len(onsets),
                      forecast_start_frame=forecast_start, forecast_start_time_ps=source['times_ps'][forecast_start],
                      windows=int(risk.sum()), first_onsets=int(((onsets > forecast_start) & (onsets < source['frame_count'])).sum()),
                      left_censored_centers=int((onsets == forecast_start).sum()), count_rows=count_rows, seconds=time.monotonic() - started,
                      files={n: sha(folder / n) for n in ('positive_windows.npz', 'events.json')})
        write_json(folder / 'audit.json', result)
        return result


def report(plan):
    root = result_folders(resolve_path(plan['config']['output']))
    results, per_source, events = [], [], []
    expected = digest(contract(plan))
    for source in plan['sources']:
        folder = data.source_folder(plan, source['id'])
        path = folder / 'audit.json'
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        if result['identity'] != expected:
            raise ValueError(f'Changed external result identity: {path}')
        results.append(result)
        per_source.extend(dict(material=source['material'], potential=source['potential'], source=source['id'],
                               ancestry_group=source['ancestry_group'], **row) for row in result['count_rows'])
        events.extend(dict(source=source['id'], material=source['material'], **e) for e in json.loads((folder / 'events.json').read_text()))
    fields = ['material', 'potential', 'threshold', 'track', 'unit', 'horizon_ps', 'label']
    totals, branches, groups = defaultdict(int), defaultdict(set), defaultdict(set)
    for row in per_source:
        key = tuple(row[k] for k in fields)
        totals[key] += row['count']
        if row['count']:
            branches[key].add(row['source'])
            groups[key].add(row['ancestry_group'])
    aggregate = [dict(zip(fields, key), count=value, branches_with_label=len(branches[key]),
                      ancestry_groups_with_label=len(groups[key])) for key, value in sorted(totals.items())]
    snapshot_metric_docs(root, 'crystallization_origin_external')
    csv_table(root / 'tables/label-counts.csv', aggregate, fields + ['count', 'branches_with_label', 'ancestry_groups_with_label'])
    csv_table(root / 'tables/source-counts.csv', per_source, ['source', 'ancestry_group'] + fields + ['count'])
    csv_table(root / 'tables/establishments.csv', events, ['source', 'material', 'phase', 'threshold', 'event_id', 'birth_time_ps',
              'confirmation_time_ps', 'birth_size', 'kind', 'covered_centers', 'covered_baseline64', 'peak_single_lineage_size', 'ever_merged'])
    summary = dict(completed=len(results), total=len(plan['sources']), sources=[r['source'] for r in results],
                   identity=expected, windows=sum(r['windows'] for r in results))
    write_json(root / 'technical/summary.json', summary)
    lines = ['# External crystallization-origin audit', '', f'Completed **{len(results)}/{len(plan["sources"])} trajectory records**. '
             'Counts remain partial until all records finish. Branches sharing preparation ancestry are not independent replicates.', '',
             'Full periodic cells at 0.5 ps; reference establishment is 64 atoms over 1.5 ps (four observations). '
             'These are candidate establishments, not established physical critical nuclei. Length cutoffs use the existing fixed material normalization.', '',
             '| Material / potential | Completed records | Candidate isolated births | Births covered by 1% centers | Births covered by 64 centers |',
             '| --- | ---: | ---: | ---: | ---: |']
    for material, potential in sorted({(s['material'], s['potential']) for s in plan['sources']}):
        def count(track, unit):
            return totals.get((material, potential, 'primary', track, unit, 0., 'isolated_establishment'), 0)
        completed = sum(r['material'] == material and r['potential'] == potential for r in results)
        lines.append(f'| {material} / {potential} | {completed} | {count("full_cell", "distinct_clusters")} | '
                     f'{count("sampled1pct", "distinct_clusters_covered")} | {count("baseline64", "distinct_clusters_covered")} |')
    lines.extend(['', '[All 3/6 ps origin-label counts](tables/label-counts.csv) · [Per-source counts](tables/source-counts.csv) · '
                  '[Establishment catalogue](tables/establishments.csv) · [Definitions](tables/METRICS.md)', '',
                  'The million-atom Al melt and measurement are one continuous lineage, with their duplicate boundary omitted. '
                  'This exploratory population has no new train/test split and is not interchangeable with fixed Al64 benchmark windows. '
                  'Static-only Zr data cannot supply temporal origin labels.', ''])
    (root / 'RESULTS.md').write_text('\n'.join(lines))
    return summary


def watch(plan_path, workers):
    plan = json.loads(Path(plan_path).read_text())
    root = result_folders(resolve_path(plan['config']['output']))
    value = contract(plan)
    path = root / 'technical/audit-contract.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Changed external audit implementation')
    write_json(path, value)
    tasks = defaultdict(list)
    for task in plan['tasks']:
        tasks[task['source']].append(task)
    pending, submitted = {}, set()
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            while len(submitted) < len(plan['sources']) or pending:
                for source in plan['sources']:
                    sid = source['id']
                    if sid in submitted:
                        continue
                    folder = data.source_folder(plan, sid)
                    if all((folder / f'ptm-{t["start"]:04d}-{t["stop"]:04d}.json').exists() for t in tasks[sid]):
                        submitted.add(sid)
                        pending[pool.submit(analyze, str(plan_path), sid)] = sid
                if pending:
                    done, _ = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        print(json.dumps(dict(source=result['source'], seconds=result['seconds'], windows=result['windows'])), flush=True)
                        del pending[future]
                else:
                    time.sleep(15)
                failed = [p.name for p in (root / 'technical').glob('lane-*.json') if json.loads(p.read_text())['state'] == 'failed']
                if failed:
                    raise RuntimeError(f'PTM lanes failed: {failed}')
                summary = report(plan)
                write_json(root / 'technical/audit-state.json', dict(state='running', **summary))
        write_json(root / 'technical/audit-state.json', dict(state='complete', **report(plan)))
    except BaseException:
        write_json(root / 'technical/audit-state.json', dict(state='failed', traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    plan_path = resolve_path(config['output']) / 'technical/plan.json'
    if args.report_only:
        report(json.loads(plan_path.read_text()))
    else:
        watch(plan_path, config['analysis_workers'])


if __name__ == '__main__':
    main()
