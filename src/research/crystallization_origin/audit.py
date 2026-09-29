"""Count cluster births and ancestry-resolved Al64 prediction labels without fitting."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import fcntl
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.forecast_crystallization.local_metrics import first_sustained_onset
from . import ancestry


def contract(config, plan):
    return dict(config=config, release_identity=plan['identity'],
                producer_sha256=sha(Path(__file__)), ancestry_sha256=sha(Path(ancestry.__file__)),
                labels=list(ancestry.LABELS), track='all64 and exact legacy16 subset',
                units='distinct cluster establishments, tracked-atom first onsets, and fixed prediction windows')


def analyze_source(config, plan, item, identity):
    if digest(contract(config, plan)) != identity:
        raise ValueError('Audit implementation changed during execution')
    folder = resolve_path(config['output']) / 'technical/sources' / str(item['id'])
    with (folder / 'audit.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = folder / 'audit.json'
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved['identity'] != identity:
                raise ValueError(f'Changed audit contract for {item["id"]}')
            for name, checksum in saved['files'].items():
                if sha(folder / name) != checksum:
                    raise ValueError(f'Changed audit artifact: {folder / name}')
            return saved
        started = time.monotonic()
        root = resolve_path(config['release'])
        benchmark = json.loads((root / 'benchmark/manifest.json').read_text())
        source_record = next(s for s in benchmark['sources'] if s['id'] == item['id'])
        labels_path = root / 'benchmark/sources' / str(item['id']) / 'labels.npy'
        if sha(labels_path) != source_record['files']['labels.npy']:
            raise ValueError(f'Changed fixed center labels: {item["id"]}')
        if sha(root / 'benchmark/population.npz') != benchmark['population_sha256']:
            raise ValueError('Changed fixed prediction population')
        with np.load(root / 'benchmark/population.npz') as data:
            mask = data['source'] == item['id']
            population = {k: data[k][mask] for k in data.files}
        graph, graph_receipt = ancestry.build_graph(config, item, folder)
        geometry = ancestry.GeometryAccess(config, item, folder, graph)
        labels = np.load(labels_path)
        persistence = plan['config']['onset_persistence_frames']
        onsets = first_sustained_onset(np.isin(labels, [1, 2, 3]), persistence)
        centers = np.searchsorted(item['center_atom_ids'], population['atom'])
        np.testing.assert_array_equal(onsets[centers], population['onset_frame'])
        cadence = plan['config']['cadence_ps']
        np.testing.assert_allclose((onsets[centers] - population['frame']) * cadence, population['delay'], rtol=0, atol=0)
        extraction = json.loads((folder / 'ptm-complete.json').read_text())
        mismatch_onset = {m['atom'] for m in extraction['center_mismatches']
                          if onsets[item['center_atom_ids'].index(m['atom'])] <= m['frame'] <
                          onsets[item['center_atom_ids'].index(m['atom'])] + persistence}
        count_rows, all_events, center_records = [], [], []
        output_codes, output_births = [], []
        tracks = dict(all64=np.ones(len(centers), bool), legacy16=population['legacy_row'] >= 0)
        for threshold in config['lineage']['thresholds']:
            events, roots, uncertain = ancestry.establish(graph, threshold)
            events = ancestry.locate_births(config, item, graph, events, roots, geometry)
            onset_info = []
            for ci, (atom, onset) in enumerate(zip(item['center_atom_ids'], onsets)):
                rids, reason = ancestry.onset_roots(graph, roots, uncertain, int(onset), ci, persistence)
                if atom in mismatch_onset:
                    reason = 'unresolved_ptm_disagreement'
                onset_info.append((rids, reason))
                center_records.append(dict(threshold=threshold['name'], atom=atom, onset_frame=int(onset),
                                           ancestors=sorted(rids), unresolved_reason=reason))
            for e in events:
                all_events.append(dict(threshold=threshold['name'], **e))
            codes = np.zeros((len(centers), len(config['horizons_ps'])), np.int8)
            regional = np.zeros_like(codes, dtype=bool)
            for h, horizon in enumerate(config['horizons_ps']):
                positive = (population['delay'] > 0) & (population['delay'] <= horizon)
                for row in np.flatnonzero(positive):
                    ci = centers[row]
                    rids, reason = onset_info[ci]
                    label = ancestry.classify(int(population['frame'][row]), int(onsets[ci]), int(ci),
                        rids, reason, events, config['lineage']['local_birth_radius_A'])
                    codes[row, h] = ancestry.LABELS.index(label)
                for event in events:
                    if event['kind'] != 'isolated_establishment':
                        continue
                    delay = (event['birth_frame'] - population['frame']) * cadence
                    local = np.asarray(event['center_distances_A'])[centers] <= config['lineage']['local_birth_radius_A']
                    regional[:, h] |= local & (delay > 0) & (delay <= horizon)
                for track, selection in tracks.items():
                    for code, label in enumerate(ancestry.LABELS):
                        selected = selection & (codes[:, h] == code)
                        count_rows.append(dict(threshold=threshold['name'], track=track, unit='onset_windows',
                                               horizon_ps=horizon, label=label, count=int(selected.sum())))
                        if code:
                            count_rows.append(dict(threshold=threshold['name'], track=track, unit='distinct_onsets_represented',
                                horizon_ps=horizon, label=label, count=int(len(np.unique(population['atom'][selected])))))
                    count_rows.append(dict(threshold=threshold['name'], track=track, unit='regional_birth_windows',
                        horizon_ps=horizon, label='local_nucleus_establishment', count=int((selection & regional[:, h]).sum())))
            output_codes.append(codes)
            output_births.append(regional)
            for kind in ('isolated_establishment', 'interface_associated', 'left_censored_existing',
                         'unresolved_periodic_extent', 'unresolved_establishment_merge'):
                subset = [e for e in events if e['kind'] == kind]
                count_rows.append(dict(threshold=threshold['name'], track='full_cell', unit='distinct_clusters',
                                       horizon_ps=0., label=kind, count=len(subset)))
                for track in ('all64', 'legacy16'):
                    count_rows.append(dict(threshold=threshold['name'], track=track, unit='distinct_clusters_covered',
                        horizon_ps=0., label=kind, count=sum(e['covered_centers_' + track] > 0 for e in subset)))
        arrays = folder / 'window_labels.npz'
        np.savez_compressed(arrays, sample_id=population['sample_id'], frame=population['frame'],
                            atom=population['atom'], onset_frame=population['onset_frame'],
                            label_code=np.stack(output_codes, axis=1), regional_birth=np.stack(output_births, axis=1))
        write_json(folder / 'events.json', all_events)
        write_json(folder / 'center-onsets.json', center_records)
        result = dict(identity=identity, source=item['id'], role=item['role'], source_lineage=item['lineage'],
            release_identity=plan['identity'], graph_identity=graph_receipt['identity'], count_rows=count_rows,
            windows=len(centers), center_onsets=int(((onsets > 0) & (onsets < labels.shape[1])).sum()),
            centers_already_crystalline=int((onsets == 0).sum()), centers_right_censored=int((onsets == labels.shape[1]).sum()),
            ptm_center_mismatches=len(extraction['center_mismatches']), ptm_disagreements_at_onset=len(mismatch_onset),
            files={p: sha(folder / p) for p in ('window_labels.npz', 'events.json', 'center-onsets.json')},
            seconds=time.monotonic() - started)
        write_json(receipt, result)
        return result


def csv_table(path, rows, fields):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def report(config, plan, identity):
    root = result_folders(resolve_path(config['output']))
    results, event_rows = [], []
    for item in plan['sources']:
        folder = root / 'technical/sources' / str(item['id'])
        path = folder / 'audit.json'
        if path.exists():
            result = json.loads(path.read_text())
            if result['identity'] != identity:
                raise ValueError(f'Changed audit results: {path}')
            results.append(result)
            event_rows.extend(dict(source=item['id'], role=item['role'], **e) for e in json.loads((folder / 'events.json').read_text()))
    fields = ['threshold', 'track', 'unit', 'horizon_ps', 'label']
    counters, source_sets = defaultdict(int), defaultdict(set)
    source_rows = []
    for result in results:
        for row in result['count_rows']:
            source_rows.append(dict(source=result['source'], role=result['role'], **row))
            for role in (result['role'], 'all_completed'):
                key = (role, *(row[f] for f in fields))
                counters[key] += row['count']
                if row['count']:
                    source_sets[key].add(result['source'])
    aggregated = [dict(zip(['role', *fields], key), count=value, sources_with_label=len(source_sets[key]))
                  for key, value in sorted(counters.items())]
    snapshot_metric_docs(root, 'crystallization_origin')
    csv_table(root / 'tables/label-counts.csv', aggregated, ['role', *fields, 'count', 'sources_with_label'])
    csv_table(root / 'tables/source-counts.csv', source_rows, ['source', 'role', *fields, 'count'])
    csv_table(root / 'tables/cluster-establishments.csv', event_rows,
              ['source', 'role', 'threshold', 'event_id', 'birth_frame', 'confirmation_frame', 'birth_size', 'kind',
               'covered_centers_all64', 'covered_centers_legacy16', 'nearest_established_crystal_A', 'periodic_extent_ambiguous',
               'peak_single_lineage_size', 'last_single_lineage_frame', 'ever_merged', 'establishment_merge_ambiguous'])
    summary = dict(completed_sources=len(results), total_sources=len(plan['sources']),
                   source_ids=[r['source'] for r in results], pilot_complete=set(config['pilot_sources']).issubset(r['source'] for r in results),
                   windows=sum(r['windows'] for r in results), center_onsets=sum(r['center_onsets'] for r in results),
                   ptm_center_mismatches=sum(r['ptm_center_mismatches'] for r in results),
                   ptm_disagreements_at_onset=sum(r['ptm_disagreements_at_onset'] for r in results),
                   release_identity=plan['identity'], identity=identity)
    write_json(root / 'technical/summary.json', summary)
    lines = ['# Crystallization-origin audit', '',
             f'Completed **{len(results)}/{len(plan["sources"])} sources**; {summary["windows"]:,} fixed prediction windows. '
             'These are partial availability counts until all sources finish.', '',
             'The labels describe operational PTM cluster establishment and ancestry, not physical critical nuclei. '
             'Reference: 64 connected crystalline atoms for three frames (1.5 ps between first and third observations), '
             '3.6 Å periodic connectivity; births within 8 Å of a tracked center are local.', '',
             '## Distinct clusters in the full cells', '',
             '| Criterion | Isolated establishments | Near existing crystal | Already present at start | Unresolved establishment | Covered isolated births (64 centers) |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    def count(threshold, track, unit, horizon, label):
        return counters.get(('all_completed', threshold, track, unit, horizon, label), 0)
    for th in config['lineage']['thresholds']:
        name = th['name']
        values = [count(name, 'full_cell', 'distinct_clusters', 0., k) for k in
                  ('isolated_establishment', 'interface_associated', 'left_censored_existing')]
        values.append(sum(count(name, 'full_cell', 'distinct_clusters', 0., k) for k in
                          ('unresolved_periodic_extent', 'unresolved_establishment_merge')))
        values.append(count(name, 'all64', 'distinct_clusters_covered', 0., 'isolated_establishment'))
        lines.append(f'| {th["size"]} atoms, {th["persistence_frames"]} frames | ' + ' | '.join(map(str, values)) + ' |')
    lines.extend(['', '## Existing local-onset windows, primary criterion', '',
                  '| Label | 3 ps positive windows | 6 ps positive windows |', '| --- | ---: | ---: |'])
    for label in ancestry.LABELS[1:]:
        values = [count('primary', 'all64', 'onset_windows', h, label) for h in config['horizons_ps']]
        lines.append('| ' + label.replace('_', ' ') + ' | ' + ' | '.join(map(str, values)) + ' |')
    lines.extend(['', '## Regional birth target', '',
                  'A birth within the local 8 Å region is a different target from the center atom becoming crystalline. '
                  'It can occur without that atom transforming within the horizon.', ''])
    for h in config['horizons_ps']:
        value = count('primary', 'all64', 'regional_birth_windows', h, 'local_nucleus_establishment')
        lines.append(f'- {h:g} ps: {value} positive fixed windows for an isolated local establishment.')
    lines.extend(['', f'Original/full-cell PTM disagreements: {summary["ptm_center_mismatches"]} center-frames; '
                  f'{summary["ptm_disagreements_at_onset"]} tracked onset events touch a disagreement and are marked unresolved.', '',
                  'Counts are split by source role, all64/legacy16, threshold and horizon in [label-counts.csv](tables/label-counts.csv). '
                  'Distinct onsets represented are deduplicated by source/atom within each label; the same onset may have different '
                  'origin-relative labels across forecast origins. Do not sum those distinct counts across labels.', '',
                  '[Definitions and limits](tables/METRICS.md) · [Per-source counts](tables/source-counts.csv) · '
                  '[Establishment catalogue](tables/cluster-establishments.csv)', ''])
    (root / 'RESULTS.md').write_text('\n'.join(lines))
    return summary


def run(config, watch, workers):
    _, plan = read_release(config['release'])
    root = result_folders(resolve_path(config['output']))
    value = contract(config, plan)
    identity = digest(value)
    path = root / 'technical/audit-contract.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Refusing to overwrite an audit using different labeling rules')
    write_json(path, value)
    pending, submitted = {}, set()
    order = {sid: i for i, sid in enumerate(config['pilot_sources'])}
    items = sorted(plan['sources'], key=lambda s: (order.get(s['id'], len(order)), s['id']))
    # Freeze rules before examining held-out counts. Pilot sources run first.
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            while True:
                pilot_done = all((root / 'technical/sources' / str(sid) / 'audit.json').exists() for sid in config['pilot_sources'])
                for item in items:
                    folder = root / 'technical/sources' / str(item['id'])
                    if item['id'] in submitted or not (folder / 'ptm-complete.json').exists():
                        continue
                    if not pilot_done and item['id'] not in config['pilot_sources']:
                        continue
                    submitted.add(item['id'])
                    pending[pool.submit(analyze_source, config, plan, item, identity)] = item['id']
                if pending:
                    done, _ = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                    for future in done:
                        result = future.result()
                        print(json.dumps(dict(source=result['source'], seconds=result['seconds'], windows=result['windows'])), flush=True)
                        del pending[future]
                    if done:
                        summary = report(config, plan, identity)
                        write_json(root / 'technical/audit-state.json', dict(state='running', **summary))
                else:
                    if not watch or len(submitted) == len(items):
                        break
                    extraction = root / 'technical/extraction-state.json'
                    if extraction.exists() and json.loads(extraction.read_text())['state'] == 'failed':
                        raise RuntimeError('Full-cell PTM extraction failed; inspect extraction-state.json')
                    time.sleep(20)
                if not watch and not pending:
                    # In non-watching mode, allow a just-completed pilot to unlock
                    # already extracted non-pilot sources before finishing.
                    if any((root / 'technical/sources' / str(s['id']) / 'ptm-complete.json').exists()
                           and s['id'] not in submitted for s in items) and all(
                           (root / 'technical/sources' / str(sid) / 'audit.json').exists() for sid in config['pilot_sources']):
                        continue
                    break
        summary = report(config, plan, identity)
        state = 'complete' if summary['completed_sources'] == summary['total_sources'] else 'partial'
        write_json(root / 'technical/audit-state.json', dict(state=state, **summary))
    except Exception:
        write_json(root / 'technical/audit-state.json', dict(state='failed', traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--watch', action='store_true', help='Analyze sources as detached PTM extraction finishes')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if args.report_only:
        _, plan = read_release(config['release'])
        report(config, plan, digest(contract(config, plan)))
    else:
        run(config, args.watch, args.workers)


if __name__ == '__main__':
    main()
