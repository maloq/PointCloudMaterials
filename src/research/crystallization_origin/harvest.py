"""Training-only regional-emergence availability and precursor-quality audit.

References and sampling receipts are exploratory analysis artifacts, not a
released training population. Existing Al64 and its labels remain unchanged.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.spatial import cKDTree

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from src.research.geoframe_evolution.reference import bond_descriptors, ORDER_NAMES
from . import ancestry, extract
from .audit import csv_table


CAUSES = ('none_within_horizon', 'isolated_birth_first',
          'confirmed_lineage_contact_first', 'other_establishment_first',
          'simultaneous_or_ambiguous')


def load_plan(config):
    heldout=config['protocol']=='nucleus-harvest-heldout-audit-v2'
    if config['role'] != 'train' and not heldout:
        raise ValueError('This first harvest audit is restricted to training sources')
    if config['role'] not in ('train','selection','calibration','test'):raise ValueError('Unknown immutable source role')
    if heldout:
        seal=json.loads(resolve_path(config['training_definition']).read_text())
        if sha(resolve_path(config['training_definition']))!=config['training_definition_sha256']:
            raise ValueError('Training-only birth definition changed')
        for key in ('horizons_ps','lead_frames','history_ps','reference_history_ps','control_origins_per_source','control_centers_per_origin'):
            if seal['config'][key]!=config[key]:raise ValueError(f'Held-out audit changed training definition: {key}')
    audit = json.loads(resolve_path(config['audit_config']).read_text())
    if heldout and audit!=seal['audit_config']:raise ValueError('Held-out audit changed the frozen ancestry/event definition')
    _, release = read_release(audit['release'])
    root = resolve_path(audit['output'])
    state = json.loads((root / 'technical/audit-state.json').read_text())
    old = json.loads((root / 'technical/audit-contract.json').read_text())
    if state['state'] != 'complete' or old['ancestry_sha256'] != sha(Path(ancestry.__file__)):
        raise ValueError('Require complete audit and its exact ancestry implementation')
    sources = [s for s in release['sources'] if s['role'] == config['role']]
    if len(sources) != release['config']['expected_source_counts'][config['role']]:
        raise ValueError('Training source contract changed')
    files = [Path(__file__), Path(ancestry.__file__), Path(extract.__file__),
             Path(bond_descriptors.__code__.co_filename)]
    plan = dict(config=config, audit_config=audit, release_identity=release['identity'],
                audit_identity=state['identity'], cadence_ps=release['config']['cadence_ps'],
                sources=[{k: v for k, v in s.items() if k != 'cells'} for s in sources],
                implementation={str(p.relative_to(Path(__file__).resolve().parents[3])): sha(p) for p in files})
    plan['identity'] = digest(plan)
    return plan


def confirmed_times(graph, roots, events):
    """Earliest *observed confirmation*, never backfilled birth time, per node."""
    sentinel = graph['center_node'].shape[1] + 1
    return np.asarray([min((events[r - 1]['confirmation_frame'] for r in rs),
                           default=sentinel) for rs in roots], np.int32)


def prefix_check(graph, threshold, confirmation, frames):
    """Record causal invariance on actual source prefixes used by this analysis."""
    checked = []
    for frame in sorted(set(frames)):
        keep = graph['frame'] <= frame
        n = int(keep.sum())
        prefix = dict(frame=graph['frame'][:n], size=graph['size'][:n],
                      edges=graph['edges'][graph['edges'][:, 1] < n],
                      center_node=graph['center_node'][:, :frame + 1])
        events, roots, _ = ancestry.establish(prefix, threshold)
        actual = confirmed_times(prefix, roots, events) <= frame
        current = graph['frame'][:n] == frame
        if not np.array_equal(actual[current], (confirmation[:n] <= frame)[current]):
            raise ValueError(f'Causal-prefix disagreement: threshold={threshold}, frame={frame}')
        checked.append(frame)
    return checked


class Frames(ancestry.GeometryAccess):
    def __init__(self, config, item, folder, graph):
        super().__init__(config, item, folder, graph)
        self.seal = json.loads((folder / 'ptm-complete.json').read_text())
        self.verified = set()

    def frame(self, frame):
        count = self.config['ptm']['chunk_frames']
        start = frame // count * count
        stop = min(self.item['frame_count'], start + count)
        name = f'ptm-{start:04d}-{stop:04d}.npz'
        if name not in self.verified:
            if sha(self.folder / name) != self.seal['files'][name]:
                raise ValueError(f'Changed PTM input: {self.folder / name}')
            self.verified.add(name)
        points, box, dense = super().frame(frame)
        observed = np.bincount(dense, minlength=len(self.graph['size']))
        current = np.flatnonzero(self.graph['frame'] == frame)
        if not np.array_equal(observed[current], self.graph['size'][current]):
            raise ValueError(f'Reconstructed components changed: {self.item["id"]}/{frame}')
        return points, box, dense

    def ptm(self, frame):
        self.frame(frame)
        return self.labels[frame - self.chunk_start]


def near_rows(points, box, center, radius):
    delta = points - center
    delta -= box * np.rint(delta / box)
    return np.flatnonzero(np.einsum('ij,ij->i', delta, delta) <= radius ** 2)


def make_rows(events, event_atoms, controls, leads, atom_count):
    keys, associations = [], []
    for event in events:
        if event['kind'] != 'isolated_establishment':
            continue
        for lead in leads:
            origin = event['birth_frame'] - lead
            if origin < 0:
                continue
            values = origin * atom_count + event_atoms[event['event_id']]
            keys.extend(values.tolist())
            associations.extend([event['event_id']] * len(values))
    keys = np.asarray(keys, np.int64)
    all_keys, inverse = np.unique(np.r_[keys, controls], return_inverse=True)
    memberships = defaultdict(set)
    for event, row in zip(associations, inverse[:len(keys)]):
        memberships[event].add(int(row))
    return dict(atom_row=(all_keys % atom_count).astype(np.int32),
                origin=(all_keys // atom_count).astype(np.int32),
                event_memberships={e: np.asarray(sorted(rows), np.int32) for e, rows in memberships.items()},
                control=np.isin(all_keys, controls),
                candidate=np.isin(all_keys, keys))


def ordered_features(points, box, rows):
    tree = cKDTree(points, boxsize=box)
    values, counts = [], []
    for start in range(0, len(rows), 128):
        anchors = rows[start:start + 128]
        _, core = tree.query(points[anchors], k=13, workers=1)
        _, neighbors = tree.query(points[core], k=13, workers=1)
        if not np.array_equal(core[:, 0], anchors) or not np.array_equal(neighbors[:, :, 0], core):
            raise ValueError('Duplicate coordinates or inconsistent periodic nearest neighbors')
        vectors = points[neighbors[:, :, 1:]] - points[core][:, :, None]
        vectors -= box * np.rint(vectors / box)
        v, c = bond_descriptors(vectors)
        values.append(v)
        counts.append(c)
    return np.concatenate(values), np.concatenate(counts)


def review_movie_data(plan, item, events, access, folder):
    if item['id'] not in plan['config']['review_sources']:
        return []
    selected = next((e for e in events if e['kind'] == 'isolated_establishment'), None)
    if selected is None:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cfg = plan['config']
    frames = [selected['birth_frame'] + x for x in cfg['review_offsets_frames']]
    frames = [f for f in frames if 0 <= f < item['frame_count']]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    names = []
    for ax, frame in zip(axes.ravel(), frames):
        points, box, dense = access.frame(frame)
        delta = points - selected['centroid_A']
        delta -= box * np.rint(delta / box)
        mask = np.linalg.norm(delta, axis=1) <= cfg['review_radius_A']
        path = folder / f'review-{frame:04d}.npz'
        np.savez_compressed(path, displacement_A=delta[mask].astype(np.float32),
                            atom_ids=access.raw.atom_ids[mask], ptm=access.ptm(frame)[mask],
                            component_size=access.graph['size'][dense[mask]])
        names.append(path.name)
        slab = mask & (np.abs(delta[:, 2]) <= 5.)
        solid = np.isin(access.ptm(frame)[slab], [1, 2, 3])
        ax.scatter(delta[slab, 0], delta[slab, 1], c=np.where(solid, '#cc3d45', '#a8b9ce'), s=7)
        ax.add_patch(plt.Circle((0, 0), 8, fill=False, color='black', lw=.8))
        ax.set(xlim=(-24, 24), ylim=(-24, 24), aspect='equal',
               title=f'{(frame-selected["birth_frame"])*plan["cadence_ps"]:+g} ps', xlabel='x (Å)', ylabel='y (Å)')
    for ax in axes.ravel()[len(frames):]:
        ax.set_visible(False)
    fig.suptitle(f'Source {item["id"]}, event {selected["event_id"]}: PTM crystal red, other blue\n'
                 'Future-birth-centered review crop; never a predictor input. Central |z| ≤ 5 Å slab.')
    plot = resolve_path(cfg['output']) / 'plots' / f'source-{item["id"]}-birth-review.png'
    fig.savefig(plot, dpi=160)
    plt.close(fig)
    return names


def analyze(plan, item):
    if item['role'] != plan['config']['role']:
        raise ValueError('Source role differs from sealed audit')
    cfg, old_cfg = plan['config'], plan['audit_config']
    root = resolve_path(cfg['output'])
    folder = root / 'technical/sources' / str(item['id'])
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'source.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = folder / 'complete.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['identity'] != plan['identity'] or any(sha(folder / f) != h for f, h in saved['files'].items()):
                raise ValueError(f'Changed harvest artifact: source {item["id"]}')
            return saved
        started = time.monotonic()
        old_folder = resolve_path(old_cfg['output']) / 'technical/sources' / str(item['id'])
        old = json.loads((old_folder / 'audit.json').read_text())
        if old['identity'] != plan['audit_identity'] or sha(old_folder / 'events.json') != old['files']['events.json']:
            raise ValueError('Changed source audit catalogue')
        receipt = json.loads((old_folder / 'graph-complete.json').read_text())
        if sha(old_folder / 'graph.npz') != receipt['sha256']:
            raise ValueError('Changed source ancestry graph')
        with np.load(old_folder / 'graph.npz') as a:
            graph = {k: a[k] for k in a.files}
        catalogue = json.loads((old_folder / 'events.json').read_text())
        access = Frames(old_cfg, item, old_folder, graph)
        cadence = plan['cadence_ps']
        future = int(round(max(cfg['horizons_ps']) / cadence))
        padding = max(t['persistence_frames'] - 1 for t in old_cfg['lineage']['thresholds'])
        final_origin = item['frame_count'] - 1 - future - padding
        rng = np.random.default_rng(int(digest([cfg['seed'], item['id'], 'uniform_controls'])[:16], 16))
        control_origins = np.sort(rng.choice(final_origin + 1, cfg['control_origins_per_source'], replace=False))
        controls = np.concatenate([f * item['atom_count'] + rng.choice(item['atom_count'], cfg['control_centers_per_origin'], replace=False)
                                   for f in control_origins])
        control_probability = (len(control_origins) / (final_origin + 1)
                               * cfg['control_centers_per_origin'] / item['atom_count'])
        thresholds, schedule = {}, defaultdict(dict)
        radius = old_cfg['lineage']['local_birth_radius_A']
        for th in old_cfg['lineage']['thresholds']:
            name = th['name']
            events = [e for e in catalogue if e['threshold'] == name]
            fresh, roots, _ = ancestry.establish(graph, th)
            if [(e['birth_node'], e['confirmation_frame']) for e in events] != [(e['birth_node'], e['confirmation_frame']) for e in fresh]:
                raise ValueError(f'Changed event reconstruction: {item["id"]}/{name}')
            confirmed = confirmed_times(graph, roots, events)
            first = next((e for e in events if not e['left_censored']), None)
            check_frames = [0, int(control_origins[-1])]
            if first is not None:
                check_frames.extend([max(0, first['birth_frame'] - 1), first['confirmation_frame']])
            checked = prefix_check(graph, th, confirmed, check_frames)
            atoms = {}
            for event in events:
                points, box, _ = access.frame(event['birth_frame'])
                atoms[event['event_id']] = near_rows(points, box, event['centroid_A'], radius)
            rows = make_rows(events, atoms, controls, cfg['lead_frames'], item['atom_count'])
            n = len(rows['origin'])
            rows.update(contact=np.zeros((n, future + 1), bool),
                        near_large=np.zeros((n, future + 1), bool),
                        current_component=np.zeros(n, np.int32), current_ptm=np.zeros(n, np.uint8),
                        neighborhood_max_size=np.zeros(n, np.int32), neighborhood_crystal_fraction=np.zeros(n, np.float32))
            if 'support_radii_A' in cfg:
                for support in cfg['support_radii_A']:
                    rows[f'established_clear_{support:g}A']=np.ones(n,bool)
                    rows[f'ptm_all_clear_{support:g}A']=np.ones(n,bool)
            thresholds[name] = dict(threshold=th, events=events, atoms=atoms, confirmed=confirmed, rows=rows, prefix_frames=checked)
            for lag in range(future + 1):
                frames = rows['origin'] + lag
                for frame in np.unique(frames):
                    if frame >= item['frame_count']:
                        continue
                    schedule[int(frame)].setdefault(name, []).append((lag, np.flatnonzero(frames == frame)))
        for fi, frame in enumerate(sorted(schedule)):
            points, box, dense = access.frame(frame)
            ptm = access.ptm(frame)
            tree = cKDTree(points, boxsize=box)
            for name, tasks in schedule[frame].items():
                group = thresholds[name]
                rows = group['rows']
                confirmed_mask = group['confirmed'][dense] <= frame
                large_mask = graph['size'][dense] >= group['threshold']['size']
                trees = [cKDTree(points[m], boxsize=box) if m.any() else None for m in (confirmed_mask, large_mask)]
                if 'support_radii_A' in cfg:
                    masks=(confirmed_mask|large_mask,np.isin(ptm,[1,2,3]))
                    support_trees=[cKDTree(points[m],boxsize=box) if m.any() else None for m in masks]
                for lag, ids in tasks:
                    anchors = rows['atom_row'][ids]
                    for out, t in zip(('contact', 'near_large'), trees):
                        if t is not None:
                            rows[out][ids, lag] = t.query(points[anchors], workers=1)[0] <= radius
                    if lag == 0:
                        if 'support_radii_A' in cfg:
                            for label,t in zip(('established','ptm_all'),support_trees):
                                distance=np.full(len(anchors),np.inf) if t is None else t.query(points[anchors],workers=1)[0]
                                for support in cfg['support_radii_A']:
                                    rows[f'{label}_clear_{support:g}A'][ids]=distance>support
                        rows['current_component'][ids] = graph['size'][dense[anchors]]
                        rows['current_ptm'][ids] = ptm[anchors]
                        neighbors = tree.query_ball_point(points[anchors], radius, workers=1)
                        rows['neighborhood_max_size'][ids] = [graph['size'][dense[v]].max() for v in neighbors]
                        rows['neighborhood_crystal_fraction'][ids] = [np.isin(ptm[v], [1, 2, 3]).mean() for v in neighbors]
            if fi % 16 == 0 or fi == len(schedule) - 1:
                write_json(folder / 'progress.json', dict(state='geometry', completed_frames=fi + 1,
                           requested_frames=len(schedule), seconds=time.monotonic() - started))
        counts, event_rows, artifacts = [], [], []
        for name, group in thresholds.items():
            rows, events = group['rows'], group['events']
            n = len(rows['origin'])
            sentinel = item['frame_count'] + future + 1
            birth_time = np.full(n, sentinel, np.int32)
            birth_event = np.zeros(n, np.int32)
            birth_kind = np.zeros(n, np.int8)
            for event in sorted(events, key=lambda e: (e['birth_frame'], e['event_id'])):
                if event['left_censored']:
                    continue
                delay = event['birth_frame'] - rows['origin']
                mask = (delay > 0) & (delay <= future) & np.isin(rows['atom_row'], group['atoms'][event['event_id']])
                tied = mask & (birth_time == event['birth_frame'])
                earlier = mask & (birth_time > event['birth_frame'])
                birth_time[earlier] = event['birth_frame']
                birth_event[earlier] = event['event_id']
                birth_kind[earlier] = 1 if event['kind'] == 'isolated_establishment' else (4 if event['kind'].startswith('unresolved') else 3)
                birth_kind[tied] = 4
            contact_any = rows['contact'][:, 1:].any(1)
            contact_time = np.where(contact_any, rows['origin'] + 1 + rows['contact'][:, 1:].argmax(1), sentinel)
            cause = np.where(birth_time < contact_time, birth_kind, np.where(contact_time < birth_time, 2, 4)).astype(np.int8)
            first_time = np.minimum(birth_time, contact_time)
            cause[first_time == sentinel] = 0
            complete = rows['origin'] <= final_origin
            eligible = complete & ~rows['contact'][:, 0] & ~rows['near_large'][:, 0]
            codes = np.stack([np.where((first_time - rows['origin']) * cadence <= h, cause, 0) for h in cfg['horizons_ps']], 1)
            # Eligibility is separate; never label an ineligible/censored row as a negative.
            codes[~eligible] = -1
            rows.update(eligible=eligible, complete_followup=complete, first_event_frame=first_time,
                        first_birth_event=birth_event, label_code=codes,
                        observed_history_ps=(rows['origin'] * cadence).astype(np.float32))
            selected = np.zeros(n, bool)
            inclusion = np.zeros(n, np.float64)
            for event in events:
                if event['kind'] != 'isolated_establishment':
                    continue
                members = rows['event_memberships'].get(event['event_id'], np.empty(0, np.int32))
                owner = eligible[members] & (cause[members] == 1) & (birth_event[members] == event['event_id'])
                reference = owner & (rows['observed_history_ps'][members] >= cfg['reference_history_ps'])
                center_pool = np.unique(rows['atom_row'][members[reference]])
                number = min(cfg['centers_per_event'], len(center_pool))
                erng = np.random.default_rng(int(digest([cfg['seed'], item['id'], name, event['event_id']])[:16], 16))
                chosen = erng.choice(center_pool, number, replace=False) if number else center_pool
                accepted = members[reference & np.isin(rows['atom_row'][members], chosen)]
                selected[accepted] = True
                inclusion[accepted] = number / len(center_pool) if number else 0.
                rec = dict(source=item['id'], threshold=name, event_id=event['event_id'], birth_frame=event['birth_frame'],
                           candidate_centers=len(group['atoms'][event['event_id']]), enumerated_rows=len(members),
                           complete_rows=int(complete[members].sum()), eligible_rows=int(eligible[members].sum()),
                           existing_contact_exclusions=int(rows['contact'][members, 0].sum()),
                           size_crossed_exclusions=int(rows['near_large'][members, 0].sum()),
                           earlier_competing_rows=int((eligible[members] & ~owner).sum()),
                           peak_single_lineage_size=event['peak_single_lineage_size'], ever_merged=event['ever_merged'],
                           selected_centers=number, selected_rows=len(accepted))
                for h in cfg['history_ps']:
                    use = owner & (rows['observed_history_ps'][members] >= h)
                    for horizon in cfg['horizons_ps']:
                        keep = use & ((event['birth_frame'] - rows['origin'][members]) * cadence <= horizon)
                        rec[f'first_birth_rows_H{h:g}_tau{horizon:g}'] = int(keep.sum())
                event_rows.append(rec)
            rows['selected_example'] = selected
            rows['example_center_inclusion_probability'] = inclusion
            for history in cfg['history_ps']:
                history_mask = rows['observed_history_ps'] >= history
                for track, membership in [('candidate', rows['candidate']), ('uniform_control', rows['control']), ('selected_example', selected)]:
                    valid = membership & history_mask & eligible
                    for hi, horizon in enumerate(cfg['horizons_ps']):
                        for code, label in enumerate(CAUSES):
                            chosen = valid & (codes[:, hi] == code)
                            counts.append(dict(source=item['id'], threshold=name, history_ps=history, horizon_ps=horizon,
                                track=track, label=label, count=int(chosen.sum()),
                                unique_births=int(len(np.unique(birth_event[chosen]))) if code == 1 else 0))
            arrays = {k: v for k, v in rows.items() if k != 'event_memberships'}
            arrays['atom_ids'] = access.raw.atom_ids[rows['atom_row']]
            arrays['uniform_control_inclusion_probability'] = np.asarray(control_probability)
            name_npz = f'{name}-references.npz'
            np.savez_compressed(folder / name_npz, **arrays)
            artifacts.append(name_npz)
        primary = thresholds['primary']['rows']
        diagnostic_controls = np.flatnonzero(primary['control'] & primary['eligible'])
        drng = np.random.default_rng(int(digest([cfg['seed'], item['id'], 'bond_order'])[:16], 16))
        diagnostic_controls = drng.choice(diagnostic_controls, min(len(diagnostic_controls), cfg['order_controls_per_source']), replace=False)
        diagnostic = np.union1d(np.flatnonzero(primary['selected_example']), diagnostic_controls)
        order = np.empty((len(diagnostic), len(ORDER_NAMES)), np.float32)
        connections = np.empty((len(diagnostic), 3), np.int8)
        for frame in np.unique(primary['origin'][diagnostic]):
            mask = primary['origin'][diagnostic] == frame
            points, box, _ = access.frame(int(frame))
            order[mask], connections[mask] = ordered_features(points, box, primary['atom_row'][diagnostic[mask]])
        np.savez_compressed(folder / 'order-diagnostics.npz', reference_row=diagnostic,
                            order=order, coherent_bonds=connections)
        artifacts.append('order-diagnostics.npz')
        artifacts.extend(review_movie_data(plan, item, thresholds['primary']['events'], access, folder))
        write_json(folder / 'event-availability.json', event_rows)
        artifacts.append('event-availability.json')
        result = dict(identity=plan['identity'], source=item['id'], role=item['role'], ancestry=item['lineage'],
            counts=counts, events=event_rows,
            causal_prefix_frames={name: g['prefix_frames'] for name, g in thresholds.items()},
            controls=dict(origins=control_origins.tolist(), rows=len(controls), inclusion_probability=control_probability,
                          population_origins=final_origin + 1, population_atoms=item['atom_count']),
            primary_candidates=sum(e['kind'] == 'isolated_establishment' for e in thresholds['primary']['events']),
            primary_eligible_controls=int((primary['control'] & primary['eligible']).sum()),
            primary_selected_rows=int(primary['selected_example'].sum()),
            primary_selected_crystalline_centers=int((primary['selected_example'] & np.isin(primary['current_ptm'], [1, 2, 3])).sum()),
            seconds=time.monotonic() - started, files={f: sha(folder / f) for f in artifacts})
        write_json(path, result)
        write_json(folder / 'progress.json', dict(state='complete', seconds=result['seconds']))
        return result


def report(plan):
    root = result_folders(resolve_path(plan['config']['output']))
    results = []
    for source in plan['sources']:
        p = root / 'technical/sources' / str(source['id']) / 'complete.json'
        if p.exists():
            r = json.loads(p.read_text())
            if r['identity'] != plan['identity']:
                raise ValueError(f'Changed harvest receipt: {p}')
            results.append(r)
    fields = ['threshold', 'history_ps', 'horizon_ps', 'track', 'label']
    totals, events = defaultdict(lambda: [0, 0, 0]), []
    for result in results:
        events.extend(result['events'])
        for row in result['counts']:
            key = tuple(row[f] for f in fields)
            value = totals[key]
            value[0] += row['count']
            value[1] += row['unique_births']
            value[2] += row['count'] > 0
    rows = [dict(zip(fields, key), count=v[0], distinct_source_births=v[1], sources_with_rows=v[2]) for key, v in sorted(totals.items())]
    snapshot_metric_docs(root, 'nucleus_harvest')
    csv_table(root / 'tables/availability.csv', rows, fields + ['count', 'distinct_source_births', 'sources_with_rows'])
    if events:
        csv_table(root / 'tables/events.csv', events, list(events[0]))
    summary = dict(identity=plan['identity'], completed=len(results), total=len(plan['sources']),
        candidates=sum(r['primary_candidates'] for r in results),
        selected_rows=sum(r['primary_selected_rows'] for r in results),
        selected_ptm_crystalline_centers=sum(r['primary_selected_crystalline_centers'] for r in results),
        control_rows=sum(r['controls']['rows'] for r in results),
        eligible_controls=sum(r['primary_eligible_controls'] for r in results),
        source_ids=[r['source'] for r in results])
    write_json(root / 'technical/summary.json', summary)
    lines = [f'# {plan["config"]["role"].title()}-source nucleus harvest audit', '',
             f'Completed **{len(results)}/{len(plan["sources"])} {plan["config"]["role"]} sources**; '
             f'**{summary["candidates"]} primary candidate births** processed.', '',
             'No model fitting or new simulations. Source roles remain fixed. References are exploratory; '
             'this is not a released probability-training dataset. Counts from pending sources are not zeros.', '',
             '| Required history | 3 ps candidate windows | Distinct births | 6 ps candidate windows | Distinct births |',
             '| --- | ---: | ---: | ---: | ---: |']
    for history in plan['config']['history_ps']:
        numbers = []
        for h in plan['config']['horizons_ps']:
            value = totals[('primary', history, h, 'candidate', 'isolated_birth_first')]
            numbers.extend(value[:2])
        lines.append(f'| {history:g} ps | ' + ' | '.join(map(str, numbers)) + ' |')
    lines.extend(['', f'Sampled reference examples at 12 ps history: **{summary["selected_rows"]}**. '
                  f'Of these, **{summary["selected_ptm_crystalline_centers"]}** have a PTM-crystalline central atom '
                  'and would fail an instantaneous center-liquid restriction.', '',
                  f'Uniform control regions sampled: {summary["control_rows"]}; primary causal-risk eligible: {summary["eligible_controls"]}. '
                  'The control draw is outcome-blind; future contact and birth labels remain explicit.', '',
                  'Eligibility excludes any component already at the size threshold within 8 Å and any '
                  'previously confirmed lineage in that region. First future confirmed-lineage contact is a '
                  'conservative competing-event screen; simultaneous events remain ambiguous. Size-qualified '
                  'transients are retained in reference diagnostics, not automatically called nuclei.', '',
                  '[All criteria, histories and outcomes](tables/availability.csv) · [Per-event availability](tables/events.csv) · '
                  '[Exact definitions](tables/METRICS.md)', '',
                  'Bond-order diagnostics and review crops are audit-only. Review plots use future birth locations '
                  'for visualization and must never be passed to a predictor. All model-example references remain '
                  'ordinary atom-centered observations at their own forecast origins.', ''])
    (root / 'RESULTS.md').write_text('\n'.join(lines))
    return summary


def run(config, workers, source_ids):
    root = result_folders(resolve_path(config['output']))
    plan = load_plan(config)
    plan_path = root / 'technical/plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Changed harvest protocol; use a new run directory')
    write_json(plan_path, plan)
    sources = [s for s in plan['sources'] if not source_ids or s['id'] in source_ids]
    if source_ids and {s['id'] for s in sources} != set(source_ids):
        raise ValueError('Requested sources must belong to the declared fixed role')
    # Verify selected source availability and manifests before dispatching workers.
    for source in sources:
        extract.raw_source(source)
    order = {s: i for i, s in enumerate(config['review_sources'])}
    sources.sort(key=lambda s: (order.get(s['id'], len(order)), s['id']))
    try:
        write_json(root / 'technical/state.json', dict(state='running', **report(plan)))
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = {pool.submit(analyze, plan, source): source['id'] for source in sources}
            for future in as_completed(pending):
                result = future.result()
                print(json.dumps(dict(source=result['source'], seconds=result['seconds'],
                                      candidates=result['primary_candidates'], selected=result['primary_selected_rows'])), flush=True)
                write_json(root / 'technical/state.json', dict(state='running', **report(plan)))
        summary = report(plan)
        write_json(root / 'technical/state.json', dict(state='complete' if summary['completed'] == summary['total'] else 'partial', **summary))
    except BaseException:
        write_json(root / 'technical/state.json', dict(state='failed', traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=24)
    parser.add_argument('--sources', type=int, nargs='*')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if args.report_only:
        report(load_plan(config))
    else:
        run(config, args.workers, args.sources)


if __name__ == '__main__':
    main()
