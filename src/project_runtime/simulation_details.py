"""Export simulation holdings, measured timelines and storage without loading coordinates."""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import csv
import json
import os
from pathlib import Path
import re

import numpy as np

from .dataset_registry import inspect_dataset, sha
from .paths import catalog, dataset_path, storage_path


def owned_files(root, children):
    files = []
    for directory, dirs, names in os.walk(root, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(d for d in dirs if not (parent/d).is_symlink()
                         and (parent/d).resolve() not in children)
        for name in sorted(names):
            p = parent/name
            if not p.is_symlink():
                files.append(p)
    return files


def closest_fact(path, records, field):
    """Only unique values in the nearest containing producer record are usable."""
    for parent in (path, *path.parents):
        candidates = []
        for record in records.get(parent, []):
            values = record.get('facts', {}).get(field, [])
            numbers = {v['value'] for v in values if isinstance(v['value'], (float, int, str))}
            if len(numbers) == 1:
                candidates.append((next(iter(numbers)), record['path']))
        if candidates:
            if len({c[0] for c in candidates}) != 1:
                return None, 'conflicting nearest producer values'
            return candidates[0]
    return None, ''


def timeline(path, records, steps):
    dt, evidence = closest_fact(path, records, 'timestep_ps')
    if dt is None:
        dt, evidence = closest_fact(path, records, 'timestep_fs')
        if dt is not None:
            dt = float(dt)/1000
    if dt is None or steps is None or len(steps) == 0:
        return dict(duration_ps=None, sampling_ps='', timestep_ps=dt, time_evidence=evidence)
    steps = np.asarray(steps)
    if steps.ndim != 1 or np.any(np.diff(steps) <= 0):
        raise ValueError(f'Invalid saved timeline: {path}')
    intervals = np.unique(np.round(np.diff(steps)*dt, 9))
    return dict(duration_ps=round(float((steps[-1]-steps[0])*dt), 9),
                sampling_ps=';'.join(f'{x:g}' for x in intervals), timestep_ps=dt,
                time_evidence=evidence)


def base_row(dataset, path, records):
    material, evidence = closest_fact(path, records, 'material')
    if material is None:
        material = ','.join(dataset['materials']) or 'unknown'
    return dict(dataset_id=dataset['id'], material=material,
                classification=dataset['classification'], path=str(path),
                material_evidence=evidence or 'dataset registry', signature='',
                notes='', state='saved')


def inspect_collection(args):
    identifier, entry, root, children = args
    # Reuse the registry's actual manifest/header verification, not directory labels.
    dataset, records = inspect_dataset(identifier, entry, root, children, {}, [])
    by_directory = defaultdict(list)
    for record in records:
        by_directory[Path(record['path']).parent].append(record)
    files = owned_files(root, children)
    filesizes = {str(p): (p.stat().st_size, p.stat().st_blocks*512) for p in files}
    directory_sizes = defaultdict(lambda: [0, 0])
    for filename, sizes in filesizes.items():
        total = directory_sizes[Path(filename).parent]
        total[0] += sizes[0]
        total[1] += sizes[1]
    rows, covered = [], set()
    for record in records:
        if record['kind'] != 'trajectory':
            continue
        t = record['trajectory']
        path = Path(record['path']).parent
        covered.add(path)
        arrays = t['arrays']
        steps = np.load(path/'timesteps.npy', allow_pickle=False) if 'timesteps' in arrays else None
        row = base_row(dataset, path, by_directory)
        row.update(atom_count=t['atom_count'], frames=t['frame_count'],
                   format='NPY '+arrays['positions']['dtype'] if 'positions' in arrays else 'incomplete NPY',
                   velocities='yes' if 'velocities' in arrays else 'no',
                   velocity_dtype=arrays.get('velocities', {}).get('dtype', ''),
                   state='complete binary' if t['usable_binary'] else 'incomplete binary',
                   signature=t['content_signature_from_manifest'],
                   allocated_bytes=directory_sizes[path][1],
                   apparent_bytes=directory_sizes[path][0],
                   **timeline(path, by_directory, steps))
        if t['frame_count'] == 1:
            row['notes'] = 'Single configuration, not a time series.'
        rows.append(row)
    for array in dataset['loose_arrays']:
        members = array.get('members', {})
        if 'positions_A' not in members or len(members['positions_A']['shape']) != 3:
            continue
        path = root/array['path']
        description = members['positions_A']
        frames, atoms, dimension = description['shape']
        if dimension != 3:
            raise ValueError(f'Unexpected trajectory vector width: {path}')
        with np.load(path, allow_pickle=False) as archive:
            steps = archive['step'] if 'step' in members else None
        row = base_row(dataset, path, by_directory)
        row.update(atom_count=atoms, frames=frames, format='NPZ '+description['dtype'],
                   velocities='yes' if 'velocities_A_per_ps' in members else 'no',
                   velocity_dtype=members.get('velocities_A_per_ps', {}).get('dtype', ''),
                   apparent_bytes=filesizes[str(path)][0], allocated_bytes=filesizes[str(path)][1],
                   **timeline(path, by_directory, steps))
        row['notes'] = 'Legacy export; may overlap a binary, checkpoint prefix or endpoint archive.'
        rows.append(row)
    for path in files:
        if path.suffix == '.traj' and 'tracking' not in path.parts and 'code' not in path.parts:
            from ase.io import Trajectory
            with Trajectory(path, 'r') as trajectory:
                n = len(trajectory)
                if not n:
                    continue
                first = trajectory[0]
                row = base_row(dataset, path, by_directory)
                row.update(atom_count=len(first), frames=n, format='ASE .traj',
                           velocities='yes' if first.has('momenta') else 'no',
                           velocity_dtype=str(first.arrays['momenta'].dtype) if first.has('momenta') else '',
                           duration_ps=0 if n == 1 else None, sampling_ps='', timestep_ps=None,
                           time_evidence='', apparent_bytes=filesizes[str(path)][0],
                           allocated_bytes=filesizes[str(path)][1],
                           notes='Snapshot/checkpoint; saved momenta allow velocities.' if n == 1 else 'ASE trajectory; timeline unestablished.')
                rows.append(row)
        if path.name in {'trajectory.lammpstrj', 'velocities.lammpstrj'} and not any(p in path.parts for p in ('tracking','code')):
            # Do not parse a growing multi-gigabyte text stream as a complete trajectory.
            with path.open() as stream:
                header = [stream.readline().strip() for _ in range(9)]
            if header[0] != 'ITEM: TIMESTEP' or header[2] != 'ITEM: NUMBER OF ATOMS' or not header[8].startswith('ITEM: ATOMS '):
                continue
            columns = header[8].split()[2:]
            sampling, evidence = closest_fact(path, by_directory, 'sample_interval_ps')
            # These are the actual input names emitted by the maintained producers.
            for filename in ('in.lammps', 'source.in.lammps'):
                source_input = path.parent/filename
                if not source_input.is_file():
                    continue
                text = source_input.read_text()
                dt = re.findall(r'^timestep\s+([0-9.eE+-]+)\s*$', text, re.M)
                stride = re.findall(r'^dump\s+\S+\s+all\s+custom\s+(\d+)\s+'
                                    + re.escape(path.name)+r'\s+', text, re.M)
                if len(dt) == len(stride) == 1:
                    actual = round(float(dt[0])*int(stride[0]), 9)
                    if sampling is not None and not np.isclose(actual, sampling):
                        raise ValueError(f'Input cadence disagrees with producer metadata: {source_input}')
                    sampling, evidence = actual, str(source_input)
            row = base_row(dataset, path, by_directory)
            row.update(atom_count=int(header[3]), frames=None,
                       format='LAMMPS velocity text' if path.name == 'velocities.lammpstrj' else 'LAMMPS text',
                       velocities='yes' if all(k in columns for k in ('vx','vy','vz')) else 'no',
                       velocity_dtype='text' if 'vx' in columns else '', duration_ps=None,
                       sampling_ps='' if sampling is None else f'{sampling:g}', timestep_ps=None,
                       time_evidence=evidence, apparent_bytes=filesizes[str(path)][0],
                       allocated_bytes=filesizes[str(path)][1],
                       state='text; completion unverified',
                       notes='Cadence from producer; full-frame count not rescanned. May duplicate a binary or be growing.')
            rows.append(row)
    for row in rows:
        p = Path(row['path'])
        parent = p.parent
        sidecar = parent/'velocities.lammpstrj'
        row['velocity_sidecar'] = str(sidecar) if str(sidecar) in filesizes else ''
        if row['velocities'] == 'no' and row['velocity_sidecar']:
            row['velocities'] = 'separate text'
            row['notes'] += ' Separate velocity dump exists; full time/ID matching not verified by this inventory.'
    return dataset, rows, filesizes


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def compact(values):
    values = set(v for v in values if v not in (None, ''))
    values = sorted(values) if all(isinstance(v, (int,float)) for v in values) else sorted(values,key=str)
    if not values:
        return 'unknown'
    if len(values) <= 6:
        return ', '.join(f'{v:g}' if isinstance(v,float) else str(v) for v in values)
    if all(isinstance(v, (int,float)) for v in values):
        return f'{min(values):g}–{max(values):g} (varies)'
    return 'mixed; see trajectory CSV'


def export_details(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries = catalog()
    registered_roots = {dataset_path(k).resolve() for k in entries}
    # Active runs are not necessarily registered until verified publication.
    for root in sorted(storage_path('simulation_runs').iterdir()):
        if root.is_symlink() or not root.is_dir() or not (root/'config.json').is_file():
            continue
        if root.resolve() in registered_roots:
            continue
        config = json.loads((root/'config.json').read_text())
        if config.get('protocol') == 'ta-position-branches':
            entries[root.name] = dict(root='simulation_runs', path=root.name, kind='simulation',
                metadata=dict(materials=['Ta'], classification='active_or_stopped',
                              description='Live Ta shooting directory; not yet published.'))
    roots = {key:(storage_path(entry['root'])/entry['path']).resolve() for key,entry in entries.items()}
    tasks = []
    unique_roots = set()
    for key, entry in sorted(entries.items()):
        if entry['kind'] != 'simulation' and entry.get('metadata',{}).get('role') != 'raw_dynamics' and not entry.get('simulation_fixture'):
            continue
        root = roots[key]
        if root in unique_roots:
            continue
        unique_roots.add(root)
        children = {p for k,p in roots.items() if p != root and p.is_relative_to(root)}
        tasks.append((key,entry,root,children))
    started = datetime.now(timezone.utc).isoformat()
    all_rows, collections, signatures = [], [], defaultdict(list)
    with ThreadPoolExecutor(max_workers=6) as pool:
        for dataset, rows, filesizes in pool.map(inspect_collection, tasks):
            print(f'INVENTORY {dataset["id"]}: {len(rows)} representations', flush=True)
            for row in rows:
                if row['signature']:
                    signatures[row['signature']].append(row['path'])
            measured = [r for r in rows if r['frames'] is not None and r['frames'] > 1 and r['state'] != 'incomplete binary']
            dynamic = [r for r in rows if (r['frames'] is None or r['frames'] > 1) and r['state'] != 'incomplete binary']
            formats = compact([r['format'] for r in dynamic]) if dynamic else 'no time series found'
            collections.append(dict(dataset_id=dataset['id'], material=compact([r['material'] for r in rows]) if rows else ','.join(dataset['materials']) or 'unknown',
                classification=dataset['classification'], time_series_exports=len(measured),
                snapshot_exports=sum(r['frames']==1 for r in rows),
                unverified_text_exports=sum(r['frames'] is None for r in rows),
                atom_counts=compact([r['atom_count'] for r in dynamic]),
                duration_ps=compact([r['duration_ps'] for r in measured]),
                sampling_ps=compact([r['sampling_ps'] for r in dynamic]),
                velocities=compact([r['velocities'] for r in dynamic]), formats=formats,
                allocated_GiB=round(sum(v[1] for v in filesizes.values())/2**30,6),
                apparent_GiB=round(sum(v[0] for v in filesizes.values())/2**30,6),
                allocated_bytes=sum(v[1] for v in filesizes.values()),
                location=dataset['location']['resolved'], available=dataset['location']['available'],
                notes=dataset['description'], issues='; '.join(dataset['issues'])))
            all_rows.extend(rows)
    for row in all_rows:
        row['identical_binary_paths'] = ';'.join(p for p in signatures.get(row['signature'],[]) if p != row['path'])
    csv_write(output/'collections.csv', collections)
    csv_write(output/'trajectories.csv', all_rows)
    ended = datetime.now(timezone.utc).isoformat()
    metadata = dict(started_at=started, finished_at=ended, registered_and_active_collections=len(collections),
                    trajectory_and_snapshot_representations=len(all_rows),
                    implementation_sha256=sha(Path(__file__)),
                    allocated_GiB=sum(c['allocated_bytes'] for c in collections)/2**30,
                    duplicate_binary_groups=sum(len(v)>1 for v in signatures.values()))
    (output/'inventory.json').write_text(json.dumps(metadata,indent=2)+'\n')
    lines = ['# Simulation holdings', '', f'Observed {started} to {ended}.', '',
        '[Collection CSV](collections.csv) · [Trajectory and snapshot CSV](trajectories.csv)', '',
        'Sizes are measured allocated disk bytes (GiB = 2^30 bytes), including native restarts, logs and provenance. '
        'Nested registered collections are charged to their own rows; directory aliases are not followed. '
        'Different physical copies remain included. Sizes of active runs change during observation.', '',
        'Lengths are last minus first saved time, not planned durations. Sampling is measured from binary/NPZ '
        'timelines and the nearest unambiguous producer timestep. Text cadence is declared; complete text '
        'frame counts are not rescanned. Unknown means the available evidence is insufficient.', '',
        'Counts are saved representations, not independent simulations: precision variants, checkpoint prefixes, '
        'duplicate exports and shared parent lineages must not be summed as independent runs. Velocities refer '
        'to sampled trajectories; restart files and isolated ASE snapshots can contain momenta separately. '
        'Large arrays are not rehashed; duplicate groups use producer-declared array hashes.', '',
        '| Collection | Material | Series exports | Atoms | Saved length (ps) | Sampling (ps) | Velocities | Format | Disk GiB | Classification |',
        '|---|---|---:|---:|---|---|---|---|---:|---|']
    for c in sorted(collections, key=lambda c:(c['material'],c['dataset_id'])):
        values=[c['dataset_id'],c['material'],str(c['time_series_exports']),c['atom_counts'],c['duration_ps'],
                c['sampling_ps'],c['velocities'],c['formats'],f'{c["allocated_GiB"]:.3f}',c['classification']]
        lines.append('| '+' | '.join(v.replace('|','/') for v in values)+' |')
    lines.extend(['',f'Total allocated space across these owned collection rows: **{metadata["allocated_GiB"]:.3f} GiB**.', '',
        'Scope: local registered simulation collections, fixtures and detected active Ta runs. Static archives and '
        'training/feature caches are excluded. Remote H200 copies are not locally verifiable and are not added. '
        'No historical datasets are deleted or reclassified by this inventory.', ''])
    (output/'README.md').write_text('\n'.join(lines))
    return metadata
