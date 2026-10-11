"""Select catalogue views using actual saved cadence, not their display name."""
import json
import hashlib
from pathlib import Path

import numpy as np

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest


def view_id(source, cadence):
    return digest(dict(input_identity=source['input_identity'], cadence=cadence,
                       frames=source['views'][cadence]))


def select_views(source, config, raw_cache):
    """Keep frozen source payloads intact; return published views and audit rows."""
    times = np.asarray(source['times_ps'], dtype=float)
    native = source['native_cadence_ps']
    if len(times) != len(source['frames']) or len(times) < 2 or not np.allclose(
            np.diff(times), native, rtol=0, atol=1e-8):
        raise ValueError(f'Invalid declared native timeline: {source["id"]}')
    selected, audit = [], []
    for cadence, dt in config['cadences'].items():
        if cadence not in source['views']:
            continue
        frames = np.asarray(source['views'][cadence], dtype=int)
        if len(frames) < 2 or not np.allclose(np.diff(times[frames]), dt, rtol=0, atol=1e-8):
            raise ValueError(f'Nonexact {dt} ps view: {source["id"]}')
        policy = config['sampling_policy'][cadence]
        reason, intervals = 'included', []
        if policy == 'native_saved':
            if abs(native-dt) > 1e-8:
                reason = 'excluded_finer_source_view'
            else:
                # Some shooting producers preselected a regular grid from an
                # irregular fine prefix. Inspect every referenced raw part.
                for part in source['parts']:
                    key = (part['path'], part['manifest_sha256'], part['timestep_fs'])
                    if key not in raw_cache:
                        path = Path(part['path'])
                        if sha(path/'manifest.json') != part['manifest_sha256']:
                            raise ValueError(f'Changed cadence source manifest: {path}')
                        manifest = json.loads((path/'manifest.json').read_text())
                        steps_path = path/manifest['arrays']['timesteps']['file']
                        steps = np.load(steps_path, mmap_mode='r', allow_pickle=False)
                        # Trajectory manifests hash array payloads, excluding
                        # the NPY header (the conversion producer's contract).
                        if hashlib.sha256(np.ascontiguousarray(steps).tobytes()).hexdigest() != part['array_sha256']['timesteps']:
                            raise ValueError(f'Changed raw timesteps: {steps_path}')
                        delta = np.diff(steps)*part['timestep_fs']/1000
                        if len(delta) == 0 or np.any(delta <= 0):
                            raise ValueError(f'Invalid raw timeline: {steps_path}')
                        raw_cache[key] = delta
                    delta = raw_cache[key]
                    intervals.extend(np.unique(np.round(delta, 8)).tolist())
                    if not np.allclose(delta, dt, rtol=0, atol=1e-8):
                        reason = 'excluded_mixed_or_finer_raw_parts'
                if not np.array_equal(frames, np.arange(len(times))):
                    reason = 'excluded_subsampled_frame_map'
        elif policy != 'exact_saved':
            raise ValueError(f'Unknown sampling policy for {cadence}: {policy}')
        if reason == 'included':
            selected.append(cadence)
        audit.append(dict(source_id=source['input_identity'], source=source['id'],
            dataset=source['dataset'], cadence=cadence, cadence_ps=dt,
            declared_source_cadence_ps=native, sampling_policy=policy,
            selected=reason == 'included', reason=reason,
            raw_intervals_ps=json.dumps(sorted(set(intervals)))))
    return selected, audit


def retain_views(db, identifiers):
    """Reconcile a derived index after an explicit population-policy change."""
    db.execute('CREATE TEMP TABLE selected_views(view_id TEXT PRIMARY KEY)')
    db.executemany('INSERT INTO selected_views VALUES (?)', ((v,) for v in identifiers))
    removed = db.execute('''SELECT count(*) FROM views
        WHERE view_id NOT IN (SELECT view_id FROM selected_views)''').fetchone()[0]
    db.execute('''DELETE FROM events WHERE task_id IN (SELECT task_id FROM tasks
        WHERE view_id NOT IN (SELECT view_id FROM selected_views))''')
    db.execute('DELETE FROM tasks WHERE view_id NOT IN (SELECT view_id FROM selected_views)')
    db.execute('DELETE FROM views WHERE view_id NOT IN (SELECT view_id FROM selected_views)')
    db.execute('DELETE FROM sources WHERE source_id NOT IN (SELECT source_id FROM views)')
    db.execute('DROP TABLE selected_views')
    return removed
