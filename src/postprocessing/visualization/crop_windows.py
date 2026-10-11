"""Extract narrower temporal windows from completed, verified local event movies."""
import argparse
import gzip
from pathlib import Path
import zipfile

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import resolve_path
from src.postprocessing.visualization.crops import validate
from src.postprocessing.protocols.shooting_data import read

FAMILY = 'ta_event_crop_windows'
ATOM_FIELDS = ('positions', 'atom_ids', 'ptm_type', 'tracked_center')
FRAME_FIELDS = ('time_ps', 'relative_time_ps', 'source_frame', 'timesteps',
                'center_original_A', 'original_box_low_A', 'original_box_high_A')


def movie_window(source, target, first, last, expected):
    with gzip.open(source, 'rt') as src, gzip.open(target, 'wt', compresslevel=3) as dst:
        frame = 0
        while line := src.readline():
            atoms = int(line)
            block = [line, src.readline(), *[src.readline() for _ in range(atoms)]]
            if not all(block): raise ValueError('Incomplete source XYZ frame')
            if first <= frame <= last: dst.writelines(block)
            frame += 1
        if frame != expected: raise ValueError('Source XYZ frame count changed')


def run(c, progress):
    source_root, root = resolve_path(c['source']), resolve_path(c['output'])
    original = read(source_root / 'technical/plan.json')
    completed = read(source_root / 'technical/complete.json')
    if (completed['identity'] != c['source_identity'] or original['identity'] != c['source_identity']
            or digest({k:v for k,v in original.items() if k != 'identity'}) != original['identity']):
        raise ValueError('Require the bound completed source crop collection')
    if (c['radius_A'] != original['config']['radius_A'] or c['before_ps'] > original['config']['before_ps']
            or c['after_ps'] > original['config']['after_ps']):
        raise ValueError('Cannot enlarge an existing spatial or temporal crop')
    events = [e for e in original['events'] if e['kind'] == c['kind']]
    if len(events) != c['expected_events']: raise ValueError('Selected event population changed')
    plan = dict(config=c, source_identity=c['source_identity'], events=events,
                implementation=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    plan['identity'] = digest(plan)
    path = root / 'technical/plan.json'
    if path.exists() and read(path) != plan: raise ValueError('Window export changed; choose a new revision')
    write_json(path, plan)
    rows = []
    for e in events:
        relative = Path('crops') / e['kind'] / (e['source']+'__'+e['name'])
        src, out = source_root / relative, root / relative
        receipt, m = read(src / 'complete.json'), read(src / 'metadata.json')
        expected = digest(dict(plan=original['identity'], event=e))
        if receipt['identity'] != expected or m['identity'] != expected:
            raise ValueError('Wrong source event identity')
        for name, h in receipt['files'].items():
            if sha(src / name) != h: raise ValueError(f'Changed source crop: {src/name}')
        identity = digest(dict(plan=plan['identity'], event=e, source_files=receipt['files']))
        out.mkdir(parents=True, exist_ok=True)
        progress.update(event=e['name'], source=e['source'])
        if (out / 'complete.json').exists():
            done = read(out / 'complete.json')
            if done['identity'] != identity or any(sha(out/n) != h for n,h in done['files'].items()):
                raise ValueError('Completed derived crop changed')
            metadata = read(out / 'metadata.json')
        else:
            with np.load(src / 'trajectory.npz') as a:
                if set(a.files) != {*ATOM_FIELDS, *FRAME_FIELDS, 'offsets'}:
                    raise ValueError('Unexpected source NPZ schema')
                ids = np.flatnonzero((a['relative_time_ps'] >= -c['before_ps']-1e-8) &
                                    (a['relative_time_ps'] <= c['after_ps']+1e-8))
                if not len(ids) or np.any(np.diff(ids) != 1): raise ValueError('Empty/noncontiguous time window')
                first, last = map(int, ids[[0,-1]])
                left, right = map(int, a['offsets'][[first,last+1]])
                arrays = {k:a[k][left:right].copy() for k in ATOM_FIELDS}
                arrays.update({k:a[k][ids].copy() for k in FRAME_FIELDS})
                arrays['offsets'] = a['offsets'][first:last+2]-left
            np.savez_compressed(out / 'trajectory.npz', **arrays)
            movie_window(src / 'trajectory.xyz.gz', out / 'trajectory.xyz.gz', first, last, m['frames'])
            start, end = map(float, arrays['relative_time_ps'][[0,-1]])
            metadata = dict(m, identity=identity, plan=plan['identity'], derived_from=str(src),
                source_crop_files=receipt['files'],
                requested_window_ps=[e['anchor_time_ps']-c['before_ps'],e['anchor_time_ps']+c['after_ps']],
                available_window_ps=arrays['time_ps'][[0,-1]].tolist(), frames=len(ids),
                available_before_ps=-start, available_after_ps=end,
                missing_before_ps=max(0,c['before_ps']+start), missing_after_ps=max(0,c['after_ps']-end),
                full_window=bool(-start >= c['before_ps']-1e-8 and end >= c['after_ps']-1e-8))
            write_json(out / 'metadata.json', metadata)
            validate(out, metadata)
            write_json(out / 'complete.json', dict(identity=identity, frames=len(ids),
                validated_xyz_frames=len(ids), files={n:sha(out/n) for n in
                    ('trajectory.npz','trajectory.xyz.gz','metadata.json')}))
        rows.append(dict(source=e['source'],parent=e['parent'],event=e['name'],status=e['status'],
            anchor_time_ps=e['anchor_time_ps'],radius_A=c['radius_A'],
            **{k:metadata[k] for k in ('frames','available_before_ps','available_after_ps',
                                     'missing_before_ps','missing_after_ps','full_window')},
            xyz=str(out/'trajectory.xyz.gz'),npz=str(out/'trajectory.npz')))
    write_metric_rows(rows, root/'analyses/coverage-v1', family=FAMILY, name='crop-index')
    lines = ['# Ta appearance candidate cutouts', '',
        f'{len(rows)} separately saved candidates, radius{c["radius_A"]:g} Angstrom, '
        f'requested{c["before_ps"]:g}ps before and{c["after_ps"]:g}ps after, native0.1ps spacing.',
        'These are the eight ambiguous establishments excluded by weak ancestry links, not confirmed independent nucleation events.',
        'Time zero marks the start of the confirmed64-atom establishment streak, not the first crystalline atom.',
        'Source branches last24ps. Only available frames are exported, with missing time recorded; no padding or stitched trajectories.',
        'The20A sphere follows the existing tracked reference. Atom IDs, changing membership, original boxes and partial full-cell PTM labels are preserved.',
        'Open trajectory.xyz.gz in OVITO; trajectory.npz contains centered coordinate arrays. PTM=-1 means unavailable between0.5ps audit observations.', '',
        '| Parent | Branch | Event | Before (ps) | After (ps) | Frames | Movie |',
        '|---|---|---|---:|---:|---:|---|']
    for e,r in zip(events,rows):
        movie = Path(r['xyz']).relative_to(root)
        lines.append(f'| {r["parent"]} | {e["source"].split("-")[-1]} | '
            f'{r["event"]} | {r["available_before_ps"]:g} | {r["available_after_ps"]:g} | {r["frames"]} | [XYZ]({movie}) |')
    lines += ['', '[Machine-readable index](analyses/coverage-v1/tables/crop-index.csv) · '
              '[Metric definitions](analyses/coverage-v1/tables/METRICS.md)', '']
    (root/'README.md').write_text('\n'.join(lines))
    archive = root / c['archive_name']
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as z:
        for f in sorted(root.rglob('*')):
            if f.is_file() and f != archive and not f.is_relative_to(root/'technical'):
                z.write(f,str(f.relative_to(root)))
    write_json(root/'technical/complete.json', dict(identity=plan['identity'], crops=len(rows),
        full_windows=sum(r['full_window'] for r in rows), frames=sum(r['frames'] for r in rows),
        archive=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size)))
    print(dict(crops=len(rows), frames=sum(r['frames'] for r in rows), output=str(root), archive=str(archive)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--config',required=True)
    c = read(parser.parse_args().config)
    with recorded_stage(resolve_path(c['output'])/'technical/export.json') as progress:
        run(c,progress)
