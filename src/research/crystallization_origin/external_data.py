"""Explicit external-trajectory protocol and independently resumable PTM chunks."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import fcntl
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.structural_pretraining.prepare import source_arrays
from src.experiment_runner.artifacts import result_folders
from src.project_runtime.paths import resolve_path, portable_config
from .extract import full_labels


def settings(config, material, scales):
    ratio = scales[material] / scales['Al']
    old = config['lineage']
    result = {k: old[k] for k in ('minimum_tracked_size', 'minimum_shared_atoms',
                                 'minimum_overlap_fraction', 'maximum_missing_frames')}
    for name in ('neighbor_cutoff', 'local_birth_radius', 'interface_distance'):
        result[name + '_A'] = old[name + '_Al_equivalent_A'] * ratio
    result['thresholds'] = []
    for threshold in old['thresholds']:
        steps = threshold['persistence_ps'] / config['cadence_ps']
        if abs(steps - round(steps)) > 1e-8:
            raise ValueError('Persistence duration must be represented exactly')
        result['thresholds'].append(dict(threshold, persistence_frames=round(steps) + 1))
    return result


def prepare(config):
    root = result_folders(resolve_path(config['output']))
    catalog_path, structural_path = map(resolve_path, (config['source_catalog'], config['structural_plan']))
    catalog, structural = json.loads(catalog_path.read_text()), json.loads(structural_path.read_text())
    selected = [s for s in catalog['sources'] if s['kind'] == 'dynamic' and s['stratum'] in config['strata']]
    native = {s['id']: s for s in structural['sources']}
    million = [s for s in selected if 'al_meam_1m_450K_400ps_with_melt_20260913T205405Z' in s['path']]
    if len(million) != 2:
        raise ValueError('Require exactly the recorded Al million-atom melt and source phases')
    million.sort(key=lambda s: 0 if '/melt/' in s['path'] else 1)
    groups = [('al-million-continuous', million)] + [(s['id'], [s]) for s in selected if s not in million]
    sources, tasks = [], []
    for sid, parts in groups:
        material = parts[0]['material']
        if any(p['material'] != material or p['potential'] != parts[0]['potential'] for p in parts):
            raise ValueError(f'Cannot stitch different material/potential histories: {sid}')
        arrays = [source_arrays(p) for p in parts]
        ids = arrays[0]['atom_ids']
        if np.any(np.diff(ids) <= 0):
            raise ValueError(f'Unsorted/duplicate atom IDs: {sid}')
        evidence = None
        if len(parts) == 2:
            np.testing.assert_array_equal(ids, arrays[1]['atom_ids'])
            b0 = arrays[0]['box_high'][-1].astype(float) - arrays[0]['box_low'][-1].astype(float)
            b1 = arrays[1]['box_high'][0].astype(float) - arrays[1]['box_low'][0].astype(float)
            delta = arrays[0]['positions'][-1].astype(float) - arrays[1]['positions'][0].astype(float)
            delta -= b0 * np.rint(delta / b0)
            rms, maximum = float(np.sqrt(np.mean(delta ** 2))), float(np.abs(delta).max())
            if not np.allclose(b0, b1, rtol=0, atol=1e-4) or rms > 1e-5 or maximum > 1e-3:
                raise ValueError(f'Unverified continuous phase boundary: {sid}, RMS={rms}, max={maximum}')
            evidence = dict(minimum_image_rms_A=rms, maximum_A=maximum,
                            rule='retain melt endpoint and omit the duplicate source initial frame')
        frames, times = [], []
        elapsed = 0.
        for pi, (part, a) in enumerate(zip(parts, arrays)):
            raw_times = (a['timesteps'] - a['timesteps'][0]).astype(float) * part['timestep_fs'] / 1000
            dt = np.diff(raw_times)
            stride = int(round(config['cadence_ps'] / dt[0]))
            if stride < 1 or not np.allclose(dt, dt[0], rtol=0, atol=1e-8) or abs(stride * dt[0] - config['cadence_ps']) > 1e-8:
                raise ValueError(f'Cannot select the declared exact cadence: {sid}')
            selected_frames = list(range(0, len(raw_times), stride))
            if pi:
                selected_frames = selected_frames[1:]
            frames.extend([[pi, f] for f in selected_frames])
            times.extend((elapsed + raw_times[selected_frames]).tolist())
            elapsed += float(raw_times[-1])
        np.testing.assert_allclose(times, np.arange(len(times)) * config['cadence_ps'], rtol=0, atol=1e-8)
        sampling_source = native[parts[-1]['id']]
        centers = sampling_source['center_ids']
        expected = int(np.floor(len(ids) * config['center_fraction'] + .5))
        if len(centers) != expected:
            raise ValueError(f'Existing structural centers do not match declared density: {sid}')
        rows = np.searchsorted(ids, centers)
        np.testing.assert_array_equal(ids[rows], centers)
        seed = int(digest([config['seed'], sid])[:16], 16)
        baseline = np.sort(np.random.default_rng(seed).choice(len(centers), config['baseline_centers'], replace=False))
        source = dict(id=sid, material=material, potential=parts[0]['potential'], parts=parts,
                      ancestry_group=sampling_source['lineage'], atom_count=len(ids), frame_count=len(frames),
                      frames=frames, times_ps=times, cadence_ps=config['cadence_ps'],
                      center_atom_ids=centers, baseline_center_indices=baseline.tolist(),
                      center_sampling='reuse outcome-blind structural-release centers; measurement centers also tracked through Al melt',
                      phase_boundary=evidence, lineage=settings(config, material, catalog['scales']),
                      quantization=[json.loads((resolve_path(p['path']) / 'manifest.json').read_text())['provenance'].get('quantization') for p in parts])
        sources.append(source)
        for start in range(0, len(frames), config['chunk_frames']):
            stop = min(len(frames), start + config['chunk_frames'])
            tasks.append(dict(source=sid, start=start, stop=stop, atom_frames=(stop - start) * len(ids)))
    # Greedy load balancing distributes the ten-million-atom tasks and long Al
    # history across allocations, instead of assigning one long job per source.
    loads = [0] * config['extraction_lanes']
    for task in sorted(tasks, key=lambda t: (-t['atom_frames'], t['source'], t['start'])):
        lane = int(np.argmin(loads))
        task['lane'] = lane
        loads[lane] += task['atom_frames']
    plan = dict(config=config, sources=sources, tasks=tasks, lane_atom_frames=loads,
                source_catalog_sha256=sha(catalog_path), structural_plan_sha256=sha(structural_path),
                scales=catalog['scales'], sampling='every fifth saved frame: 0.5 ps; whole trajectories, no outcome filtering',
                exclusions=[dict(material='Zr', reason='registered available Zr inputs are static; no temporal origin labels')],
                ancestry_limit='branch-local counts; shared preparation groups are not independent replicates')
    plan['identity'] = digest(plan)
    dest = root / 'technical/plan.json'
    if dest.exists() and json.loads(dest.read_text()) != plan:
        raise ValueError('External audit plan changed; use a new output for a new population')
    write_json(dest, plan)
    return plan


@lru_cache(maxsize=8)
def arrays(path, manifest_sha256):
    return source_arrays(dict(path=path, manifest_sha256=manifest_sha256, kind='dynamic'))


def geometry(source, frame):
    part_index, raw_frame = source['frames'][frame]
    part = source['parts'][part_index]
    a = arrays(part['path'], part['manifest_sha256'])
    box = a['box_high'][raw_frame].astype(float) - a['box_low'][raw_frame].astype(float)
    points = np.mod(a['positions'][raw_frame].astype(float), box)
    if np.any(box <= 0) or not np.isfinite(points).all():
        raise ValueError(f'Invalid frame geometry: {source["id"]}/{frame}')
    return points, box


def source_folder(plan, sid):
    return resolve_path(plan['config']['output']) / 'technical/sources' / sid


def chunk_identity(plan, task):
    return digest(dict(plan=plan['identity'], task=task, producer=sha(Path(__file__)),
                       ptm=sha(Path(__file__).with_name('extract.py')), ovito=importlib.metadata.version('ovito')))


def extract_chunk(plan, task):
    os.environ['OVITO_THREAD_COUNT'] = '1'
    source = next(s for s in plan['sources'] if s['id'] == task['source'])
    folder = source_folder(plan, source['id'])
    folder.mkdir(parents=True, exist_ok=True)
    name = f'ptm-{task["start"]:04d}-{task["stop"]:04d}'
    path, receipt = folder / (name + '.npz'), folder / (name + '.json')
    identity = chunk_identity(plan, task)
    with (folder / (name + '.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old['identity'] != identity or old['sha256'] != sha(path):
                raise ValueError(f'Changed external PTM chunk: {path}')
            return old
        started = time.monotonic()
        labels = np.stack([full_labels(*geometry(source, f), plan['config']['ptm_rmsd_cutoff'])
                           for f in range(task['start'], task['stop'])])
        temporary = path.with_suffix('.building.npz')
        np.savez_compressed(temporary, labels=labels)
        temporary.replace(path)
        result = dict(identity=identity, **task, sha256=sha(path), seconds=time.monotonic() - started)
        write_json(receipt, result)
        return result


def initialize_worker(path):
    global WORKER_PLAN
    WORKER_PLAN = json.loads(Path(path).read_text())


def chunk_worker(task):
    return extract_chunk(WORKER_PLAN, task)


def lane(plan, index):
    tasks = sorted((t for t in plan['tasks'] if t['lane'] == index),
                   key=lambda t: (-t['atom_frames'], t['source'], t['start']))
    root = resolve_path(plan['config']['output']) / 'technical'
    done = []
    try:
        with ProcessPoolExecutor(max_workers=plan['config']['workers_per_lane'], mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_worker, initargs=(str(root / 'plan.json'),)) as pool:
            futures = [pool.submit(chunk_worker, t) for t in tasks]
            for future in as_completed(futures):
                result = future.result()
                done.append([result['source'], result['start']])
                write_json(root / f'lane-{index:02d}.json', dict(state='running', completed=done, total=len(tasks),
                           allocation=os.environ.get('SLURM_JOB_ID'), node=os.uname().nodename))
                print(json.dumps(dict(source=result['source'], start=result['start'], seconds=result['seconds'],
                                      completed=len(done), total=len(tasks))), flush=True)
        write_json(root / f'lane-{index:02d}.json', dict(state='complete', completed=done, total=len(tasks)))
    except BaseException:
        write_json(root / f'lane-{index:02d}.json', dict(state='failed', completed=done, total=len(tasks), traceback=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['plan', 'extract-lane'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if args.stage == 'plan':
        plan = prepare(config)
        print(json.dumps(dict(sources=len(plan['sources']), chunks=len(plan['tasks']), lane_atom_frames=plan['lane_atom_frames'])))
    else:
        plan = json.loads((resolve_path(config['output']) / 'technical/plan.json').read_text())
        if plan['config'] != config:
            raise ValueError('Changed external configuration')
        lane(plan, int(os.environ['SLURM_ARRAY_TASK_ID']))


if __name__ == '__main__':
    main()
