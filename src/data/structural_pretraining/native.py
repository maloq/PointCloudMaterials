"""Larger native-unit structural release; fixed evaluation sources never enter fitting.

This is distinct from the older material-rescaled temporal/TDA protocol. It
extracts only current patches and existing relaxed teachers, and produces no MD.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import math
import multiprocessing
from pathlib import Path
import time
import traceback

import numpy as np

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import centered, digest, sha, write_json
from src.project_runtime.paths import dataset_path, portable_config, resolve_path
from .prepare import ELEMENTS, chart, source_arrays


def implementation():
    repo = Path(__file__).resolve().parents[3]
    return {str(p.relative_to(repo)): sha(p) for p in (
        Path(__file__), Path(__file__).with_name('prepare.py'),
        repo/'src/data/fixed_cohort/protocol.py')}


def frames_at_spacing(steps, timestep_fs, spacing_ps):
    times = (steps - steps[0]).astype(np.float64) * timestep_fs / 1000
    if np.any(np.diff(times) <= 0):
        raise ValueError('Source timeline is not strictly increasing')
    frames = [0]
    while True:
        nxt = int(np.searchsorted(times, times[frames[-1]] + spacing_ps - 1e-8))
        if nxt >= len(times):
            return frames
        if nxt <= frames[-1]:
            raise ValueError('Invalid structural frame spacing')
        frames.append(nxt)


def build_plan(config):
    if config['protocol'] != 'native_multimaterial_structural_v1':
        raise ValueError('Undeclared native structural protocol')
    root = resolve_path(config['release'])
    path = root/'plan.json'
    _, fixed = read_release(config['fixed_dataset']['root'])
    if fixed['identity'] != config['fixed_dataset']['identity']:
        raise ValueError('Fixed benchmark identity changed')
    catalog_path = resolve_path(config['source_catalog'])
    catalog = json.loads(catalog_path.read_text())
    evidence = dict(fixed_identity=fixed['identity'], catalog_sha256=sha(catalog_path))
    if path.exists():
        plan = json.loads(path.read_text())
        if (plan['config'] != config or plan['implementation'] != implementation()
                or plan['evidence'] != evidence):
            raise ValueError('Structural recipe/producer changed; choose a new release')
        return plan
    sources = []
    for old in fixed['sources']:
        if old['role'] not in ('train', 'selection'):
            continue
        sources.append(dict(id=f'native-{old["id"]}', native_id=old['id'],
            material='Al', atomic_number=13, kind='dynamic', split=old['role'],
            stratum='al_native', lineage=old['lineage'], potential='al-lee2003-meam',
            path=portable_config(str(dataset_path(old['dataset'])/old['relative_trajectory_path'])),
            manifest_sha256=old['manifest_sha256'], timestep_fs=old['timestep_fs'],
            frame_count=old['frame_count'], inherited_center_ids=old['center_atom_ids'],
            centers_requested=config['native_train_centers'] if old['role']=='train' else len(old['center_atom_ids']),
            cells=old['cells'], eligibility='inherited exact fixed-cohort source role'))
    heldout = {s['lineage'] for s in fixed['sources'] if s['role'] != 'train'}
    blocked_manifests = {s['manifest_sha256'] for s in fixed['sources'] if s['role'] != 'train'}
    seen = {s['manifest_sha256'] for s in sources}
    for old in catalog['sources']:
        dynamic = old['kind']=='dynamic' and old['stratum'] in config['external_dynamic_strata']
        static = old['kind']=='static' and old['material'] in config['external_static_materials']
        if not (dynamic or static):
            continue
        if old['split'] != 'train' or old['lineage'] in heldout:
            raise ValueError(f'External source is not training eligible: {old["id"]}')
        key = old['manifest_sha256'] if dynamic else old['file_sha256']
        if key in blocked_manifests:
            raise ValueError(f'External source aliases held-out source: {old["id"]}')
        if key in seen:
            continue
        seen.add(key)
        source = dict(old, path=portable_config(str(resolve_path(old['path']))),
            atomic_number=ELEMENTS[old['material']], inherited_center_ids=[], cells=[],
            eligibility='external train-only family from recorded structural catalog; never used for Al validation/test')
        # Melt and measurement are one preparation lineage, not independent roots.
        if 'al_meam_1m_450K_400ps_with_melt_20260913T205405Z' in old['path']:
            source['lineage'] = 'al-1m-independent-melt-911001'
        sources.append(source)
    tasks = []
    for source in sources:
        arrays = source_arrays(source)
        n_atoms = len(arrays['atom_ids'])
        source['atom_count'] = n_atoms
        if source['kind']=='dynamic':
            frames = frames_at_spacing(arrays['timesteps'], source['timestep_fs'], config['frame_spacing_ps'])
            # Consecutive source/melt phases share their boundary snapshot.
            if source['lineage']=='al-1m-independent-melt-911001' and '/source/' in source['path']:
                frames = frames[1:]
            eligible = np.arange(n_atoms)
        else:
            frames = [0]
            x = arrays['positions']
            eligible = np.flatnonzero(((x > x.min(0)+config['radius_A']) &
                                      (x < x.max(0)-config['radius_A'])).all(-1))
        if 'native_id' not in source:
            source['centers_requested'] = max(1, math.floor(n_atoms*config['external_center_fraction']+.5))
        count = source['centers_requested']
        inherited = np.asarray(source['inherited_center_ids'], dtype=np.int64)
        if len(inherited):
            inherited_rows = np.searchsorted(arrays['atom_ids'], inherited)
            if not np.array_equal(arrays['atom_ids'][inherited_rows], inherited):
                raise ValueError(f'Inherited centers missing: {source["id"]}')
        else:
            inherited_rows = np.empty(0, dtype=np.int64)
        remaining = np.setdiff1d(eligible, inherited_rows, assume_unique=True)
        if count < len(inherited) or count-len(inherited) > len(remaining):
            raise ValueError(f'Insufficient distinct center candidates: {source["id"]}')
        seed = int(digest([config['seed'], source['id']])[:16], 16)
        extra = np.random.default_rng(seed).choice(remaining, count-len(inherited), replace=False)
        center_rows = np.sort(np.r_[inherited_rows, extra])
        source['center_ids'] = arrays['atom_ids'][center_rows].astype(np.int64).tolist()
        source['observed_frames'] = frames
        paired = {c['frame'] for c in source['cells']}
        source['paired_frames'] = sorted(paired)
        for frame in sorted(set(frames) | paired):
            for start in range(0, count, config['maximum_rows_per_shard']):
                tasks.append(dict(id=f'{source["id"]}-{frame:06d}-{start:06d}', source=source['id'],
                    frame=frame, start=start, stop=min(start+config['maximum_rows_per_shard'],count),
                    observed=frame in frames, paired=frame in paired, split=source['split']))
        del arrays
    plan = dict(protocol=config['protocol'], config=config, evidence=evidence, implementation=implementation(),
        sources=sources, tasks=tasks, excluded_fixed_sources=[
            {k:s[k] for k in ('id','role','lineage','manifest_sha256')}
            for s in fixed['sources'] if s['role'] in ('calibration','test')],
        evaluation='Fixed Al64 all64/legacy16 source and sample IDs unchanged',
        normalization='Native Angstrom, no material-dependent rescaling',
        labels='No onset/PTM labels read or used in sampling; no outcome filtering',
        ancestry_limitations='Archived non-native branches/static frames may share preparation ancestry; potential unknown for static originals; not independent-source replicates')
    plan['identity'] = digest(plan)
    write_json(path,plan)
    write_json(root/'summary.json',summary(plan))
    return plan


def summary(plan):
    rows = Counter(); sources = Counter()
    for s in plan['sources']:
        key = f'{s["split"]}/{s["material"]}/{s["kind"]}'
        sources[key] += 1
        rows[key] += len(s['observed_frames'])*len(s['center_ids'])
    return dict(identity=plan['identity'], observed_rows=dict(rows), source_records=dict(sources),
        paired_rows={role:sum(len(s['center_ids'])*len(s['cells']) for s in plan['sources'] if s['split']==role)
                     for role in ('train','selection')}, tasks=len(plan['tasks']))


def prepare_frames(arguments):
    """One source/frame per task; reuse the full-cell tree across bounded shards."""
    root_value, source, tasks, config, identity = arguments
    root = Path(root_value); results = []; pending = []
    for task in tasks:
        dest = root/'shards'/task['id']; receipt = dest/'complete.json'
        if receipt.exists():
            record = json.loads(receipt.read_text())
            if record['identity'] != identity or any(sha(dest/name)!=h for name,h in record['files'].items()):
                raise ValueError(f'Changed structural shard: {dest}')
            results.append(record)
        else:
            pending.append(task)
    if not pending:
        return results
    frame = pending[0]['frame']; started = time.monotonic(); arrays = source_arrays(source)
    x,tree,box = chart(arrays,frame,source['kind']=='static')
    all_ids = np.asarray(source['center_ids'],dtype=np.int64)
    rows = np.searchsorted(arrays['atom_ids'],all_ids)
    if not np.array_equal(arrays['atom_ids'][rows],all_ids):
        raise ValueError(f'Changed center IDs: {source["id"]}')
    cell = next((c for c in source['cells'] if c['frame']==frame),None)
    relaxed = None
    if cell is not None:
        archive = resolve_path(cell['archive']); raw_path = archive/'relaxed_binary_float16'
        if sha(archive/'metadata.json')!=cell['metadata_sha256'] or sha(raw_path/'manifest.json')!=cell['manifest_sha256']:
            raise ValueError(f'Changed relaxed ancestry: {archive}')
        cold = source_arrays(dict(kind='dynamic',path=str(raw_path),manifest_sha256=cell['manifest_sha256']))
        if not np.array_equal(cold['atom_ids'],arrays['atom_ids']) or cold['timesteps'][0]!=arrays['timesteps'][frame]:
            raise ValueError(f'Relaxed identities/timeline mismatch: {archive}')
        cold_box = cold['box_high'][0].astype(float)-cold['box_low'][0].astype(float)
        if not np.allclose(cold_box,box,atol=1e-5,rtol=0):
            raise ValueError(f'Relaxed box changed: {archive}')
        relaxed = np.mod(cold['positions'][0].astype(float),box)
    for task in pending:
        centers = rows[task['start']:task['stop']]
        neighbors = tree.query(x[centers],k=config['candidate_atoms'],workers=1)[1]
        if not np.array_equal(neighbors[:,0],centers):
            raise ValueError(f'Center is not nearest atom (quantization/ties): {task["id"]}')
        if box is None:
            hot = (x[neighbors]-x[centers,None]).astype(np.float32)
        else:
            hot = centered(x,box,centers,neighbors)
        if not np.isfinite(hot).all() or np.any(hot[:,0]):
            raise ValueError(f'Invalid coordinates: {task["id"]}')
        values = dict(hot=hot,neighbor_ids=arrays['atom_ids'][neighbors],center_ids=arrays['atom_ids'][centers])
        if task['paired']:
            values['cold'] = centered(relaxed,box,centers,neighbors)
        dest = root/'shards'/task['id']; dest.mkdir(parents=True,exist_ok=True); files = {}
        for name,value in values.items():
            temporary = dest/f'{name}.tmp.npy'; np.save(temporary,value,allow_pickle=False)
            target = dest/f'{name}.npy'; temporary.replace(target); files[target.name]=sha(target)
        record = dict(identity=identity,task=task,rows=len(centers),atomic_number=source['atomic_number'],
            material=source['material'],potential=source['potential'],lineage=source['lineage'],
            files=files,elapsed_frame_seconds=time.monotonic()-started)
        write_json(dest/'complete.json',record); results.append(record)
    return results


def run(config,workers):
    root = resolve_path(config['release']); root.mkdir(parents=True,exist_ok=True)
    with (root/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = build_plan(config)
        sources = {s['id']:s for s in plan['sources']}; grouped = {}
        for task in plan['tasks']:
            grouped.setdefault((task['source'],task['frame']),[]).append(task)
        jobs = [(str(root),sources[sid],tasks,config,plan['identity']) for (sid,_),tasks in grouped.items()]
        # Mix families early, independently of outcomes; receipts make interruption resumable.
        order = np.random.default_rng(config['seed']).permutation(len(jobs))
        completed = 0
        try:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                futures = [pool.submit(prepare_frames,jobs[i]) for i in order]
                for future in as_completed(futures):
                    records = future.result(); completed += len(records)
                    status = dict(state='preparing',identity=plan['identity'],completed_shards=completed,
                        total_shards=len(plan['tasks']),last=records[-1]['task']['id'])
                    write_json(root/'status.json',status)
                    if completed%25 < len(records): print(json.dumps(status),flush=True)
            shards = [json.loads((root/'shards'/t['id']/'complete.json').read_text()) for t in plan['tasks']]
            result = dict(state='complete',identity=plan['identity'],plan_sha256=sha(root/'plan.json'),
                shards=shards,summary=summary(plan),config=config)
            write_json(root/'manifest.json',result)
            write_json(root/'status.json',dict(state='complete',identity=plan['identity'],shards=len(shards)))
            return result['summary']
        except BaseException as error:
            write_json(root/'status.json',dict(state='failed',error=repr(error),traceback=traceback.format_exc(),
                identity=plan['identity'],completed_shards=completed))
            raise


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=('plan','prepare'))
    parser.add_argument('--config',required=True); parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args(); config=json.loads(Path(args.config).read_text())
    result=summary(build_plan(config)) if args.stage=='plan' else run(config,args.workers)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
