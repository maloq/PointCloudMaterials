"""Quench every observed full cell while preserving the frozen birth cohort."""
from collections import defaultdict
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.data.relaxed_targets.worker import AbsolutePositions, clouds, publish, verify_archive
from src.data.conversion.relaxation import convert, read_relaxed
from src.project_runtime.paths import resolve_path
from src.research.crystallization_origin.extract import raw_source, frame_geometry
from src.simulation.relaxation import relax_frame
from .data import load, read
from .extension import base


INPUT = dict(relaxation=True,
    coordinates='centered relaxed coordinates of the same nearest-80 atom IDs selected in original MD geometry',
    computational_context='full original periodic cell, fixed box, generating MEAM potential, FIRE quench',
    force_tolerance_eV_per_A=.01,
    precision='local float32 offsets extracted from full-precision quench before global float16 archival conversion',
    membership='original MD nearest-80 atom IDs; no reselection after quenching',
    eligibility='original MD crystal-free radius 8 A; no relaxed-state label filtering',
    labels='unchanged original MD first-appearance/establishment and persistent-liquid controls',
    causal='only the same observed frame is minimized; no future frames in relaxation')


def freeze(c):
    root = resolve_path(c['output']) / 'technical'
    path = root / 'relaxation-plan.json'
    if path.exists():
        p = read(path)
        if p['config'] != c or digest({k: v for k, v in p.items() if k != 'identity'}) != p['identity']:
            raise ValueError('Changed relaxed-input release; use a new output')
        return p
    b = base(c)
    positions, rows, manifest = load(b)
    ancestor = read(resolve_path(b['cache']) / 'plan.json')
    # A patch may occur in several histories. Its physical binding must agree.
    bindings = {}
    for i in range(len(rows['id'])):
        for k, patch in enumerate(rows['indices'][i]):
            key = (int(rows['source'][i]), int(rows['start_frame'][i] + k), int(rows['atom'][i]))
            if int(patch) in bindings and bindings[int(patch)] != key:
                raise ValueError(f'Conflicting physical identity for patch {patch}')
            bindings[int(patch)] = key
    if sorted(bindings) != list(range(len(positions))):
        raise ValueError('Every original patch must have exactly one physical identity')
    groups = defaultdict(list)
    for patch, (source, frame, atom) in sorted(bindings.items()):
        groups[(source, frame)].append(dict(patch=patch, atom=atom))
    tasks = [dict(id=f'{s}-{f}', source=s, frame=f, patches=v) for (s, f), v in sorted(groups.items())]
    np.random.default_rng(c['seed']).shuffle(tasks)
    potential = [str(resolve_path(p)) for p in c['potential_files']]
    for f, h in zip(potential, c['potential_sha256'], strict=True):
        if sha(Path(f)) != h:
            raise ValueError(f'Changed generating potential: {f}')
    binary = resolve_path(c['lammps_binary'])
    p = dict(config=c, ancestor_manifest=manifest, ancestor_cache=str(resolve_path(b['cache'])),
        sources=ancestor['sources'], tasks=tasks, patch_count=len(positions),
        rows_sha256=manifest['files']['rows.npz'], potential_files=potential,
        lammps=str(binary), lammps_sha256=sha(binary), input_domain=INPUT,
        mpi_launcher=[str(Path(sys.executable).parent / 'mpiexec'), '-n', '{ranks}'],
        implementation={str(Path(f).name): sha(Path(f)) for f in
            [__file__, relax_frame.__code__.co_filename, convert.__code__.co_filename]})
    p['identity'] = digest(p)
    write_json(path, p)
    # The existing temporal protocol can consume this explicitly derived release
    # through its normal, hash-checked parent contract.
    derived = dict(b, cache=c['cache'], output=c['output'], prepare_tasks=c['descriptor_workers'], input_domain=INPUT)
    write_json(root / 'derived-parent/technical/code/config.json', derived)
    return p


def study(c):
    p = freeze(c)
    parent = resolve_path(c['output']) / 'technical/derived-parent'
    return dict(c, parent_run=str(parent), parent_config_sha256=sha(parent / 'technical/code/config.json'),
                dataset_identity=p['identity'])


def settings(p):
    c = p['config']
    os.environ.update(c['mpi_environment'])
    if sha(Path(p['lammps'])) != p['lammps_sha256']:
        raise ValueError('Changed pinned LAMMPS executable')
    for f, h in zip(p['potential_files'], c['potential_sha256'], strict=True):
        if sha(Path(f)) != h:
            raise ValueError('Changed generating MEAM potential')
    return dict(c['relaxation'],
        lammps_command=[v.format(ranks=c['ranks']) for v in p['mpi_launcher']] + [p['lammps']],
        potential_files=p['potential_files'],
        pair_commands=['pair_style meam', f'pair_coeff * * {p["potential_files"][0]} Al {p["potential_files"][1]} Al'])


def checked(p, task):
    folder = resolve_path(p['config']['cache']) / 'cells' / task['id']
    done = read(folder / 'complete.json')
    if done['identity'] != p['identity'] or done['task'] != task:
        raise ValueError(f'Changed relaxed-cell binding: {task["id"]}')
    if sha(folder / 'patches.npz') != done['patches_sha256']:
        raise ValueError(f'Changed relaxed patches: {task["id"]}')
    verify_archive(Path(done['archive']))
    return done


def cell(p, task):
    c = p['config']
    folder = resolve_path(c['cache']) / 'cells' / task['id']
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return checked(p, task)
    source = next(s for s in p['sources'] if s['id'] == task['source'])
    raw = raw_source(source)  # validates manifest, exact cadence and identities
    frame = task['frame']
    observed, lengths = frame_geometry(raw, frame)
    atoms = np.array([v['atom'] for v in task['patches']], np.int64)
    patches = np.array([v['patch'] for v in task['patches']], np.int64)
    centers = np.searchsorted(raw.atom_ids, atoms)
    np.testing.assert_array_equal(raw.atom_ids[centers], atoms)
    work_root = resolve_path(c['scratch']) / 'cells' / task['id']
    archive = resolve_path(c['archive']) / 'cells' / task['id']
    if not archive.exists():
        attempts = sorted(work_root.glob('attempt-*'))
        work = attempts[-1] if attempts else work_root / 'attempt-000'
        params = settings(p)
        if not (work / 'metadata.json').exists():
            # Keep stopped minimizations and restart their full-precision dump.
            if attempts:
                previous = attempts[-1]
                if (previous / 'relaxed.dump').exists():
                    params.update(restart_dump=str(previous / 'relaxed.dump'), restart_sha256=sha(previous / 'relaxed.dump'))
                work = work_root / f'attempt-{len(attempts):03d}'
            absolute = SimpleNamespace(**vars(raw), atom_count=raw.atom_count)
            absolute.positions = AbsolutePositions(raw)
            try:
                relax_frame(absolute, frame, work, params)
            except BaseException:
                failure = resolve_path(c['archive']) / 'failures' / task['id'] / work.name
                if work.exists() and not failure.exists():
                    publish(work, failure)
                raise
        meta = read(work / 'metadata.json')
        if (meta['source_manifest_sha256'] != source['manifest_sha256'] or meta['source_frame'] != frame
                or meta['fmax_eV_per_A'] > c['relaxation']['force_tolerance']):
            raise ValueError(f'Incorrect recovered minimization: {work}')
        if not (work / 'patches.npz').exists():
            relaxed, _ = read_relaxed(work)
            hot, cold, neighbors = clouds(observed, relaxed - raw.box_low[frame], lengths, centers)
            original = np.load(Path(p['ancestor_cache']) / 'positions.npy', mmap_mode='r')[patches]
            np.testing.assert_allclose(hot, original, rtol=0, atol=1e-6,
                                       err_msg=f'Observed nearest-80 replay differs for {task["id"]}')
            if not np.isfinite(cold).all():
                raise FloatingPointError(f'Nonfinite relaxed input: {task["id"]}')
            np.savez_compressed(work / 'patches.npz', indices=patches, positions=cold,
                                neighbor_atom_ids=raw.atom_ids[neighbors], center_atom_ids=atoms)
        if not (work / 'conversion.json').exists():
            convert(work, delete_source=True, local_cloud_dtype='float32')
        publish(work, archive)
    verify_archive(archive)
    shutil.copy2(archive / 'patches.npz', folder / 'patches.npz')
    meta = read(archive / 'metadata.json')
    done = dict(identity=p['identity'], task=task, archive=str(archive),
        patches_sha256=sha(folder / 'patches.npz'), fmax_eV_per_A=meta['fmax_eV_per_A'], seconds=meta['seconds'],
        local_clouds_saved=True, relaxation_metadata_sha256=sha(archive / 'metadata.json'))
    write_json(folder / 'complete.json', done)
    # All successful state is now verified on STORE and IDS.
    if work_root.exists():
        shutil.rmtree(work_root)
    return done


def seal(c):
    p = freeze(c)
    root = resolve_path(c['cache'])
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'manifest.json').exists():
        _, _, manifest = load(read(resolve_path(c['output']) / 'technical/derived-parent/technical/code/config.json'))
        if manifest['identity'] != p['identity']:
            raise ValueError('Sealed relaxation identity changed')
        return manifest
    positions = np.lib.format.open_memmap(root / 'positions.npy', mode='w+', dtype=np.float32,
                                         shape=(p['patch_count'], 80, 3))
    coverage = np.zeros(p['patch_count'], np.int32)
    receipts = []
    for task in p['tasks']:
        done = checked(p, task)
        with np.load(root / 'cells' / task['id'] / 'patches.npz') as a:
            ids = a['indices']
            np.testing.assert_array_equal(ids, [v['patch'] for v in task['patches']])
            positions[ids] = a['positions']
            coverage[ids] += 1
        receipts.append(done)
    if not np.all(coverage == 1):
        raise ValueError('Relaxed release must cover every original patch exactly once; fits remain gated')
    positions.flush()
    del positions
    old = Path(p['ancestor_cache'])
    shutil.copy2(old / 'rows.npz', root / 'rows.npz')
    if sha(root / 'rows.npz') != p['rows_sha256']:
        raise ValueError('Relaxation changed original rows, labels or splits')
    write_json(root / 'plan.json', p)
    records, offset = [], 0
    bank = np.load(root / 'positions.npy', mmap_mode='r')
    for item in p['ancestor_manifest']['sources']:
        dest = root / 'sources' / str(item['source'])
        dest.mkdir(parents=True, exist_ok=True)
        np.save(dest / 'positions.npy', bank[offset:offset + item['patches']])
        offset += item['patches']
        record = dict(item, identity=digest(dict(ancestor=item['identity'], relaxation=p['identity'])),
                      files={'positions.npy': sha(dest / 'positions.npy')})
        write_json(dest / 'complete.json', record)
        records.append(record)
    if offset != p['patch_count']:
        raise ValueError('Source concatenation changed patch order')
    manifest = dict(identity=p['identity'], sources=records,
        files={f: sha(root / f) for f in ['plan.json', 'positions.npy', 'rows.npz']},
        summary=p['ancestor_manifest']['summary'], supported=True, cohort=p['ancestor_manifest']['cohort'],
        ancestor_identity=p['ancestor_manifest']['identity'], input_domain=INPUT,
        relaxation_cells=len(receipts), convergence_max=max(r['fmax_eV_per_A'] for r in receipts))
    write_json(root / 'manifest.json', manifest)
    return manifest
