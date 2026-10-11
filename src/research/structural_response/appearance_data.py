"""Prospective first-PTM-appearance panels from the pinned native 0.1 ps pool.

Locations and observation times are selected without reading future labels.
The risk set is defined by absence of a PTM crystal in the present observation.
This is a separate target, not a filter applied to the established-birth assay.
"""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import resolve_path
from .birth_audit import read

FAMILY = 'motion_first_appearance'


def root(c):
    return resolve_path(c['output'])


def cache(c):
    return resolve_path(c['cache'])


def prepare(c):
    from src.postprocessing.datasets.births import NucleiBirthDataset
    parent = resolve_path(c['parent'])
    old = read(parent/'technical/birth-cohort.json')
    if old['fingerprint'] != c['fingerprint']:
        raise ValueError('Parent source pin differs from the requested catalogue')
    with NucleiBirthDataset('010', expected_fingerprint=c['fingerprint']) as data:
        data.pin(root(c)/'technical/nuclei-birth-dataset.json')
    tasks = []
    for item in old['sources']:
        s = item['source']
        timeline = np.asarray(s['times_ps'])[s['views']['010']]
        if not np.allclose(np.diff(timeline), .1, rtol=0, atol=1e-8):
            raise ValueError('Nonexact 0.1 ps timeline: '+s['id'])
        for age in c['observation_offsets_ps']:
            slot = int(round(age/.1))
            past = int(round(c['history_audit_ps']/.1))
            future = int(round(max(c['horizons_ps'])/.1))+c['persistence_frames']-1
            if slot-past < 0 or slot+future >= len(timeline):
                raise ValueError('Declared fixed observation window unavailable: '+s['id'])
            tasks.append(dict(id=item['source_id']+f'-t{slot:05d}', source_id=item['source_id'],
                slot=slot, slots=list(range(slot-past, slot+future+1)),
                observation_time_ps=float(timeline[slot])))
    binding = dict(config=c, parent_cohort_sha256=sha(parent/'technical/birth-cohort.json'),
        sources=old['sources'], excluded_sources=old['excluded'], tasks=tasks,
        implementations=check_metric_docs(family=FAMILY)[FAMILY]['files'],
        sampling='Fixed elapsed offsets and hash-seeded uniform atom IDs, before any PTM/future access. No event catalogue conditioning.',
        target='First observed FCC/HCP/BCC label among the present observation atom identities, after a crystal-free present. Not first-ever appearance over unobserved history.',
        risk_set='No present PTM crystalline atom in the consumed nearest80/radius8 observation; no external neighborhood/component/distance screening.',
        history='Past 0.5 ps labels are diagnostic only; do not change primary eligibility.',
        predictor_inputs='Present original-MD Q,V only. No temperature, age, time, source or label fields.',
        catalogue_events_used=False)
    binding['identity'] = digest(binding)
    path = root(c)/'technical/plan.json'
    if path.exists() and read(path) != binding:
        raise ValueError('Appearance plan changed; use a new output revision')
    write_json(path, binding)
    return binding


def extract(c, p, task):
    from src.postprocessing.io import geometry
    from src.postprocessing.io import arrays
    from src.postprocessing.descriptors.ptm import full_labels
    item = next(s for s in p['sources'] if s['source_id'] == task['source_id'])
    s = item['source']; dest = cache(c)/'windows'/task['id']
    dest.mkdir(parents=True, exist_ok=True)
    identity = digest([p['identity'], task])
    receipt = dest/'complete.json'
    if receipt.exists():
        done = read(receipt)
        if done['identity'] != identity or any(sha(dest/n) != h for n,h in done['files'].items()):
            raise ValueError('Changed first-appearance window')
        return done
    started = time.monotonic()
    native = arrays(s['parts'][0]['path'], s['parts'][0]['manifest_sha256'])
    atom_ids = native['atom_ids']; view = np.asarray(s['views']['010'])
    frame = int(view[task['slot']]); q, box = geometry(s, frame)
    rng = np.random.default_rng(int(digest([c['seed'], task['id']])[:16], 16))
    centers = rng.choice(len(q), c['centers_per_window'], replace=False)
    _, members = cKDTree(q, boxsize=box).query(q[centers], k=80, workers=1)
    if not np.array_equal(members[:,0], centers):
        raise ValueError('Present patch identity mismatch')
    delta = q[members]-q[centers,None]; delta -= box*np.rint(delta/box)
    consumed = np.linalg.norm(delta, axis=-1) < 8
    pi, rf = s['frames'][frame]; part = s['parts'][pi]
    if sha(Path(part['path'])/'manifest.json') != part['manifest_sha256']:
        raise ValueError('Changed raw Q,V manifest')
    manifest = read(Path(part['path'])/'manifest.json')
    if manifest['velocity_units'] != 'angstrom_per_ps':
        raise ValueError('Unexpected velocity units')
    velocity = np.load(Path(part['path'])/manifest['arrays']['velocities']['file'], mmap_mode='r')
    state = np.concatenate((delta.astype(np.float32), np.asarray(velocity[rf,members],np.float32)), -1)
    # Seal the drawn locations and present identities before looking at labels.
    selection = dict(identity=identity, center_atom_ids=atom_ids[centers].tolist(),
        member_atom_ids=atom_ids[members].tolist(), consumed=consumed.tolist(),
        frame=frame, slots=task['slots'], future_used_for_selection=False)
    selected = dest/'selection.json'
    if selected.exists() and read(selected) != selection:
        raise ValueError('Outcome-independent selection changed')
    write_json(selected, selection)
    labels = []
    for slot in task['slots']:
        f = int(view[slot]); xyz, cell = geometry(s, f)
        values = full_labels(xyz, cell, c['ptm_rmsd_cutoff'])
        labels.append(values[members])
    labels = np.stack(labels, axis=1)
    zero = task['slots'].index(task['slot'])
    crystal = np.isin(labels, [1,2,3]) & consumed[:,None,:]
    current_clear = ~crystal[:,zero].any(1)
    history_clear = ~crystal[:,:zero+1].any(axis=(1,2))
    future = crystal[:,zero+1:]
    any_crystal = future.any(2)
    first = np.where(any_crystal.any(1), any_crystal.argmax(1)+1, -1)
    persistent = np.stack([future[:,i:i+c['persistence_frames']].all(1).any(1)
        for i in range(int(round(max(c['horizons_ps'])/.1)))], axis=1)
    first_persistent = np.where(persistent.any(1), persistent.argmax(1)+1, -1)
    np.savez_compressed(dest/'observations.npz', state=state, labels=labels,
        atom=atom_ids[centers], members=atom_ids[members], consumed=consumed,
        clear=current_clear, history_clear=history_clear, first_step=first,
        persistent_first_step=first_persistent)
    done = dict(identity=identity, task=task['id'], source_id=item['source_id'],
        candidates=len(centers), at_risk=int(current_clear.sum()),
        seconds=time.monotonic()-started,
        files={n:sha(dest/n) for n in ('observations.npz','selection.json')})
    write_json(receipt, done); arrays.cache_clear()
    return done


def seal(c, p):
    rows, states, aliases, coverage = [], [], [], []
    seen, by_q = {}, {}
    items = {s['source_id']:s for s in p['sources']}
    for task in p['tasks']:
        dest = cache(c)/'windows'/task['id']; done = read(dest/'complete.json')
        if done['identity'] != digest([p['identity'],task]) or any(sha(dest/n)!=h for n,h in done['files'].items()):
            raise ValueError('Changed first-appearance packet')
        coverage.append(done)
        s = items[task['source_id']]['source']
        with np.load(dest/'observations.npz') as packet:
            for i in np.flatnonzero(packet['clear']):
                x = packet['state'][i]
                h = hashlib.sha256(x.tobytes()).hexdigest()
                qh = hashlib.sha256(np.ascontiguousarray(x[:,:3]).tobytes()).hexdigest()
                role = 'train' if s['role']=='train' else 'test'
                if qh in by_q and by_q[qh] != role:
                    raise ValueError('Exact present geometry overlaps training and test')
                by_q[qh] = role
                atom = int(packet['atom'][i]); identity = task['id']+':'+str(atom)
                onset = int(packet['first_step'][i]); persistent = int(packet['persistent_first_step'][i])
                if h in seen:
                    old = rows[seen[h]]
                    if old['ancestor'] != s['ancestry_group']:
                        raise ValueError('Identical Q,V across declared independent ancestors')
                    aliases.append(dict(id=identity, retained=old['id'], same_onsets=(onset==old['first_step'] and persistent==old['persistent_first_step']),
                        reason='Exact present Q,V; deterministic source/window order, never future-selected'))
                    continue
                seen[h] = len(rows); states.append(x)
                rows.append(dict(id=identity, event=identity, pair=task['id'], atom=atom,
                    source_id=task['source_id'], source=s['id'], ancestor=s['ancestry_group'], role=s['role'],
                    observation_time_ps=task['observation_time_ps'], original_frame=int(s['views']['010'][task['slot']]),
                    first_step=onset, persistent_first_step=persistent,
                    history_clear=bool(packet['history_clear'][i]), input_hash=h, coordinate_hash=qh,
                    index=len(rows)))
    if not rows:
        raise ValueError('No crystal-free present observations')
    roles = {}
    for r in rows:
        role='train' if r['role']=='train' else 'test'
        if r['ancestor'] in roles and roles[r['ancestor']] != role:
            raise ValueError('Ancestor crosses training and pooled test')
        roles[r['ancestor']] = role
    counts = Counter(r['ancestor'] for r in rows)
    for r in rows:
        r['weight'] = 1/counts[r['ancestor']]
    train = sorted((a for a,r in roles.items() if r=='train'), key=lambda a:digest([c['seed'],a,'appearance-inner-selection']))
    n = c['inner_selection_ancestors']
    if len(train)<=n:
        raise ValueError('Insufficient independent training ancestors')
    partition = dict(inner_fit=train[n:],inner_selection=train[:n],final_train=train,
        test=sorted(a for a,r in roles.items() if r=='test'))
    np.save(cache(c)/'states.npy', np.asarray(states,np.float32))
    write_json(cache(c)/'rows.json', rows); write_json(cache(c)/'aliases.json', aliases)
    write_json(root(c)/'technical/readout-plan.json', dict(identity=digest([p['identity'],partition]),partition=partition))
    summary=[]
    for role in ('train','test'):
        selected=[r for r in rows if ('train' if r['role']=='train' else 'test')==role]
        for target,key in [('any','first_step'),('persistent','persistent_first_step')]:
            for horizon in c['horizons_ps']:
                y=np.array([0<r[key]<=round(horizon/.1) for r in selected])
                w=np.array([r['weight'] for r in selected])
                summary.append(dict(role=role,target=target,horizon_ps=horizon,rows=len(y),
                    positives=int(y.sum()),ancestors=len({r['ancestor'] for r in selected}),
                    prevalence=float(y.mean()),ancestor_weighted_prevalence=float(np.average(y,weights=w))))
    write_metric_rows(summary,root(c),family=FAMILY,name='cohort-coverage')
    write_json(cache(c)/'complete.json', dict(identity=p['identity'],rows=len(rows),windows=len(coverage),
        duplicates=len(aliases),files={n:sha(cache(c)/n) for n in ('states.npy','rows.json','aliases.json')}))
    return summary


def collect(c):
    p=read(root(c)/'technical/plan.json');cache(c).mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(extract,c,p,t) for t in p['tasks']]
        for i,future in enumerate(as_completed(futures)):
            done=future.result()
            write_json(root(c)/'technical/collection-progress.json',dict(completed=i+1,total=len(futures),last=done))
            print('appearance window',i+1,len(futures),done['task'],done['at_risk'],round(done['seconds'],1),flush=True)
    seal(c,p)


def load(c):
    done=read(cache(c)/'complete.json')
    if done['identity']!=read(root(c)/'technical/plan.json')['identity'] or any(sha(cache(c)/n)!=h for n,h in done['files'].items()):
        raise ValueError('Changed sealed first-appearance observations')
    raw=read(cache(c)/'rows.json')
    rows={k:np.asarray([r[k] for r in raw]) for k in raw[0]}
    return np.load(cache(c)/'states.npy',mmap_mode='r'),rows,done
