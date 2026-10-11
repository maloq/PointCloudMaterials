"""Periodic Ta event cutouts for visual review, preserving native observations."""
import argparse
from collections import Counter
import gzip
import os
from pathlib import Path

import numpy as np
import yaml

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue, recorded_stage
from src.experiment_runner.metric_docs import check_metric_docs, write_metric_rows
from src.project_runtime.paths import REPO, resolve_path
from src.postprocessing.io import arrays
from src.postprocessing.protocols.shooting_data import read
from src.postprocessing.protocols.ta_content import strong_origins

FAMILY = 'ta_event_crops'


def prepare(c):
    root = resolve_path(c['audit']); p = read(root / 'technical/plan.json')
    if (digest({k:v for k,v in p.items() if k != 'identity'}) != p['identity'] or
            read(root / 'technical/complete.json')['identity'] != p['identity']):
        raise ValueError('Require the completed unchanged Ta audit')
    if c['include_historical'] or c['appearance_selection'] != 'weak_link_dependent_establishments':
        raise ValueError('This crop protocol covers the newly shot ambiguous appearances only')
    if c['dissolution_selection'] not in ('all_verified', 'strict_isolated'):
        raise ValueError('Unknown dissolution population')
    records, sources, evidence = [], [], []
    for s in p['sources']:
        if s['legacy']: continue
        folder = root / 'technical/sources' / s['id']
        done = read(folder / 'complete.json')
        if done['identity'] != digest(dict(plan=p['identity'], source=s['id'])):
            raise ValueError('Audit source identity changed')
        for name in ('graph.npz', 'events.json', 'transient-episodes.json', 'ptm-complete.json'):
            if sha(folder/name) != done['files'][name]: raise ValueError(f'Changed audit file: {folder/name}')
        evidence.append(dict(source=s['id'], files=done['files']))
        with np.load(folder / 'graph.npz') as g:
            origins = strong_origins(g)
        selected = []
        for e in read(folder / 'events.json'):
            if e['kind'] == 'parent_present_embryo_growth' and origins[e['birth_node']] > 0:
                selected.append(dict(name=f'appearing-event-{e["event_id"]:03d}', kind='appearing',
                    anchor_time_ps=s['times_ps'][e['birth_frame']], reference_frame=e['birth_frame'],
                    centroid_A=e['centroid_A'], anchor_atom_id=None,
                    anchor_definition='start of the confirmed >=64-atom establishment streak',
                    earliest_strong_ancestor_ps=s['times_ps'][int(origins[e['birth_node']])],
                    status='ambiguous appearance; frozen label remains parent_present_embryo_growth', label=e))
        for e in read(folder / 'transient-episodes.json'):
            if e['terminal_status'] != 'dissolved': continue
            if c['dissolution_selection'] == 'strict_isolated' and not e['brief_failed_candidate']: continue
            selected.append(dict(name='disappearing-'+e['episode'].split(':')[-1], kind='disappearing',
                anchor_time_ps=s['times_ps'][e['last_frame']+1], reference_frame=e['first_frame'],
                centroid_A=e['centroid_A'], anchor_atom_id=e['anchor_atom'],
                anchor_definition='first audit observation after the last tracked cluster; confirmed noncrystalline core',
                disappearance_bracket_ps=[s['times_ps'][e['last_frame']],s['times_ps'][e['last_frame']+1]],
                confirmation_time_ps=s['times_ps'][e['dissolution_frame']],
                status='verified dissolution; origin and lineage reliability retained separately', label=e))
        if selected:
            sources.append(s)
            records += [dict(e, source=s['id'], parent=s['parent_id']) for e in selected]
    counts = Counter(r['kind'] for r in records)
    if counts != dict(appearing=c['expected_appearance_events'], disappearing=c['expected_dissolution_events']):
        raise ValueError(f'Unexpected event scope: {counts}')
    plan = dict(config=c, audit_identity=p['identity'], sources=sources, events=records,
                evidence=evidence, implementation=check_metric_docs(family=FAMILY)[FAMILY]['files'])
    plan['identity'] = digest(plan)
    path = resolve_path(c['output']) / 'technical/plan.json'
    if path.exists() and read(path) != plan: raise ValueError('Crop plan changed; use a new output revision')
    write_json(path, plan)
    return plan


def load(c):
    p = read(resolve_path(c['output']) / 'technical/plan.json')
    if (p['config'] != c or digest({k:v for k,v in p.items() if k != 'identity'}) != p['identity']
            or p['implementation'] != check_metric_docs(family=FAMILY)[FAMILY]['files']):
        raise ValueError('Crop producer/plan changed; resume the frozen bundle')
    return p


def displacement(points, center, box):
    d = np.asarray(points, np.float64)-center
    return d-box*np.rint(d/box)


def nearest(a, frame, center, box, chunk):
    best, row = np.inf, None
    for start in range(0,len(a['atom_ids']),chunk):
        delta = displacement(a['positions'][frame,start:start+chunk],center,box)
        d2 = np.einsum('ij,ij->i',delta,delta); k = int(np.argmin(d2))
        if d2[k] < best: best,row = float(d2[k]),start+k
    return row


def crop(points, center, box, radius):
    # A cheap periodic x-slab eliminates most atoms before the full 3D calculation.
    dx = points[:,0].astype(np.float64)-center[0]; dx -= box[0]*np.rint(dx/box[0])
    ids = np.flatnonzero(np.abs(dx) <= radius)
    d = displacement(points[ids],center,box)
    keep = np.einsum('ij,ij->i',d,d) <= radius**2
    return ids[keep], d[keep].astype(np.float32)


def destination(c,e):
    return resolve_path(c['output']) / 'crops' / e['kind'] / (e['source']+'__'+e['name'])


def process(c,p,s,progress):
    if len(s['parts']) != 1: raise ValueError('This Ta crop source must have one continuous native trajectory')
    part = s['parts'][0]; a = arrays(part['path'],part['manifest_sha256'])
    times = a['timesteps'].astype(np.float64)*part['timestep_fs']/1000
    if not np.allclose(np.diff(times),c['cadence_ps'],rtol=0,atol=1e-8):
        raise ValueError('Native Ta cadence differs from requested0.1ps')
    np.testing.assert_allclose(times[[f[1] for f in s['frames']]],s['times_ps'],rtol=0,atol=1e-8)
    folder = resolve_path(c['audit'])/'technical/sources'/s['id']
    ptm = read(folder/'ptm-complete.json')
    pending = []
    for e in [r for r in p['events'] if r['source'] == s['id']]:
        out = destination(c,e); out.mkdir(parents=True,exist_ok=True)
        identity = digest(dict(plan=p['identity'],event=e))
        if (out/'complete.json').exists():
            old = read(out/'complete.json')
            if old['identity'] != identity or any(sha(out/n)!=h for n,h in old['files'].items()):
                raise ValueError('Changed completed crop')
            continue
        reference = s['frames'][e['reference_frame']][1]
        box = a['box_high'][reference].astype(float)-a['box_low'][reference].astype(float)
        center = np.asarray(e['centroid_A'],float)
        if e['anchor_atom_id'] is None:
            anchor = nearest(a,reference,center,box,c['chunk_atoms'])
        else:
            anchor = int(np.searchsorted(a['atom_ids'],e['anchor_atom_id']))
            if a['atom_ids'][anchor] != e['anchor_atom_id']: raise ValueError('Missing tracked center atom')
        offset = displacement(center,a['positions'][reference,anchor],box)/box
        requested = (e['anchor_time_ps']-c['before_ps'],e['anchor_time_ps']+c['after_ps'])
        selected = np.flatnonzero((times >= requested[0]-1e-8)&(times <= requested[1]+1e-8))
        buffers = {k:[] for k in ('positions','atom_ids','ptm','tracked_center','counts','centers','box_low','box_high')}
        pending.append(dict(event=e,out=out,identity=identity,anchor=anchor,offset=offset,
            frames=selected,buffer=buffers,requested=requested,
            writer=gzip.open(out/'trajectory.xyz.gz','wt',compresslevel=3)))
    audit_frames = {raw:i for i,(_,raw) in enumerate(s['frames'])}
    try:
        native = sorted({int(f) for q in pending for f in q['frames']})
        for number,frame in enumerate(native):
            active = [q for q in pending if frame in q['frames']]
            box = a['box_high'][frame].astype(float)-a['box_low'][frame].astype(float)
            centers = [np.mod(a['positions'][frame,q['anchor']].astype(float)+box*q['offset'],box) for q in active]
            pieces = [[] for q in active]; row_pieces = [[] for q in active]
            for start in range(0,s['atom_count'],c['chunk_atoms']):
                points = a['positions'][frame,start:start+c['chunk_atoms']]
                for k,center in enumerate(centers):
                    ids, delta = crop(points,center,box,c['radius_A'])
                    pieces[k].append(delta);row_pieces[k].append(ids+start)
            full_labels = None
            if frame in audit_frames:
                audit_frame = audit_frames[frame]; name=f'ptm-{audit_frame:04d}-{audit_frame+1:04d}.npz'
                if sha(folder/name) != ptm['files'][name]: raise ValueError('Changed full-cell PTM frame')
                with np.load(folder/name) as z: full_labels=z['labels'][0]
            for k,q in enumerate(active):
                xyz=np.concatenate(pieces[k]);rows=np.concatenate(row_pieces[k]);ids=a['atom_ids'][rows]
                labels=np.full(len(rows),-1,np.int8) if full_labels is None else full_labels[rows].astype(np.int8)
                mark=(rows==q['anchor']).astype(np.uint8)
                if mark.sum()!=1 or np.linalg.norm(xyz,axis=1).max() > c['radius_A']+1e-5:
                    raise ValueError('Crop center/radius validation failed')
                if len(ids)!=len(np.unique(ids)) or not np.isfinite(xyz).all(): raise ValueError('Invalid cropped geometry')
                b=q['buffer']
                for key,value in [('positions',xyz),('atom_ids',ids),('ptm',labels),('tracked_center',mark),
                                  ('counts',len(ids)),('centers',centers[k]),('box_low',a['box_low'][frame]),('box_high',a['box_high'][frame])]:
                    b[key].append(value)
                f=q['writer']; f.write(str(len(ids))+'\n')
                length=2*c['radius_A']; relative=times[frame]-q['event']['anchor_time_ps']
                f.write(f'Lattice="{length} 0 0 0 {length} 0 0 0 {length}" pbc="F F F" '
                    'Properties=species:S:1:pos:R:3:id:I:1:tracked_center:I:1:ptm_type:I:1 '
                    f'Time={times[frame]:.8f} relative_time_ps={relative:.8f} ptm_available={int(full_labels is not None)}\n')
                # Coordinates translated to [0,40]^3; the sphere center is(20,20,20).
                for i,position,m,label in zip(ids,xyz+c['radius_A'],mark,labels):
                    f.write(f'Ta {position[0]:.7f} {position[1]:.7f} {position[2]:.7f} {int(i)} {int(m)} {int(label)}\n')
            if number % 10 == 0:
                progress.update(source=s['id'],frames_completed=number+1,frames_total=len(native))
    finally:
        for q in pending:q['writer'].close()
    for q in pending:
        out,b,e = q['out'],q['buffer'],q['event']; selected=q['frames']; relative=times[selected]-e['anchor_time_ps']
        np.savez_compressed(out/'trajectory.npz', positions=np.concatenate(b['positions']),atom_ids=np.concatenate(b['atom_ids']),
            ptm_type=np.concatenate(b['ptm']),tracked_center=np.concatenate(b['tracked_center']),
            offsets=np.r_[0,np.cumsum(b['counts'])], time_ps=times[selected],relative_time_ps=relative,
            source_frame=selected,timesteps=a['timesteps'][selected],center_original_A=np.asarray(b['centers']),
            original_box_low_A=np.asarray(b['box_low']),original_box_high_A=np.asarray(b['box_high']))
        metadata=dict(identity=q['identity'],plan=p['identity'],event=e,source=s,
            radius_A=c['radius_A'],requested_window_ps=list(q['requested']),
            available_window_ps=[float(times[selected[0]]),float(times[selected[-1]])],
            available_before_ps=float(-relative[0]),available_after_ps=float(relative[-1]),
            missing_before_ps=float(max(0,c['before_ps']+relative[0])),missing_after_ps=float(max(0,c['after_ps']-relative[-1])),
            full_window=bool(-relative[0]>=c['before_ps']-1e-8 and relative[-1]>=c['after_ps']-1e-8),
            frames=len(selected),center_atom_id=int(a['atom_ids'][q['anchor']]),center_fractional_offset=q['offset'].tolist(),
            coordinates='NPZ centered minimum-image displacement; XYZ translates by+20A; no crop PBC',
            temporal_membership='all atoms inside20A in each frame; identities preserved; variable atom count',
            precision='float32 local coordinates derived from existing float16 full-cell positions; no precision recovery',
            ptm='original full-cell labels every0.5ps; -1 means not evaluated at native intermediate0.1ps frames',
            use='retrospectively event-centered visual review; no predictor inputs or train/test assignment')
        write_json(out/'metadata.json',metadata)
        validate(out,metadata)
        write_json(out/'complete.json',dict(identity=q['identity'],frames=len(selected),
            files={n:sha(out/n) for n in ('trajectory.npz','trajectory.xyz.gz','metadata.json')},validated_xyz_frames=len(selected)))
        print(dict(source=s['id'],event=e['name'],frames=len(selected),output=str(out)),flush=True)
    arrays.cache_clear()


def validate(out,metadata):
    # Round-trip every production frame using the independent XYZ reader.
    from ase.io import iread
    with np.load(out/'trajectory.npz') as a:
        n=0
        for n,atoms in enumerate(iread(out/'trajectory.xyz.gz',index=':',format='extxyz'),1):
            k=n-1;left,right=a['offsets'][k:k+2]
            np.testing.assert_array_equal(atoms.arrays['id'],a['atom_ids'][left:right])
            np.testing.assert_allclose(atoms.positions-metadata['radius_A'],a['positions'][left:right],rtol=0,atol=3e-6)
            np.testing.assert_array_equal(atoms.arrays['ptm_type'],a['ptm_type'][left:right])
            if atoms.pbc.any():raise ValueError('Local crop must not impose artificial periodicity')
        if n!=metadata['frames']:raise ValueError('Incomplete XYZ export')


def collect(c):
    p=load(c); rows=[]
    for e in p['events']:
        out=destination(c,e);r=read(out/'complete.json');m=read(out/'metadata.json')
        if r['identity']!=digest(dict(plan=p['identity'],event=e)) or any(sha(out/n)!=h for n,h in r['files'].items()):
            raise ValueError('Incomplete or changed crop')
        rows.append(dict(source=e['source'],parent=e['parent'],event=e['name'],kind=e['kind'],status=e['status'],
            original_origin=e['label'].get('origin',e['label'].get('kind')),
            strict_failed_candidate=e['label'].get('brief_failed_candidate',False),
            anchor_time_ps=e['anchor_time_ps'],frames=m['frames'],radius_A=m['radius_A'],
            available_before_ps=m['available_before_ps'],available_after_ps=m['available_after_ps'],
            missing_before_ps=m['missing_before_ps'],missing_after_ps=m['missing_after_ps'],full_window=m['full_window'],
            xyz=str(out/'trajectory.xyz.gz'),npz=str(out/'trajectory.npz')))
    root=resolve_path(c['output']);write_metric_rows(rows,root/'analyses/coverage-v1',family=FAMILY,name='crop-index')
    lines=['# Ta nucleus review crops','',
        f'{len(rows)} separate event movies: eight ambiguous appearances and28 verified disappearances. Labels unchanged.',
        '20 Angstrom radius spheres at native0.1ps spacing. Requested window:18ps before and18ps after each event.',
        'Sources are24ps long: every crop is truncated, with exact available/missing durations in its metadata and index. No padding, interpolation, source stitching or new MD.',
        'Open each trajectory.xyz.gz in OVITO or ASE. trajectory.npz preserves variable-length frame arrays, IDs, original boxes, crop centers and timelines.',
        'Appearing anchors mark the start of the confirmed64-atom establishment streak, not the first crystalline atom. Disappearing anchors mark the first audit observation after the last tracked cluster, before the later disappearance-confirmation timestamp.',
        'The sphere follows a tracked atom plus a constant fractional offset, chosen to match the recorded event centroid at its reference frame. Atom membership changes across frames. Crop boundaries are nonperiodic.',
        'ptm_type contains original full-cell labels only at0.5ps audit observations; -1 at intermediate frames means unavailable, not liquid. Recomputing PTM on a small crop can introduce boundary effects.',
        'These future-selected centers and labels are for review, not causal prediction inputs. Original simulation files are untouched.','',
        '[Index with paths and window coverage](analyses/coverage-v1/tables/crop-index.csv)','',
        '| Event type | Folder |','|---|---|']
    lines += [f'| {e["kind"]} | [{e["source"]} / {e["name"]}]({destination(c,e).relative_to(root)}/trajectory.xyz.gz) |' for e in p['events']]
    (root/'README.md').write_text('\n'.join(lines)+'\n')
    write_json(root/'technical/complete.json',dict(identity=p['identity'],crops=len(rows),full_windows=sum(r['full_window'] for r in rows)))
    print(dict(crops=len(rows),output=str(root)),flush=True)


def submit(path):
    c=read(path);p=prepare(c);technical=resolve_path(c['output'])/'technical'
    if (technical/'launch.json').exists():raise ValueError('Already submitted')
    bundle=ExecutionBundle.freeze(REPO,technical/'code',c,directories=('src','configs','docs/metrics'))
    machine=yaml.safe_load(Path(os.environ.get('PCM_MACHINE_CONFIG',REPO/'machine.local.yaml')).read_text())
    machine['catalog']=str(bundle.root/'configs/datasets.json');(technical/'machine.yaml').write_text(yaml.safe_dump(machine))
    record=dict(identity=p['identity'],code=str(bundle.root),jobs={},crops=len(p['events']))
    queue=SlurmQueue(technical,bundle,'src.postprocessing.visualization.crops',
        dict(PCM_PROJECT_ROOT=str(bundle.root),PCM_MACHINE_CONFIG=str(technical/'machine.yaml'),
             OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),technical/'launch.json',record,'TA-CROPS')
    with queue.submission():
        job=queue.submit('worker',[f'--array=0-{c["workers"]-1}','--cpus-per-task=1','--mem=6G','--time=06:00:00'])
        queue.submit('collect',['--cpus-per-task=1','--mem=2G','--time=00:30:00'],dependency='afterok:'+job)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('stage',choices=('prepare','submit','worker','collect'))
    parser.add_argument('--config',required=True);args=parser.parse_args();c=read(args.config)
    if args.stage=='prepare':print(prepare(c)['identity'])
    elif args.stage=='submit':print(submit(args.config))
    else:
        index=int(os.environ.get('SLURM_ARRAY_TASK_ID','0'))
        with recorded_stage(resolve_path(c['output'])/f'technical/{args.stage}-{index}.json',job=os.environ.get('SLURM_JOB_ID')) as progress:
            if args.stage=='collect':collect(c)
            else:
                p=load(c)
                for s in p['sources'][index::c['workers']]:process(c,p,s,progress)
