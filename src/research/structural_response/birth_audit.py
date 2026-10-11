"""Matched-cohort leakage audit and interpretable frozen birth-readout controls."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows, check_metric_docs
from src.project_runtime.paths import resolve_path

FAMILY = 'motion_birth_attribution'


def read(path):
    return json.loads(Path(path).read_text())


def setup(c):
    parent = resolve_path(c['parent'])
    recipe = read(parent/'technical/code/config.json')
    cache = resolve_path(recipe['birth_evaluation']['cache'])
    receipt = read(cache/'complete.json')
    if receipt['cohort'] != c['cohort_identity']:
        raise ValueError('Birth cohort changed')
    raw = read(cache/'rows.json')
    rows = {k: np.asarray([r[k] for r in raw]) for k in raw[0]}
    return parent, recipe, cache, raw, rows, resolve_path(c['output'])


def kinematics(x):
    q, v = x[:, :3], x[:, 3:]
    relative = v[1:]-v[:1]
    radius = np.linalg.norm(q[1:], axis=1)
    radial = np.sum(relative*q[1:], axis=1)/radius
    speed = np.linalg.norm(v, axis=1)
    local = v[:13]-v[:13].mean(0)
    values = [speed[0], np.mean(speed**2), np.std(speed),
              np.mean(relative**2), np.mean(radial), np.std(radial),
              np.mean(radial[:12]), np.mean(local**2), np.linalg.norm(v.mean(0))]
    names = ['center_speed', 'mean_speed_squared', 'speed_std', 'relative_velocity_power',
             'radial_mean', 'radial_std', 'first_shell_radial_mean',
             'first_shell_velocity_variance', 'collective_speed']
    return values, ['kinematics/'+n for n in names]


def source_packet(args):
    c, item, indices = args
    from src.postprocessing.io import arrays
    from src.postprocessing.io import geometry
    from src.postprocessing.extraction import Frames
    from src.research.liquid_predictability.descriptors import patch_descriptors
    _, recipe, cache, raw, _, out = setup(c)
    dest = out/'technical/sources'/item['source_id']
    dest.mkdir(parents=True, exist_ok=True)
    identity = digest([c, item['source_id'], indices, sha(Path(__file__))])
    if (dest/'complete.json').exists():
        record = read(dest/'complete.json')
        if record['identity'] != identity or sha(dest/'features.npz') != record['sha256']:
            raise ValueError('Changed audit source packet')
        return record
    x = np.load(cache/'states.npy', mmap_mode='r')
    cohort = read(resolve_path(c['parent'])/'technical/birth-cohort.json')
    events = {e['event_id']: e for e in cohort['events']}
    folder = Path(item['label_folder'])
    if sha(folder/'events.json') != item['label_files']['events.json']:
        raise ValueError('Birth labels changed')
    upstream = read(folder/'events.json')
    lineage = read(folder.parents[3]/'plan.json')
    pp = resolve_path(lineage['config']['parent_output'])/'technical/plan.json'
    if sha(pp) != lineage['parent_plan_sha256']:
        raise ValueError('Extraction plan changed')
    extraction = read(pp)
    s = item['source']
    original = next(a for a in extraction['sources'] if a['input_identity'] == s['input_identity'])
    access = Frames(extraction, original, 'ptm')
    if extraction['config']['minimum_tracked_atoms'] != 4:
        raise ValueError('Expected four-atom minimum in original component bank')
    native = arrays(s['parts'][0]['path'], s['parts'][0]['manifest_sha256'])
    atom_ids = native['atom_ids']
    velocities = []
    for part in s['parts']:
        m = read(Path(part['path'])/'manifest.json')
        velocities.append(np.load(Path(part['path'])/m['arrays']['velocities']['file'], mmap_mode='r'))
    observations = defaultdict(list)
    leads = recipe['birth_evaluation']['leads_ps']
    view, times = np.asarray(s['views']['010']), np.asarray(s['times_ps'])
    for index in indices:
        row = raw[index]
        event = upstream[events[row['event_view']]['event_row']]
        member = row['atom'] in event['first_members']
        if member != bool(row['label']):
            raise ValueError('Founder/control identity mismatch')
        slot = int(np.searchsorted(view, event['first_frame']))
        for j, lead in enumerate(leads):
            frame = int(view[slot-int(round(lead/.1))])
            if abs(times[event['first_frame']]-times[frame]-lead) > 1e-8:
                raise ValueError('Noncausal or incorrect lead')
            observations[frame].append((index, j))
    physical = np.empty((len(indices), len(leads), 455), np.float32)
    auxiliary = np.empty((len(indices), len(leads), 6), np.float64)
    where = {index: k for k, index in enumerate(indices)}
    names = None
    max_error = 0.
    for frame, requests in sorted(observations.items()):
        q, box = geometry(s, frame)
        centers = np.searchsorted(atom_ids, [raw[i]['atom'] for i, _ in requests])
        if not np.array_equal(atom_ids[centers], [raw[i]['atom'] for i, _ in requests]):
            raise ValueError('Wrong source atom mapping')
        _, neighbors = cKDTree(q, boxsize=box).query(q[centers], k=80, workers=1)
        delta = q[neighbors]-q[centers, None]
        delta -= box*np.rint(delta/box)
        pi, rf = s['frames'][frame]
        state = np.concatenate((delta.astype(np.float32), np.asarray(velocities[pi][rf, neighbors], np.float32)), -1)
        labels = access.labels(frame)
        sizes = np.bincount(labels)
        sizes[0] = 0
        large = sizes[labels] >= 64
        distance = cKDTree(q[large], boxsize=box).query(q[centers])[0] if large.any() else np.full(len(centers), np.inf)
        for k, (index, j) in enumerate(requests):
            error = float(np.max(np.abs(state[k]-x[index, j])))
            max_error = max(error, max_error)
            if error != 0:
                raise ValueError(f'Raw Q,V reconstruction differs: {s["id"]}/{frame}/{index}: {error}')
            desc, cols = patch_descriptors(state[k, :, :3])
            motion, mn = kinematics(state[k])
            # Geometric quantization artifacts only; no absolute-coordinate or metadata inputs.
            grid = [float(np.mean(np.abs(state[k, 1:, :3]/step-np.rint(state[k, 1:, :3]/step)) < 1e-5))
                    for step in (1/64, 1/32, 1/16, 1/8)]
            current_names = cols+mn+[f'precision/grid_{step:g}A' for step in (1/64, 1/32, 1/16, 1/8)]
            if names is None:
                names = current_names
            if names != current_names or len(names) != physical.shape[-1]:
                raise ValueError('Descriptor schema changed')
            physical[where[index], j] = np.r_[desc, motion, grid]
            local = sizes[labels[neighbors[k]]]
            auxiliary[where[index], j] = [float(np.mean(local >= 4)), float(sizes[labels[centers[k]]]),
                float(local.max()), distance[k], float(large.mean()), float(np.mean(local >= 64))]
    np.savez_compressed(dest/'features.npz', indices=indices, values=physical, auxiliary=auxiliary)
    write_json(dest/'columns.json', names)
    record = dict(identity=identity, source=s['id'], rows=len(indices), observations=len(indices)*len(leads),
                  max_reconstruction_error=max_error, sha256=sha(dest/'features.npz'))
    write_json(dest/'complete.json', record)
    arrays.cache_clear()
    return record


def prepare(c):
    parent, recipe, cache, raw, rows, out = setup(c)
    check_metric_docs(family=FAMILY)
    (out/'technical').mkdir(parents=True, exist_ok=True)
    for name, expected in read(cache/'complete.json')['files'].items():
        if sha(cache/name) != expected:
            raise ValueError('Prepared birth arrays changed')
    train = rows['role'] == 'train'
    checks = []
    def check(name, passed, **evidence):
        checks.append(dict(check=name, passed=bool(passed), evidence=evidence))
        write_json(out/'technical/integrity.json', checks)
        if not passed:
            raise ValueError(f'Audit failed: {name}: {evidence}')
    check('ancestor_disjoint', not(set(rows['ancestor'][train]) & set(rows['ancestor'][~train])),
          training=len(set(rows['ancestor'][train])), heldout=len(set(rows['ancestor'][~train])))
    check('exact_input_hash_disjoint', not(set(rows['input_hash'][train]) & set(rows['input_hash'][~train])))
    pairs = defaultdict(list)
    for i, pair in enumerate(rows['pair']): pairs[pair].append(i)
    matched = all(len(ids) == 5 and rows['label'][ids].sum() == 1 and
                  all(len(set(rows[k][ids])) == 1 for k in ('source', 'event_view', 'ancestor', 'role', 'weight'))
                  for ids in pairs.values())
    check('same_source_frame_matched_controls', matched, pairs=len(pairs))
    # Exact coordinate-only overlap is checked independently of velocity and at each lead.
    states = np.load(cache/'states.npy', mmap_mode='r')
    for j, lead in enumerate(recipe['birth_evaluation']['leads_ps']):
        hashes = [hashlib.sha256(np.ascontiguousarray(a[:, :3]).tobytes()).hexdigest() for a in states[:, j]]
        check(f'coordinate_overlap_{lead:g}ps', not({h for h,t in zip(hashes,train) if t} & {h for h,t in zip(hashes,train) if not t}))
    # The broad training manifest contains train ancestors only; inspect the actual producer shards.
    packed = read(resolve_path(recipe['loader']['packed_cache'])/'manifest.json')
    spatial_anc = {s['task']['lineage'] for s in packed['shards']}
    check('spatial_pretraining_disjoint', not(spatial_anc & set(rows['ancestor'][~train])),
          structural_shards=len(packed['shards']), arrays_opened=False)
    cohort = read(parent/'technical/birth-cohort.json')
    active = {s['source_id']:s for s in cohort['sources']}
    groups = defaultdict(list)
    for i, row in enumerate(raw): groups[row['source_id']].append(i)
    with ProcessPoolExecutor(max_workers=c['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
        tasks = [pool.submit(source_packet, (c,active[k],indices)) for k,indices in groups.items()]
        for n, future in enumerate(as_completed(tasks),1):
            record = future.result()
            write_json(out/'technical/progress.json', dict(stage='raw_and_physical_audit', sources=n, total=len(tasks), last=record))
            print('audit source',n,len(tasks),record['source'],flush=True)
    features = np.empty((len(raw),3,455),np.float32)
    auxiliary = np.empty((len(raw),3,6),np.float64)
    for key in groups:
        folder = out/'technical/sources'/key
        with np.load(folder/'features.npz') as a:
            features[a['indices']] = a['values'];auxiliary[a['indices']] = a['auxiliary']
    names = read(folder/'columns.json')
    np.savez_compressed(out/'technical/physical.npz',values=features,auxiliary=auxiliary)
    write_json(out/'technical/columns.json', names)
    check('all_raw_observations_reconstructed', True, observations=len(raw)*3, max_error=0.)
    write_json(out/'technical/prepared.json',dict(rows=len(raw),physical_sha256=sha(out/'technical/physical.npz'),
        auxiliary_columns=['component_ge4_fraction','center_component_size','max_component_in_patch',
                           'distance_component_ge64_A','global_component_ge64_fraction','local_component_ge64_fraction'],
        neighborhood_filter=False,cohort=c['cohort_identity']))


def groups(columns):
    result = {f:np.array([i for i,n in enumerate(columns) if n.startswith(f+'/')])
              for f in ('geometry','bond_order','cna','tda','kinematics','precision')}
    result['rich'] = np.arange(442)
    result['radial'] = np.array([i for i,n in enumerate(columns) if n.startswith('geometry/') and
                                not any(k in n for k in ('pair_hist','angle','shape'))])
    result['q6'] = np.array([columns.index('bond_order/l6_q')])
    result['qbar6'] = np.array([columns.index('bond_order/l6_qbar')])
    result['order_small'] = np.array([columns.index('bond_order/'+n) for n in
        ('l4_q','l6_q','l4_qbar','l6_qbar','l6_coherence_mean','l6_coherence_q90')])
    return result


def measured(rows, probability, **context):
    from src.research.birth_prediction.fit import scores
    from src.research.birth_prediction.analysis import source_draws, bootstrap_scores
    mask = rows['role'] != 'train'
    y,w,p = rows['label'][mask],rows['weight'][mask],probability[mask]
    _,slot,draws = source_draws(rows['ancestor'][mask],500,20261007)
    val = scores(y,p,w);boot,_ = bootstrap_scores(y,p,w,slot,draws,None)
    return [dict(context,metric=k,value=val[k],lower=float(np.quantile(boot[k],.025)),
                 upper=float(np.quantile(boot[k],.975))) for k in ('ap','auroc','nll','brier')]


def controls(c):
    from src.research.response_dense_birth.pooled import fit
    parent, recipe, _, _, rows, out = setup(c)
    saved = np.load(out/'technical/physical.npz');x,aux = saved['values'],saved['auxiliary']
    columns = read(out/'technical/columns.json');families = groups(columns)
    plan = read(parent/'analyses/masked_teacher-derivative_joint/analyses/birth-prediction/technical/readout-plan.json')
    fit_config = dict(recipe['birth_evaluation'], seed=c['seed'], output=str(out))
    fit_config['fit_threads'] = 2
    score_rows, distribution = [],[]
    for j,lead in enumerate(recipe['birth_evaluation']['leads_ps']):
        for space in c['physical_controls']:
            indices = families[space]
            for kind in ('linear','catboost'):
                model = dict(name='physical',schema_sha256=sha(out/'technical/columns.json'))
                fit(fit_config,dict(config=fit_config),rows,x[:,j,indices],model,space,lead,kind,plan=plan)
                path=out/'analyses/fits-v1/physical'/space/f'lead-{lead:g}ps'/kind/'technical/predictions.npz'
                with np.load(path) as values:probability=values['probability']
                score_rows.extend(measured(rows,probability,features=space,lead_ps=lead,readout=kind))
                write_metric_rows(score_rows,out,family=FAMILY,name='physical-controls')
        for column in ('bond_order/l6_q','bond_order/l6_qbar','bond_order/l6_coherence_mean',
                       'geometry/neighbor_distance_12','kinematics/relative_velocity_power'):
            for label in (0,1):
                ix=(rows['role']!='train') & (rows['label']==label)
                distribution.append(dict(lead_ps=lead,feature=column,label=label,
                    mean=float(np.average(x[ix,j,columns.index(column)],weights=rows['weight'][ix]))))
        for label in (0,1):
            ix=(rows['role']!='train') & (rows['label']==label);w=rows['weight'][ix]
            for name,v in [('any_component_ge4',aux[ix,j,0]>0),('center_in_component_ge8',aux[ix,j,1]>=8),
                           ('center_in_component_ge64',aux[ix,j,1]>=64),('any_component_ge64',aux[ix,j,2]>=64),
                           ('component_ge4_fraction',aux[ix,j,0]),('component_ge64_within8A',aux[ix,j,3]<=8)]:
                distribution.append(dict(lead_ps=lead,feature=name,label=label,mean=float(np.average(v,weights=w))))
    write_metric_rows(distribution,out,family=FAMILY,name='present-state-distributions')
    write_json(out/'technical/controls-complete.json',dict(rows=len(score_rows)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('prepare','controls'))
    p.add_argument('--config',required=True);a=p.parse_args();c=read(a.config)
    (prepare if a.stage=='prepare' else controls)(c)


if __name__=='__main__':main()
