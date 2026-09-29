"""Frozen synthetic-label assays and paired archived-relaxation observations."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import softmax
from scipy.spatial import cKDTree

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from .data import config, population
from .descriptor_data import load, parent, _block
from .descriptor_fit import targets
from .descriptors import summarize_context

ROLES = ('train', 'selection', 'calibration', 'test')


def seal_cache(root, x, rows, columns, geometry, provenance):
    """A derived cohort never changes roles, IDs, or targets implicitly."""
    root.mkdir(parents=True, exist_ok=True)
    np.save(root/'features.npy', x)
    np.savez(root/'rows.npz', **rows)
    write_json(root/'columns.json', columns)
    write_json(root/'geometry.json', geometry)
    record = dict(provenance=provenance, rows={r: int((rows['role'] == r).sum()) for r in ROLES},
                  train_sources=np.unique(rows['source'][rows['role'] == 'train']).tolist(),
                  files={n: sha(root/n) for n in ('features.npy', 'rows.npz', 'columns.json', 'geometry.json')})
    record['identity'] = digest(record)
    write_json(root/'manifest.json', record)
    return record


def freeze(c):
    root = resolve_path(c['output']); tech = root/'technical'; tech.mkdir(parents=True, exist_ok=True)
    if (tech/'data-plan.json').exists():
        saved = config(tech/'data-plan.json')
        if saved['config'] != c: raise ValueError('Control plan configuration changed')
        return saved
    pc = config(resolve_path(c['descriptor_parent']))
    x, rows, columns, descriptor_manifest = load(pc)
    original, manifest, plan, meta, _ = population(parent(pc))
    ids = rows['ids']
    geom_root = resolve_path(c['cache'])/'original'; geom_root.mkdir(parents=True, exist_ok=True)
    np.savez(geom_root/'index.npz', indices=meta['indices'][ids], actual=meta['actual'][ids])
    banks = [dict(path=str(original/'sources'/str(s['source'])/'positions.npy'),
                  rows=s['patches'], sha256=s['files']['positions.npy']) for s in manifest['sources']]
    geometry = dict(index=str(geom_root/'index.npz'), index_sha256=sha(geom_root/'index.npz'), banks=banks)
    write_json(geom_root/'geometry.json', geometry)
    cells = {}; releases = []
    for name in c['relaxation_plans']:
        p = resolve_path(name); release = config(p); releases.append(dict(path=str(p), sha256=sha(p)))
        for folder in sorted((resolve_path(release['config']['cache'])/'cells').iterdir()):
            f = folder/'complete.json'
            if not f.exists(): continue
            record = config(f)
            if record['identity'] != release['identity']: raise ValueError(f'Changed relaxation receipt {f}')
            key = (record['task']['source'], record['task']['frame'])
            cells.setdefault(key, dict(source=key[0], frame=key[1], path=str(f), sha256=sha(f), archive=record['archive']))
    chosen = np.array([i for i, old in enumerate(ids) if (int(meta['source'][old]), int(meta['frame'][old])) in cells], dtype=np.int64)
    if c.get('require_full_coverage',False) and len(chosen)!=len(ids):
        raise ValueError(f'Full study requires every original row: {len(chosen)}/{len(ids)} available; finish relaxation first')
    if any(not np.any(rows['role'][chosen] == r) for r in ROLES): raise ValueError('Relaxed intersection loses an entire source role')
    np.save(tech/'paired-parent-rows.npy', chosen)
    used = {(int(meta['source'][ids[i]]), int(meta['frame'][ids[i]])) for i in chosen}
    record = dict(config=c, descriptor_identity=descriptor_manifest['identity'], original_geometry=geometry,
                  parent_manifest_sha256=sha(original/'manifest.json'), releases=releases,
                  paired_rows_file=str(tech/'paired-parent-rows.npy'), paired_rows_sha256=sha(tech/'paired-parent-rows.npy'),
                  cells=[cells[k] for k in sorted(used)],
                  counts={r: int((rows['role'][chosen] == r).sum()) for r in ROLES},
                  sources=plan['sources'], original_dataset=str(original))
    record['identity'] = digest(record); write_json(tech/'data-plan.json', record)
    return record


def synthetic(c):
    """Train-calibrated exponential tilts; exact oracle moments include within-bin noise."""
    if not c['signals']:return
    pc = config(resolve_path(c['descriptor_parent'])); x, rows, columns, manifest = load(pc)
    plan = config(resolve_path(c['output'])/'technical/data-plan.json')
    names = [v['name'] for v in columns]; train = np.flatnonzero(rows['role'] == 'train')
    w = rows['weights'][train]; w = w/w.sum(); edge = np.asarray(pc['distance_edges_A'], float)
    mid = np.r_[(edge[:-1]+edge[1:])/2, edge[-1]]
    second = np.r_[(edge[:-1]**2+edge[:-1]*edge[1:]+edge[1:]**2)/3, edge[-1]**2]
    p0 = np.bincount(targets(rows['target'][train], pc), weights=w, minlength=len(mid))+1e-8; p0 /= p0.sum()
    direction = (mid-p0@mid)/np.sqrt(p0@second-(p0@mid)**2)
    _, _, _, meta, _ = population(parent(pc))
    keys = np.c_[meta['source'][rows['ids']], meta['frame'][rows['ids']], meta['atom'][rows['ids']]]
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    uniforms = np.random.default_rng(c['synthetic_seed']).random((int(inv.max())+1, 2))[inv]
    del meta
    for arm in c['signals']:
        dest = resolve_path(c['cache'])/arm['name']; dest.mkdir(parents=True, exist_ok=True)
        if (dest/'manifest.json').exists(): continue
        if arm['kind'] == 'null':
            value = np.zeros(len(rows['ids'])); column = None; mu = 0.; scale = 1.
        else:
            column = c['signal_features'][arm['kind']]; j = names.index(column)
            value = np.asarray(x[:, j], float); mu = float(w@value[train]); scale = float(np.sqrt(w@(value[train]-mu)**2))
            if scale < 1e-8: raise ValueError(f'Constant synthetic generator feature: {column}')
            value = np.tanh((value-mu)/scale)
        def probability(eta, ix): return softmax(np.log(p0)[None]+eta*value[ix, None]*direction[None], axis=1)
        def oracle(eta):
            p = probability(eta, train); m = p@mid; s = p@second
            risk = float(w@(s-m*m)); baseline = float(w@s-(w@m)**2)
            return 1-np.sqrt(risk/baseline)
        goal = arm['oracle_rmse_gain']; eta = 0. if goal == 0 else brentq(lambda z: oracle(z)-goal, 0., 64., xtol=1e-10)
        p = probability(eta, np.arange(len(value)))
        # Common random numbers across signal strengths; independent per original
        # source/frame/query identity, including duplicate sampling-role entries.
        label = (uniforms[:, :1] > np.cumsum(p, axis=1)).sum(1).clip(max=len(mid)-1)
        y = np.full(len(label), edge[-1]); finite = label < len(mid)-1
        y[finite] = edge[label[finite]]+uniforms[finite, 1]*np.diff(edge)[label[finite]]
        derived = {k: v.copy() for k, v in rows.items()}; derived['target'] = y
        # Features are immutable shared inputs; avoid copying 4.6 GB per control.
        for name in ('features.npy', 'columns.json'):
            path=dest/name; target=resolve_path(pc['cache'])/name
            if path.is_symlink():
                if path.resolve()!=target.resolve():raise ValueError(f'Changed shared input {path}')
            else:path.symlink_to(target)
        np.savez(dest/'rows.npz', **derived); np.save(dest/'oracle.npy', p.astype(np.float32))
        write_json(dest/'geometry.json', plan['original_geometry'])
        theoretical = {}
        for role in ROLES:
            ix = np.flatnonzero(rows['role'] == role); ww = rows['weights'][ix]; ww = ww/ww.sum()
            m = p[ix]@mid; s = p[ix]@second; marginal = ww@p[ix]
            gain = 1-np.sqrt((ww@(s-m*m))/(ww@s-(ww@m)**2))
            mi = float(np.sum(ww[:, None]*p[ix]*(np.log(p[ix])-np.log(marginal))))
            theoretical[role] = dict(oracle_rmse_gain=float(gain), oracle_information_nats=mi)
        record = dict(parent_identity=manifest['identity'], signal=arm, generator_feature=column,
                      generator_mean=mu, generator_scale=scale, eta=float(eta), prior=p0.tolist(),
                      theoretical=theoretical, synthetic_not_physical_distance=True,
                      rows=manifest['rows'], files={n: sha(dest/n) for n in ('features.npy','columns.json','rows.npz','geometry.json','oracle.npy')})
        record['identity'] = digest(record); write_json(dest/'manifest.json', record)
        print(json.dumps(dict(signal=arm['name'], eta=eta, oracle=theoretical)), flush=True)


def paired_source(c, sid, pool):
    """Track hot patch/neighbor IDs through a verified existing full-cell quench."""
    from src.data.trajectories.shooting import ShootingBinaryTrajectory
    from src.project_runtime.paths import dataset_path
    from src.research.crystallization_origin.extract import frame_geometry, full_labels
    from src.research.crystallization_origin.ancestry import components
    from src.research.crystal_vector.data import context_atoms
    from src.research.structured_context.reuse import archived_positions
    plan = config(resolve_path(c['output'])/'technical/data-plan.json')
    dest = resolve_path(c['cache'])/'paired-sources'/str(sid); dest.mkdir(parents=True, exist_ok=True)
    if (dest/'complete.json').exists():
        done = config(dest/'complete.json')
        if done['plan_identity'] != plan['identity']: raise ValueError('Paired preparation identity changed')
        return
    source = next(s for s in plan['sources'] if s['id'] == sid)
    raw = ShootingBinaryTrajectory.load(dataset_path(source['dataset'])/source['relative_trajectory_path'])
    if sha(raw.root/'manifest.json') != source['manifest_sha256']: raise ValueError('Raw ancestor manifest changed')
    original = Path(plan['original_dataset'])/'sources'/str(sid)
    with np.load(original/'rows.npz') as a: meta = {k: a[k] for k in a.files}
    hot_bank = np.load(original/'positions.npy', mmap_mode='r')
    valid = (meta['kind'] < 2)&~meta['inside_crystal']&~meta['crystal_visible_context']&np.isfinite(meta['crystal_distance'])
    receipts = []
    for cell in [r for r in plan['cells'] if r['source'] == sid]:
        frame = cell['frame']; rows = np.flatnonzero(valid & (meta['frame'] == frame))
        folder = dest/str(frame); folder.mkdir(exist_ok=True)
        if (folder/'complete.json').exists(): receipts.append(config(folder/'complete.json')); continue
        cold, box, atom_ids, provenance = archived_positions(
            {'reuse_config': {'relaxation_plan': c['relaxation_plans'][0]}}, cell, source)
        if not np.array_equal(atom_ids, raw.atom_ids): raise ValueError('Relaxation changed atom identities')
        hot, hotbox = frame_geometry(raw, frame)
        if not np.allclose(box, hotbox, atol=1e-6, rtol=0): raise ValueError('Relaxation changed the box')
        centers = np.searchsorted(raw.atom_ids, meta['atom'][rows]); tree = cKDTree(hot, boxsize=box)
        chosen = [context_atoms(hot, box, tree, int(i), [(4,14,12),(14,24,12)]) for i in centers]
        query = np.stack([a[0] for a in chosen]); actual = np.stack([a[1] for a in chosen])
        if not np.allclose(actual, meta['actual'][rows], atol=2e-5, rtol=0): raise ValueError(f'Patch identity replay failed {sid}/{frame}')
        unique, inv = np.unique(query, return_inverse=True); inv = inv.reshape(-1,25)
        nn = tree.query(hot[unique], k=80, workers=1)[1]
        hot_xyz = hot[nn]-hot[unique,None]; hot_xyz -= box*np.rint(hot_xyz/box)
        for j in range(len(rows)):
            if not np.allclose(hot_xyz[inv[j]], hot_bank[meta['indices'][rows[j]]], atol=2e-5, rtol=0):
                raise ValueError(f'Nearest-neighbor replay changed {sid}/{frame}/{rows[j]}')
        xyz = cold[nn]-cold[unique,None]; xyz -= box*np.rint(xyz/box)
        # Never introduce atoms that were outside the original consumed support.
        xyz[np.linalg.norm(hot_xyz, axis=-1) >= 8] = 100.
        offset = cold[query]-cold[centers,None]; offset -= box*np.rint(offset/box)
        xyz = xyz.astype(np.float32)
        ptm=full_labels(cold,box,c['relaxed_labels']['ptm_rmsd_cutoff'])
        cluster,sizes=components(cold,box,ptm,dict(neighbor_cutoff_A=c['relaxed_labels']['neighbor_cutoff_A'],minimum_tracked_size=1))
        solid=(cluster>0)&(sizes[cluster]>=c['relaxed_labels']['minimum_cluster_atoms'])
        distance=np.full(len(rows),np.inf)
        if solid.any():distance=cKDTree(cold[solid],boxsize=box).query(cold[centers],workers=1)[0]
        visible=(solid[nn]&(np.linalg.norm(hot_xyz,axis=-1)<8)).any(1)[inv].any(1)
        common=~solid[centers]&~visible&np.isfinite(distance)
        np.savez(folder/'relaxed-labels.npz',distance=distance,inside=solid[centers],visible=visible,common=common,
                 ptm=ptm,qualifying_crystal=solid,atom_ids=atom_ids)
        blocks = (xyz[i:i+c['preparation']['chunk_size']] for i in range(0,len(xyz),c['preparation']['chunk_size']))
        desc = np.concatenate(list(pool.map(_block, blocks)))
        # Frozen hot shell assignments, but physical cold offsets in contractions.
        pieces = []
        for sl in (slice(0,1),slice(1,13),slice(13,25)):
            v=desc[inv][:,sl]; pieces += [v.mean(1),v.std(1)]
        values=desc[inv]; centered=values-values.mean(1,keepdims=True); direction=offset/24
        pieces.append(np.linalg.norm(np.einsum('npf,npd->nfd',centered,direction)/25,axis=-1))
        q=np.einsum('npi,npj->npij',direction,direction)-np.eye(3)*np.sum(direction**2,axis=-1)[...,None,None]/3
        pieces.append(np.linalg.norm(np.einsum('npf,npij->nfij',centered,q)/25,axis=(-1,-2)))
        np.save(folder/'features.npy',np.concatenate(pieces,axis=1).astype(np.float32))
        np.save(folder/'positions.npy',xyz); np.savez(folder/'index.npz',local_rows=rows,indices=inv,actual=offset.astype(np.float32))
        receipt=dict(source=sid,frame=frame,rows=len(rows),patches=len(xyz),provenance=provenance,
                     path=str(folder),files={n:sha(folder/n) for n in ('features.npy','positions.npy','index.npz','relaxed-labels.npz')})
        write_json(folder/'complete.json',receipt); receipts.append(receipt)
    write_json(dest/'complete.json',dict(plan_identity=plan['identity'],source=sid,frames=receipts))


def paired_prepare(c, index):
    plan=config(resolve_path(c['output'])/'technical/data-plan.json')
    sources=sorted({r['source'] for r in plan['cells']})[index::c['preparation']['tasks']]
    with ProcessPoolExecutor(max_workers=c['preparation']['workers']) as pool:
        for sid in sources:
            paired_source(c,sid,pool);print(json.dumps(dict(prepared_source=sid)),flush=True)


def paired_seal(c):
    plan=config(resolve_path(c['output'])/'technical/data-plan.json');pc=config(resolve_path(c['descriptor_parent']))
    x, rows, columns, parent_manifest=load(pc);chosen=np.load(plan['paired_rows_file'])
    if sha(Path(plan['paired_rows_file']))!=plan['paired_rows_sha256']:raise ValueError('Changed paired row selection')
    _,_,_,meta,_=population(parent(pc));old=rows['ids'][chosen]
    pair={k:v[chosen].copy() for k,v in rows.items()}
    for role in ROLES:
        take=pair['role']==role;pair['weights'][take]/=pair['weights'][take].sum()
    out=np.empty((len(chosen),len(columns)),np.float32);ix=np.empty((len(chosen),25),np.int64);actual=np.empty((len(chosen),25,3),np.float32)
    banks=[];offset=0;seen=np.zeros(len(chosen),bool);relaxed_distance=np.empty(len(chosen));common=np.zeros(len(chosen),bool)
    inside=np.zeros(len(chosen),bool);visible=np.zeros(len(chosen),bool)
    for sid in sorted({r['source'] for r in plan['cells']}):
        receipt=config(resolve_path(c['cache'])/'paired-sources'/str(sid)/'complete.json')
        if receipt['plan_identity']!=plan['identity']:raise ValueError('Changed paired-source receipt')
        source_global=np.flatnonzero(meta['source']==sid)
        for frame in receipt['frames']:
            folder=Path(frame['path'])
            for n,h in frame['files'].items():
                if sha(folder/n)!=h:raise ValueError(f'Changed paired frame {folder/n}')
            with np.load(folder/'index.npz') as a:
                global_ids=source_global[a['local_rows']];order=np.argsort(old);dest=order[np.searchsorted(old[order],global_ids)]
                if not np.array_equal(old[dest],global_ids) or seen[dest].any():raise ValueError('Paired row mismatch')
                ix[dest]=a['indices']+offset;actual[dest]=a['actual']
            out[dest]=np.load(folder/'features.npy');seen[dest]=True
            with np.load(folder/'relaxed-labels.npz') as a:
                relaxed_distance[dest]=a['distance'];common[dest]=a['common'];inside[dest]=a['inside'];visible[dest]=a['visible']
            banks.append(dict(path=str(folder/'positions.npy'),rows=frame['patches'],sha256=frame['files']['positions.npy']))
            offset+=frame['patches']
    if not seen.all() or not np.isfinite(out).all():raise ValueError('Incomplete relaxed cohort')
    root=resolve_path(c['cache'])
    # Both inputs and both target definitions use this exact common row set.
    # Preserve every excluded observation in a separate auditable challenge file.
    audit=resolve_path(c['output'])/'analyses/paired-coverage-v1';(audit/'technical').mkdir(parents=True,exist_ok=True)
    np.savez(audit/'technical/eligibility.npz',ids=old,common=common,inside_relaxed=inside,visible_relaxed=visible,
             old_target=pair['target'],relaxed_target=relaxed_distance,role=pair['role'],source=pair['source'])
    from .control_train import table
    table(audit,'coverage',[dict(role=r,available=int((pair['role']==r).sum()),common=int((common&(pair['role']==r)).sum()),
        inside_relaxed=int((inside&(pair['role']==r)).sum()),visible_relaxed=int((visible&(pair['role']==r)).sum()),
        no_relaxed_crystal=int((~np.isfinite(relaxed_distance)&(pair['role']==r)).sum())) for r in ROLES])
    if any(not np.any(common&(pair['role']==r)) for r in ROLES):raise ValueError('Joint crystal-free paired population loses a role; inspect coverage, no fits launched')
    pair={k:v[common] for k,v in pair.items()}
    for r in ROLES:
        take=pair['role']==r;pair['weights'][take]/=pair['weights'][take].sum()
    for domain,values,indices,positions,source_banks in [
        ('relaxed',out[common],ix[common],actual[common],banks),
        ('matched_raw',np.asarray(x[chosen[common]]),meta['indices'][old[common]],meta['actual'][old[common]],plan['original_geometry']['banks'])]:
        folder=root/domain;folder.mkdir(parents=True,exist_ok=True)
        np.savez(folder/'index.npz',indices=indices,actual=positions)
        geom=dict(index=str(folder/'index.npz'),index_sha256=sha(folder/'index.npz'),banks=source_banks)
        seal_cache(folder,values,pair,columns,geom,dict(domain=domain,plan=plan['identity'],parent=parent_manifest['identity'],
                   labels='original MD crystal distance',conditional_on_archived_relaxation=True,
                   observed_identity_support_frozen=True,full_cell_relaxation_sees_external_context=True))
        new=folder.with_name(domain+'_newlabels');new.mkdir(parents=True,exist_ok=True)
        for n in ('features.npy','columns.json','geometry.json'):
            p=new/n
            if p.is_symlink():
                if p.resolve()!=(folder/n).resolve():raise ValueError(f'Changed paired shared file {p}')
            else:p.symlink_to(folder/n)
        newrows={k:v.copy() for k,v in pair.items()};newrows['target']=relaxed_distance[common]
        np.savez(new/'rows.npz',**newrows)
        record=dict(provenance=dict(domain=domain,labels='relaxed instantaneous PTM clusters, no temporal establishment',
                    protocol=c['relaxed_labels'],paired_original_manifest_sha256=sha(folder/'manifest.json')),
                    rows={r:int((newrows['role']==r).sum()) for r in ROLES},
                    files={n:sha(new/n) for n in ('features.npy','columns.json','geometry.json','rows.npz')})
        record['identity']=digest(record);write_json(new/'manifest.json',record)
    print(json.dumps(dict(paired_counts=plan['counts'],relaxed_patches=offset)),flush=True)
