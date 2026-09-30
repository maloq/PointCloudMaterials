"""Full-snapshot descriptors and frozen-model assignments for the MD viewer."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from sklearn.cluster._kmeans import _labels_inertia_threadpool_limit

from src.data.fixed_cohort.protocol import centered, sha, write_json
from src.research.crystallization_origin.extract import raw_source, frame_geometry
from src.research.crystal_vector.interface import interface_mask
from src.research.spatial_vicreg_bias.data import descriptors
from .viewer_payload import write_asset


def read(config):
    c = json.loads(Path(config).read_text())
    if c['protocol'] != 'interface_dense_md_v1':
        raise ValueError('Wrong dense MD protocol')
    pc = json.loads(Path(c['pacmap_config']).read_text())
    corr = json.loads(Path(pc['correspondence_config']).read_text())
    parent = json.loads(Path(corr['parent']).read_text())
    for value in (c['output'], c['publication'], parent['cache'], parent['output']):
        if not Path(value).is_absolute():
            raise ValueError('Use the resolved dense MD submission config')
    return c, pc, corr, parent


def block(folder, start, stop):
    folder = Path(folder); dest = folder/'chunks'/f'{start:06d}-{stop:06d}.npy'
    if dest.exists():
        return str(dest)
    patches = np.load(folder/'patches.npy', mmap_mode='r')
    values = np.stack([descriptors(p)[0] for p in patches[start:stop]])
    if values.shape != (stop-start, 412) or not np.isfinite(values).all():
        raise ValueError(f'Invalid descriptors in {dest}')
    temp = dest.with_suffix('.building.npy'); np.save(temp, values); temp.replace(dest)
    return str(dest)


def prepare(config):
    c, pc, corr, parent = read(config)
    out = Path(c['output']); dest = Path(c['publication'])/'md-data'
    dest.mkdir(parents=True, exist_ok=True)
    cache = Path(parent['cache']); plan = json.loads((cache/'plan.json').read_text())
    if plan['identity'] != corr['cache_identity']:
        raise ValueError('Changed fixed descriptor inputs')
    manifest = dict(protocol=c['protocol'], selection=c['selection'], snapshots=[])
    for selected in c['snapshots']:
        sid, frame = selected['source'], selected['frame']; tag = f'{sid}-{frame}'
        folder = out/'data'/tag; (folder/'chunks').mkdir(parents=True, exist_ok=True)
        receipt = folder/'complete.json'
        if receipt.exists():
            r = json.loads(receipt.read_text())
            if sha(folder/'physical.npz') != r['physical_sha256'] or sha(folder/'targets.npy') != r['targets_sha256']:
                raise ValueError('Changed completed dense snapshot')
            if sha(dest/(tag+'.js')) != r['asset_sha256']:
                raise ValueError('Changed completed dense asset')
            manifest['snapshots'].append(r['snapshot']); continue
        started = time.monotonic()
        item = next(s for s in plan['sources'] if s['id']==sid)
        if item['role'] != 'test' or frame not in parent['assay']['frames']:
            raise ValueError('Dense display must use a recorded held-out snapshot')
        raw = raw_source(item); points, box = frame_geometry(raw, frame)
        tree = cKDTree(points, boxsize=box); neighbors = tree.query(points, k=80, workers=c['workers'])[1]
        patches = centered(points, box, np.arange(len(points)), neighbors)
        audit = Path(parent['ptm_audit'])/'technical/sources'/str(sid)
        p = audit/f'ptm-{frame//32*32:04d}-{min(item["frame_count"],frame//32*32+32):04d}.npz'
        pr = json.loads((audit/'ptm-complete.json').read_text())
        if sha(p) != pr['files'][p.name]:
            raise ValueError('Changed full-source PTM labels')
        with np.load(p) as z: ptm = z['labels'][frame%32]
        solid = np.isin(ptm, [1, 2, 3]); support = solid[neighbors].mean(1)
        # Preserve the original patch exactly for every previously observed atom,
        # including historical nearest-neighbor ordering at distance ties.
        assay = cache/'assay'
        old = np.flatnonzero((np.load(assay/'source.npy')==sid)&(np.load(assay/'frame.npy')==frame))
        old_atoms = np.load(assay/'atom.npy')[old]; atom_rows = np.searchsorted(raw.atom_ids, old_atoms)
        if not np.array_equal(raw.atom_ids[atom_rows], old_atoms) or len(np.unique(atom_rows)) != len(atom_rows):
            raise ValueError('Ambiguous existing dense display identities')
        patches[atom_rows] = np.load(assay/'parents.npy', mmap_mode='r')[old, :80]
        support[atom_rows] = np.load(assay/'support_fraction.npy')[old]
        np.save(folder/'patches.npy', patches)
        np.savez(folder/'old-rows.npz', original_row=old, atom_row=atom_rows)
        large = np.zeros(len(points), bool); ids = np.flatnonzero(solid)
        if len(ids):
            pairs = cKDTree(points[ids], boxsize=box).query_pairs(corr['interface']['cutoff_A'], output_type='ndarray')
            graph = coo_matrix((np.ones(len(pairs), np.uint8), (pairs[:, 0], pairs[:, 1])), shape=(len(ids), len(ids))).tocsr()
            _, comp = connected_components(graph, directed=False)
            large[ids] = np.bincount(comp)[comp] >= corr['interface']['minimum_crystal_component']
        boundary, accepted = interface_mask(points, box, solid, corr['interface']['cutoff_A'], corr['interface']['minimum_disordered_component'])
        boundary &= large
        distance = np.full(len(points), np.inf)
        if boundary.any(): distance = cKDTree(points[boundary], boxsize=box).query(points, workers=c['workers'])[0]
        region = np.full(len(points), 4, np.uint8); near = distance<=12
        region[near & solid]=1; region[near & accepted]=2; region[near & ~solid & ~accepted]=3
        region[boundary]=0; region[~np.isfinite(distance)]=5
        np.savez_compressed(folder/'physical.npz', coords=points.astype(np.float32), box=box.astype(np.float32),
            atom=raw.atom_ids, ptm=ptm, support_fraction=support, distance=distance, region=region)
        print(f'Prepared full geometry: {tag}, {len(points)} atoms; calculating descriptors', flush=True)
        chunks = [(i, min(i+c['chunk_atoms'], len(points))) for i in range(0, len(points), c['chunk_atoms'])]
        with ProcessPoolExecutor(max_workers=c['workers']) as pool:
            tasks = [pool.submit(block, str(folder), lo, hi) for lo, hi in chunks]
            for n, f in enumerate(as_completed(tasks), 1):
                f.result()
                if n%20==0: print(f'Descriptors {tag}: {n}/{len(tasks)} chunks', flush=True)
        targets = np.concatenate([np.load(folder/'chunks'/f'{lo:06d}-{hi:06d}.npy') for lo, hi in chunks])
        known = np.load(assay/'targets.npy', mmap_mode='r')[old]
        if not np.allclose(targets[atom_rows], known, rtol=1e-5, atol=1e-7):
            raise ValueError(f'Dense descriptor producer disagrees with saved observations: {tag}')
        targets[atom_rows] = known
        np.save(folder/'targets.npy', targets)
        fields = {}; models = {}; bindings = {}; assignment_checks = {}
        uniform_rows = np.flatnonzero(np.load(assay/'uniform.npy'))
        old_indices = np.searchsorted(uniform_rows, old)
        if not np.array_equal(uniform_rows[old_indices], old):
            raise ValueError('Expected original uniform observations for this snapshot')
        for family, label in [('tda','TDA clusters'), ('bond_order','Bond-order clusters'), ('cna','CNA clusters'), ('joint','Joint descriptor clusters')]:
            mp = Path(corr['output'])/'data'/f'interface12-{family}-k7-descriptor-model.npz'
            with np.load(mp) as z:
                # Match the cluster producer's float32 cast BEFORE balancing.
                x = np.ascontiguousarray((targets[:, z['columns']]-z['mean'])/z['sd'], dtype=np.float32)
                x /= z['balance']
                values = _labels_inertia_threadpool_limit(x, np.ones(len(x), np.float32), z['centers'], n_threads=1, return_inertia=False)
                seed = np.flatnonzero(z['cluster_seeds']==corr['primary_cluster_seed'])
                if len(seed)!=1:raise ValueError('Missing descriptor cluster seed')
                expected = z['assignments'][seed[0], old_indices]
                assignment_checks[family] = int(np.sum(values[atom_rows]!=expected))
                if assignment_checks[family]:raise ValueError(f'Changed saved descriptor assignments: {family}')
            fields[label] = values.tolist(); models[family] = values; bindings[str(mp)] = sha(mp)
        signed = np.clip(np.where(solid, -1, 1)*distance, -20, 20)
        fields.update({'PTM type':ptm.tolist(), 'Physical region':region.tolist(),
                       'Input crystal fraction':np.round(support, 4).tolist(),
                       'Interface distance (Å, clipped ±20)':[round(v,4) if np.isfinite(d) else None for v,d in zip(signed,distance)]})
        np.savez_compressed(folder/'descriptor-labels.npz', **models)
        asset = dest/(tag+'.js')
        write_asset(asset, tag, dict(source=sid, frame=frame, count=len(points), box=box.tolist(),
            x=np.round(points[:,0],4).tolist(), y=np.round(points[:,1],4).tolist(), z=np.round(points[:,2],4).tolist(), fields=fields))
        snapshot = dict(key=tag, source=sid, frame=frame, count=len(points), asset='../md-data/'+asset.name)
        r = dict(snapshot=snapshot, neural_training=False, selection=c['selection'], original_rows_preserved=len(old),
                 physical_sha256=sha(folder/'physical.npz'), targets_sha256=sha(folder/'targets.npy'),
                 asset_sha256=sha(asset), descriptor_models=bindings, ptm_sha256=sha(p),
                 assignment_checks=assignment_checks,
                 source_manifest_sha256=item['manifest_sha256'], data_identity=plan['identity'],
                 implementation_sha256=sha(__file__), descriptor_sha256=sha(Path(descriptors.__code__.co_filename)), seconds=time.monotonic()-started)
        write_json(receipt, r); manifest['snapshots'].append(snapshot)
        print(f'Dense descriptor snapshot complete: {tag}', flush=True)
    if c.get('publish_manifest', True):
        write_json(out/'technical/manifest.json', manifest)
        write_json(dest/'manifest.json', manifest)


def infer(config):
    import torch
    from omegaconf import OmegaConf
    from src.research.spatial_vicreg_bias.train import PairEncoder
    c, pc, corr, parent = read(config)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    out = Path(c['output']); dest = Path(c['publication'])/'md-data'
    for selected in c['snapshots']:
        tag = f'{selected["source"]}-{selected["frame"]}'; folder=out/'data'/tag
        physical = np.load(folder/'physical.npz'); patches=np.load(folder/'patches.npy', mmap_mode='r')
        dense_receipt=json.loads((folder/'complete.json').read_text())
        if sha(folder/'physical.npz') != dense_receipt['physical_sha256']:
            raise ValueError('Changed dense physical observations')
        for item in c['models']:
            run, epoch = item['run'], item['epoch']; key=f'{tag}-{run}-epoch{epoch}'
            receipt=out/'technical'/f'{key}.json'
            if receipt.exists(): continue
            root=Path(parent['output'])/run; cp=root/'checkpoints'/f'epoch-{epoch:02d}.pt'
            original=json.loads((root/'analyses'/f'epoch-{epoch:02d}'/'technical/complete.json').read_text())
            if sha(cp) != original['checkpoint_sha256']: raise ValueError('Changed checkpoint')
            saved=torch.load(cp,map_location='cpu',weights_only=False)
            if saved['epoch']!=epoch or saved['data_identity']!=corr['cache_identity']:
                raise ValueError('Wrong frozen checkpoint identity')
            torch.manual_seed(saved['seed']); np.random.seed(saved['seed'])
            model=PairEncoder(OmegaConf.create(saved['recipe'])).cuda().eval(); model.requires_grad_(False)
            model.load_state_dict(saved['model'],strict=True)
            centers={}; values={rep:np.empty(len(patches),np.uint8) for rep in ('encoder','projector')}
            for rep in values:
                with np.load(root/'analyses'/f'epoch-{epoch:02d}'/'data'/f'{rep}-k7-assignments.npz') as z:
                    centers[rep]=z['centers']
            with torch.inference_mode():
                for start in range(0,len(patches),256):
                    x=torch.as_tensor(np.asarray(patches[start:start+256])/parent['geometry']['length_scale_A'],device='cuda')
                    z,y=model(x)
                    for rep, features in [('encoder',z),('projector',y)]:
                        xx=np.ascontiguousarray(features.cpu().numpy())
                        values[rep][start:start+len(xx)]=_labels_inertia_threadpool_limit(xx,np.ones(len(xx),np.float32),centers[rep],n_threads=1,return_inertia=False)
                    if start%16384==0:print(f'Dense frozen inference {key}: {start}/{len(patches)}',flush=True)
            # Existing displayed observations keep original saved assignments;
            # independently audit numerical replay before preserving them.
            with np.load(folder/'old-rows.npz') as old:
                disagreement={}
                for rep in values:
                    with np.load(root/'analyses'/f'epoch-{epoch:02d}'/'data'/f'{rep}-k7-assignments.npz') as z:
                        expected=z['cluster'][old['original_row']]
                    disagreement[rep]=int(np.sum(values[rep][old['atom_row']]!=expected))
                    if disagreement[rep]>2:raise ValueError(f'Dense GPU assignment mismatch: {key}/{rep}/{disagreement[rep]}')
                    values[rep][old['atom_row']]=expected
            np.savez_compressed(folder/(key+'-labels.npz'),**values)
            asset=dest/(key+'.js');write_asset(asset,key,{k:v.tolist() for k,v in values.items()},'MD_NEURAL')
            write_json(receipt,dict(key=key,asset='../md-data/'+asset.name,asset_sha256=sha(asset),
                checkpoint_sha256=sha(cp),rows=len(patches),device='cuda-float32',neural_training=False,
                conditions=[],geometry='observed periodic centered nearest 80 atoms',history=0,
                assignment_disagreements=disagreement,implementation_sha256=sha(__file__)))
            shutil.copy2(receipt,dest/receipt.name)
            del model;torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','infer']);p.add_argument('--config',required=True)
    args=p.parse_args();globals()[args.stage](args.config)
