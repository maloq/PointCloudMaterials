"""Publish all-training descriptor fits beside unchanged frozen neural embeddings."""
import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import pairwise_distances_argmin

from src.analysis.representative_style import sparse_geometry
from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_table
from src.project_runtime.paths import resolve_config
from .cluster_matching import summarize
from .sample_lattice import fit_batch
from .viewer_payload import read_payload, read_asset, write_asset

FIELDS = {'joint':'Joint descriptor clusters', 'tda':'TDA clusters',
          'bond_order':'Bond-order clusters', 'cna':'CNA clusters'}
REMOVED_FIELDS = ('Physical region', 'Interface distance (Å, clipped ±20)')


def load(config):
    c = resolve_config(json.loads(Path(config).read_text()))
    if c['protocol'] != 'general_descriptor_comparison_v1': raise ValueError('Wrong comparison protocol')
    return c


def transform(targets, model):
    # Match correspondence.descriptor_clusters: float32 BEFORE family balancing.
    z = ((targets[:, model['columns']] - model['mean']) / model['sd']).astype(np.float32)
    z /= model['balance']
    if not np.isfinite(z).all(): raise ValueError('Nonfinite standardized descriptors')
    return z


def asset_entry(path, key, publication):
    return dict(key=key, asset=os.path.relpath(path, Path(publication)/'interactive'))


def samples(folder, out, key, family, labels):
    patches = np.load(folder/'patches.npy', mmap_mode='r')
    with np.load(folder/'physical.npz') as z: atoms = z['atom']
    clusters, examples = [], {}
    for cluster in range(7):
        members = np.flatnonzero(labels == cluster)
        seed = int.from_bytes(hashlib.sha256(f'20260929/{key}/descriptors/{family}/{cluster}'.encode()).digest()[:8], 'little')
        chosen = np.random.default_rng(seed).choice(members, min(5, len(members)), replace=False)
        clusters.append(dict(count=int(len(members)), rows=chosen.tolist()))
        for row in chosen:
            xyz = np.asarray(patches[row]); radius = np.sort(np.linalg.norm(xyz, axis=1))
            oriented, edges, _ = sparse_geometry(xyz, float(1.2*np.median(radius[1:13])), orientation='pca')
            examples[str(row)] = dict(atom=int(atoms[row]), xyz=oriented.tolist(), edges=edges)
    name = key+'-all-'+family
    write_asset(out/'sample-data'/f'{name}.js', name, dict(clusters=clusters, patches=examples), 'CLUSTER_SAMPLES')
    rows = list(examples)
    fits = fit_batch([examples[r]['xyz'] for r in rows]) if rows else []
    write_asset(out/'lattice-data'/f'{name}.js', name, dict(zip(rows, fits)), 'LATTICE_SAMPLES')
    return name


def build(config, kind):
    c = load(config); spec = c['datasets'][kind]; out = Path(c['output'])/'analyses'/kind
    for folder in ('data', 'technical', 'projection-data', 'md-data', 'sample-data', 'lattice-data', 'travel-data'):
        (out/folder).mkdir(parents=True, exist_ok=True)
    reference = Path(spec['publications'][0]); payload = read_payload(reference/'index.html')
    fit_root = Path(c['fits']); assay = Path(c['assay'])
    fit_record = json.loads((fit_root/'technical/descriptor-clustering.json').read_text())
    plan = json.loads((assay.parent/'plan.json').read_text())
    if plan['fixed_identity'] != c['fixed_identity']: raise ValueError('Changed fixed data release')
    uniform = np.flatnonzero(np.load(assay/'uniform.npy'))
    roles = np.load(assay/'role.npy')[uniform]
    train = roles == 'train'; sources = np.load(assay/'source.npy')[uniform]
    models, provenance = {}, {}
    for family in FIELDS:
        path = fit_root/'data'/f'all-{family}-k7-descriptor-model.npz'
        with np.load(path) as z: models[family] = dict(z)
        if models[family]['cluster_seeds'][0] != 17: raise ValueError('Changed primary clustering seed')
        r = fit_record[f'all-{family}-k7']
        if r['training_rows'] != int(train.sum()) or r['training_sources'] != len(np.unique(sources[train])):
            raise ValueError('All-training descriptor fit identity mismatch')
        provenance[family] = dict(path=str(path), sha256=sha(path), training_rows=r['training_rows'],
            training_sources=r['training_sources'], active_features=r['active_features'])
    if kind == 'matched':
        selected = uniform[roles == 'test']
        for field in ('source', 'frame', 'atom'):
            if not np.array_equal(np.load(assay/(field+'.npy'))[selected], payload[field]):
                raise ValueError(f'Changed held-out observation order: {field}')
        displayed = np.asarray(np.load(assay/'targets.npy', mmap_mode='r')[selected])
    else:
        parts = []
        for snapshot in payload['md']['snapshots']:
            folder = Path(spec['source'])/'data'/snapshot['key']
            with np.load(folder/'physical.npz') as z:
                rows = z['sample']; atoms = z['atom'][rows]
            mask = np.asarray(payload['frame']) == snapshot['frame']
            if not np.array_equal(atoms, np.asarray(payload['atom'])[mask]): raise ValueError('Changed static display order')
            parts.append(np.asarray(np.load(folder/'targets.npy', mmap_mode='r')[rows]))
        displayed = np.concatenate(parts)
    labels = {}; projections = {}
    import pacmap, numba, faiss
    pc = c['pacmap']; params = {k:v for k,v in pc.items() if k != 'version'}
    if version('pacmap') != pc['version']: raise ValueError('Changed PaCMAP version')
    numba.set_num_threads(2); faiss.omp_set_num_threads(2)
    for family, model in models.items():
        z = transform(displayed, model)
        labels[family] = pairwise_distances_argmin(z, model['centers']).astype(np.int16)
        if kind == 'matched' and not np.array_equal(labels[family], model['assignments'][0][roles == 'test']):
            raise ValueError(f'All-fit label reproduction failed: {family}')
        target = out/'data'/f'{family}-projection.npz'
        if target.exists():
            with np.load(target) as data:
                if not np.array_equal(data['labels'], labels[family]): raise ValueError('Changed cached labels')
                y3 = data['y3']
        else:
            print(f'{kind}/{family} PaCMAP: {z.shape}', flush=True)
            y3 = pacmap.PaCMAP(n_components=3, **params).fit_transform(z, init='pca')
            if not np.isfinite(y3).all(): raise ValueError('Nonfinite PaCMAP')
            np.savez_compressed(target, y3=y3, labels=labels[family])
        name = 'all-training-'+family
        write_asset(out/'projection-data'/(name+'.js'), name, dict(y3=np.round(y3,6).tolist()), 'PACMAP_LAYOUTS')
        projections[family] = name
    np.savez_compressed(out/'data/display-labels.npz', **labels)
    records = {}
    for snapshot in payload['md']['snapshots']:
        key = snapshot['key']; folder = Path(spec['source'])/'data'/key
        old = read_asset(reference/'interactive'/snapshot['asset'])
        targets = np.load(folder/'targets.npy', mmap_mode='r')
        dense = {}
        for family, model in models.items():
            parts = [pairwise_distances_argmin(transform(np.asarray(targets[start:start+8192]), model), model['centers'])
                     for start in range(0,len(targets),8192)]
            dense[family] = np.concatenate(parts).astype(np.int16)
            old['fields'][FIELDS[family]] = dense[family].tolist()
        for field in REMOVED_FIELDS: old['fields'].pop(field, None)
        np.savez_compressed(out/'data'/f'{key}-labels.npz', **dense)
        write_asset(out/'md-data'/(key+'.js'), key, old, 'MD_SNAPSHOTS')
        names = {family:samples(folder, out, key, family, values) for family,values in dense.items()}
        # Preserve frozen travel observations/vectors, update their descriptor IDs.
        travel = read_asset(reference/'interactive'/payload['travel'][key]['geometry']['asset'])
        travel['descriptor_clusters'] = dense['joint'][travel['rows']].tolist()
        write_asset(out/'travel-data'/(key+'.js'), key, travel, 'TRAVEL_GEOMETRY')
        records[key] = dict(samples=names, rows=len(targets), targets_sha256=sha(folder/'targets.npy'),
                            labels_sha256=sha(out/'data'/f'{key}-labels.npz'))
        print(f'{kind}/{key}: {len(targets):,} dense labels and descriptor examples complete', flush=True)
    write_json(out/'technical/build.json', dict(config_sha256=sha(config), implementation_sha256=sha(__file__),
        fit_population='all training environments', fits=provenance, fixed_identity=c['fixed_identity'],
        training_rows=int(train.sum()), primary_seed=17, neural_training=False, projections=projections, snapshots=records,
        travel='Frozen observed centers and neural vectors retained; descriptor IDs updated. Original sampling: uniform plus per-cluster supplements.'))


def apply_overlay(payload, overlay):
    for field in ('source', 'frame', 'atom'):
        if payload[field] != overlay['identity'][field]: raise ValueError(f'General descriptor overlay mismatched {field}')
    if {n['id'] for n in payload['paired']['neural']} != set(overlay['matching']): raise ValueError('Changed neural inventory')
    payload['fields'].update(overlay['fields'])
    for field in REMOVED_FIELDS: payload['fields'].pop(field, None)
    payload.pop('region', None)
    payload['paired']['descriptors'] = overlay['descriptors']
    payload['md']['snapshots'] = overlay['snapshots']
    payload['matching'] = overlay['matching']
    for key in overlay['samples']:
        payload['samples'][key]['descriptors'] = overlay['samples'][key]
        payload['lattice'][key]['descriptors'] = overlay['lattice'][key]
        payload['travel'][key]['geometry'] = overlay['travel'][key]
    payload['descriptor_fit'] = overlay['fit']
    payload['title'] = 'Neural and descriptor clusters'


def publish(config, kind):
    from .comparison_layout import refresh
    c = load(config); spec = c['datasets'][kind]; out = Path(c['output'])/'analyses'/kind
    receipt = json.loads((out/'technical/build.json').read_text())
    if receipt['config_sha256'] != sha(config) or receipt['implementation_sha256'] != sha(__file__):
        raise ValueError('Build provenance changed before publication')
    with np.load(out/'data/display-labels.npz') as z: display = dict(z)
    metrics = {}
    for publication in spec['publications']:
        dest = Path(publication); payload = read_payload(dest/'index.html'); overlay = {}
        overlay['identity'] = {k:payload[k] for k in ('source','frame','atom')}
        overlay['fields'] = {FIELDS[f]:v.tolist() for f,v in display.items()}
        overlay['descriptors'] = [dict(id='descriptors-'+f, title=FIELDS[f], field=FIELDS[f],
            **asset_entry(out/'projection-data'/(name+'.js'),name,dest)) for f,name in receipt['projections'].items()]
        overlay['snapshots'] = [dict(s, **asset_entry(out/'md-data'/(s['key']+'.js'),s['key'],dest)) for s in payload['md']['snapshots']]
        for section in ('samples','lattice','travel'): overlay[section] = {}
        for key,r in receipt['snapshots'].items():
            for section,folder in [('samples','sample-data'),('lattice','lattice-data')]:
                overlay[section][key] = {f:asset_entry(out/folder/(name+'.js'),name,dest) for f,name in r['samples'].items()}
            overlay['travel'][key] = asset_entry(out/'travel-data'/(key+'.js'),key,dest)
        matching = {}; model_metrics = {}
        for neural in payload['paired']['neural']:
            identity = neural['id']; tables = {}
            if kind == 'matched':
                left = np.asarray(payload['fields'][neural['field']]) if neural['field'] in payload['fields'] else np.asarray(read_asset(dest/'interactive'/neural['asset'])['clusters'])
                right = display
            else:
                model = next(m for m in payload['md']['models'] if m['id'] == identity)
                lefts = []; rights = {f:[] for f in FIELDS}
                for snapshot in payload['md']['snapshots']:
                    key = snapshot['key']; entry = model['snapshots'][key]
                    lefts.append(read_asset(dest/'interactive'/entry['asset'])[model['representation']])
                    with np.load(out/'data'/f'{key}-labels.npz') as z:
                        for f in FIELDS: rights[f].append(z[f])
                left = np.concatenate(lefts); right = {f:np.concatenate(v) for f,v in rights.items()}
            matching[identity] = {}; model_metrics[identity] = {}
            for family in FIELDS:
                if len(left) != len(right[family]): raise ValueError('Unmatched cluster populations')
                table = np.bincount(left.astype(int)*7+right[family],minlength=49).reshape(7,7)
                rows, columns = linear_sum_assignment(table,maximize=True)
                if not np.array_equal(rows,np.arange(7)): raise ValueError('Incomplete color assignment')
                result = summarize(table, columns)
                result['adjusted_rand_index'] = float(adjusted_rand_score(left,right[family]))
                matching[identity][family] = dict(neural_to_descriptor=columns.tolist(), contingency=table.tolist(), reference=result)
                model_metrics[identity][family] = result
        overlay['matching'] = matching
        overlay['fit'] = dict(population='all training environments', training_rows=receipt['training_rows'],
            primary_seed=17, metrics=os.path.relpath(out/'tables/METRICS.md',dest/'interactive'))
        history = dest/'technical/rendering/history'; history.mkdir(parents=True,exist_ok=True)
        prior = dest/'index.json'; backup = history/('before-all-descriptors-'+sha(prior)+'.json')
        if not backup.exists(): backup.write_bytes(prior.read_bytes())
        write_json(dest/'technical/general-descriptors.json',overlay)
        metrics[dest.parent.parent.name] = model_metrics
        refresh(dest,kind)
    write_metric_table(metrics,out,family='general_descriptor_comparison')
    write_json(out/'technical/metrics.json',metrics)
    print(f'Published {kind}: all-training descriptor comparisons',flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    p.add_argument('--dataset',choices=['matched','static'],required=True)
    p.add_argument('--stage',choices=['build','publish'],required=True)
    a=p.parse_args();globals()[a.stage](a.config,a.dataset)
