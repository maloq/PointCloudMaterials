"""Publish a frozen rich-MACE state beside the original descriptor references."""
import copy
import hashlib
import html
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from src.analysis.representative_style import sparse_geometry
from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_table
from .cluster_matching import summarize
from .embedding_travel import payload
from .mace_checkpoint import read
from .sample_lattice import fit_batch
from .comparison_layout import comparison_template, population_controls
from .viewer_payload import read_payload, write_comparison, write_asset, write_vector_asset

FAMILIES = {'joint': 'Joint descriptor clusters', 'tda': 'TDA clusters',
            'bond_order': 'Bond-order clusters', 'cna': 'CNA clusters'}


def rebase_assets(value, reference, dest):
    if isinstance(value, dict):
        return {k: os.path.relpath((reference/'interactive'/v).resolve(), dest/'interactive')
                if k == 'asset' else rebase_assets(v, reference, dest) for k, v in value.items()}
    if isinstance(value, list):
        return [rebase_assets(v, reference, dest) for v in value]
    return value


def contingency(left, right):
    return np.bincount(np.asarray(left, int)*7+np.asarray(right, int), minlength=49).reshape(7, 7)


def score(left, right, mapping):
    result = summarize(contingency(left, right), mapping)
    result['adjusted_rand_index'] = float(adjusted_rand_score(left, right)) if len(left) else None
    return result


def samples(folder, dest, key, identity, labels):
    patches = np.load(folder/'patches.npy', mmap_mode='r')
    with np.load(folder/'physical.npz') as z: atoms = z['atom']
    clusters, examples = [], {}
    for cluster in range(7):
        members = np.flatnonzero(labels == cluster)
        seed = int.from_bytes(hashlib.sha256(f'20260929/{key}/neural/{identity}/{cluster}'.encode()).digest()[:8], 'little')
        chosen = np.random.default_rng(seed).choice(members, min(5, len(members)), replace=False)
        clusters.append(dict(count=len(members), rows=chosen.tolist()))
        for row in chosen:
            xyz = np.asarray(patches[row])
            radius = np.sort(np.linalg.norm(xyz, axis=1))
            oriented, edges, _ = sparse_geometry(xyz, float(1.2*np.median(radius[1:13])), orientation='pca')
            examples[str(row)] = dict(atom=int(atoms[row]), xyz=oriented.tolist(), edges=edges)
    name = key+'-'+identity
    write_asset(dest/'sample-data'/f'{name}.js', name, dict(clusters=clusters, patches=examples), 'CLUSTER_SAMPLES')
    rows = list(examples)
    fits = fit_batch([examples[r]['xyz'] for r in rows])
    write_asset(dest/'lattice-data'/f'{name}.js', name, dict(zip(rows, fits)), 'LATTICE_SAMPLES')
    return (dict(key=name, asset='../sample-data/'+name+'.js'),
            dict(key=name, asset='../lattice-data/'+name+'.js'))


def projection(c, dest, population, z):
    import faiss
    import numba
    import pacmap
    pc = json.loads(Path(c['pacmap_config']).read_text())['pacmap']
    if version('pacmap') != pc['version']:
        raise ValueError('PaCMAP version changed')
    target = dest/'data'/f'pacmap-{population}.npz'
    if target.exists():
        with np.load(target) as data: return data['y3']
    faiss.omp_set_num_threads(2); numba.set_num_threads(2)
    params = {k: v for k, v in pc.items() if k != 'version'}
    print(f'PaCMAP {dest.name}/{population}: {z.shape}', flush=True)
    y = pacmap.PaCMAP(n_components=3, **params).fit_transform(np.asarray(z, np.float32), init='pca')
    if not np.isfinite(y).all(): raise ValueError('Nonfinite projection')
    np.savez_compressed(target, y3=y)
    return y


def template(c, kind):
    options = ''.join(f'<option value="../../{k}/interactive/comparison-{("all_test" if k=="matched" else "all_static")}.html"'
                      + (' selected' if k == kind else '') + f'>{title}</option>'
                      for k, title in [('matched', 'Held-out Al MD'), ('static', 'Al static · 166–240 ps')])
    reference = Path(c['datasets'][kind]['reference'])
    dest = Path(c['datasets'][kind]['publication'])
    link = os.path.relpath(reference/'index.html', dest/'interactive')
    nav = f'<nav><label>Data <select onchange="location.href=this.value">{options}</select> <a href="{link}">GeoFormer comparison</a></label></nav>'
    note = '<p class="status">Frozen step '+str(c['model']['update'])+' · 256-D embedding · trained on rich local descriptors</p>'
    methods = '<details><summary>Methods and checkpoint</summary><p>'+html.escape(c['model']['title'])+'. '
    methods += 'The frozen scalar encoder state is clustered into seven groups using the original 74,880 training-source observations. '
    methods += 'PaCMAP, dense MD, examples and travel use the same observations and descriptors as the GeoFormer comparison. '
    methods += ('Color matching maximizes overlap on the 24,960 fixed held-out display observations.' if kind == 'matched' else 'Color matching maximizes overlap on the 684,723 centers in the six static MD snapshots.')
    methods += ' Colors stay fixed across frames. The encoder was trained to predict 442 rich descriptors: descriptor correspondence is not an independent discovery test. '
    methods += 'The six static snapshots are relaxed inputs; this MACE was trained on raw dynamic patches. '
    methods += 'Descriptor clusters use all training environments. <a id="metricDefinitions" href="../tables/METRICS.md">Metric definitions</a>.</p><p>Checkpoint: '+html.escape(c['checkpoint'])+'</p></details>'
    controls = dict(FRAME_CONTROLS=population_controls('All snapshots')) if kind == 'matched' else {}
    return comparison_template(nav, HEADLINE='MACE ↔ descriptors', HEADER_NOTE=note, METHODS=methods, **controls)


def publish(config):
    c, out = read(config); identity = c['model']['id']
    if not (out/'technical/inference.json').exists(): raise ValueError('Frozen inference is incomplete')
    import torch
    saved = torch.load(c['checkpoint'], map_location='cpu', weights_only=False)
    plan = json.loads((Path(c['assay']).parent/'plan.json').read_text())
    if plan['fixed_identity'] != saved['config']['fixed_dataset']['identity']:
        raise ValueError('Checkpoint and analysis use different fixed Al64 releases')
    roles = np.load(Path(c['assay'])/'role.npy'); uniform = np.load(Path(c['assay'])/'uniform.npy')
    with np.load(out/'data/clusters.npz') as z:
        if not np.array_equal(z['fit_rows'], np.flatnonzero((roles == 'train') & uniform)):
            raise ValueError('Neural clustering rows differ from the frozen training contract')
    del saved
    for kind, ds in c['datasets'].items():
        reference, dest, dense = map(Path, (ds['reference'], ds['publication'], ds['source']))
        for folder in ('interactive', 'data', 'assets', 'projection-data', 'md-data', 'sample-data', 'lattice-data', 'travel-data', 'technical/rendering'):
            (dest/folder).mkdir(parents=True, exist_ok=True)
        base = payload(reference)
        with np.load(out/'data'/('heldout.npz' if kind == 'matched' else 'static.npz')) as z:
            vectors, labels = z['z'], z['labels']
            if not np.array_equal(z['frame'], base['frame']) or not np.array_equal(z['atom'], base['atom']):
                raise ValueError('Projection observation order differs from reference')
        d = rebase_assets(copy.deepcopy(base), reference, dest)
        d['frozen_model'] = c['model']
        # Never carry a historical neural field into the new checkpoint view.
        neural_fields = {entry['field'] for entry in d['paired']['neural']}
        d['fields'] = {k: v for k, v in d['fields'].items() if k not in neural_fields}
        field = 'neural:'+identity; d['fields'][field] = labels.tolist()
        md_model = dict(id=identity, title=c['model']['title'], representation='encoder', snapshots={})
        dense_tables = {f: np.zeros((7, 7), np.int64) for f in FAMILIES}
        snapshot_data = {}
        for snap in d['md']['snapshots']:
            key = snap['key']; stored = out/'data'/kind/key; folder = dense/'data'/key
            with np.load(stored/'labels.npz') as z: assigned = z['encoder']
            with np.load(folder/'descriptor-labels.npz') as z: descriptors = dict(z)
            with np.load(folder/'physical.npz') as z: physical = dict(z)
            snapshot_data[key] = (assigned, descriptors, physical['distance'])
            for family in FAMILIES: dense_tables[family] += contingency(assigned, descriptors[family])
            name = key+'-'+identity
            write_asset(dest/'md-data'/f'{name}.js', name, dict(encoder=assigned.tolist()), 'MD_NEURAL')
            md_model['snapshots'][key] = dict(key=name, asset='../md-data/'+name+'.js')
            sample_entry, lattice_entry = samples(folder, dest, key, identity, assigned)
            d['samples'][key]['neural'] = {identity: sample_entry}
            d['lattice'][key]['neural'] = {identity: lattice_entry}
            with np.load(stored/'travel.npz') as z:
                write_vector_asset(dest/'travel-data'/f'{name}.js', name, z['z'],
                    dict(neighbors=z['neighbors'].tolist(), clusters=z['labels'].tolist()))
            d['travel'][key]['models'] = {identity: dict(key=name, asset='../travel-data/'+name+'.js')}
            print(f'Prepared viewer assets: {kind}/{key}', flush=True)
        d['md']['models'] = [md_model]
        matching = {}; metrics = {}
        for family, descriptor_field in FAMILIES.items():
            right = np.asarray(d['fields'][descriptor_field])
            table = contingency(labels, right) if kind == 'matched' else dense_tables[family]
            rows, mapping = linear_sum_assignment(table, maximize=True)
            if not np.array_equal(rows, np.arange(7)): raise ValueError('Incomplete color assignment')
            matching[family] = dict(neural_to_descriptor=mapping.tolist(), contingency=table.tolist(), reference=summarize(table, mapping))
            near = np.array([v is not None and v <= 12 for v in d['distance']])
            metrics[family] = dict(display=score(labels, right, mapping), interface12=score(labels[near], right[near], mapping), snapshots={})
            for key, (left, descriptors, distance) in snapshot_data.items():
                near = distance <= 12
                metrics[family]['snapshots'][key] = dict(all=score(left, descriptors[family], mapping),
                    interface12=score(left[near], descriptors[family][near], mapping))
        d['matching'] = {identity: matching}
        write_json(dest/'technical/matches.json', matching)
        write_metric_table(metrics, dest, family='rich_mace_interface')
        populations = ('all_test', 'interface20') if kind == 'matched' else ('all_static', 'interface20')
        for population in populations:
            if population == populations[0]: subset = np.arange(len(labels)); view = copy.deepcopy(d)
            else:
                old_path = reference/'interactive'/('comparison-interface20.html' if kind == 'matched' else 'S1-seed17-epoch24-encoder-interface20.html')
                previous = read_payload(old_path)
                lookup = {k: i for i, k in enumerate(zip(base['source'], base['frame'], base['atom']))}
                subset = np.array([lookup[k] for k in zip(previous['source'], previous['frame'], previous['atom'])])
                view = copy.deepcopy(d)
                for key in ('source', 'frame', 'atom', 'distance', 'solid', 'region'): view[key] = previous[key]
                view['fields'] = {k: np.asarray(v, dtype=object)[subset].tolist() for k, v in d['fields'].items()}
                view['paired']['descriptors'] = rebase_assets(previous['paired']['descriptors'], reference, dest)
            if kind == 'matched': view['explorer']['population'] = population
            projected = projection(c, dest, population, vectors[subset])
            name = identity+'-'+population
            write_asset(dest/'projection-data'/f'{name}.js', name,
                        dict(y3=np.round(projected, 6).tolist(), clusters=labels[subset].tolist()), 'PACMAP_LAYOUTS')
            entry = dict(id=identity, title=c['model']['title'], representation='encoder', field=field,
                         key=name, asset='../projection-data/'+name+'.js')
            view['paired']['neural'] = [entry]; view['paired']['default_neural'] = identity
            for filename in ('cluster_comparison.js', 'viewer_extensions.js'):
                shutil.copy2(Path(__file__).with_name(filename), dest/'assets'/filename)
            shutil.copy2(reference/'assets/plotly.min.js', dest/'assets/plotly.min.js')
            path = dest/'interactive'/f'comparison-{population}.html'
            write_comparison(path, template(c, kind), view, title='MACE and descriptor clusters')
            if population == populations[0]:
                write_comparison(dest/'index.html', template(c, kind), view, title='MACE and descriptor clusters', index=True)
        write_json(dest/'technical/rendering/provenance.json', dict(checkpoint_sha256=c['checkpoint_sha256'],
            reference=str(reference), reference_payload_sha256=sha(reference/'index.json'), model=c['model'],
            input_record=str(out/'technical/inference.json'), neural_training=False, descriptor_refit=False,
            original_observations_preserved=True, fixed_release_identity=plan['fixed_identity'],
            assay_plan_sha256=sha(Path(c['assay']).parent/'plan.json'), frozen_clusters_sha256=sha(out/'data/clusters.npz')))
        print(f'Published {dest}', flush=True)
