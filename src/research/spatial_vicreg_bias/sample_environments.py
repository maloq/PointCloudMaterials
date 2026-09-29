"""Export real local neighborhoods from saved dense cluster assignments."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib.colors as mcolors

from src.data.fixed_cohort.protocol import sha, write_json
from src.analysis.representative_style import radial_colors, sparse_geometry
from .dense_md import write_asset


def export(source, publication, dataset, snapshot_key=None):
    source = Path(source).resolve(); dest = Path(publication).resolve()
    page = dest/'index.html'
    payload = json.loads(page.read_text().split('const D=', 1)[1].split(';\nconst palette', 1)[0])
    if dataset == 'matched':
        payload['md']['snapshots'] = json.loads((source/'technical/manifest.json').read_text())['snapshots']
    if snapshot_key is not None:
        payload['md']['snapshots'] = [s for s in payload['md']['snapshots'] if s['key'] == snapshot_key]
        if len(payload['md']['snapshots']) != 1: raise ValueError(f'Unknown snapshot: {snapshot_key}')
    folder = dest/'sample-data'; folder.mkdir(exist_ok=True)
    manifest = {}; provenance = {}
    for snapshot in payload['md']['snapshots']:
        key = snapshot['key']; data = source/'data'/key
        patches = np.load(data/'patches.npy', mmap_mode='r')
        with np.load(data/'physical.npz') as z: atoms = z['atom']
        if patches.shape != (snapshot['count'], 80, 3) or len(atoms) != len(patches):
            raise ValueError(f'Unexpected dense neighborhoods: {key}')
        spaces = {'neural': {}, 'descriptors': {}}
        inputs = {str(data/'patches.npy'): sha(data/'patches.npy'), str(data/'physical.npz'): sha(data/'physical.npz')}
        loaded = {}
        for model in payload['md']['models']:
            identity = model['id'].removesuffix('-'+model['representation'])
            filename = (key+'-' if dataset == 'matched' else '')+identity+'-labels.npz'
            path = data/filename
            if path not in loaded:
                with np.load(path) as z: loaded[path] = dict(z)
                inputs[str(path)] = sha(path)
            spaces['neural'][model['id']] = loaded[path][model['representation']]
        path = data/'descriptor-labels.npz'
        with np.load(path) as z: spaces['descriptors'] = dict(z)
        inputs[str(path)] = sha(path)
        result = dict(neural={}, descriptors={}, patches={})
        palette = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2']
        for kind, families in spaces.items():
            for identity, labels in families.items():
                if labels.shape != (len(patches),) or np.any((labels<0)|(labels>6)):
                    raise ValueError(f'Invalid saved cluster labels: {key}/{identity}')
                clusters = []
                for cluster in range(7):
                    members = np.flatnonzero(labels == cluster)
                    seed = int.from_bytes(hashlib.sha256(f'20260929/{key}/{kind}/{identity}/{cluster}'.encode()).digest()[:8], 'little')
                    chosen = np.random.default_rng(seed).choice(members, size=min(5,len(members)), replace=False)
                    clusters.append(dict(count=len(members), rows=chosen.tolist()))
                    for row in chosen:
                        row = int(row); xyz = np.asarray(patches[row])
                        if str(row) in result['patches']: continue
                        if not np.isfinite(xyz).all() or not np.allclose(xyz[0], 0, rtol=0, atol=1e-6):
                            raise ValueError(f'Invalid centered atom neighborhood: {key}/{row}')
                        radius = np.sort(np.linalg.norm(xyz, axis=1)); cutoff = float(1.2*np.median(radius[1:13]))
                        oriented, edges, _ = sparse_geometry(xyz, cutoff, orientation='pca')
                        colors = [radial_colors(oriented, base) for base in palette]
                        result['patches'][str(row)] = dict(atom=int(atoms[row]), xyz=oriented.tolist(), edges=edges,
                            point_colors=[[mcolors.to_hex(c) for c in shade] for shade in colors],
                            edge_colors=[[mcolors.to_hex(.78*.5*(shade[a]+shade[b])) for a,b in edges] for shade in colors])
                result[kind][identity] = clusters
        manifest[key] = dict(neural={}, descriptors={}); assets = {}
        for kind in ('neural', 'descriptors'):
            for identity, clusters in result[kind].items():
                asset_key = f'{key}-{kind}-{identity}'; asset = folder/(asset_key+'.js')
                selected = {str(row) for cluster in clusters for row in cluster['rows']}
                write_asset(asset, asset_key, dict(clusters=clusters, patches={row:result['patches'][row] for row in selected}), 'CLUSTER_SAMPLES')
                manifest[key][kind][identity] = dict(key=asset_key, asset='../sample-data/'+asset.name)
                assets[asset.name] = sha(asset)
        provenance[key] = dict(inputs=inputs, asset_sha256=assets, distinct_examples=len(result['patches']))
        print(f'Exported {key}: {len(result["patches"])} real 80-atom environments', flush=True)
    suffix = '-'+snapshot_key if snapshot_key is not None else ''
    write_json(folder/f'manifest{suffix}.json', manifest)
    receipt = dict(source=str(source), snapshots=provenance,
        selection='up to five uniform draws without replacement per cluster, deterministic identity-derived seed',
        coordinates='original centered 80-atom input in angstroms, PCA oriented for display; first atom is the center',
        display_producer=str(Path(sparse_geometry.__code__.co_filename)), display_producer_sha256=sha(sparse_geometry.__code__.co_filename),
        display_style='existing analysis radial_colors and sparse_geometry; adaptive 1.2×median focal-12 distance; core edges and mutual-2NN outer connections',
        scope='entire selected MD snapshot; independent of projection filters and MD z slab',
        representative_centroids=False, neural_training=False, cluster_refit=False, implementation_sha256=sha(__file__))
    write_json(dest/f'technical/rendering/environment-samples{suffix}.json', receipt)
    return manifest, receipt


def parallel_export(source, publication, dataset, workers):
    if dataset != 'matched': raise ValueError('Parallel timeline export expects held-out MD')
    snapshots = json.loads((Path(source)/'technical/manifest.json').read_text())['snapshots']
    with ProcessPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(export, source, publication, dataset, s['key']) for s in snapshots]
        results = [task.result() for task in tasks]
    manifest = {k:v for entries,_ in results for k,v in entries.items()}
    receipt = dict(results[0][1], snapshots={k:v for _,r in results for k,v in r['snapshots'].items()})
    dest = Path(publication)
    write_json(dest/'sample-data/manifest.json', manifest)
    write_json(dest/'technical/rendering/environment-samples.json', receipt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--publication', required=True)
    parser.add_argument('--dataset', choices=['matched', 'static'], required=True)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.workers > 1: parallel_export(args.source, args.publication, args.dataset, args.workers)
    else: export(args.source, args.publication, args.dataset)
