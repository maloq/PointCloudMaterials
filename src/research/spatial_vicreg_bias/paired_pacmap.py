"""Display saved neural and descriptor 3D PaCMAP side by side; no numerical refit."""
import argparse
import html
import json
from pathlib import Path
import shutil

import numpy as np

from src.data.fixed_cohort.protocol import sha, write_json
from .dense_md import write_asset
from .cluster_matching import build as build_matching
from .comparison_layout import dataset_navigation


def publish(source, publication):
    source = Path(source).resolve(); dest = Path(publication).resolve()
    template_path = Path(__file__).with_name('cluster_comparison.html')
    script_path = Path(__file__).with_name('cluster_comparison.js')
    template = template_path.read_text().replace('__DATASET_NAV__', dataset_navigation('static'))
    matching, matching_root = build_matching(source)
    shutil.copytree(matching_root, dest.parent/matching_root.name, dirs_exist_ok=True)
    shutil.copy2(script_path, dest/'assets/cluster_comparison.js')
    shutil.copy2(Path(__file__).with_name('viewer_extensions.js'), dest/'assets/viewer_extensions.js')
    prior = dest/'technical/rendering/paired-pacmap.json'
    if prior.exists():
        saved = dest/'technical/rendering/history'/('paired-pacmap-'+sha(prior)+'.json')
        saved.parent.mkdir(exist_ok=True); shutil.copy2(prior, saved)
    assets = dest/'projection-data'; assets.mkdir(exist_ok=True)
    receipts = sorted((source/'technical/views').glob('*.json'))
    pages = {}; populations = {}; bindings = {}
    for receipt in receipts:
        record = json.loads(receipt.read_text()); name = record['name']; population = record['population']
        data_path = source/'data'/f'{name}.npz'
        if sha(data_path) != record['coordinate_sha256']:
            raise ValueError(f'Changed saved projection: {name}')
        page_path = source/'interactive'/f'{name}.html'
        payload = json.loads(page_path.read_text().split('const D=', 1)[1].split(';\nconst palette', 1)[0])
        with np.load(data_path) as data:
            ids = {k: data[k] for k in ('sample_row', 'frame', 'atom', 'grid_row')}
            y3 = data['pacmap3']
            if not np.array_equal(data['frame'], payload['frame']) or not np.array_equal(data['atom'], payload['atom']):
                raise ValueError(f'Projection identity mismatch: {name}')
            if not np.array_equal(np.round(y3, 6), np.asarray(payload['y3'], dtype=y3.dtype)):
                raise ValueError(f'Changed rendered coordinates: {name}')
        if population in populations:
            if any(not np.array_equal(ids[k], populations[population]['ids'][k]) for k in ids):
                raise ValueError(f'Cannot pair different observations: {name}')
            if payload['fields'] != populations[population]['fields']:
                raise ValueError(f'Inconsistent coloring of the same observations: {name}')
        else:
            populations[population] = dict(ids=ids, fields=payload['fields'], neural=[], descriptors=[])
        kind = 'descriptors' if name.startswith('descriptors-') else 'neural'
        identity = name.removesuffix('-'+population)
        if kind == 'neural':
            model = next(m for m in payload['md']['models'] if m['id'] == identity)
            title = model['title']; field = title
        else:
            family = identity.removeprefix('descriptors-')
            field = {'joint': 'Joint descriptor clusters', 'tda': 'TDA clusters',
                     'bond_order': 'Bond-order clusters', 'cna': 'CNA clusters'}[family]
            title = field
        if field not in payload['fields']: raise ValueError(f'Missing color field: {name}')
        asset = assets/(name+'.js')
        write_asset(asset, name, dict(y3=np.round(y3, 6).tolist()), 'PACMAP_LAYOUTS')
        populations[population][kind].append(dict(id=identity, title=title, field=field,
            asset='../projection-data/'+asset.name, key=name))
        pages[name] = (payload, population, kind, identity)
        bindings[name] = dict(coordinate_sha256=record['coordinate_sha256'],
            original_html_sha256=sha(page_path), asset_sha256=sha(asset))
    for name, (payload, population, kind, identity) in pages.items():
        group = populations[population]
        if len(group['neural']) != 4 or len(group['descriptors']) != 4:
            raise ValueError(f'Incomplete paired static feature spaces: {population}')
        payload['paired'] = {k: group[k] for k in ('neural', 'descriptors')}
        payload['paired'].update(default_neural=identity if kind == 'neural' else payload['md']['default_model'],
                                 default_descriptor=identity if kind == 'descriptors' else 'descriptors-joint')
        payload['md']['default_model'] = payload['paired']['default_neural']
        payload['matching'] = matching
        payload.pop('y2'); payload.pop('y3')
        payload['title'] = 'Neural and non-neural 3D PaCMAP · '+population
        encoded = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
        path = dest/'interactive'/f'{name}.html'; temporary = path.with_suffix('.html.building')
        temporary.write_text(template.replace('__TITLE__', html.escape(payload['title'])).replace('__DATA__', encoded)
                             .replace('__SCRIPT_HASH__', sha(script_path)[:16]).replace('__EXTENSION_HASH__', sha(Path(__file__).with_name('viewer_extensions.js'))[:16]))
        temporary.replace(path); bindings[name]['published_html_sha256'] = sha(path)
    path = dest/'index.html'
    rows = []
    for name, (payload, population, kind, identity) in pages.items():
        rows.append(f'<tr><td>{html.escape(identity)}</td><td>{population}</td>'
            f'<td><a href="interactive/{name}.html">3D comparison + overlap</a></td>'
            f'<td><a href="plots/{name}.png">2D figure</a></td></tr>')
    path.write_text('<!doctype html><meta charset="utf-8"><title>Al cluster comparison</title>'
        '<style>body{font:16px system-ui;margin:32px;max-width:1350px}td,th{text-align:left;padding:10px;border-bottom:1px solid #ddd}table{border-collapse:collapse}a{color:#145bc0}</style>'
        '<h1>Al: neural and descriptor clusters</h1><p>166 · 170 · 174 · 175 · 177 · 240 ps</p>'
        '<p><a href="RESULTS.md">Findings</a> · <a href="tables/metrics.csv">Correspondence metrics</a> · '
        '<a href="../cluster-color-matching-v1/tables/metrics.csv">Optimal-match metrics</a></p>'
        '<table><tr><th>Default feature space</th><th>Population</th><th>Interactive</th><th>Saved figure</th></tr>'+''.join(rows)+'</table>')
    write_json(dest/'technical/rendering/paired-pacmap.json', dict(source=str(source),
        operation='saved 3D coordinates; optimal color assignment and overlap display',
        neural_training=False, projection_refit=False, original_metric_recomputation=False,
        added_metric_bundle=str(matching_root),
        atom_linkage=False, implementation_sha256=sha(__file__), template_sha256=sha(template_path),
        script_sha256=sha(script_path), matching_sha256=sha(matching_root/'technical/matches.json'),
        index_sha256=sha(path), views=bindings))
    print(f'Published {len(pages)} paired 3D views: {dest}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--publication', required=True)
    args = parser.parse_args(); publish(args.source, args.publication)
