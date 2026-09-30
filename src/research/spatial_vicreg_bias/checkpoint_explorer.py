"""Publish existing matched GeoFormer checkpoints in a single descriptor explorer."""
import argparse
import html
from importlib.metadata import version
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlencode

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from src.data.fixed_cohort.protocol import sha, write_json
from src.experiment_runner.metric_docs import write_metric_table
from .cluster_matching import summarize
from .dense_md import read
from .comparison_layout import comparison_template, dataset_navigation, population_controls
from .viewer_payload import read_payload, write_asset, write_comparison


FIELDS = {'joint': 'Joint descriptor clusters', 'tda': 'TDA clusters',
          'bond_order': 'Bond-order clusters', 'cna': 'CNA clusters'}
COMMON_FIELDS = [*FIELDS.values(), 'PTM type', 'Physical region',
                 'Input crystal fraction', 'Interface distance (Å, clipped ±20)']
TITLES = {'joint': 'TDA + bond order + CNA', 'tda': 'TDA',
          'bond_order': 'Bond order', 'cna': 'CNA'}
DEFAULT = 'S1-seed17-epoch24-encoder'


def explorer_template():
    neural = ('<span>GeoFormer encoder</span><select id="nnSpace" hidden></select>'
        '<label>Checkpoint <select id="checkpoint"><option value="24">Epoch 24 · final</option>'
        '<option value="12">Epoch 12</option><option value="4">Epoch 4 · early</option></select></label>')
    advanced = ('<details><summary>Training variants and repeats</summary><div class="controls">'
        '<label>VICReg pairing <select id="recipe"><option value="1">Spatial neighbors</option>'
        '<option value="0.5">50% same center + 50% neighbors</option><option value="0">Same center only</option></select></label>'
        '<label>Training repeat <select id="repeat"><option value="17">Repeat 1 (seed 17)</option>'
        '<option value="29">Repeat 2 (seed 29)</option><option value="43">Repeat 3 (seed 43)</option></select></label>'
        '<label>Representation <select id="representation"><option value="encoder">Encoder embedding</option>'
        '<option value="projector">VICReg projector</option></select></label></div>'
        '<p>S0 = same-center views; S0.5 = equal same-center and neighbor alignment; S1 = neighbor alignment. '
        'These are training variants of the same architecture. Seeds are independent training repeats. '
        'Repeat 1 is a fixed example, not a selected winner.</p></details><div id="selection" class="status"></div>')
    methods = ('<details><summary>Methods and checkpoint</summary>'
        '<p>The same observations appear in both 3D plots. Colors maximize shared cluster membership by a one-to-one '
        'assignment on the 24,960 original held-out observations. This mapping stays fixed across filters, '
        'MD and PaCMAP. Shared colors identify assigned pairs; overlap and IoU show their actual correspondence.</p>'
        '<p>Frame numbers are saved-frame indices. PaCMAP layouts have independent axes; distances between separate '
        'layouts are not comparable. Correspondence uses displayed PaCMAP centers. The shared frame slider '
        'updates PaCMAP, the full 70,304-atom source-908 MD snapshot and local examples. MD z controls are '
        'independent of PaCMAP filters. Neural clusters and neural projected coordinates are preserved. Descriptor fitting uses all training environments.</p>'
        '<p>Encoder embeddings are the default; the optional projector is the training-loss head. '
        'Epoch 24 is the final checkpoint, epoch 12 is an intermediate checkpoint and epoch 4 is an early reference. '
        'This display makes no claim that any is scientifically best.</p>'
        '<p>Checkpoint: <code id="checkpointPath"></code></p>'
        '<p><a id="metricDefinitions" href="../../checkpoint-cluster-matching-v1/tables/METRICS.md">Metric definitions</a> · '
        '<a id="metricTable" href="../../checkpoint-cluster-matching-v1/tables/metrics.csv">Overlap table</a></p></details>')
    return comparison_template(dataset_navigation('matched'), HEADLINE='GeoFormer ↔ descriptors',
        NEURAL_CONTROLS=neural, EXTRA_CONTROLS=advanced, FRAME_CONTROLS=population_controls(), METHODS=methods)


def publish(config):
    c, pc, _, parent = read(config)
    dest = Path(c['publication']); dense = Path(c['output']); source = Path(pc['output'])
    roots = [Path(pc['inherit_descriptors_from']), source]
    views = {p.stem: (root, p) for root in roots for p in sorted((root/'technical/views').glob('*.json'))}
    assets = dest/'projection-data'; assets.mkdir(exist_ok=True)
    groups = {}; bindings = {}; old_links = {}; assignments = {}
    for name, (root, receipt) in sorted(views.items()):
        population = next(p for p in ('all_test', 'interface20') if name.endswith('-'+p))
        r = json.loads(receipt.read_text()); path = root/'data'/f'{name}.npz'
        if sha(path) != r['coordinate_sha256']: raise ValueError(f'Changed saved projection: {name}')
        original = root/'interactive'/f'{name}.html'
        payload = read_payload(original)
        with np.load(path) as z:
            ids = {k:z[k] for k in ('original_row', 'source', 'frame', 'atom')}; coordinates = z['pacmap3']
        for key in ('source', 'frame', 'atom'):
            if not np.array_equal(ids[key], payload[key]): raise ValueError(f'Changed identities: {name}/{key}')
        if not np.array_equal(np.round(coordinates, 6), np.asarray(payload['y3'], dtype=coordinates.dtype)):
            raise ValueError(f'Changed rendered coordinates: {name}')
        fields = {k:payload['fields'][k] for k in COMMON_FIELDS}
        metadata = {k:payload[k] for k in ('source', 'frame', 'atom', 'distance', 'solid', 'region')}
        if population not in groups:
            groups[population] = dict(ids=ids, fields=fields, metadata=metadata, neural=[], descriptors=[], labels={})
        group = groups[population]
        if any(not np.array_equal(ids[k], group['ids'][k]) for k in ids) or fields != group['fields'] or metadata != group['metadata']:
            raise ValueError(f'Unmatched observations or physical coloring: {name}')
        match = re.fullmatch(r'(S(0|0\.5|1)-seed(17|29|43))-epoch(\d+)-(encoder|projector)-'+population, name)
        asset = assets/(name+'.js'); data = dict(y3=np.round(coordinates, 6).tolist())
        if match:
            run, alpha, seed, epoch, rep = match.groups(); epoch = int(epoch)
            identity = f'{run}-epoch{epoch}-{rep}'; field = 'neural:'+identity
            assignment = Path(parent['output'])/run/'analyses'/f'epoch-{epoch:02d}'/'data'/f'{rep}-k7-assignments.npz'
            if identity not in assignments:
                if sha(assignment) != r['inputs'][str(assignment)]: raise ValueError(f'Changed assignments: {identity}')
                with np.load(assignment) as z: assignments[identity] = z['cluster']
            labels = assignments[identity][ids['original_row']]
            if not np.array_equal(labels, payload['fields']['Neural clusters']): raise ValueError(f'Changed labels: {name}')
            data['clusters'] = labels.tolist(); group['labels'][identity] = labels
            entry = dict(id=identity, title=f'GeoFormer · epoch {epoch}', field=field, alpha=float(alpha), seed=int(seed),
                epoch=epoch, representation=rep, checkpoint=str(Path(parent['output'])/run/'checkpoints'/f'epoch-{epoch:02d}.pt'))
            kind = 'neural'; old_links[name] = (population, identity, 'descriptors-joint')
        else:
            family = name.removesuffix('-'+population).removeprefix('descriptors-')
            entry = dict(id='descriptors-'+family, title=TITLES[family], field=FIELDS[family])
            kind = 'descriptors'; old_links[name] = (population, DEFAULT, entry['id'])
        write_asset(asset, name, data, 'PACMAP_LAYOUTS')
        entry.update(asset='../projection-data/'+asset.name, key=name); group[kind].append(entry)
        bindings[name] = dict(coordinate_sha256=r['coordinate_sha256'], original_payload_sha256=sha(original.with_suffix('.json')),
            inputs=r['inputs'], asset_sha256=sha(asset))
    if set(groups) != {'all_test', 'interface20'}: raise ValueError('Missing fixed population')
    for population, group in groups.items():
        if len(group['neural']) != 54 or len(group['descriptors']) != 4: raise ValueError(f'Incomplete saved views: {population}')
    matches = {}; metrics = {}
    for neural in groups['all_test']['neural']:
        identity = neural['id']; matches[identity] = {}; metrics[identity] = {}
        for family, field in FIELDS.items():
            tables = {}; diagnostics = {}
            for population, group in groups.items():
                a = group['labels'][identity].astype(np.int64); b = np.asarray(group['fields'][field], np.int64)
                if np.any((a<0)|(a>6)|(b<0)|(b>6)): raise ValueError('Expected original K=7 cluster IDs')
                tables[population] = np.bincount(a*7+b, minlength=49).reshape(7, 7)
                diagnostics[population] = float(adjusted_rand_score(a, b))
            reference = tables['all_test']
            if reference.sum() != 24960: raise ValueError('Changed color reference population')
            rows, mapping = linear_sum_assignment(reference, maximize=True)
            if not np.array_equal(rows, np.arange(7)): raise ValueError('Incomplete optimal assignment')
            matches[identity][family] = dict(neural_to_descriptor=mapping.tolist())
            metrics[identity][family] = {p:dict(**summarize(t, mapping), adjusted_rand_index=diagnostics[p]) for p,t in tables.items()}
    match_root = source.parent/'checkpoint-cluster-matching-v1'
    write_metric_table(metrics, match_root, family='checkpoint_cluster_matching')
    write_json(match_root/'technical/matches.json', matches)
    write_json(match_root/'technical/provenance.json', dict(reference='all_test', rows=24960, scipy_version=version('scipy'),
        inputs=bindings, implementation_sha256=sha(__file__), neural_training=False, cluster_refit=False, projection_refit=False))
    shutil.copytree(match_root, dest.parent/match_root.name, dirs_exist_ok=True)
    snapshots = json.loads((dense/'technical/manifest.json').read_text())['snapshots']; models = []
    for nn in groups['all_test']['neural']:
        run = nn['id'].split('-epoch')[0]; records = {}
        for snap in snapshots:
            key = f'{snap["key"]}-{run}-epoch{nn["epoch"]}'
            receipt = dense/'technical'/f'{key}.json'; record = json.loads(receipt.read_text())
            if sha(dest/'md-data'/Path(record['asset']).name) != record['asset_sha256']: raise ValueError(f'Changed MD asset: {key}')
            if record['checkpoint_sha256'] != bindings[nn['key']]['inputs'][nn['checkpoint']]: raise ValueError(f'Wrong MD checkpoint: {key}')
            records[snap['key']] = dict(asset=record['asset'], key=key)
        models.append(dict(id=nn['id'], representation=nn['representation'], snapshots=records))
    template = explorer_template(); script = Path(__file__).with_name('cluster_comparison.js')
    shutil.copy2(script, dest/'assets'/script.name)
    shutil.copy2(Path(__file__).with_name('viewer_extensions.js'), dest/'assets/viewer_extensions.js')
    prior = dest/'technical/rendering/checkpoint-explorer.json'
    if prior.exists():
        history = prior.parent/'history'; history.mkdir(exist_ok=True); shutil.copy2(prior, history/(sha(prior)+'.json'))
    pages = {}
    for population, group in groups.items():
        payload = dict(**group['metadata'], fields=group['fields'], explorer=dict(population=population), matching=matches,
            paired=dict(neural=group['neural'], descriptors=group['descriptors'], default_neural=DEFAULT, default_descriptor='descriptors-joint'),
            md=dict(models=models, snapshots=snapshots, default_model=DEFAULT))
        page = dest/'interactive'/f'comparison-{population}.html'
        write_comparison(page, template, payload, title='GeoFormer versus descriptors'); pages[population] = sha(page)
        if population == 'all_test':
            index = dest/'index.html'
            if not (dest/'technical/rendering/original-gallery.html').exists(): shutil.copy2(index, dest/'technical/rendering/original-gallery.html')
            write_comparison(index, template, payload, title='GeoFormer versus descriptors', index=True)
    for name, (population, neural, descriptor) in old_links.items():
        url = f'comparison-{population}.html?'+urlencode(dict(model=neural, descriptor=descriptor))
        (dest/'interactive'/f'{name}.html').write_text('<!doctype html><meta charset="utf-8"><title>GeoFormer comparison</title>'
            f'<meta http-equiv="refresh" content="0;url={html.escape(url, quote=True)}"><a href="{html.escape(url, quote=True)}">Open comparison</a>')
    write_json(prior, dict(source=str(source), config=str(config), implementation_sha256=sha(__file__), script_sha256=sha(script),
        template_sha256=sha(Path(__file__).with_name('cluster_comparison.html')), source_views=bindings, pages=pages,
        index_sha256=sha(dest/'index.html'), checkpoints=27, neural_spaces=54, descriptor_spaces=4, default_checkpoint=DEFAULT,
        default_policy='final epoch, spatial-neighbor recipe, fixed first repeat; no score or appearance selection',
        neural_training=False, projection_refit=False, added_metric_bundle=str(match_root)))
    (dest/'README.md').write_text('# GeoFormer versus descriptors\n\n[Open explorer](index.html). Choose epoch 4, 12 or 24. '
        'The default is spatial-neighbor VICReg, raw encoder, repeat 1 (seed 17). Other pairing recipes, repeats and '
        'the loss projector are under advanced controls. All 27 frozen checkpoints have matching full-snapshot MD coloring. '
        'The 116 original PaCMAP coordinate files and 2D figures are preserved.\n')
    print(f'Published checkpoint explorer: {dest / "index.html"}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__); parser.add_argument('--dense-config', required=True)
    publish(parser.parse_args().dense_config)
