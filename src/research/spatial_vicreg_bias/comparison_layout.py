"""Refresh comparison-page layout without recalculating scientific artifacts."""
import argparse
import html
import json
from pathlib import Path
import shutil

from src.data.fixed_cohort.protocol import sha, write_json


def dataset_navigation(selected):
    choices = [
        ('matched', 'Held-out Al MD', 'matched-al64-20260929/analyses/interface-pacmap-v1/interactive/comparison-all_test.html'),
        ('static', 'Al static · 166–240 ps', 'static-al-six-20260929/analyses/interface-v1/interactive/S1-seed17-epoch24-encoder-all_static.html'),
    ]
    options = ''.join(f'<option value="../../../../{path}"'+(' selected' if kind == selected else '')+f'>{label}</option>'
                      for kind, label, path in choices)
    model = ''
    latest = Path(__file__).resolve().parents[3]/'configs/analysis/mace_rich_interface_20260929.json'
    if latest.exists():
        from src.project_runtime.paths import resolve_config
        frozen = resolve_config(json.loads(latest.read_text()))
        target = Path(frozen['datasets'][selected]['publication'])/'index.html'
        if target.exists():
            import os
            origin = Path(frozen['datasets'][selected]['reference'])/'interactive'
            url = html.escape(os.path.relpath(target, origin))
            title = html.escape('MACE · step '+str(frozen['model']['update']))
            model = '<label>Model <select onchange="if(this.value)location.href=this.value"><option value="">GeoFormer</option><option value="'+url+'">'+title+'</option></select></label>'
    return '<nav class="controls"><label>Data <select id="dataset" onchange="location.href=this.value">'+options+'</select></label>'+model+'</nav>'


def refresh(publication, dataset, dense_source=None):
    dest = Path(publication).resolve()
    if dataset == 'matched':
        from .checkpoint_explorer import explorer_template
        template = explorer_template()
        pages = sorted((dest/'interactive').glob('comparison-*.html'))
        default = 'comparison-all_test.html'
    else:
        template = Path(__file__).with_name('cluster_comparison.html').read_text().replace('__DATASET_NAV__', dataset_navigation('static'))
        pages = sorted((dest/'interactive').glob('*.html'))
        default = 'S1-seed17-epoch24-encoder-all_static.html'
    if not pages: raise ValueError(f'No saved comparison pages: {dest}')
    script = Path(__file__).with_name('cluster_comparison.js')
    records = {}
    for page in pages:
        original = page.read_text()
        payload_text = original.split('const D=', 1)[1].split(';\nconst palette', 1)[0]
        payload = json.loads(payload_text)
        if 'paired' not in payload or 'matching' not in payload: raise ValueError(f'Not a paired comparison: {page}')
        if dense_source is not None:
            if dataset != 'matched': raise ValueError('Dense timeline override applies to held-out MD')
            dense = Path(dense_source)
            snapshots = json.loads((dense/'technical/manifest.json').read_text())['snapshots']
            for model in payload['md']['models']:
                identity = model['id'].removesuffix('-'+model['representation'])
                expected = json.loads((dense/'technical'/f'908-768-{identity}.json').read_text())['checkpoint_sha256']
                snapshot_records = {}
                for snapshot in snapshots:
                    key = snapshot['key']+'-'+identity
                    record = json.loads((dense/'technical'/f'{key}.json').read_text())
                    if record['checkpoint_sha256'] != expected: raise ValueError(f'Wrong timeline checkpoint: {key}')
                    if sha(dest/'md-data'/Path(record['asset']).name) != record['asset_sha256']: raise ValueError(f'Changed timeline asset: {key}')
                    snapshot_records[snapshot['key']] = dict(key=key, asset=record['asset'])
                model['snapshots'] = snapshot_records
            payload['md']['snapshots'] = snapshots
        samples = dest/'sample-data/manifest.json'
        if samples.exists(): payload['samples'] = json.loads(samples.read_text())
        for extra in ('lattice','travel'):
            manifest=dest/(extra+'-data')/'manifest.json'
            if manifest.exists():payload[extra]=json.loads(manifest.read_text())
        for kind,field in [('sample','samples'),('travel','travel')]:
            compact=dest/(kind+'-compact')/'manifest.json'
            if compact.exists():payload[field]=json.loads(compact.read_text())
        payload_text = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c')
        rendered = template.replace('__TITLE__', html.escape(payload.get('title', 'GeoFormer versus descriptors'))).replace(
            '__DATA__', payload_text).replace('__SCRIPT_HASH__', sha(script)[:16]).replace('__EXTENSION_HASH__', sha(Path(__file__).with_name('viewer_extensions.js'))[:16])
        if page == dest/'interactive'/default:
            index = dest/'index.html'; archive = dest/'technical/rendering/history'/('index-'+sha(index)+'.html')
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists(): shutil.copy2(index, archive)
            index.write_text(rendered.replace('<head>', '<head><base href="./interactive/">', 1))
        records[page.name] = dict(previous_html_sha256=sha(page), scientific_payload_unchanged=True,
                                 sample_assets_added=samples.exists(), dense_timeline_added=dense_source is not None)
        temporary = page.with_suffix('.html.building'); temporary.write_text(rendered); temporary.replace(page)
        records[page.name]['html_sha256'] = sha(page)
    shutil.copy2(script, dest/'assets'/script.name)
    shutil.copy2(Path(__file__).with_name('viewer_extensions.js'), dest/'assets/viewer_extensions.js')
    write_json(dest/'technical/rendering/compact-layout.json', dict(operation='layout only', dataset=dataset,
        template_sha256=sha(Path(__file__).with_name('cluster_comparison.html')), script_sha256=sha(script),
        implementation_sha256=sha(__file__), neural_training=False, metric_recomputation=False,
        index_sha256=sha(dest/'index.html'), pages=records))
    print(f'Updated {len(pages)} pages and index: {dest}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--publication', required=True); parser.add_argument('--dataset', choices=['matched', 'static'], required=True)
    parser.add_argument('--dense-source')
    args = parser.parse_args(); refresh(args.publication, args.dataset, args.dense_source)
