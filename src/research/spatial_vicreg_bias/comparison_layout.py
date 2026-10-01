"""Refresh comparison-page layout without recalculating scientific artifacts."""
import argparse
import html
import json
from pathlib import Path
import shutil

from src.data.fixed_cohort.protocol import sha, write_json
from .viewer_payload import fill_template, read_payload, write_comparison


def dataset_navigation(selected):
    choices = [
        ('matched', 'Held-out Al MD', 'matched-al64-20260929/analyses/interface-pacmap-v1/interactive/comparison-all_test.html'),
        ('static', 'Al static · 166–240 ps', 'static-al-six-20260929/analyses/interface-v1/interactive/S1-seed17-epoch24-encoder-all_static.html'),
    ]
    options = ''.join(f'<option value="../../../../{path}"'+(' selected' if kind == selected else '')+f'>{label}</option>'
                      for kind, label, path in choices)
    model = ''
    latest = Path(__file__).resolve().parents[3]/'configs/analysis/mace_rich_current.json'
    if latest.exists():
        from src.project_runtime.paths import resolve_config
        frozen = resolve_config(json.loads(latest.read_text()))
        target = Path(frozen['datasets'][selected]['publication'])/'index.html'
        if target.exists():
            import os
            origin = Path(frozen['datasets'][selected]['reference'])/'interactive'
            url = html.escape(os.path.relpath(target, origin))
            title = html.escape(frozen['model']['title'])
            model = '<label>Model <select onchange="if(this.value)location.href=this.value"><option value="">GeoFormer</option><option value="'+url+'">'+title+'</option></select></label>'
    return '<nav class="controls"><label>Data <select id="dataset" onchange="location.href=this.value">'+options+'</select></label>'+model+'</nav>'


def comparison_template(navigation, **controls):
    slots = dict(DATASET_NAV=navigation, HEADLINE='Neural ↔ descriptor clusters', EXTRA_CONTROLS='', HEADER_NOTE='',
        NEURAL_CONTROLS='<label>Neural <select id="nnSpace"></select></label>',
        FRAME_CONTROLS='<label>Frame <select id="frame"><option value="all">All snapshots</option></select></label>',
        METHODS=Path(__file__).with_name('cluster_comparison_methods.html').read_text().rstrip())
    return fill_template(Path(__file__).with_name('cluster_comparison.html').read_text(), **(slots | controls))


def population_controls(frame_label='All frames'):
    return ('<label>Source <select id="source"><option value="all">All sources</option></select></label>'
        '<label>Frame <select id="frame"><option value="all">'+frame_label+'</option></select></label>')


def publish_run_entrypoint(publication, dataset):
    """Expose the complete saved viewer directly in its experiment output folder."""
    dest = Path(publication).resolve()
    if dest.parent.name != 'analyses':
        raise ValueError(f'Comparison bundle must be under RUN/analyses/: {dest}')
    run = dest.parent.parent
    target = run / {'matched': 'heldout-al.html', 'static': 'al-static.html'}[dataset]
    source = dest/'index.html'
    page = source.read_text()
    original_base = '<base href="./interactive/">'
    if page.count(original_base) != 1:
        raise ValueError(f'Expected one interactive asset base in {source}')
    asset_base = (dest.relative_to(run)/'interactive').as_posix()+'/'
    temporary = target.with_suffix('.html.building')
    temporary.write_text(page.replace(original_base, '<base href="'+html.escape(asset_base)+'">'))
    temporary.replace(target)
    shutil.copy2(source.with_suffix('.json'), target.with_suffix('.json'))
    write_json(dest/'technical/rendering/run-entrypoint.json', dict(
        operation='publish existing interactive viewer at experiment root', dataset=dataset,
        source=str(source), target=str(target), asset_base=asset_base,
        source_html_sha256=sha(source), html_sha256=sha(target),
        payload_sha256=sha(target.with_suffix('.json')), implementation_sha256=sha(__file__),
        neural_training=False, metric_recomputation=False, scientific_payload_unchanged=True))
    print(f'Interactive experiment page: {target}', flush=True)
    return target


def refresh(publication, dataset, dense_source=None):
    dest = Path(publication).resolve()
    primary = dest/'interactive'/('comparison-all_test.html' if dataset == 'matched' else 'comparison-all_static.html')
    if dataset == 'static' and not primary.exists():
        primary = dest/'interactive/S1-seed17-epoch24-encoder-all_static.html'
    frozen = read_payload(primary).get('frozen_model')
    if frozen:
        from .mace_checkpoint import read
        from .mace_publication import template as mace_template
        c, _ = read(Path(__file__).resolve().parents[3]/'configs/analysis/mace_rich_current.json')
        if c['model']['id'] != frozen['id']:
            raise ValueError('Viewer checkpoint differs from its frozen recipe')
        template = mace_template(c, dataset)
        pages = sorted((dest/'interactive').glob('comparison-*.html'))
        default = 'comparison-all_test.html' if dataset == 'matched' else 'comparison-all_static.html'
    elif dataset == 'matched':
        from .checkpoint_explorer import explorer_template
        template = explorer_template()
        pages = sorted((dest/'interactive').glob('comparison-*.html'))
        default = 'comparison-all_test.html'
    else:
        template = comparison_template(dataset_navigation('static'))
        pages = sorted((dest/'interactive').glob('*.html'))
        default = 'S1-seed17-epoch24-encoder-all_static.html'
    if not pages: raise ValueError(f'No saved comparison pages: {dest}')
    script = Path(__file__).with_name('cluster_comparison.js')
    records = {}
    for page in pages:
        if 'interface20' in page.stem:
            archive = dest/'technical/rendering/history'/('retired-'+sha(page)+'.html')
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists(): shutil.copy2(page, archive)
            full = page.name.replace('interface20', 'all_test' if dataset == 'matched' else 'all_static')
            page.write_text('<!doctype html><meta charset="utf-8"><title>Cluster comparison</title>'
                '<script>location.replace('+json.dumps(full)+'+location.search)</script>'
                '<a href="'+html.escape(full)+'">Open the full comparison</a>')
            records[page.name] = dict(retired_interface_view=True, full_view=full)
            continue
        payload = read_payload(page)
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
        general = dest/'technical/general-descriptors.json'
        if general.exists():
            from .general_descriptors import apply_overlay
            apply_overlay(payload, json.loads(general.read_text()))
        title = payload.get('title', 'GeoFormer versus descriptors')
        if page == dest/'interactive'/default:
            index = dest/'index.html'; archive = dest/'technical/rendering/history'/('index-'+sha(index)+'.html')
            archive.parent.mkdir(parents=True, exist_ok=True)
            if not archive.exists(): shutil.copy2(index, archive)
            write_comparison(index, template, payload, title=title, index=True)
        records[page.name] = dict(previous_html_sha256=sha(page), scientific_payload_unchanged=not general.exists(),
                                 all_training_descriptor_overlay=general.exists(),
                                 sample_assets_added=samples.exists(), dense_timeline_added=dense_source is not None)
        write_comparison(page, template, payload, title=title)
        records[page.name]['html_sha256'] = sha(page)
    shutil.copy2(script, dest/'assets'/script.name)
    shutil.copy2(Path(__file__).with_name('viewer_extensions.js'), dest/'assets/viewer_extensions.js')
    write_json(dest/'technical/rendering/compact-layout.json', dict(operation='layout only', dataset=dataset,
        template_sha256=sha(Path(__file__).with_name('cluster_comparison.html')), script_sha256=sha(script),
        implementation_sha256=sha(__file__), neural_training=False, metric_recomputation=False,
        index_sha256=sha(dest/'index.html'), pages=records))
    publish_run_entrypoint(dest, dataset)
    print(f'Updated {len(pages)} pages and index: {dest}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--publication', required=True); parser.add_argument('--dataset', choices=['matched', 'static'], required=True)
    parser.add_argument('--dense-source')
    args = parser.parse_args(); refresh(args.publication, args.dataset, args.dense_source)
