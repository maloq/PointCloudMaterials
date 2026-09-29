"""Publish named scientific analyses without changing their original evidence."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re
import uuid

from src.experiment_runner.artifacts import write_json
from src.experiment_runner.result_records import file_hash, href, identity, location, resolve_reference, save_record

VISUALS = {'.png', '.jpg', '.jpeg', '.svg', '.pdf', '.html', '.gif'}
SKIP = {'code', 'source', 'source_snapshot', 'frozen_source', 'tracking', 'wandb',
        '__pycache__', 'dataset-registry', 'analyses', 'metric-contracts', 'table-contracts',
        'collector-recovery-code', 'preflight-history'}


def section_for(relative):
    parts = relative.parts
    if parts[0] == 'snapshots' and len(parts) > 2:
        return f'snapshots/{parts[1]}'
    if parts[0].startswith('real_md') and len(parts) > 2:
        return parts[1].replace('_', '-')
    if parts[0] == 'plots' and len(parts) > 2:
        return parts[1]
    if relative.name.startswith('latent_'):
        return 'latent'
    if len(parts) > 1:
        return parts[0].replace('_', '-')
    return 'overview'


def artifact_role(path):
    if path.suffix.lower() in VISUALS:
        return 'figure'
    if path.name in {'METRICS.md', 'METRIC_INDEX.md'} or 'metric-definitions' in path.parts:
        return 'metric_definition'
    if 'metric-contract' in path.name or {'metric-contracts','table-contracts'} & set(path.parts):
        return 'metric_contract'
    if path.name in {'analysis_metrics.json', 'metrics.json', 'comparison.json'}:
        return 'metrics'
    if 'inference_cache' in path.name:
        return 'cache'
    if path.suffix == '.csv':
        return 'table'
    if path.suffix in {'.npz', '.npy'}:
        return 'scientific_data'
    if path.suffix in {'.pt', '.ckpt', '.model'}:
        return 'model'
    if path.suffix in {'.log', '.out', '.err', '.lock'}:
        return 'execution'
    if path.suffix in {'.json', '.yaml', '.yml', '.md'}:
        return 'provenance'
    return 'unclassified'


def inventory(source, bundle):
    artifacts = []
    for directory, dirs, files in os.walk(source, followlinks=False):
        parent = Path(directory)
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not d.startswith('.')
                         and not (parent/d).is_symlink() and (parent/d) != bundle)
        for name in sorted(files):
            path = parent/name
            if path == bundle.parent.parent/'run.json':
                # The destination receipt is rewritten after this inventory. It
                # cannot also be immutable evidence within its own artifact list.
                continue
            if path.is_symlink() or name.startswith('.'):
                continue  # Existing aliases do not create independent artifacts.
            relative = path.relative_to(source)
            role = artifact_role(relative)
            size = path.stat().st_size
            # Hash all visual evidence; never reread multi-GB caches/checkpoints for a gallery.
            checksum = file_hash(path) if role == 'figure' or size <= 16*1024**2 else None
            artifacts.append(dict(id=identity(location(path)),path=location(path),role=role,
                relative=str(relative),section=section_for(relative),bytes=size,sha256=checksum,
                verification='hashed' if checksum else 'indexed; large artifact not rehashed',
                retention='disposable_with_verified_prerequisites' if role=='cache' else 'preserve'))
    return artifacts


def original_contracts(source, bundle):
    """Link the definitions actually exported with these numbers, never current docs."""
    roots = [source]
    if source.name in {'technical', 'data'}:
        roots.append(source.parent)
    if source.name == 'archived_analysis':
        roots.append(source.parent.parent)
    artifacts = []
    for root in roots:
        for directory in ('tables', 'technical/metric-contracts', 'technical/table-contracts'):
            folder = root/directory
            if not folder.is_dir():
                continue
            for path in sorted(folder.rglob('*')):
                if path.is_file() and path.suffix in {'.md', '.csv', '.json'}:
                    artifacts.append(dict(id=identity(location(path)),path=location(path),
                        relative=str(path.relative_to(root)),role=artifact_role(path),section='tables',
                        bytes=path.stat().st_size,sha256=file_hash(path),verification='hashed',retention='preserve'))
        path = root/'technical/metric-contract.json'
        if path.is_file():
            artifacts.append(dict(id=identity(location(path)),path=location(path),relative='technical/metric-contract.json',
                role='metric_contract',section='provenance',bytes=path.stat().st_size,
                sha256=file_hash(path),verification='hashed',retention='preserve'))
    return artifacts


def publish_artifact_links(bundle, artifacts, *, include_paper_svg=False):
    """Expose selected visuals; HTML launchers preserve original relative dependencies."""
    for a in artifacts:
        if a['role'] != 'figure':
            continue
        relative = Path(a['relative'])
        original = resolve_reference(a['path'])
        if a.get('superseded_by') or (relative.name == 'cluster_proportions_stacked_area_paper.svg' and not include_paper_svg):
            a['visibility'] = 'superseded' if a.get('superseded_by') else 'opt_in'
            if 'published_path' in a:
                old = resolve_reference(a['published_path'])
                if old.is_symlink() and old.resolve() == original.resolve():
                    old.unlink()
                elif (old.is_file() and a.get('publication_mode') == 'html_launcher'
                      and file_hash(old) == a.get('published_sha256')):
                    old.unlink()
                elif old.exists():
                    raise FileExistsError(f'Cannot remove unrelated publication artifact: {old}')
                del a['published_path']
                a.pop('publication_mode',None)
                a.pop('published_sha256',None)
                if old.parent.is_dir() and not any(old.parent.iterdir()):
                    old.parent.rmdir()
            continue
        a.pop('visibility',None)
        target = bundle/'plots'/relative if len(relative.parts)>1 else bundle/'plots'/a['section']/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if relative.suffix.lower() == '.html':
            url = html.escape(href(a['path'],target.parent),quote=True)
            content = ('<!doctype html><meta charset="utf-8"><title>Interactive plot</title>'
                f'<meta http-equiv="refresh" content="0;url={url}">'
                f'<p><a href="{url}">Open the interactive plot</a></p>\n')
            if (target.exists() or target.is_symlink()) and (target.is_symlink() or target.read_text()!=content):
                raise FileExistsError(f'Publication would replace unrelated HTML: {target}')
            target.write_text(content)
            a.update(publication_mode='html_launcher',published_sha256=file_hash(target))
        elif target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve()!=original.resolve():
                raise FileExistsError(f'Publication would replace unrelated artifact: {target}')
        else:
            target.symlink_to(os.path.relpath(original,target.parent))
        a['published_path'] = location(target)


def render_bundle(bundle, analysis):
    groups = defaultdict(list)
    links = []
    interactive = []
    for artifact in analysis['artifacts']:
        if artifact.get('visibility') in ('opt_in','superseded'):
            continue
        target = artifact.get('published_path', artifact['path'])
        url = href(target, bundle)
        label = Path(artifact['relative']).stem.replace('_', ' ')
        if artifact['role'] == 'figure':
            is_html = Path(artifact['relative']).suffix.lower() == '.html'
            if is_html:
                interactive.append(f'<li><a href="{url}">{html.escape(artifact["section"]+" · "+label)}</a></li>')
            preview = (f'<img loading="lazy" src="{url}" alt="{html.escape(label)}">'
                       if Path(artifact['relative']).suffix.lower() in {'.png','.jpg','.jpeg','.svg','.gif'} else '')
            groups[artifact['section']].append(f'<figure><a href="{url}">{preview or ("Open interactive view" if is_html else "Open document")}</a>'
                f'<figcaption>{html.escape(label)}</figcaption><small>{html.escape(artifact["relative"])}</small></figure>')
        else:
            links.append(f'<li>{html.escape(artifact["role"])} · <a href="{url}">{html.escape(artifact["relative"])}</a>'
                         f' · {html.escape(artifact["verification"])}</li>')
    def order(s):
        return re.sub(r'\d+', lambda m:m[0].zfill(10), s)
    sections = []
    nav = [f'<a href="#interactive">Interactive views ({len(interactive)})</a>'] if interactive else []
    for n, name in enumerate(sorted(groups,key=order)):
        label = name.replace('snapshots/', 'Snapshot ').replace('-', ' ').title()
        nav.append(f'<a href="#section-{n}">{html.escape(label)} ({len(groups[name])})</a>')
        sections.append(f'<section id="section-{n}"><h2>{html.escape(label)}</h2><div class="figures">'
                        + ''.join(groups[name]) + '</div></section>')
    stages = ''.join(f'<li>{html.escape(name)}: <strong>{html.escape(value["state"])}</strong> — '
                     f'{html.escape(value.get("note", ""))}</li>' for name,value in analysis['stages'].items())
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title><style>
body{font:16px system-ui;margin:36px auto;max-width:1400px;padding:0 24px;background:#f4f6f8;color:#182e3b}
a{color:#18599a}p{max-width:1000px;line-height:1.65}nav{display:flex;flex-wrap:wrap;gap:12px;background:white;padding:18px;border-radius:8px}
.figures{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
figure{margin:0;padding:16px;background:white;border-radius:8px;border:1px solid #d8e0e8;overflow-wrap:anywhere}
img{width:100%;height:300px;object-fit:contain}figcaption{margin:12px 0}small{color:#526576}li{margin:8px 0}
section{margin-top:32px}summary{cursor:pointer;padding:12px;background:white}pre{white-space:pre-wrap}
input{font:inherit;padding:12px;margin:18px 0;width:min(90%,540px)}figure[hidden]{display:none}
</style><h1>__TITLE__</h1><p>__CONTEXT__</p><p><a href="analysis.json">Evaluation receipt</a> ·
<a href="artifacts.json">Complete artifact inventory</a></p><nav>__NAV__</nav>
<input id="search" placeholder="Filter views, representatives, clusters or frames">
__INTERACTIVE__
<details><summary>Analysis coverage and provenance</summary><ul>__STAGES__</ul><pre>__PROVENANCE__</pre></details>
__SECTIONS__<details><summary>Tables, scientific data, execution details and unclassified artifacts</summary><ul>__LINKS__</ul></details>
<script>document.querySelector('#search').oninput=e=>{for(const f of document.querySelectorAll('figure'))f.hidden=!f.textContent.toLowerCase().includes(e.target.value.toLowerCase());};</script></html>'''
    replacements = dict(TITLE=html.escape(analysis['title']),CONTEXT=html.escape(analysis['context']),
        NAV=''.join(nav),STAGES=stages,PROVENANCE=html.escape(json.dumps({k:analysis[k] for k in
        ('protocol','checkpoint_sha256','inputs','population','selection')},indent=2)),
        SECTIONS=''.join(sections),LINKS=''.join(links),
        INTERACTIVE=('<section id="interactive"><h2>Interactive views</h2>'
            '<p>Open a view to rotate, zoom and inspect its geometry.</p><ul>'+''.join(interactive)+'</ul></section>') if interactive else '')
    for key,value in replacements.items():
        page = page.replace('__'+key+'__',value)
    temporary = bundle/f'.index.{os.getpid()}.html'
    temporary.write_text(page)
    temporary.replace(bundle/'index.html')


def publish_bundle(source, destination, *, name='standard-v1', title=None, context=None,
                   study=None, checkpoint_sha256=None, stages=None, metadata=None, refresh=False,
                   numerical_file='analysis_metrics.json', components=None, activity='analysis', execution=None,
                   include_paper_svg=False):
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', name):
        raise ValueError(f'Analysis name must be a single readable path component: {name!r}')
    metrics_path = source/numerical_file
    if not metrics_path.is_file():
        raise FileNotFoundError(f'Standard analysis evidence missing: {metrics_path}')
    metrics = json.loads(metrics_path.read_text()) if metrics_path.suffix=='.json' else {}
    checkpoint_sha256 = checkpoint_sha256 or metrics.get('checkpoint_sha256') or metrics.get('topology',{}).get('checkpoint_sha256')
    bundle = destination/'analyses'/name
    bundle.mkdir(parents=True,exist_ok=True)
    evidence_id = identity(dict(checkpoint=checkpoint_sha256,metrics=file_hash(metrics_path),
                                metadata=metadata or {}))
    previous = bundle/'analysis.json'
    if previous.exists() and json.loads(previous.read_text())['id'] != evidence_id:
        raise ValueError(f'{bundle}: numerical evidence changed; choose a new analysis revision')
    artifacts = inventory(source,bundle) + original_contracts(source,bundle)
    artifacts = list({a['id']:a for a in artifacts}.values())
    # Carry existing aliases into the layout pass so opt-out can remove them.
    if previous.exists():
        old_artifacts = {a['id']:a for a in json.loads(previous.read_text())['artifacts']}
        for a in artifacts:
            if a['id'] in old_artifacts and 'published_path' in old_artifacts[a['id']]:
                a['published_path'] = old_artifacts[a['id']]['published_path']
            if a['id'] in old_artifacts and old_artifacts[a['id']].get('superseded_by'):
                a['superseded_by'] = old_artifacts[a['id']]['superseded_by']
    publish_artifact_links(bundle,artifacts,include_paper_svg=include_paper_svg)
    stages = stages or {'saved_analysis':dict(state='complete',evidence=location(metrics_path),
        note='Saved metrics exist; historical optional-stage completion is not inferred.'),
        'optional_stages':dict(state='unknown',note='Consult the original configuration and reports.')}
    values = metadata or {}
    analysis = dict(schema_version=1,id=evidence_id,title=title or destination.name,
        context=context or 'Descriptive analysis of saved embeddings. Consult the original protocol for population and limitations.',
        checkpoint_sha256=checkpoint_sha256,protocol=values.get('protocol','standard-analysis/historical'),
        inputs=values.get('inputs',{'state':'not recorded in publication; consult saved producer'}),
        population=values.get('population',{'state':'not recorded in publication'}),
        selection=values.get('selection',{'state':'historical selector; consult training record'}),
        stages=stages,artifacts=artifacts,page=location(bundle/'index.html'),
        source=location(source),numerical_evidence=dict(path=location(metrics_path),sha256=file_hash(metrics_path)))
    write_json(bundle/'analysis.json',analysis)
    write_json(bundle/'artifacts.json',dict(schema_version=1,analysis_id=evidence_id,artifacts=artifacts,
        counts=dict(Counter(a['role'] for a in artifacts))))
    write_json(bundle/'technical/rendering.json',dict(rendered_at=datetime.now(timezone.utc).isoformat(),
        producer_sha256=file_hash(__file__),analysis_id=evidence_id,mode='existing_evidence_only',
        include_paper_svg=include_paper_svg))
    render_bundle(bundle,analysis)
    if not (bundle/'README.md').exists():
        (bundle/'README.md').write_text(f'# {analysis["title"]}\n\n{analysis["context"]}\n\n'
            '[Open the grouped gallery](index.html) · [Evidence and stage coverage](analysis.json).\n')
    existing_run = destination/'run.json'
    previous_run = json.loads(existing_run.read_text()) if existing_run.exists() else {}
    run_id = previous_run.get('id') or 'report:'+uuid.uuid4().hex
    analyses = [a for a in previous_run.get('analyses',[]) if a.get('page') != analysis['page']] + [analysis]
    record = dict(schema_version=1,id=run_id,kind='research',activity=previous_run.get('activity',activity),title=previous_run.get('title',analysis['title']),
        study=study or previous_run.get('study',{}),
        execution=execution or previous_run.get('execution') or dict(state='historical',note='No training or inference executed by publication'),
        evidence=previous_run.get('evidence',dict(state='available',note='Coverage is recorded per analysis; optional stages may be unknown')),
        interpretation=previous_run.get('interpretation',dict(state='exploratory',note=analysis['context'])),analyses=analyses,
        components=components if components is not None else previous_run.get('components',[]))
    save_record(destination,record,refresh=refresh)
    if not (destination/'index.html').exists():
        (destination/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>{html.escape(analysis["title"])}</title>'
            f'<h1>{html.escape(analysis["title"])}</h1><p><a href="analyses/{name}/index.html">Open the analysis gallery</a></p>')
    return analysis


def refresh_publication_record(path, *, include_paper_svg=False):
    """Update navigation while preserving recorded identities, evidence and annotations."""
    path = Path(path).absolute()
    record = json.loads(path.read_text())
    for analysis in record['analyses']:
        bundle = resolve_reference(analysis['page']).parent
        publish_artifact_links(bundle,analysis['artifacts'],include_paper_svg=include_paper_svg)
        write_json(bundle/'analysis.json',analysis)
        write_json(bundle/'artifacts.json',dict(schema_version=1,analysis_id=analysis['id'],
            artifacts=analysis['artifacts'],counts=dict(Counter(a['role'] for a in analysis['artifacts']))))
        render_bundle(bundle,analysis)
        write_json(bundle/'technical/rendering.json',dict(rendered_at=datetime.now(timezone.utc).isoformat(),
            producer_sha256=file_hash(__file__),analysis_id=analysis['id'],mode='existing_evidence_only',
            include_paper_svg=include_paper_svg))
    save_record(path.parent,record,refresh=True)
    return dict(id=record['id'],analyses=len(record['analyses']))


def publish_plan(path, *, include_paper_svg=False):
    from src.project_runtime.paths import resolve_path
    plan = json.loads(Path(path).read_text())
    if plan['schema_version'] != 1:
        raise ValueError(f'Unsupported publication plan: {path}')
    results = []
    for item in plan['analyses']:
        components = []
        for component in item.get('components',[]):
            path = resolve_path(component['receipt'])
            saved = json.loads(path.read_text())
            components.append(dict(component,id=identity(dict(producer=saved['identity'],name=component['name'])),
                receipt_sha256=file_hash(path),recorded=saved))
        result = publish_bundle(resolve_path(item['source']),resolve_path(item['destination']),
            name=item['name'],title=item['title'],context=item['context'],study=item['study'],
            metadata=item['metadata'],numerical_file=item.get('numerical_file','analysis_metrics.json'),
            components=components,activity=item.get('activity','analysis'),stages=item.get('stages'),
            include_paper_svg=include_paper_svg)
        results.append(dict(id=result['id'],title=result['title'],artifacts=len(result['artifacts'])))
    from src.experiment_runner.result_records import refresh_results
    refresh_results()
    return results
