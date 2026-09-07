"""File-backed experiment catalogue and explicitly planned output pruning."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
from urllib.parse import quote


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.building')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def files_under(root: Path):
    """Never traverse dataset symlinks into external storage."""
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != 'registry' and not Path(directory, d).is_symlink())
        for name in sorted(names):
            path = Path(directory, name)
            if not path.is_symlink():
                yield path


def category(path: Path) -> str:
    name = path.name.lower()
    if path.suffix in {'.ckpt', '.pt', '.model'}:
        return 'checkpoints'
    if path.suffix in {'.png', '.jpg', '.svg', '.gif', '.pdf', '.html'}:
        return 'plots'
    if path.suffix == '.md':
        return 'reports'
    if path.suffix in {'.yaml', '.yml'} or ('config' in name and path.suffix == '.json'):
        return 'configs'
    if path.suffix in {'.csv', '.jsonl'} or (path.suffix == '.json' and any(
            key in name for key in ('metric', 'comparison', 'summary', 'results', 'evaluation'))):
        return 'metrics'
    if path.suffix == '.json':
        return 'metadata'
    if path.suffix in {'.log', '.out', '.err', '.wandb'}:
        return 'logs'
    if path.suffix in {'.py', '.sh', '.sbatch'}:
        return 'reproduction'
    return 'data'


def run_id(relative: Path) -> str:
    # Existing Hydra and experiment-runner containers hold multiple independent runs.
    first = relative.parts[0]
    containers = {'detached', 'experiments', 'synthetic_data', 'temporal_cache'}
    if len(relative.parts) > 2 and (first in containers or
            (len(first) == 10 and first[4] == '-' and first[7] == '-')):
        return '/'.join(relative.parts[:2])
    return first


def scalar_metrics(value, prefix=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from scalar_metrics(item, f'{prefix}.{key}' if prefix else key)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        # Lists remain in the original metric artifact; do not mix observations.
        yield prefix, value


def build(repo: Path) -> dict:
    output = repo / 'output'
    registry = output / 'registry'
    registry.mkdir(exist_ok=True)
    captured = datetime.now(timezone.utc).isoformat()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True)
    # Versioned configs can be reorganized later without losing today's exact bytes.
    snapshot_index = []
    for base in ('configs', 'experiments'):
        for path in files_under(repo / base):
            if path.suffix not in {'.json', '.yaml', '.yml', '.md'}:
                continue
            relative = path.relative_to(repo)
            target = registry / 'config_snapshot' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            snapshot_index.append({'source': str(relative), 'snapshot': str(target.relative_to(repo)),
                                   'sha256': sha256(target)})
    write_json(registry / 'config_snapshot/index.json', {
        'captured_at': captured, 'catalogue_commit': commit,
        'note': 'Current config/recipe snapshot, not evidence of historical run configuration.',
        'files': snapshot_index})
    (registry / 'working_tree_status.txt').write_text(dirty)
    groups = defaultdict(list)
    for path in files_under(output):
        if path == output / 'README.md':
            continue
        groups[run_id(path.relative_to(output))].append(path)
    entries = []
    for identifier, paths in sorted(groups.items()):
        artifacts = []
        metrics = []
        recorded_commits = []
        statuses = []
        for path in paths:
            stat = path.stat()
            kind = category(path)
            artifact = {'path': str(path.relative_to(repo)), 'kind': kind,
                        'bytes': stat.st_size, 'allocated_bytes': stat.st_blocks * 512}
            if kind in {'configs', 'checkpoints', 'reports', 'metrics', 'metadata', 'reproduction'}:
                artifact['sha256'] = sha256(path)
            if path.suffix == '.json' and kind in {'configs', 'metrics', 'metadata'}:
                try:
                    value = json.loads(path.read_text())
                except json.JSONDecodeError as error:
                    artifact['parse_error'] = str(error)
                else:
                    flat = dict(scalar_metrics(value))
                    for key, val in flat.items():
                        if key.split('.')[-1] in {'git_commit', 'git_sha', 'commit_hash'}:
                            recorded_commits.append({'source': artifact['path'], 'key': key, 'value': val})
                    if path.name == 'status.json':
                        statuses.append({'source': artifact['path'], 'value': value})
                    if kind == 'metrics' and len(metrics) < 30:
                        # Strict JSON: old scientific outputs sometimes contain NaN.
                        clean = {k: v for k, v in list(flat.items())[:40]
                                 if not isinstance(v, float) or __import__('math').isfinite(v)}
                        metrics.append({'source': artifact['path'], 'values': clean})
            artifacts.append(artifact)
        first = identifier.split('/')[0]
        kind = 'experiment'
        if first in {'synthetic_data', 'temporal_cache'}:
            kind = 'dataset'
        elif first in {'wandb', 'training_jobs', 'slurm_outputs'} or first.startswith(('ids_', 'cache_float16', 'simulation_audit')):
            kind = 'maintenance'
        recipe = repo / 'experiments' / identifier / 'README.md'
        entries.append({'id': identifier, 'kind': kind, 'artifacts': artifacts,
                        'recipe': str(recipe.relative_to(repo)) if recipe.is_file() else None,
                        'git_commits': recorded_commits, 'git_status': 'recorded' if recorded_commits else 'not recorded',
                        'statuses': statuses, 'metric_preview': metrics,
                        'bytes': sum(a['bytes'] for a in artifacts),
                        'allocated_bytes': sum(a['allocated_bytes'] for a in artifacts)})
    catalogue = {'schema_version': 1, 'generated_at': captured, 'catalogue_commit': commit,
                 'catalogue_working_tree_dirty': bool(dirty),
                 'note': 'Catalogue commit is not a historical experiment commit. Original output paths are preserved.',
                 'experiments': entries}
    write_json(registry / 'experiments.json', catalogue)
    render(registry, catalogue)
    print(f'Indexed {len(entries)} entries, {sum(len(e["artifacts"]) for e in entries)} files', flush=True)
    return catalogue


def render(registry: Path, catalogue: dict) -> None:
    cards = []
    for entry in catalogue['experiments']:
        groups = defaultdict(list)
        for artifact in entry['artifacts']:
            groups[artifact['kind']].append(artifact)
        sections = []
        for kind in ('reports', 'configs', 'metrics', 'checkpoints', 'plots', 'metadata', 'reproduction', 'data', 'logs'):
            links = []
            for artifact in groups[kind]:
                path = artifact['path']
                url = '../../' + quote(path)
                label = path.removeprefix('output/' + entry['id'] + '/')
                link = f'<a href="{url}">{html.escape(label)}</a>'
                if kind == 'plots' and Path(path).suffix in {'.png', '.jpg', '.svg', '.gif'}:
                    link = f'<a href="{url}"><img loading="lazy" src="{url}" alt="{html.escape(label)}"></a>' + link
                links.append('<li>' + link + '</li>')
            if links:
                sections.append(f'<details><summary>{kind.title()} ({len(links)})</summary><ul class="{kind}">' + ''.join(links) + '</ul></details>')
        preview = html.escape(json.dumps(entry['metric_preview'], indent=2))
        recipe = '' if not entry['recipe'] else f'<a href="../../{quote(entry["recipe"])}">Research recipe</a> · '
        statuses = html.escape(json.dumps(entry['statuses']))
        cards.append(f'<article data-kind="{entry["kind"]}"><h2>{html.escape(entry["id"])}</h2>'
                     f'<p>{recipe}{entry["kind"]} · {entry["allocated_bytes"] / 2**30:.2f} GiB · '
                     f'Run git commit: {html.escape(entry["git_status"])}</p>'
                     f'<details><summary>Recorded status / metric preview</summary><p>{statuses}</p><pre>{preview}</pre></details>'
                     + ''.join(sections) + '</article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Materials experiment registry</title><style>
body{font:16px system-ui;background:#f4f6f8;color:#18232d;max-width:1200px;margin:30px auto;padding:0 20px}
header{position:sticky;top:0;background:#f4f6f8;padding:12px 0;z-index:1}h1{margin:8px 0}h2{font-size:19px;overflow-wrap:anywhere}
input,select{font:inherit;padding:10px;border:1px solid #bccbd8;border-radius:6px}input{width:60%}
article{background:white;border:1px solid #d9e1e8;border-radius:9px;padding:18px;margin:16px 0}
a{color:#17528d;overflow-wrap:anywhere}summary{cursor:pointer;padding:7px 0}li{margin:6px 0;overflow-wrap:anywhere}
.plots{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:15px;list-style:none;padding:0}.plots img{width:100%;height:190px;object-fit:contain}
pre{white-space:pre-wrap;max-height:400px;overflow:auto;font-size:12px}article[hidden]{display:none}
</style><header><h1>Materials experiment registry</h1>
<p>Configs, results, checkpoints and plots at their original paths. <a href="experiments.json">JSON registry</a> · <a href="../../docs/output_registry.md">Retention &amp; reproduction guide</a></p>
<input id="search" type="search" placeholder="Search experiment, model, metric or artifact…">
<select id="kind"><option value="experiment">Experiments</option><option value="">Everything</option><option value="dataset">Datasets</option><option value="maintenance">Maintenance</option></select>
<span id="count"></span></header>'''
    page += ''.join(cards) + '''<script>
const search=document.querySelector('#search'),kind=document.querySelector('#kind'),cards=[...document.querySelectorAll('article')];
function filter(){let n=0;const q=search.value.toLowerCase();for(const c of cards){c.hidden=!!((kind.value&&c.dataset.kind!==kind.value)||!c.textContent.toLowerCase().includes(q));if(!c.hidden)n++;}document.querySelector('#count').textContent=` ${n} entries`;}
search.addEventListener('input',filter);kind.addEventListener('change',filter);filter();
</script></html>'''
    (registry / 'index.html').write_text(page)


def checked_path(repo: Path, name: str) -> Path:
    path = repo / name
    if not path.is_relative_to(repo / 'output') or '..' in path.parts:
        raise ValueError(f'Cleanup path must be inside output/: {name}')
    if path.is_symlink() or path.resolve() != path or (repo / 'output/registry') in path.parents:
        raise ValueError(f'Refusing symlink, registry, or redirected cleanup path: {name}')
    return path


def prune(repo: Path, plan_path: Path, apply: bool) -> dict:
    plan = json.loads(plan_path.read_text())
    items = plan['remove']
    # Verify the complete plan before deleting anything, including retained prerequisites.
    for item in items + plan['required']:
        path = checked_path(repo, item['path'])
        if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
            raise RuntimeError(f'Artifact changed since cleanup plan: {path}')
    required = {item['path'] for item in plan['required']}
    removing = {item['path'] for item in items}
    if required & removing:
        raise ValueError(f'Cleanup removes a retained prerequisite: {required & removing}')
    for item in items:
        if not item['reason']:
            raise ValueError(f'Missing deletion reason: {item["path"]}')
    result = {'state': 'preview', 'files': len(items),
              'allocated_bytes': sum(checked_path(repo, i['path']).stat().st_blocks * 512 for i in items)}
    if apply:
        audit = repo / 'output/registry/cleanup_applied.jsonl'
        with audit.open('a') as log:
            for item in items:
                path = checked_path(repo, item['path'])
                # Recheck immediately before each unlink; an active writer invalidates the plan.
                if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
                    raise RuntimeError(f'Artifact changed during cleanup: {path}')
                path.unlink()
                log.write(json.dumps({'deleted_at': datetime.now(timezone.utc).isoformat(), **item}) + '\n')
                log.flush()
                os.fsync(log.fileno())
        result['state'] = 'complete'
        write_json(repo / 'output/registry/cleanup_result.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'prune'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--apply', action='store_true', help='Apply the verified explicit deletion plan; default is preview.')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    if args.command == 'build':
        build(repo)
    else:
        if args.plan is None:
            parser.error('prune requires --plan PATH')
        prune(repo, args.plan, args.apply)


if __name__ == '__main__':
    main()
