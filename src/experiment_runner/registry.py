"""File-backed experiment catalogue and explicitly planned output pruning."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
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
    if name.startswith('diagnostics-') and name.endswith('.tar.gz'):
        return 'logs'
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
    if first == 'runs' and len(relative.parts) >= 5:
        return '/'.join(relative.parts[:4])
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
            checksum = sha256(path)
            target = registry / 'config_snapshot/objects' / (checksum + path.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(path, target)
            if sha256(target) != checksum:
                raise RuntimeError(f'Config snapshot verification failed: {path}')
            snapshot_index.append({'source': str(relative), 'snapshot': str(target.relative_to(repo)),
                                   'sha256': checksum})
    config_capture = {
        'captured_at': captured, 'catalogue_commit': commit,
        'note': 'Current config/recipe snapshot, not evidence of historical run configuration.',
        'files': snapshot_index}
    write_json(registry / 'config_snapshot/index.json', config_capture)
    write_json(registry / 'config_snapshot/captures' / (captured.replace(':', '-') + '.json'), config_capture)
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
                    if path.name == 'status.json' or (path.name == 'run_record.json' and 'tracking' not in path.parts):
                        statuses.append({'source': artifact['path'], 'value': value,
                                         'observed_mtime': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
                    if kind == 'metrics' and len(metrics) < 30:
                        # Strict JSON: old scientific outputs sometimes contain NaN.
                        clean = {k: v for k, v in list(flat.items())[:40]
                                 if not isinstance(v, float) or math.isfinite(v)}
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
    settings_path = repo / 'experiments/registry.json'
    if settings_path.exists():
        settings = json.loads(settings_path.read_text())
        catalogue['external_runs'] = external_runs(repo, settings['storage_roots'])
        catalogue['ideas'] = json.loads((repo / 'experiments/ideas.json').read_text())['ideas']
    write_json(registry / 'experiments.json', catalogue)
    render(registry, catalogue)
    print(f'Indexed {len(entries)} entries, {sum(len(e["artifacts"]) for e in entries)} files', flush=True)
    return catalogue


def external_runs(repo: Path, roots: list[dict]) -> list[dict]:
    """Read-only summaries of IDS runs; large arrays are counted, never loaded."""
    records = []
    for item in roots:
        root = Path(item['path'])
        if not root.is_dir():
            raise FileNotFoundError(f'Registered storage root is unavailable: {root}')
        link = repo / 'output/registry/storage' / item['id']
        link.parent.mkdir(exist_ok=True)
        if not link.is_symlink():
            link.symlink_to(root, target_is_directory=True)
        elif link.resolve() != root.resolve():
            raise RuntimeError(f'Registry storage link targets a different root: {link}')
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or directory.is_symlink():
                continue
            total = 0
            count = 0
            states = Counter()
            links = []
            progress = []
            for path in files_under(directory):
                stat = path.stat()
                total += stat.st_blocks * 512
                count += 1
                relative = path.relative_to(root)
                if path.name in {'status.json', 'sequence_status.json', 'run_record.json'}:
                    value = json.loads(path.read_text())
                    if 'state' in value:
                        states[value['state']] += 1
                    if len(path.relative_to(directory).parts) <= 3:
                        progress.append({'path': str(path), 'recorded': value,
                                         'observed_mtime': datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat()})
                if (len(path.relative_to(directory).parts) <= 3 and
                    (path.suffix in {'.md', '.yaml', '.yml'} or path.name in {
                        'status.json', 'sequence_status.json', 'run_record.json', 'config.json',
                        'manifest.json', 'summary.json', 'active_submission.json', 'training_summary.json'})):
                    links.append({'path': str(Path('storage') / item['id'] / relative), 'name': str(path.relative_to(directory))})
            records.append({'id': item['id'] + '/' + directory.name, 'kind': item['kind'],
                            'path': str(directory), 'allocated_bytes': total, 'file_count': count,
                            'recorded_status_counts': dict(states), 'progress': progress, 'links': links})
    return records


def render(registry: Path, catalogue: dict) -> None:
    cards = []
    for idea in catalogue.get('ideas', []):
        cards.append('<article data-kind="idea"><h2>' + html.escape(idea['title']) + '</h2><p>'
                     + html.escape(idea['state'] + ' · ' + idea['id']) + '</p><p>'
                     + html.escape(idea['question']) + '</p><p>Next: ' + html.escape(idea['next_action'])
                     + '</p><a href="../../' + quote(idea['source']) + '">Evidence / research record</a></article>')
    for run in catalogue.get('external_runs', []):
        links = ''.join('<li><a href="' + quote(a['path']) + '">' + html.escape(a['name']) + '</a></li>' for a in run['links'])
        progress = html.escape(json.dumps(run['progress'], indent=2))
        cards.append('<article data-kind="' + run['kind'] + '"><h2>' + html.escape(run['id'])
                     + f'</h2><p>{run["allocated_bytes"]/2**30:.2f} GiB · {run["file_count"]} files · '
                     + 'Recorded statuses: ' + html.escape(json.dumps(run['recorded_status_counts']))
                     + '</p><p>Filesystem observation; scheduler liveness and scientific success are not inferred.</p>'
                     + '<details><summary>Progress and timestamps</summary><pre>' + progress + '</pre></details>'
                     + '<details><summary>Config, manifests, reports and status</summary><ul>' + links + '</ul></details></article>')
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
        primary = [s['value'] for s in entry['statuses'] if s['source'] == 'output/' + entry['id'] + '/status.json']
        if not primary:
            primary = [s['value'] for s in entry['statuses'] if s['source'] == 'output/' + entry['id'] + '/run_record.json']
        state = str(primary[0].get('state', 'see recorded status')) if primary else 'status not recorded at run root'
        cards.append(f'<article data-kind="{entry["kind"]}"><h2>{html.escape(entry["id"])}</h2>'
                     f'<p>{recipe}{entry["kind"]} · {entry["allocated_bytes"] / 2**30:.2f} GiB · '
                     f'Run git commit: {html.escape(entry["git_status"])}</p>'
                     f'<p>Recorded progress: <strong>{html.escape(state)}</strong></p>'
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
<select id="kind"><option value="experiment">Experiments</option><option value="simulation">Simulations</option><option value="idea">Ideas</option><option value="">Everything</option><option value="dataset">Datasets &amp; caches</option><option value="maintenance">Maintenance</option></select>
<span id="count"></span></header>'''
    page += ''.join(cards) + '''<script>
const search=document.querySelector('#search'),kind=document.querySelector('#kind'),cards=[...document.querySelectorAll('article')];
function filter(){let n=0;const q=search.value.toLowerCase();for(const c of cards){c.hidden=!!((kind.value&&c.dataset.kind!==kind.value)||!c.textContent.toLowerCase().includes(q));if(!c.hidden)n++;}document.querySelector('#count').textContent=` ${n} entries`;}
search.addEventListener('input',filter);kind.addEventListener('change',filter);filter();
</script></html>'''
    (registry / 'index.html').write_text(page)
    rows = ['# Run registry', '', '[Open the searchable dashboard](index.html)', '',
            f'Generated: {catalogue["generated_at"]}', '',
            '| Experiment | Allocated GiB | Files |', '| --- | ---: | ---: |']
    for entry in catalogue['experiments']:
        if entry['kind'] == 'experiment':
            rows.append(f'| [{entry["id"]}](../../output/{quote(entry["id"])}/) | {entry["allocated_bytes"]/2**30:.2f} | {len(entry["artifacts"])} |')
    (registry / 'README.md').write_text('\n'.join(rows) + '\n')


def checked_path(repo: Path, name: str, *, retained: bool = False) -> Path:
    path = repo / name
    if not path.is_relative_to(repo / 'output') or '..' in path.parts:
        raise ValueError(f'Cleanup path must be inside output/: {name}')
    if path.is_symlink() or path.resolve() != path or (not retained and (repo / 'output/registry') in path.parents):
        raise ValueError(f'Refusing symlink, registry, or redirected cleanup path: {name}')
    return path


def prune(repo: Path, plan_path: Path, apply: bool) -> dict:
    plan = json.loads(plan_path.read_text())
    items = plan['remove']
    # Verify the complete plan before deleting anything, including retained prerequisites.
    for item in items + plan['required']:
        path = checked_path(repo, item['path'], retained=item in plan['required'])
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


def pack_logs(repo: Path, before: str, apply: bool) -> list[dict]:
    """Losslessly pack old local experiment logs and W&B journals, not simulation logs."""
    cutoff = datetime.fromisoformat(before)
    if cutoff.tzinfo is None:
        raise ValueError('--before must include a UTC offset')
    stamp = cutoff.strftime('%Y%m%dT%H%M%S')
    catalogue = json.loads((repo / 'output/registry/experiments.json').read_text())
    result = []
    for entry in catalogue['experiments']:
        if entry['kind'] != 'experiment' and entry['id'] != 'wandb':
            continue
        paths = [repo / a['path'] for a in entry['artifacts'] if a['kind'] == 'logs'
                 and Path(a['path']).suffix in {'.log', '.out', '.err', '.wandb'}]
        paths = [p for p in paths if p.stat().st_mtime < cutoff.timestamp() and not p.is_symlink()]
        if not paths:
            continue
        directory = repo / 'output' / entry['id']
        archive_path = directory / f'diagnostics-before-{stamp}.tar.gz'
        if archive_path.exists():
            raise FileExistsError(f'Existing diagnostics archive: {archive_path}; rebuild registry before another packing pass')
        record = {'run': entry['id'], 'files': len(paths), 'archive': str(archive_path.relative_to(repo)),
                  'source_allocated_bytes': sum(p.stat().st_blocks * 512 for p in paths)}
        if apply:
            specs = {str(p.relative_to(directory)): sha256(p) for p in paths}
            with tarfile.open(archive_path, 'w:gz') as archive:
                for path in paths:
                    archive.add(path, arcname=str(path.relative_to(directory)), recursive=False)
            with tarfile.open(archive_path, 'r:gz') as archive:
                for name, expected in specs.items():
                    digest = hashlib.sha256()
                    with archive.extractfile(name) as handle:
                        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                            digest.update(block)
                    if digest.hexdigest() != expected:
                        raise RuntimeError(f'Packed log verification failed: {archive_path}: {name}')
            # Preserve member hashes next to the archive, including original paths.
            write_json(archive_path.with_suffix('.manifest.json'), {'members': specs, 'sha256': sha256(archive_path)})
            for path in paths:
                if sha256(path) != specs[str(path.relative_to(directory))]:
                    raise RuntimeError(f'Log changed while archiving; original retained: {path}')
            for path in paths:
                path.unlink()
            record['archive_allocated_bytes'] = archive_path.stat().st_blocks * 512
        result.append(record)
    write_json(repo / 'output/registry/log_archive_result.json', {'applied': apply, 'before': before, 'runs': result})
    print(f'{"Packed" if apply else "Would pack"} {sum(r["files"] for r in result)} old logs in {len(result)} archives', flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'prune', 'pack-logs', 'run', 'status', 'idea'])
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--spec', type=Path, help='Explicit run specification for the run command.')
    parser.add_argument('--record', type=Path, help='run_record.json for the status command.')
    parser.add_argument('--before', help='Timezone-aware ISO cutoff for packing old experiment logs.')
    parser.add_argument('--id', help='Existing idea ID to update.')
    parser.add_argument('--state', choices=['proposed', 'planned', 'running', 'blocked', 'completed', 'needs_review'])
    parser.add_argument('--next-action', help='Concrete next action for the idea.')
    parser.add_argument('--apply', action='store_true', help='Apply the verified explicit deletion plan; default is preview.')
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[2]
    if args.command == 'build':
        build(repo)
    elif args.command == 'idea':
        if not args.id or not args.state or not args.next_action:
            parser.error('idea requires --id, --state and --next-action')
        path = repo / 'experiments/ideas.json'
        backlog = json.loads(path.read_text())
        matches = [item for item in backlog['ideas'] if item['id'] == args.id]
        if len(matches) != 1:
            raise ValueError(f'Expected one registered idea with ID {args.id!r}, found {len(matches)}')
        item = matches[0]
        item.setdefault('history', []).append({'updated_at': datetime.now(timezone.utc).isoformat(),
                                               'previous_state': item['state'], 'state': args.state,
                                               'next_action': args.next_action})
        item.update(state=args.state, next_action=args.next_action)
        write_json(path, backlog)
        print(f'Updated idea {args.id}; run build to refresh the dashboard')
    elif args.command == 'pack-logs':
        if args.before is None:
            parser.error('pack-logs requires --before ISO_TIMESTAMP')
        pack_logs(repo, args.before, args.apply)
    elif args.command == 'run':
        if args.spec is None:
            parser.error('run requires --spec PATH')
        from .tracking import execute_spec
        execute_spec(args.spec)
    elif args.command == 'status':
        if args.record is None:
            parser.error('status requires --record PATH')
        from .tracking import observed_record
        print(json.dumps(observed_record(args.record), indent=2))
    else:
        if args.plan is None:
            parser.error('prune requires --plan PATH')
        prune(repo, args.plan, args.apply)


if __name__ == '__main__':
    main()
