"""Inspect storage and prepare verified deletion of our completed inference caches."""

from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

from .registry import checked_path, files_under, prune, sha256, write_json


def local_roots(repo, roots):
    selected = [repo / name for name in (roots or ['output', 'outputs'])]
    for root in selected:
        if not root.is_dir():
            raise FileNotFoundError(f'Storage root does not exist: {root}')
        checked_path(repo, str(root.relative_to(repo)))
    return selected


def inventory(repo, roots, destination, min_mib):
    if min_mib < 0:
        raise ValueError('--min-mib must be nonnegative')
    totals = Counter()
    largest = []
    for root in local_roots(repo, roots):
        for path in files_under(root):
            stat = path.stat()
            size = stat.st_blocks * 512
            totals[path.suffix or '(no extension)'] += size
            if size >= min_mib * 2**20:
                largest.append((size, path.relative_to(repo)))
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'large-files.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('allocated_mib', 'path', 'retention'))
        for size, path in sorted(largest, reverse=True):
            writer.writerow((round(size / 2**20, 2), str(path), 'keep unless a producer-specific verified plan permits deletion'))
    lines = ['# Local output storage', '', '| File type | Allocated GiB |', '| --- | ---: |']
    lines += [f'| {suffix} | {size/2**30:.3f} |' for suffix, size in totals.most_common()]
    lines += ['', '[Large files](large-files.csv). Symlink targets and registry snapshots are excluded.', '',
              'Inference caches can be rebuilt from the selected checkpoint and exact analysis inputs. '
              'Test predictions support paired comparisons and must stay. Keep selected checkpoints, '
              'training scalers, source snapshots, trajectories, IDs, timelines and restarts. '
              'PNG duplicates can be avoided in future configs by disabling redundant figure sets; '
              'existing plots are not deleted based on size.']
    (destination / 'STORAGE.md').write_text('\n'.join(lines) + '\n')
    print(f'{sum(totals.values())/2**30:.2f} GiB; storage report: {destination / "STORAGE.md"}')


def _record(repo, path, reason):
    checked_path(repo, str(path.relative_to(repo)), retained=True)
    return dict(path=str(path.relative_to(repo)), bytes=path.stat().st_size,
                sha256=sha256(path), reason=reason)


def clean_caches(repo, roots, *, apply=False, inactive=False):
    """Only recognize the exact metadata schema produced by analysis.inference_cache."""
    if apply and not inactive:
        raise ValueError('--apply requires --inactive after checking that no analysis or queue is using these runs.')
    selected = local_roots(repo, roots)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')
    audit = repo / 'output/maintenance' / f'cache-cleanup-{stamp}'
    technical = audit / 'technical'
    technical.mkdir(parents=True)
    remove, required, skipped = [], {}, []
    for root in selected:
        for metadata in files_under(root):
            if {'maintenance', 'tracking', 'retained-cache-metadata'} & set(metadata.relative_to(repo).parts):
                continue
            if not metadata.name.endswith('.npz.meta.json'):
                continue
            data = metadata.with_name(metadata.name.removesuffix('.meta.json'))
            spec = json.loads(metadata.read_text())
            if 'spec' not in spec or spec['spec'].get('version') != 8:
                skipped.append((str(metadata.relative_to(repo)), 'not the current inference-cache producer'))
                continue
            if data.is_symlink():
                skipped.append((str(data.relative_to(repo)), 'linked cache; external storage is retained'))
                continue
            if not data.is_file():
                raise FileNotFoundError(f'Orphan inference-cache metadata: {metadata}; inspect its producer before cleanup.')
            # A completed analysis is the retained scientific result; topology caches sit one level below it.
            analysis = metadata.parent.parent if metadata.parent.name == 'topology' else metadata.parent
            metrics = analysis / 'analysis_metrics.json'
            if not metrics.is_file():
                skipped.append((str(data.relative_to(repo)), 'analysis has no completed metrics'))
                continue
            checkpoint = Path(spec['spec']['checkpoint']['path'])
            if not checkpoint.is_relative_to(repo / 'output') and not checkpoint.is_relative_to(repo / 'outputs'):
                skipped.append((str(data.relative_to(repo)), 'external checkpoint; use an explicit retention plan'))
                continue
            if checkpoint.resolve() != checkpoint:
                skipped.append((str(data.relative_to(repo)), 'redirected checkpoint; use an explicit retention plan'))
                continue
            if not checkpoint.is_file():
                skipped.append((str(data.relative_to(repo)), 'reconstruction checkpoint is unavailable'))
                continue
            identity = spec['spec']['checkpoint']
            stat = checkpoint.stat()
            if stat.st_size != identity['size_bytes'] or stat.st_mtime_ns != identity['mtime_ns']:
                skipped.append((str(data.relative_to(repo)), 'checkpoint identity changed since cache creation'))
                continue
            archived = technical / 'retained-cache-metadata' / metadata.relative_to(repo)
            archived.parent.mkdir(parents=True, exist_ok=True)
            archived.write_bytes(metadata.read_bytes())
            if sha256(archived) != sha256(metadata):
                raise RuntimeError(f'Cache provenance copy failed verification: {metadata}')
            for keep in (metrics, checkpoint, archived):
                if str(keep) not in required:
                    required[str(keep)] = _record(repo, keep, 'result or cache reconstruction provenance')
            for path in (data, metadata):
                remove.append(_record(repo, path, 'completed inference cache; metadata preserved in cleanup technical folder'))
    plan = technical / 'cleanup-plan.json'
    write_json(plan, dict(remove=remove, required=list(required.values())))
    lines = ['# Inference cache cleanup', '', f'{len(remove)//2} cache/sidecar pairs selected.', '',
             'Rebuild with the original analysis config and checkpoint, with `figure_set.figure_only=false`. '
             'The original metadata under `technical/retained-cache-metadata/` records the dataset, '
             'sample selection, batching, checkpoint and seed. Keep the original datasets and source snapshots. '
             'This command does not delete them or training caches. It does not verify scheduler quiescence.', '',
             '| File | Why retained |', '| --- | --- |']
    lines += [f'| {path} | {reason} |' for path, reason in skipped]
    (audit / 'README.md').write_text('\n'.join(lines) + '\n')
    # Reuse the maintained hash/size preflight, per-file recheck and durable unlink audit.
    result = prune(repo, plan, apply)
    print(f'{result["allocated_bytes"]/2**30:.3f} GiB {"removed" if apply else "can be reclaimed"}; {audit / "README.md"}')
    return result
