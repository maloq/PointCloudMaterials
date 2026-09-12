"""Verified relocation of explicit, immutable repository training caches."""

import hashlib
import json
import os
from pathlib import Path
import shutil


def _inventory(root):
    records = {}
    for path in sorted(root.rglob('*')):
        name = str(path.relative_to(root))
        if path.is_symlink():
            records[name] = {'link': os.readlink(path)}
        elif path.is_file():
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            records[name] = {'bytes': path.stat().st_size, 'sha256': digest}
    return records


def relocate_caches(plan_path, apply=False):
    plan = json.loads(Path(plan_path).read_text())
    audit = Path(plan['audit'])
    if apply:
        audit.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for item in plan['moves']:
        source, destination = Path(item['source']).absolute(), Path(item['destination']).absolute()
        if source.is_symlink() and source.resolve() == destination:
            results.append(dict(**item, state='already_relocated'))
            continue
        if source.is_symlink() or destination.exists():
            raise FileExistsError(f'Relocation requires an ordinary source directory and a new destination: {source} -> {destination}')
        before = _inventory(source)
        if not before:
            raise ValueError(f'No cache files found at {source}')
        result = dict(**item, files=before, bytes=sum(r.get('bytes',0) for r in before.values()), state='preview')
        results.append(result)
        if apply:
            shutil.copytree(source, destination, symlinks=True)
            if _inventory(destination) != before or _inventory(source) != before:
                raise RuntimeError(f'Cache changed or failed verification during relocation: {source}. Source retained.')
            result['state'] = 'copy_verified'
            audit.write_text(json.dumps(results, indent=2)+'\n')
            # No active writers may use a cache selected for relocation.
            shutil.rmtree(source)
            source.symlink_to(destination, target_is_directory=True)
            if not source.is_dir() or source.resolve() != destination:
                raise RuntimeError(f'Compatibility path did not resolve after relocation: {source}')
            result['state'] = 'complete'
            audit.write_text(json.dumps(results, indent=2)+'\n')
        print(f'{result["state"]}: {source} -> {destination} ({result["bytes"]/2**30:.3f} GiB)', flush=True)
    return results
