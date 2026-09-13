"""Explicit, verified copies for portable bundles and research snapshots."""

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil

from src.experiment_runner.cache_storage import _inventory
from .paths import REPO, catalog, dataset_path, machine, resolve_path, resolve_text, storage_path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.building')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def verified_copy(source, destination, audit, *, move=False):
    source, destination, audit = Path(source).absolute(), Path(destination).absolute(), Path(audit)
    if source.is_symlink():
        raise ValueError(f'Pass the physical source directory, not its alias: {source}')
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f'Choose an unused destination: {destination}')
    if destination.is_relative_to(source) or audit.is_relative_to(source):
        raise ValueError('Destination and copy audit must be outside the source tree.')
    before = _inventory(source)
    if not before:
        raise ValueError(f'No files to copy: {source}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = sum(item.get('bytes', 0) for item in before.values())
    free = shutil.disk_usage(destination.parent).free
    if free < required:
        raise OSError(f'Copy needs {required} logical bytes; {free} available at {destination.parent}. Source retained.')
    record = dict(state='copying', source=str(source), destination=str(destination),
                  started_at=datetime.now(timezone.utc).isoformat(), files=before)
    write_json(audit, record)
    shutil.copytree(source, destination, symlinks=True)
    if _inventory(destination) != before or _inventory(source) != before:
        raise RuntimeError(f'Copy differs or source changed: {source}. Originals retained; audit: {audit}')
    record['state'] = 'verified'
    write_json(audit, record)
    if move:
        backup = source.with_name(source.name + '.verified-original')
        if backup.exists() or backup.is_symlink():
            raise FileExistsError(f'Previous relocation backup exists: {backup}')
        source.rename(backup)
        try:
            source.symlink_to(destination, target_is_directory=True)
        except OSError:
            backup.rename(source)
            raise
        if source.resolve() != destination.resolve():
            raise RuntimeError(f'Compatibility alias failed: {source}; original retained at {backup}')
        record.update(state='alias_installed', backup=str(backup))
        write_json(audit, record)
        shutil.rmtree(backup)
    record.update(state='complete', completed_at=datetime.now(timezone.utc).isoformat())
    write_json(audit, record)
    return record


def publish_simulation(source, *, identifier, move=False):
    """Publish a completed elemental campaign; do not reinterpret incomplete run state."""
    return _publish_run(source, identifier=identifier, move=move, expected_state='complete')


def archive_failed_simulation(source, *, identifier):
    """Copy stopped failure evidence and precise restart state; retain the working run."""
    return _publish_run(source, identifier=identifier, move=False, expected_state='failed')


def _publish_run(source, *, identifier, move, expected_state):
    if not identifier or Path(identifier).name != identifier or identifier in {'.', '..'}:
        raise ValueError(f'Dataset ID must be a single directory name: {identifier!r}')
    source = resolve_path(source).resolve()
    status = json.loads((source / 'status.json').read_text())
    if status['state'] != expected_state:
        raise RuntimeError(f'Only {expected_state} campaigns can use this operation: {source}, state={status["state"]}')
    destination = storage_path('archive') / 'simulations' / identifier
    audit = destination.parent / f'{identifier}.publication.json'
    entries = catalog()
    if identifier in entries:
        raise ValueError(f'Dataset ID is already registered: {identifier}')
    dependencies = set()
    launch = source / 'technical/launch_config.json'
    if launch.is_file():
        config = json.loads(launch.read_text())
        for potential in config['potential_files']:
            potential_path = resolve_path(potential['path']).resolve()
            matches = [(key, dataset_path(key).resolve()) for key in entries
                       if potential_path.is_relative_to(dataset_path(key).resolve())]
            if not matches:
                raise ValueError(f'Register the potential directory before publication: {potential_path}')
            dependencies.add(max(matches, key=lambda item: len(str(item[1])))[0])
    result = verified_copy(source, destination, audit, move=move)
    entry = dict(root='archive', path=f'simulations/{identifier}', kind='simulation',
                 state=expected_state, dependencies=sorted(dependencies), aliases=[str(source)] if move else [])
    catalog_path = resolve_path(machine()['catalog'])
    # Independent Slurm branches may finish together; serialize catalog updates.
    with catalog_path.with_suffix('.json.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = json.loads(catalog_path.read_text())
        if identifier in value['datasets']:
            raise ValueError(f'Dataset ID was registered during publication: {identifier}; verified copy retained at {destination}')
        value['datasets'][identifier] = entry
        write_json(catalog_path, value)
    return result


def snapshot(destination):
    """Copy the checkout including dirty/untracked files, preserving external links."""
    destination = Path(destination).absolute()
    if destination.is_relative_to(REPO):
        raise ValueError('A project snapshot must be outside the checkout.')
    audit = destination.with_name(destination.name + '.snapshot.json')
    result = verified_copy(REPO, destination, audit)
    external = []
    for relative, item in result['files'].items():
        if 'link' not in item:
            continue
        path = REPO / relative
        target = path.resolve()
        if not target.is_relative_to(REPO):
            external.append(dict(path=relative, target=str(target), exists=path.exists()))
    result['external_links'] = external
    result['storage'] = machine()['roots']
    result['note'] = 'Full checkout snapshot; external data is referenced, not duplicated. Use bundle for a self-contained selection.'
    write_json(audit, result)
    return result


def _selection(identifiers):
    entries = catalog()
    selected = set()
    def add(identifier):
        if identifier in selected:
            return
        if identifier not in entries:
            raise KeyError(f'Unknown dataset ID: {identifier}')
        selected.add(identifier)
        for dependency in entries[identifier]['dependencies']:
            add(dependency)
    for identifier in identifiers:
        add(identifier)
    return {identifier: entries[identifier] for identifier in sorted(selected)}


def bundle(plan_path, destination, *, apply=False):
    """Bundle explicitly selected data and checkpoint/config files with the source code."""
    plan = json.loads(resolve_path(plan_path).read_text())
    selected = _selection(plan['datasets'])
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    sources = {identifier: dataset_path(identifier).resolve() for identifier in selected}
    inventories = {identifier: _inventory(source) for identifier, source in sources.items()}
    for identifier, files in inventories.items():
        if not files:
            raise FileNotFoundError(f'Dataset {identifier} has no files at {sources[identifier]}')
    total = sum(r.get('bytes', 0) for inv in inventories.values() for r in inv.values())
    report = dict(state='preview', datasets=list(selected), logical_data_bytes=total,
                  files=plan['files'], destination=str(destination))
    if not apply:
        return report
    if destination.is_relative_to(REPO):
        raise ValueError('Create bundles outside the source checkout.')
    destination.mkdir(parents=True)
    # Only declared project sources and requested artifacts; no implicit full dataset copy.
    for name in ['src', 'scripts', 'configs', 'environments', 'docs']:
        shutil.copytree(REPO / name, destination / name,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    for name in ['README.md', 'requirements.txt']:
        shutil.copy2(REPO / name, destination / name)
    for name in plan['files']:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Bundle files must be checkout-relative: {name}')
        source, target = REPO / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True, dirs_exist_ok=True)
        else:
            shutil.copy2(source, target)
    new_entries = {}
    for identifier, entry in selected.items():
        target = destination / 'data' / 'bundle' / identifier
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(sources[identifier], target, symlinks=True)
        if _inventory(target) != inventories[identifier] or _inventory(sources[identifier]) != inventories[identifier]:
            raise RuntimeError(f'Bundle copy failed verification: {identifier}')
        new_entries[identifier] = dict(entry, root='repo', path=f'data/bundle/{identifier}',
                                      aliases=list(dict.fromkeys(entry.get('aliases', []) + [str(sources[identifier])])) )
    # Rewrite only bundle symlinks, preserving dataset manifest bytes.
    mappings = [(source, destination / 'data/bundle' / identifier) for identifier, source in sources.items()]
    mappings.append((REPO, destination))
    for directory, dirs, files in os.walk(destination, followlinks=False):
        for name in dirs + files:
            link = Path(directory) / name
            if not link.is_symlink():
                continue
            original = link.resolve()
            if original.is_relative_to(destination):
                if not link.exists():
                    raise ValueError(f'Bundle has an unselected relative-link dependency: {link} -> {original}')
                continue
            match = next(((old, new) for old, new in sorted(mappings, key=lambda pair: -len(str(pair[0])))
                          if original.is_relative_to(old)), None)
            if match is None:
                raise ValueError(f'Bundle needs an unselected dependency: {link} -> {original}. Add its dataset ID to the plan.')
            old, new = match
            target = new / original.relative_to(old)
            link.unlink()
            link.symlink_to(os.path.relpath(target, link.parent))
    write_json(destination / 'configs/datasets.json', dict(schema_version=1, datasets=new_entries))
    # Legacy config references retain their scientific bytes and resolve to bundled data.
    import yaml
    local = yaml.safe_load((REPO / 'configs/machines/local.yaml').read_text())
    local['legacy_paths'] = dict(machine()['legacy_paths'])
    local['legacy_paths'][str(REPO)] = '${storage:repo}'
    for identifier, source in sources.items():
        local['legacy_paths'][str(source)] = '${dataset:' + identifier + '}'
        for old, target in machine()['legacy_paths'].items():
            previous_root = Path(resolve_text(target))
            if source.is_relative_to(previous_root):
                alias = str(Path(old) / source.relative_to(previous_root))
                local['legacy_paths'][alias] = '${dataset:' + identifier + '}'
    (destination / 'machine.local.yaml').write_text(yaml.safe_dump(local, sort_keys=False))
    actual = _inventory(destination)
    report.update(state='complete', inventory=actual,
                  note='Use this machine.local.yaml or adapt it; all hashes refer to bundled files.')
    write_json(destination / 'bundle.json', report)
    verify_bundle(destination)
    return {key: value for key, value in report.items() if key != 'inventory'}


def verify_bundle(root):
    root = Path(root).absolute()
    manifest = json.loads((root / 'bundle.json').read_text())
    actual = _inventory(root)
    actual.pop('bundle.json')
    if actual != manifest['inventory']:
        raise RuntimeError(f'Bundle file/link inventory changed: {root}')
    for name, record in actual.items():
        if 'link' in record and (not (root/name).exists() or not (root/name).resolve().is_relative_to(root)):
            raise RuntimeError(f'Bundle has a broken or external alias: {root/name}')
    return dict(state='verified', files=len(actual), root=str(root))
