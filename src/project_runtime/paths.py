"""Resolve explicitly named storage roots without changing scientific settings."""

import json
import os
from pathlib import Path
import re
from contextlib import contextmanager
from contextvars import ContextVar

import yaml


REPO = Path(os.environ.get('PCM_PROJECT_ROOT', Path(__file__).resolve().parents[2])).absolute()
TOKENS = re.compile(r'\$\{(storage|dataset):([^{}]+)\}')
ROOT_DEFAULTS = {
    'repo': '.', 'datasets': 'datasets', 'simulations': 'data/simulations',
    'cache': 'data/cache', 'archive': 'data/archive', 'training_storage': 'data/archive/experiments',
    'analysis': 'output/analysis', 'output': 'output', 'scratch': 'tmp',
    'simulation_runs': 'tmp/simulations',
}


def machine():
    """Explicit environment selection, then ignored local settings, then portable defaults."""
    selected = os.environ.get('PCM_MACHINE_CONFIG')
    path = Path(selected).expanduser() if selected else REPO / 'machine.local.yaml'
    if selected and not path.is_file():
        raise FileNotFoundError(f'PCM_MACHINE_CONFIG does not exist: {path}')
    settings = yaml.safe_load(path.read_text()) if path.is_file() else {}
    unknown = set(settings) - {'roots', 'execution', 'legacy_paths', 'catalog'}
    if unknown:
        raise ValueError(f'Unknown machine settings in {path}: {sorted(unknown)}')
    roots = dict(ROOT_DEFAULTS, **settings.get('roots', {}))
    if set(roots) != set(ROOT_DEFAULTS):
        raise ValueError(f'Unknown storage roots in {path}: {set(roots)-set(ROOT_DEFAULTS)}')
    roots['repo'] = str(REPO)
    expanded = {}
    for key, value in roots.items():
        value = os.path.expandvars(value)
        if '$' in value:
            raise ValueError(f'Unexpanded environment variable in {path}, roots.{key}: {value}')
        root = Path(value).expanduser()
        expanded[key] = str(root if root.is_absolute() else REPO / root)
    return dict(roots=expanded,
                execution={'backend': 'local', 'device': 'cpu', 'lammps': 'lmp',
                           'mpi_launcher': [], 'mpi_environment': {},
                           **settings.get('execution', {})},
                legacy_paths=settings.get('legacy_paths', {}),
                catalog=settings.get('catalog', 'configs/datasets.json'))


def storage_path(name):
    roots = machine()['roots']
    if name not in roots:
        raise KeyError(f'Unknown storage root {name!r}; choose one of {sorted(roots)}')
    return Path(roots[name])


def catalog():
    filename = Path(machine()['catalog'])
    filename = filename if filename.is_absolute() else REPO / filename
    value = json.loads(filename.read_text())
    if value['schema_version'] != 1:
        raise ValueError(f'Unsupported dataset catalog schema: {filename}')
    return value['datasets']


def dataset_path(identifier):
    entries = catalog()
    if identifier not in entries:
        raise KeyError(f'Unknown dataset ID {identifier!r}; register it in the dataset catalog.')
    entry = entries[identifier]
    return storage_path(entry['root']) / entry['path']


class _PathSnapshot:
    """One consistent machine/catalogue read for a resolution operation."""

    def __init__(self):
        settings = machine()
        self.roots = settings['roots']
        filename = Path(settings['catalog'])
        filename = filename if filename.is_absolute() else REPO / filename
        value = json.loads(filename.read_text()) if filename.is_file() else None
        if value is not None and value['schema_version'] != 1:
            raise ValueError(f'Unsupported dataset catalog schema: {filename}')
        self.entries = value['datasets'] if value is not None else {}
        aliases = dict(settings['legacy_paths'])
        for identifier, entry in self.entries.items():
            for alias in entry.get('aliases', []):
                aliases[alias] = '${dataset:' + identifier + '}'
        self.aliases = sorted(aliases.items(), key=lambda item: -len(item[0]))

    def tokens(self, text):
        def replace(match):
            kind, key = match.groups()
            if kind == 'storage':
                return self.roots[key]
            entry = self.entries[key]
            return str(Path(self.roots[entry['root']]) / entry['path'])
        return TOKENS.sub(replace, text)

    def resolve(self, text):
        text = self.tokens(text)
        for prefix, base in [('datasets', Path(self.roots['datasets'])),
                             ('output', Path(self.roots['output'])),
                             *[(name, REPO / name) for name in ('experiments', 'configs', 'src', 'scripts')]]:
            if text.startswith(prefix + '/'):
                text = str(base / text[len(prefix)+1:])
                break
        for old, target in self.aliases:
            if text == old:
                text = self.tokens(target).rstrip('/')
            elif old.rstrip('/') + '/' in text:
                text = text.replace(old.rstrip('/') + '/', self.tokens(target).rstrip('/') + '/')
        return text


_path_snapshot = ContextVar('path_snapshot', default=None)


@contextmanager
def path_snapshot():
    """Reuse paths during one preparation operation, then observe future changes.

    Nesting shares the caller's snapshot. No process-global persistent cache is
    installed, so subsequent operations read the current machine and catalogue.
    """
    token = _path_snapshot.set(_path_snapshot.get() or _PathSnapshot())
    try:
        yield
    finally:
        _path_snapshot.reset(token)


def resolve_text(text):
    """Expand explicit tokens and the finite legacy roots declared by this machine."""
    return (_path_snapshot.get() or _PathSnapshot()).resolve(text)


def resolve_path(value):
    path = Path(resolve_text(str(value))).expanduser()
    return path if path.is_absolute() else REPO / path


@path_snapshot()
def resolve_config(value):
    """Repository JSON/YAML config trees: dictionaries, lists and scalar values."""
    if isinstance(value, dict):
        return {key: resolve_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_config(item) for item in value]
    if isinstance(value, str):
        return resolve_text(value)
    return value


@path_snapshot()
def load_json(path):
    return resolve_config(json.loads(resolve_path(path).read_text()))


@path_snapshot()
def portable_config(value):
    """Canonical path spelling for config comparisons, independent of mount points."""
    if isinstance(value, dict):
        return {key: portable_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_config(item) for item in value]
    if not isinstance(value, str):
        return value
    value = resolve_text(value)
    roots = _path_snapshot.get().roots
    for role, root in sorted(roots.items(), key=lambda item: -len(item[1])):
        token = '${storage:' + role + '}'
        if value == root:
            return token
        value = value.replace(root.rstrip('/') + '/', token + '/')
    return value


def register_resolvers():
    from omegaconf import OmegaConf
    for name, function in [('storage', lambda role: str(storage_path(role))),
                           ('dataset', lambda identifier: str(dataset_path(identifier)))]:
        if not OmegaConf.has_resolver(name):
            OmegaConf.register_new_resolver(name, function)
