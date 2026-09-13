"""Resolve explicitly named storage roots without changing scientific settings."""

import json
import os
from pathlib import Path
import re

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


def _tokens(text):
    return TOKENS.sub(lambda match: str(storage_path(match[2]) if match[1] == 'storage'
                                      else dataset_path(match[2])), text)


def resolve_text(text):
    """Expand explicit tokens and the finite legacy roots declared by this machine."""
    text = _tokens(text)
    for prefix, base in [('datasets', storage_path('datasets')), ('output', storage_path('output')),
                         ('experiments', REPO / 'experiments'), ('configs', REPO / 'configs'),
                         ('src', REPO / 'src'), ('scripts', REPO / 'scripts')]:
        if text.startswith(prefix + '/'):
            text = str(base / text[len(prefix)+1:])
            break
    aliases = dict(machine()['legacy_paths'])
    catalog_file = Path(machine()['catalog'])
    catalog_file = catalog_file if catalog_file.is_absolute() else REPO / catalog_file
    if catalog_file.is_file():
        for identifier, entry in catalog().items():
            for alias in entry.get('aliases', []):
                aliases[alias] = '${dataset:' + identifier + '}'
    for old, target in sorted(aliases.items(), key=lambda item: -len(item[0])):
        target = _tokens(target).rstrip('/')
        if text == old:
            text = target
        else:
            text = text.replace(old.rstrip('/') + '/', target + '/')
    return text


def resolve_path(value):
    path = Path(resolve_text(str(value))).expanduser()
    return path if path.is_absolute() else REPO / path


def resolve_config(value):
    """Repository JSON/YAML config trees: dictionaries, lists and scalar values."""
    if isinstance(value, dict):
        return {key: resolve_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_config(item) for item in value]
    if isinstance(value, str):
        return resolve_text(value)
    return value


def load_json(path):
    return resolve_config(json.loads(resolve_path(path).read_text()))


def portable_config(value):
    """Canonical path spelling for config comparisons, independent of mount points."""
    if isinstance(value, dict):
        return {key: portable_config(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_config(item) for item in value]
    if not isinstance(value, str):
        return value
    value = resolve_text(value)
    roots = machine()['roots']
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
