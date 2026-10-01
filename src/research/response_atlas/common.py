"""Explicit immutable protocol, output and metric bindings."""
import json
from pathlib import Path

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import write_metric_rows


def read(path):
    return json.loads(Path(path).read_text())


def output(c):
    return resolve_path(c['output'])


def table(c, analysis, name, rows):
    return write_metric_rows(rows, output(c) / 'analyses' / analysis,
                             family='response_atlas', name=name)


def protocol(c):
    import importlib.metadata
    sources = {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}
    return dict(config=c, sources=sources, libraries={k: importlib.metadata.version(k)
                for k in ('torch', 'numpy', 'scipy', 'ase', 'mace-torch')})


def bind(c):
    value = protocol(c)
    if sha(resolve_path(c['potential'])) != c['potential_sha256']:
        raise ValueError('Changed fixed MACE potential')
    if sha(resolve_path(c['shooting_config'])) != c['shooting_config_sha256']:
        raise ValueError('Changed shooting assay protocol')
    root = output(c) / 'technical'
    path = root / 'protocol.json'
    record = dict(identity=digest(value), binding=value)
    if path.exists() and read(path) != record:
        raise ValueError('Response protocol changed: use a new run')
    write_json(path, record)
    write_json(root / 'prediction-context.json', dict(
        synthetic_predictor_inputs=['u', 'v'], synthetic_encoder='toy MLP, explicit capacity exception',
        atomistic_predictor='none: fixed-potential oracle feasibility only',
        oracle_state='complete periodic 256-atom Al cell; fixed species/box/potential/450K thermostat',
        conditions_as_model_inputs=[], history_as_model_input=False, relaxation=False,
        future_targets='smooth radial statistics of same complete cell; fixed joint-prefix RFF',
        ancestry='controlled FCC perturbations share a prototype; no independent-melt generalization claim',
        scientific_training='online W&B for toy likelihood/response fits; numerical gates and frozen diagnostics local'))
    return record
