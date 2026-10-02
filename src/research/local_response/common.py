"""Local-response protocol, immutable inputs and source provenance."""
import json
from pathlib import Path

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.project_runtime.paths import resolve_path

FAMILY = 'local_response'


def read(path):
    return json.loads(Path(path).read_text())


def root(c):
    return resolve_path(c['output'])


def save(path, value):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.building.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def bind(c):
    base = Path(__file__).resolve().parents[3]
    paths = list(Path(__file__).parent.glob('*.py')) + [base / p for p in (
        'src/research/response_performance/oracle.py',
        'src/research/response_performance/collection.py',
        'src/research/response_training/model.py',
        'src/research/supervised_onset/model.py',
        'src/models/encoders/spatial_mace.py', 'src/models/encoders/mace_backend.py')]
    record = dict(config=c, implementation={str(p.relative_to(base)): sha(p) for p in paths})
    record = dict(identity=digest(record), **record)
    path = root(c)/'technical/binding.json'
    if path.exists() and read(path) != record:
        raise ValueError('Local response producer/config changed; require a new run revision')
    write_json(path, record)
    write_json(root(c)/'technical/prediction-context.json', dict(
        encoder_inputs=dict(spatial_support='nearest80, center-relative, radius8A',
            edge_cutoff_A=5, halo=None, history=0, motion=False, conditions=[], relaxation=False,
            atom_channel='one constant channel', center_indicator=True,
            fixed_length_normalization='Al multiplier1', channels=128, exported_embedding=128),
        predictor_inputs=dict(embedding='z128', conditions=[], history=0,
            head='128 SiLU to256 normalized future Fourier features'),
        training_only_teacher=dict(potential='MACE-MPA-0 medium; changed from parent MEAM physics',
            potential_sha256=c['oracle']['potential_sha256'], conditions=c['oracle'],
            support='open moving spherical environment; radius chosen only by training-source convergence gate',
            boundary='all included atoms move; no frozen shell; not exact full-cell dynamics',
            responses='two initial local80 displacements; center fixed, exterior initially zero; tangents evolve everywhere'),
        evaluation=dict(fixed_release_identity=c['fixed_identity'],
            track='new source-held-out response-query pilot; not the all64 crystallization-window benchmark',
            downstream_track='all64 required for a subsequent crystallization comparison',
            population='one outcome-independent fixed64 center per non-calibration source; identical queries for every arm'),
        interpretation='fixed-initial-environment responses are regularization under partial observation, not a local sufficiency guarantee',
        training_deviation=c['training']['deviation']))
    return record
