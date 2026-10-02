"""Sealed synthetic full-cell protocol and explicit prediction inputs."""
import json
from importlib.metadata import version
from pathlib import Path

from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.project_runtime.paths import resolve_path

FAMILY = 'response_training'


def metric_family(c):
    return 'response_training_fast' if 'simulation_profile' in c else FAMILY


def simulation_profile(c):
    return read(Path(__file__).resolve().parents[3]/c['simulation_profile']) if 'simulation_profile' in c else None


def read(path):
    return json.loads(Path(path).read_text())


def root(c):
    return resolve_path(c['output'])


def oracle_config(c):
    path = resolve_path(c['oracle_recipe'])
    if sha(path) != c['oracle_recipe_sha256']:
        raise ValueError('Original oracle recipe changed')
    o = read(path)
    o.update(output=c['output'], horizons_steps=c['horizons_steps'])
    if sha(resolve_path(o['potential'])) != o['potential_sha256']:
        raise ValueError('Fixed simulator potential changed')
    return o


def parents(c):
    rows = [dict(index=i, role=role, sigma_index=i % 4)
            for role, (lo, hi) in c['parent_roles'].items() for i in range(lo, hi)]
    if [r['index'] for r in rows] != list(range(56)):
        raise ValueError('Expected sealed 32/8/16 configuration roles')
    return rows


def bind(c):
    o = oracle_config(c)
    files = [*Path(__file__).parent.glob('*.py'),
        Path(__file__).parents[1]/'response_atlas/atomistic.py',
        Path(__file__).parents[1]/'response_atlas/reference.py',
        Path(__file__).parents[1]/'supervised_onset/model.py',
        Path(__file__).parents[2]/'models/encoders/spatial_mace.py',
        Path(__file__).parents[2]/'models/encoders/mace_backend.py']
    profile = simulation_profile(c)
    if profile is not None:
        files.extend([Path(__file__).parents[1]/'response_performance/oracle.py',
                      Path(__file__).parents[1]/'response_performance/collection.py',
                      Path(__file__).resolve().parents[3]/c['simulation_profile']])
    binding = dict(config=c, oracle=o, parents=parents(c),
        sources={str(p.relative_to(Path(__file__).parents[3])):sha(p) for p in files},
        packages={k:version(k) for k in ('torch', 'numpy', 'ase', 'e3nn', 'mace-torch')})
    if profile is not None:
        binding['simulation_profile'] = profile
        binding['packages']['cuequivariance-torch'] = version('cuequivariance-torch')
    record = dict(identity=digest(binding), binding=binding)
    path = root(c)/'technical/binding.json'
    if path.exists() and read(path) != record:
        raise ValueError('Atomistic training protocol changed; use a new run revision')
    write_json(path, record)
    write_json(root(c)/'technical/prediction-context.json', dict(
        encoder_inputs=dict(spatial_support='complete periodic 256-atom cell', edge_cutoff_A=5,
            halo=None, history=0, motion=False, conditions=[], relaxation=False,
            species_channel='one constant atom channel', center_indicator='zero; no distinguished atom',
            fixed_length_normalization='Al multiplier 1', capacity='channels128, z128',
            pool='mean and population variance of atomwise scalar channels',
            training_only_teacher='fixed-MACE BAOAB future values; two full-cell coordinate responses in responses8 arm'),
        predictor_inputs=dict(embedding='z128', head='hidden128 SiLU to256 normalized future features',
            conditions=[], history=0),
        response='J_(head o encoder)(q) V, through live radial/angular geometry and all learned layers',
        oracle_conditions=dict(temperature_K=450, timestep_fs=1, horizon_fs=[20,100],
            potential_sha256=o['potential_sha256'], full_cell=True,
            numerical_profile=profile or dict(dtype='float64', backend='e3nn', graph='ase', replicas=1)),
        track='separate shared-FCC-prototype mechanism experiment; not Al64 window evaluation or MEAM shooting',
        ancestry=('same 56 generating geometries and fixed roles as the October1 mechanism cohort; all share an ideal FCC prototype; new stochastic namespace, not new independent geometries; numerical acceptance may inspect fixed test-role streams without model fitting or selection'
                  if profile is not None else
                  'parents0:15 seen only in numerical development, now training only; all held-out configurations fresh; all share ideal FCC prototype'),
        training_deviation=c['training']['deviation']))
    return record
