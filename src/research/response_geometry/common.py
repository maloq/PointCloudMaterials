"""Versioned unrelaxed-only response experiment contracts."""
from pathlib import Path
import torch
from src.experiment_runner.artifacts import file_hash as sha, json_digest as digest, write_json
from src.project_runtime.paths import resolve_path
from src.research.local_response.common import read, save

FAMILY='response_geometry_training'


def root(c):return resolve_path(c['output'])


def legacy(c):return read(resolve_path(c['reference'])/'technical/binding.json')['config']


def bind(c):
    repo=Path(__file__).resolve().parents[3]
    old=read(resolve_path(c['reference'])/'technical/binding.json')
    if old['identity']!=c['reference_identity']:raise ValueError('Changed original teacher protocol')
    for name,expected in old['implementation'].items():
        if sha(repo/name)!=expected:raise ValueError('Changed inherited producer: '+name)
    files={str(p.relative_to(repo)):sha(p)for p in Path(__file__).parent.glob('*.py')}
    record=dict(config=c,inherited=old['implementation'],implementation=files)
    record['identity']=digest(record)
    path=root(c)/'technical/binding.json'
    if path.exists() and read(path)!=record:raise ValueError('Changed run producer; require new revision')
    write_json(path,record)
    write_json(root(c)/'technical/prediction-context.json',dict(
        encoder_inputs=dict(geometry='unrelaxed center-relative nearest80',support_A=8,edge_cutoff_A=5,
            channels=128,embedding=128,constant_atom_channel=True,center_indicator=True,
            halo=None,history=0,motion=False,conditions=[],relaxation=False),
        predictor_inputs=dict(embedding='z128 only',conditions=[],history=0,motion=False),
        training_only_teacher=dict(potential='MACE-MPA-0 medium',parent_potential='Lee2003 MEAM',
            halo_A=24,reference_A=36,temperature_K=450,horizons_fs=[20,100],
            targets=['legacy changes','rich changes','rich future endpoints plus changes'],
            geometry_labels=False,crystal_labels=False,metric_targets='shot-noise-corrected future feature mean distances'),
        evaluation=dict(fixed_identity=c['fixed_identity'],track='all64 uniform unrelaxed structural observations',
            primary_clustering='K7 many-to-one descriptor agreement',selection='validation future-feature Gaussian NLL',
            test='original reused test sources; exploratory, not an untouched benchmark'),
        training_deviation='batch256; microbatch8 for higher-derivative memory; 90-parent arms sample with replacement, 720-parent arms without replacement'))
    return record


def load(c):
    r=read(root(c)/'technical/data-complete.json');p=resolve_path(c['cache'])/'training.pt'
    if r['identity']!=read(root(c)/'technical/binding.json')['identity'] or sha(p)!=r['sha256']:
        raise ValueError('Changed sealed response bank')
    return torch.load(p,map_location='cpu',weights_only=False)
