"""Shared historical inputs; independent identities for this factorial study."""
from itertools import product
from pathlib import Path
import numpy as np

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.project_runtime.paths import resolve_path
from src.research.shooting_laws.common import read
from src.research.shooting_laws.features import bank, encode
from src.research.predictive_baseline.data import load as base_load, cache, output

FAMILY = 'predictive_followup'


def jobs(c, kind):
    sources = ['joint'] if kind == 'joint' else [s for s in c['sources'] if s != 'joint']
    return [dict(source=s, target=t, variance=v, seed=seed)
            for s, t, v, seed in product(sources, c['target_arms'], c['variance_arms'], c['fit_seeds'])]


def name(arm):
    return f'{arm["source"]}-{arm["target"]}-{arm["variance"]}'


def folder(c, arm):
    return output(c)/'analyses'/f'{name(arm)}-seed-{arm["seed"]}'


def load(c):
    data, manifest = base_load(c)
    if manifest['identity'] != c['baseline_target_identity']:
        raise ValueError('Follow-up must reuse the exact existing target map, rows and roles')
    return data, manifest


def features(c, data, manifest, source):
    if source == 'joint':
        return np.asarray(data['positions'])
    if source in ('mace_vicreg', 'mace_epi'):
        return np.asarray(data[source])
    if source == 'mm_tda_block_direct_full':
        return bank(c, data, manifest, dict(bank=source))
    raise ValueError(f'Unknown frozen feature producer: {source}')


def prepare(c):
    data, manifest = load(c)
    spec = c['encoders'][0]
    checkpoint = resolve_path(spec['checkpoint'])
    producer = resolve_path(spec['producer'])
    if sha(checkpoint) != spec['checkpoint_sha256']:
        raise ValueError('Selected MM-TDA checkpoint changed')
    for filename, expected in spec['inference_dependencies'].items():
        if sha(producer/filename) != expected:
            raise ValueError(f'Changed MM-TDA inference dependency: {filename}')
    recipe = read(checkpoint.parent/'config.json')
    complete = read(checkpoint.parent/'complete.json')
    if recipe['name'] != spec['label'] or complete['best_sha256'] != spec['checkpoint_sha256']:
        raise ValueError('Wrong MM-TDA training run or selected checkpoint')
    descriptor_root = resolve_path(recipe['cache'])
    descriptor = read(descriptor_root/'plan.json')
    descriptor_manifest = read(descriptor_root/'manifest.json')
    if (descriptor['identity'] != recipe['prepared_identity'] or
            descriptor_manifest['identity'] != recipe['prepared_identity'] or
            sha(descriptor_root/'plan.json') != descriptor_manifest['plan_sha256']):
        raise ValueError('MM-TDA descriptor ancestry plan differs from its sealed producer')
    packed_root = resolve_path(recipe['loader']['packed_cache'])
    packed = read(packed_root/'manifest.json')
    if (packed['binding']['dataset'] != recipe['prepared_identity'] or
            sha(descriptor_root/'manifest.json') != packed['binding']['descriptor_manifest_sha256']):
        raise ValueError('MM-TDA fitting release differs from recorded packed data')
    tasks = {t['id']: t for t in descriptor['tasks']}
    actual_sources = {tasks[s['task']['id']]['source'] for s in packed['shards'] if s['rows'] > 0}
    if any(tasks[s['task']['id']]['role'] != 'train' for s in packed['shards'] if s['rows'] > 0):
        raise ValueError('MM-TDA packed fitting rows include non-training roles')
    native_root = resolve_path(recipe['structural_dataset']['root'])
    native = read(native_root/'plan.json')
    native_manifest = read(native_root/'manifest.json')
    if (native['identity'] != recipe['structural_dataset']['identity'] or
            sha(native_root/'plan.json') != native_manifest['plan_sha256'] or
            sha(native_root/'manifest.json') != descriptor['source_manifest_sha256']):
        raise ValueError('MM-TDA original structural source ancestry changed')
    used = [s for s in native['sources'] if s['id'] in actual_sources]
    if len(used) != len(actual_sources):
        raise ValueError('Missing MM-TDA source ancestry')
    paths = [str(resolve_path(s['path'])) for s in used]
    overlap = [s for s in np.unique(data['sources']) if any(str(s) in path for path in paths)]
    if overlap:
        raise ValueError(f'MM-TDA has exact shooting source overlap: {overlap}')
    audit = dict(actual_training_sources=[{k:s[k] for k in ('id','lineage','material','potential','path')} for s in used],
        exact_shooting_source_overlap=overlap,
        limitations=native['ancestry_limitations'],
        ancestry_claim='No exact source/path overlap found; archived non-native preparation ancestry remains limited',
        input_manifests={str(p):sha(p) for p in (descriptor_root/'plan.json', descriptor_root/'manifest.json',
            native_root/'plan.json', packed_root/'manifest.json')})
    tech = output(c)/'technical'
    binding = dict(config=c, target_identity=manifest['identity'], target_files=manifest['files'],
                   checkpoint=spec, ancestry=audit)
    record = dict(identity=digest(binding), binding=binding)
    if (tech/'binding.json').exists() and read(tech/'binding.json') != record:
        raise ValueError('Frozen follow-up binding changed')
    write_json(tech/'binding.json', record)
    write_json(tech/'prediction-context.json', dict(
        encoder_inputs=dict(atoms=80, radius_A=8, cutoff_A=5, halo=None, history=0, motion=False,
            conditions=[], relaxation=False, constant_atom_channel=True, center_indicator=True,
            material_normalization='fixed Al multiplier 1', new_training_teachers=[]),
        predictor_inputs=dict(joint='learned z128', frozen_vicreg_epi='recorded z128',
            frozen_mm_tda='recorded z256; training-only coordinate standardization',
            head='128-hidden SiLU; same target transform and selector', conditions=[], history=0),
        mm_tda=dict(encoder=recipe['encoder_config'], original_training=recipe['training'],
            original_objective=recipe['objective'], frozen=True,
            support='same nearest80 / radius8 input as baseline; no relaxed view',
            comparison='historical larger encoder and pretraining population; not capacity/pretraining matched'),
        target='Reuse all normalization, Fourier frequencies, phases and block weights from the sealed baseline',
        selector=c['selector'], track=c['track'], ancestry_limitations=native['ancestry_limitations']))
    return record


def encode_mm(c):
    prepare(c)
    encode(c, 0)
    data, manifest = load(c)
    z = features(c, data, manifest, 'mm_tda_block_direct_full')
    if z.shape != (len(data['parent']), 256) or not np.isfinite(z).all():
        raise ValueError(f'MM-TDA export shape/value mismatch: {z.shape}')
    write_json(output(c)/'technical/mm-encoding-complete.json', dict(state='complete', rows=len(z),
        dimensions=256, checkpoint_sha256=c['encoders'][0]['checkpoint_sha256'],
        target_identity=manifest['identity']))
