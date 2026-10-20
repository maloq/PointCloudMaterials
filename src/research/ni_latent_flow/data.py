"""Reuse sealed native observations and ancestral splits without recollection."""
from pathlib import Path
from collections import Counter
import torch

from src.research.continuous_motion.data import (
    read, root, cache, resolve_path, sha, digest, write_json, atomic_save,
)
from src.research.ni_kinetic_latent import data as native
from src.research.ni_spatial_scales import data as spatial

FAMILY = 'ni_latent_flow'


def prepare(c):
    parents = {}
    for name in ('base', 'spatial'):
        path = resolve_path(c[name + '_run']) / 'technical/plan.json'
        if sha(path) != c[name + '_plan_sha256']:
            raise ValueError('Changed parent plan: ' + name)
        parents[name] = read(path)
    base = parents['base']
    # Parent producers already establish exact timelines, roles and identities.
    # Bind the reused files rather than silently rerunning their normalization.
    files = {}
    for key, path in (
        ('native', cache(base['config']) / 'data.pt'),
        ('labels', cache(base['config']) / 'labels.pt'),
        ('context', cache(parents['spatial']['config']) / 'context.pt'),
    ):
        receipt = read(path.with_suffix('.json'))
        if sha(path) != receipt['sha256']:
            raise ValueError('Changed parent cache: ' + str(path))
        files[key] = dict(path=str(path), sha256=receipt['sha256'])
    checkpoints = {}
    for arm in c['arms']:
        if arm['source_arm'] is None:
            continue
        parent = parents['spatial' if arm['scale'] == 'r20' else 'base']
        folder = root(parent['config']) / 'analyses' / arm['source_arm'] / 'technical'
        receipt = read(folder / 'complete.json')
        path = folder / 'final.pt'
        if sha(path) != receipt['checkpoint_sha256']:
            raise ValueError('Changed source checkpoint: ' + str(path))
        checkpoints[arm['name']] = dict(path=str(path), sha256=receipt['checkpoint_sha256'],
            identity=receipt['identity'], source_arm=arm['source_arm'],
            source_config=parent['config'])
    p = dict(base, config=c, base_config=base['config'],
             spatial_config=parents['spatial']['config'], files=files,
             checkpoints=checkpoints, base_identity=base['identity'],
             spatial_identity=parents['spatial']['identity'],
             data_identity=digest([base['data_identity'], parents['spatial']['data_identity'], files]),
             scope='Native NPT conditional local path distributions on the existing enriched cohort; no equilibrium, reversibility, natural-rate or whole-cell generation claim',
             birth_input_policy='Present measurement encoders only; no generated birth path from fixed-membership history',
             inherited_cadence_audit=base['timelines'])
    if 'expansion' in c:
        from .expansion import directory
        expanded = read(root(c)/'technical/expansion-complete.json')
        for filename, checksum in expanded['files'].items():
            if sha(directory(c)/filename) != checksum:
                raise ValueError('Changed expanded payload: '+filename)
        p['expanded_measurements'] = expanded
        p['expanded_plan_sha256'] = sha(root(c)/'technical/expansion-plan.json')
        expanded_plan = read(root(c)/'technical/expansion-plan.json')
        p['rows'] = [dict(row, source_cohort='old60') for row in base['rows']] + expanded_plan['rows']
        p['sources'] = base['sources'] + expanded_plan['sources']
        p['timelines'] = base['timelines'] + expanded_plan['timelines']
        p['roles'] = dict(Counter(row['role'] for row in p['rows']))
        p['ancestors'] = {role: sorted({row['ancestor'] for row in p['rows'] if row['role'] == role})
                          for role in ('train', 'test')}
        p['files']['labels'] = dict(path=str(directory(c)/'labels.pt'), sha256=expanded['files']['labels.pt'])
        p['data_identity'] = digest([p['data_identity'], expanded, p['expanded_plan_sha256']])
    p.pop('identity')
    p['identity'] = digest(p)
    target = root(c) / 'technical/plan.json'
    if target.exists() and read(target) != p:
        raise ValueError('Changed frozen latent-flow plan; use a new revision')
    write_json(target, p)
    return p


def load(c, arm=None):
    if 'expansion' in c:
        import numpy as np
        from .expansion import load_expanded
        plan, bank = load_expanded(c, arm)
        bank['reference_train_rows'] = np.asarray([i for i, row in enumerate(bank['rows'])
            if row['role'] == 'train' and row['source_cohort'] == 'old60'], dtype=np.int64)
        if arm is not None and arm['input_dimension'] == 331:
            bank['input'] = torch.cat((bank['x'], torch.zeros((*bank['x'].shape[:-1],87))), -1)
        return plan, bank
    p = read(root(c) / 'technical/plan.json')
    if arm is not None and arm['scale'] == 'r20':
        _, bank = spatial.load(p['spatial_config'], arm)
    else:
        _, bank = native.load(p['base_config'])
        bank['input'] = bank['x']
        if arm is not None and arm['input_dimension'] == 331:
            bank['input'] = torch.cat((bank['x'], torch.zeros((*bank['x'].shape[:-1],87))), -1)
    return p, bank


def labels_path(c):
    if 'expansion' in c:
        from .expansion import labels_path as expanded_labels_path
        return expanded_labels_path(c)
    p = read(root(c) / 'technical/plan.json')
    path = Path(p['files']['labels']['path'])
    if sha(path) != p['files']['labels']['sha256']:
        raise ValueError('Changed reused PTM annotations')
    return path


def load_births(c, bank, arm=None):
    p = read(root(c) / 'technical/plan.json')
    if arm is not None and arm['scale'] == 'r20':
        return spatial.load_births(p['spatial_config'], bank, arm)
    cohort, history = native.load_births(p['base_config'], bank)
    inputs = history
    if arm is not None and arm['input_dimension'] == 331:
        inputs = torch.cat((history, torch.zeros((*history.shape[:-1],87))), -1)
    return cohort, history, inputs
