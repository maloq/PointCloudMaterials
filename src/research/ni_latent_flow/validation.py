"""Purpose-matched follow-up: fixed predictors, native objectives and source expansion."""
from collections import defaultdict

import numpy as np

from src.experiment_runner.metric_docs import write_metric_rows
from src.research.ni_kinetic_latent import assessment as native
from src.research.ni_kinetic_latent import births
from . import data, learning, assessment
from .data import root, read, sha, resolve_path


def prepare(c):
    reference = resolve_path(c['reference_run'])/'technical/plan.json'
    if sha(reference) != c['reference_plan_sha256']:
        raise ValueError('Changed reference experiment plan')
    for arm in c['reference_arms']:
        checkpoint = reference.parents[1]/'analyses'/arm['reference_arm']['name']/'technical/final.pt'
        if sha(checkpoint) != arm['checkpoint_sha256']:
            raise ValueError('Changed reference checkpoint: '+arm['name'])
    return data.prepare(c)


def kinetics(c, arm, deadline):
    from . import kinetics as objectives
    if arm.get('stages') == ['representation']:
        return objectives.evaluate(c, arm, deadline, load_provider=data.load,
            fitted_provider=learning.fitted, family=c['metric_family'])
    previous = read(resolve_path(c['reference_run'])/'technical/plan.json')['config']
    original = arm['reference_arm']
    return objectives.evaluate(c, arm, deadline,
        load_provider=lambda _c, _a: data.load(previous, original),
        fitted_provider=lambda _c, _a: learning.fitted(previous, original),
        family=c['metric_family'])


def diagnostics(c, arm, deadline):
    from . import diagnostics as convergence
    return convergence.evaluate(c, arm, deadline, family=c['metric_family'])


def compare(c):
    """Retain all-source contrasts and expose old/new source coverage separately."""
    family = c['metric_family']
    assessment.compare(c, family=family)
    out = root(c)/'analyses/comparison'
    _, bank = data.load(c, c['arms'][0])
    cohorts = {row['ancestor']: row['source_cohort'] for row in bank['rows']}
    for table in ('trajectory', 'retention', 'validity'):
        scopes = (('method','regime','horizon_ps','metric') if table == 'validity' else
                  ('method','regime','subset','metric','feature','horizon_ps','lag_ps'))
        bundles = defaultdict(lambda: defaultdict(dict))
        individual = []
        for arm in c['arms']:
            path = root(c)/'analyses'/arm['name']/'tables'/f'{table}_sources.csv'
            if not path.exists():
                continue
            groups = defaultdict(dict)
            for row in births._rows(path):
                cohort = cohorts[row['ancestor']]
                key = (cohort, *(row.get(field, '') for field in scopes))
                value = float(row['value']) if row['value'] else np.nan
                groups[key][row['ancestor']] = value
                bundles[arm['treatment'], key][arm['seed']][row['ancestor']] = value
            for key, sources in groups.items():
                estimate = native.estimate(list(sources.values()), list(sources),
                    c['seed'], c['evaluation']['bootstrap'])
                individual.append(dict(model=arm['name'], treatment=arm['treatment'], seed=arm['seed'],
                    source_cohort=key[0], **dict(zip(scopes,key[1:])), **estimate))
        paired = []
        for left, right in c['comparisons']:
            keys = sorted(key for treatment,key in bundles if treatment == left and (right,key) in bundles)
            for key in keys:
                result = births._source_pair(bundles[left,key], bundles[right,key], c['model_seeds'],
                                            c['evaluation']['bootstrap'], c['seed'])
                if result is not None:
                    paired.append(dict(left=left,right=right,source_cohort=key[0],
                        **dict(zip(scopes,key[1:])), **result))
        if individual:
            write_metric_rows(individual,out,family=family,name=table+'_by_collection')
        if paired:
            write_metric_rows(paired,out,family=family,name=table+'_collection_paired')
    method_arms = c['reference_arms'] + [a for a in c['arms'] if a.get('stages') == ['representation']]
    for prefix, arms in (('kinetics_', method_arms), ('diagnostics', c['arms'])):
        tables = defaultdict(list)
        for arm in arms:
            for path in (root(c)/'analyses'/arm['name']/'tables').glob(prefix+'*.csv'):
                if path.stem.endswith('_sources'):
                    continue
                tables[path.stem].extend(dict(row, treatment=arm['treatment'], seed=arm['seed'])
                                         for row in births._rows(path))
        for name, rows in tables.items():
            write_metric_rows(rows,out,family=family,name=name+'_individual')
