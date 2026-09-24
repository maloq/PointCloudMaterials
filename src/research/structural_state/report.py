"""Collect matched evaluations with paired, temperature-stratified root intervals."""
import csv
import json

import numpy as np

from src.experiment_runner.metric_docs import snapshot_metric_docs
from .common import write_json
from .data import Corpus
from .dynamics import PROTOCOL


def paired(reference, candidate, temperatures, draws, seed):
    if set(reference) != set(candidate):
        raise ValueError('Paired comparisons require the exact same development roots')
    keys = sorted(reference)
    a, b = (np.array([v[k] for k in keys], dtype=float) for v in (reference, candidate))
    rng = np.random.default_rng(seed)
    temp = np.array([temperatures[int(k)] for k in keys])
    sampled = np.concatenate([rng.choice(np.flatnonzero(temp == t), size=(draws, int((temp == t).sum())))
                              for t in np.unique(temp)], axis=1)
    change = 100*(b[sampled].mean(1)/a[sampled].mean(1)-1)
    lo, hi = np.quantile(change, [.025, .975])
    return dict(reference=float(a.mean()), candidate=float(b.mean()), change_percent=float(100*(b.mean()/a.mean()-1)),
                ci95_lower=float(lo), ci95_upper=float(hi), roots=len(keys), roots_improved=int((b < a).sum()))


def table(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(study):
    corpus = Corpus(study)
    temperatures = {r['source']: r['temperature_K'] for r in corpus.records}
    regression, neighbors, hazard, comparisons, native, auxiliary, ranks = [], [], [], [], [], [], []
    root_metrics = {}
    completed = []
    for arm in study.config['arms']:
        name = arm['name']
        root = study.technical / 'evaluation' / name
        if (root/'auxiliary-heads.json').exists():
            for checkpoint, targets in json.loads((root/'auxiliary-heads.json').read_text()).items():
                for target, scores in targets.items():
                    for population, values in scores['groups'].items():
                        auxiliary.append(dict(arm=name,checkpoint=checkpoint,target=target,population=population,**values))
        for path in sorted(root.glob('*/embedding-geometry.json')):
            for population, scores in json.loads(path.read_text()).items():
                ranks.append(dict(arm=name,representation=path.parent.name,population=population,n=scores['n'],
                    rank=None if scores['rank'] is None else scores['rank']['rank'],
                    conditional_rank=float(np.mean([v['rank'] for v in scores['per_source'].values()])) if scores['per_source'] else None))
        if (root / 'native-heads.json').exists():
            scores = json.loads((root / 'native-heads.json').read_text())
            for checkpoint, blocks in scores['scores'].items():
                for block, values in blocks.items():
                    for population, row in values['groups'].items():
                        native.append(dict(arm=name, checkpoint=checkpoint, domain=scores['domain'],
                                           block=block, population=population, **row))
        if (root / 'complete.json').exists():
            receipt = json.loads((root / 'complete.json').read_text())
            if receipt['identity'] != study.identity:
                raise ValueError('Collector encountered a different study identity')
            completed.append(name)
        for path in sorted(root.glob('*/*/metrics.json')):
            representation, target = path.parts[-3:-1]
            metrics = json.loads(path.read_text())
            if target.startswith('hazard_'):
                kind = target.removeprefix('hazard_')
                hazard.append(dict(arm=name, representation=representation, readout=kind, horizon_ps='all',
                                   nll=metrics['nll'], rows=metrics['rows'], events=metrics['events_12ps']))
                root_metrics[(name, representation, 'hazard', kind)] = metrics['per_source']
                for horizon, values in metrics['horizons'].items():
                    hazard.append(dict(arm=name, representation=representation, readout=kind, horizon_ps=horizon,
                                       **{k: v for k, v in values.items() if k != 'calibration'}))
            else:
                for kind, values in metrics.items():
                    root_metrics[(name, representation, target, kind)] = values['per_source']
                    for population, row in values['groups'].items():
                        regression.append(dict(arm=name, representation=representation, target=target,
                            readout=kind, population=population, **row,
                            rmse=None if row['mse'] is None else float(np.sqrt(row['mse']))))
        for path in sorted(root.glob('*/neighbors.json')):
            representation = path.parent.name
            for family, values in json.loads(path.read_text()).items():
                neighbors.append(dict(arm=name, representation=representation, target=family, population='all', mse=values['mse']))
                root_metrics[(name, representation, family, 'neighbors')] = values['per_source']
                if study.config['protocol'] == PROTOCOL:
                    liquid = values['noncrystalline']
                    neighbors.append(dict(arm=name, representation=representation, target=family,
                                          population='noncrystalline', mse=liquid['mse']))
                    root_metrics[(name, representation, family, 'neighbors_noncrystalline')] = liquid['per_source']
        persistence = root / 'persistence.json'
        if persistence.exists():
            for family, values in json.loads(persistence.read_text()).items():
                root_metrics[(name, 'persistence', family, 'direct')] = values['per_source']
                regression.append(dict(arm=name, representation='persistence', target=family,
                    readout='direct', population='all', **values['groups']['all'],
                    rmse=float(np.sqrt(values['groups']['all']['mse']))))
    for key, candidate in root_metrics.items():
        arm, representation, target, readout = key
        references = []
        if representation in ('pooled', 'exported'):
            references.append(('training', (arm, 'initial_' + representation, target, readout)))
            if arm == 'C-relaxed-teacher':
                references.append(('relaxed_teacher', ('A-observed', representation, target, readout)))
            if arm == 'D-physical-distance':
                references.append(('physical_distance', ('B-relaxed', representation, target, readout)))
            if study.config['protocol'] == PROTOCOL:
                references += [(label,(reference,representation,target,readout))
                               for label,reference,candidate in study.config['contrasts']
                               if candidate==arm and label!='physical_distance']
            if representation == 'exported' and arm in ('A-observed', 'B-relaxed'):
                references.append(('descriptor_control', (arm, 'descriptor', target, readout)))
        if target.startswith('future_order') and representation in ('pooled', 'exported') and readout in ('ridge', 'residual'):
            references.append(('persistence', (arm, 'persistence', target, 'direct')))
        for contrast, reference_key in references:
            if reference_key in root_metrics:
                summary = paired(root_metrics[reference_key], candidate, temperatures,
                                 study.config['bootstrap'], study.config['seed'])
                comparisons.append(dict(contrast=contrast, arm=arm, representation=representation,
                    target=target, readout=readout, reference_arm=reference_key[0],
                    reference_representation=reference_key[1], **summary))
    snapshot_metric_docs(study.root, 'structural_state')
    for name, rows in [('physical', regression), ('neighbors', neighbors), ('onset', hazard), ('comparisons', comparisons), ('training_heads', native)]:
        table(study.root / 'tables' / f'{name}.csv', rows)
    if study.config['protocol'] == PROTOCOL:
        table(study.root/'tables/auxiliary_heads.csv',auxiliary)
        table(study.root/'tables/embedding_geometry.csv',ranks)
    state = 'complete' if len(completed) == len(study.config['arms']) else 'partial'
    write_json(study.technical / 'collection.json', dict(state=state, completed=completed,
        identity=study.identity, tables=dict(physical=len(regression), neighbors=len(neighbors), onset=len(hazard), comparisons=len(comparisons), training_heads=len(native))))
    lines = ['# Fixed structural targets and dynamics', '', f'Queue collection: **{state}**; {len(completed)}/{len(study.config["arms"])} encoder/evaluation pipelines complete.', '',
        'One seed; 25 fitting, five tuning and 15 reused development roots. Native MACE with cuEquivariance, '
        f'fixed {study.config["training"]["updates"]:,}-update exports and matched full-radius observations. No final-test claim.', '',
        '- [Actual training heads](tables/training_heads.csv): calibrated initialization, final checkpoint, and constant-mean controls; no refitting.',
        '- [Physical readouts](tables/physical.csv): training and withheld target families, including future physical order.',
        '- [Embedding neighbors](tables/neighbors.csv): current-temperature/PTM-matched neighbors in original feature space.',
        '- [Onset predictions](tables/onset.csv): natural at-risk population, NLL, AP, Brier, false alarms and timing with misses.',
        '- [Paired comparisons](tables/comparisons.csv): negative changes mean smaller error; intervals resample whole sources within temperature.',
        '- [Metric definitions](tables/METRICS.md).', '',
        ('Angular/l6 observations and onset labels are withheld from encoder losses. Current order8 is trained in every arm; '
         'the declared future arms train a fixed9ps order residual.3ps/12ps order is not an encoder target. '
         if study.config['protocol']==PROTOCOL else
         'Angular and l6 moments, cached current-order measurements, future order, and onset labels were withheld from encoder losses. ')+
        'Current coarse PTM labels only define matched sampling strata. '
        'The sparse four-frame cohort is a mechanism screen; event counts and intervals must accompany interpretation.', '']
    (study.root / 'README.md').write_text('\n'.join(lines))
    return state
