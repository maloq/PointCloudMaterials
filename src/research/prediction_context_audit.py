"""Recover input contracts for the saved September 24–25 onset comparisons.

This is a historical audit, not a trainer. Never change a saved score or infer
that missing metadata means an experiment was unconditioned.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch

from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def context_row(collection, row):
    return dict(collection=collection, model=row['model'], readout=row['readout'],
                task='local sustained onset', input_frames=1, history_ps=0,
                velocities=False, encoder_temperature=False, encoder_time=False,
                predictor_time=False, encoder_external_physics=False,
                encoder_auxiliary_inputs=[], training_teacher=None,
                training_predictor_context=None, context_evidence=[],
                AP3=float(row.get('AP3', row.get('test_AP3'))),
                AP6=float(row.get('AP6', row.get('test_AP6'))))


def screen_context(row):
    collection = row['collection']
    out = context_row(collection, row)
    prediction = Path(row['prediction_path'])
    folder = prediction.parent
    root = folder.parents[2]
    out['prediction_path'] = str(prediction)
    out['prediction_sha256'] = sha(prediction)
    out['predictor_temperature'] = True
    out['temperature_encoding'] = 'five fitting-temperature indicators'
    out['condition_normalization'] = 'raw indicators'
    if collection in ('robust_onset', 'spatial_hierarchy'):
        identity_path = root / 'technical/identity.json'
        config = json.loads(identity_path.read_text())['config']
        arm = next(a for a in config['arms'] if a['name'] == row['model'])
        out.update(encoder_input=arm['input'] + ' current geometry',
                   encoder_kind='native MACE', code_dim=128,
                   support='all cached atoms inside 8 A; smooth boundary taper',
                   focal_radius_A=8., context_radius_A=8., maximum_atoms=None,
                   edge_cutoff_A=config['encoder']['cutoff'], message_passing_layers=2,
                   predictor_input='128-D state + temperature', predictor_input_dim=133,
                   predictor_physical_side_inputs=[],
                   encoder_auxiliary_inputs=['Al species', 'center indicator'],
                   training_predictor_context='event head: state + temperature; physical heads: state only',
                   training_teacher='relaxed-geometry teacher' if arm['teacher'] else None,
                   input_verification='saved experiment identity and producer source',
                   context_evidence=[str(identity_path), 'src/research/robust_onset/data.py:targets',
                                     'src/research/robust_onset/evaluate.py:run'])
        if collection == 'spatial_hierarchy':
            radii = dict(local=[4., 6., 8.], near=[8., 10., 12.], wide=[8., 12., 16.])[arm['context']]
            out.update(context_radius_A=max(radii), spatial_summary_radii_A=radii,
                       spatial_fusion='early' if arm['early'] else 'late')
            out['context_evidence'] += ['src/research/spatial_hierarchy/data.py:RADII',
                                        'src/research/spatial_hierarchy/model.py']
    else:
        # These saved probes use the same producer, but native input geometry
        # differs. Read each inference task rather than guessing from its name.
        probe = folder / f"{row['readout']}-hazard-readout.pt"
        saved = torch.load(probe, map_location='cpu', weights_only=False)
        dimension = len(saved['mean'])
        actual = list(saved['model']['weight'].shape)
        if actual != [5, dimension + 5 + 8 + 89]:
            raise ValueError(f'Unexpected historical predictor shape {actual}: {probe}')
        out.update(encoder_input='relaxed current geometry', code_dim=dimension,
                   predictor_input=f'{dimension}-D representation + temperature + current order + relaxed descriptors',
                   predictor_input_dim=actual[1], predictor_physical_side_inputs=[
                       'current_order: 8 columns', 'relaxed geometry descriptors: 89 columns'],
                   input_verification='saved predictor input width plus producer source',
                   context_evidence=[str(probe), 'src/research/geoframe_evolution/prediction.py:evaluate'],
                   probe_sha256=sha(probe), context_radius_A=8.)
        out['condition_normalization'] = 'all 102 side-input columns standardized on fitting rows'
        task_path = folder / 'task.json'
        if collection == 'Geoformer':
            out.update(encoder_kind='Geoformer', support='nearest80', maximum_atoms=80,
                       focal_radius_A=8., edge_cutoff_A=None, message_passing_layers=None,
                       coordinate_divisor=9.192189)
            out['context_evidence'] += ['src/research/geoframe_evolution/prediction.py:prepare']
        else:
            task = json.loads(task_path.read_text())
            radius = 8. if task['kind'] == 'geometry' else 8. * task['scales']['Al'] / 9.192189
            out.update(encoder_kind=task['architecture'], native_kind=task['kind'],
                       support=task['support'], maximum_atoms=80 if task['support']=='nearest80' else None,
                       focal_radius_A=8. if task['support']=='nearest80' else radius,
                       coordinate_scale=task['scales']['Al'],
                       checkpoint_sha256=task['checkpoint_sha256'],
                       edge_cutoff_A=5. if task['kind']=='geometry' else (
                           5.*task['scales']['Al']/9.192189 if task['support']=='radius' else None),
                       message_passing_layers=None)
            if task['kind'] in ('shared', 'neighborhood'):
                out['encoder_auxiliary_inputs'] = ['Al species', 'material log-scale; fixed within Al assay']
            elif task['kind'] == 'geometry':
                out['encoder_auxiliary_inputs'] = ['Al species', 'center indicator']
            # Depth is deliberately unfilled here: native MACE, GATr and
            # Geoformer variants need their own checkpoint architecture audit.
            out['context_evidence'] += [str(task_path), 'src/research/encoder_screen/native.py:main',
                                       'src/research/encoder_screen/prepare.py']
    out['relaxation_context'] = ('full periodic current cell before cropping' if
        'relaxed' in out['encoder_input'] or out['predictor_physical_side_inputs'] else None)
    return out


def supervised_context(row, root):
    out = context_row('supervised_ap36_large', row)
    name, kind = row['model'], row['readout']
    config_path = root / 'technical/code/configs/supervised_onset/ap36_20260924.json'
    config = json.loads(config_path.read_text())
    arms = {a['name']: a for a in config['arms']}
    external = 'five temperature indicators + age/600 ps + (age/600 ps)^2'
    out.update(predictor_temperature=True, predictor_time=True,
               temperature_encoding='five fitting-temperature indicators',
               time_encoding='age/600 ps and its square',
               condition_normalization='raw temperature indicators; age scaling as recorded',
               predictor_physical_side_inputs=[], context_radius_A=8., focal_radius_A=8.,
               maximum_atoms=80, support='nearest 80 including center, cropped at 8 A',
               edge_cutoff_A=5., message_passing_layers=2,
               encoder_auxiliary_inputs=['Al species', 'center indicator'],
               training_predictor_context='state or descriptors + '+external,
               identity=row['identity'], input_verification='frozen source and full 31,609-row condition audit',
               context_evidence=[str(config_path), str(root/'technical/context-audit-20260925.json'),
                                 str(root/'technical/code/src/research/supervised_onset/model.py'),
                                 str(root/'technical/code/src/research/supervised_onset/evaluate.py')])
    if name in arms:
        arm = arms[name]
        out.update(encoder_kind='native MACE', encoder_input=dict(hot='observed', cold='relaxed', paired='observed and relaxed')[arm['input']]+' current geometry',
                   code_dim=128, predictor_input='128-D state + '+external, predictor_input_dim=135,
                   training_teacher=config['teacher_arm'] if arm['teacher'] else None)
    elif name.endswith('-descriptors') or name == 'conditions':
        dimension = {'observed-descriptors':237, 'relaxed-descriptors':237, 'paired-descriptors':474, 'conditions':0}[name]
        out.update(encoder_kind='no learned encoder', encoder_input=None, code_dim=None,
                   predictor_input=f'{dimension} physical descriptors + '+external,
                   predictor_input_dim=dimension+7, message_passing_layers=None, edge_cutoff_A=None,
                   encoder_auxiliary_inputs=[])
    elif name == 'selected-ensemble':
        out.update(encoder_kind='risk ensemble', encoder_input='relaxed current geometry',
                   predictor_input='mixture of R-AP36 and R-NLL risks; each constituent receives '+external,
                   predictor_input_dim=None, code_dim=None)
    else:
        raise ValueError(f'Unaudited supervised model: {name}')
    if kind in ('linear', 'mlp'):
        path = root / 'technical/readouts' / name / kind / 'best.pt'
        state = torch.load(path, map_location='cpu', weights_only=False)['model']
        weight = state['weight' if kind=='linear' else '0.weight']
        if weight.shape[1] != out['predictor_input_dim']:
            raise ValueError(f'Unexpected historical readout width: {path}')
        out['context_evidence'].append(str(path))
        out['input_verification'] += '; saved readout width'
    out['relaxation_context'] = ('full periodic current cell before cropping' if
        name in ('relaxed-descriptors','paired-descriptors','selected-ensemble') or
        (name in arms and arms[name]['input'] in ('cold','paired')) else None)
    return out


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    root = resolve_path(config['output'])
    for sub in ('tables', 'technical'):
        (root/sub).mkdir(parents=True, exist_ok=True)
    screen = resolve_path(config['screen_table'])
    supervised = resolve_path(config['supervised_root'])
    rows = [screen_context(r) for r in read_rows(screen)]
    rows += [supervised_context(r, supervised) for r in read_rows(supervised/'tables/comparison.csv')]
    keys = [(r['collection'], r['model'], r['readout']) for r in rows]
    if len(rows) != config['expected_rows'] or len(set(keys)) != len(keys):
        raise ValueError(f'Unexpected or duplicate result coverage: {len(rows)} rows')
    files = {str(screen):sha(screen), str(supervised/'tables/comparison.csv'):sha(supervised/'tables/comparison.csv')}
    for row in rows:
        for evidence in row['context_evidence']:
            path = Path(evidence.split(':')[0])
            if path.is_file():
                files[str(path)] = sha(path)
    payload = dict(audit_date='2026-09-25', rows=rows, evidence_sha256=files,
                   scope='246 saved horizon comparisons plus 29 larger supervised comparisons; not a universal audit of every archived training study')
    (root/'technical/contexts.json').write_text(json.dumps(payload, indent=2)+'\n')
    columns = sorted(set().union(*(r.keys() for r in rows)))
    with (root/'tables/prediction-context.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v) if isinstance(v, (list,dict)) else v for k,v in row.items()})
    snapshot_metric_docs(root, 'prediction_context')
    print(json.dumps(dict(output=str(root), rows=len(rows), temperature_conditioned=sum(r['predictor_temperature'] for r in rows),
                          age_conditioned=sum(r['predictor_time'] for r in rows),
                          physical_side_conditioned=sum(bool(r['predictor_physical_side_inputs']) for r in rows))))


if __name__ == '__main__':
    main()
