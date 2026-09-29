"""Result receipts for the maintained supervised-onset producer, without fitting."""
import json
from pathlib import Path

from src.analysis.publication import inventory, original_contracts, render_bundle
from .artifacts import write_json
from .result_records import file_hash, identity, location, save_record


def supervised_onset_record(study):
    c = study.config
    context_path = study.technical/'prediction-context.json'
    context = json.loads(context_path.read_text())
    if context['identity'] != study.identity:
        raise ValueError(f'{context_path}: prediction-context identity differs from study')
    analyses = []
    components = []
    available = set()
    checkpoints = {p.parent.parent.name:json.loads(p.read_text())['checkpoint_sha256']
                   for p in (study.technical/'evaluation').glob('*/best/metrics.json')}
    for path in sorted((study.technical/'evaluation').glob('*/*/metrics.json')):
        result = json.loads(path.read_text())
        if result['identity'] != study.identity:
            raise ValueError(f'{path}: saved evaluation belongs to another study')
        name, readout = result['name'], result['kind']
        available.add((name,readout))
        evaluation_id = identity(dict(study=study.identity,component=name,readout=readout,
                                      numerical_evidence=file_hash(path)))
        bundle = study.root/'analyses'/f'{name}-{readout}-{evaluation_id[:12]}'
        bundle.mkdir(parents=True,exist_ok=True)
        definitions = [a for a in original_contracts(study.root,bundle)
                       if a['role'] in {'metric_contract','metric_definition'}
                       and Path(a['relative']).name!='METRIC_INDEX.md']
        artifacts = inventory(path.parent,bundle)+definitions
        artifacts = list({a['id']:a for a in artifacts}.values())
        diagnostics = path.parent/'diagnostics.json'
        stages = dict(prediction=dict(state='complete',evidence=location(path)),
                      representation_diagnostics=dict(state='complete' if diagnostics.exists() else 'unavailable',
                          evidence=location(diagnostics) if diagnostics.exists() else None,
                          note='Readout-only evaluations do not imply an encoder diagnostic pass.'))
        if diagnostics.exists():
            values = json.loads(diagnostics.read_text())
            stages['movement'] = dict(state='unavailable' if values.get('movement') is None else 'complete',
                evidence=location(diagnostics),note=values.get('movement_unavailable','Recorded in diagnostics.'))
        analysis = dict(schema_version=1,id=evaluation_id,title=f'{name} / {readout}',
            context='Saved likelihood-based evaluation. AP is diagnostic; source uncertainty is not seed replication.',
            checkpoint_sha256=result.get('checkpoint_sha256',checkpoints.get(name)),protocol=c['protocol'],
            inputs=dict(encoder=context['encoder_inputs'],predictor=context['predictor_inputs'],
                full_context=location(context_path),context_sha256=file_hash(context_path),
                actual_input_domain=result.get('input'),external_inputs=result['external_inputs']),
            population=dict(fixed_dataset=c.get('fixed_dataset'),producer_identity=study.identity,
                cohort_manifest=location(study.cache/'manifest.json'),
                cohort_manifest_sha256=file_hash(study.cache/'manifest.json')),
            selection=dict(objective=c['objective'],checkpoint_selector=c['selection_metric'],
                readout_selector=result.get('selected_by'),calibration=result['calibration']),
            stages=stages,artifacts=artifacts,page=location(bundle/'index.html'),
            numerical_evidence=dict(path=location(path),sha256=file_hash(path)))
        write_json(bundle/'analysis.json',analysis)
        write_json(bundle/'artifacts.json',dict(schema_version=1,analysis_id=evaluation_id,artifacts=artifacts))
        render_bundle(bundle,analysis)
        analyses.append(analysis)
        if readout in {'linear','mlp'}:
            receipt=study.technical/'readouts'/name/readout/'complete.json'
            if receipt.exists():
                components.append(dict(id=identity([study.identity,name,readout]),name=f'{name}/{readout}',
                    activity='frozen_probe',parent=identity([study.identity,name]),
                    receipt=location(receipt),receipt_sha256=file_hash(receipt)))
    expected = {(a['name'],readout) for a in c['arms'] for readout in ('best','linear','mlp')}
    completed = []
    for arm in c['arms']:
        name=arm['name'];receipt=study.technical/'runs'/name/'evaluation-complete.json'
        if receipt.exists():
            saved=json.loads(receipt.read_text())
            if saved['identity']!=study.identity:
                raise ValueError(f'{receipt}: completed arm identity differs')
            completed.append(name)
        components.append(dict(id=identity([study.identity,name]),name=name,activity='encoder',
            seed=c['seed'],evaluation_state='complete' if name in completed else 'pending',
            checkpoint_sha256=checkpoints.get(name),
            input=arm['input'],objective=c['objective'],selector=c['selection_metric'],
            tracking_receipt_directory=location(study.technical/'wandb'/name)))
    record=dict(schema_version=1,id='supervised-onset:'+study.identity,kind='research',activity='campaign',
        title=study.root.name,study=dict(id='supervised_information_20260925',
            record='${storage:repo}/experiments/supervised_information_20260925/README.md'),
        execution=dict(state='consult_producer',receipt=location(study.technical/'queue-state.json')),
        evidence=dict(state='complete' if expected<=available and len(completed)==len(c['arms']) else 'partial',
            expected_encoder_readouts=len(expected),available_encoder_readouts=len(expected & available),
            missing=[f'{name}/{readout}' for name,readout in sorted(expected-available)]),
        interpretation=dict(state='exploratory',note='Consult the authored study findings; no ranking or selection is performed by the catalogue.'),
        components=components,analyses=analyses)
    return save_record(study.root,record,refresh=True)
