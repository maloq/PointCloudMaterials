"""Mandatory online W&B records, with one stable identity per trained encoder."""
from contextlib import contextmanager
import hashlib
import json
import os
import time
from types import SimpleNamespace
import numpy as np

from .common import write_json


DEFAULTS = dict(entity='teshbek', project='PointCloudMaterials', mode='online')


def require_online(settings):
    if settings['mode'] != 'online' or os.environ.get('WANDB_MODE', 'online') != 'online':
        raise ValueError('W&B must remain online for new runs; offline/disabled mode is not authorized')
    if os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1', 'yes'):
        raise ValueError('WANDB_DISABLED conflicts with mandatory online experiment tracking')


@contextmanager
def tracked_run(study, name, *, job_type='encoder'):
    if study.config.get('tracking_scope') == 'diagnostic':
        raise ValueError('Diagnostic studies must use local_evaluation; they cannot create W&B runs')
    if job_type not in ('encoder', 'predictor', 'control'):
        raise ValueError(f'W&B is restricted to scientific training runs, not {job_type!r}; keep diagnostics local')
    import wandb
    settings = study.config['wandb']
    require_online(settings)
    folder = study.technical/'wandb'/name
    folder.mkdir(parents=True, exist_ok=True)
    run_id = hashlib.sha256(f'{study.identity}:{name}'.encode()).hexdigest()[:20]
    display=settings.get('display_name',f'{study.root.parent.name}/{study.root.name}/{name}')
    if job_type=='control' and 'display_name' in settings:display=f'{display} | {name} control'
    run = wandb.init(entity=settings['entity'], project=settings['project'], mode='online',
        id=run_id, resume='allow', name=display,
        group=settings.get('group',study.root.parent.name), job_type=job_type,
        config=dict(study.config, experiment_identity=study.identity, tracked_component=name),
        tags=[study.config['branch'], 'no-temperature-or-time-inputs', job_type],
        dir=str(folder), save_code=False, force=True, settings=wandb.Settings(init_timeout=60))
    if run is None or run.settings.mode != 'online':
        if run is not None:
            run.finish(exit_code=1)
        raise RuntimeError('W&B did not establish an online run; refusing untracked training')
    try:
        run.define_metric('optimizer_update', hidden=True)
        for prefix in ('train', 'validation'):
            run.define_metric(f'{prefix}/*', step_metric='optimizer_update', summary='last')
        run.summary['prediction_external_inputs'] = []
        run.summary['training_log_semantics'] = ('Label-free encoder objective; no onset labels or event selection'
            if study.config['branch']=='self_supervised' else 'Predictive likelihood training; AP is an evaluation diagnostic only')
        write_json(folder/'run.json', dict(id=run.id, url=run.url, mode=run.settings.mode,
            entity=run.entity, project=run.project, identity=study.identity, component=name))
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)


@contextmanager
def local_evaluation(study, name, kind):
    """Keep diagnostic probe progress and summaries without opening a W&B run."""
    folder = study.technical/'evaluation-tracking'/name/kind
    folder.mkdir(parents=True, exist_ok=True)
    receipt = dict(identity=study.identity, component=name, kind=kind,
        mode='local', created_online_runs=0, state='running', started_at=time.time())
    summary = {}
    write_json(folder/'run.json', receipt)
    try:
        with (folder/'progress.jsonl').open('a') as stream:
            def log(values):
                stream.write(json.dumps(values)+'\n')
                stream.flush()
            yield SimpleNamespace(log=log, summary=summary)
    except BaseException as error:
        receipt.update(state='failed', error=repr(error))
        raise
    else:
        receipt['state'] = 'complete'
    finally:
        write_json(folder/'summary.json', summary)
        write_json(folder/'run.json', dict(receipt, finished_at=time.time()))


def update_training_summary(study, name, fields, *, evaluation):
    """Update a recorded training run through the API; never create/restart one."""
    settings = study.config['wandb']
    require_online(settings)
    folder = study.technical/'wandb'/name
    receipt = json.loads((folder/'run.json').read_text())
    expected_id = hashlib.sha256(f'{study.identity}:{name}'.encode()).hexdigest()[:20]
    expected = dict(id=expected_id, identity=study.identity, component=name,
        entity=settings['entity'], project=settings['project'], mode='online')
    for key,value in expected.items():
        if receipt[key] != value:
            raise ValueError(f'Associated training receipt mismatch: {folder}/run.json: {key}')
    target = folder/'evaluations'/f'{evaluation}.json'
    record = dict(state='pending', run_id=receipt['id'], url=receipt['url'],
        identity=study.identity, evaluation=evaluation, fields=fields, created_online_runs=0)
    write_json(target, record)
    try:
        import wandb
        run = wandb.Api(timeout=60).run(f"{receipt['entity']}/{receipt['project']}/{receipt['id']}")
        run.summary.update(fields)
    except Exception as error:
        write_json(target, dict(record, state='failed', error=repr(error)))
        raise
    write_json(target, dict(record, state='complete', updated_at=time.time()))


def training_record(run, record, optimizer):
    values={'train/event_nll':record['nll'], 'train/gradient_norm':record['gradient_norm'],
            'train/encoder_learning_rate':optimizer.param_groups[0]['lr'],
            'train/head_learning_rate':optimizer.param_groups[1]['lr']}
    if record['distillation']:
        values['train/teacher_distillation_loss']=record['distillation']
    run.log(dict(optimizer_update=record['update'], **values))


def risk_diagnostics(events, risks, sources):
    """Source-weighted, uncalibrated validation scores; no new readout or selector."""
    from src.research.local_predictability.metrics import weighted_scores
    result={}
    for horizon,column in ((3,1),(6,2)):
        values=weighted_scores(events<=column,risks[:,column],sources)
        result.update({f'AP{horizon}':values['average_precision'],
                       f'brier{horizon}':values['brier'], f'log_loss{horizon}':values['log_loss']})
    return result


def validation_fields(scores):
    result={'validation/event_nll':scores['nll']}
    for horizon in (3,6):
        for key,label in (('AP','average_precision'),('brier','brier_score'),('log_loss','binary_log_loss')):
            if f'{key}{horizon}' in scores:
                result[f'validation/{label}_{horizon}ps']=scores[f'{key}{horizon}']
    return result


def population_summary(run, corpus, config):
    for role,ids in corpus.split.items():
        label='validation' if role=='selection' else role
        run.summary[f'data/{label}_windows']=len(ids)
        run.summary[f'data/{label}_sources']=len(np.unique(corpus.pop['source'][ids]))
    if 'fixed_dataset' in config:
        run.summary['data/release_identity']=config['fixed_dataset']['identity']
        run.summary['data/evaluation_track']=config['fixed_dataset']['track']
    run.summary['checkpoint/selection_rule']='minimum source-weighted validation event NLL'


def final_fields(metrics, metadata):
    """Compact scalar columns, instead of hiding final scores in a nested JSON summary."""
    fields={'checkpoint/selected_update':metadata['selected_update'],
            'checkpoint/validation_event_nll':metadata['selection_nll'],
            'model/predictor_parameters':metadata['parameters']}
    for role in ('selection','calibration','test'):
        label='validation' if role=='selection' else role
        fields[f'{label}/selected_event_nll']=metrics['event_nll'][role]
    for horizon in ('3','6'):
        test=metrics['horizons'][horizon]['test']
        for key,label in (('average_precision','average_precision'),('raw_brier','brier_score_raw'),
                          ('brier','brier_score_calibrated'),('raw_log_loss','binary_log_loss_raw'),
                          ('log_loss','binary_log_loss_calibrated'),
                          ('recall','recall_at_calibration_fpr05'),('false_positive_rate','false_positive_rate')):
            fields[f'test/{label}_{horizon}ps']=test[key]
    return fields


def evaluation_record(study, name, kind, scores, metadata):
    """Associated scores update the trained encoder; diagnostic controls stay local."""
    is_encoder = name in {a['name'] for a in study.config['arms']}
    fields = {f'evaluation/{kind}': dict(scores=scores, metadata=metadata)}
    for horizon, blocks in scores.items():
        for role, values in blocks.items():
            for key in ('average_precision', 'brier', 'raw_brier', 'recall', 'false_positive_rate'):
                if key in values:
                    fields[f'{role}/{kind}/{key}_{horizon}ps'] = values[key]
    if is_encoder:
        update_training_summary(study, name, fields, evaluation=kind)
    else:
        write_json(study.technical/'evaluation-tracking'/name/kind/'evaluation.json',
            dict(identity=study.identity, mode='local', created_online_runs=0, fields=fields))
