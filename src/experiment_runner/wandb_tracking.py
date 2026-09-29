"""Online scientific training and API-only updates to recorded training runs."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import time

from .artifacts import write_json

DEFAULTS = dict(entity='teshbek', project='PointCloudMaterials', mode='online')


def require_online(settings):
    if settings['mode'] != 'online' or os.environ.get('WANDB_MODE', 'online') != 'online':
        raise ValueError('W&B must remain online for scientific training')
    if os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1', 'yes'):
        raise ValueError('WANDB_DISABLED conflicts with online experiment tracking')


def start_online_run(settings, *, run_id, name, config, folder, receipt_path,
                     job_type, group=None, tags=(), receipt_fields=None):
    """Open a resumable training run and persist its identity before yielding it."""
    require_online(settings)
    if job_type not in ('encoder', 'predictor', 'control'):
        raise ValueError(f'{job_type!r} is not scientific training; keep diagnostics local')
    import wandb
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        entity=settings['entity'], project=settings['project'], mode='online',
        id=run_id, resume='allow', name=name, group=group, job_type=job_type,
        config=config, tags=list(tags), dir=str(folder), save_code=False,
        force=True, settings=wandb.Settings(init_timeout=60),
    )
    if (run is None or run.settings.mode != 'online' or run.id != run_id
            or run.entity != settings['entity'] or run.project != settings['project']):
        if run is not None:
            run.finish(exit_code=1)
        raise RuntimeError(f'W&B did not establish the requested online run: '
                           f'{settings["entity"]}/{settings["project"]}/{run_id}')
    try:
        write_json(receipt_path, dict(
            receipt_fields or {}, id=run.id, url=run.url, mode='online',
            entity=run.entity, project=run.project,
        ))
    except BaseException:
        run.finish(exit_code=1)
        raise
    return run


@contextmanager
def online_training(settings, **arguments):
    run = start_online_run(settings, **arguments)
    try:
        yield run
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)


def update_recorded_summary(receipt_path, fields, *, evaluation, expected):
    """Update the receipt's training ID through the API, without wandb.init."""
    receipt_path = Path(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    for key, value in expected.items():
        if receipt[key] != value:
            raise ValueError(f'Training receipt mismatch: {receipt_path}: {key}')
    target = receipt_path.parent / 'evaluations' / f'{evaluation}.json'
    record = dict(state='pending', run_id=receipt['id'], url=receipt['url'],
                  evaluation=evaluation, fields=fields, created_online_runs=0)
    if 'identity' in receipt:
        record['identity'] = receipt['identity']
    write_json(target, record)
    try:
        import wandb
        run = wandb.Api(timeout=60).run(
            f"{receipt['entity']}/{receipt['project']}/{receipt['id']}"
        )
        run.summary.update(fields)
    except Exception as error:
        write_json(target, dict(record, state='failed', error=repr(error)))
        raise
    write_json(target, dict(record, state='complete', updated_at=time.time()))
