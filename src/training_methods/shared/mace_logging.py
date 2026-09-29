"""Checkpoint and online tracking utilities for MACE training."""

import torch


def save_checkpoint(
    path, model, optimizer, cfg, epoch, step, seen, validation
):
    temp = path.with_suffix('.tmp')
    torch.save(
        dict(
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            config=cfg,
            epoch=epoch,
            step=step,
            anchor_exposures=seen,
            validation=validation,
        ),
        temp,
    )
    temp.replace(path)


def flatten_metrics(prefix, values):
    result = {}
    for key, value in values.items():
        if isinstance(value, dict):
            result.update(flatten_metrics(prefix + '/' + key, value))
        else:
            result[prefix + '/' + key] = value
    return result


def start_wandb(cfg, out):
    from src.experiment_runner.wandb_tracking import DEFAULTS, start_online_run
    settings = DEFAULTS | cfg['wandb']
    run = start_online_run(
        settings, run_id=settings['id'], name=settings['name'], config=cfg,
        folder=out, receipt_path=out/'wandb_run.json', job_type='encoder',
        group=settings.get('group'),
    )
    run.define_metric('training_step')
    run.define_metric('train/*', step_metric='training_step')
    run.define_metric('validation/*', step_metric='training_step')
    return run
