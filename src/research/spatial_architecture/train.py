"""Twelve full, identically ordered epochs with explicit training/I/O timings."""
import math
import time

import numpy as np
import torch

from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from src.research.continuous_motion.data import atomic_save
from src.research.structural_response.data import spatial_data, root, cache, read, digest, sha, write_json
from src.research.structural_response.model import spatial_backward
from src.research.structural_response.train import validate_spatial
from .model import Spatial, teacher
from src.models.encoders.esen_runtime import configure, direct_backward
from . import parallel

FAMILY = 'spatial_architecture'
METHOD = 'vicreg_descriptors'


def folder(c, architecture):
    p = root(c) / 'analyses' / architecture / 'technical'
    p.mkdir(parents=True, exist_ok=True)
    return p


def identity(c, architecture):
    return digest(dict(plan=read(root(c) / 'technical/plan.json')['identity'], architecture=architecture))


def fit(c, architecture, deadline):
    p = folder(c, architecture)
    signature = identity(c, architecture)
    cfg = c['spatial_training']
    micro, batch = cfg['microbatch'], cfg['batch_size']
    data, columns, weights = spatial_data(c)
    weights = torch.from_numpy(weights).cuda()
    torch.manual_seed(c['seed'])
    model = Spatial(c, architecture).cuda()
    ema = teacher(c, architecture, model.geometry)
    variant = c.get('execution_variants', {}).get(architecture, 'eager')
    configure(model.geometry, architecture, variant)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    start, history = 0, []
    timing = dict(training_seconds=0., validation_seconds=0., checkpoint_seconds=0.)
    if (p / 'last.pt').exists():
        saved = torch.load(p / 'last.pt', map_location='cuda', weights_only=False)
        if saved['identity'] != signature:
            raise ValueError('Changed spatial architecture resume identity')
        model.load_state_dict(saved['model'])
        ema.load_state_dict(saved['ema'])
        opt.load_state_dict(saved['optimizer'])
        start, history, timing = saved['update'], saved['history'], saved['timing']
    parallel_step = parallel.create(c, architecture, model)
    if parallel_step is None and variant.endswith('direct') and micro != batch:
        raise ValueError('Direct full-batch training requires microbatch=batch')
    if start == cfg['updates']:
        # Recover final publication after a completed fit whose export failed.
        val = read(p / 'validation.json')
        if val.pop('update') != start:
            raise ValueError('Missing final validation for completed training')
    steps = math.ceil(len(data) / batch)
    if cfg['updates'] != steps * cfg['epochs'] or len(data) != c['data']['training_rows']:
        raise ValueError('Declared epochs do not cover the pinned training rows')

    def save(update):
        before = time.monotonic()
        atomic_save(p / 'last.pt', dict(identity=signature, model=model.state_dict(),
            ema=ema.state_dict(), optimizer=opt.state_dict(), update=update, history=history,
            timing=timing, config=c, architecture=architecture))
        timing['checkpoint_seconds'] += time.monotonic() - before

    epoch, order = -1, None
    write_json(p / 'parameters.json', dict(encoder=sum(x.numel() for x in model.geometry.parameters()),
        total=sum(x.numel() for x in model.parameters()), architecture=architecture,
        gpu=torch.cuda.get_device_name(), effective_batch=batch, microbatch=micro,
        gpu_count=len(parallel.devices(c)), per_device_batch=math.ceil(batch/len(parallel.devices(c))),
        execution_variant=variant))
    encoder = c['reference_encoder' if architecture == 'mace' else 'encoder']
    label = (f"tensornet-{encoder['num_interactions']}layers" if architecture == 'tensornet'
             else f"{architecture}-l{encoder['max_ell']}")
    if architecture == 'mace':
        label += f"-c{encoder['correlation']}"
    target_label='-'.join(dict.fromkeys(data.columns[i]['family'].replace('bond_order','bond') for i in columns))
    with online_training(c['wandb'], run_id='spatial-'+signature[:16], name=label+'-vicreg-'+target_label,
            config=dict(c, architecture=architecture, identity=signature), folder=p,
            receipt_path=p / 'wandb.json', job_type='encoder', group=c['protocol'],
            receipt_fields=dict(identity=signature)) as run:
        for update in range(start + 1, cfg['updates'] + 1):
            if time.time() > deadline - 300:
                save(update - 1)
                write_json(p / 'progress.json', dict(state='checkpointed_for_continuation',
                    update=update-1, total=cfg['updates'], **timing))
                data.close()
                if parallel_step is not None:
                    parallel_step.close()
                return False
            current_epoch = (update - 1) // steps
            if epoch != current_epoch:
                epoch = current_epoch
                order = np.random.default_rng(np.random.SeedSequence([c['seed'], epoch])).permutation(len(data))
            step = (update - 1) % steps
            parallel.synchronize(c)
            begin = time.monotonic()
            model.train()
            opt.zero_grad(set_to_none=True)
            b = data.batch(order[step*batch:(step+1)*batch])
            q = torch.from_numpy(b['positions']).cuda()
            target = torch.from_numpy(b['target'][:, columns]).cuda()
            fraction = min(1., update / cfg['warmup'])
            floor = cfg['min_lr'] / cfg['learning_rate']
            cosine = floor + (1-floor)*.5*(1+math.cos(math.pi*max(0, update-cfg['warmup'])/(cfg['updates']-cfg['warmup'])))
            opt.param_groups[0]['lr'] = cfg['learning_rate'] * fraction * cosine
            if parallel_step is not None:
                loss, stats = parallel_step.backward(q, target, weights, c['seed']+update)
            elif variant.endswith('direct'):
                loss, stats = direct_backward(model, q, target, weights, c['seed']+update)
            else:
                loss, stats = spatial_backward(model, ema, q, target, weights, METHOD, c['seed']+update, micro)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for a, b in zip(ema.parameters(), model.geometry.parameters(), strict=True):
                    a.lerp_(b, cfg['ema'])
            parallel.synchronize(c)
            seconds = time.monotonic() - begin
            timing['training_seconds'] += seconds
            if not np.isfinite(loss):
                raise FloatingPointError('Nonfinite spatial objective')
            row = dict(update=update, epoch=epoch+(step+1)/steps, update_seconds=seconds,
                learning_rate=opt.param_groups[0]['lr'], loss=loss, gradient_norm=float(norm), **stats)
            if update % cfg['selection_every'] == 0 or update == cfg['updates']:
                begin = time.monotonic()
                val = validate_spatial(model, ema, c, METHOD, micro)
                for material in c.get('monitoring', {}).get('materials', []):
                    panel_config = dict(c, derived_cache=str(cache(c)/('monitor-'+material)))
                    panel = validate_spatial(model, ema, panel_config, METHOD, micro)
                    val.update({material+'_'+k:v for k,v in panel.items()})
                torch.cuda.synchronize()
                timing['validation_seconds'] += time.monotonic() - begin
                row.update({'validation_'+k: v for k, v in val.items()})
                write_json(p / 'validation.json', dict(update=update, **val))
            history.append(row)
            run.log({k: v for k, v in row.items() if k not in (
                'masked_half_mse', 'validation_masked_half_mse')}, step=update)
            if update % cfg['save_every'] == 0 or update == cfg['updates']:
                save(update)
            if update == 1 or update % 25 == 0:
                progress = dict(state='training', architecture=architecture, update=update,
                    total=cfg['updates'], epoch=row['epoch'], loss=loss, update_seconds=seconds,
                    estimated_remaining_training_seconds=np.median([r['update_seconds'] for r in history[-100:]])*(cfg['updates']-update), **timing)
                write_json(p / 'progress.json', progress)
                print(progress, flush=True)
        atomic_save(p / 'final.pt', dict(identity=signature, geometry=model.geometry.state_dict(),
            model=model.state_dict(), config=c, architecture=architecture, update=cfg['updates']))
        run.summary.update({'validation/'+k: v for k, v in val.items()})
        run.summary.update(timing)
    write_metric_rows(history, p.parent, family=FAMILY, name='learning',
        columns=list(dict.fromkeys(k for row in history for k in row)))
    epochs = []
    for epoch in range(cfg['epochs']):
        rows = history[epoch*steps:(epoch+1)*steps]
        epochs.append(dict(architecture=architecture, epoch=epoch+1, rows=len(data),
            updates=len(rows), training_seconds=sum(r['update_seconds'] for r in rows),
            mean_loss=float(np.mean([r['loss'] for r in rows]))))
    write_metric_rows(epochs, p.parent, family=FAMILY, name='epochs')
    write_json(p / 'complete.json', dict(state='complete', identity=signature, update=cfg['updates'],
        epochs=cfg['epochs'], checkpoint_sha256=sha(p / 'final.pt'), validation=val, timing=timing))
    write_json(p / 'progress.json', dict(state='complete', architecture=architecture,
        update=cfg['updates'], total=cfg['updates'], epoch=cfg['epochs'], validation=val, **timing))
    data.close()
    if parallel_step is not None:
        parallel_step.close()
    return True
