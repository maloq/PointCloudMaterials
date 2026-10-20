"""Fixed-budget literature objectives with sealed data and independent evaluation."""
import math
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn

from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from src.research.ni_kinetic_latent.model import Model as KineticModel, mlp
from src.research.ni_kinetic_latent.learning import _latents
from .data import load, root, read, sha, digest, atomic_save, write_json, FAMILY
from .model import BlockFlow, MemoryTransition
from .representation import make_model, representation_loss, representation_parameters, fit_flow_prior


class Model(nn.Module):
    """Common measured-state/latent interface; predicted futures never use saved context."""
    def __init__(self, c, arm, bank=None):
        super().__init__()
        self.arm = arm
        self.input_dimension = arm['input_dimension']
        kind = arm['representation']
        self.dimension = 244 if kind == 'direct' else c['model']['latent']
        self.support = None
        if arm.get('measurement_support', False):
            from .support import MeasurementSupport
            self.support = MeasurementSupport(bank, epsilon=c['support']['epsilon'])
        if kind != 'direct':
            self.representation = (make_model(c, kind, self.input_dimension)
                if arm['source_arm'] is None else KineticModel(c, kind, self.input_dimension))
        if arm['predictor'] == 'block':
            self.predictor = BlockFlow(self.dimension,
                dict(c['flow'], history_noise=arm['history_noise']))
        else:
            self.predictor = MemoryTransition(self.dimension, c['model']['width'],
                c['model']['mixtures'], arm['memory'], history_noise=arm['history_noise'])

    def encode(self, x):
        if self.support is not None:
            return self.support.encode(x)
        return x if self.arm['representation'] == 'direct' else self.representation.encode(x)

    def decode(self, z):
        if self.support is not None:
            return self.support.decode(z)
        return z if self.arm['representation'] == 'direct' else self.representation.decode(z)

    @torch.no_grad()
    def rollout(self, historyx, steps, particles, seed, deterministic=False):
        h = self.encode(historyx)
        if self.arm['predictor'] == 'block':
            z = self.predictor.sample(h, particles, seed, deterministic=deterministic)[:, :, :steps]
        else:
            h = h[:, None].expand(-1, particles, -1, -1).flatten(0, 1)
            rng = torch.Generator(device=h.device).manual_seed(seed)
            future = []
            for _ in range(steps):
                z = self.predictor.mean(h) if deterministic else self.predictor.sample(h, rng)
                future.append(z)
                h = torch.cat((h[:, 1:], z[:, None]), 1)
            z = torch.stack(future, 1).reshape(len(historyx), particles, steps, -1)
        return z, self.decode(z)


def fitted(c, arm):
    folder = root(c) / 'analyses' / arm['name'] / 'technical'
    done = read(folder / 'complete.json')
    if sha(folder / 'final.pt') != done['checkpoint_sha256']:
        raise ValueError('Changed completed model: ' + arm['name'])
    state = torch.load(folder / 'final.pt', map_location='cuda', weights_only=False)
    expected = digest([read(root(c) / 'technical/plan.json')['identity'], arm])
    if state['identity'] != expected or done['identity'] != expected:
        raise ValueError('Checkpoint/plan identity differs: ' + arm['name'])
    bank = load(c, arm)[1] if arm.get('measurement_support', False) else None
    model = Model(c, arm, bank).cuda()
    model.load_state_dict(state['model'])
    if arm['predictor'] == 'block':
        model.predictor.load_state_dict(state['predictor_ema'])
    return model.eval(), state


def _initialize(model, plan, arm, inputs, targets):
    if model.support is not None:
        model.support.fit(targets)
    if arm['source_arm'] is not None:
        source = plan['checkpoints'][arm['name']]
        if sha(source['path']) != source['sha256']:
            raise ValueError('Changed reused encoder checkpoint: ' + source['path'])
        state = torch.load(source['path'], map_location='cuda', weights_only=False)
        if state['identity'] != source['identity']:
            raise ValueError('Source encoder identity mismatch')
        model.representation.load_state_dict(state['model'])
    elif arm['representation'].startswith('fmrc'):
        fit_flow_prior(model.representation, targets.flatten(0, 1))


def _objective(model, stage, x, inputs, latent, indices, update, weights, c):
    if stage == 'representation':
        rows, frames = indices // 51, indices % 51 + 3
        lag = c['representation']['lags'][(update-1) % len(c['representation']['lags'])]
        return representation_loss(model.representation, inputs[rows, frames],
            inputs[rows, frames+lag], dict(c['representation'], lag=lag))
    if stage == 'decoder':
        rows, frames = indices // 51, indices % 51 + 3
        prediction = model.decode(latent[rows, frames])
        loss = ((prediction.float() - x[rows, frames]).square()*weights).sum(-1).mean()
        return loss, dict(reconstruction=loss.detach())
    history = latent[indices, :4]
    if model.arm['predictor'] == 'block':
        clean, loss, scores = model.predictor.training_prediction(history, latent[indices, 4:])
        if model.arm['decoded_weight']:
            decoded = model.decode(clean)
            auxiliary = ((decoded.float()-x[indices, 4:]).square()*weights).sum(-1).mean()
            loss = loss + model.arm['decoded_weight']*auxiliary
            scores = dict(scores, decoded_auxiliary=auxiliary.detach())
        return loss, scores
    # Every chosen sequence supplies all sixty observed one-step transitions.
    frames = torch.arange(3, 63, device=x.device)
    h = latent[indices[:, None, None], frames[None, :, None]+torch.arange(-3, 1, device=x.device)]
    following = latent[indices, 4:]
    loss = -model.predictor.log_prob(h, following).mean()/model.dimension
    return loss, dict(nll_per_dimension=loss.detach())


def fit(c, arm, deadline, *, debug=False, max_updates=None, family=FAMILY):
    folder = root(c) / 'analyses' / arm['name'] / 'technical'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'complete.json').exists():
        return True
    plan, bank = load(c, arm)
    train = np.asarray(bank['train_rows'])
    if arm.get('training_sources') is not None:
        sources = sorted({bank['rows'][i]['ancestor'] for i in train},
                         key=lambda source: digest([c['seed'], source]))[:arm['training_sources']]
        train = np.asarray([i for i in train if bank['rows'][i]['ancestor'] in sources])
    write_json(folder/'training-rows.json', dict(indices=train.tolist(),
        ancestors=sorted({bank['rows'][i]['ancestor'] for i in train}),
        preprocessing='Shared reference normalization from all 16 declared training ancestors; predictor normalization and support fit use only selected rows'))
    x = bank['x'][train].cuda()
    inputs = x if arm['input_dimension'] == 244 else bank['input'][train].cuda()
    weights = bank['weights'].cuda()
    settings = c['training']
    identity = digest([plan['identity'], arm])
    torch.manual_seed(arm['seed'])
    rng = np.random.default_rng(arm['seed'])
    model = Model(c, arm, bank).cuda()
    del bank
    previous = None
    stages = arm.get('stages', (['representation', 'decoder'] if arm['source_arm'] is None and
              arm['representation'] != 'direct' else []) + ['dynamics'])
    completed, records = [], []
    global_update = 0
    ema = None
    if (folder / 'last.pt').exists():
        previous = torch.load(folder / 'last.pt', map_location='cuda', weights_only=False)
        if previous['identity'] != identity:
            raise ValueError('Resume identity mismatch: ' + arm['name'])
        model.load_state_dict(previous['model'])
        completed, records = previous['completed'], previous['history']
        global_update = previous['global_update']
        ema = previous['predictor_ema']
        rng.bit_generator.state = previous['rng']
        torch.set_rng_state(previous['torch_rng'].cpu())
        torch.cuda.set_rng_state(previous['cuda_rng'].cpu())
    else:
        _initialize(model, plan, arm, inputs, x)
    optimizer = None
    stage, stage_update = stages[0], 0
    latent = None

    def save(final=False, filename=None):
        atomic_save(folder / (filename or ('final.pt' if final else 'last.pt')), dict(
            identity=identity, data_identity=plan['data_identity'], arm=arm,
            model=model.state_dict(), predictor_ema=ema,
            optimizer=optimizer.state_dict() if optimizer else None,
            stage=stage, stage_update=stage_update, global_update=global_update,
            update=global_update, completed=completed, history=records,
            rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state()))

    write_json(folder / f'execution-{time.time_ns()}.json', dict(
        allocation=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(),
        parameters=sum(p.numel() for p in model.parameters()), train_sequences=len(x),
        input_dimension=inputs.shape[-1], target_dimension=244, stages=stages,
        precision='BF16 matrix operations, FP32 parameters/optimizer/loss reductions; no physical integration',
        checkpoint='fixed final stage budgets', resumed_update=global_update,
        input='fixed central atom; four observed native0.1ps states; optional observed20A context; no conditions',
        debug=debug))
    context = (nullcontext(None) if debug else online_training(c['wandb'],
        run_id='niflow-'+identity[:16], name=arm['name'], config=dict(c, arm=arm),
        folder=folder, receipt_path=folder/'wandb.json',
        job_type='encoder' if stages == ['representation'] else 'predictor',
        group=c['protocol'], receipt_fields=dict(identity=identity)))
    with context as tracking:
        for stage_index, stage in enumerate(stages):
            if stage in completed:
                continue
            start = previous['stage_update'] if previous is not None and previous['stage'] == stage else 0
            for p in model.parameters():
                p.requires_grad_(False)
            if stage == 'representation':
                parameters = list(representation_parameters(model.representation))
                count = len(x)*51
                total = math.ceil(count/settings['batch_size'])*settings['repr_epochs']
                warmup = math.ceil(count/settings['batch_size'])
            else:
                if latent is None:
                    with torch.no_grad():
                        latent = _latents(model, inputs)
                if stage == 'decoder':
                    if start == 0:
                        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                            torch.manual_seed(arm['seed']+207)
                            model.representation.decoder = mlp(model.dimension, c['model']['width'], 244).cuda()
                    parameters = list(model.representation.decoder.parameters())
                    count = len(x)*51
                    total = math.ceil(count/settings['batch_size'])*settings['decoder_epochs']
                    warmup = math.ceil(count/settings['batch_size'])
                else:
                    if start == 0:
                        flat = latent.flatten(0, 1)
                        model.predictor.location.copy_(flat.mean(0))
                        model.predictor.scale.copy_(flat.std(0, unbiased=False).clamp_min(1e-5))
                    parameters = list(model.predictor.parameters())
                    if arm['predictor'] == 'block' and ema is None:
                        ema = {key:value.detach().clone() for key,value in model.predictor.state_dict().items()}
                    count = len(x)
                    total = settings['dynamics_updates']
                    warmup = settings['dynamics_warmup']
            for p in parameters:
                p.requires_grad_(True)
            optimizer = torch.optim.AdamW(parameters, lr=settings['lr'], weight_decay=settings['weight_decay'])
            if start:
                optimizer.load_state_dict(previous['optimizer'])
            batches = math.ceil(count/settings['batch_size'])
            order, order_epoch = None, -1
            model.train()
            for stage_update in range(start+1, total+1):
                if time.time() > deadline-180:
                    stage_update -= 1
                    save()
                    return False
                begin = time.monotonic()
                epoch, offset = divmod(stage_update-1, batches)
                if epoch != order_epoch:
                    order = np.random.default_rng(arm['seed']+100000*stage_index+epoch).permutation(count)
                    order_epoch = epoch
                indices = torch.as_tensor(order[offset*settings['batch_size']:(offset+1)*settings['batch_size']], device='cuda')
                warm = min(stage_update/warmup, 1.)
                cosine = .01+.99*.5*(1+math.cos(math.pi*max(0,stage_update-warmup)/max(1,total-warmup)))
                for group in optimizer.param_groups:
                    group['lr'] = settings['lr']*warm*cosine
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=c['training']['mixed_precision']):
                    loss, scores = _objective(model, stage, x, inputs, latent, indices, stage_update, weights, c)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite {stage} objective at {stage_update}: {arm["name"]}')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, settings['gradient_clip'], error_if_nonfinite=True)
                optimizer.step()
                if stage == 'dynamics' and ema is not None:
                    with torch.no_grad():
                        parameter_names = dict(model.predictor.named_parameters())
                        for key,value in model.predictor.state_dict().items():
                            if key in parameter_names:
                                ema[key].lerp_(value.detach(), 1-c['flow']['ema_decay'])
                            else:
                                ema[key].copy_(value)
                global_update += 1
                record = dict(update=global_update, stage=stage, stage_update=stage_update,
                    loss=float(loss.detach()), gradient_norm=float(norm), lr=optimizer.param_groups[0]['lr'],
                    **{key: float(value.detach()) if torch.is_tensor(value) else float(value) for key,value in scores.items()})
                records.append(record)
                if tracking is not None:
                    tracking.log({f'{stage}/{key}': value for key,value in record.items()
                                  if key not in ('stage','update','stage_update')}, step=global_update)
                if stage_update == 1 or stage_update % settings['save_every'] == 0 or stage_update == total:
                    save()
                    write_json(folder/'progress.json', dict(record, total_stage_updates=total,
                        stages_completed=completed, last_update_seconds=time.monotonic()-begin,
                        updated=time.time(), target_stage_updates=total))
                    print(arm['name'], stage, stage_update, '/', total, 'loss', record['loss'], flush=True)
                if stage == 'dynamics' and stage_update in settings.get('snapshot_updates', []):
                    save(filename=f'update-{stage_update}.pt')
                if max_updates is not None and global_update >= max_updates:
                    save()
                    return False
            completed.append(stage)
            save()
            previous = None
        model.eval()
        save(final=True)
        if not debug:
            columns = sorted({key for record in records for key in record})
            write_metric_rows(({key:r.get(key) for key in columns} for r in records),
                folder.parent, family=family, name='training', columns=columns)
        write_json(folder/'complete.json', dict(identity=identity, updates=global_update,
            completed_stages=completed, checkpoint_sha256=sha(folder/'final.pt'), debug=debug))
    return True
