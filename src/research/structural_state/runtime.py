"""Matched scratch fits, physical heads and replayable allocation checkpoints."""
import copy
import json
import math
import time

import numpy as np
import torch

from src.models.encoders.mace_backend import mace_backend_metadata
from .common import save_checkpoint, write_json, remaining, sha
from .data import Corpus, PairStream
from .model import StructuralModel, GraphBank, block_error, objective, pair_distances, teacher_distances, calibrate_heads
from .data import BLOCKS
from .dynamics import PROTOCOL, targets as dynamics_targets


def setup(config, corpus, domain, device):
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    ec = dict(config['encoder'], d0=corpus.manifest['d0'], n_ref=corpus.manifest['n_ref'])
    model = StructuralModel(ec, dynamics=config.get('protocol') in (PROTOCOL, 'fixed_geometry_parameter_search_v4')).to(device)
    return model


def load_bank(study, model, domain, device):
    with np.load(study.cache / f'{domain}-graphs.npz') as arrays:
        return GraphBank(dict(arrays), model.encoder, device)


@torch.no_grad()
def exports(model, bank, indices, chunk):
    pooled, exported = [], []
    model.eval()
    for start in range(0, len(indices), chunk):
        p = model.encoder.pooled_graph(bank.batch(indices[start:start+chunk]))
        pooled.append(p.cpu().numpy())
        exported.append(model.encoder.export_pooled(p).cpu().numpy())
    return dict(pooled=np.concatenate(pooled), exported=np.concatenate(exported))


@torch.no_grad()
def initialize(model, bank, corpus, target, config):
    fit = corpus.split['fit']
    tc = config['training']
    pooled = exports(model, bank, fit, tc['microbatch'])['pooled'].astype(np.float64)
    model.encoder.pooled_mean.copy_(torch.as_tensor(pooled.mean(0)))
    model.encoder.pooled_scale.copy_(torch.as_tensor(pooled.std(0).clip(1e-6)))
    features = exports(model, bank, fit, tc['microbatch'])['exported']
    receipt = calibrate_heads(model, torch.as_tensor(features, device=bank.device),
        {d: y[fit] for d, y in target.items()}, tc['head_ridge'], tc['head_norm_bound'])
    return receipt


@torch.no_grad()
def physical_diagnostics(model, bank, corpus, target, arm, chunk):
    """Score the actual training decoder before any fresh probe rescaling."""
    result = {}
    for role in ('fit', 'tune'):
        ids = corpus.split[role]
        z = torch.as_tensor(exports(model, bank, ids, chunk)['exported'], device=bank.device)
        prediction = model.heads[arm['input']](z)
        y = target[arm['input']][ids]
        blocks = {key: float((prediction[:, sl]-y[:, sl]).square().mean()) for key, sl in BLOCKS.items()}
        result[role] = dict(blocks=blocks, mse=sum(blocks.values())/len(blocks),
            constant_mse=float(block_error(torch.zeros_like(y), y).mean()),
            export_rms_std=float(z.double().std(0, correction=0).square().mean().sqrt()))
        if 'current_order' in model.heads:
            result[role]['auxiliary_mse'] = {name: float((model.heads[name](z)-target[name][ids]).square().mean())
                                             for name in ('current_order','future_residual')}
    return result


def train(study, name, device='cuda', deadline=None, stop_after=None, directory=None):
    config = study.config
    arm = study.arm(name)
    corpus = Corpus(study)
    folder = directory or study.technical / 'fits' / name
    folder.mkdir(parents=True, exist_ok=True)
    complete = folder / 'complete.json'
    if complete.exists():
        if json.loads(complete.read_text())['identity'] != study.identity:
            raise ValueError(f'Completed fit identity changed: {name}')
        return True
    model = setup(config, corpus, arm['input'], device)
    bank = load_bank(study, model, arm['input'], device)
    target = {d: torch.as_tensor((v - corpus.scalers[d]['mean']) / corpus.scalers[d]['scale'], device=device)
              for d, v in corpus.geometry.items()}
    auxiliary_receipt = None
    if config.get('protocol') in (PROTOCOL, 'fixed_geometry_parameter_search_v4'):
        auxiliary, auxiliary_receipt, _ = dynamics_targets(corpus, config)
        target.update({d: torch.as_tensor(v, device=device) for d, v in auxiliary.items()})
        write_json(folder/'future-targets.json', auxiliary_receipt)
    tc = config['training']
    if tc['batch_size'] % tc['microbatch'] or tc['microbatch'] % 2:
        raise ValueError('Even microbatches must divide the statistical batch and preserve pairs')
    if (folder / 'initial.pt').exists():
        initial_checkpoint = torch.load(folder / 'initial.pt', weights_only=False, map_location=device)
        if initial_checkpoint['identity'] != study.identity:
            raise ValueError('Initial calibration identity changed')
        model.load_state_dict(initial_checkpoint['model'])
        calibration_receipt = initial_checkpoint['calibration']
        initial_diagnostics = initial_checkpoint['initial_diagnostics']
    else:
        calibration_receipt = initialize(model, bank, corpus, target, config)
        initial_diagnostics = physical_diagnostics(model, bank, corpus, target, arm, tc['microbatch'])
    initial = copy.deepcopy(model.state_dict())
    optimizer = torch.optim.AdamW([
        dict(params=model.encoder.parameters(), lr=arm.get('encoder_lr', tc['encoder_lr'])),
        dict(params=model.heads.parameters(), lr=tc['head_lr'])], weight_decay=tc['weight_decay'])
    stream = PairStream(corpus.records, corpus.split['fit'], corpus.targets['phase'], config['seed'] + 1)
    step, best = 0, float('inf')
    # Fixed scale initialization is train-only and does not consume the fit stream.
    calibration = PairStream(corpus.records, corpus.split['fit'], corpus.targets['phase'], config['seed'] + 2).draw(256)
    with torch.no_grad():
        z = exports(model, bank, calibration, tc['microbatch'])['exported']
        scale_z = float(pair_distances(torch.as_tensor(z)).mean())
        scale_g = float(teacher_distances(target[arm['input']][calibration]).mean())
    if min(scale_z, scale_g) <= 1e-10:
        raise ValueError('Collapsed initial embedding or teacher relation')
    if (folder / 'last.pt').exists():
        saved = torch.load(folder / 'last.pt', weights_only=False, map_location=device)
        if saved['identity'] != study.identity or saved['arm'] != name:
            raise ValueError('Resume identity changed')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        stream.load_state_dict(saved['stream'])
        step, best, scale_z, scale_g = saved['step'], saved['best'], saved['scale_z'], saved['scale_g']
        torch.set_rng_state(saved['torch_rng'].cpu())
        np.random.set_state(saved['numpy_rng'])
        if torch.device(device).type == 'cuda':
            torch.cuda.set_rng_state(saved['cuda_rng'].cpu(), device)
    started = time.monotonic()

    def state():
        return dict(identity=study.identity, arm=name, model=model.state_dict(), optimizer=optimizer.state_dict(),
                    step=step, best=best, scale_z=scale_z, scale_g=scale_g, stream=stream.state_dict(),
                    torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                    cuda_rng=torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None,
                    encoder_config=dict(config['encoder'],
                        d0=corpus.manifest['d0'], n_ref=corpus.manifest['n_ref']),
                    target_scalers=corpus.scalers, calibration=calibration_receipt,
                    initial_diagnostics=initial_diagnostics, auxiliary_targets=auxiliary_receipt)

    if step == 0:
        save_checkpoint(folder / 'initial.pt', state())
        write_json(folder / 'initial-diagnostics.json', initial_diagnostics)
    write_json(folder / 'environment.json', dict(backend=mace_backend_metadata(config['encoder']['backend']),
        torch=torch.__version__, gpu=torch.cuda.get_device_name(device) if torch.device(device).type == 'cuda' else 'cpu',
        precision='float32; TF32 disabled', geometry_resident=True, config=config,
        parameters=sum(p.numel() for p in model.parameters())))
    limit = min(tc['updates'], stop_after if stop_after is not None else tc['updates'])
    try:
        while step < limit:
            remaining(deadline)
            indices = stream.draw(tc['batch_size'])
            factor = min(1., (step+1)/tc['warmup']) * (.02 + .98*.5*(1+math.cos(math.pi*step/tc['updates'])))
            for group, rate in zip(optimizer.param_groups, (arm.get('encoder_lr', tc['encoder_lr']), tc['head_lr']), strict=True):
                group['lr'] = factor * rate
            optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)
            distance = torch.zeros((), device=device)
            model.train()
            current_arm = dict(arm, relation_weight=arm['relation_weight'] *
                               min(1., step / tc['relation_warmup']))
            for start in range(0, len(indices), tc['microbatch']):
                ix = indices[start:start+tc['microbatch']]
                z = model(bank.batch(ix))
                value, amplitude = objective(model, z, {d: v[ix] for d, v in target.items()}, current_arm, (scale_z, scale_g))
                weighted = value * (len(ix)/len(indices))
                weighted.backward()
                total += weighted.detach()
                distance += amplitude * (len(ix)/len(indices))
            if not torch.isfinite(total):
                raise FloatingPointError(f'Nonfinite objective in {name} at update {step}, rows={indices.tolist()}')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            model.bound_heads(tc['head_norm_bound'])
            # Fixed fit-only reference scales. A moving detached denominator can
            # chase shrinking features and oppose the reconstruction objective.
            step += 1
            if step in tc.get('milestones', []):
                save_checkpoint(folder/f'step-{step}.pt',state())
            if step % tc['log_every'] == 0 or step == limit:
                record = dict(step=step, loss=float(total), gradient_norm=float(norm), scale_z=scale_z,
                              current_pair_distance=float(distance), relation_weight=current_arm['relation_weight'],
                              anchor_exposures=stream.exposures, seconds=time.monotonic()-started)
                with (folder / 'training.jsonl').open('a') as stream_file:
                    stream_file.write(json.dumps(record) + '\n')
                print(json.dumps(dict(arm=name, **record)), flush=True)
            if step % tc['evaluate_every'] == 0 or step == limit:
                diagnostics = physical_diagnostics(model, bank, corpus, target, arm, tc['microbatch'])
                spread = diagnostics['fit']['export_rms_std'] / initial_diagnostics['fit']['export_rms_std']
                if not np.isfinite(spread) or spread < tc['minimum_spread_ratio']:
                    save_checkpoint(folder / 'last.pt', state())
                    raise FloatingPointError(f'Export amplitude failure in {name}: final/initial RMS spread={spread:g}')
                diagnostics.update(step=step, spread_ratio=spread,
                    retained_blocks={k: diagnostics['tune']['blocks'][k] <=
                        (1 + config['retention_tolerance_relative_mse']) * v
                        for k, v in initial_diagnostics['tune']['blocks'].items()})
                with (folder / 'diagnostics.jsonl').open('a') as stream_file:
                    stream_file.write(json.dumps(diagnostics) + '\n')
                model.eval()
                errors = []
                with torch.no_grad():
                    for start in range(0, len(corpus.split['tune']), tc['microbatch']):
                        ix = corpus.split['tune'][start:start+tc['microbatch']]
                        z = model(bank.batch(ix))
                        error = block_error(model.heads[arm['input']](z), target[arm['input']][ix])
                        if arm['relaxed_weight']:
                            error += arm['relaxed_weight'] * block_error(model.heads['relaxed'](z), target['relaxed'][ix])
                        errors.append(error.cpu().numpy())
                tuning = float(np.concatenate(errors).mean())
                with (folder / 'tuning.jsonl').open('a') as stream_file:
                    stream_file.write(json.dumps(dict(step=step, physical_mse=tuning)) + '\n')
                if tuning < best:
                    best = tuning
                    save_checkpoint(folder / 'best.pt', state())
                save_checkpoint(folder / 'last.pt', state())
            elif step % tc['save_every'] == 0:
                save_checkpoint(folder / 'last.pt', state())
        save_checkpoint(folder / 'last.pt', state())
    except TimeoutError:
        save_checkpoint(folder / 'last.pt', state())
        write_json(folder / 'status.json', dict(state='checkpointed', step=step, identity=study.identity))
        return False
    if step != tc['updates']:
        write_json(folder / 'status.json', dict(state='checkpointed', step=step, identity=study.identity))
        return False
    # Primary exports are the matched final update, not a selected best checkpoint.
    all_rows = np.arange(len(corpus.records))
    final = exports(model, bank, all_rows, tc['microbatch'])
    model.load_state_dict(initial)
    reference = exports(model, bank, all_rows, tc['microbatch'])
    np.savez(folder / 'features.npz', **final, **{'initial_' + k: v for k, v in reference.items()})
    write_json(complete, dict(state='complete', identity=study.identity, step=step,
        feature_sha256=sha(folder / 'features.npz'), checkpoint_sha256=sha(folder / 'last.pt'),
        selection='Fixed final update; best tuning checkpoint retained as secondary only'))
    write_json(folder / 'status.json', dict(state='complete', step=step, identity=study.identity))
    return True
