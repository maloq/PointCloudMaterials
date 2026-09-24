"""Matched fresh decoder fits on immutable frozen encoder codes."""
import copy
import json

import numpy as np
import torch

from src.training_methods.bcr.data import BalancedStream, pack, corrupt
from src.training_methods.bcr.objective import per_environment
from src.training_methods.bcr.runtime import lr_at, save
from .common import decode, remaining, fixed_batch, write_json


def train_decoder(model, initial_decoder, codes, patches, records, fit, levels, config, seed,
                  root, identity, device, deadline=None, stop_after=None):
    """Codes are fixed tensors. Encoder parameters never enter the optimizer."""
    model.requires_grad_(False)
    model.decoder.load_state_dict(initial_decoder)
    model.decoder.requires_grad_(True)
    params = list(model.decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=config['learning_rate'], betas=(.9, .95), weight_decay=1e-5)
    stream = BalancedStream([records[i] for i in fit], seed+101)
    noise = torch.Generator().manual_seed(seed+211)
    root.mkdir(parents=True, exist_ok=True); start = 0
    if (root/'last.pt').exists():
        old = torch.load(root/'last.pt', map_location=device, weights_only=False)
        if old['identity'] != identity: raise ValueError('Fresh decoder resume identity changed')
        model.decoder.load_state_dict(old['decoder']); optimizer.load_state_dict(old['optimizer'])
        stream.load_state_dict(old['stream']); noise.set_state(old['noise_rng'].cpu()); start = old['step']
    elif (root/'initial-decoder.pt').exists():
        raise ValueError('Decoder initial state exists without resumable last state')
    else:
        save(root/'initial-decoder.pt', initial_decoder)
    codes = torch.as_tensor(codes, device=device)
    def persist(step):
        save(root/'last.pt', dict(identity=identity, step=step, decoder=model.decoder.state_dict(),
            optimizer=optimizer.state_dict(), stream=stream.state_dict(), noise_rng=noise.get_state(),
            config=config, initial_decoder_shared=True, encoder_frozen=True))
    persist(start)
    for step in range(start, min(config['updates'], stop_after or config['updates'])):
        try: remaining(deadline)
        except TimeoutError:
            persist(step); raise
        ids = [fit[i] for i in stream.draw(config['batch_size'])]
        clean = pack([patches[i] for i in ids], device)
        noisy, epsilon, sigma, _ = corrupt(clean, levels, model.config['encoder']['d0'], noise)
        optimizer.zero_grad(set_to_none=True); total = 0.
        for group in optimizer.param_groups:
            group['lr'] = lr_at(step, config['updates'], config['learning_rate'], config['minimum_learning_rate'])
        for first in range(0, len(ids), config['microbatch']):
            sl = slice(first, first+config['microbatch'])
            a, b = ({k: v[sl] for k, v in x.items()} for x in (clean, noisy))
            pred = decode(model.decoder, b, codes[ids[sl]], sigma[sl], model.config['encoder']['d0'])
            loss = per_environment(pred, epsilon[sl], a, model.config['encoder']['radius']).sum()/len(ids)
            if not torch.isfinite(loss): raise FloatingPointError(f'Fresh decoder loss at update {step}')
            loss.backward(); total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); optimizer.step()
        if (step+1) % 50 == 0:
            with (root/'training.jsonl').open('a') as stream_file:
                stream_file.write(json.dumps(dict(step=step+1, loss=total, examples=stream.exposures))+'\n')
        if (step+1) % config['save_every'] == 0: persist(step+1)
    done = min(config['updates'], stop_after or config['updates']); persist(done)
    return done


@torch.no_grad()
def evaluate_decoder(study, model, codes, device, deadline):
    patches = [study.patches[i] for i in study.chosen]
    code = torch.as_tensor(codes[study.chosen], device=device)
    errors = np.empty((len(study.manifest['noise_levels']), study.pilot['evaluation_draws'], len(patches)))
    for k in range(len(study.manifest['noise_levels'])):
        for draw in range(study.pilot['evaluation_draws']):
            for start in range(0, len(patches), 16):
                remaining(deadline); stop = min(start+16, len(patches))
                clean, noisy, epsilon, sigma = fixed_batch(patches, start, stop, k, draw,
                    study.manifest['noise_levels'], study.manifest['d0'], device)
                prediction = decode(model.decoder, noisy, code[start:stop], sigma, study.manifest['d0'])
                errors[k, draw, start:stop] = per_environment(prediction, epsilon, clean, study.manifest['radius_A']).cpu().numpy()
    return errors


def run(study, device, deadline=None):
    initial = study.model(0, 'cpu')
    decoder_state = copy.deepcopy(initial.decoder.state_dict())
    variants = [(str(step), step) for step in study.config['decoder_steps']]
    if study.config['vicreg_checkpoint'] is not None: variants.append(('vicreg', None))
    for name, step in variants:
        root = study.technical/'fresh_decoders'/name
        if (root/'complete.json').exists(): continue
        remaining(deadline)
        if step is None:
            from src.project_runtime.paths import resolve_path
            from src.research.bcr_pilot.compare import load_model
            saved = torch.load(resolve_path(study.config['vicreg_checkpoint']), map_location='cpu', weights_only=False)
            if saved['config']['arm'] != 'vicreg' or saved['config']['encoder'] != initial.config['encoder']:
                raise ValueError('VICReg comparator must have identical input support/export architecture')
            model = load_model(resolve_path(study.config['vicreg_checkpoint']), study.manifest, device)
            with torch.no_grad():
                codes = torch.cat([model.encode(pack(study.patches[i:i+16], device)).cpu()
                                   for i in range(0, len(study.patches), 16)]).numpy()
        else:
            model = study.model(step, device); codes = study.features(step)['exported']
        before = {k: v.detach().cpu().clone() for k, v in model.encoder.state_dict().items()}
        done = train_decoder(model, decoder_state, codes, study.patches, study.records, study.split['train'],
            study.manifest['noise_levels'], study.config['decoder'], study.config['seed'], root,
            study.identity, device, deadline)
        for key, value in model.encoder.state_dict().items():
            torch.testing.assert_close(value.cpu(), before[key], atol=0, rtol=0)
        model.eval().requires_grad_(False)
        errors = evaluate_decoder(study, model, codes, device, deadline)
        np.savez(root/'errors.npz', errors=errors, indices=np.array(study.chosen),
                 roots=np.array([study.records[i]['root'] for i in study.chosen]))
        write_json(root/'complete.json', dict(identity=study.identity, encoder=name, decoder_updates=done,
            encoder_unchanged=True, encoder_checkpoint_sha256=study.receipt['checkpoints'][name] if step is not None else study.receipt['vicreg_checkpoint_sha256'],
            initial_decoder_source='Original BCR initial.pt, identical tensors across all arms',
            mean_nmse_by_level={str(level): float(errors[k].mean()) for k, level in enumerate(study.manifest['noise_levels'])}))
        print(f'Finished fresh decoder on frozen encoder {name}', flush=True)
