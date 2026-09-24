"""Matched scratch MACE fits with exact pass accounting and checkpoint resume."""
import json
import signal
import time
from pathlib import Path

import torch
from torch import nn

from src.data.structural_pretraining.batches import move
from src.data.structural_pretraining.prepare import digest, file_hash
from src.research.encoder_screen.common import write
from src.training_methods.neighborhood_jepa.regularization.model import Encoder
from src.training_methods.neighborhood_jepa.execution import loader, training_step, prime_encoder
from src.training_methods.shared_pretraining.compilation import compile_encoder
from src.training_methods.shared_pretraining.runtime import learning_rate
from src.experiment_runner.metric_docs import write_metric_table
from .data import Data, PassBatches
from .objective import Objective


class Model(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.encoder = Encoder(channels, 'raw')


def build(config, item, data, device='cuda'):
    torch.manual_seed(item['seed'])
    model = Model(config['encoder_channels']).to(device)
    model.encoder.geometry_scales.copy_(torch.tensor(data.manifest['geometry_scales'], device=device))
    objective = Objective(item['treatment'], config['epi_weight']).to(device)
    return model, objective


@torch.no_grad()
def calibrate(model, objective, data, config, seed):
    from src.training_methods.neighborhood_jepa.regularization.objective import epiplexity
    batches = PassBatches(data.train, config['batch_size'], seed+41, 0, 4)
    values = []
    model.eval()
    for packed, target in loader(data, batches, config['microbatch']):
        z = []
        for b in packed:
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config['precision']=='bf16'):
                z.append(model.encoder(move(b, 'cuda')).float()[:, :128])
        z = torch.cat(z).reshape(len(target['index']), 2, 128)
        h = target['reservoir'].cuda()
        values.extend(epiplexity(z[:, i], h[:, i]) for i in (0, 1))
    scale = torch.stack(values).mean()
    if not torch.isfinite(scale) or scale <= 1e-6:
        raise FloatingPointError(f'Degenerate training-only initial Epi scale: {scale}')
    objective.epi_initial_scale.copy_(scale)
    model.train()


def train(config, item, deadline):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = Path(config['output'])/'technical/fits'/item['name']
    root.mkdir(parents=True, exist_ok=True)
    data = Data(config)
    batches_per_pass = len(data.train)//config['batch_size']
    if batches_per_pass*config['batch_size'] != len(data.train):
        raise ValueError('This study requires complete equal statistical batches')
    total = config['epochs']*batches_per_pass
    files = json.loads((Path(config['output'])/'technical/code-files.json').read_text())
    metadata = dict(protocol=config['protocol'], config=config, item=item, files=files,
                    data_identity=data.manifest['identity'], views=[[1, 0], [2, 0]],
                    sampling='one permutation of every fitting anchor per pass',
                    selection='fixed epochs; no development-selected best checkpoint',
                    loss_inputs=['current/next invariant state', 'frozen random reservoir'],
                    train_anchors=len(data.train), steps_per_pass=batches_per_pass, updates=total)
    identity = digest(metadata)
    complete = root/'complete.json'
    if complete.exists():
        if json.loads(complete.read_text())['identity'] != identity:
            raise ValueError('Completed fit identity changed')
        return True
    manifest = root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError('Immutable paired-MACE manifest changed')
    write(manifest, metadata)
    model, objective = build(config, item, data)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    saved = torch.load(root/'last.pt', map_location='cuda', weights_only=False) if (root/'last.pt').exists() else None
    step = 0
    if saved:
        if saved['identity'] != identity:
            raise ValueError('Resume identity changed')
        model.load_state_dict(saved['model'], strict=True)
        objective.load_state_dict(saved['objective'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        step = saved['step']
    example = next(iter(loader(data, [data.train[:2]], config['microbatch'])))[0][0]
    if config['compile']:
        compile_encoder(model.encoder, move(example, 'cuda'), config['precision'])
        prime_encoder(model.encoder, example, config['precision'])
    if not saved:
        calibrate(model, objective, data, config, item['seed'])
    else:
        torch.set_rng_state(saved['rng'].cpu())
        torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    stop = False

    def request_stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)

    def checkpoint(name):
        tmp = root/(name+'.building')
        torch.save(dict(identity=identity, manifest=metadata, spec=dict(export_norm='raw'),
                        model=model.state_dict(), objective=objective.state_dict(),
                        optimizer=optimizer.state_dict(), step=step, epoch=step/batches_per_pass,
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), tmp)
        tmp.replace(root/name)

    if not saved:
        checkpoint('epoch-000.pt')
        checkpoint('last.pt')
    started = time.monotonic()
    sampler = PassBatches(data.train, config['batch_size'], item['seed'], step, total)
    stream = loader(data, sampler, config['microbatch'], config['loader_workers'])
    last_record = json.loads((root/'training.jsonl').read_text().splitlines()[-1]) if step == total else None
    for packed, target in stream:
        if stop or time.time() > deadline-240:
            checkpoint('last.pt')
            write(root/'status.json', dict(state='checkpointed', step=step, epoch=step/batches_per_pass))
            return False
        rate = learning_rate(step+1, total, config['learning_rate'], .1, .01)
        for group in optimizer.param_groups:
            group['lr'] = rate
        optimizer.zero_grad(set_to_none=True)
        value, terms, diagnostics = training_step(model, objective, packed, move(target, 'cuda'),
            config['precision'], diagnose=step==0, retain_chunks=0, gpu_cache=True)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        step += 1
        last_record = dict(step=step, epoch=step/batches_per_pass, anchor_exposures=step*config['batch_size'],
            loss=value, learning_rate=rate, gradient_norm=float(norm), terms=terms,
            diagnostics=diagnostics, elapsed_seconds=time.monotonic()-started)
        if step % config['log_every'] == 0 or step == 1:
            with (root/'training.jsonl').open('a') as f:
                f.write(json.dumps(last_record)+'\n')
            write(root/'status.json', dict(state='training', **last_record))
            print(json.dumps(last_record), flush=True)
        if step % batches_per_pass == 0:
            epoch = step//batches_per_pass
            if epoch in config['milestones']:
                checkpoint(f'epoch-{epoch:03d}.pt')
            checkpoint('last.pt')
    checkpoint('last.pt')
    write_metric_table(last_record, root, family='mace_epi', name='final-training')
    write(complete, dict(state='complete', identity=identity, steps=step, epochs=config['epochs'],
                         checkpoint_sha256=file_hash(root/'last.pt')))
    write(root/'status.json', dict(state='complete', steps=step, epochs=config['epochs']))
    return True
