"""Matched VICReg warm-start pilot; no topology labels enter optimization."""

from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.registry import sha256, write_json
from .data import read_json, training_clouds
from .engine import augment, encode, load_model, loss_from_features, replay_step
from .evaluate import extract


def view_major(data, rows):
    return [data[row][view] for view in range(3) for row in rows]


def train(config, mode):
    if mode not in config['training_modes']:
        raise ValueError(f'Mode is not part of the matched training protocol: {mode}')
    root = Path(config['output'])/'technical'/f'train-{mode}'
    root.mkdir(parents=True,exist_ok=True)
    if (root/'status.json').exists():
        raise FileExistsError(f'Existing training attempt: {root}; preserve or explicitly resume its state')
    if read_json(Path(config['output'])/'technical/verification.json')['state'] != 'complete':
        raise ValueError('Full-graph and gradient-replay verification must pass before training')
    model, original_cfg = load_model(config)
    data = training_clouds(config)
    probes = np.load(Path(config['diagnostics'])/'technical/probes.npz')
    train_ids = np.flatnonzero(probes['split']=='train')
    val_ids = np.flatnonzero(probes['split']=='val')
    source_sets = [set(probes['source'][probes['split']==s]) for s in ['train','val','test']]
    if any(source_sets[a]&source_sets[b] for a,b in [(0,1),(0,2),(1,2)]):
        raise ValueError('Source leakage in context split')
    rng = np.random.default_rng(config['seed'])
    val_ids = np.random.default_rng(config['seed']+1).permutation(val_ids)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config['learning_rate'],weight_decay=float(original_cfg.decay_rate))
    history, best = [], float('inf')
    started = time.monotonic()
    write_json(root/'config.json',dict(config=config,mode=mode,
        initial_checkpoint_sha256=sha256(Path(config['checkpoint'])),
        cache_manifest_sha256=sha256(Path(config['cache'])/'manifest.json'),
        objective='original spatial/temporal VICReg, no TDA',optimizer='AdamW',
        weight_decay=float(original_cfg.decay_rate),schedule='constant warm-start learning rate',
        training_rows=len(train_ids),validation_rows=len(val_ids)))
    try:
        for epoch in range(config['epochs']):
            model.train()
            order = rng.permutation(train_ids)
            losses = []
            for step, start in enumerate(range(0,len(order)-config['batch_size']+1,config['batch_size'])):
                rows = order[start:start+config['batch_size']]
                clouds = augment(config,model,view_major(data,rows),mode)
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = replay_step(config,model,clouds,mode)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(),float(original_cfg.gradient_clip_val),error_if_nonfinite=True)
                optimizer.step()
                losses.append(loss)
                write_json(root/'status.json',dict(state='training',mode=mode,epoch=epoch+1,
                    epochs=config['epochs'],step=step+1,steps=len(order)//config['batch_size'],
                    loss=loss,elapsed_seconds=time.monotonic()-started))
                print(f'TRAIN {mode} epoch={epoch+1} step={step+1} loss={loss:.6f} gradient_norm={float(norm):.4f}',flush=True)
            model.eval()
            val_losses, weights = [], []
            with torch.no_grad():
                for start in range(0,len(val_ids),config['batch_size']):
                    rows=val_ids[start:start+config['batch_size']]
                    z=encode(config,model,view_major(data,rows),mode)
                    val_losses.append(float(loss_from_features(model,z)[0]));weights.append(len(rows))
            validation=float(np.average(val_losses,weights=weights))
            record=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),validation_loss=validation,
                        elapsed_seconds=time.monotonic()-started)
            history.append(record)
            payload=dict(protocol='mace_context_vicreg_warm_start_v1',mode=mode,config=config,
                model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),history=history,
                epoch=epoch+1,numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),
                cuda_rng_state=torch.cuda.get_rng_state(),initial_checkpoint_sha256=sha256(Path(config['checkpoint'])))
            temporary=root/'last.building.pt';torch.save(payload,temporary);temporary.replace(root/'last.pt')
            if validation < best:
                best=validation
                temporary=root/'best.building.pt';torch.save(payload,temporary);temporary.replace(root/'best.pt')
            write_json(root/'history.json',history)
            print(f'EPOCH COMPLETE {mode} {record}',flush=True)
        best_state=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(best_state['model_state'],strict=True)
        write_json(root/'status.json',dict(state='evaluating',mode=mode,best_epoch=best_state['epoch'],
                    best_validation_loss=best,elapsed_seconds=time.monotonic()-started))
        extract(config,mode,model=model,trained=True)
        write_json(root/'status.json',dict(state='complete',mode=mode,best_epoch=best_state['epoch'],
                    best_validation_loss=best,epochs=config['epochs'],elapsed_seconds=time.monotonic()-started,
                    best_checkpoint_sha256=sha256(root/'best.pt')))
    except BaseException as error:
        write_json(root/'status.json',dict(state='failed',mode=mode,error=repr(error),
                    completed_epochs=len(history),elapsed_seconds=time.monotonic()-started))
        raise
