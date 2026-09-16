"""Frozen-backbone local-motion fits with atomic, exact-epoch resumption."""
from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import portable_config
from src.research.mace_velocity.inventory import read
from .motion import MotionState, batch_arrays, objective, specifications
from .motion_data import check_deadline, load
from .smooth import FAMILIES, within_covariance


def identity(config):
    scientific = {k:v for k,v in config.items() if k not in ('runtime','devices','cpu_threads','label_workers_per_lane')}
    files = ['motion.py','motion_data.py','motion_train.py','motion_evaluate.py']
    return dict(config=portable_config(scientific),
        code={f:sha256(Path(__file__).with_name(f)) for f in files},
        data_sha256=sha256(Path(config['cache'])/'sequences.npz'),
        checkpoint_sha256=sha256(Path(config['checkpoint'])))


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.building.pt')
    torch.save(payload,temporary);temporary.replace(path)


def fit(config, root, lane):
    torch.set_num_threads(config['cpu_threads']);device = config['devices'][lane]
    data = load(config)
    train = batch_arrays(data,0,config,device);val = batch_arrays(data,1,config,device)
    expected = identity(config)
    initial = torch.load(config['checkpoint'],map_location='cpu',weights_only=False)
    specs = list(specifications(config))[lane::len(config['devices'])]
    for number,spec in enumerate(specs):
        check_deadline(config)
        directory = root/'technical'/spec['name'];directory.mkdir(exist_ok=True)
        last = directory/'last.pt';best_path = directory/'best.pt'
        torch.manual_seed(spec['seed'])
        model = MotionState(spec['dimension'],spec['rank']).to(device)
        with torch.no_grad():
            if spec['dimension']:
                raw = model.mapping(train['x']).reshape(-1,spec['dimension'])
                scale = within_covariance(raw,train['weights'],train['context']).diag().sqrt()
                if torch.any(scale<=1e-8): raise FloatingPointError('Degenerate initial state map')
                model.initial_scale.copy_(scale)
            else:
                state = {k.removeprefix('structure.'):v for k,v in initial['head_state'].items() if k.startswith('structure.')}
                model.readout.load_state_dict(state,strict=True)
        optimizer = torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
        epoch_start = 0;best = float('inf');history = [];started = time.monotonic()
        if last.exists():
            saved = torch.load(last,map_location=device,weights_only=False)
            if saved['identity']!=expected or saved['spec']!=spec: raise ValueError(f'Changed local-motion resume: {last}')
            model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
            epoch_start = saved['epoch']+1;best = saved['best'];history = saved['history']
        if epoch_start>config['epochs']: continue
        for epoch in range(epoch_start,config['epochs']+1):
            if epoch:
                model.train();optimizer.zero_grad(set_to_none=True)
                ramp = min(1.,max(0.,(epoch-config['warmup_epochs'])/config['ramp_epochs']))
                loss,metrics,_ = objective(model,train,spec,config,ramp=ramp)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(model.mapping.parameters())+list(model.readout.parameters()),5.,error_if_nonfinite=True)
                torch.nn.utils.clip_grad_norm_(model.directions.parameters(),5.,error_if_nonfinite=True)
                optimizer.step()
            if epoch%config['validation_every']==0 or epoch==config['epochs']:
                model.eval()
                with torch.no_grad():
                    _,metrics,_ = objective(model,val,spec,config,training=False)
                physical = metrics['physical'].cpu().numpy()
                score = float(physical.mean())+sum(spec[k]*float(metrics[k]) for k in ('temporal','direction','curvature'))
                row = dict(epoch=epoch,physical={k:float(v) for k,v in zip(FAMILIES,physical,strict=True)},
                    **{k:float(v) for k,v in metrics.items() if k!='physical'},selection_score=score)
                history.append(row)
                payload = dict(protocol=config['protocol'],identity=expected,spec=spec,epoch=epoch,
                    model=model.state_dict(),optimizer=optimizer.state_dict(),history=history,best=min(score,best))
                if score<best: save_checkpoint(best_path,payload);best=score
                save_checkpoint(last,payload);write_json(directory/'history.json',history)
                status = dict(state='training',lane=lane,variant=spec['name'],variant_number=number+1,
                    total_variants=len(specs),epoch=epoch,epochs=config['epochs'],best_validation=best,
                    elapsed_seconds=time.monotonic()-started)
                write_json(root/f'technical/fit-lane{lane}.json',status)
                if epoch==0 or epoch%100==0: print('MOTION FIT',status,flush=True)
                check_deadline(config)
        write_json(directory/'status.json',dict(state='complete',completed_epochs=config['epochs']))
    write_json(root/f'technical/fit-lane{lane}.json',dict(state='complete',variants=len(specs)))


def load_model(config, root, spec, device):
    checkpoint = torch.load(root/'technical'/spec['name']/'best.pt',map_location=device,weights_only=False)
    if checkpoint['identity'] != identity(config): raise ValueError(f'Changed evaluation identity: {spec["name"]}')
    model = MotionState(spec['dimension'],spec['rank']).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    return model,checkpoint['epoch']
