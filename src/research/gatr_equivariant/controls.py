"""Diagnostic interventions on the frozen model, never used as physical data."""
import argparse
import json
from pathlib import Path
import socket

import numpy as np
import torch
from src.data.structural_pretraining.prepare import save_json
from .model import Capture, VECTOR_INDICES


@torch.no_grad()
def controls(config):
    if socket.gethostname().split('.')[0] != config['required_hostname']:
        raise RuntimeError('Controls must run on the requested node07')
    if config['required_gpu'] not in torch.cuda.get_device_name(0):
        raise RuntimeError('Controls require the requested A100')
    torch.set_num_threads(2)
    root = Path(config['output'])/'technical'
    parent = json.loads((Path(config['parent_audit'])/'technical/plan.json').read_text())
    model = Capture(root/'encoder.pt',config['checkpoint_sha256']).cuda().eval()
    positions, centers, rows = [], [], []
    # First test source per temperature; early, middle and late observations.
    test = [s for s in parent['sources'] if s['split']=='test'][::2]
    for source in test:
        folder = Path(config['parent_audit'])/'technical/sources'/str(source['id'])
        a = dict(np.load(folder/'observations.npz'))
        x = np.load(folder/'positions.npy',mmap_mode='r')
        for frame in (0,400,800):
            for c in range(len(a['centers'])):
                row = frame*len(a['centers'])+c
                lo,hi = a['offsets'][row:row+2]
                positions.append(x[lo:hi]); centers.append(a['center_indices'][row])
                rows.append(dict(source=source['id'],frame=frame,atom=int(a['centers'][c])))
    batch = model.make_batch(positions,centers)
    results = []
    random = torch.Generator(device='cuda').manual_seed(config['seed'])
    x = batch['positions']
    random_unit = torch.randn(x.shape,generator=random,device=x.device,dtype=x.dtype)
    random_unit /= random_unit.norm(dim=-1,keepdim=True)
    shuffled = dict(batch,positions=x.norm(dim=-1,keepdim=True)*random_unit)
    scaled = dict(batch,positions=1.01*x)
    for precision in ('bf16','float32'):
        model.precision = precision
        z,_ = model.evaluate(batch)
        def record(name,changed):
            delta = changed.double()-z.double()
            results.append(dict(precision=precision,intervention=name,
                rms=float(delta.square().mean().sqrt()),max_abs=float(delta.abs().max()),
                mean_per_state_l2=float(delta.norm(dim=-1).mean())))
        record('independent_atom_angle_scramble_fixed_radii',model.evaluate(shuffled)[0])
        record('radii_times_1p01_fixed_support_weights',model.evaluate(scaled)[0])
        for layer in ('input','block1','block2_mlp_input','block2_output'):
            if layer=='block2_mlp_input':
                def before(module,args,kwargs):
                    mv = args[0].clone(); mv[...,list(VECTOR_INDICES)] = 0
                    return (mv,*args[1:]),kwargs
                handle = model.encoder.spatial[1].mlp.register_forward_pre_hook(before,with_kwargs=True)
            else:
                module = model.encoder.input if layer=='input' else model.encoder.spatial[0 if layer=='block1' else 1]
                def after(module,args,output):
                    mv = output[0].clone(); mv[...,list(VECTOR_INDICES)] = 0
                    return mv,output[1]
                handle = module.register_forward_hook(after)
            changed,_ = model.evaluate(batch); handle.remove()
            record(f'erase_all_token_vector_triplets_after_{layer}',changed)
    save_json(root/'interventions.json',dict(rows=rows,results=results,
        description='60 observations, five held-out sources, frames 0/400/800. Synthetic, out-of-distribution mechanistic controls; not plausible atomic trajectories. Scalar metadata and support weights retained.'))
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config',required=True)
    args = parser.parse_args(); controls(json.loads(Path(args.config).read_text()))
