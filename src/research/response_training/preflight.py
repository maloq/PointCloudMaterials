"""Local numerical gates for the live student geometry and fixed response oracle."""
import time
import numpy as np
import torch

from src.research.response_atlas.atomistic import (configuration, gate, MACEPotential,
    PathFeatures, basis_for, oracle_for)
from .common import root, oracle_config, bind, write_json
from .data import prepare
from .model import Predictor, initialize, responses


def run(c):
    torch.set_num_threads(2)
    record=bind(c);states=prepare(c);o=oracle_config(c)
    physical=gate(o,'cuda')
    # The simulator calculator sets torch's default dtype; the student contract
    # explicitly returns to float32 before constructing its layers.
    torch.set_default_dtype(torch.float32);torch.manual_seed(c['fit_seeds'][0])
    q=torch.stack([s['q'] for s in states if s['role']=='train']).float().cuda()
    model=Predictor(c,states[0]['box']).float().cuda()
    initialize(model,q,c['training']['microbatch'])
    micro=c['training']['microbatch'];x=q[:micro]
    basis=torch.stack([basis_for(v.flatten(),2,20261201+i).reshape(256,3,2) for i,v in enumerate(x)])
    began=time.monotonic();torch.cuda.reset_peak_memory_stats()
    h=responses(model,x,basis,create_graph=True)
    loss=(h-.5*h.detach()).square().mean();loss.backward();torch.cuda.synchronize()
    gradient=float(torch.nn.utils.clip_grad_norm_(model.encoder.parameters(),10.,error_if_nonfinite=True))
    if gradient<=0 or not torch.isfinite(h).all() or float(h.detach().norm())<=0:
        raise ValueError('Response loss has no finite input/encoder gradient')
    seconds=time.monotonic()-began;memory=torch.cuda.max_memory_allocated()/2**30
    errors=[]
    for eps in (.03,.01):
        with torch.no_grad():fd=(model(x+eps*basis[...,0])-model(x-eps*basis[...,0]))/(2*eps)
        errors.append(dict(epsilon_A=eps,relative_error=float((fd-h[:,:,0].detach()).norm()/h[:,:,0].detach().norm())))
    if min(e['relative_error'] for e in errors)>.015:raise ValueError(f'Student JVP/FD mismatch: {errors}')
    with torch.no_grad():
        reference=model(x)
        translated=x+torch.tensor([.32,.18,.71],device=x.device)
        imaged=x.clone();imaged[:,17,0]+=model.box[0]
        permutation=torch.randperm(256,device=x.device)
        invariance=dict(translation=float((model(translated)-reference).abs().max()),
            periodic_image=float((model(imaged)-reference).abs().max()),
            permutation=float((model(x[:,permutation])-reference).abs().max()))
    if max(invariance.values())>2e-5:raise ValueError(f'Full-cell invariance failure: {invariance}')
    del model,h,loss
    torch.cuda.empty_cache()
    # A physical-horizon common-random-number check, plus actual collection timings.
    atoms=configuration(o,0);point=torch.tensor(atoms.positions.flatten(),dtype=torch.float64,device='cuda')
    potential=MACEPotential(o,atoms,'cuda')
    features=PathFeatures(atoms.cell.lengths(),c['horizons_steps'],o,'cuda')
    directions=basis_for(point,2,20261203)
    oracle=oracle_for(o,potential,features,point,c['horizons_steps'])
    start=time.monotonic();ad=oracle.query(point,directions,[69000000]);torch.cuda.synchronize();ad_seconds=time.monotonic()-start
    start=time.monotonic();value=oracle.query(point,point.new_empty(len(point),0),[69000000]);torch.cuda.synchronize();value_seconds=time.monotonic()-start
    if float((ad.values-value.values).abs().max())>1e-10:raise ValueError('AD changes the forward trajectory')
    eps=.003
    minus=oracle.query(point-eps*directions[:,0],point.new_empty(len(point),0),[69000000])
    plus=oracle.query(point+eps*directions[:,0],point.new_empty(len(point),0),[69000000])
    fd=(plus.values-minus.values)/(2*eps)
    physical_error=float((fd-ad.responses[:,:,0]).norm()/ad.responses[:,:,0].norm())
    if physical_error>.01:raise ValueError(f'100fs coupled FD/AD mismatch: {physical_error}')
    result=dict(state='complete',identity=record['identity'],physical_short_gate=physical,
        student_jvp_fd=errors,response_only_encoder_gradient=gradient,student_invariance=invariance,
        student_microbatch=micro,student_response_backward_seconds=seconds,student_peak_GiB=memory,
        physical_100fs_fd_relative_error=physical_error,value_branch_seconds=value_seconds,
        response_branch_seconds=ad_seconds,device=torch.cuda.get_device_name(),
        approximate_collection_hours=(56*32*value_seconds+48*8*ad_seconds)/3600,
        note='Local numerical checks only; no online training runs')
    write_json(root(c)/'technical/preflight.json',result)
    print(result,flush=True)
    return result
