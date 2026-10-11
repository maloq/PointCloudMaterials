"""Local numerical audit of the new eSEN derivative path; no scientific updates."""
import time
import torch
from .data import read,root,Source,write_json
from .model import Motion
from . import simulator

def run(c,p):
    torch.set_num_threads(2);torch.manual_seed(c['seed']);model=Motion(c).cuda().eval()
    item=p['sources'][0];job=p['jobs'][0]
    x=Source(item['source']).patches(job['frame'],item['centers'][:2])[0]*c['length_factor'];x=torch.from_numpy(x).float().cuda()
    q=x[...,:3];d=x[...,3:]*model.time_scale
    started=time.monotonic()
    value,exact=torch.autograd.functional.jvp(model.geometry,q,d,create_graph=False,strict=True)
    with torch.no_grad():
        eps=model.velocity_epsilon;secant=(model.geometry(q+eps*d)-model.geometry(q-eps*d))/(2*eps)
    error=float((secant-exact).norm()/exact.norm().clamp_min(1e-10))
    if not torch.isfinite(exact).all() or error>.1:raise ValueError(f'eSEN velocity secant vs exact JVP discrepancy: {error}')
    model.geometry.zero_grad(set_to_none=True)
    model.encode(x,create_graph=True).square().mean().backward()
    grads=[v.grad for v in model.geometry.parameters() if v.grad is not None]
    if not grads or not all(torch.isfinite(v).all() for v in grads):raise FloatingPointError('Nonfinite geometry parameter gradients')
    result=dict(velocity_secant_relative_l2=error,exact_motion_rms=float(exact.square().mean().sqrt()),gradient_norm=float(torch.cat([g.flatten() for g in grads]).norm()),seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name(),geometry_precision='float32',student_velocity_epsilon=eps)
    write_json(root(c)/'technical/numerical-probe.json',result);print(result,flush=True)
    del model,x,value,exact,secant,grads;torch.cuda.empty_cache()
    receipt=simulator.collect(c,p,job,time.time()+600)
    print('physical teacher',receipt,flush=True)

if __name__=='__main__':
    import sys
    c=read(sys.argv[1]);run(c,read(root(c)/'technical/plan.json'))
