"""Local real-data precision, higher-gradient and resource measurements (no W&B)."""
import time
from pathlib import Path
import numpy as np
import torch
from .data import root,read,resolve_path,write_json,spatial_data,digest,sha
from .model import Spatial,Motion,spatial_backward,spatial_loss,views,fixed_geometry,reference_contractions,geometry_jvp
from .train import short_objective,spatial_start


def run(c,spatial_only=False):
    if 'spatial_initialization' in c and not spatial_only:return motion_probe(c)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(c['seed']);data,columns,weights=spatial_data(c)
    b=data.batch(np.arange(256));data.close()
    q=torch.from_numpy(b['positions']).cuda();target=torch.from_numpy(b['target'][:,columns]).cuda();weight=torch.from_numpy(weights).cuda()
    model=Spatial(c).cuda();ema=fixed_geometry(c,model.geometry)
    seen=[];hook=model.geometry.products[0].register_forward_hook(lambda module,args,out:seen.append(str(out.dtype)))
    with torch.no_grad():
        f32=model.geometry(q[:16])
        with torch.autocast('cuda',dtype=torch.bfloat16):amp=model.geometry(q[:16])
        rotation=torch.linalg.qr(torch.randn(3,3,device='cuda')).Q
        rotated=model.geometry(q[:16]@rotation)
    hook.remove();error=float((amp-f32).square().mean().sqrt()/f32.square().mean().sqrt().clamp_min(1e-6))
    rotation_error=float((rotated-f32).abs().max())
    if error>.1 or rotation_error>.005:raise ValueError(f'Precision or invariance error: {error}, {rotation_error}')
    records=[]
    for method in c['methods']:
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        timings=[];torch.cuda.reset_peak_memory_stats()
        for step in range(3):
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();begin=time.monotonic()
            loss,stats=spatial_backward(model,ema,q,target,weight,method,c['seed']+step,256)
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True));optimizer.step();torch.cuda.synchronize()
            timings.append(time.monotonic()-begin)
        records.append(dict(method=method,loss=loss,gradient_norm=norm,seconds_per_update=timings,peak_memory_GB=torch.cuda.max_memory_allocated()/2**30,**stats))
        print(records[-1],flush=True)
    model.zero_grad(set_to_none=True)
    spatial_backward(model,ema,q[:8],target[:8],weight,'masked_teacher',771,4)
    cached={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        z=[model.geometry(x,keep) for x,keep in views(q[:8],'masked_teacher',771)]
        with torch.no_grad():teacher=ema(q[:8])
    loss,_=spatial_loss(model,z,target[:8],weight,'masked_teacher',teacher);loss.backward()
    numerator=sum((p.grad-cached[k]).square().sum() for k,p in model.named_parameters() if k in cached)
    denominator=sum(v.square().sum() for v in cached.values())
    gradient_cache_relative=float((numerator/denominator).sqrt())
    if gradient_cache_relative>.05:raise ValueError(f'Cached batch gradients disagree: {gradient_cache_relative}')
    restored=fixed_geometry(c,model.geometry)
    if any(not torch.equal(v,restored.state_dict()[k]) for k,v in model.geometry.state_dict().items()):raise ValueError('Spatial checkpoint tensors changed')
    with torch.no_grad():restore_error=float((model.geometry(q[:2])-restored(q[:2])).abs().max())
    if restore_error>1e-4:raise ValueError(f'Spatial checkpoint restoration changed geometry: {restore_error}')
    spatial_result=dict(passed=True,encoder=c['encoder'],device=torch.cuda.get_device_name(),amp_relative_rms=error,rotation_max_abs=rotation_error,gradient_cache_relative=gradient_cache_relative,restore_max_abs=restore_error,message_dtypes=sorted(set(seen)),spatial=records,online_runs_created=0)
    write_json(root(c)/'technical/spatial-probe.json',spatial_result);print('spatial checks passed',spatial_result,flush=True)
    if spatial_only:return spatial_result
    del restored
    del model,ema,optimizer,z,loss,cached;torch.cuda.empty_cache()
    # Actual saved full-cell MEAM response and tangents, not synthetic derivative targets.
    campaign=resolve_path(c['response_campaign']);plan=read(campaign/'technical/plan.json');old=plan['config']
    job=next(j for j in plan['jobs'] if next(s for s in plan['sources'] if s['key']==j['source_key'])['role']=='train')
    record=torch.load(resolve_path(old['simulation_archive'])/job['key']/'response.pt',weights_only=False)
    motion=Motion(c).cuda();anchor=fixed_geometry(c,motion.geometry)
    slot=record['horizons_fs'].index(100);future=record['future'][:,slot,:,:3].float().cuda();responses=[]
    def teacher_geometry(q):
        with reference_contractions(anchor):return anchor(q)
    for k in range(2):
        _,j=torch.autograd.functional.jvp(teacher_geometry,future,record['tangent'][:,slot,:,:3,k].float().cuda(),create_graph=False,strict=True);responses.append(j.mean(0,keepdim=True).detach())
    with torch.no_grad():end=anchor(future).mean(0,keepdim=True)
    r=dict(state=record['state'][None].float(),basis=record['basis'][None].float(),target=end,response=torch.stack(responses,-1))
    with torch.no_grad():fast=motion.geometry(r['state'][...,:3].cuda())
    with reference_contractions(motion.geometry),torch.no_grad():reference=motion.geometry(r['state'][...,:3].cuda())
    reference_error=float((reference-fast).abs().max())
    if reference_error>.0001:raise ValueError(f'Reference contraction mismatch: {reference_error}')
    start=time.monotonic();v,d=short_objective(motion,r,c,dict(frozen=False,derivative=True),91)
    (v+.1*d).backward();norm=float(torch.nn.utils.clip_grad_norm_(motion.parameters(),10.,error_if_nonfinite=True));torch.cuda.synchronize()
    derivative=dict(endpoint=float(v.detach()),response=float(d.detach()),gradient_norm=norm,seconds=time.monotonic()-start)
    print('short derivative',derivative,'reference maximum error',reference_error,flush=True)
    if not np.isfinite(list(derivative.values())).all():raise FloatingPointError('Invalid higher-order derivative training')
    motion.zero_grad(set_to_none=True);del v,d
    observed=torch.load(resolve_path(old['cache'])/'dataset.pt',weights_only=False)
    torch.cuda.reset_peak_memory_stats();start=time.monotonic()
    from src.research.continuous_motion.model import mixture_nll
    total=0.;micro=c['response_training']['microbatch']
    for offset in range(0,256,micro):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            z=motion.encode(observed['state'][offset:offset+micro].cuda(),create_graph=True)
            path=motion.path(z,c['horizons_ps'],particles=4,seed=72,random_batch_size=256,sample_offset=offset)
            mu,lv=motion.prediction(path)
        target_y=((observed['target'][offset:offset+micro]-observed['mean'])/observed['scale']).float().cuda()
        objective=(mixture_nll(mu.float(),lv.float(),target_y).mean()+.1*path[...,:128].square().mean())*micro/256
        objective.backward();total+=float(objective.detach())
    norm=float(torch.nn.utils.clip_grad_norm_(motion.parameters(),10.,error_if_nonfinite=True));torch.cuda.synchronize()
    long=dict(loss=total,microbatch=micro,gradient_norm=norm,seconds=time.monotonic()-start,peak_memory_GB=torch.cuda.max_memory_allocated()/2**30)
    print('long motion',long,flush=True)
    # Exercise checkpoint restoration after cuEq has initialized its kernels.
    state=motion.geometry.state_dict();restored=type(motion.geometry)(c).cuda();restored.load_state_dict(state)
    with torch.no_grad():
        restore_error=float((motion.geometry(q[:2])-restored(q[:2])).abs().max())
    if restore_error>1e-4:raise ValueError(f'Checkpoint restoration changed geometry: {restore_error}')
    result=dict(passed=True,encoder=c['encoder'],device=torch.cuda.get_device_name(),amp_relative_rms=error,rotation_max_abs=rotation_error,gradient_cache_relative=gradient_cache_relative,restore_max_abs=restore_error,reference_max_abs=reference_error,message_dtypes=sorted(set(seen)),spatial=records,derivative=derivative,long=long,precision='BF16 neural autocast; float32 geometry/normalization/losses and nested derivative island; float32 parameters/optimizer',online_runs_created=0)
    write_json(root(c)/'technical/probe.json',result);print(result,flush=True)


def motion_probe(c):
    """Audit the repaired derivative route on completed spatial checkpoints."""
    from src.research.continuous_motion.model import mixture_nll
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(c['seed']);model=Motion(c).cuda()
    plan=read(resolve_path(c['response_campaign'])/'technical/plan.json');old=plan['config']
    observed=torch.load(resolve_path(old['cache'])/'dataset.pt',weights_only=False)
    job=next(j for j in plan['jobs'] if next(s for s in plan['sources'] if s['key']==j['source_key'])['role']=='train')
    record=torch.load(resolve_path(old['simulation_archive'])/job['key']/'response.pt',weights_only=False)
    # Check the densest graphs: peak memory depends on edge count, not row order.
    edges=[]
    with torch.no_grad():
        for q in observed['state'][...,:3].split(128):
            q=q.cuda();q=q-q[:,:1];valid=q.norm(dim=-1)<8
            edges.append((((q[:,None]-q[:,:,None]).norm(dim=-1)<5)&valid[:,:,None]&valid[:,None,:]).sum((1,2)).cpu())
    ids=torch.cat(edges).argsort(descending=True)[:16];micro=c['response_training']['microbatch'];records=[]
    for method in c['methods']:
        saved=torch.load(spatial_start(c,method)/'best.pt',weights_only=False,map_location='cpu')
        model.geometry.load_state_dict(saved['geometry']);model.zero_grad(set_to_none=True)
        x=record['state'][None].float().cuda()
        with reference_contractions(model.geometry):
            old_s,old_m=torch.autograd.functional.jvp(model.geometry,x[...,:3],x[...,3:]*.1,strict=True)
        fresh=model.encode(x,create_graph=False);error=float((fresh-torch.cat((old_s,old_m),-1)).abs().max())
        if error>1e-4:raise ValueError(f'Geometry/motion JVP disagreement: {error}')
        with reference_contractions(model.geometry),torch.no_grad():
            eps=.001;direction=x[...,3:]*.1
            difference=(model.geometry(x[...,:3]+eps*direction)-model.geometry(x[...,:3]-eps*direction))/(2*eps)
        fd_error=float((difference-old_m).square().mean().sqrt()/old_m.square().mean().sqrt().clamp_min(1e-6))
        if fd_error>.02:raise ValueError(f'Geometry JVP fails local finite-difference audit: {fd_error}')
        slot=record['horizons_fs'].index(100);future=record['future'][:,slot,:,:3].float().cuda();responses=[]
        for k in range(2):
            _,j=geometry_jvp(model.geometry,future,record['tangent'][:,slot,:,:3,k].float().cuda(),create_graph=False)
            responses.append(j.mean(0,keepdim=True).cpu())
        with torch.no_grad():end=model.geometry(future).mean(0,keepdim=True).cpu()
        r=dict(state=x.cpu(),basis=record['basis'][None].float(),target=end,response=torch.stack(responses,-1))
        start=time.monotonic();torch.cuda.reset_peak_memory_stats()
        v,d=short_objective(model,r,c,dict(frozen=False,derivative=True),91)
        (.1*v+.1*d).backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True));torch.cuda.synchronize()
        short=dict(endpoint=float(v.detach()),response=float(d.detach()),gradient_norm=norm,seconds=time.monotonic()-start,peak_GB=torch.cuda.max_memory_allocated()/2**30)
        if not np.isfinite(list(short.values())).all():raise FloatingPointError('Nonfinite repaired response gradients')
        print(method,'short derivative',short,flush=True)
        del v,d;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats();start=time.monotonic();total=0.
        for start_row in range(0,len(ids),micro):
            block=ids[start_row:start_row+micro]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=model.encode(observed['state'][block].cuda(),create_graph=True)
                path=model.path(z,c['horizons_ps'],particles=4,seed=72,random_batch_size=16,sample_offset=start_row)
                mu,lv=model.prediction(path)
            y=((observed['target'][block]-observed['mean'])/observed['scale']).float().cuda()
            loss=(mixture_nll(mu.float(),lv.float(),y).mean()+.1*path[...,:128].square().mean())*len(block)/len(ids)
            loss.backward();total+=float(loss.detach())
            del z,path,mu,lv,loss
        norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True));torch.cuda.synchronize()
        long=dict(loss=total,gradient_norm=norm,seconds=time.monotonic()-start,peak_GB=torch.cuda.max_memory_allocated()/2**30,microbatch=micro,densest_rows=ids.tolist())
        print(method,'long backward',long,flush=True)
        records.append(dict(method=method,jvp_max_abs=error,finite_difference_relative_rms=fd_error,short=short,long=long));model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    result=dict(passed=True,encoder=c['encoder'],config_identity=digest(c),model_sha256=sha(Path(__file__).with_name('model.py')),device=torch.cuda.get_device_name(),methods=records,online_runs_created=0)
    write_json(root(c)/'technical/probe.json',result);print(result,flush=True);return result
