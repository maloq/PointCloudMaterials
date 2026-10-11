"""Local numerical and complete-update measurements for the secant campaign."""
import time
from pathlib import Path
import numpy as np
import torch
from .data import read,root,write_json,digest,sha
from .fast import FastMotion,short_objective,long_backward,cached_targets
from .model import geometry_jvp
from .train import spatial_start,load_motion_data


def run(c):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(c['seed']);model=FastMotion(c).cuda()
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('geometry.')}
    data,bank,future=load_motion_data(c);records=[];train=np.array([i for i,r in enumerate(data['rows']) if r['role']=='train'])
    # Actual full-sized training batch, with dense neighborhoods represented.
    indices=torch.from_numpy(np.random.default_rng(c['seed']).choice(train,256,replace=False))
    for method in c['methods']:
        saved=torch.load(spatial_start(c,method)/'best.pt',map_location='cpu',weights_only=False)
        state=model.state_dict();state.update(initial);model.load_state_dict(state);model.geometry.load_state_dict(saved['geometry'])
        model.set_arm(dict(frozen=False))
        p=root(c)/'technical'/('probe-'+method);p.mkdir(parents=True,exist_ok=True)
        target,_,short=cached_targets(c,method,model,None,bank,future,data['state'],False,c['response_training']['microbatch'],time.time()+7200,p)
        r=next(r for r in short if r['role']=='train');x=r['state'].cuda()
        samples=torch.cat((x,data['state'][indices[:3]].cuda()))
        errors=[];convergence=[]
        for sample in samples.split(1):
            s,m=geometry_jvp(model.geometry,sample[...,:3],sample[...,3:]*.1,create_graph=False)
            z=model.base_encode(sample,create_graph=False)
            errors.append(float((z[:,128:]-m).norm()/m.norm().clamp_min(1e-6)))
            model.velocity_epsilon*=.5;half=model.base_encode(sample,create_graph=False);model.velocity_epsilon*=2
            convergence.append(float((z[:,128:]-half[:,128:]).norm()/m.norm().clamp_min(1e-6)))
        if max(errors)>.03 or max(convergence)>.03:raise ValueError(f'Neural secant error: {method}, {errors}, {convergence}')
        with torch.no_grad():
            center=model.base_encode(x,create_graph=False).cpu();eps=model.response_epsilon
            pairs=torch.stack([torch.stack([model.base_encode(x+sign*eps*r['basis'][...,k].cuda(),create_graph=False).cpu() for sign in (-1,1)]) for k in range(2)])
        cached=dict(r,z=center,z_pairs=pairs)
        model.set_arm(dict(frozen=False))
        v,d=short_objective(model,r,c,dict(frozen=False,derivative=True),91,create_graph=False)
        model.set_arm(dict(frozen=True))
        av,ad=short_objective(model,cached,c,dict(frozen=True,derivative=True),91,create_graph=False)
        identity_error=max(abs(float(v-av)),abs(float(d-ad)))
        identity_code_error=float((model.adapt(center.cuda())-center.cuda()).abs().max())
        identity_response_relative=abs(float(d-ad))/max(1.,abs(float(d)))
        # Independently executed fused MACE scatters vary at float32 roundoff;
        # paired differences amplify it. Compare the adapter map exactly on
        # shared codes, and retain a separate relative response reproducibility
        # bound (the local repeated-forward audit measured about1e-4 relative).
        if identity_code_error>1e-7 or abs(float(v-av))>1e-4 or identity_response_relative>1e-3:
            raise ValueError(f'Adapter identity mismatch: codes={identity_code_error}, response={identity_response_relative}')
        model.set_arm(dict(frozen=False));zbase=torch.cat([model.base_encode(data['state'][ids].cuda(),create_graph=False).cpu() for ids in indices.split(c['response_training']['microbatch'])])
        # Verify nonlinear adapter's analytic JVP independently on a nonidentity map.
        with torch.no_grad():model.adapter[-1].weight.normal_(std=.001)
        model.set_arm(dict(frozen=True));z=zbase[:4].cuda();s,m=z.split(128,-1)
        _,exact=torch.autograd.functional.jvp(lambda q:q+model.adapter(q),s,m,strict=True)
        adapter_error=float((model.adapt(z)[:,128:]-exact).abs().max())
        if adapter_error>1e-4:raise ValueError('Incorrect adapter velocity derivative')
        # Restore the common identity adapter before timing matched updates.
        state=model.state_dict();state.update(initial);model.load_state_dict(state)
        for arm in c['response_arms']:
            model.set_arm(arm);optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
            local_data={k:(v[indices] if k in ('state','target') else v) for k,v in data.items()}
            timings=[]
            for repeat in range(2):
                model.zero_grad(set_to_none=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
                nll,latent=long_backward(model,local_data,target[indices],zbase if arm['frozen'] else None,torch.arange(256),c,c['response_training']['microbatch'],93+repeat)
                for k in range(c['response_training']['short_batch']):
                    value,tangent=short_objective(model,cached if arm['frozen'] else r,c,arm,91+k)
                    loss=(c['response_training']['short_weight']*value+c['response_training']['derivative_weight']*tangent)/c['response_training']['short_batch']
                    loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True));optimizer.step();torch.cuda.synchronize()
                seconds=time.monotonic()-start;peak=torch.cuda.max_memory_allocated()/2**30;timings.append(seconds)
                if not np.isfinite([nll,latent,norm,seconds]).all():raise FloatingPointError('Nonfinite full update')
                print(method,arm['name'],dict(repeat=repeat,seconds=seconds,peak_GiB=peak,nll=nll,gradient_norm=norm),flush=True)
                del value,tangent,loss
            records.append(dict(method=method,arm=arm['name'],seconds_per_update=timings,peak_GiB=peak,velocity_relative_errors=errors,
                velocity_step_halving_errors=convergence,identity_adapter_error=identity_error,
                identity_adapter_code_error=identity_code_error,identity_response_relative=identity_response_relative,adapter_jvp_max_abs=adapter_error))
            model.geometry.load_state_dict(saved['geometry']);state=model.state_dict();state.update(initial);model.load_state_dict(state)
            del optimizer;model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
        print(method,'audits passed',errors,flush=True)
    result=dict(passed=True,encoder=c['encoder'],config_identity=digest(c),model_sha256=sha(Path(__file__).with_name('model.py')),
        fast_sha256=sha(Path(__file__).with_name('fast.py')),training_sha256=sha(Path(__file__).with_name('train.py')),
        device=torch.cuda.get_device_name(),records=records,online_runs_created=0)
    write_json(root(c)/'technical/probe.json',result);print('passed',result,flush=True);return result
