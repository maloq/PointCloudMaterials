"""Fixed-budget trajectory fits, exact JVPs and checkpoint recovery."""
import math
import os
import time
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from .data import root, read, load, normalization, digest, write_json, atomic_save, sha, FAMILY
from .model import PhaseProcess


def objective(model, initial, target, direction, response, norm, arm, c, *, prediction=None):
    length=target.shape[1]
    if prediction is None: prediction=model(initial,length)
    residual=(prediction-target).reshape(len(initial),length,80,2,3)
    scaled=residual/norm['path_scale'][:length][None,:,None,:,None]
    scores={'path':scaled.square().mean()}
    # Coupled Q/V check in normalized units (V scale / Q scale = 2 / ps).
    joined=torch.cat((initial[:,None],prediction),1)
    dq=joined[:,1:,:,:3]-joined[:,:-1,:,:3]
    trapezoid=.005*(joined[:,1:,:,3:]+joined[:,:-1,:,3:])
    scores['kinematic']=((dq-trapezoid)/.002).square().mean()
    if model.latent:
        true=torch.cat((initial[:,None],target),1).flatten(0,1)
        codes=model.encode(true)
        decoded=model.decode(codes).flatten(-2)
        scores['reconstruction']=(decoded-true).square().mean()
        next_codes=model.step(model.encode(initial))
        encoded_next=model.encode(target[:,0])
        scores['latent_value']=(next_codes-encoded_next).square().mean()
    if arm['response']:
        _, dp=torch.func.jvp(lambda x:model(x,length),(initial,),(direction,))
        error=(dp-response).reshape(len(initial),length,80,2,3)
        # Normalizer is selected per sampled direction by the caller.
        scores['physical_response']=(error.square()/norm['selected_power'][None,:,None,:,None]).mean()
        if arm['latent_response']:
            _, predicted=torch.func.jvp(lambda x:model.step(model.encode(x)),(initial,),(direction,))
            _, expected=torch.func.jvp(model.encode,(target[:,0],),(response[:,0],))
            power=expected.detach().square().mean().clamp_min(1e-6)
            scores['latent_response']=(predicted-expected).square().mean()/power
    weights=c['training']['loss_weights']
    loss=sum(weights[name]*value for name,value in scores.items())
    return loss,scores


def batch(state,tangent,ids,starts,length,k):
    rows=torch.as_tensor(ids,device=state.device)
    starts=torch.as_tensor(starts,device=state.device)
    indices=starts[:,None]+torch.arange(1,length+1,device=state.device)[None]
    return (state[rows,starts],state[rows[:,None],indices],
            tangent[rows,starts,...,k],tangent[rows[:,None],indices,...,k])


def fit(c,arm,deadline):
    folder=root(c)/'analyses'/arm['name']/'technical'; folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists(): return True
    p,records,state,tangent=load(c,'train'); norm=normalization(state,tangent)
    settings=c['training']; count=settings['updates']; n=len(state)
    torch.manual_seed(arm['seed']); model=PhaseProcess(c,arm).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=1e-5)
    rng=np.random.default_rng(arm['seed']); identity=digest([p['identity'],arm]); history=[]; start=0
    checkpoint=folder/'last.pt'
    if checkpoint.exists():
        previous=torch.load(checkpoint,map_location='cuda',weights_only=False)
        if previous['identity']!=identity: raise ValueError('Changed resumed phase-space fit')
        model.load_state_dict(previous['model']); optimizer.load_state_dict(previous['optimizer'])
        rng.bit_generator.state=previous['rng']; start=previous['update']; history=previous['history']
    def save(update,final=False):
        payload=dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=rng.bit_generator.state,history=history,update=update,arm=arm,
            normalization={key:value.detach().cpu() for key,value in norm.items()})
        atomic_save(folder/('final.pt' if final else 'last.pt'),payload)
    write_json(folder/'execution-segments'/f'{time.time_ns()}.json',dict(
        allocation=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),start_update=start,
        parameters=sum(v.numel() for v in model.parameters()),train_states=n,
        train_ancestors=len({r['ancestor'] for r in records}),precision='FP32 neural state, JVP and losses',
        checkpoint_rule='Final fixed update only, no test selection'))
    with online_training(c['wandb'],run_id='phase-'+identity[:16],name=arm['name'],config=dict(c,arm=arm),
            folder=folder,receipt_path=folder/'wandb.json',job_type='predictor',group=c['protocol'],
            receipt_fields=dict(identity=identity)) as tracking:
        for update in range(start+1,count+1):
            if time.time()>deadline-360: save(update-1); return False
            begin=time.monotonic(); optimizer.zero_grad(set_to_none=True)
            warm=min(update/settings['warmup'],1.)
            cosine=.01+.99*.5*(1+math.cos(math.pi*max(0,update-settings['warmup'])/(count-settings['warmup'])))
            for group in optimizer.param_groups: group['lr']=settings['lr']*warm*cosine
            length=int(rng.choice(settings['unroll_steps'],p=settings['unroll_probabilities']))
            k=int(rng.integers(2)); norm['selected_power']=norm['response_power'][:length,:,k]
            size=settings['batch_size']; ids=rng.integers(n,size=size); starts=rng.integers(21-length,size=size)
            totals={}
            for offset in range(0,size,settings['microbatch']):
                end=offset+settings['microbatch']; weight=len(ids[offset:end])/size
                loss,scores=objective(model,*batch(state,tangent,ids[offset:end],starts[offset:end],length,k),norm,arm,c)
                (weight*loss).backward()
                for key,value in scores.items(): totals[key]=totals.get(key,0.)+weight*float(value.detach())
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True)
            optimizer.step()
            row=dict(update=update,unroll_steps=length,query=k,loss=sum(settings['loss_weights'][k]*v for k,v in totals.items()),
                     gradient_norm=float(grad),**{name:totals.get(name,0.) for name in settings['loss_weights']})
            history.append(row); tracking.log(row,step=update)
            if update==start+1 or update%settings['save_every']==0: save(update)
            write_json(folder/'progress.json',dict(state='training',**row,total=count,seconds_per_update=time.monotonic()-begin))
            if update==start+1 or update%50==0: print(arm['name'],row,flush=True)
        tracking.summary.update(dict(final_update=count,checkpoint_rule='fixed_final'))
    save(count); save(count,final=True)
    write_metric_rows(history,folder.parent,family=FAMILY,name='learning')
    write_json(folder/'complete.json',dict(identity=identity,updates=count,checkpoint_sha256=sha(folder/'final.pt')))
    return True


def benchmark(c):
    """Local integration/performance exercise; no W&B run or scientific fit."""
    from .data import collect,archive
    p=read(root(c)/'technical/plan.json')
    jobs=[j for j in p['jobs'] if j['role']=='train'][:2]
    packets=[]
    for job in jobs:
        collect(c,p,job,time.time()+600)
        packets.append(torch.load(archive(c)/job['key']/'path.pt',weights_only=False))
    scale=torch.tensor([5.,5.,5.,10.,10.,10.],device='cuda')
    state=torch.stack([r['state'] for r in packets]).cuda().float()/scale
    tangent=torch.stack([r['tangent'] for r in packets]).cuda().float()/scale[:,None]
    norm=normalization(state,tangent); result=[]
    for arm in c['arms'][:4]:
        torch.manual_seed(arm['seed']); model=PhaseProcess(c,arm).cuda()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
        for length in [4,20]:
            elapsed=[]
            for repeat in range(3):
                optimizer.zero_grad(set_to_none=True); norm['selected_power']=norm['response_power'][:length,:,0]
                torch.cuda.synchronize(); begin=time.monotonic()
                size=c['training']['batch_size']; micro=c['training']['microbatch']
                for offset in range(0,size,micro):
                    n=min(micro,size-offset); args=batch(state,tangent,np.arange(n)%2,np.zeros(n,dtype=int),length,0)
                    loss,_=objective(model,*args,norm,arm,c); (loss*n/size).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True); optimizer.step()
                torch.cuda.synchronize(); elapsed.append(time.monotonic()-begin)
            row=dict(arm=arm['name'],steps=length,seconds=elapsed,loss=float(loss.detach()),gpu=torch.cuda.get_device_name(),
                     memory_GiB=torch.cuda.max_memory_allocated()/2**30)
            result.append(row); print(row,flush=True)
        # Rotation equivariance and exact-JVP/secant comparison on a real path.
        x=state[:1,0]; u=tangent[:1,0,...,0]
        rotation=torch.linalg.qr(torch.randn(3,3,device='cuda'))[0]
        rotated=(x.reshape(1,80,2,3)@rotation).flatten(-2)
        with torch.no_grad():
            a=model(rotated,4); b=(model(x,4).reshape(1,4,80,2,3)@rotation).flatten(-2)
        _,jvp=torch.func.jvp(lambda value:model(value,4),(x,),(u,))
        with torch.no_grad(): secant=(model(x+.1*u,4)-model(x-.1*u,4))/.2
        error=float((jvp-secant).norm()/jvp.norm().clamp_min(1e-8))
        eq=float((a-b).abs().max())
        if error>.02 or eq>1e-4: raise ValueError(f'Phase-model integration audit: secant {error}, rotation {eq}')
        result.append(dict(arm=arm['name'],jvp_secant_relative_error=error,equivariance_max_abs=eq))
    write_json(root(c)/'technical/benchmark.json',result)
