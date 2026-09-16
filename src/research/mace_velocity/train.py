"""Joint backbone training with structural retention and local velocity readouts."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from src.experiment_runner.registry import sha256, write_json
from src.models.encoders.mace_context import context_features, make_context_graph
from src.models.encoders.mace_velocity import MACEVelocityEncoder
from src.project_runtime.paths import load_json
from src.research.mace_context.engine import load_model
from .data import load_cache
from .inventory import read


GROUPS={'bond_order':slice(0,16),'instantaneous_TDA_H0':slice(16,32),
        'instantaneous_TDA_H1':slice(32,96),'instantaneous_TDA_H2':slice(96,160),
        'motion_even':slice(160,166),'motion_odd':slice(166,169)}


def backbone(config,device):
    if sha256(Path(config['initial_checkpoint']))!=config['initial_checkpoint_sha256']:
        raise ValueError('Selected dual-physics initialization checkpoint changed')
    original=load_json(config['encoder_config'])
    original.update(device=device,cpu_threads=config['cpu_threads'],seed=config['seed'])
    model,_=load_model(original)
    state=torch.load(config['initial_checkpoint'],map_location='cpu',weights_only=False)
    if state['protocol']!='mace_context_recovery_joint_v1' or state['variant']!='dual_physics':
        raise ValueError('Expected selected jointly trained structure encoder')
    model.load_state_dict(state['model_state'],strict=True)
    return model.encoder.mace


def graph(config,clouds,device):
    g=make_context_graph([x for x,v in clouds],'halo_inner',device=device,
                         inner=config['inner_radius_A'],outer=config['outer_radius_A'])
    v=torch.as_tensor(np.concatenate([v for x,v in clouds]),device=device,dtype=torch.float32)
    return g,v


def encode(config,model,clouds,device,gradients=False):
    values=[]
    with torch.set_grad_enabled(gradients):
        for start in range(0,len(clouds),config['micro_batch_size']):
            g,v=graph(config,clouds[start:start+config['micro_batch_size']],device)
            values.append(model(g,v))
    result=torch.cat(values)
    if not torch.isfinite(result).all(): raise FloatingPointError('Nonfinite phase-space embedding')
    return result


def teacher(config,args):
    root=Path(config['output'])/'technical';clouds,y,meta=load_cache(config)
    target=root/'teacher.npz'
    if target.exists(): raise FileExistsError(f'Preserve teacher extraction: {target}')
    model=backbone(config,args.device);model.eval();features=[];start=time.monotonic()
    with torch.no_grad():
        for a in range(0,len(clouds),config['micro_batch_size']):
            g,_=graph(config,clouds[a:a+config['micro_batch_size']],args.device)
            features.append(context_features(model,g).cpu().numpy())
            if a%400==0:
                write_json(root/'teacher-status.json',dict(state='extracting',clouds=a,total=len(clouds),elapsed_seconds=time.monotonic()-start))
                print('TEACHER',a,len(clouds),flush=True)
    z=np.concatenate(features)
    pair_train=np.array([m['split']=='train' for m in meta]);train=np.repeat(pair_train,2)
    counts=Counter(m['balance_group'] for m in meta if m['split']=='train')
    weights=np.array([1/counts[m['balance_group']] if m['split']=='train' else 0 for m in meta])
    weights=np.repeat(weights,2);weights/=weights.sum()
    mean=weights@z;scale=np.sqrt(weights@((z-mean)**2));floor=.05*np.median(scale)
    if floor<=0: raise ValueError('Degenerate structural feature normalization')
    scale=np.maximum(scale,floor)
    ymean=weights@y;yscale=np.sqrt(weights@((y-ymean)**2))
    for section in (slice(16,32),slice(32,96),slice(96,160)):
        yscale[section]=np.sqrt(np.mean(yscale[section]**2))
    # Time-odd labels are centered on exactly zero to preserve reversal parity.
    ymean[166:]=0;yscale[166:]=np.sqrt(weights@(y[:,166:]**2))
    if np.any(yscale<=0) or not np.isfinite(z).all():
        raise ValueError('Degenerate target scales or nonfinite teacher features')
    np.savez(target,z=z,feature_mean=mean.astype(np.float32),feature_scale=scale.astype(np.float32),
             target_mean=ymean.astype(np.float32),target_scale=yscale.astype(np.float32))
    write_json(root/'teacher-status.json',dict(state='complete',clouds=len(z),elapsed_seconds=time.monotonic()-start,
        sha256=sha256(target),cache_manifest_sha256=sha256(Path(config['cache'])/'manifest.json'),
        initial_checkpoint_sha256=config['initial_checkpoint_sha256']))


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        self.structure=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,160))
        self.activity=nn.Sequential(nn.Linear(288,64),nn.SiLU(),nn.Linear(64,6))
        self.flow=nn.Linear(16,3,bias=False)

    def forward(self,z):
        return torch.cat((self.structure(z[:,:256]),self.activity(z[:,:288]),self.flow(z[:,288:])),1)


def loss(config,heads,z,target,old,weights,lag):
    predicted=heads(z)
    error=(predicted-target).square()
    group={name:(error[:,section].mean(1)*weights).mean() for name,section in GROUPS.items()}
    structural=torch.stack([group[k] for k in list(GROUPS)[:4]]).mean()
    motion=(group['motion_even']*2+group['motion_odd'])/3
    retention=(((z[:,:256]-old).square().mean(1))*weights).mean()
    difference=(z[::2,:256]-z[1::2,:256]).square().mean(1)
    previous=(old[::2]-old[1::2]).square().mean(1)
    # Penalize only excess change over the selected smooth teacher, at measured
    # short physical lags. Never force fast velocity channels to be stationary.
    smooth=(torch.relu(difference-previous)*weights[::2]*(lag<=config['maximum_smoothness_lag_ps'])).mean()
    value=structural+config['motion_weight']*motion+config['teacher_weight']*retention+config['smoothness_weight']*smooth
    metrics={k:float(v.detach()) for k,v in group.items()}
    metrics.update(retention=float(retention.detach()),excess_temporal_change=float(smooth.detach()),total=float(value.detach()))
    if not torch.isfinite(value): raise FloatingPointError(f'Invalid training loss: {metrics}')
    return value,metrics


def replay(config,model,heads,clouds,target,old,weights,lag,device):
    z=encode(config,model,clouds,device).detach().requires_grad_(True)
    value,metrics=loss(config,heads,z,target,old,weights,lag);value.backward()
    for start in range(0,len(clouds),config['micro_batch_size']):
        actual=encode(config,model,clouds[start:start+config['micro_batch_size']],device,gradients=True)
        actual.backward(z.grad[start:start+len(actual)])
    return metrics


def verify(config,args):
    """Actual MACE forward symmetries and gradient replay on real local groups."""
    root=Path(config['output'])/'technical'
    files=sorted(Path(config['cache']).glob('source-*.npz'))
    if not files: raise FileNotFoundError('Prepare at least one real source before verification')
    a=np.load(files[0]);p=a['pointers'];x,v=a['positions'],a['velocities']
    clouds=[(x[i:j],v[i:j]) for i,j in zip(p[:4],p[1:5],strict=True)]
    mace=backbone(config,args.device)
    model=MACEVelocityEncoder(mace,np.zeros(256),np.ones(256)).to(args.device).eval()
    rng=np.random.default_rng(config['seed']);x,v=clouds[0]
    rotation,_=np.linalg.qr(rng.normal(size=(3,3)));permutation=np.r_[0,rng.permutation(np.arange(1,len(x)))]
    variants=[(x,v),(x,v),(x@rotation,v@rotation),(x[permutation],v[permutation]),
              (x+np.array([5.,-4.,7.]),v),(x,v+np.array([10.,-7.,3.])),(x,-v),(x,np.zeros_like(v))]
    zz=encode(config,model,[(a.astype(np.float32),b.astype(np.float32)) for a,b in variants],args.device).cpu().numpy()
    errors={}
    for name,index in [('repeat',1),('rotation',2),('permutation',3),('translation',4),('velocity_boost',5)]:
        errors[name]=[]
        for section in (slice(0,256),slice(256,288),slice(288,304)):
            error=float(np.linalg.norm(zz[index,section]-zz[0,section])/max(np.linalg.norm(zz[0,section]),1e-12))
            errors[name].append(error)
        if max(errors[name])>2e-4: raise AssertionError(f'Broken symmetry {name}: {errors[name]}')
    np.testing.assert_allclose(zz[6,:288],zz[0,:288],rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(zz[6,288:],-zz[0,288:],rtol=2e-5,atol=2e-6)
    np.testing.assert_allclose(zz[7,256:],0,atol=1e-7)
    if np.linalg.norm(zz[0,256:])<1e-6: raise AssertionError('Velocity branch ignores measured velocities')
    # Reordered clouds and different microbatch sizes must give the same result.
    z1=encode(config,model,clouds,args.device)
    with torch.no_grad():
        original_graph,_=graph(config,clouds,args.device)
        original=context_features(mace,original_graph)
        structural_replay_error=float((z1[:,:256]-original).norm()/original.norm())
    if structural_replay_error>2e-6:
        raise AssertionError(f'Original structural representation changed: {structural_replay_error}')
    z2=encode(dict(config,micro_batch_size=1),model,clouds[::-1],args.device).flip(0)
    batch_error=float((z1-z2).norm()/z1.norm())
    if batch_error>2e-5: raise AssertionError(f'Batch-order sensitivity: {batch_error}')
    heads=Heads().to(args.device)
    target=torch.as_tensor(a['targets'][:4],device=args.device)
    old=z1[:,:256].detach();weights=torch.ones(4,device=args.device);lag=torch.ones(2,device=args.device)*.3
    model.zero_grad(set_to_none=True);heads.zero_grad(set_to_none=True)
    direct=encode(config,model,clouds,args.device,gradients=True)
    value,_=loss(config,heads,direct,target,old,weights,lag);value.backward()
    parameters=list(model.named_parameters())+[(f'head.{n}',p) for n,p in heads.named_parameters()]
    expected={n:p.grad.detach().clone() for n,p in parameters if p.grad is not None}
    model.zero_grad(set_to_none=True);heads.zero_grad(set_to_none=True)
    replay(config,model,heads,clouds,target,old,weights,lag,args.device)
    numerator=denominator=0.
    for n,p in parameters:
        if n in expected:
            if p.grad is None or not torch.isfinite(p.grad).all(): raise AssertionError(f'Invalid gradient: {n}')
            numerator+=float((p.grad-expected[n]).double().square().sum())
            denominator+=float(expected[n].double().square().sum())
    gradient_error=numerator/denominator
    if gradient_error>1e-6: raise AssertionError(f'Gradient replay mismatch: {gradient_error}')
    backbone_gradient=sum(float(p.grad.square().sum()) for n,p in model.mace.named_parameters() if p.grad is not None)
    if backbone_gradient==0: raise AssertionError('MACE backbone does not receive training gradients')
    report=dict(state='complete',symmetry_relative_errors_by_structure_activity_flow=errors,
                original_structural_readout_relative_error=structural_replay_error,
                reordered_batch_relative_error=batch_error,replay_gradient_relative_squared_error=gradient_error,
                backbone_gradient_squared_norm=backbone_gradient,velocity_motion_block_norm=float(np.linalg.norm(zz[0,256:])),
                zero_velocity_motion_block_norm=float(np.linalg.norm(zz[7,256:])),
                implementation_hashes=implementation_hashes(config))
    write_json(root/'verification.json',report);print('VERIFIED',report,flush=True)


def setup(config,args):
    root=Path(config['output'])/'technical'
    verification=read(root/'verification.json')
    if verification['state']!='complete' or verification['implementation_hashes']!=implementation_hashes(config):
        raise ValueError('Real GPU verification must pass for the current implementation')
    report=read(root/'teacher-status.json')
    if report['state']!='complete' or sha256(root/'teacher.npz')!=report['sha256'] or sha256(Path(config['cache'])/'manifest.json')!=report['cache_manifest_sha256']:
        raise ValueError('Incomplete or changed teacher/cache')
    clouds,y,meta=load_cache(config);norm=dict(np.load(root/'teacher.npz'))
    model=MACEVelocityEncoder(backbone(config,args.device),norm['feature_mean'],norm['feature_scale'],
        use_velocity=args.variant=='coordinates_velocity',velocity_scale=config['velocity_scale_A_per_ps']).to(args.device)
    # Identical head initialization and optimizer batches in both variants.
    torch.manual_seed(config['seed']+1);heads=Heads().to(args.device)
    target=torch.as_tensor((y-norm['target_mean'])/norm['target_scale'],device=args.device)
    old=torch.as_tensor((norm['z']-norm['feature_mean'])/norm['feature_scale'],device=args.device)
    return model,heads,clouds,target,old,meta,norm


def validation(config,model,heads,clouds,target,old,meta,split,device):
    ids=np.array([i for i,m in enumerate(meta) if m['split']==split])
    result=[];model.eval();heads.eval()
    for start in range(0,len(ids),config['batch_pairs']):
        pairs=ids[start:start+config['batch_pairs']];rows=np.ravel(np.c_[2*pairs,2*pairs+1])
        z=encode(config,model,[clouds[i] for i in rows],device)
        with torch.no_grad():
            _,metrics=loss(config,heads,z,target[rows],old[rows],torch.ones(len(rows),device=device),
                    torch.as_tensor([meta[i]['lag_ps'] for i in pairs],device=device))
        result.append((len(pairs),metrics))
    return {k:sum(n*m[k] for n,m in result)/len(ids) for k in result[0][1]}


def implementation_hashes(config):
    paths=[Path(__file__),Path(__file__).with_name('data.py'),Path(__file__).with_name('inventory.py'),
           Path('src/models/encoders/mace_velocity.py'),Path('src/models/encoders/mace_context.py'),
           Path('src/research/mace_local_state/physics.py'),Path('src/analysis/liquid_structure.py')]
    return {str(p):sha256(p) for p in paths}


def train(config,args):
    model,heads,clouds,target,old,meta,norm=setup(config,args)
    root=Path(config['output'])/'technical'/args.variant;root.mkdir(exist_ok=True,parents=True)
    optimizer=torch.optim.AdamW([
        dict(params=[p for p in model.mace.parameters() if p.requires_grad],lr=config['backbone_learning_rate']),
        dict(params=list(model.activity.parameters())+list(model.flow.parameters())+list(heads.parameters()),lr=config['head_learning_rate'])],weight_decay=.0001)
    rng=np.random.default_rng(config['seed']);history=[];best=float('inf');epoch_start=1
    hashes=implementation_hashes(config)
    if args.resume:
        checkpoint=torch.load(root/'last.pt',map_location=args.device,weights_only=False)
        if checkpoint['implementation_hashes']!=hashes: raise ValueError('Resume code changed; use a separately named continuation protocol')
        model.load_state_dict(checkpoint['encoder_state'],strict=True);heads.load_state_dict(checkpoint['head_state'],strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state']);rng.bit_generator.state=checkpoint['numpy_rng_state']
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu());torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu(),args.device)
        history=checkpoint['history'];best=checkpoint['best_validation'];epoch_start=checkpoint['epoch']+1
    elif (root/'last.pt').exists(): raise FileExistsError(f'Use --resume for existing run: {root}')
    ids=np.array([i for i,m in enumerate(meta) if m['split']=='train'])
    counts=Counter(meta[i]['balance_group'] for i in ids)
    weight=np.array([1/counts[meta[i]['balance_group']] for i in ids]);weight/=weight.mean()
    all_weight=np.zeros(len(meta));all_weight[ids]=weight
    started=time.monotonic();deadline=datetime.fromisoformat(config['deadline_utc']).timestamp()
    def save(name,epoch):
        payload=dict(protocol='mace_local_phase_space_v1',variant=args.variant,config=config,epoch=epoch,
            encoder_state=model.state_dict(),head_state=heads.state_dict(),optimizer_state=optimizer.state_dict(),
            numpy_rng_state=rng.bit_generator.state,torch_rng_state=torch.get_rng_state(),
            cuda_rng_state=torch.cuda.get_rng_state(args.device),history=history,best_validation=best,
            normalization={k:v for k,v in norm.items() if k!='z'},implementation_hashes=hashes,
            cache_manifest_sha256=sha256(Path(config['cache'])/'manifest.json'),
            teacher_sha256=sha256(Path(config['output'])/'technical/teacher.npz'))
        temporary=root/(name+'.building');torch.save(payload,temporary);temporary.replace(root/name)
    if epoch_start==1:
        initial=validation(config,model,heads,clouds,target,old,meta,'val',args.device)
        best=np.mean([initial[k] for k in GROUPS]);history.append(dict(epoch=0,validation=initial))
        save('best.pt',0);save('last.pt',0)
    write_json(root/'config.json',dict(config=config,variant=args.variant,implementation_hashes=hashes,
        objective='Current local geometry and velocity observables, smooth structural teacher retention; no future target.'))
    for epoch in range(epoch_start,config['epochs']+1):
        if time.time()>deadline-600:
            write_json(root/'status.json',dict(state='paused_at_deadline',completed_epochs=epoch-1,best_validation=best));return
        start=time.monotonic();model.train();heads.train();metrics=[]
        order=rng.permutation(ids)
        for step,a in enumerate(range(0,len(order),config['batch_pairs'])):
            pairs=order[a:a+config['batch_pairs']];rows=np.ravel(np.c_[2*pairs,2*pairs+1])
            optimizer.zero_grad(set_to_none=True)
            record=replay(config,model,heads,[clouds[i] for i in rows],target[rows],old[rows],
                torch.as_tensor(np.repeat(all_weight[pairs],2),device=args.device,dtype=torch.float32),
                torch.as_tensor([meta[i]['lag_ps'] for i in pairs],device=args.device),args.device)
            torch.nn.utils.clip_grad_norm_(model.mace.parameters(),1.,error_if_nonfinite=True)
            torch.nn.utils.clip_grad_norm_(list(model.activity.parameters())+list(model.flow.parameters())+list(heads.parameters()),5.,error_if_nonfinite=True)
            optimizer.step();metrics.append(record)
            if step%10==0:
                status=dict(state='training',epoch=epoch,step=step+1,steps=(len(order)+config['batch_pairs']-1)//config['batch_pairs'],
                    metrics=record,elapsed_seconds=time.monotonic()-started)
                write_json(root/'status.json',status);print('TRAIN',args.variant,status,flush=True)
            if time.time()>deadline-120:
                # Last is the exact completed epoch. Never label a partial epoch
                # as complete; save its weights separately for diagnostic recovery.
                save('interrupted-partial.pt',epoch-1)
                write_json(root/'status.json',dict(state='paused_mid_epoch',epoch=epoch,step=step+1,
                    exact_resume_checkpoint='last.pt'));return
        val=validation(config,model,heads,clouds,target,old,meta,'val',args.device)
        record=dict(epoch=epoch,training={k:float(np.mean([m[k] for m in metrics])) for k in metrics[0]},
                    validation=val,epoch_seconds=time.monotonic()-start)
        history.append(record);score=np.mean([val[k] for k in GROUPS])
        if score<best:best=score;save('best.pt',epoch)
        save('last.pt',epoch);write_json(root/'history.json',history)
        print('EPOCH',args.variant,record,flush=True)
    write_json(root/'status.json',dict(state='trained',completed_epochs=config['epochs'],best_validation=best))
    evaluate(config,args)


def evaluate(config,args):
    from .evaluate import evaluate_checkpoint
    evaluate_checkpoint(config,args)
