"""Matched spatial starts and value/derivative, frozen/joint motion continuations."""
import math
import time
import numpy as np
import torch
from src.experiment_runner.wandb_tracking import online_training
from src.experiment_runner.metric_docs import write_metric_rows
from src.research.continuous_motion.model import mixture_nll
from src.research.continuous_motion.train import Sampler
from src.research.continuous_motion import teacher as physical_teacher
from src.research.continuous_motion.data import atomic_save
from .data import FAMILY,read,root,cache,resolve_path,digest,sha,write_json,spatial_data
from .model import Spatial,Motion,spatial_backward,spatial_loss,views,fixed_geometry,reference_contractions,geometry_jvp


def tech(c,name):
    p=root(c)/'analyses'/name/'technical';p.mkdir(parents=True,exist_ok=True);return p


def spatial_start(c,method):
    if 'spatial_initialization' not in c:return tech(c,'spatial-'+method)
    spec=c['spatial_initialization'];source=resolve_path(spec['output'])
    if read(source/'technical/plan.json')['identity']!=spec['plan_identity']:raise ValueError('Changed spatial initialization campaign')
    p=source/'analyses'/('spatial-'+method)/'technical';done=read(p/'complete.json')
    if done['update']!=c['spatial_training']['updates'] or done['checkpoint_sha256']!=spec['checkpoints'][method] or sha(p/'best.pt')!=spec['checkpoints'][method]:
        raise ValueError('Spatial initialization is not the bound complete epoch-12 checkpoint')
    return p


def tracking(c,name,identity,path):
    return online_training(c['wandb'],run_id='sresp-'+identity[:16],name=name,config=dict(c,task=name,identity=identity),
        folder=path,receipt_path=path/'wandb.json',job_type='encoder' if name.startswith('spatial-') or name.endswith('_joint') else 'predictor',
        group=c['protocol'],receipt_fields=dict(identity=identity))


def signature(c,name):return digest(dict(plan=read(root(c)/'technical/plan.json')['identity'],task=name))


def common_dynamics(c,model,path):
    """Bind every arm to one actual initialization, including across GPU types."""
    import fcntl
    base=root(c)/'technical';target=base/'motion-initialization.pt';receipt=base/'motion-initialization.json'
    with (base/'motion-initialization.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not receipt.exists():
            initial={k:v.detach().cpu() for k,v in model.state_dict().items() if not k.startswith('geometry.')}
            atomic_save(target,initial);write_json(receipt,dict(seed=c['seed'],sha256=sha(target)))
        record=read(receipt)
        if sha(target)!=record['sha256']:raise ValueError('Changed common latent-dynamics initialization')
        initial=torch.load(target,weights_only=False,map_location='cpu')
    current=model.state_dict()
    if set(initial)!={k for k in current if not k.startswith('geometry.')}:raise ValueError('Changed latent-dynamics parameterization')
    current.update(initial);model.load_state_dict(current)
    write_json(path/'initialization.json',record)


def spectrum(z):
    z=z.float();cov=torch.cov(z.T);eig=torch.linalg.eigvalsh(cov).clamp_min(0)
    return dict(mean_std=float(z.std(0).mean()),minimum_std=float(z.std(0).min()),
        participation_rank=float(eig.sum().square()/eig.square().sum().clamp_min(1e-20)))


def validate_spatial(model,ema,c,method,micro):
    a=np.load(cache(c)/'spatial-selection.npz');weights=torch.tensor(read(cache(c)/'spatial.json')['weights'],device='cuda')
    metrics=[];vectors=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(a['positions']),256):
            q=torch.from_numpy(a['positions'][start:start+256]).cuda();y=torch.from_numpy(a['target'][start:start+256]).cuda()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                z=[torch.cat([model.geometry(x[i:i+micro],None if keep is None else keep[i:i+micro]) for i in range(0,len(x),micro)]) for x,keep in views(q,method,c['seed']+start)]
                target=None if method!='masked_teacher' else torch.cat([ema(x) for x in q.split(micro)])
            loss,stats=spatial_loss(model,z,y,weights,method,target)
            metrics.append(dict(loss=float(loss),**{k:float(v) for k,v in stats.items()}));vectors.append(z[0].cpu())
    return {**{k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]},**spectrum(torch.cat(vectors))}


def spatial(c,method,deadline,micro):
    name='spatial-'+method;p=tech(c,name);identity=signature(c,name);cfg=c['spatial_training']
    data,columns,weights=spatial_data(c);weights=torch.from_numpy(weights).cuda()
    torch.manual_seed(c['seed']);model=Spatial(c).cuda();ema=fixed_geometry(c,model.geometry)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    start=0;history=[]
    if (p/'last.pt').exists():
        saved=torch.load(p/'last.pt',map_location='cuda',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Changed spatial resume')
        model.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);opt.load_state_dict(saved['optimizer']);start=saved['update'];history=saved['history']
    def save(update):
        atomic_save(p/'last.pt',dict(identity=identity,model=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),update=update,history=history,config=c))
    batch=cfg['batch_size'];steps=math.ceil(len(data)/batch);epoch=-1;order=None
    if cfg['updates']!=steps*cfg['epochs']:raise ValueError('Spatial budget must cover the declared complete epochs')
    with tracking(c,name,identity,p) as run:
        for update in range(start+1,cfg['updates']+1):
            if time.time()>deadline-240:
                save(update-1);data.close()
                write_json(p/'progress.json',dict(state='checkpointed_for_continuation',update=update-1,total=cfg['updates'],microbatch=micro))
                return False
            current_epoch=(update-1)//steps
            if epoch!=current_epoch:
                epoch=current_epoch;order=np.random.default_rng(np.random.SeedSequence([c['seed'],epoch])).permutation(len(data))
            step=(update-1)%steps
            model.train();opt.zero_grad(set_to_none=True);ids=order[step*batch:(step+1)*batch]
            b=data.batch(ids);q=torch.from_numpy(b['positions']).cuda();y=torch.from_numpy(b['target'][:,columns]).cuda()
            fraction=min(1.,update/cfg['warmup']);floor=cfg['min_lr']/cfg['learning_rate']
            cosine=floor+(1-floor)*.5*(1+math.cos(math.pi*max(0,update-cfg['warmup'])/(cfg['updates']-cfg['warmup'])))
            opt.param_groups[0]['lr']=cfg['learning_rate']*fraction*cosine
            loss,stats=spatial_backward(model,ema,q,y,weights,method,c['seed']+update,micro)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True);opt.step()
            with torch.no_grad():
                for target,current in zip(ema.parameters(),model.geometry.parameters(),strict=True):target.lerp_(current,cfg['ema'])
            row=dict(update=update,epoch=epoch+(step+1)/steps,learning_rate=opt.param_groups[0]['lr'],loss=loss,gradient_norm=float(norm),**stats)
            if update%cfg['selection_every']==0 or update==cfg['updates']:
                val=validate_spatial(model,ema,c,method,micro);row.update({'validation_'+k:v for k,v in val.items()})
                write_json(p/'validation.json',dict(update=update,**val))
            if not np.isfinite(loss):raise FloatingPointError('Nonfinite spatial objective')
            history.append(row);run.log(row,step=update)
            if update%cfg['save_every']==0 or update==cfg['updates']:save(update)
            if update==1 or update%25==0:
                write_json(p/'progress.json',dict(state='training',update=update,total=cfg['updates'],microbatch=micro,loss=loss));print(name,row,flush=True)
        save(cfg['updates'])
        # Fixed final-update export: held-out structural observations do not select it.
        atomic_save(p/'best.pt',dict(identity=identity,geometry=model.geometry.state_dict(),model=model.state_dict(),config=c,update=cfg['updates']))
        run.summary.update({'validation/'+k:v for k,v in val.items()})
    columns=list(dict.fromkeys(k for row in history for k in row));write_metric_rows(history,p.parent,family=FAMILY,name='learning',columns=columns)
    write_json(p/'complete.json',dict(identity=identity,state='complete',checkpoint_sha256=sha(p/'best.pt'),update=cfg['updates'],validation=val))
    data.close();return True


def load_motion_data(c):
    campaign=resolve_path(c['response_campaign']);p=read(campaign/'technical/plan.json');old=p['config']
    path=resolve_path(old['cache'])/'dataset.pt'
    if sha(path)!=read(campaign/'technical/data-complete.json')['sha256']:raise ValueError('Changed motion observations')
    data=torch.load(path,weights_only=False);bank=physical_teacher.load(old,p)
    if sha(cache(c)/'future-positions.pt')!=read(cache(c)/'complete.json')['files']['future-positions.pt']:raise ValueError('Changed future positions')
    future=torch.load(cache(c)/'future-positions.pt',weights_only=False)
    if future.shape[:2]!=(len(data['rows']),len(c['horizons_ps'])):raise ValueError('Wrong future tensor')
    return data,bank,future


def short_objective(model,record,c,arm,seed,*,create_graph=True):
    if c.get('neural_derivative')=='centered_student_v1':
        from .fast import short_objective as fast_short
        return fast_short(model,record,c,arm,seed,create_graph=create_graph)
    with reference_contractions(model.geometry):
        return short_objective_reference(model,record,c,arm,seed,create_graph=create_graph)


def short_objective_reference(model,record,c,arm,seed,*,create_graph=True):
    # FP32 island for nested input derivatives; spatial and long prediction use BF16 AMP.
    def predicted_z(z):return model.path(z,[.1],particles=4,seed=seed)[...,:128].mean(1)[:,0]
    if arm['frozen']:
        x=record['z'].cuda();direction=record['dz'].cuda();fn=predicted_z
    else:
        x=record['state'].cuda();direction=record['basis'].cuda()
        def fn(state):return predicted_z(model.encode(state,create_graph=True))
    with torch.set_grad_enabled(create_graph):prediction=fn(x)
    if not create_graph:prediction=prediction.detach()
    value=.5*(prediction-record['target'].cuda()).square().mean();derivative=value.new_zeros(())
    if arm['derivative']:
        for k in range(2):
            _,dp=torch.autograd.functional.jvp(fn,x,direction[...,k],create_graph=create_graph,strict=True)
            dt=record['response'][...,k].cuda()
            derivative=derivative+.25*(dp-dt).square().mean()/dt.square().mean().clamp_min(1e-6)
    return value,derivative


def predict_batches(model,data,future_target,zall,indices,c,micro):
    nll=[];mse=[];latent=[];predictions=[];codes=[];coverage=[];variances=[]
    center=data['mean'].float().cuda();scale=data['scale'].float().cuda()
    model.eval()
    for offset in range(0,len(indices),256):
        block=indices[offset:offset+256];z=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            if zall is not None:
                z=zall[block].cuda()
                if c.get('neural_derivative')=='centered_student_v1':z=model.adapt(z)
            else:z=torch.cat([model.encode(data['state'][part].cuda(),create_graph=False) for part in np.array_split(block,max(1,math.ceil(len(block)/micro)))])
            path=model.path(z,c['horizons_ps'],particles=8,seed=c['seed']+811)
            mu,lv=model.prediction(path)
        mu=mu.float();lv=lv.float();y=(data['target'][block].float().cuda()-center)/scale
        nll.append(mixture_nll(mu,lv,y).detach().cpu());mse.append((mu.mean(1)-y).square().mean(-1).detach().cpu())
        pit=(.5*(1+torch.erf((y[:,None]-mu)/torch.sqrt(2*lv.exp())))).mean(1)
        coverage.append(((pit>=.05)&(pit<=.95)).float().mean(-1).detach().cpu())
        variances.append((lv.exp()+mu.square()).mean(1).sub(mu.mean(1).square()).detach().cpu())
        latent.append((path[...,:128].mean(1)-future_target[block].cuda()).square().mean(-1).detach().cpu())
        predictions.append(mu.mean(1).detach().cpu());codes.append(z[:,:128].detach().cpu())
    return dict(nll=torch.cat(nll).numpy(),descriptor_mse=torch.cat(mse).numpy(),structural_mse=torch.cat(latent).numpy(),coverage90=torch.cat(coverage).numpy(),variance=torch.cat(variances).numpy(),mean=torch.cat(predictions).numpy(),structural=torch.cat(codes).numpy())


def motion(c,method,arm,deadline,micro):
    name=method+'-'+arm['name'];p=tech(c,name);identity=signature(c,name);cfg=c['response_training']
    pretrained=spatial_start(c,method);receipt=read(pretrained/'complete.json')
    if sha(pretrained/'best.pt')!=receipt['checkpoint_sha256']:raise ValueError('Changed spatial start')
    start_state=torch.load(pretrained/'best.pt',map_location='cpu',weights_only=False)
    fast=c.get('neural_derivative')=='centered_student_v1'
    if fast:
        from .fast import FastMotion
        constructor=FastMotion
    else:constructor=Motion
    torch.manual_seed(c['seed']);model=constructor(c).cuda();model.geometry.load_state_dict(start_state['geometry'])
    if fast:model.set_arm(arm)
    common_dynamics(c,model,p)
    anchor=fixed_geometry(c,model.geometry)
    replay=None
    if not arm['frozen'] and cfg['replay_weight']>0:
        if cfg['replay_every']<1:raise ValueError('Enabled spatial replay needs a positive interval')
        replay=Spatial(c).cuda();replay.load_state_dict(start_state['model']);replay.requires_grad_(False);replay.geometry=model.geometry
    model.geometry.requires_grad_(not arm['frozen'])
    dynamics=[p for n,p in model.named_parameters() if not n.startswith('geometry.')]
    opt=torch.optim.AdamW([dict(params=dynamics,lr=cfg['learning_rate']),dict(params=model.geometry.parameters(),lr=cfg['learning_rate']*cfg['encoder_lr_scale'])],weight_decay=cfg['weight_decay'])
    data,bank,future=load_motion_data(c)
    if fast:
        from .fast import cached_targets
    else:
        from .targets import cached_targets
    try:
        target,zall,short=cached_targets(c,method,model,anchor,bank,future,data['state'],arm['frozen'],micro,deadline,p)
    except TimeoutError:
        write_json(p/'progress.json',dict(state='teacher_preparation_checkpointed_for_continuation',update=0,total=cfg['updates']))
        return False
    # Sample from the physical producer's source/ancestor metadata. Cached
    # targets retain job IDs, but older target packets do not include source.
    if [r['job'] for r in bank]!=[r['job'] for r in short]:raise ValueError('Teacher targets no longer match physical bank order')
    short_sampler=Sampler(bank,'train')
    del future,bank
    spatial_data_object=None
    if replay is not None:spatial_data_object,columns,weights=spatial_data(c);weights=torch.from_numpy(weights).cuda()
    from collections import Counter
    write_json(p/'motion-data.json',dict(response_campaign=c['response_campaign'],response_identity=c['response_identity'],
        observation_rows=len(data['rows']),roles=dict(Counter(r['role'] for r in data['rows'])),
        encoder_inputs='unrelaxed local80 positions and same-frame saved velocities; no conditions',
        predictor_inputs='128 structural coordinates and128 directional motion coordinates; no conditions',
        broad_spatial_replay=replay is not None,spatial_rows=0 if spatial_data_object is None else len(spatial_data_object),
        spatial_initialization=c.get('spatial_initialization'),birth_evaluation=c.get('birth_evaluation')))
    sampler=Sampler(data['rows'],'train');rng=np.random.default_rng(c['seed'])
    selection=np.unique(Sampler(data['rows'],'selection').draw(np.random.default_rng(c['seed']+81),256))
    center=data['mean'].float().cuda();scale=data['scale'].float().cuda()
    start=0;history=[];best=float('inf');selected=0
    if (p/'last.pt').exists():
        saved=torch.load(p/'last.pt',map_location='cuda',weights_only=False)
        if saved['identity']!=identity:raise ValueError('Changed motion resume')
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);rng.bit_generator.state=saved['rng']
        start=saved['update'];history=saved['history'];best=saved['best'];selected=saved['selected']
    def payload(update):return dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,update=update,history=history,best=best,selected=selected,config=c,method=method,arm=arm,spatial_sha256=receipt['checkpoint_sha256'])
    step_seconds=30. if arm['frozen'] or fast else 1200.
    with tracking(c,name,identity,p) as run:
        for update in range(start+1,cfg['updates']+1):
            if time.time()>deadline-max(600.,2*step_seconds+300.):
                atomic_save(p/'last.pt',payload(update-1))
                write_json(p/'progress.json',dict(state='checkpointed_for_continuation',update=update-1,total=cfg['updates'],microbatch=micro))
                return False
            begin=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
            indices=sampler.draw(rng,256);noise_seed=int(rng.integers(2**31));total_nll=total_latent=0.
            if fast:
                from .fast import long_backward
                total_nll,total_latent=long_backward(model,data,target,zall,torch.from_numpy(indices),c,micro,noise_seed)
            else:
                long_micro=(256 if micro==cfg['microbatch'] else 128) if arm['frozen'] else micro
                for offset in range(0,256,long_micro):
                    ids=indices[offset:offset+long_micro]
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        z=zall[ids].cuda() if arm['frozen'] else model.encode(data['state'][ids].cuda(),create_graph=True)
                        path=model.path(z,c['horizons_ps'],particles=4,seed=noise_seed,random_batch_size=256,sample_offset=offset)
                        mu,lv=model.prediction(path)
                    y=(data['target'][ids].float().cuda()-center)/scale
                    nll=mixture_nll(mu.float(),lv.float(),y).mean();latent=.5*(path[...,:128].mean(1)-target[ids].cuda()).square().mean()
                    loss=(nll+cfg['long_structural_weight']*latent)*len(ids)/256
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite long future objective')
                    loss.backward();total_nll+=float(nll.detach())*len(ids)/256;total_latent+=float(latent.detach())*len(ids)/256
            value=tangent=0.
            for i in short_sampler.draw(rng,cfg['short_batch']):
                v,d=short_objective(model,short[i],c,arm,int(rng.integers(2**31)))
                loss=(cfg['short_weight']*v+cfg['derivative_weight']*d)/cfg['short_batch']
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite short response')
                loss.backward();value+=float(v.detach())/cfg['short_batch'];tangent+=float(d.detach())/cfg['short_batch']
            replay_value=0.
            if replay is not None and update%cfg['replay_every']==0:
                # Scaling the full objective (including cotangents) keeps head and encoder gradients consistent.
                replay_rng=np.random.default_rng(np.random.SeedSequence([c['seed'],update,442]))
                ids=replay_rng.choice(len(spatial_data_object),256,replace=False);b=spatial_data_object.batch(ids)
                before={n:p.grad.detach().clone() if p.grad is not None else None for n,p in model.geometry.named_parameters()}
                replay_micro=256 if micro==cfg['microbatch'] else 128
                replay_value,_=spatial_backward(replay,anchor,torch.from_numpy(b['positions']).cuda(),torch.from_numpy(b['target'][:,columns]).cuda(),weights,method,c['seed']+update,replay_micro)
                for n,param in model.geometry.named_parameters():
                    if param.grad is None:continue
                    old=before[n]
                    param.grad.mul_(cfg['replay_weight'])
                    if old is not None:param.grad.add_(old,alpha=1-cfg['replay_weight'])
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step()
            row=dict(update=update,future_nll=total_nll,long_structural_half_mse=total_latent,short_half_mse=value,relative_tangent=tangent,spatial_replay=replay_value,gradient_norm=float(norm))
            if update%cfg['selection_every']==0 or update==cfg['updates']:
                val=predict_batches(model,data,target,zall,selection,c,micro);score=float(val['nll'].mean())
                if not math.isfinite(score):raise FloatingPointError('Nonfinite inner-selection future NLL')
                row.update(selection_nll=score,selection_structural_mse=float(val['structural_mse'].mean()),**{'selection_'+k:v for k,v in spectrum(torch.from_numpy(val['structural'])).items()})
                if score<best:
                    best=score;selected=update;atomic_save(p/'best.pt',payload(update))
            history.append(row);run.log(row,step=update)
            if not arm['frozen'] or update%cfg['save_every']==0 or update==cfg['updates']:atomic_save(p/'last.pt',payload(update))
            step_seconds=time.monotonic()-begin
            if update==1 or update%10==0:
                write_json(p/'progress.json',dict(state='training',update=update,total=cfg['updates'],microbatch=micro,last_update_seconds=time.monotonic()-begin,selection_nll=best if math.isfinite(best) else None));print(name,row,flush=True)
        if time.time()>deadline-3600:
            write_json(p/'progress.json',dict(state='evaluation_checkpointed_for_continuation',update=cfg['updates'],total=cfg['updates']))
            return False
        saved=torch.load(p/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model'])
        indices=np.array([i for i,r in enumerate(data['rows']) if r['role']=='test']);evaluation=predict_batches(model,data,target,zall,indices,c,micro)
        from src.research.response_geometry.features import observed
        fit=torch.tensor([r['role']=='train' for r in data['rows']])
        fit_y=(data['target'][fit].float().cuda()-center)/scale
        marginal_mu=fit_y.mean(0);marginal_lv=fit_y.var(0,unbiased=False).clamp_min(1e-5).log()
        persistence=[];marginal=[]
        with torch.no_grad():
            for block in np.array_split(indices,max(1,math.ceil(len(indices)/128))):
                y=(data['target'][block].float().cuda()-center)/scale
                present=(observed(data['state'][block,:,:3].double().cuda())-center)/scale
                persistence.append((present[:,None]-y).square().mean(-1).cpu())
                marginal.append(mixture_nll(marginal_mu[None,None].expand(len(block),1,-1,-1),marginal_lv[None,None].expand(len(block),1,-1,-1),y).cpu())
        evaluation.update(persistence_mse=torch.cat(persistence).numpy(),marginal_nll=torch.cat(marginal).numpy())
        np.savez_compressed(p/'future-predictions.npz',row=indices,**evaluation)
        from src.research.continuous_motion.evaluate import summarize
        metrics=[]
        for track,mask in [('all64',np.ones(len(indices),bool)),('legacy16',np.array([data['rows'][i]['legacy'] for i in indices]))]:
            ancestors=[data['rows'][i]['ancestor'] for i,k in zip(indices,mask,strict=True) if k]
            for j,h in enumerate(c['horizons_ps']):
                row=dict(track=track,horizon_ps=h,observations=int(mask.sum()),ancestors=len(set(ancestors)))
                for k in ('nll','descriptor_mse','structural_mse','coverage90','persistence_mse','marginal_nll'):
                    summary=summarize(evaluation[k][mask,j],ancestors,c['seed']);row.update({k:summary['mean'],k+'_lower':summary['lower'],k+'_upper':summary['upper']})
                metrics.append(row)
        write_metric_rows(metrics,p.parent,family=FAMILY,name='future-scores')
        response_rows=[]
        for r in short:
            if r['role']!='test':continue
            v,d=short_objective(model,r,c,dict(arm,derivative=True),c['seed']+981,create_graph=False)
            response_rows.append(dict(job=r['job'],ancestor=r['ancestor'],endpoint_half_mse=float(v.detach()),relative_tangent=float(d.detach())))
        write_metric_rows(response_rows,p.parent,family=FAMILY,name='heldout-responses')
        run.summary.update({f"evaluation/{r['track']}/{r['horizon_ps']}ps/{k}":r[k] for r in metrics for k in ('nll','descriptor_mse','structural_mse','coverage90','persistence_mse','marginal_nll')})
        run.summary.update({'evaluation/short_'+k:summarize([r[k] for r in response_rows],[r['ancestor'] for r in response_rows],c['seed'])['mean'] for k in ('endpoint_half_mse','relative_tangent')})
        run.summary.update({'selected_update':selected,'selection_nll':best})
        if fast and 'birth_evaluation' in c:
            from .birth import evaluate as evaluate_birth
            try:birth_metrics=evaluate_birth(c,model,method,arm,p/'best.pt',p,deadline)
            except TimeoutError:
                write_json(p/'progress.json',dict(state='birth_evaluation_checkpointed_for_continuation',update=cfg['updates'],total=cfg['updates']))
                return False
            run.summary.update({f"birth_010_no_neighborhood_filter/{r['space']}/{r['lead_ps']}ps/{r['readout']}/{r['metric']}":r['value'] for r in birth_metrics})
    write_metric_rows(history,p.parent,family=FAMILY,name='learning',columns=list(dict.fromkeys(k for r in history for k in r)))
    write_json(p/'complete.json',dict(identity=identity,state='complete',checkpoint_sha256=sha(p/'best.pt'),selected_update=selected,metrics=metrics))
    if spatial_data_object is not None:spatial_data_object.close()
    return True
