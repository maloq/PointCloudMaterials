"""Compiled MACE, exact joint-gradient replay, fixed-target selection and resumable training."""
import copy,json,math,signal,time
from pathlib import Path
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.batches import move
from src.data.structural_pretraining.prepare import save_json,digest
from src.training_methods.shared_pretraining.compilation import compile_encoder,compilation_counters
from src.training_methods.structural_pretraining.objective import block_errors,PHYSICAL_BLOCKS,TDA_BLOCKS
from src.experiment_runner.metric_docs import write_metric_table
from .model import NeighborhoodModel
from .objective import Objective
from .data import NeighborhoodData,MixedBatches,loader


def encode(model,batches,precision):
    result=[]
    with torch.no_grad():
        for b in batches:
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):result.append(model.encoder(move(b,'cuda')).float())
    return torch.cat(result)


def training_step(model,objective,batches,target,precision):
    z=encode(model,batches,precision).detach().requires_grad_(True)
    loss,terms=objective(model,z,target)
    if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite neighborhood loss: {terms}')
    loss.backward();offset=0
    # Both predictor-input and target derivatives enter this replay; no detached teacher.
    for b in batches:
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):replay=model.encoder(move(b,'cuda')).float()
        n=len(replay);replay.backward(z.grad[offset:offset+n]);offset+=n
    if offset!=len(z):raise ValueError('Gradient replay lost observations')
    return float(loss.detach()),{k:float(v.detach()) for k,v in terms.items()}


def fixed_indices(data,per_group,seed):
    rng=np.random.default_rng(seed+315);return [int(i) for g in data.groups for i in rng.choice(g,min(per_group,len(g)),replace=False)]


@torch.no_grad()
def calibrate(model,objective,data,config):
    ids=fixed_indices(data,64,config['seed']);batches=[ids[i:i+16] for i in range(0,len(ids),16)]
    means=[];groups=[];model.eval();objective.eval()
    for packed,target in loader(data,batches,config['microbatch'],workers=0):
        z=encode(model,packed,config['precision']).reshape(len(target['group']),3,data.neighbors+1,-1)
        means.append(z[:,1:,0,:128].flatten(0,1).double());groups.append(target['group'].cuda().repeat_interleave(2))
    z=torch.cat(means);g=torch.cat(groups)
    for group in range(len(data.groups)):
        x=z[g==group];objective.z_mean[group].copy_(x.mean(0));objective.z_scale[group].copy_((x.var(0,unbiased=False)+1e-6).sqrt())


@torch.no_grad()
def evaluate(model,objective,data,config):
    calibrate(model,objective,data,config);records=[];sources=[]
    ids=data.selection;batches=[ids[i:i+16] for i in range(0,len(ids),16)]
    for packed,target in loader(data,batches,config['microbatch'],workers=0):
        target=move(target,'cuda');raw=encode(model,packed,config['precision']).reshape(len(target['group']),3,data.neighbors+1,-1)
        z=objective.normalized(raw,target['group']);current=z[:,1,0,:128]
        p=block_errors(model.physical(current,target['group']),(target['physical'][:,0]-objective.physical_mean)/objective.physical_std,PHYSICAL_BLOCKS).mean(-1)
        t=block_errors(model.tda(current,target['group']),(target['tda'][:,0]-objective.tda_mean)/objective.tda_std,TDA_BLOCKS).mean(-1)
        baseline=block_errors(objective.baseline_physical[target['group']],(target['physical'][:,0]-objective.physical_mean)/objective.physical_std,PHYSICAL_BLOCKS).mean(-1)
        baseline+=.25*block_errors(objective.baseline_tda[target['group']],(target['tda'][:,0]-objective.tda_mean)/objective.tda_std,TDA_BLOCKS).mean(-1)
        values=[p,t,baseline,raw[:,1,0,:128].square().mean(-1)]
        if objective.spec['prediction']=='temporal':
            inv,_=model.predict(z[:,1,0],z[:,0,0],target['position'][:,:1],target['times'][:,2:3],target['times'][:,0])
            future=(target['physical'][:,1]-objective.physical_mean)/objective.physical_std
            fp=block_errors(model.physical(inv[:,0],target['group']),future,PHYSICAL_BLOCKS).mean(-1)
            persistence=block_errors((target['physical'][:,0]-objective.physical_mean)/objective.physical_std,future,PHYSICAL_BLOCKS).mean(-1)
            values.extend((fp,persistence))
        records.append(torch.stack(values,-1).cpu().numpy());sources.extend(data.rows[int(i)][0]['source'] for i in target['index'].cpu())
    values=np.concatenate(records);source=np.asarray(sources);average=np.mean([values[source==s].mean(0) for s in np.unique(source)],0)
    metrics=dict(physical=float(average[0]),tda=float(average[1]),selection_score=float(average[0]+.25*average[1]),
        training_mean_baseline=float(average[2]),sources=len(np.unique(source)),windows=len(values))
    if values.shape[1]>4:metrics.update(future_physical=float(average[4]),physical_persistence=float(average[5]))
    return metrics,dict(values=values,sources=source,indices=np.array(ids))


def run(config,spec,deadline):
    root=resolve_path(config['output'])/'technical/runs'/spec['name'];root.mkdir(parents=True,exist_ok=True)
    data=NeighborhoodData(resolve_path(config['cache']),spec['neighbors']);identity=digest(dict(config=config,spec=spec,data=data.manifest['identity']))
    torch.manual_seed(config['seed']);torch.set_num_threads(1)
    model=NeighborhoodModel('mace',spec['previous_context'],len(data.groups)).cuda();objective=Objective(data.manifest,spec).cuda()
    objective.baseline_physical.copy_(torch.as_tensor(data.baselines['physical'],device='cuda'));objective.baseline_tda.copy_(torch.as_tensor(data.baselines['tda'],device='cuda'))
    last=root/'last.pt';saved=torch.load(last,map_location='cuda',weights_only=False) if last.exists() else None
    if saved is not None:
        if saved['identity']!=identity:raise ValueError('Training resume identity changed')
        model.load_state_dict(saved['model']);objective.load_state_dict(saved['objective'])
    first=next(iter(loader(data,MixedBatches(data,config['batch_size'],config['seed'],0,1),config['microbatch'],workers=0)))
    if config['compile']:compile_encoder(model.encoder,move(first[0][0],'cuda'),config['precision'])
    encoder=list(model.encoder.parameters());encoder_ids={id(p) for p in encoder};heads=[p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer=torch.optim.AdamW([{'params':encoder,'peak':spec['encoder_lr']},{'params':heads,'peak':spec['head_lr']}],weight_decay=1e-4)
    per_epoch=math.ceil(data.train_size/config['batch_size']);updates=spec['epochs']*per_epoch;step=0;best=float('inf');stop=False
    if saved is not None:
        optimizer.load_state_dict(saved['optimizer']);step=saved['step'];best=saved['best'];torch.set_rng_state(saved['rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    def request_stop(*_):
        nonlocal stop;stop=True
    signal.signal(signal.SIGTERM,request_stop);signal.signal(signal.SIGUSR1,request_stop)
    def checkpoint():
        torch.save(dict(identity=identity,spec=spec,model=model.state_dict(),objective=objective.state_dict(),optimizer=optimizer.state_dict(),
            step=step,best=best,rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),root/'last.building.pt')
        (root/'last.building.pt').replace(last)
    save_json(root/'spec.json',spec);save_json(root/'parameters.json',dict(encoder=sum(p.numel() for p in encoder),predictors=sum(p.numel() for p in heads)))
    wb=None
    if config['wandb']:
        import wandb
        wb_group=config.get('wandb_group','neighborhood-jepa-20260920')
        wb_id=f"{config.get('wandb_prefix','nj20')}-{spec['name']}"
        wb=wandb.init(entity='teshbek',project='PointCloudMaterials',group=wb_group,name=spec['name'],
            id=wb_id,resume='allow',dir=str(root),config=dict(config=config,spec=spec))
        wb.define_metric('epoch',hidden=True)
        for metric_prefix in ['loss','validation','optimization']:wb.define_metric(metric_prefix+'/*',step_metric='epoch')
    def validate():
        nonlocal best
        metrics,predictions=evaluate(model,objective,data,config);metrics['step']=step
        with (root/'validation.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        if metrics['selection_score']<best:
            best=metrics['selection_score'];torch.save(dict(model=model.state_dict(),objective=objective.state_dict(),spec=spec,identity=identity,step=step,metrics=metrics),root/'best.pt')
            torch.save(dict(encoder=model.encoder.state_dict(),architecture='neighborhood_mace_v1',spec=spec,identity=identity,step=step,irreps=str(__import__('src.training_methods.neighborhood_jepa.model',fromlist=['IRREPS']).IRREPS)),root/'encoder.pt')
            np.savez_compressed(root/'selection_predictions.npz',**predictions)
        if wb:wb.log({'epoch':step/per_epoch,**{f'validation/{k}':v for k,v in metrics.items() if k!='step'}})
        return metrics
    if step==0:validate();checkpoint()
    started=time.monotonic();stream=iter(loader(data,MixedBatches(data,config['batch_size'],config['seed'],step,updates),config['microbatch'],workers=config['loader_workers']))
    for packed,target in stream:
        if stop or time.time()>deadline-300:
            checkpoint();save_json(root/'status.json',dict(state='checkpointed',step=step,updates=updates));
            if wb:wb.finish()
            return False
        model.train();objective.train();target=move(target,'cuda');optimizer.zero_grad(set_to_none=True)
        warm=max(1,int(updates*.1));factor=min((step+1)/warm,1.) if step<warm else .01+.99*.5*(1+math.cos(math.pi*(step-warm)/(updates-warm)))
        for group in optimizer.param_groups:group['lr']=group['peak']*factor
        loss,terms=training_step(model,objective,packed,target,config['precision'])
        torch.nn.utils.clip_grad_norm_(encoder,1.,error_if_nonfinite=True);torch.nn.utils.clip_grad_norm_(heads,5.,error_if_nonfinite=True);optimizer.step();step+=1
        if step%16==0:
            record=dict(state='running',step=step,updates=updates,epoch=step/per_epoch,loss=loss,terms=terms,seconds=time.monotonic()-started)
            save_json(root/'status.json',record)
            with (root/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if wb:wb.log(dict(epoch=step/per_epoch,**{'loss/total':loss},**{f'loss/{k}':v for k,v in terms.items()},**{'optimization/encoder_lr':spec['encoder_lr']*factor}))
        if step%per_epoch==0 or step==updates:validate();checkpoint()
        elif step%128==0:checkpoint()
    selected=torch.load(root/'best.pt',map_location='cpu',weights_only=False)
    metrics=selected['metrics'];metrics.update(selected_step=selected['step'],total_updates=step)
    save_json(root/'metrics.json',metrics);save_json(root/'compilation.json',compilation_counters())
    write_metric_table(metrics,resolve_path(config['output']),family='neighborhood_jepa',name=spec['name'])
    save_json(root/'status.json',dict(state='complete',step=step,selection_score=best,selected_step=selected['step'],learned=best<metrics['training_mean_baseline']))
    if wb:wb.finish()
    return True
