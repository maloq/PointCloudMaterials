"""MACE256 patch reconstruction with full epochs and global distributed VCReg."""
import gc
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torch.distributed as dist

from src.data.fixed_cohort.protocol import digest,sha,write_json
from src.experiment_runner.metric_docs import check_metric_docs
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.project_runtime.paths import resolve_path
from src.research.crystal_vector import parallel
from src.research.crystal_vector.model import JointCrystalVector,vcreg
from src.research.encoder_context.geometry import graph
from src.research.equivariant_context.features import TypedPatchExport,prepared_frames
from src.research.supervised_onset.model import CapacityEncoder
from src.research.supervised_onset.tracking import tracked_run,update_training_summary
from .control_train import FAMILIES,table
from .data import config
from .rich_encoder import learning_rate
from .rich_descriptor_head import NormalizedResidualHead
from .rich_multimaterial_data import CachedPatches


def allocation_deadline():
    """Checkpoint before Slurm takes the GPU away; epochs control completion."""
    job=os.environ.get('SLURM_JOB_ID')
    if not job:return math.inf
    raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
    fields=dict(x.split('=',1) for x in raw.split() if '=' in x)
    return datetime.fromisoformat(fields['EndTime']).timestamp()-240


class RichPatchMACE(nn.Module):
    # Reuse the measured, checkpointed cuEq patch path; no context predictor.
    encode=JointCrystalVector.encode

    def __init__(self,c,outputs):
        super().__init__();self.config=c;self.patch=TypedPatchExport(CapacityEncoder(**c['encoder_config']))
        self.latent_dim=c['encoder_config']['code_dim'];f=c['vector_channels']
        self.vector_export=nn.Linear(2*self.encoder.channels,f,bias=False)
        self.register_buffer('scalar_mean',torch.zeros(self.latent_dim))
        self.register_buffer('scalar_scale',torch.ones(self.latent_dim))
        self.register_buffer('vector_scale',torch.ones(f))
        self.register_buffer('output_mask',torch.ones(outputs))
        self.readout=NormalizedResidualHead(self.latent_dim,outputs,c['descriptor_head'])

    @property
    def encoder(self):return self.patch.encoder

    def forward(self,positions):
        z,v=self.encode(positions)
        return dict(prediction=self.readout(z).float()*self.output_mask,
            state=z.float(),z=z[:,None].float(),v=v[:,None].float())


def upload(values,device):
    return {k:torch.from_numpy(v).to(device,non_blocking=True) for k,v in values.items()}


def initialize(c,data,device):
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    # cuEq allocates CUDA workspaces outside PyTorch's caching allocator.
    torch.cuda.set_per_process_memory_fraction(c['batch_search']['memory_fraction'],device)
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model=RichPatchMACE(c,len(data.mean)).to(device);model.output_mask.copy_(torch.as_tensor(data.active,device=device))
    ids=np.random.default_rng(c['seed']+17).choice(len(data),8192,replace=False)
    model.eval();pools=[]
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        for begin in range(0,len(ids),c['patch_chunk']):
            b=upload(data.batch(ids[begin:begin+c['patch_chunk']]),device)
            pools.append(model.encoder.pooled_graph(graph(b['positions'],model.encoder)).float())
        pools=torch.cat(pools)
        model.encoder.pooled_mean.copy_(pools.mean(0))
        model.encoder.pooled_scale.copy_(pools.std(0,unbiased=False).clamp_min(1e-4))
        b=upload(data.batch(ids[:1024]),device);_,v=model.encode(b['positions'])
        model.vector_scale.copy_(v.square().mean((0,2)).sqrt().clamp_min(1e-4))
    return model,ids


def compile_model(model,data,c,device):
    if c['runtime']['compile']:
        x=upload(data.batch(np.arange(c['patch_chunk'])),device)['positions']
        with torch.autocast('cuda',dtype=torch.bfloat16):compile_spatial_encoder(model.patch,graph(x,model.encoder))


def task_loss(out,target,weight):
    return .5*((out['prediction']-target).square()*weight).sum(1)+.5*math.log(2*math.pi)


def representation_loss(out,settings):
    value,stats=vcreg(out,settings)
    z=out['state'];count=z.new_tensor(float(len(z)))
    if parallel.world_size()>1:dist.all_reduce(count)
    mean=parallel.differentiable_sum(z.sum(0))/count
    centered=mean.square().mean()
    stats['scalar_mean_rms']=centered.detach().sqrt()
    stats['vcreg_loss']=value.detach()
    return value+settings['scalar_mean_weight']*centered,stats


def optimizer_for(model,c):
    encoder=list(model.encoder.parameters());ids={id(p) for p in encoder}
    return torch.optim.AdamW([
        dict(params=encoder,lr_scale=c['training']['encoder_lr_scale']),
        dict(params=[p for p in model.parameters() if id(p) not in ids],lr_scale=1.)],
        lr=c['training']['max_lr'],weight_decay=c['training']['weight_decay'],fused=True)


def set_lr(optimizer,lr):
    for group in optimizer.param_groups:group['lr']=lr*group['lr_scale']


def is_allocation_failure(error):
    # cuEq cudaMallocAsync failures are RuntimeError, not torch.OutOfMemoryError.
    return isinstance(error,torch.OutOfMemoryError) or (
        'cudaErrorMemoryAllocation' in str(error) and 'cudaMallocAsync' in str(error))


def probe(c):
    """Separate local numerical job: all families, real mixtures, consecutive updates."""
    data=CachedPatches(c,'train');device=torch.device('cuda:0');model,_=initialize(c,data,device)
    compile_model(model,data,c,device);model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=0.,weight_decay=c['training']['weight_decay'],fused=True)
    weight=torch.as_tensor(data.loss_weight,device=device);settings=c['batch_search']
    total=torch.cuda.get_device_properties(0).total_memory;budget=total*settings['memory_fraction']-settings['headroom_bytes']
    tech=resolve_path(c['output'])/'technical';records=[];rng=np.random.default_rng(c['seed']+37)
    def measure(n,repeats=1):
        gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();started=time.monotonic()
        b=out=loss=reg=stats=None;record=dict(per_gpu_batch=n,consecutive_steps=repeats)
        try:
            for _ in range(repeats):
                opt.zero_grad(set_to_none=True)
                b=upload(data.batch(rng.choice(len(data),n,replace=False)),device)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b['positions'])
                reg,stats=representation_loss(out,c['regularization']);loss=task_loss(out,b['target'],weight).mean()+reg
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip'],error_if_nonfinite=True)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite numerical loss')
                grads=[math.sqrt(sum(float(p.grad.square().sum()) for p in m.parameters() if p.grad is not None)) for m in model.encoder.interactions]
                if len(grads)!=3 or not all(math.isfinite(g) and g>0 for g in grads):raise ValueError(f'Missing spatial gradients: {grads}')
                if out['state'].shape!=(n,256):raise ValueError('Wrong exported state dimension')
                opt.step();torch.cuda.synchronize()
                record.update(loss=float(loss.detach()),gradient_norm=float(norm),interaction_gradient_norms=grads)
                b=out=loss=reg=stats=None
            peak=torch.cuda.max_memory_allocated()
            record.update(finite=True,peak_allocated_bytes=peak,within_budget=peak<=budget)
            free,_=torch.cuda.mem_get_info(device)
            record.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(),free_device_bytes=free)
            record['within_budget'] &= free>=settings['headroom_bytes']
        except RuntimeError as error:
            if not is_allocation_failure(error):raise
            record.update(finite=None,out_of_memory=True,within_budget=False,error=str(error))
        finally:
            del b,out,loss,reg,stats;opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
        record['seconds']=time.monotonic()-started;records.append(record)
        write_json(tech/'batch-probe-progress.json',dict(config_sha256=digest(c),records=records))
        print(json.dumps(record),flush=True);return record['within_budget']
    world=c['runtime']['gpus'];global_batch=c['training']['batch_size']
    if global_batch%world:raise ValueError('Requested global batch must divide the GPU count')
    local=global_batch//world
    if not measure(local,settings['consecutive_steps']):
        raise RuntimeError('The declared batch does not fit without changing the execution plan; inspect batch-probe-progress.json')
    steps=math.ceil(c['data']['training_rows']/global_batch)
    write_json(tech/'batch-candidate.json',dict(config_sha256=digest(c),dataset_identity=data.identity,
        per_gpu_batch=local,global_batch=global_batch,world_size=world,
        minimum_device_memory_bytes=total,gpu=torch.cuda.get_device_name(0),memory_budget_bytes=budget,
        updates_per_epoch=steps,total_updates=steps*c['training']['epochs'],train_rows=c['data']['training_rows'],
        encoder_parameters=sum(p.numel() for p in model.encoder.parameters()),total_parameters=sum(p.numel() for p in model.parameters()),
        records=records,online_runs_created=0,allocator='expandable_segments:True'))
    # A separate short engineering check exercises the repaired head at the
    # requested peak LR. Its weights are discarded before scientific training.
    opt=optimizer_for(model,c);learning=[];small=c['head_check']['batch_size']
    model.config=dict(c,patch_chunk=small)
    for update in range(c['head_check']['updates']):
        opt.zero_grad(set_to_none=True)
        b=upload(data.batch(rng.choice(len(data),small,replace=False)),device)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b['positions'])
        nll=task_loss(out,b['target'],weight).mean();reg,stats=representation_loss(out,c['regularization'])
        nll.backward(retain_graph=True)
        task_grads=[math.sqrt(sum(float(p.grad.double().square().sum()) for p in m.parameters() if p.grad is not None)) for m in model.encoder.interactions]
        spread=float(out['prediction'].detach().std(0,unbiased=False)[torch.as_tensor(data.active,device=device)].mean())
        if not all(math.isfinite(g) and g>1e-10 for g in task_grads) or not math.isfinite(spread) or spread<1e-7:
            raise FloatingPointError(f'Head check lost descriptor learning at step {update}: gradients={task_grads}, prediction_std={spread}')
        reg.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip'],error_if_nonfinite=True)
        opt.step()
        entry=dict(update=update+1,descriptor_nll=float(nll.detach()),regularization=float(reg.detach()),
            descriptor_interaction_gradient_norms=task_grads,prediction_std=spread,
            embedding_mean_rms=float(stats['scalar_mean_rms']),gradient_norm=float(norm))
        learning.append(entry);print(json.dumps(entry),flush=True)
        del b,out,nll,reg,stats
    write_json(tech/'optimization-check.json',dict(finite=True,consecutive_steps=settings['consecutive_steps'],
        batch=local,head_check_batch=small,head_check_peak_lr=c['training']['max_lr'],head_check=learning,
        scope='local numerical and short head-learning checks; all fitted diagnostic weights discarded',online_runs_created=0))
    data.close()


def select_subset(c):
    """Freeze the declared sample count; no timing-dependent data or stop rules."""
    tech=resolve_path(c['output'])/'technical';candidate=config(tech/'batch-candidate.json')
    if candidate['config_sha256']!=digest(c):raise ValueError('Numerical batch protocol changed')
    data=CachedPatches(c,'train');population=len(data);rows=c['data']['training_rows']
    if not 0<rows<=population:raise ValueError('Declared training rows exceed the available population')
    ids=np.sort(np.random.default_rng(c['seed']+71).choice(population,rows,replace=False))
    frozen_ids=tech/'training-pool-row-ids.npy'
    if frozen_ids.exists() and not np.array_equal(np.load(frozen_ids),ids):
        raise ValueError('Declared deterministic draw differs from the existing frozen sample IDs')
    data.select(ids)
    counts={material:int(sum(s['rows'] for s in data.shards if s['task']['material']==material))
        for material in sorted({s['task']['material'] for s in data.shards})}
    if set(counts)!={'Al','Mg','Ti','Ta'}:raise ValueError('Fitting subset lost a required material')
    np.save(frozen_ids,ids)
    transform=tech/'target-standardization.npz'
    if transform.exists():
        with np.load(transform) as previous:
            for k in ('mean','scale','active','loss_weight'):
                if not np.array_equal(previous[k],getattr(data,k)):raise ValueError(f'Frozen target transform changed: {k}')
    else:np.savez(transform,mean=data.mean,scale=data.scale,active=data.active,loss_weight=data.loss_weight)
    steps=math.ceil(len(data)/candidate['global_batch'])
    result=dict(candidate,train_rows=len(data),pool_rows=population,
        updates_per_epoch=steps,total_updates=c['training']['epochs']*steps,
        subset_sha256=sha(tech/'training-pool-row-ids.npy'),transform_sha256=sha(tech/'target-standardization.npz'),
        material_rows=counts,source_count=len({s['task']['source'] for s in data.shards}))
    write_json(tech/'batch-plan.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
    data.close()


def mse_metrics(mse,baseline):
    """Compare with the fitting-subset mean, not the evaluated population mean."""
    if not math.isfinite(mse) or not math.isfinite(baseline):
        raise FloatingPointError(f'Nonfinite descriptor MSE: model={mse}, training_mean={baseline}')
    relative=mse/baseline if baseline>1e-10 else None
    return dict(standardized_mse=mse,training_mean_mse=baseline,
        relative_mse_to_training_mean=relative,
        skill_over_training_mean=None if relative is None else 1-relative)


def descriptor_mse_metrics(data,error,baseline):
    """Aggregate squared errors before division; give each descriptor family equal mass."""
    families=np.array([v['family'] for v in data.columns])
    result={'all':mse_metrics(float(error@data.loss_weight),float(baseline@data.loss_weight))}
    for f in FAMILIES:
        mask=(families==f)&data.active
        result[f]=mse_metrics(float(error[mask].mean()),float(baseline[mask].mean()))
    return result


@torch.no_grad()
def training_diagnostics(out,target,embedding_gradient,data):
    prediction=out['prediction'].detach().double();target=target.double();n=len(prediction)
    moments=torch.stack((prediction.sum(0),prediction.square().sum(0),
        (prediction-target).square().sum(0),target.square().sum(0)))
    total_count,gradient_square=parallel.sum_values([n,float(embedding_gradient.double().square().sum())])
    first,second,error,baseline=parallel.sum_values(moments.double().cpu().numpy())/total_count
    spread=np.sqrt(np.maximum(second-first*first,0))
    metrics=descriptor_mse_metrics(data,error,baseline)
    result={'prediction_std':float(spread@data.loss_weight),
        'descriptor_embedding_gradient_rms':math.sqrt(gradient_square/(total_count*out['state'].shape[1]))}
    for family,values in metrics.items():
        prefix='' if family=='all' else family+'_'
        result[prefix+'relative_mse_to_training_mean']=values['relative_mse_to_training_mean']
    return result


@torch.no_grad()
def validation(model,data,batch,rank,world,device):
    model.eval();totals=np.zeros((2,len(data.columns)))
    ids=np.arange(rank,len(data),world)
    for begin in range(0,len(ids),batch):
        b=upload(data.batch(ids[begin:begin+batch]),device)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b['positions'])
        error=(out['prediction']-b['target']).square()
        # Standardized zero is the frozen training mean; held-out means are never fitted.
        totals[0]+=error.sum(0,dtype=torch.float64).cpu().numpy()
        totals[1]+=b['target'].square().sum(0,dtype=torch.float64).cpu().numpy()
    error,baseline=parallel.sum_values(totals)/len(data)
    metrics=descriptor_mse_metrics(data,error,baseline)
    scores={'descriptor_nll':.5*metrics['all']['standardized_mse']+.5*math.log(2*math.pi)}
    for family,values in metrics.items():
        prefix='' if family=='all' else family+'_'
        scores.update({prefix+k:v for k,v in values.items() if k!='skill_over_training_mean'})
    return scores


def train(c):
    session_started=time.monotonic()
    rank,world=parallel.initialize();device=torch.device('cuda',torch.cuda.current_device())
    root=resolve_path(c['output']);tech=root/'technical';plan=config(tech/'batch-plan.json')
    if (root/'analyses/descriptor-v1/technical/complete.json').exists():
        if world>1:dist.destroy_process_group()
        return
    if plan['config_sha256']!=digest(c):raise ValueError('Batch protocol changed')
    if plan['global_batch']!=c['training']['batch_size']:raise ValueError('Global batch differs from the recipe')
    if world not in c['runtime']['allowed_world_sizes'] or plan['global_batch']%world:
        raise ValueError('Execution GPU count cannot partition the frozen global batch')
    if torch.cuda.get_device_properties(device).total_memory<plan['minimum_device_memory_bytes']:raise ValueError('Resume GPU has less VRAM than the measured batch')
    data=CachedPatches(c,'train');selection=CachedPatches(c,'selection')
    if data.identity!=plan['dataset_identity']:raise ValueError('Training population changed')
    if sha(tech/'training-pool-row-ids.npy')!=plan['subset_sha256']:raise ValueError('Measured fitting subset changed')
    if sha(tech/'target-standardization.npz')!=plan['transform_sha256']:raise ValueError('Measured fitting transform changed')
    data.select(np.load(tech/'training-pool-row-ids.npy'))
    if len(data)!=c['data']['training_rows']:raise ValueError('Frozen fitting row count differs from the recipe')
    if plan['total_updates']!=plan['updates_per_epoch']*c['training']['epochs']:
        raise ValueError('Frozen update count differs from the epoch schedule')
    with np.load(tech/'target-standardization.npz') as a:
        for k in ('mean','scale','active','loss_weight'):
            if not np.array_equal(getattr(data,k),a[k]):raise ValueError(f'Subset transform mismatch: {k}')
    selection.use_transform(data)
    implementation=check_metric_docs(family=c['metric_family'])[c['metric_family']]['files']
    binding=dict(config=c,dataset=data.identity,batch_plan_sha256=sha(tech/'batch-plan.json'),implementation=implementation)
    if (tech/'identity.json').exists() and config(tech/'identity.json')!=binding:
        original=config(tech/'identity.json');amendment=config(tech/'protocol-amendment.json')
        if amendment['base_identity']!=digest(original) or amendment['binding']!=binding:
            raise ValueError('Protocol differs from the explicitly recorded training amendment')
        binding=original
    identity=digest(binding);study=SimpleNamespace(root=root,technical=tech,identity=identity,config=c)
    model,normalization_ids=initialize(c,data,device)
    optimizer=optimizer_for(model,c)
    epoch=step=update=0;best=float('inf');elapsed_before=0.
    if (tech/'last.pt').exists():
        saved=torch.load(tech/'last.pt',map_location=device,weights_only=False)
        if saved['identity']!=identity:raise ValueError('Checkpoint belongs to a different experiment')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        epoch,step,update,best=(saved[k] for k in ('epoch','step','update','best'))
        elapsed_before=saved['elapsed_fit_seconds']
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu());del saved
    parallel.broadcast_model(model);compile_model(model,data,c,device)
    batch=plan['global_batch'];local_batch=batch//world;steps=plan['updates_per_epoch'];n=len(data)
    weight=torch.as_tensor(data.loss_weight,device=device)
    if rank==0:
        write_json(tech/'identity.json',binding);np.save(tech/'normalization-rows.npy',normalization_ids)
        write_json(tech/'prediction-context.json',dict(
            encoder=dict(channels=256,message_passing_layers=3,exported_embedding=256,
                max_ell=model.encoder.max_ell,correlation=model.encoder.correlation,
                input='one current nearest-80 patch, fixed material-normalized coordinates',radius=8,edge_cutoff=5,halo=False,
                atom_channel='constant',species=False,material_id=False,history=False,motion=False,conditions=[]),
            predictor=dict(input='only the exported 256-D patch embedding',head=c['descriptor_head'],
                outputs=442,spatial_context=False,conditions=[]),
            relaxation=False,initialization='scratch',training_only_teacher='fixed geometry/bond-order/CNA/TDA descriptors of the same patch',
            coordinate_normalization=c['structural_dataset']['normalization'],
            sampling='fixed uniform subset of raw dynamic pool; every selected row once per epoch; no phase filter',
            training_rows=len(data),pool_rows=plan['pool_rows'],material_rows=plan['material_rows'],
            selection='Al-only structural selection sources',test='exact fixed Al64 calibration/test sample IDs',
            batch_size=batch,microbatch=batch,allowed_world_sizes=c['runtime']['allowed_world_sizes']))
        with (tech/'executions.jsonl').open('a') as f:
            f.write(json.dumps(dict(time=time.time(),owner=os.environ['PCM_RICH_OWNER'],
                job=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(device),world_size=world,
                global_batch=batch,per_gpu_batch=local_batch,resumed_update=update,
                elapsed_fit_seconds=elapsed_before,epochs=c['training']['epochs'],
                implementation=implementation))+'\n')
    def save(path):
        if rank:return
        tmp=path.with_suffix('.building.pt')
        torch.save(dict(identity=identity,model=model.state_dict(),encoder=model.encoder.state_dict(),encoder_config=c['encoder_config'],
            config=c,coordinate_normalization=c['structural_dataset']['normalization'],optimizer=optimizer.state_dict(),
            epoch=epoch,step=step,update=update,best=best,
            elapsed_fit_seconds=elapsed_before+time.monotonic()-session_started,
            implementation=implementation,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),tmp)
        tmp.replace(path)
    stop=allocation_deadline()
    requested=[False];signal.signal(signal.SIGUSR1,lambda *_:requested.__setitem__(0,True))
    collapsed_checks=0
    def handoff_requested():
        request=tech/'handoff-request.json'
        return request.exists() and config(request)['from']==os.environ['PCM_RICH_OWNER']
    tracking=tracked_run(study,'fit',job_type='encoder') if rank==0 else parallel.local_tracking()
    with tracking as log:
        if rank==0:log.config.update(c,allow_val_change=True)
        if rank==0:log.summary.update(dict(training_rows=n,validation_rows=len(selection),batch_size=batch,per_gpu_batch=local_batch,
            epochs_requested=c['training']['epochs'],parameters=plan['total_parameters'],encoder_parameters=plan['encoder_parameters'],
            mace_correlation=model.encoder.correlation,mace_max_ell=model.encoder.max_ell,
            encoder_patch_chunk=c['patch_chunk'],activation_checkpointing=c['activation_checkpointing'],
            descriptor_head=c['descriptor_head'],scalar_mean_regularization=c['regularization']['scalar_mean_weight'],
            checkpoint_selector='Al selection family-balanced descriptor Gaussian NLL',peak_head_learning_rate=c['training']['max_lr'],
            peak_encoder_learning_rate=c['training']['max_lr']*c['training']['encoder_lr_scale'],
            execution_gpus=world,
            data_relaxed=False,materials=['Al','Mg','Ti','Ta'],descriptor_outputs=442))
        while epoch<c['training']['epochs']:
            model.train();started=time.monotonic()
            stream=data.epoch(batch,epoch,c['seed'],start_batch=step,shards_per_block=c['loader']['shards_per_block'])
            def prepare(item):
                index,ids=item;values=data.batch(ids[rank::world])
                return index,len(ids),{k:torch.from_numpy(v).pin_memory() for k,v in values.items()}
            with prepared_frames(stream,prepare,workers=1,capacity=c['loader']['prefetch']) as ready:
                for index,count,host in ready:
                    handoff=handoff_requested()
                    if parallel.stop_requested(requested[0] or time.time()>stop or handoff):
                        save(tech/'last.pt')
                        if rank==0:write_json(tech/'state.json',dict(state='checkpointed',epoch=epoch,step=step,update=update,
                            reason='handoff' if handoff else ('signal' if requested[0] else 'allocation_end'),
                            elapsed_fit_seconds=elapsed_before+time.monotonic()-session_started))
                        break
                    if index!=step:raise ValueError('Shuffled streaming batch order changed')
                    b={k:v.to(device,non_blocking=True) for k,v in host.items()};optimizer.zero_grad(set_to_none=True)
                    lr=learning_rate(update,steps,c)
                    set_lr(optimizer,lr)
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b['positions'])
                    nll=task_loss(out,b['target'],weight).sum()*(world/count)
                    reg,stats=representation_loss(out,c['regularization']);reg=reg*min((update+1)/(steps*c['regularization']['warmup_epochs']),1.)
                    loss=nll+reg
                    if parallel.stop_requested(not bool(torch.isfinite(loss))):raise FloatingPointError(f'Nonfinite objective at update {update}')
                    logging=update==0 or (update+1)%8==0 or step+1==steps
                    if logging:
                        # Only the small head is differentiated here, excluding
                        # VCReg; a second MACE backward is not needed.
                        embedding_gradient=torch.autograd.grad(nll,out['state'],retain_graph=True)[0].detach()*(count/world)
                        diagnostics=training_diagnostics(out,b['target'],embedding_gradient,data)
                        del embedding_gradient
                    loss.backward();parallel.average_gradients(model)
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),c['training']['gradient_clip'],error_if_nonfinite=True)
                    optimizer.step();step+=1;update+=1
                    if logging:
                        vcreg_value=stats['vcreg_loss']*min(update/(steps*c['regularization']['warmup_epochs']),1.)
                        means=parallel.sum_values([float(nll.detach())/world,float(vcreg_value)/world,float(reg.detach()-vcreg_value)/world])
                        if rank==0:
                            record=dict(optimizer_update=update,**{'train/epoch':epoch+step/steps,'train/descriptor_nll':float(means[0]),
                                'train/vcreg':float(means[1]),'train/head_learning_rate':lr,
                                'train/embedding_mean_penalty':float(means[2]),
                                'train/encoder_learning_rate':lr*c['training']['encoder_lr_scale'],'train/gradient_norm':float(norm),
                                'train/patches_seen':epoch*n+min(step*batch,n)})
                            record.update({'train/'+k:v for k,v in diagnostics.items()})
                            record.update({'train/embedding_mean_rms':float(stats['scalar_mean_rms']),
                                'train/embedding_minimum_std':float(stats['scalar_minimum_std'])})
                            log.log(record)
                            with (tech/'training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                        collapsed=diagnostics['prediction_std']<1e-7 and diagnostics['descriptor_embedding_gradient_rms']<1e-10
                        collapsed_checks=collapsed_checks+1 if collapsed else 0
                        if collapsed_checks>=3:
                            save(tech/'last.pt')
                            if rank==0:write_json(tech/'state.json',dict(state='failed',reason='constant_predictor_and_missing_descriptor_gradient',update=update,diagnostics=diagnostics))
                            raise FloatingPointError(f'Descriptor learning collapsed at update {update}: {diagnostics}')
                    if update==1 or update%c['training']['save_every_updates']==0:save(tech/'last.pt')
                    if rank==0:write_json(tech/'state.json',dict(state='training',epoch=epoch,step=step,update=update,total_epochs=c['training']['epochs']))
                    del b,out,loss,nll,reg,stats,host
            if step<steps:break
            epoch+=1;step=0;scores=None
            if epoch%c['training']['validation_every_epochs']==0 or epoch==c['training']['epochs']:
                scores=validation(model,selection,local_batch,rank,world,device)
                if not all(math.isfinite(v) for v in scores.values() if v is not None):raise FloatingPointError('Nonfinite selection score')
                if scores['descriptor_nll']<best:
                    best=scores['descriptor_nll'];save(tech/'best.pt')
                    if rank==0:log.summary['checkpoint/selected_epoch']=epoch
            save(tech/'last.pt')
            if epoch%5==0:save(tech/f'epoch-{epoch:02d}.pt')
            if rank==0 and scores is not None:
                record=dict(optimizer_update=update,epoch=epoch,seconds=time.monotonic()-started,**{'validation/'+k:v for k,v in scores.items()})
                log.log(record)
                with (tech/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                print(json.dumps(record),flush=True)
        finished=epoch==c['training']['epochs']
        if finished and rank==0:
            write_json(tech/'complete.json',dict(identity=identity,epochs=epoch,updates=update,patch_visits=epoch*n,
                elapsed_fit_seconds=elapsed_before+time.monotonic()-session_started,
                best_sha256=sha(tech/'best.pt'),last_sha256=sha(tech/'last.pt')))
            log.summary['epochs_completed']=epoch
    if world>1:dist.destroy_process_group()
    if finished and rank==0:
        if time.time()>stop-3600:
            write_json(tech/'state.json',dict(state='checkpointed',phase='evaluation_pending',epoch=epoch,step=step,update=update))
        else:
            model.load_state_dict(torch.load(tech/'best.pt',map_location=device,weights_only=False)['model'])
            fields=export(model,data,c,study,local_batch,device)
            update_training_summary(study,'fit',fields,evaluation='multimaterial-local-rich-descriptors')
            write_json(tech/'state.json',dict(state='complete',epochs=epoch,updates=update))
    data.close();selection.close()


@torch.no_grad()
def export(model,training,c,study,batch,device):
    model.eval();root=study.root/'analyses/descriptor-v1';tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    scores=[];features=[];summary={};files={}
    audits=training.audit_ids(c['evaluation']['training_audit_per_material'],c['seed']+29)
    views=[('train_audit_'+material,training,ids) for material,ids in audits.items()]
    for role in ('selection','calibration','test'):
        data=CachedPatches(c,role);data.use_transform(training);views.append((role,data,np.arange(len(data))))
    for label,data,ids in views:
        error=np.zeros(len(data.mean));baseline=error.copy();target_mean=error.copy();n=len(ids)
        predictions=np.lib.format.open_memmap(tech/f'{label}-predictions.npy',mode='w+',dtype=np.float32,shape=(n,len(data.mean)))
        states=np.lib.format.open_memmap(tech/f'{label}-states.npy',mode='w+',dtype=np.float32,shape=(n,256))
        for begin in range(0,n,batch):
            b=upload(data.batch(ids[begin:begin+batch]),device)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b['positions'])
            pred=out['prediction'].cpu().numpy();y=b['target'].cpu().numpy().astype(float)
            predictions[begin:begin+len(y)]=pred;states[begin:begin+len(y)]=out['state'].cpu().numpy()
            error+=((pred.astype(float)-y)**2).sum(0);baseline+=(y*y).sum(0);target_mean+=y.sum(0)
        predictions.flush();states.flush();del predictions,states
        error/=n;baseline/=n;target_mean/=n
        metrics=descriptor_mse_metrics(data,error,baseline)
        for family,values in metrics.items():
            scores.append(dict(population=label,rows=n,family=family,**values))
            prefix='' if family=='all' else family+'_'
            summary.update({f'evaluation/{label}/{prefix}{k}':v
                for k,v in values.items() if k!='skill_over_training_mean'})
        summary[f'evaluation/{label}/balanced_gaussian_nll']=.5*metrics['all']['standardized_mse']+.5*math.log(2*math.pi)
        for j,col in enumerate(data.columns):
            variance=baseline[j]-target_mean[j]**2
            features.append(dict(population=label,feature=col['name'],family=col['family'],trained=bool(data.active[j]),
                **mse_metrics(float(error[j]),float(baseline[j])),
                rmse_descriptor_units=float(np.sqrt(error[j])*data.scale[j]),
                r2=1-float(error[j]/variance) if variance>1e-10 else None))
        np.save(tech/f'{label}-row-ids.npy',ids)
        if label in ('calibration','test'):
            sample_ids=np.concatenate([np.asarray(s['task']['sample_ids']) for s in data.shards])
            global_rows=np.concatenate([np.asarray(s['task']['fixed_global_rows']) for s in data.shards])
            if len(sample_ids)!=n:raise ValueError('Held-out sample identities do not match predictions')
            np.savez(tech/f'{label}-fixed-ids.npz',sample_ids=sample_ids,global_rows=global_rows)
        for name in (f'{label}-predictions.npy',f'{label}-states.npy',f'{label}-row-ids.npy'):
            files[name]=sha(tech/name)
        if data is not training:data.close()
    table(root,'scores',scores,family=c['metric_family']);table(root,'features',features,family=c['metric_family'])
    write_json(tech/'complete.json',dict(identity=study.identity,checkpoint_sha256=sha(study.technical/'best.pt'),files=files,summary=summary))
    return summary
