"""Random target-population batches and resumable joint localization training."""
import json
import math
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch

from src.data.fixed_cohort.protocol import digest,sha,write_json as store_json
from src.project_runtime.paths import resolve_path
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.research.encoder_context.geometry import graph
from src.research.supervised_onset.tracking import tracked_run
from src.research.spatial_distance.model import capped_mean,cdf
from .data import ResidentContexts
from .sampling import RandomBatches
from .model import JointCrystalVector,objective,vcreg
from . import parallel


def initialize(c,device):
    path=resolve_path(c['initial_encoder'])
    if sha(path)!=c['initial_encoder_sha256']:raise ValueError('Parent checkpoint changed')
    saved=torch.load(path,map_location=device,weights_only=False)
    if saved['config']['protocol']!='joint_mace_distance_early_v1' or saved['epoch']!=12:
        raise ValueError('Expected completed snapshot CD-MACE128 parent')
    for k,v in c['encoder'].items():
        if saved['encoder_config'][k]!=v:raise ValueError(f'Wrong parent architecture {k}')
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model=JointCrystalVector(saved['encoder_config'],c).to(device)
    model.encoder.load_state_dict(saved['encoder'],strict=True)
    return model,saved['encoder_config']


@torch.no_grad()
def calibrate(model,data,c):
    # Fixed training-only feature scales; never batch-normalize Cartesian axes.
    rng=np.random.default_rng(c['seed']);ids=rng.choice(data.split['train'],256,replace=True,p=data.weights['train'])
    patches=torch.unique(data.indices[torch.as_tensor(ids,device=data.indices.device)].flatten())[:4096]
    model.eval()
    with torch.autocast('cuda',dtype=torch.bfloat16):z,v=model.encode(data.positions[patches])
    model.scalar_mean.copy_(z.mean(0));model.scalar_scale.copy_(z.std(0,unbiased=False).clamp_min(1e-4))
    # One scalar scale per vector channel; component directions remain coupled.
    model.vector_scale.copy_(v.square().mean((0,2)).sqrt().clamp_min(1e-4))


def compile_model(model,data,c):
    if c['runtime']['compile']:
        with torch.autocast('cuda',dtype=torch.bfloat16):
            compile_spatial_encoder(model.patch,graph(data.positions[:c['patch_chunk']],model.encoder))


def deadline(c):
    stop=time.time()+c['runtime']['hours']*3600
    job=os.environ.get('SLURM_JOB_ID')
    if job:
        raw=subprocess.check_output(['scontrol','show','job',job,'-o'],text=True)
        fields=dict(x.split('=',1) for x in raw.split() if '=' in x)
        from datetime import datetime
        end=datetime.fromisoformat(fields['EndTime']).timestamp()
        stop=min(stop,end-240)
    return stop


@torch.no_grad()
def validate(model,data,c,directional):
    rank=torch.distributed.get_rank() if parallel.world_size()>1 else 0
    ids=data.split['selection'][rank::parallel.world_size()];weights=data.weights['selection'][rank::parallel.world_size()];model.eval()
    totals=np.zeros(7);valid_mass=0.
    for begin in range(0,len(ids),c['microbatch']):
        index=ids[begin:begin+c['microbatch']];b=data.batch(index)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b)
        loss=objective(out,b,c['loss'],directional)
        risk=cdf(out['parts'],b['distance'].new_tensor([20.,32.]))
        squared=(capped_mean(out['parts'],64)-b['distance'].clamp(max=64)).square()
        brier=(risk-(b['distance'][:,None]<=b['distance'].new_tensor([20.,32.])).to(risk.dtype)).square()
        values=torch.stack((loss['objective'],loss['distance_nll'],loss['proximity_log_loss'],loss['direction_nll'],squared,brier[:,0],brier[:,1]),-1)
        w=weights[begin:begin+len(index)];totals+=(values.double().cpu().numpy()*w[:,None]).sum(0)
        valid_mass+=float(w@b['valid'].cpu().numpy())
    reduced=parallel.sum_values(np.r_[totals,valid_mass]);totals=reduced[:7];valid_mass=float(reduced[7])
    return dict(predictive_objective=float(totals[0]),distance_nll=float(totals[1]),proximity_log_loss=float(totals[2]),
        direction_nll_on_valid=float(totals[3]/valid_mass) if valid_mass else None,
        capped_distance_rmse_A=float(np.sqrt(totals[4])),brier_within20A=float(totals[5]),brier_within32A=float(totals[6]),
        direction_valid_mass=valid_mass)


def identity(c,variant,data):
    repo=Path(__file__).resolve().parents[3]
    files=list(Path(__file__).parent.glob('*.py'))+[repo/p for p in (
        'src/models/encoders/spatial_mace.py','src/models/encoders/mace_backend.py',
        'src/research/equivariant_context/model.py','src/research/equivariant_context/features.py',
        'src/research/supervised_onset/model.py','src/research/encoder_context/geometry.py',
        'src/research/distance_encoder/model.py','src/research/spatial_distance/model.py')]
    return dict(config=c,variant=variant,dataset_identity=data.identity,
        implementation={str(p.relative_to(repo)):sha(p) for p in files})


def run(config_path,variant):
    c=json.loads(Path(config_path).read_text());rank,world=parallel.initialize();primary=rank==0
    if c['batch_size']%world:raise ValueError('Global batch must divide evenly across GPUs')
    if c['runtime'].get('gpus',1)!=world:raise ValueError('Launch must match the declared GPU count')
    def write_json(path,value):
        if primary:store_json(path,value)
    if variant not in c['variants']:raise ValueError(variant)
    if c['batch_size']!=c['microbatch']:raise ValueError('VCReg uses a full context batch; use patch activation checkpointing for memory')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    device=torch.device('cuda');root=resolve_path(c['output'])/variant;tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    if (tech/'complete.json').exists():return True
    started=time.monotonic();stop=deadline(c);data=ResidentContexts(c,device)
    binding=identity(c,variant,data);run_identity=digest(binding)
    old=tech/'identity.json'
    if old.exists() and json.loads(old.read_text())!=binding:raise ValueError('Training identity changed; use a fresh run')
    write_json(old,binding)
    model,encoder_config=initialize(c,device)
    if c['sampling']['method']!='independent_random_target_population':raise ValueError('Expected direct random population sampling')
    sampler=RandomBatches(data,c['batch_size'])
    if 'updates_per_epoch' in c['training']:sampler.steps=c['training']['updates_per_epoch']
    encoder_params=list(model.encoder.parameters())+list(model.vector_export.parameters())
    encoder_ids={id(p) for p in encoder_params}
    head_params=[p for p in model.parameters() if id(p) not in encoder_ids]
    optimizer=torch.optim.AdamW([dict(params=encoder_params,lr=c['training']['encoder_lr']),
        dict(params=head_params,lr=c['training']['head_lr'])],weight_decay=c['training']['weight_decay'],fused=True)
    epoch=step=update=0;best=float('inf');rng=np.random.default_rng(np.random.SeedSequence([c['seed'],0]))
    if (tech/'last.pt').exists():
        saved=torch.load(tech/'last.pt',map_location=device,weights_only=False)
        if saved['identity']!=run_identity:raise ValueError('Checkpoint identity mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        epoch,step,update,best=saved['epoch'],saved['step'],saved['update'],saved['best']
        rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state(saved['cuda_rng'].cpu())
    elif primary:calibrate(model,data,c)
    parallel.broadcast_model(model)
    compile_model(model,data,c)
    directional=variant!='distance_only';regularized=variant=='distance_direction_vcreg'
    inputs=dict(encoder=dict(geometry='single observed nearest-80 patch, radius 8, edge cutoff 5, two interactions',
        trainable=True,scalar_export=128,vector_channels=c['predictor']['field_channels'],species=False,history=False,motion=False,conditions=[]),
        predictor=dict(type='vector_messages',trainable=True,shared_encoder_for_all_patches=True,patches=25,
        sampling='distance-only farthest covering; 12 in (4,14] and 12 in (14,24]',max_support_A=32,
        fields=['scalar embeddings','equivariant vectors','actual relative patch positions'],central_bypass=False,conditions=[]),
        relaxation=False,training_only_teacher=None,label_history='past-only established-crystal mask, not a model input')
    write_json(tech/'prediction-context.json',inputs)
    interface=c.get('target',{}).get('kind')=='crystal_interface_layer'
    liquid=c.get('observation_filter')=='liquid_no_visible_crystal'
    if interface:
        inputs['target']=c['target']
        inputs['label_history']='past-only established crystal plus current spatial interface graph; no label fields are inputs'
        write_json(tech/'prediction-context.json',inputs)
    if c.get('observation_filter')=='no_visible_interface':
        inputs['training_and_selection_population']='original population conditioned on no visible interface in any patch'
        inputs['predictor']['visibility_input']=False
        write_json(tech/'prediction-context.json',inputs)
        write_json(tech/'observation-filter.json',data.view_audit)
    if liquid:
        inputs['target']=c['target']
        inputs['training_and_selection_population']='liquid query; no established crystal in any consumed patch; nearest crystal exists elsewhere'
        inputs['predictor']['visibility_input']=False
        inputs['predictor']['crystal_existence_input']=False
        inputs['label_history']='past-confirmed established-crystal mask for target/eligibility only; subcritical liquid ordering remains observable'
        write_json(tech/'prediction-context.json',inputs);write_json(tech/'observation-filter.json',data.view_audit)
    sampling=sampler.audit(data.meta['distance'],data.meta['inside_crystal'] if interface or liquid else None)
    if c.get('observation_filter')=='no_visible_interface':
        sampling.update(target='original half-fixed half-uniform, source-weighted population conditioned on no visible interface',
            uses_labels_for_sampling=True,label_use='interface visibility for eligibility only; no distance quotas',
            updates_per_epoch=sampler.steps)
    if liquid:
        sampling.update(target=c['sampling']['target_population'],uses_labels_for_sampling=True,
            label_use='liquid/crystal visibility and existence eligibility only; no distance quotas',updates_per_epoch=sampler.steps)
    write_json(tech/'sampling.json',sampling)
    def save(path):
        if not primary:return
        temporary=path.with_suffix('.building.pt')
        torch.save(dict(identity=run_identity,model=model.state_dict(),encoder=model.encoder.state_dict(),encoder_config=encoder_config,
            optimizer=optimizer.state_dict(),config=c,variant=variant,epoch=epoch,step=step,update=update,best=best,
            numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()),temporary)
        temporary.replace(path)
    name='CIV-MACE128' if interface else 'CDV-MACE128'
    if c.get('observation_filter')=='no_visible_interface':name='CIV-MACE128-Unseen'
    if liquid:name='LCD-MACE128-VC'
    tracking=SimpleNamespace(config=dict(c,wandb=dict(c['wandb'],display_name=f'{name} | {variant} | random spatial Al64')),
        root=root,technical=tech,identity=run_identity)
    with (tracked_run(tracking,'joint',job_type='encoder') if primary else parallel.local_tracking()) as run:
        run.summary.update({'prediction_context':inputs,'model/parameters':sum(p.numel() for p in model.parameters()),
            'data/train_rows':len(data.split['train']),'data/selection_rows':len(data.split['selection']),
            'training/batch_size':c['batch_size'],'training/microbatch':c['microbatch'],
            'training/gpus':world,'training/per_gpu_batch':c['batch_size']//world,
            'training/vcreg_statistics':'global differentiable moments; averaged parameter gradients',
            'sampling/expected_near_0_8_per_batch':sampling['coverage']['near_0_8']['expected_per_batch'],
            'sampling/expected_near_0_20_per_batch':sampling['coverage']['near_0_20']['expected_per_batch'],
            'sampling/empty_near_0_8_probability':sampling['coverage']['near_0_8']['empty_batch_probability'],
            'sampling/fixed_quotas':False,'sampling/importance_corrected':False,
            'checkpoint/selection':'configured-population predictive likelihood; epochs 12–16; VCReg excluded'})
        total_updates=sampler.steps*c['training']['epochs']
        while epoch<c['training']['epochs']:
            epoch_started=time.monotonic();model.train()
            while step<sampler.steps:
                if update%64==0 and parallel.stop_requested(time.time()>stop):
                    save(tech/'last.pt');write_json(tech/'state.json',dict(state='checkpointed',epoch=epoch,step=step,update=update))
                    run.summary['training/status']='checkpointed_for_resume';return False
                global_ids=sampler.batch(rng);ids=global_ids[rank::world];b=data.batch(ids)
                if c.get('observation_filter')=='no_visible_interface' and data.meta['visible_context'][ids].any():
                    raise ValueError('Visible interface entered an excluded training batch')
                if liquid and not data.eligibility[ids].all():
                    raise ValueError('Non-liquid, crystal-visible or crystal-absent example entered localization training')
                optimizer.zero_grad(set_to_none=True)
                warm=min((update+1)/c['training']['warmup_updates'],1.)
                factor=warm*(.05+.95*.5*(1+math.cos(math.pi*update/total_updates)))
                for group,base in zip(optimizer.param_groups,(c['training']['encoder_lr'],c['training']['head_lr'])):group['lr']=base*factor
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b)
                loss=objective(out,b,c['loss'],directional)
                task=loss['objective'].mean();penalty=task.new_zeros(());stats={}
                if regularized:
                    penalty,stats=vcreg(out,c['regularization'])
                    penalty=penalty*min((update+1)/c['regularization']['warmup_updates'],1.)
                value=task+penalty
                if not torch.isfinite(value):raise FloatingPointError(f'Nonfinite objective {variant}/{epoch}/{step}')
                value.backward();parallel.average_gradients(model);norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                optimizer.step();update+=1;step+=1
                if update%c['training']['log_every']==0:
                    means=parallel.sum_values([float(task.detach()),float(loss['distance_nll'].mean().detach()),float(loss['direction_nll'].mean().detach())])/world
                    global_distance=data.meta['distance'][global_ids]
                    record=dict(optimizer_update=update,**{'train/epoch':epoch+step/sampler.steps,
                        'train/predictive_objective':float(means[0]),'train/distance_nll':float(means[1]),
                        'train/direction_nll':float(means[2]) if directional else None,
                        'train/batch_zero_distance_count':int((global_distance==0).sum()),
                        'train/batch_inside_crystal_count':int(data.meta['inside_crystal'][global_ids].sum()) if interface else int((global_distance==0).sum()),
                        'train/batch_near_0_8_count':int(((global_distance>0)&(global_distance<=8)).sum()),
                        'train/batch_near_0_20_count':int(((global_distance>0)&(global_distance<=20)).sum()),
                        'train/vcreg':float(penalty.detach()),'train/gradient_norm':float(norm),
                        'train/encoder_lr':optimizer.param_groups[0]['lr'],'train/predictor_lr':optimizer.param_groups[1]['lr'],
                        'train/peak_vram_GiB':torch.cuda.max_memory_allocated()/2**30},
                        **{f'train/{k}':float(v) for k,v in stats.items()})
                    run.log(record)
                    if primary:
                        with (tech/'training.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                    write_json(tech/'state.json',dict(state='training',epoch=epoch+step/sampler.steps,update=update,total_updates=total_updates))
                    if primary:print(json.dumps(dict(variant=variant,**record)),flush=True)
                if update%c['training']['save_every']==0:save(tech/'last.pt')
            # Keep the current epoch/step checkpoint until validation and selection finish.
            save(tech/'last.pt')
            if parallel.stop_requested(time.time()>stop-120):
                write_json(tech/'state.json',dict(state='checkpointed_before_validation',epoch=epoch,step=step,update=update));return False
            scores=validate(model,data,c,directional)
            if not np.isfinite(scores['predictive_objective']):raise FloatingPointError('Nonfinite selection likelihood')
            epoch+=1;step=0;rng=np.random.default_rng(np.random.SeedSequence([c['seed'],epoch]))
            if epoch>=c['training']['minimum_selection_epoch'] and scores['predictive_objective']<best:
                best=scores['predictive_objective'];save(tech/'best.pt')
                run.summary['checkpoint/selected_epoch']=epoch;run.summary['checkpoint/validation_objective']=best
            record=dict(optimizer_update=update,**{f'validation/{k}':v for k,v in scores.items()},
                **{'train/epoch':epoch,'train/epoch_seconds':time.monotonic()-epoch_started})
            run.log(record)
            if primary:
                with (tech/'validation.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
            save(tech/'last.pt')
            if primary:print(json.dumps(dict(variant=variant,**record)),flush=True)
        run.summary['training/status']='complete'
        if world>1:torch.distributed.barrier()
        write_json(tech/'complete.json',dict(identity=run_identity,epochs=epoch,updates=update,seconds=time.monotonic()-started,
            best_sha256=sha(tech/'best.pt'),dataset_identity=data.identity))
        write_json(tech/'state.json',dict(state='training_complete',epoch=epoch,update=update))
    return True
