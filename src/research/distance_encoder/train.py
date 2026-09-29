"""Two-GPU end-to-end distance likelihood training with resident geometry."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from src.data.fixed_cohort.protocol import digest,sha,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.research.encoder_context.geometry import graph
from src.research.supervised_onset.tracking import tracked_run
from src.research.spatial_approach.evaluate import csv_rows
from .data import ResidentDataset
from .model import DistanceEncoder,loss_terms,cdf,capped_mean
from .regularization import variance_covariance


def load_model(c,device):
    path=resolve_path(c['initial_encoder'])
    if sha(path)!=c['initial_encoder_sha256']:raise ValueError('Initial encoder checkpoint changed')
    saved=torch.load(path,map_location=device,weights_only=False)
    if c['protocol'] in ('joint_mace_distance_history_v1','joint_mace_distance_dense_history_v1'):
        from .history import HistoryDistanceEncoder
        encoder_config=saved['encoder_config']
        for key,value in c['encoder'].items():
            if encoder_config[key]!=value:raise ValueError(f'Encoder configuration mismatch: {key}')
        model=HistoryDistanceEncoder(encoder_config,c['history']).to(device)
        if c['initialization']=='distance_checkpoint':
            if saved['epoch']!=12 or saved['config']['protocol']!='joint_mace_distance_early_v1':
                raise ValueError('Expected the declared completed CD-MACE128 distance checkpoint')
            model.encoder.load_state_dict(saved['encoder'],strict=True)
            model.head.distance.load_state_dict(saved['head'],strict=True)
        elif c['initialization']=='native_al_onset_encoder':
            if saved['arm']['input']!='hot' or saved['config']['objective']!='hazard_nll' or saved['config']['output']!='${storage:analysis}/encoder_context/multimaterial256-b1024-20260925/scratch/base-hot':
                raise ValueError('Expected the native Al-only observed onset checkpoint; external held-out families must be unseen')
            model.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.')},strict=True)
        elif c['initialization']=='history_distance_checkpoint':
            if saved['epoch']<12 or saved['config']['protocol']!='joint_mace_distance_dense_history_v1':
                raise ValueError('Expected a completed dense-history distance checkpoint')
            if saved['config']['history']['mode']!='real' or c['history']['mode']!='real':
                raise ValueError('Material adaptation requires real history in both parent and child')
            if len(saved['config']['history']['offsets_ps'])!=len(c['history']['offsets_ps']):
                raise ValueError('Parent and fine-tuning history lengths differ')
            model.load_state_dict(saved['model'],strict=True)
        else:raise ValueError(f'Unknown history initialization: {c["initialization"]}')
        return model,encoder_config
    if saved['arm']['input']!='hot' or saved['config']['objective']!='hazard_nll':
        raise ValueError('Expected the declared observed likelihood-trained initialization')
    encoder_config=saved['encoder_config']
    for key,value in c['encoder'].items():
        if encoder_config[key]!=value:raise ValueError(f'Encoder configuration mismatch: {key}')
    model=DistanceEncoder(encoder_config).to(device)
    model.encoder.load_state_dict({k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.')},strict=True)
    return model,encoder_config


@torch.no_grad()
def validate(model,data,c,rank,world,device):
    model.eval();size=c['microbatch'];totals=torch.zeros(7,device=device,dtype=torch.float64)
    for start in range(rank*size,data.n,world*size):
        stop=min(start+size,data.n);positions,distance,weight=data.batch(slice(start,stop))
        geometry=graph(positions.reshape(-1,80,3),model.encoder)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['runtime']['precision']=='bf16'):
            parts=model(geometry)
        objective,nll,early=loss_terms(parts,distance,c['loss'])
        risk=cdf(parts,distance.new_tensor([20.,32.]))
        error=(capped_mean(parts,c['loss']['distance_cap'])-distance.clamp(max=c['loss']['distance_cap'])).square()
        observed=(distance[:,None]<=distance.new_tensor([20.,32.])).to(risk.dtype)
        brier=(risk-observed).square()
        values=torch.stack((objective,nll,early,error,brier[:,0],brier[:,1],torch.ones_like(weight)),-1)
        totals+=(values*weight[:,None]).double().sum(0)
    dist.all_reduce(totals)
    values=(totals[:6]/totals[6]).cpu().tolist()
    return dict(objective=values[0],distance_nll=values[1],early_log_loss=values[2],
        capped_mean_rmse=math.sqrt(values[3]),brier_within20=values[4],brier_within32=values[5])


def run(config_path):
    c=json.loads(Path(config_path).read_text());rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    accumulation=c.get('accumulation_steps',1)
    if world!=c['world_size'] or c['batch_size']!=world*c['microbatch']*accumulation:
        raise ValueError('Global batch must equal world_size * microbatch * accumulation_steps')
    if accumulation>1 and (c['regularization']['variance_weight'] or c['regularization']['covariance_weight']):
        raise ValueError('Global-moment regularization is not defined across accumulated microbatches')
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(1)
    torch.set_float32_matmul_precision(c['runtime']['float32_matmul_precision'])
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    root=result_folders(resolve_path(c['output']));technical=root/'technical'
    started=time.monotonic();deadline=time.time()+c['runtime']['hours']*3600
    label_root=resolve_path(c['labels']['root'])
    while not (label_root/'manifest.json').exists():
        failed=technical/'labels-failed.json'
        if failed.exists():raise RuntimeError('Label preparation failed; inspect labels-failed.json')
        if time.time()>deadline-600:raise TimeoutError('Waiting for distance labels')
        time.sleep(15)
    torch.manual_seed(c['seed']);torch.cuda.manual_seed_all(c['seed'])
    model,encoder_config=load_model(c,device)
    repo=Path(__file__).resolve().parents[3]
    files=list(Path(__file__).parent.glob('*.py'))+[repo/p for p in (
        'src/models/encoders/spatial_mace.py','src/models/encoders/mace_backend.py',
        'src/research/encoder_context/geometry.py','src/research/spatial_distance/model.py')]
    binding=dict(config=c,label_manifest_sha256=sha(label_root/'manifest.json'),encoder_config=encoder_config,
        implementation={str(p.relative_to(repo)):sha(p) for p in files})
    if c['protocol']=='joint_mace_distance_dense_history_v1':
        binding['dense_manifest_sha256']=sha(resolve_path(c['dense_history']['root'])/'manifest.json')
    identity=digest(binding)
    if rank==0:
        target=technical/'identity.json'
        if target.exists() and json.loads(target.read_text())!=binding:raise ValueError('Training contract changed')
        if not target.exists():write_json(target,binding)
        write_json(technical/'state.json',dict(state='loading',identity=identity))
    # Immutable normalized coordinates fit in ~12 GiB/GPU. There is no per-step
    # filesystem access, worker shuffle stall, or cache of trainable features.
    if 'material_finetune' in c:
        from .material_data import MaterialHistoryDataset
        dataset=MaterialHistoryDataset
    elif c['protocol']=='joint_mace_distance_dense_history_v1':
        from .dense_history import DenseHistoryDataset
        dataset=DenseHistoryDataset
    elif c['protocol']=='joint_mace_distance_history_v1':
        from .history import HistoryDataset
        dataset=HistoryDataset
    else:
        dataset=ResidentDataset
    train=dataset(c,'train',device);selection=dataset(c,'selection',device)
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=c['training']['encoder_lr']),
        dict(params=model.head.parameters(),lr=c['training']['head_lr'])],weight_decay=c['training']['weight_decay'],fused=True)
    steps=math.ceil(train.n/c['batch_size']);total=steps*c['training']['epochs'];update=0;best=float('inf')
    last=technical/'last.pt'
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False)
        if saved['identity']!=identity:raise ValueError('Resume contract mismatch')
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);update=saved['update'];best=saved['best']
    if c['runtime']['compile']:
        example=train.batch(slice(0,c['microbatch']))[0].reshape(-1,80,3)
        compile_spatial_encoder(model.encoder,graph(example,model.encoder))
    ddp=DistributedDataParallel(model,device_ids=[local],broadcast_buffers=False,static_graph=True,gradient_as_bucket_view=True)
    context=dict(encoder=dict(trainable=True,channels=128,embedding=128,interaction_blocks=2,
        geometry='nearest80 within normalized radius8; edge cutoff5; no halo',atom_channel='one constant channel',
        materials=' '.join(train.material_names)+'; fixed material length normalization in preprocessing only',
        history=False,velocities=False,conditions=[],material_id_input=False,scale_input=False,relaxation=False),
        predictor=dict(input='one 128-dimensional exported embedding',trainable=True,conditions=[],training_only_teacher=None),
        loss=c['loss'],representation_regularization=c['regularization'],distance_units='Al-equivalent Angstrom; native Al unchanged',
        initialization='earlier observed Al onset-likelihood encoder; fresh distance head')
    if c['protocol'] in ('joint_mace_distance_history_v1','joint_mace_distance_dense_history_v1'):
        context['predictor'].update(input=f'{len(c["history"]["offsets_ps"])} ordered 128-dimensional MACE exports and adjacent differences',
            history=c['history'],state_dimension=128,explicit_time_inputs=False,
            history_gradients='encoder recomputed and differentiated at every observed frame',
            surrounding_patches=False)
        context['initialization']=c['initialization']
        context['history_population']=dict(train=train.history_metadata,selection=selection.history_metadata)
        if 'material_finetune' in c:
            context['material_adaptation']=c['material_finetune']
    tracking=SimpleNamespace(config=dict(c,wandb=dict(c['wandb'],display_name=c['encoder_name']+' | distance + early detection')),
                             root=root,technical=technical,identity=identity)
    if rank==0:write_json(technical/'prediction-context.json',context)
    def save(path):
        if rank!=0:return
        temporary=path.with_suffix('.building.pt')
        torch.save(dict(identity=identity,model=model.state_dict(),encoder=model.encoder.state_dict(),head=model.head.state_dict(),
            encoder_config=encoder_config,optimizer=optimizer.state_dict(),update=update,epoch=update/steps,
            best=best,config=c,label_identity=train.identity,parameters=sum(p.numel() for p in model.parameters())),temporary)
        temporary.replace(path)
    record_folder=root/'analyses/learning-v1'
    history=[json.loads(line) for line in (technical/'validation.jsonl').read_text().splitlines()] if (technical/'validation.jsonl').exists() else []
    manager=tracked_run(tracking,'joint-mace-distance',job_type='encoder') if rank==0 else nullcontext(None)
    try:
        with manager as online:
            if rank==0:
                online.summary.update({'data/train_rows':train.n,'data/validation_rows':selection.n,
                    'data/train_material_counts':train.counts,'data/label_identity':train.identity,'prediction_context':context,
                    'data/target_counts_by_material':train.target_counts,
                    'model/encoder_parameters':sum(p.numel() for p in model.encoder.parameters()),
                    'model/head_parameters':sum(p.numel() for p in model.head.parameters()),
                    'training/global_batch':c['batch_size'],'training/per_gpu_batch':c['microbatch'],
                    'training/accumulation_steps':accumulation,
                    'training/epochs_requested':c['training']['epochs'],'encoder/trainable':True,
                    'checkpoint/selection_rule':'minimum declared distance+early likelihood after epoch 12',
                    'data/coordinate_storage':'resident float32 GPU tensors, all dynamic rows; no learned-feature cache'})
                family=c.get('metric_family','distance_encoder_history' if c['protocol']=='joint_mace_distance_history_v1' else 'distance_encoder')
                snapshot_metric_docs(record_folder,family)
            dist.barrier()
            running=torch.zeros(4,device=device,dtype=torch.float64)
            regularizer_totals=torch.zeros(3,device=device);tick=time.monotonic()
            for epoch in range(update//steps,c['training']['epochs']):
                # Same CPU permutation on both ranks makes exact row coverage
                # independent of device RNG streams. Only the final batch pads.
                order=torch.from_numpy(np.random.default_rng(np.random.SeedSequence([c['seed'],epoch])).permutation(train.n)).to(device)
                skip=update%steps if epoch==update//steps else 0
                ddp.train()
                for step in range(skip,steps):
                    stop=torch.zeros((),device=device,dtype=torch.int32)
                    if step%32==0:
                        if rank==0:stop.fill_(int(time.time()>deadline-300))
                        dist.broadcast(stop,0)
                    if bool(stop):
                        save(last)
                        if rank==0:
                            write_json(technical/'state.json',dict(state='checkpointed_time_limit',identity=identity,update=update,epoch=update/steps))
                            online.summary.update({'training/status':'checkpointed_time_limit','training/completed_epochs':update/steps})
                        dist.barrier();return
                    start=step*c['batch_size'];actual=min(c['batch_size'],train.n-start)
                    fraction=.05+.95*.5*(1+math.cos(math.pi*update/total))
                    warm=min((update+1)/c['training']['warmup_updates'],1.)
                    for group,base in zip(optimizer.param_groups,(c['training']['encoder_lr'],c['training']['head_lr'])):group['lr']=base*warm*fraction
                    optimizer.zero_grad(set_to_none=True)
                    for micro in range(accumulation):
                        lo=start+(micro*world+rank)*c['microbatch'];hi=min(lo+c['microbatch'],train.n)
                        ids=order[lo:hi];real=len(ids)
                        if real<c['microbatch']:ids=torch.cat((ids,torch.zeros(c['microbatch']-real,device=device,dtype=torch.int64)))
                        x,distance,weight=train.batch(ids)
                        if real<c['microbatch']:weight=weight.clone();weight[real:]=0
                        geometry=graph(x.reshape(-1,80,3),model.encoder)
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=c['runtime']['precision']=='bf16'):
                            parts,z=ddp(geometry,return_embedding=True)
                        objective,nll,early=loss_terms(parts,distance,c['loss'])
                        loss=(objective*weight).sum()*world/actual
                        if c['regularization']['variance_weight'] or c['regularization']['covariance_weight']:
                            regularizer,variance,covariance=variance_covariance(z,weight,c['regularization'])
                        else:
                            regularizer=variance=covariance=z.new_zeros(())
                        regularizer_warm=min((update+1)/c['regularization']['warmup_updates'],1.)
                        loss=loss+regularizer_warm*regularizer
                        if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite objective rank={rank} update={update}, micro={micro}')
                        # All-reduce each microbatch; accumulated gradients still
                        # equal the weighted global objective, including padding.
                        loss.backward()
                        running+=torch.stack(((objective.detach()*weight).sum(),(nll.detach()*weight).sum(),(early.detach()*weight).sum(),weight.sum())).double()
                        regularizer_totals+=torch.stack((regularizer.detach()*regularizer_warm,variance.detach(),covariance.detach()))/accumulation
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True,foreach=True)
                    optimizer.step();update+=1
                    if update%c['training']['log_every']==0:
                        dist.all_reduce(running);torch.cuda.synchronize()
                        if rank==0:
                            means=(running[:3]/running[3]).tolist();elapsed=time.monotonic()-tick
                            values={'optimizer_update':update,'train/epoch':epoch+(step+1)/steps,
                                'train/objective':means[0],'train/distance_nll':means[1],'train/early_log_loss':means[2],
                                'train/gradient_norm':float(norm),'train/examples_per_second':c['training']['log_every']*c['batch_size']/elapsed,
                                'train/encoder_learning_rate':optimizer.param_groups[0]['lr'],
                                'train/gpu_peak_allocated_GiB':torch.cuda.max_memory_allocated()/2**30}
                            regularizer_means=(regularizer_totals/c['training']['log_every']).tolist()
                            values.update({'train/regularizer':regularizer_means[0],
                                'train/variance_penalty':regularizer_means[1],
                                'train/covariance_penalty':regularizer_means[2],
                                'train/total_objective':means[0]+regularizer_means[0]})
                            online.log(values)
                            with (technical/'training.jsonl').open('a') as f:f.write(json.dumps(values)+'\n')
                            print(json.dumps(values),flush=True)
                            write_json(technical/'state.json',dict(state='training',identity=identity,update=update,epoch=values['train/epoch']))
                        running.zero_();regularizer_totals.zero_();tick=time.monotonic()
                    if update%c['training']['save_every']==0:save(last)
                scores=validate(model,selection,c,rank,world,device)
                if not all(math.isfinite(v) for v in scores.values()):raise FloatingPointError('Nonfinite selection score')
                if epoch+1>=c['training']['minimum_selection_epoch'] and scores['objective']<best:
                    best=scores['objective'];save(technical/'best.pt')
                save(last)
                if rank==0:
                    online.log(dict(optimizer_update=update,**{f'validation/{k}':v for k,v in scores.items()}))
                    history.append(dict(epoch=epoch+1,update=update,**scores));csv_rows(record_folder/'tables/validation.csv',history)
                    with (technical/'validation.jsonl').open('a') as f:f.write(json.dumps(history[-1])+'\n')
                    print(json.dumps(dict(stage='validation',epoch=epoch+1,**scores)),flush=True)
                dist.barrier();tick=time.monotonic()
            if rank==0:
                receipt=dict(state='complete',identity=identity,epochs=c['training']['epochs'],updates=update,train_rows=train.n,
                    seconds=time.monotonic()-started,best_sha256=sha(technical/'best.pt'),last_sha256=sha(last))
                write_json(technical/'complete.json',receipt);write_json(technical/'state.json',receipt)
                online.summary.update({'training/completed_epochs':c['training']['epochs'],'checkpoint/validation_objective':best})
    except BaseException:
        if rank==0:write_json(technical/'failure.json',dict(identity=identity,update=update,traceback=traceback.format_exc()))
        raise
    finally:dist.destroy_process_group()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
