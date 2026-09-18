"""Separate preflight measurements on actual training observations, never a fit stage."""
import argparse
import json
import resource
import statistics
import time
from pathlib import Path
import torch
from src.data.structural_pretraining.batches import Release
from src.data.structural_pretraining.batches import move
from src.data.structural_pretraining.prepare import save_json
from src.models.encoders.structural import StructuralModel,ARCHITECTURE_REVISION
from src.project_runtime.paths import resolve_path
from src.training_methods.structural_pretraining.objective import Objective
from src.training_methods.structural_pretraining.train import prepare_batch,implementation
from src.training_methods.shared_pretraining.runtime import cached_update
from src.training_methods.shared_pretraining.compilation import compile_encoder as enable_compile,compilation_counters


def measure(config,micro,output,material=None,memory_limit_gib=40.,*,precision,warmup_updates=2,measured_updates=5,compile_encoder=False):
    if warmup_updates<1 or measured_updates<1:raise ValueError('Benchmark requires warmup and measured updates')
    _,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,65536),hard))
    torch.set_num_threads(2);torch.manual_seed(config['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(memory_limit_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    release=Release(resolve_path(config['release']),materials=config['materials'])
    # Largest-support training records in each material/potential group. For
    # histories use the sum of frame counts as a conservative union-size proxy.
    ranked={}
    for group,rows in release.groups.items():
        if material is not None and (group[0]!=material or group[2]):continue
        def size(i):
            name,row,record=release.rows[i];a=release.arrays[name];slots=[0,1,2] if config['history_frames']==3 and not record['static'] else [2]
            views=a['views'][row,slots];return int((a['offsets'][views+1]-a['offsets'][views]).sum())
        ranked[group]=sorted(rows,key=size,reverse=True)[:config['batch_size']]
    group=max(ranked,key=lambda g:sum(size_for(release,i,config['history_frames']) for i in ranked[g]))
    indices=ranked[group];temporal=not group[2]
    delta=[0. if group[2] else float(release.arrays[release.rows[i][0]]['times'][3]) for i in indices]
    cfg=dict(config,microbatch_size=micro)
    batches,*_=prepare_batch(release,indices,temporal,delta,cfg)
    model=StructuralModel(config['architecture'],history=config['history_frames']>1).cuda()
    objective=Objective(release.manifest['normalization'],config['method'],config['physical_correlation_weight']).cuda()
    parameters=sum(p.numel() for p in model.encoder.parameters())
    if compile_encoder:
        enable_compile(model.encoder,move(batches[0],'cuda'),precision)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    warmup_start=time.monotonic()
    for _ in range(warmup_updates):cached_update(model,objective,batches,optimizer,temporal,delta,precision=precision)
    torch.cuda.synchronize();warmup_seconds=time.monotonic()-warmup_start
    torch.cuda.reset_peak_memory_stats();timings=[]
    for _ in range(measured_updates):
        start=time.monotonic()
        terms=cached_update(model,objective,batches,optimizer,temporal,delta,precision=precision)
        torch.cuda.synchronize();timings.append(time.monotonic()-start)
    value=dict(device=torch.cuda.get_device_name(),microbatch=micro,batch_size=config['batch_size'],memory_limit_GiB=memory_limit_gib,group=list(group),indices=indices,
        precision=precision,tf32=False,warmup_updates=warmup_updates,measured_updates=measured_updates,
        architecture_revision=ARCHITECTURE_REVISION,encoder_parameters=parameters,implementation=implementation(),
        compiled=compile_encoder,compilation=compilation_counters(),warmup_seconds=warmup_seconds,
        update_seconds=timings,median_seconds=statistics.median(timings),mean_seconds=statistics.mean(timings),
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_GiB=torch.cuda.max_memory_reserved()/2**30,terms=terms,data_identity=release.manifest['identity'])
    save_json(output,value);print(json.dumps({k:v for k,v in value.items() if k not in ('indices','terms')},indent=2))


def size_for(release,i,history):
    name,row,r=release.rows[i];a=release.arrays[name];slots=[0,1,2] if history==3 and not r['static'] else [2]
    v=a['views'][row,slots];return int((a['offsets'][v+1]-a['offsets'][v]).sum())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--microbatch',type=int,required=True);p.add_argument('--output',required=True)
    p.add_argument('--material')
    p.add_argument('--memory-limit-gib',type=float,default=40.)
    p.add_argument('--precision',choices=['float32','bf16'],required=True)
    p.add_argument('--warmup-updates',type=int,default=2);p.add_argument('--measured-updates',type=int,default=5)
    p.add_argument('--compile',action='store_true',dest='compile_encoder',help='Compile the encoder only, with precision casts preserved')
    args=p.parse_args();measure(json.loads(Path(args.config).read_text()),args.microbatch,args.output,args.material,args.memory_limit_gib,
        precision=args.precision,warmup_updates=args.warmup_updates,measured_updates=args.measured_updates,compile_encoder=args.compile_encoder)
