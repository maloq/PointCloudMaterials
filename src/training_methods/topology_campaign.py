"""Analyze interrupted MACE work, verify topology training, then run detached."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import numpy as np
import torch
from src.data_utils.temporal_campaign import write_json
from src.training_methods.pretrained_mace_queue import deadline,run_command


def finalize_partial(plan):
    from src.training_methods.pretrained_mace import Learner,validate
    from src.data_utils.pretrained_mace import Quadruplets
    cfg=json.loads(Path(plan['partial_config']).read_text());out=Path(cfg['output'])
    saved=torch.load(plan['partial_checkpoint'],map_location='cpu',weights_only=False)
    data=Quadruplets(cfg);model=Learner(cfg).cuda().eval();model.load_state_dict(saved['model'],strict=True)
    with np.load(out/'scaling.npz') as f:sc=dict(f)
    val=data.validation_indices(cfg['validation_anchors_per_material'],np.random.default_rng(cfg['seed']))
    metrics=validate(model,data,val,sc,cfg)
    saved['validation']=metrics;saved.pop('optimizer');torch.save(saved,out/'best.pt')
    initial=json.loads((out/'initial_validation.json').read_text())
    write_json(out/'training_summary.json',dict(stop_reason='user_requested_topology_switch',epochs_completed=saved['epoch']-1,partial_epoch=True,steps=saved['step'],anchor_exposures=saved['anchor_exposures'],view_exposures=4*saved['anchor_exposures'],best_epoch=saved['epoch'],best_validation=metrics,selected_epoch=saved['epoch'],selected_validation=metrics,checkpoint_selection='last_saved_partial',initial_validation=initial,converged=False,source_checkpoint=plan['partial_checkpoint']))


def preflight(plan):
    from src.training_methods.pretrained_mace import Learner,load_warm_start,gpu_batch,encode,objective,gradient_cached_step
    from src.training_methods.topology_objective import neighborhood_density,attraction,density_matched_pairs
    from src.data_utils.pretrained_mace import Quadruplets
    cfg=json.loads(Path(plan['training_config']).read_text());data=Quadruplets(cfg)
    with np.load(Path(cfg['warm_start']['scaling_dir'])/'scaling.npz') as f:sc=dict(f)
    model=Learner(cfg).cuda();load_warm_start(model,cfg['warm_start'])
    # Check exact cached gradients against ordinary full backprop with real
    # mixed Al shooting/continuation, Mg and Ta data.
    indices=np.concatenate([p[:1] for p in data.al_subpools['train']]+[p[:2] for p in data.pools['train'][1:]])
    small=gpu_batch(data.get(indices),sc);x,t,c,m=small;mask=torch.tensor(data.temporal_mask(indices),device='cuda')
    # Six examples cannot densely sample density. Widen only this numerical
    # gradient-equivalence test; the real 1536-example preflight uses 3%.
    tiny=dict(cfg,microbatch_size=3,topology=dict(cfg['topology'],density_relative_tolerance=1.));z=encode(model,x,m,3)
    loss,_=objective(model,z,t,c,m,tiny,mask,neighborhood_density(x));loss.backward()
    reference={k:p.grad.detach().clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True);gradient_cached_step(model,small,tiny,mask)
    error=max(float((p.grad-reference[k]).abs().max()) for k,p in model.named_parameters() if k in reference)
    for k,p in model.named_parameters():
        if k in reference:torch.testing.assert_close(p.grad,reference[k],rtol=3e-4,atol=2e-5)
    a=torch.zeros(6,32,device='cuda');w=a.new_tensor(cfg['topology']['reliability_weights']);materials=torch.tensor([0,0,1,1,2,2],device='cuda')
    close=attraction(a,a,materials,w,[1,1,1],.05);far=attraction(a,a+2,materials,w,[1,1,1],.05)
    assert bool((close>far).all())
    density=a.new_tensor([1,1.01,2,2.01,3,3.01]);i,j=density_matched_pairs(density,materials,.03);assert bool((materials[i]==materials[j]).all())
    del reference,z,loss,small,x,t,c,m;model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    ids=next(data.epoch('train',cfg['batch_size'],np.random.default_rng(cfg['seed'])));batch=gpu_batch(data.get(ids),sc);mask=torch.tensor(data.temporal_mask(ids),device='cuda')
    start=time.monotonic();value,parts=gradient_cached_step(model,batch,cfg,mask)
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=cfg['learning_rate']),dict(params=list(model.tda.parameters())+list(model.forecast.parameters()),lr=cfg['head_learning_rate'])],fused=True)
    optimizer.step();torch.cuda.synchronize()
    with torch.no_grad():
        x,t,c,m=batch;z=encode(model,x,m,cfg['microbatch_size']);after,_=objective(model,z,t,c,m,cfg,mask,neighborhood_density(x))
    if not bool(torch.isfinite(after)):raise FloatingPointError('Topology preflight failed at requested peak LR')
    result=dict(batch_shape=list(batch[0].shape),temporal_pairs=int(mask.sum()),gradient_cache_max_error=error,loss_before=value,loss_after_peak_lr_step=float(after),gradient_norm=float(norm),peak_encoder_lr=cfg['learning_rate'],peak_head_lr=cfg['head_learning_rate'],peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,seconds=time.monotonic()-start,parts=parts)
    write_json(Path(plan['output'])/'preflight.json',result);print('TOPOLOGY_PREFLIGHT',json.dumps(result),flush=True)


def run(plan):
    out=Path(plan['output'])
    def status(state,**extra):write_json(out/'status.json',dict(state=state,**extra))
    try:
        status('analyzing_partial_model');finalize_partial(plan);torch.cuda.empty_cache()
        cfg=json.loads(Path(plan['partial_config']).read_text())
        run_command([sys.executable,'-m','src.training_methods.pretrained_mace','--config',plan['partial_config'],'--stage','analysis'],Path(cfg['output'])/'analysis.log',deadline(plan))
        run_command([sys.executable,'-m','src.analysis.pretrained_mace_ablation','--plan',plan['comparison_plan'],'--run','no_temporal_partial'],Path(cfg['output'])/'probe.log',deadline(plan))
        status('preflight');preflight(plan);torch.cuda.empty_cache()
        cfg=json.loads(Path(plan['training_config']).read_text());directory=Path(cfg['output']);directory.mkdir(exist_ok=True)
        # Reserve fifteen minutes for selected-encoder analysis after training.
        seconds=min(cfg['max_training_seconds'],int(deadline(plan)-time.time())-900)
        if seconds<1800:raise RuntimeError(f'Only {seconds}s remain for topology training after analysis reserve')
        cfg['max_training_seconds']=seconds;Path(plan['training_config']).write_text(json.dumps(cfg,indent=2)+'\n')
        shutil.copy2(Path(plan['prepared_output'])/'data_summary.json',directory/'data_summary.json')
        status('training_topology_model',config=plan['training_config'])
        run_command([sys.executable,'-m','src.training_methods.pretrained_mace','--config',plan['training_config'],'--stage','train'],directory/'run.log',deadline(plan))
        status('probing_topology_model')
        run_command([sys.executable,'-m','src.analysis.pretrained_mace_ablation','--plan',plan['comparison_plan'],'--run','topology_aware'],directory/'probe.log',deadline(plan))
        status('static_analysis_topology_model')
        run_command([sys.executable,'-m','src.training_methods.pretrained_mace','--config',plan['training_config'],'--stage','analysis'],directory/'analysis.log',deadline(plan))
        run_command([sys.executable,'-m','src.analysis.pretrained_mace_ablation','--plan',plan['comparison_plan'],'--collect'],out/'collect.log',deadline(plan))
        status('complete')
    except BaseException:
        status('failed',traceback=traceback.format_exc());raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    def interrupted(signum,frame):raise InterruptedError(f'Topology campaign interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');run(json.loads(Path(args.plan).read_text()))


if __name__=='__main__':main()
