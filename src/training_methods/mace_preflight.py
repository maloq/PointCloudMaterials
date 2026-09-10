"""Real-MACE checks for full-graph pooling, TDA80, phased gradients and memory."""
import copy
from pathlib import Path
import time
import numpy as np
import torch
from src.analysis.liquid_structure import persistence_image
from src.data_utils.pretrained_mace import Quadruplets
from src.data_utils.pretrained_mace_gpu import GPUQuadruplets
from src.data_utils.temporal_campaign import write_json
from src.training_methods.pretrained_mace import Learner,fit_scaling
from src.training_methods.mace_objective import objective,cached_step,training_views
from src.training_methods.mace_performance import encode_views


def native_descriptor(model,x,m):
    """Independent public MACE forward, including all nodes and both layers."""
    g=model.encoder.build_geometry(x,m);b,n,_=x.shape
    data=dict(positions=x.flatten(0,1),node_attrs=g.attrs,edge_index=g.edges,
        shifts=x.new_zeros((g.edges.shape[1],3)),unit_shifts=x.new_zeros((g.edges.shape[1],3)),
        cell=x.new_zeros((3*b,3)),batch=torch.arange(b,device=x.device).repeat_interleave(n),
        ptr=torch.arange(b+1,device=x.device)*n,head=torch.zeros(b,device=x.device,dtype=torch.long))
    return model.encoder.backbone(data,compute_force=False)['node_feats'].reshape(b,n,256).mean(1)


def preflight(cfg):
    out=Path(cfg['output']).parent.parent;scaling_dir=out/'preflight_scaling';scaling_dir.mkdir(exist_ok=True)
    torch.manual_seed(cfg['seed']);data=Quadruplets(cfg);model=Learner(cfg).cuda()
    scaling=fit_scaling(model,data,cfg,scaling_dir);gpu=GPUQuadruplets(data,scaling)
    ids=np.concatenate([p[:4] for p in data.pools['train']]);batch=gpu.get(ids)
    views=training_views(cfg['loss'])
    x,targets,_,material=batch;mask=torch.tensor(data.temporal_mask(ids),device='cuda')
    with torch.no_grad():
        actual=model.encoder.raw_features(x[:,0],material)
        reference=native_descriptor(model,x[:,0],material)
        torch.testing.assert_close(actual,reference,rtol=1e-4,atol=1e-5)
        q=torch.linalg.qr(torch.randn(3,3,device='cuda'))[0]
        rotated=model.encoder.raw_features(x[:,0]@q,material)
        permuted=model.encoder.raw_features(x[:,0,torch.randperm(80,device='cuda')],material)
        torch.testing.assert_close(actual,rotated,rtol=2e-3,atol=2e-5)
        torch.testing.assert_close(actual,permuted,rtol=2e-3,atol=2e-5)
        altered=x[:,0].clone();altered[:,-15:]*=.9
        outer_effect=float((model.encoder.raw_features(altered,material)-actual).abs().max())
        if outer_effect<=1e-6:raise AssertionError('Changing atoms 66–80 did not affect the pooled representation')
    for i,j in ids:
        for view in range(3):
            target=persistence_image(data.clouds[i][j,view+(4 if cfg['protocol']=='thermal80' else 0)].astype(np.float32))
            np.testing.assert_allclose(data.tda[i][j,view],target,rtol=0,atol=0)
    errors={};tiny=copy.deepcopy(cfg);tiny['microbatch_size']=6
    tda_start=cfg['loss']['tda_start_epoch']
    for epoch in (tda_start-1,tda_start):
        model.tda.requires_grad_(epoch>=tda_start);model.zero_grad(set_to_none=True)
        cached_step(model,batch,mask,tiny,epoch)
        gradients={n:p.grad.clone() for n,p in model.named_parameters() if p.requires_grad}
        if epoch<tda_start and any(p.grad is not None for p in model.tda.parameters()):raise AssertionError('TDA head received early gradients')
        model.zero_grad(set_to_none=True)
        z=encode_views(model,x[:,views],material,6)
        loss,_=objective(model,z,targets[:,views],mask,cfg['loss'],epoch);loss.backward()
        errors[str(epoch)]=max(float((p.grad-gradients[n]).abs().max()) for n,p in model.named_parameters() if p.requires_grad)
        for name,p in model.named_parameters():
            if p.requires_grad:torch.testing.assert_close(p.grad,gradients[name],rtol=1e-3,atol=5e-5,msg=name)
        del gradients,z,loss
    model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    ids=next(data.epoch('train',cfg['batch_size'],np.random.default_rng(cfg['seed'])))
    batch=gpu.get(ids);mask=torch.tensor(data.temporal_mask(ids),device='cuda')
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=cfg['learning_rate']),dict(params=model.tda.parameters(),lr=cfg['head_learning_rate'])],weight_decay=cfg['weight_decay'],fused=True)
    timings=[];torch.cuda.reset_peak_memory_stats()
    for epoch in (tda_start-1,tda_start,tda_start):
        model.tda.requires_grad_(epoch>=tda_start);optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();start=time.monotonic()
        value,parts=cached_step(model,batch,mask,cfg,epoch)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize()
        timings.append(dict(epoch=epoch,seconds=time.monotonic()-start,loss=value,gradient_norm=float(norm),parts=parts))
    with torch.no_grad():
        z=encode_views(model,batch[0][:,views],batch[3],cfg['microbatch_size'])
        after,_=objective(model,z,batch[1][:,views],mask,cfg['loss'],tda_start)
        if not bool(torch.isfinite(after)):raise FloatingPointError('Nonfinite loss after peak-LR preflight')
    report=dict(state='passed',native_forward_max_error=float((actual-reference).abs().max()),
        rotation_max_error=float((actual-rotated).abs().max()),permutation_max_error=float((actual-permuted).abs().max()),
        outer_atom_feature_effect=outer_effect,tda_80_cache_checks=len(ids[:12])*3,
        gradient_max_errors_by_epoch=errors,batch_size=cfg['batch_size'],microbatch_size=cfg['microbatch_size'],
        peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,updates=timings,
        protocol=f'No preflight optimizer updates retained. Native MACE all-node mean verified; TDA recomputed on all 80 atoms; both sides of epoch-{tda_start} switch tested.')
    write_json(out/'preflight.json',report)
    print('PREFLIGHT',report,flush=True)
