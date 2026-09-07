"""Verify strict 80-point support, MLIP first-layer preservation and gradients."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
from src.data_utils.pretrained_mace import Quadruplets
from src.models.encoders.pretrained_mace import PretrainedMACEEncoder
from src.training_methods.pretrained_mace import Learner,encode,objective,gradient_cached_step


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());torch.set_num_threads(4);torch.manual_seed(42)
    data=Quadruplets(cfg);indices=np.concatenate([p[:2] for p in data.pools['train']]);x,t,c,m=data.get(indices)
    assert x.shape==(6,4,80,3)
    x=torch.from_numpy(x).cuda();material=torch.from_numpy(m).cuda()
    full=np.stack([np.load(Path(data.records[i]['directory'])/'clouds.npy',mmap_mode='r')[j,0] for i,j in indices])
    full=torch.from_numpy(full).cuda()
    compact=PretrainedMACEEncoder(cfg['pretrained_checkpoint'],outer_radius_A=cfg['outer_radius_A']).cuda().eval()
    original=PretrainedMACEEncoder(cfg['pretrained_checkpoint']).cuda().eval()
    with torch.no_grad():
        z=compact(x[:,0],material);wide=compact(full,material);mlip=original(full,material)
        torch.testing.assert_close(z,wide,rtol=2e-4,atol=2e-5)
        torch.testing.assert_close(z[:,:128],mlip[:,:128],rtol=2e-4,atol=2e-5)
        rotation=torch.linalg.qr(torch.randn(3,3,device='cuda')).Q
        rotated=compact(x[:,0]@rotation,material)
        torch.testing.assert_close(z,rotated,rtol=3e-4,atol=3e-5)
        # Replacing the outermost atom cannot affect a compact descriptor because
        # its source weight is zero outside 6.5 A, before the KNN boundary.
        changed=x[:,0].clone();changed[:,-1]*=1.2
        boundary=compact(changed,material)
        torch.testing.assert_close(z,boundary,rtol=1e-5,atol=1e-6)
    result=dict(input_shape=list(x.shape),compact_vs_wide_tapered_max_error=float((z-wide).abs().max()),first_layer_vs_full_MLIP_max_error=float((z[:,:128]-mlip[:,:128]).abs().max()),rotation_max_error=float((z-rotated).abs().max()),outer_neighbor_replacement_max_error=float((z-boundary).abs().max()))
    del compact,original,full,z,wide,mlip,rotated,boundary
    model=Learner(cfg).cuda().train();batch=[x,torch.from_numpy(t[:,:,:32]).cuda(),torch.from_numpy(c).cuda(),material]
    z=encode(model,x,material,24);loss,_=objective(model,z,*batch[1:],cfg);loss.backward()
    expected={name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad();cached,_=gradient_cached_step(model,batch,dict(cfg,microbatch_size=3))
    error=0.
    for name,p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad,expected[name],rtol=2e-3,atol=3e-5)
            error=max(error,float((p.grad-expected[name]).abs().max()))
    np.testing.assert_allclose(float(loss.detach()),cached,rtol=1e-6);result['cached_gradient_max_error']=error
    del expected,z,loss,batch,x
    indices=next(data.epoch('train',cfg['batch_size'],np.random.default_rng(5)))
    x,t,c,m=data.get(indices);batch=[torch.from_numpy(v).cuda() for v in (x,t[:,:,:32],c,m)]
    for _ in range(2):
        model.zero_grad();torch.cuda.reset_peak_memory_stats();start=time.monotonic();gradient_cached_step(model,batch,cfg);torch.cuda.synchronize()
        result['batch_seconds']=time.monotonic()-start;result['peak_allocated_GB']=torch.cuda.max_memory_allocated()/1e9
    (Path(cfg['output'])/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
