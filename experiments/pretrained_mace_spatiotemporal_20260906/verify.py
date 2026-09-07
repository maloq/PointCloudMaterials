"""Reproduce receptive-field, rotation, and full-batch gradient equivalence checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from src.data_utils.pretrained_mace import Quadruplets
from src.models.encoders.pretrained_mace import PretrainedMACEEncoder
from src.models.encoders.atomic_graph import graph_data
from src.training_methods.temporal_campaign import local_graph
from src.training_methods.pretrained_mace import Learner,encode,objective,gradient_cached_step


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text());torch.set_num_threads(4);torch.manual_seed(42)
    data=Quadruplets(cfg);indices=np.concatenate([p[:2] for p in data.pools['train']]);x,t,c,m=data.get(indices)
    x=torch.from_numpy(x).cuda();m=torch.from_numpy(m).cuda();cloud=x[::2,0];material=m[::2]
    native=PretrainedMACEEncoder(cfg['pretrained_checkpoint'],accelerated=False).cuda().eval()
    fast=PretrainedMACEEncoder(cfg['pretrained_checkpoint']).cuda().eval()
    edges,counts=local_graph(cloud,5.);graph=graph_data(cloud,material,edges,counts)
    graph['node_attrs']=torch.nn.functional.one_hot(native.element_indices[material],len(native.backbone.atomic_numbers)).float().repeat_interleave(cfg['points'],0)
    with torch.no_grad():
        reference=native.backbone(graph,compute_force=False)['node_feats'][graph['ptr'][:-1]]
        pruned=native(cloud,material);converted=fast(cloud,material)
        torch.testing.assert_close(pruned,reference,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(converted,reference,rtol=3e-4,atol=3e-5)
        rotation=torch.linalg.qr(torch.randn(3,3,device='cuda')).Q
        rotated=fast(cloud@rotation,material)
        torch.testing.assert_close(rotated,converted,rtol=4e-4,atol=4e-5)
    print('full_vs_pruned_max_error',float((pruned-reference).abs().max()))
    print('converted_vs_native_max_error',float((converted-reference).abs().max()))
    print('rotation_max_error',float((rotated-converted).abs().max()))
    del native,fast
    model=Learner(cfg).cuda().train()
    # Synthetic targets are sufficient here: this check is about gradient algebra,
    # not TDA scaling or the physical quality of a trained representation.
    batch=[x,torch.from_numpy(t[:,:,:cfg['tda_components']]).cuda(),torch.from_numpy(c).cuda(),m]
    z=encode(model,x,m,24);loss,_=objective(model,z,*batch[1:],cfg);loss.backward()
    expected={name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None}
    model.zero_grad();loss_cached,_=gradient_cached_step(model,batch,dict(cfg,microbatch_size=3))
    error=0.
    for name,p in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(p.grad,expected[name],rtol=2e-3,atol=3e-5)
            error=max(error,float((p.grad-expected[name]).abs().max()))
    np.testing.assert_allclose(float(loss.detach()),loss_cached,rtol=1e-6)
    print('gradient_cache_max_gradient_error',error,flush=True)

if __name__=='__main__':main()
