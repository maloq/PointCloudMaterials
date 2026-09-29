"""VCReg on the exported state, using the actual global weighted batch."""
import torch
import torch.distributed as dist
from torch.distributed.nn.functional import all_gather


def variance_covariance(z,weight,config,*,distributed=True):
    # Differentiable all-gather sums the replicated regularizer gradients;
    # subsequent DDP averaging cancels that factor. Never detach remote rows.
    if distributed and dist.get_world_size()>1:
        z=torch.cat(all_gather(z.contiguous()),0)
        weights=[torch.empty_like(weight) for _ in range(dist.get_world_size())]
        dist.all_gather(weights,weight.contiguous());weight=torch.cat(weights)
    z=z.float()/config['scale'];weight=weight.float()/weight.sum()
    centered=z-(weight[:,None]*z).sum(0)
    covariance=(centered.T*weight)@centered
    variance=torch.relu(config['std_floor']-(covariance.diagonal()+config['epsilon']).sqrt()).mean()
    d=z.shape[1]
    off=covariance-torch.diag_embed(covariance.diagonal())
    redundancy=off.square().sum()/(d*(d-1))
    penalty=config['variance_weight']*variance+config['covariance_weight']*redundancy
    return penalty,variance,redundancy
