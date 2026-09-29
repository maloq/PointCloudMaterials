"""Synchronous data parallelism with differentiable global VCReg moments."""
import os
from contextlib import nullcontext
from types import SimpleNamespace
import torch
import torch.distributed as dist


def initialize():
    world=int(os.environ.get('WORLD_SIZE','1'));rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
    if world>1 and not dist.is_initialized():dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    return rank,world


def world_size():return dist.get_world_size() if dist.is_initialized() else 1


def differentiable_sum(value):
    if world_size()==1:return value
    from torch.distributed.nn.functional import all_reduce
    return all_reduce(value,op=dist.ReduceOp.SUM)


def global_covariance(x):
    count=x.new_tensor(float(len(x)))
    if world_size()>1:dist.all_reduce(count)
    mean=differentiable_sum(x.sum(0))/count
    centered=x-mean
    return differentiable_sum(torch.einsum('bcm,bdm->cd',centered,centered))/(count*x.shape[-1])


@torch.no_grad()
def average_gradients(model):
    """One packed gradient collective; unused parameters keep grad=None.

    The model is under one million parameters, so explicit packed synchronization
    avoids wrapping the variable-size, checkpointed compiled patch graph in DDP.
    Autograd all-reduces in VCReg sum their adjoints; this gradient average yields
    the same regularization gradient as one global batch (not a mean of local VCReg).
    """
    world=world_size()
    if world==1:return
    params=[p for p in model.parameters() if p.requires_grad]
    used=torch.tensor([p.grad is not None for p in params],dtype=torch.int32,device=params[0].device)
    dist.all_reduce(used)
    grads=[p.grad if p.grad is not None else torch.zeros_like(p) for p in params]
    flat=torch.cat([g.flatten() for g in grads]);dist.all_reduce(flat);flat/=world
    begin=0
    for p,g,active in zip(params,grads,used.tolist()):
        n=p.numel()
        if active:g.copy_(flat[begin:begin+n].view_as(p));p.grad=g
        begin+=n


def sum_values(values):
    t=torch.as_tensor(values,dtype=torch.float64,device='cuda')
    if world_size()>1:dist.all_reduce(t)
    return t.cpu().numpy()


def stop_requested(flag):
    t=torch.tensor(int(flag),dtype=torch.int32,device='cuda')
    if world_size()>1:dist.all_reduce(t,op=dist.ReduceOp.MAX)
    return bool(t.item())


@torch.no_grad()
def broadcast_model(model):
    if world_size()>1:
        for tensor in model.state_dict().values():dist.broadcast(tensor,src=0)


def local_tracking():return nullcontext(SimpleNamespace(summary={},log=lambda record:None))
