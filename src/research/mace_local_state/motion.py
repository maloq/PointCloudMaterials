"""Shared local directions and time-aware bending; no forecasting objective."""
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree

from .smooth import FAMILIES, within_covariance


def time_differences(z, times):
    """Native increments, physical velocities, acceleration and finite-lag bend."""
    dt = times[:,1:]-times[:,:-1]
    if torch.any(dt <= 0): raise ValueError('Physical sequence times must increase')
    dt = dt.to(z.dtype)
    increment = z[:,1:]-z[:,:-1]
    velocity = increment/dt[...,None]
    change = velocity[:,1:]-velocity[:,:-1]
    acceleration = 2*change/(dt[:,1:]+dt[:,:-1])[...,None]
    harmonic_dt = 2*dt[:,1:]*dt[:,:-1]/(dt[:,1:]+dt[:,:-1])
    bend = change*harmonic_dt[...,None]
    return increment, velocity, acceleration, bend


def orthogonal_basis(raw):
    return torch.linalg.qr(raw, mode='reduced').Q


def projection_residual(delta, basis):
    coefficients = torch.einsum('nd,ndr->nr',delta,basis)
    residual = delta-torch.einsum('ndr,nr->nd',basis,coefficients)
    return residual.square().sum(-1)


def projector_distance(a, b):
    """Squared projector distance/(2r), invariant to basis signs/rotations."""
    cross = torch.einsum('ndr,nds->nrs',a,b)
    return 1-cross.square().sum((-1,-2))/a.shape[-1]


class MotionState(nn.Module):
    def __init__(self, dimension, rank):
        super().__init__()
        self.dimension = dimension; self.rank = rank; d = dimension or 256
        self.mapping = (nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,128),
            nn.SiLU(),nn.Linear(128,d)) if dimension else nn.Identity())
        self.register_buffer('initial_scale',torch.ones(d))
        self.readout = nn.Sequential(nn.Linear(d,128),nn.SiLU(),nn.Linear(128,160))
        # A single small function shared by all groups/sources, using current state only.
        self.directions = nn.Sequential(nn.Linear(d,32),nn.SiLU(),nn.Linear(32,d*rank))

    def forward(self, x):
        z = self.mapping(x)/self.initial_scale
        return z,self.readout(z)

    def basis(self, z):
        return orthogonal_basis(self.directions(z.detach()).reshape(-1,z.shape[-1],self.rank))


def lag_weights(dt, eligible, weights):
    """Equal nonempty native cadence bins; source-balanced within each bin."""
    result = np.zeros_like(dt,dtype=np.float64)
    bins = np.unique(np.round(dt[eligible],9))
    if not len(bins): raise ValueError('No eligible physical-lag observations')
    expanded = np.broadcast_to(weights[:,None],dt.shape)
    for value in bins:
        mask = eligible & np.isclose(dt,value,atol=1e-9,rtol=0)
        result[mask] = expanded[mask]/expanded[mask].sum()/len(bins)
    return result.astype(np.float32)


def batch_arrays(data, split, config, device):
    ids = np.flatnonzero(data['split']==split)
    x = data['embedding'][ids,:,:256]; n,nt,_ = x.shape
    source = data['source_id'][ids]; unique,inverse = np.unique(source,return_inverse=True)
    context = inverse[:,None]*nt+np.arange(nt)[None,:]
    w = data['weights'][ids] if split==0 else np.ones(n,dtype=np.float32)
    low = data['raw_target'][ids,:,4] < config['low_order_threshold']
    times = data['time_ps'][ids];dt = np.diff(times,axis=1)
    edge = []; curve = []
    for subset in ('all','low_order'):
        eligible = dt<=config['maximum_training_lag_ps']+1e-9
        triples = eligible[:,1:] & eligible[:,:-1]
        if subset=='low_order':
            eligible &= low[:,1:] & low[:,:-1]
            triples &= low[:,:-2] & low[:,1:-1] & low[:,2:]
        edge.append(lag_weights(dt,eligible,w))
        curve.append(lag_weights((dt[:,1:]+dt[:,:-1])/2,triples,w))
    rng = np.random.default_rng(config['seed'])
    edge_ids = np.flatnonzero(np.stack(edge).sum(0).ravel()>0)
    if len(edge_ids)>config['direction_samples']:
        edge_ids = np.sort(rng.choice(edge_ids,config['direction_samples'],replace=False))
    si,ti = np.unravel_index(edge_ids,(n,nt-1))
    # Only physical targets define this training locality comparison. The encoder
    # and direction network never receive targets, source IDs or time as inputs.
    physical = data['target'][ids,:,:16].reshape(-1,16)
    anchors = si*nt+ti
    groups = np.repeat(data['lineage_id'][ids],nt)
    if split==0:
        # Balance the locality pool so dense sibling families cannot monopolize it.
        pool = np.concatenate([rng.choice(np.flatnonzero(groups==g),min(16,int(np.sum(groups==g))),replace=False)
            for g in np.unique(groups)])
        count = min(len(pool),64)
        _, nearest = cKDTree(physical[pool]).query(physical[anchors],k=count,workers=1)
        neighbors = pool[nearest]
        other = groups[neighbors] != groups[anchors,None]
        if np.any(~other.any(1)):
            raise ValueError('No cross-source physical neighbor in the balanced locality pool')
        neighbor = neighbors[np.arange(len(anchors)),other.argmax(1)]
    else:
        neighbor = anchors  # Unused for validation: no held-out basis fitting.
    output = dict(x=x,y=data['target'][ids,:,:160],times=times,weights=np.repeat(w,nt),
        context=context.ravel(),low=low.ravel(),edge_weights=np.stack(edge),curve_weights=np.stack(curve),
        direction_weights=np.stack(edge).reshape(2,-1)[:,edge_ids],
        direction_ids=edge_ids,anchor_ids=anchors,neighbor_ids=neighbor)
    tensors = {k:torch.as_tensor(v,device=device) for k,v in output.items()}
    tensors['weights'] = tensors['weights'].float()
    tensors['ids'] = ids
    return tensors


def objective(model, batch, spec, config, ramp=1., training=True):
    z,pred = model(batch['x'])
    flat = z.reshape(-1,z.shape[-1]);w = batch['weights'];low = batch['low']
    error = (pred-batch['y']).square().reshape(-1,160)
    physical = torch.stack([(error[:,s].mean(1)*w).sum()/w.sum() for s in FAMILIES.values()])
    cov = within_covariance(flat,w,batch['context'])
    cov_low = within_covariance(flat[low],w[low],batch['context'][low])
    spread = torch.stack([cov.trace(),cov_low.trace()])
    if torch.any(spread <= 1e-12): raise FloatingPointError('Collapsed within-context local state')
    increments,velocity,acceleration,bend = time_differences(z,batch['times'])
    energy = increments.square().sum(-1)
    temporal = ((batch['edge_weights']*energy[None]).sum((1,2))/(2*spread)).mean()
    curvature = ((batch['curve_weights']*bend.square().sum(-1)[None]).sum((1,2))/(2*spread)).mean()
    delta = increments.reshape(-1,z.shape[-1])[batch['direction_ids']]
    q = model.basis(flat[batch['anchor_ids']])
    dw = batch['direction_weights'];dw = dw/dw.sum(1,keepdim=True)
    energies = (dw*delta.square().sum(-1)[None]).sum(1)
    if torch.any(energies <= 1e-12): raise FloatingPointError('Zero temporal energy in direction assay')
    # Train direction predictor for EVERY control. Map receives its gradient only
    # through the explicitly enabled direction penalty, not via basis fitting.
    residual = projection_residual(delta,q.detach())
    direction = ((dw*residual[None]).sum(1)/energies).mean()
    basis_fit = ((dw*projection_residual(delta.detach(),q)[None]).sum(1)/energies.detach()).mean()
    neighbor = z.new_zeros(())
    if training:
        neighbor = projector_distance(q,model.basis(flat[batch['neighbor_ids']])).mean()
    eye = torch.eye(z.shape[-1],device=z.device)
    calibration = ((cov-eye).square().sum()+(cov_low-eye).square().sum())/(2*z.shape[-1])
    value = physical.mean()+ramp*(spec['temporal']*temporal+spec['direction']*direction+spec['curvature']*curvature)
    if model.dimension: value = value+config['covariance_weight']*calibration
    if training: value = value+basis_fit+config['basis_neighbor_weight']*neighbor
    if not torch.isfinite(value): raise FloatingPointError('Nonfinite local-motion objective')
    return value,dict(physical=physical,temporal=temporal,direction=direction,
        curvature=curvature,basis_fit=basis_fit,basis_neighbor=neighbor,calibration=calibration),z


def specifications(config):
    for seed in config['seeds']:
        for rank in config['ranks']:
            yield dict(name=f'reference-r{rank}-s{seed}',dimension=0,rank=rank,seed=seed,
                temporal=0.,direction=0.,curvature=0.,kind='reference')
        for d in config['dimensions']:
            for rank in config['ranks']:
                yield dict(name=f'physics-d{d}-r{rank}-s{seed}',dimension=d,rank=rank,seed=seed,
                    temporal=0.,direction=0.,curvature=0.,kind='physics')
                for temporal in config['temporal_weights']:
                    for kind,direction,curvature in [('slow',0.,0.),('directions',config['direction_weight'],0.),
                        ('curvature',0.,config['curvature_weight']),
                        ('combined',config['direction_weight'],config['curvature_weight'])]:
                        yield dict(name=f'{kind}-d{d}-r{rank}-t{temporal:g}-s{seed}',dimension=d,rank=rank,
                            seed=seed,temporal=temporal,direction=direction,curvature=curvature,kind=kind)
