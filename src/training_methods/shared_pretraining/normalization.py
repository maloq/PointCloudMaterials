"""Refresh decoder moments from fixed training observations, never evaluation peers."""
import numpy as np
import torch

from src.data.structural_pretraining.batches import collate,move
from src.data.structural_pretraining.prepare import digest


def calibration_indices(release,seed,count):
    """Proportional deterministic sample across the selected training groups."""
    if count<2*len(release.group_keys):
        raise ValueError('Head calibration needs at least two examples per training group')
    raw=release.group_weights*count
    sizes=np.floor(raw).astype(int)
    sizes[np.argsort(-(raw-sizes),kind='stable')[:count-int(sizes.sum())]]+=1
    if (sizes<2).any():raise ValueError('Increase head calibration sample for small groups')
    indices=[]
    for group,n in zip(release.group_keys,sizes,strict=True):
        rows=release.groups[group]
        if n>len(rows):raise ValueError(f'Calibration sample exceeds training group {group}')
        rng=np.random.default_rng(np.random.SeedSequence([seed,83071,int(digest(str(group))[:8],16)]))
        indices.extend(rng.choice(rows,int(n),replace=False).tolist())
    return indices


@torch.no_grad()
def refresh_head_moments(model,states):
    """Fit input BN and then projector-hidden BN in evaluation order.

    Float64 moments resolve small state differences around a common offset.
    This changes only running buffers, not learned parameters or the encoder.
    Evaluation remains independent of batch membership and ordering.
    """
    if model.training:raise ValueError('Head moment refresh requires evaluation mode')
    if len(states)<2 or not torch.isfinite(states).all():
        raise ValueError('Head calibration requires finite training states with N >= 2')
    def fit(bn,values):
        x=values.double()
        bn.running_mean.copy_(x.mean(0).float())
        bn.running_var.copy_(x.var(0,unbiased=True).float())
    for name in ('projector','physical','tda'):fit(getattr(model,name)[0],states)
    hidden=model.projector[1](model.projector[0](states.float()))
    fit(model.projector[2],hidden)


@torch.no_grad()
def calibrate_heads(model,release,config):
    """Re-encode a fixed training sample at the current weights before selection."""
    indices=calibration_indices(release,config['seed'],config['head_calibration_rows'])
    was_training=model.training;model.eval();device=next(model.parameters()).device
    states=[];micro=min(config['microbatch_size'],64)
    for start in range(0,len(indices),micro):
        samples=[release.observation(i,'anchor',config['history_frames']>1,config['architecture']=='mace')
                 for i in indices[start:start+micro]]
        batch=move(collate(samples,config['architecture']),device)
        with torch.autocast(device.type,dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
            states.append(model.encoder(batch).float())
    states=torch.cat(states);refresh_head_moments(model,states)
    model.train(was_training)
    return dict(rows=len(indices),indices_sha256=digest(indices),
        groups=[list(k) for k in release.group_keys],
        state_std=float(states.double().std(0).mean()))
