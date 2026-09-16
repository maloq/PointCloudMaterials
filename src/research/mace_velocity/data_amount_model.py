"""Direct regularization of native MACE features; auxiliary heads are not embeddings."""
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.experiment_runner.registry import sha256
from src.models.encoders.pretrained_mace import PretrainedMACEEncoder
from src.models.encoders.mace_velocity import MACEVelocityEncoder
from src.research.mace_local_state.motion import orthogonal_basis, projection_residual, time_differences
from .train import GROUPS, Heads, encode


class Directions(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank=rank
        self.network=nn.Sequential(nn.Linear(256,32),nn.SiLU(),nn.Linear(32,256*rank))

    def forward(self,z):
        return orthogonal_basis(self.network(z.detach()).reshape(-1,256,self.rank))


def initialize(config, norm, seed, device):
    if sha256(Path(config['foundation_checkpoint'])) != config['foundation_sha256']:
        raise ValueError('Foundation checkpoint changed')
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    mace=PretrainedMACEEncoder(config['foundation_checkpoint'], performance=dict(
        geometry_cache=False,dense_precision='highest',compile_radial_mlp=False,
        bf16_mode='radial_compensated')).to(device)
    model=MACEVelocityEncoder(mace,norm['feature_mean'],norm['feature_scale'],
        velocity_scale=config['velocity_scale_A_per_ps']).to(device)
    # Reset independently of foundation loading/conversion internals.
    torch.manual_seed(seed+1); heads=Heads().to(device)
    directions=Directions(config['direction_rank']).to(device)
    return model,heads,directions


def load_native_encoder(checkpoint_path,device='cuda:0'):
    """Load the trained native output without any cache or auxiliary-head input."""
    saved=torch.load(Path(checkpoint_path),map_location='cpu',weights_only=False)
    if saved['protocol']!='mace_native_data_amount_v1':
        raise ValueError(f'Expected native data-amount encoder: {checkpoint_path}')
    config=saved['config']
    model,_,_=initialize(config,saved['normalization'],saved['fit']['seed'],device)
    model.load_state_dict(saved['encoder_state'],strict=True)
    return model.eval(),config


def normalized_targets(raw, norm, device):
    return torch.as_tensor((raw-norm['target_mean'])/norm['target_scale'],dtype=torch.float32,device=device)


def objective(config, heads, directions, z, targets, times, ramp):
    """z/targets have source, center, time ordering. Always four centers/source."""
    nt=targets.shape[1]
    if len(z) != targets.shape[0]*nt or targets.shape[0]%4:
        raise ValueError('Objective requires complete four-center source/time contexts')
    predicted=heads(z); error=(predicted-targets.flatten(0,1)).square()
    groups={k:error[:,section].mean() for k,section in GROUPS.items()}
    physical=torch.stack(list(groups.values())[:4]).mean()
    motion=(2*groups['motion_even']+groups['motion_odd'])/3
    state=z[:,:256].reshape(-1,nt,256)
    context=state.reshape(-1,4,nt,256)
    centered=context-context.mean(1,keepdim=True)
    variance=centered.square().mean((0,1,2))
    spread=2*variance.sum().clamp_min(1e-12)
    delta,_,_,bend=time_differences(state,times)
    slow=delta.square().sum(-1).mean()/spread
    curvature=bend.square().sum(-1).mean()/spread
    increments=delta.flatten(0,1)
    basis=directions(state[:,:-1].reshape(-1,256))
    energy=increments.square().sum(-1).mean().clamp_min(1e-12)
    # Neither the auxiliary basis fit nor a post-encoder map smooths the encoder.
    # The actual MACE parameters receive the detached-basis residual gradients.
    direction=projection_residual(increments,basis.detach()).mean()/energy
    basis_fit=projection_residual(increments.detach(),basis).mean()/energy.detach()
    floor=torch.relu(.5-torch.sqrt(variance+1e-6)).square().mean()
    score=physical+config['motion_weight']*motion+ramp*(config['temporal_weight']*slow+
        config['curvature_weight']*curvature+config['direction_weight']*direction)
    total=score+config['variance_weight']*floor+basis_fit
    if not torch.isfinite(total): raise FloatingPointError('Nonfinite native-encoder objective')
    metrics={k:float(v.detach()) for k,v in groups.items()}
    metrics.update(physical=float(physical.detach()),motion=float(motion.detach()),
        slow=float(slow.detach()),curvature=float(curvature.detach()),direction=float(direction.detach()),
        basis_fit=float(basis_fit.detach()),variance_floor=float(floor.detach()),
        within_context_trace=float(variance.sum().detach()),selection_score=float(score.detach()),
        total=float(total.detach()))
    return total,metrics


def replay(config, model, heads, directions, clouds, targets, times, ramp, device):
    z=encode(config,model,clouds,device).detach().requires_grad_(True)
    value,metrics=objective(config,heads,directions,z,targets,times,ramp); value.backward()
    for start in range(0,len(clouds),config['micro_batch_size']):
        actual=encode(config,model,clouds[start:start+config['micro_batch_size']],device,gradients=True)
        actual.backward(z.grad[start:start+len(actual)])
    return metrics


def calibrate(config, model, sources, core, device):
    from .data_amount_data import batch
    clouds,y,_=batch(sources,core)
    z=encode(config,model,clouds,device).cpu().numpy()[:,:256].reshape(-1,4,9,256)
    mean=z.mean((0,1,2))
    scale=np.sqrt(np.mean((z-z.mean(1,keepdims=True))**2,axis=(0,1,2)))
    floor=.05*np.median(scale)
    if floor<=0: raise ValueError('Degenerate core feature calibration')
    y=y.reshape(-1,169).astype(np.float64); ym=y.mean(0); ys=y.std(0)
    for section in (slice(16,32),slice(32,96),slice(96,160)):
        ys[section]=np.sqrt(np.mean(ys[section]**2))
    ym[166:]=0.; ys[166:]=np.sqrt(np.mean(y[:,166:]**2,axis=0))
    if np.any(ys<=0): raise ValueError(f'Degenerate core target scales: {np.flatnonzero(ys<=0)}')
    return dict(feature_mean=mean,feature_scale=np.maximum(scale,floor),
                target_mean=ym.astype(np.float32),target_scale=ys.astype(np.float32))
