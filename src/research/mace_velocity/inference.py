"""Load a trained local phase-space encoder without training-cache access."""
from pathlib import Path

import numpy as np
import torch

from src.models.encoders.mace_velocity import MACEVelocityEncoder
from .train import backbone,encode


def load_encoder(checkpoint_path,device='cuda:0'):
    checkpoint=torch.load(Path(checkpoint_path),map_location='cpu',weights_only=False)
    if checkpoint['protocol']!='mace_local_phase_space_v1':
        raise ValueError(f'Not a local phase-space encoder: {checkpoint_path}')
    config=checkpoint['config'];normalization=checkpoint['normalization']
    encoder=MACEVelocityEncoder(backbone(config,device),normalization['feature_mean'],
        normalization['feature_scale'],use_velocity=checkpoint['variant']=='coordinates_velocity',
        velocity_scale=config['velocity_scale_A_per_ps']).to(device)
    encoder.load_state_dict(checkpoint['encoder_state'],strict=True)
    return encoder.eval(),config


def embed_local_groups(encoder,config,positions,velocities,*,device='cuda:0'):
    """Lists of complete halos; coordinates in A, matched velocities in A/ps."""
    if len(positions)!=len(velocities) or not len(positions):
        raise ValueError('Supply a nonempty, matched list of positions and velocities')
    clouds=[]
    for index,(x,v) in enumerate(zip(positions,velocities,strict=True)):
        x=np.asarray(x,dtype=np.float32);v=np.asarray(v,dtype=np.float32)
        if x.ndim!=2 or x.shape[1]!=3 or v.shape!=x.shape or len(x)<80:
            raise ValueError(f'Group {index}: expected aligned [N>=80,3] positions/velocities, got {x.shape}/{v.shape}')
        if not np.isfinite(x).all() or not np.isfinite(v).all():
            raise ValueError(f'Group {index}: nonfinite positions or velocities')
        clouds.append((x,v))
    return encode(config,encoder,clouds,device).cpu().numpy()
