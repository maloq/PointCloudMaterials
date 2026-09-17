"""Fresh v2 task models; preserve the existing head definitions and native v1."""
import torch
from torch import nn
from .native_model import NativeEncoder
from src.models.encoders.mace_backend import with_mace_backend


class TaskModel(nn.Module):
    def __init__(self, encoder, objective):
        super().__init__()
        self.encoder = encoder
        self.objective = objective
        if objective == 'physical_means':
            self.present = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
            self.future = nn.Linear(135, 6 * 128)
        elif objective == 'onset':
            self.hazard = nn.Linear(135, 6)
        else:
            raise ValueError(f'Unknown v2 objective: {objective}')

    def forward(self, observations, conditions):
        state = self.encoder(observations)
        if state.shape != (len(conditions), 128) or conditions.shape[1:] != (7,):
            raise ValueError('Require [B,128] state and [B,7] conditions')
        conditional = torch.cat((state, conditions), -1)
        if self.objective == 'onset':
            return dict(state=state, logits=self.hazard(conditional))
        return dict(state=state, present=self.present(state), future=self.future(conditional).reshape(-1, 6, 128))


def build_model(encoder_kind, objective, variant, config):
    if encoder_kind == 'mace':
        encoder = NativeEncoder(variant=variant, **config['mace'])
        # No optimizer exists yet. Old e3nn AdamW states are never remapped.
        encoder = with_mace_backend(encoder, config['mace_backend'])
    elif encoder_kind == 'axial_gatr':
        from src.models.encoders.axial_gatr import AxialGATrEncoder
        encoder = AxialGATrEncoder(variant=variant, **config['axial_gatr'])
    else:
        raise ValueError(f'Unknown encoder kind: {encoder_kind}')
    return TaskModel(encoder, objective)
