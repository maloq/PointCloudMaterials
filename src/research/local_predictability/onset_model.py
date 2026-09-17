"""Supervised onset diagnostic; the separate physical encoder never uses this loss."""
import torch
from torch import nn
from .native_model import NativeEncoder


class OnsetModel(nn.Module):
    def __init__(self,variant='snapshot',**encoder_options):
        super().__init__()
        self.encoder=NativeEncoder(variant=variant,**encoder_options)
        self.hazard=nn.Linear(128+7,6)

    def forward(self,observations,conditions):
        z=self.encoder(observations)
        if conditions.shape!=(len(z),7):
            raise ValueError(f'Expected seven frozen conditions per row, got {conditions.shape}')
        return dict(state=z,logits=self.hazard(torch.cat((z,conditions),-1)))
