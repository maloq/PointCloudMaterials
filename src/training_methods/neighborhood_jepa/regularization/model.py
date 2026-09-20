"""Explicit projector/export treatments with the same MACE and prediction heads."""
import torch
from torch import nn
from ..v2.model import Model as BaseModel,Encoder as BaseEncoder


class Encoder(BaseEncoder):
    def __init__(self,channels,export_norm):
        super().__init__(channels)
        if export_norm=='raw':self.output_norm=nn.Identity()
        elif export_norm!='layernorm':raise ValueError(export_norm)
        if any(isinstance(module,nn.BatchNorm1d) for module in self.modules()):
            raise ValueError('Snapshot export must not contain batch-dependent normalization')


class Model(BaseModel):
    def __init__(self,channels,spec,seed):
        super().__init__(channels)
        self.encoder=Encoder(channels,spec['export_norm'])
        with torch.random.fork_rng():
            torch.manual_seed(seed+2)
            self.order_decoder=nn.Sequential(nn.Linear(128,64),nn.SiLU(),nn.Linear(64,8))
            torch.manual_seed(seed+3)
            p=spec['projector']
            if p=='mlp_ln':self.projector=nn.Sequential(nn.Linear(128,128),nn.LayerNorm(128),nn.SiLU(),nn.Linear(128,64))
            elif p=='mlp_plain':self.projector=nn.Sequential(nn.Linear(128,128),nn.SiLU(),nn.Linear(128,64))
            elif p=='linear':self.projector=nn.Linear(128,64,bias=False)
            elif p=='identity':self.projector=nn.Identity()
            else:raise ValueError(p)

    def initialize(self,saved):
        """Explicit v2 width64 transfer; projectors/order readouts start fresh in every arm."""
        parent=saved['manifest']['config']
        if parent['encoder_channels']!=self.encoder.base.channels:raise ValueError('Warm-start MACE width differs')
        state={k:v for k,v in saved['model'].items() if not k.startswith('projector.')}
        result=self.load_state_dict(state,strict=False)
        expected={k for k in self.state_dict() if k.startswith(('projector.','order_decoder.'))}
        if set(result.missing_keys)!=expected or result.unexpected_keys:
            raise ValueError(f'Warm transfer mismatch: {result}')
