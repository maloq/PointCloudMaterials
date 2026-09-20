"""Snapshot export and causally isolated query inputs; all heads see raw export."""
import torch
from torch import nn
from ..model import NeighborhoodModel, NeighborhoodEncoder
from .contracts import LAYOUT


class Encoder(NeighborhoodEncoder):
    layout = LAYOUT

    def __init__(self,channels=32):
        super().__init__('mace',channels=channels)
        self.register_buffer('geometry_scales', torch.ones(len(LAYOUT.degrees), LAYOUT.channels))

    def export(self, batch):
        encoded = self(batch)
        return dict(invariant=encoded[:,:LAYOUT.invariant_dim],
                    equivariant=encoded[:,LAYOUT.invariant_dim:],
                    layout=LAYOUT.metadata(), geometry_scales=self.geometry_scales)


class Model(NeighborhoodModel):
    def __init__(self,channels=32):
        super().__init__('mace', previous_context=False, groups=1)
        self.encoder = Encoder(channels)
        # No free inverse bond decoder can absorb a rescaling of anchored outputs.
        del self.bond
        # Known causal thermodynamic condition belongs to the predictor, not E(x).
        self.condition = nn.Linear(1, LAYOUT.invariant_dim, bias=False)

    def predictions(self, encoded, target, plan):
        """This method cannot read future labels or target feature slots."""
        z = encoded.reshape(len(target['group']), len(plan.views), LAYOUT.packed_dim)
        current = z[:,plan.slot(1,0)]
        conditioned = torch.cat((current[:,:LAYOUT.invariant_dim]
                                 +self.condition((target['temperature_K'][:,None]-460.)/100.),
                                 current[:,LAYOUT.invariant_dim:]), -1)
        ni = [q.neighbor_index for q in plan.query_list]
        ti = [q.time_index for q in plan.query_list]
        return self.predict(conditioned, None, target['position'][:,ni], target['times'][:,ti])
