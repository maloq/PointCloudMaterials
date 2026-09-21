"""Joint invariant/equivariant future targets, with fixed physical/TDA anchors."""
import torch
from ..regularization.objective import Objective as OriginalObjective
from ..v2.geometry import scaled_error
from .data import view_plan


def masked_horizon_means(values,valid):
    if values.shape!=valid.shape:raise ValueError(f'Horizon loss/mask shape mismatch: {values.shape}, {valid.shape}')
    counts=valid.sum(0)
    # A completely censored minibatch contributes no loss for that horizon.
    return torch.where(valid,values,0.).sum(0)/counts.clamp_min(1)


class Objective(OriginalObjective):
    def __init__(self,manifest,order_manifest,spec):
        super().__init__(manifest,order_manifest,spec)
        self.plan=view_plan(spec)

    def forward(self,model,encoded,target):
        _,terms=super().forward(model,encoded,target)
        z=encoded.reshape(len(target['group']),len(self.plan.views),-1)
        current=z[:,self.plan.slot(1,0)]
        actual=z[:,[self.plan.slot(t,0) for t in (3,4,5)]]
        inv,eq=model.future_embeddings(current,target['temperature_K'])
        valid=target['future_valid']
        inv_error=masked_horizon_means((inv-actual[:,:,:128]).square().mean(-1),valid)
        eq_error=masked_horizon_means(scaled_error(eq,actual[:,:,128:],model.encoder.geometry_scales).mean(-1),valid)
        group=target['group'].repeat_interleave(3)
        p=self.physical_errors(model,inv.flatten(0,1),target['future_physical'].flatten(0,1),group).reshape(-1,3)
        t=self.tda_errors(model,inv.flatten(0,1),target['future_tda'].flatten(0,1),group).reshape(-1,3)
        physical=masked_horizon_means(p,valid);tda=masked_horizon_means(t,valid)
        # Original .1 future-center family weight is shared equally by four lags.
        # Present-neighbor and .75 ps neighbor families retain their old weights.
        terms['future_embeddings_invariant']=.075*inv_error.mean()
        terms['future_embeddings_equivariant']=.075*eq_error.mean()
        # Total fixed-future weight also remains .25 across all four horizons.
        terms['future_anchors']=.1875*(physical+.25*tda).mean()
        for h,ps in enumerate(self.spec['horizons_ps']):
            self.diagnostics.update({f'future/{ps:g}ps/{name}':float(value[h].detach()) for name,value in
                [('invariant_mse',inv_error),('equivariant_mse',eq_error),('physical',physical),('tda',tda)]})
        return sum(terms.values()),terms
