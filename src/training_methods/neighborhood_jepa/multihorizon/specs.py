"""Matched additions to the three existing MLP regularizer comparisons."""
from ..v2.contracts import variants as base_variants


def variants(config):
    base=base_variants(config)[-1]
    return [dict(base,name=f'{short}-mlp-order-h369',regularizer=regularizer,projector='mlp_ln',export_norm='layernorm',
        regularizer_weight=.1,order_weight=.25,initialization='warm',encoder_lr=.0001,head_lr=.001,
        sigreg_weight=0.,updates=config['updates'],horizons_ps=config['horizons_ps'],
        family_weights=dict(present_neighbors=1.,future_neighbors=1.,future_center=.25),future_weight=.0625)
        for short,regularizer in [('sig','sigreg'),('vic','vicreg'),('epi','epi')]]
