"""Predeclared one-seed factorial comparisons; no test-selected hyperparameters."""
from ..v2.contracts import variants as original


def variants(config):
    if config.get('protocol')=='neighborhood_jepa_multihorizon_v1':
        from ..multihorizon.specs import variants as multi_variants
        return multi_variants(config)
    prototype=original(config)[-1]
    rows=[
        ('sig-mlp-no-order','sigreg','mlp_ln','layernorm',.1,0.,'warm'),
        ('none-mlp-order','none','mlp_ln','layernorm',0.,.25,'warm'),
        ('sig-mlp-order','sigreg','mlp_ln','layernorm',.1,.25,'warm'),
        ('sig-mlp-strong-order','sigreg','mlp_ln','layernorm',1.,.25,'warm'),
        ('sig-linear-order','sigreg','linear','layernorm',.1,.25,'warm'),
        ('sig-plain-order','sigreg','mlp_plain','layernorm',.1,.25,'warm'),
        ('sig-direct-raw-order','sigreg','identity','raw',.1,.25,'warm'),
        ('vic-mlp-order','vicreg','mlp_ln','layernorm',.1,.25,'warm'),
        ('vic-linear-order','vicreg','linear','layernorm',.1,.25,'warm'),
        ('vic-direct-raw-order','vicreg','identity','raw',.1,.25,'warm'),
        ('epi-mlp-order','epi','mlp_ln','layernorm',.1,.25,'warm'),
        ('epi-linear-order','epi','linear','layernorm',.1,.25,'warm'),
        ('epi-direct-order','epi','identity','layernorm',.1,.25,'warm'),
        ('epi-mlp-strong-order','epi','mlp_ln','layernorm',.3,.25,'warm'),
        ('scratch-sig-mlp-order','sigreg','mlp_ln','layernorm',.1,.25,'scratch'),
        ('scratch-vic-direct-raw-order','vicreg','identity','raw',.1,.25,'scratch'),
    ]
    return [dict(prototype,name=name,regularizer=reg,projector=proj,export_norm=norm,
                 regularizer_weight=weight,order_weight=order,initialization=init,
                 encoder_lr=.0001,head_lr=.001,sigreg_weight=0.,updates=config['updates'])
            for name,reg,proj,norm,weight,order,init in rows]
