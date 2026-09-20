"""Matched one-seed screens, with promotion on fixed physical information retention."""
import copy

def screens(config):
    scale=config.get('lr_scale', 4.0)
    base=dict(architecture='mace',neighbors=6,prediction='temporal',previous_context=False,
        regularizer='sigreg',prediction_weight=.1,sigreg_weight=.01,temporal_pair=True,
        head_lr=.002*scale,encoder_lr=.0002*scale,future_weight=.25,epochs=config['screen_epochs'])
    result=[]
    def add(name,**changes):
        s=dict(base,**changes);s['name']=name;result.append(s)
    add('spatial-six',prediction='spatial')
    add('spacetime-six')
    add('spacetime-six-history',previous_context=True)
    add('spacetime-four',neighbors=4)
    add('spacetime-four-history',neighbors=4,previous_context=True)
    add('spacetime-pred003',prediction_weight=.03)
    add('spacetime-pred03',prediction_weight=.3)
    add('spacetime-sig0003',sigreg_weight=.003)
    add('spacetime-sig003',sigreg_weight=.03)
    add('spacetime-lr0001',encoder_lr=.0001*scale,head_lr=.001*scale)
    add('spacetime-lr00005',encoder_lr=.00005*scale,head_lr=.0005*scale)
    add('spatial-four',prediction='spatial',neighbors=4)
    add('spacetime-history-pred003',previous_context=True,prediction_weight=.03)
    add('spacetime-no-fixed-future',future_weight=0.)
    add('anchors-sigreg',prediction='none')
    add('anchors-vicreg-temporal',prediction='none',regularizer='vicreg',sigreg_weight=.1)
    add('anchors-vicreg-spatial',prediction='none',regularizer='vicreg',temporal_pair=False,sigreg_weight=.1)
    return result


def promotions(specs,statuses,epochs):
    # One control and two predictive settings; never rank moving latent targets.
    controls=[s for s in specs if s['prediction']=='none'];predictive=[s for s in specs if s['prediction']!='none']
    chosen=sorted(controls,key=lambda s:statuses[s['name']]['selection_score'])[:1]+sorted(predictive,key=lambda s:statuses[s['name']]['selection_score'])[:2]
    result=[]
    for s in chosen:
        item=copy.deepcopy(s);item.update(name=s['name']+'-long',epochs=epochs,promoted_from=s['name']);result.append(item)
    return result
