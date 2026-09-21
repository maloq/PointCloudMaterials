"""Predeclared matched comparisons and development-only promotions."""
import copy
import json
from pathlib import Path
from src.project_runtime.paths import resolve_path
from src.data.structural_pretraining.prepare import file_hash
from src.training_methods.neighborhood_jepa.regularization.specs import variants


def encoder_specs(config):
    prototype=next(s for s in variants(config['encoder']) if s['name']=='sig-direct-raw-order')
    settings=[('continued-control',.25,.1,0.,0.),('strong-order',1.,.01,0.,0.),
              ('linear-information',1.,.01,1.,.25),('linear-no-reg',1.,0.,1.,.25)]
    return [dict(prototype,name=name,initialization='information_warm',checkpoint=config['encoder']['warm_checkpoint'],
        encoder_lr=.00005,head_lr=.0005,order_weight=order,regularizer_weight=reg,
        linear_order_weight=linear,linear_angular_weight=angular) for name,order,reg,linear,angular in settings]


def path_specs(config):
    original=resolve_path(config['path']['reference_path_output'])/'technical';promotion=json.loads((original/'promotions.json').read_text());result=[]
    selected={s['method']:s for s in promotion['selection']}
    for method in ('direct','ar_mse','mixture','diffusion'):
        parent=original/'runs'/selected[method]['source'];base=json.loads((parent/'spec.json').read_text())
        variants=('control','shells','history','both','descriptors-only') if method in ('direct','ar_mse') else ('control','both')
        for context in variants:
            spec=copy.deepcopy(base);spec.update(name=f'{method}-{context}-E18',information_context='both' if context=='descriptors-only' else context,
                remove_encoder_context=context=='descriptors-only',
                initial_checkpoint=str(parent/'best.pt'),initial_checkpoint_sha256=file_hash(parent/'best.pt'),
                head_lr=.0001,patience=6,minimum_epochs=6)
            spec['training']['epochs']=18;result.append(spec)
    return result


def choose_encoder(root,specs):
    rows=[]
    for spec in specs:
        folder=Path(root)/'technical/runs'/spec['name'];status=json.loads((folder/'status.json').read_text())
        if status['state']!='complete':raise ValueError(f'Incomplete encoder screen: {spec["name"]}')
        metrics=json.loads((folder/'metrics.json').read_text());rows.append((metrics['selection_score'],spec['name'],spec))
    score,name,spec=min(rows,key=lambda x:x[:2])
    return dict(name=name,score=score,spec=spec,criterion='Minimum development Physical85 + .25 TDA144 + .25 nonlinear order8; no test scores')


def choose_paths(root,specs):
    result=[]
    for method in ('direct','ar_mse','mixture','diffusion'):
        rows=[]
        for spec in specs:
            if spec['method']!=method:continue
            status=json.loads((Path(root)/'technical/runs'/spec['name']/'status.json').read_text())
            if status['state']!='complete':raise ValueError('Incomplete path screen')
            rows.append((spec,status))
        ceiling=min(s['best_selection_physical_mse'] for _,s in rows)*1.1
        spec,status=min([(s,r) for s,r in rows if r['best_selection_physical_mse']<=ceiling],key=lambda x:(x[1]['best_selection_brier'],x[0]['name']))
        new=copy.deepcopy(spec);new['name']=f'{method}-selected-E36';new['training']['epochs']=36;new['patience']=8
        # A fresh matched longer-budget fit from the same original parent, not
        # an optimizer continuation with a silently changed cosine schedule.
        result.append(new)
    return result
