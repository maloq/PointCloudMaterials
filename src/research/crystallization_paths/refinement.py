"""One-seed targeted screens, then validation-only quality-constrained promotions."""
import copy
import fcntl
import json
from src.data.structural_pretraining.prepare import save_json


def screens(config,reference):
    result=[];settings=config['refinement']
    def add(method,label,**changes):
        spec=copy.deepcopy(reference['spec']);spec.update(protocol='path_refinement_v2',method=method,
            state_weight=1.,event_weight=1.,residual_anchor=False,present_weight=0.,dropout=0.,motion_input=False,
            teacher_mode='likelihood' if method=='ar_gaussian' else 'scheduled',teacher_epochs=6.,
            gaussian_rank=0,mixture_style='free',mixture_boundaries=[12,64,128],
            diffusion_prediction='v',terminal_fraction=.2,onset_projection='step',diffusion_steps=16,ema_decay=0.,
            patience=4,minimum_epochs=4,
            training=dict(budget='epochs',epochs=settings['screen_epochs'],sources=90,window_fraction=1.))
        spec.update(changes);spec['name']=f'{method}-{label}-E{settings["screen_epochs"]}';result.append(spec)
    for method in ('direct','ar_mse'):
        add(method,'control')
        add(method,'lr1e-4',head_lr=1e-4)
        add(method,'regularized',dropout=.1,weight_decay=1e-3)
        add(method,'state025',state_weight=.25)
        add(method,'anchor',residual_anchor=True,present_weight=.25)
        add(method,'history48',history_ps=48)
        add(method,'motion',motion_input=True)
    add('ar_mse','free',teacher_mode='free')
    add('ar_mse','teacher2',teacher_epochs=2.)
    add('ar_gaussian','likelihood')
    add('ar_gaussian','rank8',gaussian_rank=8)
    add('ar_gaussian','rank16',gaussian_rank=16)
    add('ar_gaussian','lr1e-4',head_lr=1e-4)
    add('mixture','control')
    add('mixture','stratified4',mixture_style='stratified')
    add('mixture','stratified2',mixture_style='stratified',mixture_boundaries=[128])
    add('mixture','state3',state_weight=3.)
    add('diffusion','v128',patience=6)
    add('diffusion','v384',head_width=384,heads=6,patience=6)
    add('diffusion','x0',diffusion_prediction='x0',patience=6)
    add('diffusion','event3',event_weight=3.,patience=6)
    add('diffusion','steps32',diffusion_steps=32,patience=6)
    add('diffusion','ema',ema_decay=.999,patience=6)
    return result


def expanded_tasks(config,root):
    reference=json.loads((root/'reference-selection.json').read_text());base=screens(config,reference)
    with (root/'promotion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);path=root/'promotions.json'
        if path.exists():return base+json.loads(path.read_text())['tasks']
        results=[]
        for spec in base:
            status=root/'runs'/spec['name']/'status.json'
            if not status.exists():return base
            record=json.loads(status.read_text())
            if record['state']=='failed':raise RuntimeError(f'Failed screen blocks promotion: {spec["name"]}')
            if record['state']!='complete':return base
            results.append((spec,record))
        chosen=[];promoted=[]
        for method in ('direct','ar_mse','ar_gaussian','mixture','diffusion'):
            candidates=[(s,r) for s,r in results if s['method']==method]
            best_physical=min(r['best_selection_physical_mse'] for _,r in candidates)
            eligible=[(s,r) for s,r in candidates if r['best_selection_physical_mse']<=1.1*best_physical]
            spec,record=min(eligible,key=lambda x:(x[1]['best_selection_brier'],x[0]['name']))
            chosen.append(dict(method=method,source=spec['name'],selection_brier=record['best_selection_brier'],
                selection_physical_mse=record['best_selection_physical_mse'],physical_ceiling=1.1*best_physical))
            spec=copy.deepcopy(spec);spec['promoted_from']=spec['name'];spec['training']['epochs']=config['refinement']['long_epochs']
            spec['name']=f'{method}-selected-E{spec["training"]["epochs"]}';spec['patience']=8;promoted.append(spec)
        save_json(path,dict(criterion='Lowest selection Brier among candidates within 10% of the best selection physical MSE in their family; never test metrics',selection=chosen,tasks=promoted))
        save_json(root/'queue.json',base+promoted);return base+promoted
