"""Validation-only screening and longer, independently scheduled confirmation fits."""
import copy
import fcntl
import json
from src.data.structural_pretraining.prepare import save_json


def variants(settings):
    result=[]
    def add(mode,label,**changes):
        s=dict(protocol='adaptive_v1',mode=mode,history_ps=12,radius_A=25,aggregation='attention',attention='factorized',
            equivariant=False,baseline=None,repeat=False,head_width=128,depth=1,heads=4,geometry_bias=True,norm_eps=1e-8,
            head_lr=5e-4,encoder_lr=1e-6 if mode=='finetune' else (3e-5 if mode=='scratch' else 0.),
            warmup_epochs=1 if mode=='finetune' else 0,encoder_clip=1.,weight_decay=1e-4,
            training=dict(budget='epochs',epochs=settings['screen_epochs'],sources=90,window_fraction=1.))
        s.update(changes);s['name']=f'{mode}-{label}-E{settings["screen_epochs"]}'
        # Parameter-identical reference duplicates need no extra random rerun.
        comparable={k:v for k,v in s.items() if k!='name'}
        if any({k:v for k,v in p.items() if k!='name'}==comparable for p in result):return
        result.append(s)
    add('finetune','reference');add('scratch','reference');add('frozen','reference')
    for mode in ('finetune','scratch'):
        for lr in settings['encoder_lrs'][mode]:add(mode,f'encoder-lr{lr:g}',encoder_lr=lr)
        for lr in settings['head_lrs']:add(mode,f'head-lr{lr:g}',head_lr=lr)
        for attention in ('mean','spatial','temporal','joint'):add(mode,attention,attention=attention)
        add(mode,'depth2',depth=2);add(mode,'joint-depth2',attention='joint',depth=2)
        add(mode,'width256',head_width=256);add(mode,'heads8',heads=8)
        add(mode,'no-spatial-bias',geometry_bias=False)
        for radius in (12,18):add(mode,f'radius{radius}',radius_A=radius)
        for history in (0,3,48):add(mode,f'history{history}',history_ps=history)
        add(mode,'repeated48',history_ps=48,repeat=True)
        add(mode,'norm-eps1e-6',norm_eps=1e-6)
    add('finetune','warmup0',warmup_epochs=0);add('finetune','warmup2',warmup_epochs=2)
    add('finetune','tensor',equivariant=True)
    for attention in ('mean','spatial','temporal','joint'):add('frozen',attention,attention=attention)
    add('frozen','tensor',equivariant=True)
    return result


def expanded_tasks(config,root):
    screens=variants(config['refinement']);path=root/'promotions.json'
    with (root/'promotion.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():return screens+json.loads(path.read_text())['tasks']
        scores=[]
        for spec in screens:
            status=root/'runs'/spec['name']/'status.json'
            if not status.exists():return screens
            record=json.loads(status.read_text())
            if record['state']=='failed':raise RuntimeError(f'Screen failed; promotion blocked: {spec["name"]}')
            if record['state']!='complete':return screens
            # Never read test metrics to rank, extend, or stop these fits.
            scores.append((record['best_selection_nll'],spec))
        tasks=[];selection=[]
        for mode in ('finetune','scratch','frozen'):
            count=1 if mode=='frozen' else config['refinement']['promote_per_mode']
            chosen=sorted((row for row in scores if row[1]['mode']==mode),key=lambda row:(row[0],row[1]['name']))[:count]
            for rank,(score,spec) in enumerate(chosen,1):
                selection.append(dict(mode=mode,rank=rank,screen=spec['name'],selection_nll=score))
                for epochs in config['refinement']['long_epochs']:
                    item=copy.deepcopy(spec);item['name']=f'{mode}-rank{rank}-E{epochs}'
                    item['training']['epochs']=epochs;item['promoted_from']=spec['name'];tasks.append(item)
        save_json(path,dict(criterion='Lowest selection NLL only; no test-based choices; restart with the longer cosine schedule',selection=selection,tasks=tasks))
        save_json(root/'queue.json',screens+tasks)
        return screens+tasks
