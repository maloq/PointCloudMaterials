"""Frozen evaluation with explicit split access, training-only transforms."""
import json
from pathlib import Path
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_table
from .runtime import load_data
from .model import BCR
from .data import pack,balanced_subset
from .evaluate import reconstruction,matching_bins
from .probes import descriptors,frozen_probes,retrieval


def analyze(config,data,output,checkpoint,split,device,probes=False):
    if checkpoint is None:raise ValueError('Specify an explicit frozen --checkpoint')
    patches,manifest=load_data(data);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if saved['data_identity']!=manifest['identity']:raise ValueError('Checkpoint dataset differs')
    if split=='test' and manifest['test_status']!='untouched_lineages':raise ValueError('No untouched final test declared; use development and label historical evidence')
    model=BCR(saved['config']).to(device).eval();model.load_state_dict(saved['model']);model.requires_grad_(False)
    records=manifest['records'];ids={s:[i for i,r in enumerate(records) if r['split']==s] for s in ('train','development','test')}
    if not ids[split]:raise ValueError(f'No {split} anchors')
    targets,cov=descriptors(patches,manifest['radius_A']);bins=matching_bins(cov[ids['train']]);chosen=ids[split]
    evaluation=config.get('evaluation',{});n=min(len(chosen),evaluation.get('anchors',256));chosen=balanced_subset(records,chosen,n)
    if not probes:
        unconditional=None
        if 'unconditional_checkpoint' in evaluation:
            old=torch.load(evaluation['unconditional_checkpoint'],map_location='cpu',weights_only=False)
            if old['config']['arm']!='unconditional' or old['data_identity']!=manifest['identity']:raise ValueError('Unconditional comparator is not matched')
            unconditional=BCR(old['config']).to(device).eval();unconditional.load_state_dict(old['model'])
        if model.arm in ('vicreg','denoising'):raise ValueError('This arm has no trained conditional decoder; use structural probes')
        result=reconstruction(model,[patches[i] for i in chosen],[records[i] for i in chosen],manifest['noise_levels'],
            draws=evaluation.get('draws',2),seed=evaluation.get('noise_bank_seed',731),unconditional=unconditional,
            bins=bins,covariates=cov[chosen],shuffles=evaluation.get('shuffles',4))
        result['matching_bins_training_only']=bins
    else:
        with torch.no_grad():z=torch.cat([model.encode(pack(patches[i:i+8],device)).cpu() for i in range(0,len(patches),8)]).numpy()
        if not ids['development']:raise ValueError('Structural probes need root-disjoint development sources')
        result={}
        for family,y in targets.items():
            result[family]=frozen_probes(z[ids['train']],z[ids['development']],z[chosen],y[ids['train']],y[ids['development']],y[chosen],[records[i] for i in chosen])
            # Covariate-only comparator is retrained with identical budgets.
            result[family]['density_order_only']=frozen_probes(cov[ids['train']],cov[ids['development']],cov[chosen],y[ids['train']],y[ids['development']],y[chosen],[records[i] for i in chosen])
        noncrystalline=np.flatnonzero(cov[chosen,2]<.35)
        if len(noncrystalline):
            result['noncrystalline_q6_below_035']={}
            for family,y in targets.items():
                q=np.array(chosen)[noncrystalline]
                result['noncrystalline_q6_below_035'][family]=frozen_probes(z[ids['train']],z[ids['development']],z[q],y[ids['train']],y[ids['development']],y[q],[records[i] for i in q])
        target=targets['rich'];tr=ids['train'];ref=(target-target[tr].mean(0))/target[tr].std(0).clip(1e-6)
        roots=[records[i]['root'] for i in chosen]
        if all(sum(r!=s for s in roots)>=20 for r in roots):
            mean=z[tr].mean(0);std=z[tr].std(0).clip(1e-6)
            result['recall20']={name:float(retrieval(value,ref[chosen],roots).mean()) for name,value in [('raw',z[chosen]),('train_standardized',(z[chosen]-mean)/std)]}
        else:result['recall20_unavailable']='Fewer than 20 cross-root candidates'
    out=Path(output);(out/'technical').mkdir(parents=True,exist_ok=True);name=('probes' if probes else 'reconstruction')+'-'+split
    (out/'technical'/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
    write_metric_table(result,out,family='bcr',name=name)
