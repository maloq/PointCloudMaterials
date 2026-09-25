"""Matched predictive, structural, exact-lag and noise diagnostics after fitting."""
import json
import time
from types import SimpleNamespace
import numpy as np
import torch
from src.project_runtime.paths import resolve_path
from src.data.fixed_cohort.dataset import read_release
from src.research.structural_state.common import sha,write_json
from src.research.supervised_onset.common import Study as BaseStudy
from src.research.supervised_onset.data import Corpus
from src.research.supervised_onset.train import make_model,make_banks,encode
from src.research.supervised_onset.evaluate import evaluate_arm,readout,save_result,score_predictions
from src.research.trajectory_stability.spectrum import spectrum,source_weights,analyze
from src.experiment_runner.metric_docs import write_metric_table
from .geometry import graph,physical_targets


def population_view(pop,ids):
    selected={k:v[ids] for k,v in pop.items()}
    return SimpleNamespace(pop=selected,split={role:np.flatnonzero(selected['role']==role)
        for role in ('train','selection','calibration','test')})


def legacy_scores(campaign,pop,risks,calibrated):
    fixed,_=read_release(campaign.config['fixed_dataset']['root'])
    ids=np.load(fixed/'benchmark/legacy_order.npy')
    return score_predictions(population_view(pop,ids),risks[ids],campaign.config['bootstrap'],
        campaign.config['seed'],calibrated[ids])


def targets(corpus,domain,device,chunk):
    arrays=corpus.arrays[domain];offset=arrays['offsets'];result=[]
    for begin in range(0,len(offset)-1,chunk):
        ids=np.arange(begin,min(begin+chunk,len(offset)-1))
        x=np.full((len(ids),80,3),100.,dtype=np.float32)
        for j,i in enumerate(ids):x[j,:offset[i+1]-offset[i]]=arrays['positions'][offset[i]:offset[i+1]]
        result.append(physical_targets(torch.as_tensor(x,device=device)).cpu().numpy())
    return np.concatenate(result)


def physical_readout(features,target,corpus):
    fit=corpus.split['train'];weights=source_weights(corpus.pop['source'][fit])
    x=features.astype(float);y=target.astype(float)
    mean=weights@x[fit];scale=np.sqrt(weights@(x[fit]-mean)**2).clip(1e-5)
    ym=weights@y[fit];ys=np.sqrt(weights@(y[fit]-ym)**2).clip(1e-5)
    design=np.c_[np.ones(len(x)),(x-mean)/scale];standard=(y-ym)/ys
    regularizer=np.eye(design.shape[1])*1e-3;regularizer[0,0]=0
    beta=np.linalg.solve((design[fit].T*weights)@design[fit]+regularizer,
        (design[fit].T*weights)@standard[fit])
    prediction=design@beta;output={}
    for role in ('selection','test'):
        ids=corpus.split[role];w=source_weights(corpus.pop['source'][ids]);err=(prediction[ids]-standard[ids])**2
        variance=w@(standard[ids]-w@standard[ids])**2
        output[role]=dict(radial_standardized_mse=float((w@err[:,:24]).mean()),
            count_standardized_mse=float((w@err[:,24:26]).mean()),
            angular_standardized_mse=float((w@err[:,26:]).mean()),
            r2_per_target=[float(1-e/v) if v>1e-12 else None for e,v in zip(w@err,variance,strict=True)])
    return output


@torch.no_grad()
def dense_diagnostics(campaign,base,corpus,model,features,device,deadline):
    cache=campaign.cache/'dense-observed';manifest=json.loads((cache/'manifest.json').read_text())
    if manifest['release_identity']!=campaign.config['fixed_dataset']['identity']:raise ValueError('Dense release changed')
    all_z=[];sources=[];atoms=[];frames=[];labels=[];chunk=base.config['training']['microbatch']
    for source in manifest['sources']:
        if time.time()>deadline-180:raise TimeoutError('Dense evaluation checkpoint boundary; resume this stage')
        folder=cache/str(source['source'])
        for name,checksum in source['files'].items():
            if sha(folder/name)!=checksum:raise ValueError(f'Changed dense geometry: {folder/name}')
        x=np.load(folder/'positions.npy',mmap_mode='r').reshape(-1,80,3);encoded=[]
        for start in range(0,len(x),chunk):
            value=torch.as_tensor(np.array(x[start:start+chunk]),device=device)
            encoded.append(model.encoder(graph(value,model.encoder)).cpu().numpy())
        all_z.append(np.concatenate(encoded))
        with np.load(folder/'observations.npz') as a:
            sources.append(a['source']);atoms.append(a['atom']);frames.append(a['frame']);labels.append(a['labels'])
    z=np.concatenate(all_z);source=np.concatenate(sources);atom=np.concatenate(atoms);time_ps=np.concatenate(frames)*.75
    fit=corpus.split['train'];n=len(fit)
    # Training reference and dense test remain different sources. Synthetic reference
    # times are audit-only identifiers; no temporal pairs use reference rows.
    values=np.concatenate((features[fit],z));src=np.r_[corpus.pop['source'][fit],source]
    ats=np.r_[corpus.pop['atom'][fit],atom];times=np.r_[corpus.pop['frame'][fit]*.75,time_ps]
    liquid=np.r_[np.ones(n,dtype=bool),~np.isin(np.concatenate(labels),[1,2,3])]
    result=analyze(values,src,ats,times,np.arange(n),np.arange(n,len(values)),lags_ps=[.75],domains={'noncrystalline':liquid})
    # The dense test trajectory rank is distinct from the fixed at-risk dataset rank.
    return result,dict(z=z,source=source,atom=atom,time_ps=time_ps,labels=np.concatenate(labels))


def run(campaign,context,domain,device,deadline):
    base=BaseStudy(resolve_path(context.config['base_configs'][domain]));base.bind()
    arm=base.config['arms'][0]['name'];dest=base.technical/'full-evaluation';dest.mkdir(exist_ok=True)
    done=dest/'complete.json'
    if done.exists():
        record=json.loads(done.read_text())
        if record['identity']!=base.identity or record['metrics_sha256']!=sha(dest/'metrics.json'):
            raise ValueError('Completed evaluation identity or metrics changed')
        return
    corpus=Corpus(base);banks=make_banks(base,corpus,device)
    if not (base.technical/'runs'/arm/'evaluation-complete.json').exists():
        evaluate_arm(base,corpus,banks,arm,device,deadline)
    selected=base.technical/'evaluation'/arm/'best'
    features=np.load(selected/'features.npy')
    with np.load(selected/'predictions.npz') as p:
        np.testing.assert_array_equal(p['sample_id'],corpus.pop['sample_id'])
        legacy=legacy_scores(campaign,corpus.pop,p['risks'],p['calibrated_risks'])
    physical=targets(corpus,domain,device,base.config['training']['microbatch'])
    retention=physical_readout(features,physical,corpus)
    # Independent frozen geometry control, trained by the same likelihood recipe.
    name=f'physical-{domain}'
    risks=readout(base,corpus,physical,name,'mlp',device)
    save_result(base,corpus,name,'mlp',risks,input='32 current-geometry targets; no learned encoder',
        selected_by='minimum selection hazard NLL; no temperature/time inputs')
    fit=corpus.split['train'];w=source_weights(corpus.pop['source'][fit])
    constant=np.tile(np.array([w@(corpus.pop['event'][fit]<=k) for k in range(5)]),(len(features),1))
    save_result(base,corpus,'constant','prior',constant,input='training-source event prior only')
    result=dict(identity=base.identity,release_identity=campaign.config['fixed_dataset']['identity'],
        encoder_domain=domain,legacy16=legacy,physical_linear_readout=retention,
        encoder_diagnostics=json.loads((selected/'diagnostics.json').read_text()))
    if domain=='hot':
        model=make_model(base,corpus,base.arm(arm),device)
        model.load_state_dict(torch.load(base.technical/'runs'/arm/'best.pt',map_location=device,weights_only=False)['model']);model.eval()
        from src.models.encoders.spatial_mace import compile_spatial_encoder
        compile_spatial_encoder(model.encoder,banks['hot'].batch(fit[:base.config['training']['microbatch']]))
        del banks;torch.cuda.empty_cache()
        dynamics,values=dense_diagnostics(campaign,base,corpus,model,features,device,deadline)
        np.savez_compressed(dest/'dense-embeddings.npz',**values)
        result['dynamics_075ps']=dynamics
    else:
        result['dynamics_075ps']=dict(available=False,reason='No dense relaxed trajectories at 0.75 ps; observed-input substitution is not the deployed encoder.')
    for variant in context.config['variants']:
        p=context.root/f'{domain}-{variant}'/'technical/predictions.npz'
        with np.load(p) as values:
            np.testing.assert_array_equal(values['sample_id'],corpus.pop['sample_id'])
            record=legacy_scores(campaign,corpus.pop,values['risks'],values['calibrated'])
        write_json(p.parent/'legacy16-metrics.json',record)
    write_json(dest/'metrics.json',result);write_metric_table(result,base.root,family='encoder_context',name='full-evaluation')
    # Associated scientific evaluation updates the original encoder run; never a debug run.
    import wandb
    receipt=json.loads((base.technical/'wandb'/arm/'run.json').read_text())
    run=wandb.Api(timeout=60).run(f"{receipt['entity']}/{receipt['project']}/{receipt['id']}")
    spec=result['encoder_diagnostics']['test_spectrum']
    summary={'embedding/test_d95':spec['d95'],'embedding/test_participation_rank':spec['participation_rank'],
        'evaluation/full_complete':True}
    if domain=='hot':
        lag=result['dynamics_075ps']['domains']['all']['lags']['0.75']
        summary.update({'embedding/jump_rms_075ps':lag['rms_jump'],
            'embedding/movement_d95_075ps':lag['movement']['d95']})
    run.summary.update(summary)
    write_json(done,dict(state='complete',identity=base.identity,metrics_sha256=sha(dest/'metrics.json')))
