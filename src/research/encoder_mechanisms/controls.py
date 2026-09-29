"""Missing capacity and distance controls; reuse completed likelihood fits."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.experiment_runner.metric_docs import write_metric_table
from src.research.structural_state.common import sha, digest, write_json, save_checkpoint
from src.research.encoder_quality.common import corpus, probe_study
from src.research.encoder_quality.run import (load_encoder, encode, positions_for,
    physical_cache, publish_prediction)
from src.research.encoder_quality.metrics import score, paired_scores, neighbor_forecast
from src.research.supervised_onset.evaluate import readout
from src.research.trajectory_stability.spectrum import source_weights
from src.research.equivariant_context.cache import RetainedCache


def standardize(x,c):
    fit=c.split['train'];w=source_weights(c.pop['source'][fit])
    mean=w@x[fit].astype(float);scale=np.sqrt(w@(x[fit]-mean)**2).clip(1e-5)
    return ((x-mean)/scale).astype(np.float32),dict(mean=mean,scale=scale)


def pca32(x,c):
    x,stats=standardize(x,c);fit=c.split['train'];w=source_weights(c.pop['source'][fit])
    _,vectors=np.linalg.eigh((x[fit].T*w)@x[fit])
    projection=vectors[:,-32:][:,::-1].astype(np.float32)
    return x@projection,dict(stats,projection=projection)


def population_features(config,spec,c,device):
    """Lease only disposable features; original predictions/checkpoints are retained."""
    model,_=load_encoder(spec,device)
    if spec['kind']=='supervised':
        folder=Path(spec['origin'])/'technical/evaluation/O-NLL/best'
        with np.load(folder/'predictions.npz') as p:
            for key in ('sample_id','source','role','event'):np.testing.assert_array_equal(p[key],c.pop[key])
        metrics=json.loads((folder/'metrics.json').read_text())
        if metrics['checkpoint_sha256']!=spec['checkpoint_sha256']:raise ValueError('Saved export has another checkpoint')
        z=np.load(folder/'features.npy')
        ids=np.r_[c.split['train'][:64],c.split['test'][:64]]
        replay=encode(model,positions_for(c,'hot',ids),256,device,compile_first=True)
        np.testing.assert_allclose(replay,z[ids],atol=config['parity_atol'],rtol=config['parity_rtol'])
        return z
    key=digest(dict(checkpoint=spec['checkpoint_sha256'],population=sha(Path(config['population_cache'])/'population.npz'),domain='hot'))
    metadata=dict(artifact='mechanism-controls',checkpoint_sha256=spec['checkpoint_sha256'])
    if 'feature_reuse_record' in spec:
        record=json.loads(Path(spec['feature_reuse_record']).read_text())
        key=record['key'];metadata=dict(artifact='encoder-quality',model=spec['name'],checkpoint_sha256=spec['checkpoint_sha256'])
    managed=RetainedCache(config['feature_cache'],6)
    with managed.lease(key,deadline=time.time()+24*3600,metadata=metadata) as cache:
        path=cache/'population.npy'
        if path.exists():
            z=np.load(path);ids=np.r_[c.split['train'][:64],c.split['test'][:64]]
            replay=encode(model,positions_for(c,'hot',ids),256,device,compile_first=True)
            np.testing.assert_allclose(replay,z[ids],atol=config['parity_atol'],rtol=config['parity_rtol'])
            return z
        values=[];rows=len(c.pop['event'])
        for first in range(0,rows,256):
            x=positions_for(c,'hot',np.arange(first,min(rows,first+256)))
            values.append(encode(model,x,256,device,compile_first=first==0))
        z=np.concatenate(values);np.save(path,z)
        return z


def retained_initial(config):
    spec=next(s for s in config['models'] if s['name']=='epi_variance-pretrained')
    if sha(spec['checkpoint'])!=spec['checkpoint_sha256']:raise ValueError('Epi source changed')
    saved=torch.load(spec['checkpoint'],weights_only=False,map_location='cpu')
    if saved['reference'] is None:raise ValueError('Matched initial encoder was not retained')
    path=Path(config['output'])/'technical/inputs/epi-matched-initial.pt'
    if not path.exists():
        save_checkpoint(path,dict(encoder=saved['reference'],encoder_config=saved['encoder_config'],
            identity=digest(dict(parent=spec['checkpoint_sha256'],field='reference')),
            epoch=0,selection='Exact retained frozen initial Epi reference, including train-only pool normalization'))
    return dict(spec,name='epi-matched-initial',checkpoint=str(path),checkpoint_sha256=sha(path),kind='initial')


def soap_features(config,c):
    """Constant-species SOAP on the same finite current patch; no periodic halo."""
    from ase import Atoms
    from dscribe.descriptors import SOAP
    dest=Path(config['output'])/'technical/physical';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'hot-soap50.npy';receipt=path.with_suffix('.json')
    if path.exists():
        if sha(path)!=json.loads(receipt.read_text())['sha256']:raise ValueError('SOAP cache changed')
        return np.load(path)
    descriptor=SOAP(species=['H'],periodic=False,r_cut=8.,n_max=4,l_max=4,sigma=.3,dtype='float32')
    values=[]
    for first in range(0,len(c.pop['event']),256):
        ids=np.arange(first,min(first+256,len(c.pop['event'])))
        patches=positions_for(c,'hot',ids)
        systems=[]
        for x in patches:
            x=x[np.linalg.norm(x,axis=1)<8.]
            systems.append(Atoms(numbers=np.ones(len(x),int),positions=x,pbc=False))
        values.append(descriptor.create(systems,centers=[[0]]*len(systems),n_jobs=1)[:,0,:])
    result=np.concatenate(values)
    if result.shape!=(len(c.pop['event']),50) or not np.isfinite(result).all():raise ValueError('Unexpected SOAP50 features')
    np.save(path,result);write_json(receipt,dict(sha256=sha(path),parameters=dict(r_cut=8.,n_max=4,l_max=4,sigma=.3),
        inputs='Same 80 candidates within 8 A; all atoms assigned constant pseudo-species H; center included; no halo or covariates'))
    return result


@torch.no_grad()
def physical_neighbors(c,x,y,device):
    """Fixed k=64 physical readout, independent of onset and test labels."""
    fit=c.split['train'];test=c.split['test'];reference=torch.as_tensor(x[fit],device=device)
    values=[];weights=source_weights(c.pop['source'][fit]);weights/=weights.mean()
    rnorm=reference.square().sum(1)
    for first in range(0,len(test),256):
        ids=test[first:first+256];q=torch.as_tensor(x[ids],device=device)
        distance=(q.square().sum(1)[:,None]+rnorm[None]-2*q@reference.T).clamp_min_(0)
        ni=distance.topk(64,largest=False).indices.cpu().numpy();w=weights[ni]
        values.append((w[:,:,None]*y[fit[ni]]).sum(1)/w.sum(1)[:,None])
    prediction=np.concatenate(values);error=((prediction-y[test])**2).mean(1)
    per={str(s):float(error[c.pop['source'][test]==s].mean()) for s in np.unique(c.pop['source'][test])}
    return dict(mean_source_nmse=float(np.mean(list(per.values()))),per_source=per,k=64,
        targets='32 train-standardized geometric descriptors; held-out sources',
        caveat='Descriptors are withheld targets for z distances; descriptor-distance prediction is a geometric ceiling control')


def run(config,identity,previous_output,device):
    root=Path(config['output']);c=corpus(config);physical=physical_cache(config,c,'hot',device)
    y,_=standardize(physical,c);old=Path(previous_output)
    specs=[s for s in config['models'] if s['domain']=='hot']+[retained_initial(config)]
    basis=np.linalg.qr(np.random.default_rng(config['seed']+73).normal(size=(128,32)))[0].astype(np.float32)
    np.save(root/'technical/projection128x32.npy',basis)
    shared={}
    for name,x in [('density26',physical[:,:26]),('soap50',soap_features(config,c))]:
        owner=probe_study(config,identity,name,dict(encoder=None,predictor=name,external_inputs=[],history=1,motion=False,relaxation=False))
        for kind in ('linear','mlp','mlp256'):
            risk=readout(owner,c,x,name,kind,device)
            metric,_=publish_prediction(owner,c,name,kind,risk,dict(selected_by='validation hazard NLL',external_inputs=[]))
            shared[f'{name}-{kind}']=metric
    write_metric_table(shared,root/'analyses/shared-controls',family='encoder_mechanisms')
    for spec in specs:
        name=spec['name'];dest=root/'analyses/readout-controls'/name
        if (dest/'technical/complete.json').exists():continue
        if spec['kind']=='pretrained':spec=dict(spec,feature_reuse_record=str(old/'technical/evaluations'/name/'feature-cache.json'))
        z=population_features(config,spec,c,device)
        normalized,stats=standardize(z,c)
        owner=probe_study(config,identity,name,dict(encoder=spec,predictors=['z128','z128+d32','z128+P32(z)'],
            spatial='80 candidates; 8 A mask; 5 A edges; two blocks; no halo',history=1,motion=False,conditions=[],
            relaxation='observed inference; paired structural pretraining where recorded',external_inputs=[]))
        dest.mkdir(parents=True,exist_ok=True);np.savez(dest/'projection-normalization.npz',**stats,projection=basis)
        variants={'z':z,'joint':np.c_[z,physical],'redundant':np.c_[z,normalized@basis]}
        predictions={};metrics={};reused=[]
        for input_name,x in variants.items():
            for kind in ('linear','mlp','mlp256'):
                key=f'{input_name}-{kind}'
                previous=old/'technical/models'/name/'technical/evaluation'/input_name/kind
                if kind in ('linear','mlp') and input_name!='redundant' and (previous/'predictions.npz').exists():
                    with np.load(previous/'predictions.npz') as p:
                        for field in ('sample_id','source','event','role'):np.testing.assert_array_equal(p[field],c.pop[field])
                        risks=p['risks']
                    record,calibrated=score(c,risks)
                    reused.append(dict(input=key,path=str(previous),sha256=sha(previous/'predictions.npz')))
                else:
                    risks=readout(owner,c,x,input_name,kind,device)
                    record,calibrated=publish_prediction(owner,c,input_name,kind,risks,
                        dict(inputs=input_name,selected_by='validation hazard NLL',external_inputs=[]))
                metrics[key]=record;predictions[key]=(risks,calibrated)
        comparisons={}
        for kind in ('linear','mlp','mlp256'):
            for baseline in ('z','redundant'):
                comparisons[f'joint-minus-{baseline}-{kind}']={scale:paired_scores(c,predictions[f'joint-{kind}'][j],
                    predictions[f'{baseline}-{kind}'][j],config['bootstrap'],config['seed']) for j,scale in enumerate(('raw','calibrated'))}
        pca,pca_stats=pca32(z,c);np.savez(dest/'pca32.npz',**pca_stats)
        for distance,x in [('raw128',z),('standard128',normalized),('pca32',pca),('descriptor32',y)]:
            candidates=[]
            for k in config['neighbors']:
                risk=neighbor_forecast(c,x,k,config['neighbor_prior_strength'],device)
                ids=c.split['selection'];p=np.diff(np.c_[np.zeros(len(risk)),risk.astype(float),np.ones(len(risk))],axis=1)
                nll=float(source_weights(c.pop['source'][ids])@-np.log(p[ids,c.pop['event'][ids]].clip(1e-12)))
                candidates.append((nll,k,risk))
            nll,k,risk=min(candidates,key=lambda v:v[0])
            metric,_=publish_prediction(owner,c,distance,'neighbors',risk,dict(k=k,selection_nll=nll,
                candidates=[dict(nll=a,k=b) for a,b,_ in candidates],distance=distance))
            metric['physical_readout']=physical_neighbors(c,x,y,device);metrics[f'distance-{distance}']=metric
        result=dict(model=spec,readouts=metrics,paired=comparisons,reused=reused,
            inference='Finite readout/metric comparisons do not establish information-theoretic absence')
        write_json(dest/'technical/results.json',result)
        write_metric_table(result,dest,family='encoder_mechanisms')
        write_json(dest/'technical/complete.json',dict(identity=identity,model=name))
        print(json.dumps(dict(stage='controls',model=name,state='complete')),flush=True)
