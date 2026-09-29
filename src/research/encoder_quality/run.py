"""Apply current native MACE exports to fixed structural and predictive assays."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.research.structural_state.common import sha,digest,write_json
from src.research.equivariant_context.cache import RetainedCache
from src.research.supervised_onset.model import CapacityEncoder
from src.research.encoder_context.geometry import graph,physical_targets
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.research.geoframe_evolution.evaluate import frame_metrics
from src.research.encoder_parameter_search.metrics import frame as liquid_metrics
from src.research.supervised_onset.evaluate import readout
from src.research.trajectory_stability.spectrum import source_weights
from src.experiment_runner.metric_docs import write_metric_table
from .common import load,bind,corpus,probe_study,static_coordinate_agreement
from .metrics import score,paired_scores,neighbor_forecast,movement_relation


def load_encoder(spec,device):
    if sha(spec['checkpoint'])!=spec['checkpoint_sha256']:
        raise ValueError(f'Checkpoint changed: {spec["name"]}')
    repo=Path(__file__).resolve().parents[3]
    for name in ('src/models/encoders/spatial_mace.py','src/models/encoders/mace_backend.py',
                 'src/research/supervised_onset/model.py','src/research/encoder_context/geometry.py'):
        if sha(repo/name)!=sha(Path(spec['producer'])/name):
            raise ValueError(f'Native inference producer differs from checkpoint: {name}')
    saved=torch.load(spec['checkpoint'],weights_only=False,map_location='cpu')
    if spec['kind'] in ('supervised','adapted'):
        state={k.removeprefix('encoder.'):v for k,v in saved['model'].items() if k.startswith('encoder.')}
        metadata=dict(selected_update=saved['update'],selection=saved['best'],branch=saved['branch'])
    elif spec['kind'] in ('pretrained','initial'):
        state=saved['encoder'];metadata=dict(epoch=saved['epoch'],selection=saved['selection'],
            branch='initialized' if spec['kind']=='initial' else 'self_supervised')
    else:raise ValueError(spec['kind'])
    model=CapacityEncoder(**saved['encoder_config']).to(device)
    model.load_state_dict(state,strict=True);model.eval().requires_grad_(False)
    return model,dict(metadata,encoder_config=saved['encoder_config'],checkpoint_identity=saved['identity'])


@torch.no_grad()
def encode(model,positions,chunk,device,compile_first=False):
    result=[]
    for first in range(0,len(positions),chunk):
        part=np.array(positions[first:first+chunk],dtype=np.float32,copy=True)
        if part.shape[1:]!=(80,3) or not np.isfinite(part).all():raise ValueError('Require finite 80-candidate centered patches')
        if np.any(part[:,0]):raise ValueError('Focal atom must be first and exactly at zero')
        n=len(part)
        padded=np.full((chunk,80,3),100.,dtype=np.float32);padded[:n]=part
        value=torch.as_tensor(padded,device=device);g=graph(value,model)
        if compile_first and first==0:compile_spatial_encoder(model,g)
        z=model(g)[:n].cpu().numpy()
        if not np.isfinite(z).all():raise FloatingPointError('Nonfinite native export')
        result.append(z)
    return np.concatenate(result)


def positions_for(corpus,domain,ids):
    a=corpus.arrays[domain];offset=a['offsets']
    x=np.full((len(ids),80,3),100.,dtype=np.float32)
    for row,i in enumerate(ids):
        part=a['positions'][offset[i]:offset[i+1]]
        if len(part)>80 or np.any(part[0]):raise ValueError('Unexpected native prediction support')
        x[row,:len(part)]=part
    return x


def geometric_checks(model,positions,config,device):
    x=positions[:128].copy();chunk=config['batch_size'];z=encode(model,x,chunk,device)
    rng=np.random.default_rng(config['seed']);rotation,_=np.linalg.qr(rng.normal(size=(3,3)))
    rotation[:,0]*=np.linalg.det(rotation)
    permutation=np.r_[0,rng.permutation(np.arange(1,80))]
    translated=x+np.array([11.,-7.,3.],np.float32);translated-=translated[:,:1]
    wrapped=np.mod(x+np.array([31.,3.,60.]),64.)
    wrapped-=wrapped[:,:1];wrapped-=64.*np.round(wrapped/64.)
    comparisons={'repeat':x,'rotation':(x@rotation).astype(np.float32),
                 'permutation':x[:,permutation],'translation_recentered':translated,
                 'periodic_image_recentered':wrapped.astype(np.float32)}
    results={}
    for name,value in comparisons.items():
        zz=encode(model,value,chunk,device);difference=zz-z
        close=np.allclose(z,zz,atol=config['parity_atol'],rtol=config['parity_rtol'])
        results[name]=dict(max_abs=float(abs(difference).max()),rms=float(np.sqrt(np.mean(difference**2))),passed=bool(close))
        if not close:raise ValueError(f'Native geometric consistency failed: {name}, {results[name]}')
    from src.research.geoframe_evolution.metrics import perturbation
    from src.research.robust_onset.metrics import perturb_patch
    noise={}
    for fraction in config['noise_rms_fractions']:
        random=np.random.default_rng(config['seed'])
        changed=[perturb_patch(p,fraction,random) for p in x]
        zz=encode(model,np.stack([p for p,_ in changed]),chunk,device)
        noise[str(fraction)]=dict(perturbation(z,zz),reference='Same 128 clean environments; independent-pair RMS',
            noise_rms_fraction=fraction,neighbor_candidates='Fixed 80 candidates; graph edges and radius mask rebuilt')
    results['noise']=noise
    return results


def static_run(config,spec,model,cache,output,device):
    reference=Path(config['reference']);manifest=json.loads((reference/'manifest.json').read_text())
    original=json.loads((Path(config['native_inputs'])/'manifest.json').read_text())
    result={};chunk=config['batch_size'];saved_frames={}
    for record in manifest['frames']:
        index=record['frame_index'];key=f'frame_{index:02d}_{record["material"]}_encoder'
        path=reference/f'frame-{index:02d}.npz'
        if sha(path)!=manifest['files'][path.name]:raise ValueError(f'Changed static reference: {path}')
        with np.load(path) as loaded:a=dict(loaded)
        with np.load(Path(config['native_inputs'])/f'frame-{index:02d}.npz') as loaded:native=loaded['nearest80']
        # Rebuilt labels must use exactly the previously analyzed atom neighborhoods.
        coordinate_check=static_coordinate_agreement(a['clouds']*record['radius_A'],native)
        old=original['frames'][index]
        if old['input_sha256']!=record['input_sha256'] or old['class_counts']!=record['class_counts']:
            raise ValueError(f'Rebuilt static population/reference changed at frame {index}')
        scale=config['normalization']['reference_scale_A']/config['normalization']['scales_A'][record['material']]
        positions=native*scale
        feature=cache/f'frame-{index:02d}.npy'
        if feature.exists():z=np.load(feature)
        else:z=encode(model,positions,chunk,device);np.save(feature,z)
        with threadpool_limits(limits=1):
            metrics,clusters,clusterer=frame_metrics(z,a,record,return_clusterer=True)
            metrics['supplement']=liquid_metrics(z,a,record)
            metrics['coordinate_agreement']=coordinate_check
        result[key]=metrics
        saved_frames[index]=(z,clusters,record,clusterer)
        np.save(output/f'clusters-{index:02d}.npy',clusters)
        np.save(output/f'cluster-centers-{index:02d}.npy',clusterer.cluster_centers_)
        write_json(output/'static-metrics.json',result)
        if index==0:
            write_json(output/'geometry-checks.json',geometric_checks(model,positions,config,device))
        print(json.dumps(dict(stage='static',model=spec['name'],frame=index)),flush=True)
    from .report import static_plots
    static_plots(config,spec,model,saved_frames,cache,device,encode)
    return result


def physical_cache(config,corpus,domain,device):
    dest=Path(config['output'])/'technical/physical';dest.mkdir(exist_ok=True)
    path=dest/f'{domain}.npy'
    if path.exists():
        receipt=json.loads((dest/f'{domain}.json').read_text())
        if receipt['sha256']!=sha(path):raise ValueError('Changed physical target cache')
        return np.load(path)
    result=[];chunk=config['batch_size']
    for start in range(0,len(corpus.pop['event']),chunk):
        ids=np.arange(start,min(start+chunk,len(corpus.pop['event'])))
        x=torch.as_tensor(positions_for(corpus,domain,ids),device=device)
        result.append(physical_targets(x).cpu().numpy())
    values=np.concatenate(result);np.save(path,values)
    write_json(dest/f'{domain}.json',dict(sha256=sha(path),population_sha256=sha(Path(config['population_cache'])/'population.npz')))
    return values


@torch.no_grad()
def temporal_run(config,spec,model,cache,output,corpus,z,physical,device):
    if spec['domain']=='cold':
        result=dict(available=False,reason='No dense relaxed trajectory; observed inputs are not substituted')
        write_json(output/'temporal-metrics.json',result);return result
    dense=Path(config['dense_observed']);manifest=json.loads((dense/'manifest.json').read_text())
    if manifest['release_identity']!=config['fixed_dataset']['identity']:raise ValueError('Dense release changed')
    if manifest['cadence_ps']!=.75 or manifest['domain']!='observed':raise ValueError('Dense observation cadence/domain differs')
    np.testing.assert_array_equal(sorted(r['source'] for r in manifest['sources']),np.unique(corpus.pop['source'][corpus.split['test']]))
    rng=np.random.default_rng(config['seed']);values=[];targets=[];sources=[];pairs=[];records=[];cursor=0
    for record in manifest['sources']:
        folder=dense/str(record['source']);positions=np.load(folder/'positions.npy',mmap_mode='r')
        with np.load(folder/'observations.npz') as a:observations=dict(a)
        frame_ids=np.sort(rng.choice(positions.shape[0]-1,config['temporal_frames_per_source'],replace=False))
        rows=np.stack([frame_ids*64+i for i in range(64)],axis=1).ravel()
        following=rows+64;indices=np.r_[rows,following]
        x=np.array(positions.reshape(-1,80,3)[indices]);n=len(rows)
        zz=encode(model,x,config['batch_size'],device)
        yy=[]
        for first in range(0,len(x),config['batch_size']):
            yy.append(physical_targets(torch.as_tensor(x[first:first+config['batch_size']],device=device)).cpu().numpy())
        values.append(zz);targets.append(np.concatenate(yy));sources.append(observations['source'][indices])
        pairs.append(np.c_[np.arange(cursor,cursor+n),np.arange(cursor+n,cursor+2*n)])
        records.append(dict(source=record['source'],frames=frame_ids.tolist(),atoms=64,observation_sha256=sha(folder/'observations.npz')))
        cursor+=2*n
    fit=corpus.split['train'];w=source_weights(corpus.pop['source'][fit])
    trace=float(np.sum(w@(z[fit]-w@z[fit])**2))
    mean=w@physical[fit];scale=np.sqrt(w@(physical[fit]-mean)**2).clip(1e-5)
    zz=np.concatenate(values);yy=np.concatenate(targets);ss=np.concatenate(sources);pp=np.concatenate(pairs)
    result=movement_relation(zz,yy,ss,pp,scale,trace)
    result.update(available=True,reference_trace=trace,observations=records,selection='Outcome-independent frame draw; all 64 fixed centers; test sources only')
    np.savez(cache/'temporal-sample.npz',z=zz,physical=yy,source=ss,pairs=pp)
    write_json(output/'temporal-metrics.json',result)
    return result


def publish_prediction(study,corpus,name,kind,risks,metadata):
    dest=study.technical/'evaluation'/name/kind;dest.mkdir(parents=True,exist_ok=True)
    result,calibrated=score(corpus,risks)
    np.savez(dest/'predictions.npz',sample_id=corpus.pop['sample_id'],source=corpus.pop['source'],
        event=corpus.pop['event'],role=corpus.pop['role'],risks=risks,calibrated=calibrated)
    result.update(metadata,identity=study.identity,name=name,kind=kind)
    write_json(dest/'metrics.json',result)
    return result,calibrated


def predictive_run(config,identity,spec,model,cache,output,device,corpus):
    domain=spec['domain'];ids=np.arange(len(corpus.pop['event']));chunk=config['batch_size']
    if spec['kind']=='supervised':
        arm='O-NLL' if domain=='hot' else 'R-NLL'
        folder=Path(spec['origin'])/'technical/evaluation'/arm/'best'
        metadata=json.loads((folder/'metrics.json').read_text())
        if metadata['checkpoint_sha256']!=spec['checkpoint_sha256']:raise ValueError('Existing feature checkpoint mismatch')
        with np.load(folder/'predictions.npz') as p:
            for key in ('sample_id','source','role','event'):
                np.testing.assert_array_equal(p[key],corpus.pop[key])
        z=np.load(folder/'features.npy')
        replay_ids=np.r_[corpus.split['train'][:64],corpus.split['test'][:64]]
        replay=encode(model,positions_for(corpus,domain,replay_ids),chunk,device)
        np.testing.assert_allclose(replay,z[replay_ids],atol=config['parity_atol'],rtol=config['parity_rtol'])
        write_json(output/'feature-reuse.json',dict(path=str(folder/'features.npy'),sha256=sha(folder/'features.npy'),
            checkpoint_sha256=spec['checkpoint_sha256'],replay_rows=replay_ids.tolist(),max_abs=float(abs(replay-z[replay_ids]).max())))
        full=Path(spec['origin'])/'technical/full-evaluation/metrics.json'
        write_json(output/'original-evaluation.json',dict(source=str(full),sha256=sha(full),metrics=json.loads(full.read_text())))
    else:
        feature=cache/'population.npy'
        if feature.exists():z=np.load(feature)
        else:
            values=[]
            for start in range(0,len(ids),chunk):
                values.append(encode(model,positions_for(corpus,domain,ids[start:start+chunk]),chunk,device))
            z=np.concatenate(values);np.save(feature,z)
    if z.shape!=(len(ids),128):raise ValueError('Wrong exported population or dimension')
    physical=physical_cache(config,corpus,domain,device)
    temporal=temporal_run(config,spec,model,cache,output,corpus,z,physical,device)
    input_record=dict(encoder=spec,encoder_inputs=['current centered geometry','constant atom channel','center indicator'],
        spatial=dict(candidate_atoms=80,radius_A=8.,edge_cutoff_A=5.,blocks=2,halo=False),
        relaxation='full current-cell quench' if domain=='cold' else None,history_frames=1,motion=False,
        external_inputs=[],fixed_dataset=config['fixed_dataset'],
        training_only_context={'pretraining':('none' if spec['name'].startswith('scratch') else
            'paired current observed and full-cell-relaxed structural views; fixed material normalization'),
            'teacher':('fixed initial random MACE projection reservoir during Epi pretraining'
                       if spec['name'].startswith('epi_variance') else None),
            'supervised_teacher':None,'current_encoder_training':False},
        predictors={'z':'128-dimensional frozen export','physical':'32 same-domain geometric descriptors',
                    'joint':'128-dimensional frozen export + 32 same-domain geometric descriptors'},
        target='Original-MD sustained local onset; predominantly existing-crystal arrival',
        encoder_training_unchanged=True,probe_selection='selection-source hazard NLL')
    if 'training_context' in spec:input_record['training_only_context']=spec['training_context']
    study=probe_study(config,identity,spec['name'],input_record)
    fit=corpus.split['train'];w=source_weights(corpus.pop['source'][fit])
    prior=w@np.stack([corpus.pop['event'][fit]<=i for i in range(5)],axis=1)
    record,calibrated=publish_prediction(study,corpus,'constant','empirical',np.tile(prior,(len(ids),1)),
        dict(input='none',fit='source-weighted fitting event frequencies; categorical maximum likelihood',external_inputs=[]))
    result={'constant':record};predictions={'constant':(np.tile(prior,(len(ids),1)),calibrated)}
    # The shared descriptor control is fitted once per input domain with the same recipe.
    controls=probe_study(config,identity,f'physical-{domain}',dict(domain=domain,inputs='32 current-geometry descriptors',external_inputs=[]))
    for name,x,owner in [('physical',physical,controls),('z',z,study),('joint',np.c_[z,physical],study)]:
        for kind in ('linear','mlp'):
            risks=readout(owner,corpus,x,name,kind,device)
            record,calibrated=publish_prediction(owner,corpus,name,kind,risks,
                dict(input=name,domain=domain,selected_by='selection-source hazard NLL',external_inputs=[]))
            result[f'{name}-{kind}']=record;predictions[f'{name}-{kind}']=(risks,calibrated)
            print(json.dumps(dict(stage='readout',model=spec['name'],input=name,kind=kind)),flush=True)
    fit=corpus.split['train'];w=source_weights(corpus.pop['source'][fit])
    mean=w@physical[fit];scale=np.sqrt(w@(physical[fit]-mean)**2).clip(1e-5)
    for name,x in [('z',z),('physical',(physical-mean)/scale)]:
        candidates=[]
        for k in config['neighbors']:
            risk=neighbor_forecast(corpus,x,k,config['neighbor_prior_strength'],device)
            selection=corpus.split['selection'];prob=np.diff(np.c_[np.zeros(len(risk)),risk,np.ones(len(risk))],axis=1)
            nll=float(source_weights(corpus.pop['source'][selection])@-np.log(prob[selection,corpus.pop['event'][selection]].clip(1e-12)))
            candidates.append((nll,k,risk))
        nll,k,risk=min(candidates,key=lambda item:item[0])
        record,calibrated=publish_prediction(study,corpus,name,'neighbors',risk,
            dict(neighbors=k,selection_event_nll=nll,selection_candidates=[dict(k=b,nll=a) for a,b,_ in candidates],
                 fitting_predictions='unused fitting prior',distance='raw embedding' if name=='z' else 'train-standardized descriptors'))
        result[f'{name}-neighbors']=record;predictions[f'{name}-neighbors']=(risk,calibrated)
    comparisons={}
    for kind in ('linear','mlp','neighbors'):
        for first,second in ([('z','physical')] if kind=='neighbors' else [('z','physical'),('joint','z'),('joint','physical')]):
            key=f'{first}-minus-{second}-{kind}'
            comparisons[key]={scale:paired_scores(corpus,predictions[f'{first}-{kind}'][j],predictions[f'{second}-{kind}'][j],config['bootstrap'],config['seed'])
                for j,scale in enumerate(('raw','calibrated'))}
    write_json(output/'predictive-metrics.json',dict(readouts=result,paired=comparisons))
    return dict(readouts=result,paired=comparisons,temporal=temporal)


def run(config,identity,spec,device):
    output=Path(config['output'])/'technical/evaluations'/spec['name'];output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():
        record=json.loads((output/'complete.json').read_text())
        if record['identity']!=identity:raise ValueError('Completed evaluation identity changed')
        if record['metrics_sha256']!=sha(output/'metrics.json'):raise ValueError('Completed evaluation metrics changed')
        return
    start=time.monotonic();write_json(output/'state.json',dict(state='running',started_at=time.time()))
    model,metadata=load_encoder(spec,device)
    c=corpus(config);example=positions_for(c,spec['domain'],c.split['train'][:config['batch_size']])
    encode(model,example,config['batch_size'],device,compile_first=True)
    key=digest(dict(identity=identity,checkpoint=spec['checkpoint_sha256']))
    managed=RetainedCache(config['feature_cache'],6)
    with managed.lease(key,deadline=time.time()+8*3600,metadata=dict(artifact='encoder-quality',model=spec['name'],checkpoint_sha256=spec['checkpoint_sha256'])) as cache:
        write_json(output/'feature-cache.json',dict(key=key,path=str(cache)))
        static=static_run(config,spec,model,cache,output,device)
        predictive=predictive_run(config,identity,spec,model,cache,output,device,c)
    result=dict(model=spec,metadata=metadata,static=static,predictive=predictive,
        limitations={'nucleus_birth':'Separate versioned outcome population not yet released; no precursor-prediction claim',
                     'static_domain':'Relaxed snapshots; hot-trained encoders are evaluated in transfer',
                     'replication':'One trained seed per latest recipe; bootstrap conditions on fitted encoders'})
    write_json(output/'metrics.json',result)
    write_metric_table(result,Path(config['output'])/'models'/spec['name'],family='encoder_quality')
    write_json(output/'complete.json',dict(identity=identity,state='complete',metrics_sha256=sha(output/'metrics.json'),seconds=time.monotonic()-start))
    write_json(output/'state.json',dict(state='complete',completed_at=time.time()))


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--config',required=True);parser.add_argument('--name',required=True)
    parser.add_argument('--check-only',action='store_true');a=parser.parse_args()
    config=load(a.config);identity=None if a.check_only else bind(config);spec=next(m for m in config['models'] if m['name']==a.name)
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    try:
        if a.check_only:
            model,meta=load_encoder(spec,'cuda');c=corpus(config)
            x=positions_for(c,spec['domain'],c.split['train'][:config['batch_size']])
            encode(model,x,config['batch_size'],'cuda',compile_first=True)
            # Real input checks are local diagnostics; no W&B run is created.
            checks=geometric_checks(model,x,config,'cuda')
            if spec['kind']=='supervised':
                arm='O-NLL' if spec['domain']=='hot' else 'R-NLL'
                features=np.load(Path(spec['origin'])/'technical/evaluation'/arm/'best/features.npy',mmap_mode='r')
                ids=np.r_[c.split['train'][:64],c.split['test'][:64]]
                replay=encode(model,positions_for(c,spec['domain'],ids),config['batch_size'],'cuda')
                np.testing.assert_allclose(replay,features[ids],atol=config['parity_atol'],rtol=config['parity_rtol'])
                checks['saved_feature_replay']=dict(rows=len(ids),max_abs=float(abs(replay-features[ids]).max()),passed=True)
            write_json(Path(config['output'])/'technical'/f'checks-{spec["name"]}.json',dict(metadata=meta,checks=checks,wandb_runs=0,
                run_source_sha256=sha(Path(__file__)),config_sha256=sha(a.config),checkpoint_sha256=spec['checkpoint_sha256']))
        else:run(config,identity,spec,'cuda')
    except BaseException as exc:
        dest=Path(config['output'])/'technical/evaluations'/spec['name'];dest.mkdir(parents=True,exist_ok=True)
        write_json(dest/('check-failed.json' if a.check_only else 'failed.json'),dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


if __name__=='__main__':main()
