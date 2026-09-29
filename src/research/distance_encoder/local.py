"""Local-only distance, transfer-readout and representation-quality evaluation."""
import argparse
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch

from src.data.fixed_cohort.protocol import sha,digest,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import write_metric_table
from src.research.encoder_quality.common import corpus as load_corpus
from src.research.encoder_quality.run import encode,positions_for,publish_prediction
from src.research.equivariant_context.cache import encoder_cache
from src.research.local_predictability.metrics import source_weights
from src.research.spatial_approach.train import spatial_labels
from src.research.spatial_distance import train as distance_readout
from src.research.supervised_onset.evaluate import readout
from src.research.supervised_onset.tracking import update_training_summary
from src.research.trajectory_stability.spectrum import spectrum
from src.research.robust_onset.metrics import perturb_patch
from .model import DistanceEncoder
from .context import joint_evaluation


def compact_spectrum(z,source):
    return {k:v for k,v in spectrum(z,source_weights(source)).items() if k!='eigenvalues'}


def quality(encoder,z,original,labels,c):
    fit=original.split['train'];test=original.split['test'];source=original.pop['source']
    reference=spectrum(z[fit],source_weights(source[fit]))['total_energy']
    result={'all_fixed_rows':compact_spectrum(z,source),'test':compact_spectrum(z[test],source[test])}
    clear=test[~labels['ptm_local'][test]]
    result['test_local_ptm_clear']=compact_spectrum(z[clear],source[clear])
    rng=np.random.default_rng(c['seed'])
    ids=np.concatenate([rng.choice(test[source[test]==sid],4,replace=False) for sid in np.unique(source[test])])
    x=positions_for(original,'hot',ids);clean=z[ids];noise={}
    for fraction in (.001,.005,.01,.03):
        changed=[];relative=[];absolute=[]
        random=np.random.default_rng(c['seed'])
        for p in x:
            valid=p[np.linalg.norm(p,axis=1)<8.]
            perturbed,record=perturb_patch(valid,fraction,random)
            padded=np.full((80,3),100.,np.float32);padded[:len(perturbed)]=perturbed
            changed.append(padded);relative.append(record['input_relative_mse']);absolute.append(record['input_mse_A2'])
        zz=encode(encoder,np.stack(changed),256,'cuda')
        movement=np.linalg.norm(zz-clean,axis=1)/np.sqrt(2*reference)
        noise[str(fraction)]=dict(rows=len(ids),input_rms_fraction_of_d12=float(np.sqrt(np.mean(relative))),
            input_rms_A=float(np.sqrt(np.mean(absolute))),embedding_rms=float(np.sqrt(np.mean(movement**2))),
            embedding_p95=float(np.quantile(movement,.95)))
    result['noise']=noise
    dense=resolve_path(c['local_evaluation']['dense_observed']);manifest=json.loads((dense/'manifest.json').read_text())
    if manifest['release_identity']!=c['fixed_dataset']['identity'] or manifest['cadence_ps']!=.75 or manifest['domain']!='observed':
        raise ValueError('Wrong dense observed release or cadence')
    np.testing.assert_array_equal(sorted(r['source'] for r in manifest['sources']),np.unique(source[test]))
    delta=[];sources=[];observations=[];frames=[]
    for record in manifest['sources']:
        folder=dense/str(record['source']);positions=np.load(folder/'positions.npy',mmap_mode='r')
        chosen=np.sort(rng.choice(positions.shape[0]-1,8,replace=False))
        rows=(chosen[:,None]*64+np.arange(64)).ravel();indices=np.r_[rows,rows+64]
        zz=encode(encoder,np.array(positions.reshape(-1,80,3)[indices]),256,'cuda');n=len(rows)
        delta.append(zz[n:]-zz[:n]);sources.append(np.full(n,record['source']));observations.append(zz)
        frames.append(dict(source=record['source'],frame_indices=chosen.tolist(),observations_sha256=sha(folder/'observations.npz')))
    delta=np.concatenate(delta);sources=np.concatenate(sources);w=source_weights(sources)
    movement=spectrum(delta/.75,w,centered=False)
    result['temporal_0_75ps']=dict(pairs=len(delta),rms_jump=float(np.sqrt(w@np.square(delta).sum(1)/(2*reference))),
        movement={k:v for k,v in movement.items() if k!='eigenvalues'},
        state=compact_spectrum(np.concatenate(observations),np.concatenate([np.r_[s,s] for s in np.split(sources,len(frames))])),
        frames=frames,reference_trace=reference)
    return result


def export_other(encoder,s,cache,records,kind):
    output=[]
    for i,record in enumerate(records):
        origin=(s.geometry if kind=='uniform' else s.parent/'technical/sources')/str(record['source'])/record['file']
        key=record['frame'] if kind=='uniform' else record['path_id'];dest=cache/f'{kind}-{record["source"]}-{key}.npz'
        receipt=dest.with_suffix('.json')
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['identity']!=s.identity or old['sha256']!=sha(dest):raise ValueError('Changed local feature cache')
        else:
            if sha(origin)!=record['sha256']:raise ValueError(f'Changed geometry {origin}')
            with np.load(origin) as a:
                # Explicitly take only focal patch zero; never encode its neighbors.
                inverse=a['inverse'][:,0];positions=np.array(a['positions'][inverse],dtype=np.float32)
                z=encode(encoder,positions,256,'cuda')[:,None,:]
                values={k:a[k] for k in ('actual','atom','distance','visible_local','visible_context','ptm_local','ptm_context')}
                if kind=='scan':values['travel_A']=a['travel_A']
                np.savez(dest,z=z,**values)
            write_json(receipt,dict(identity=s.identity,sha256=sha(dest),geometry_sha256=record['sha256']))
        output.append(dict(record,features=str(dest)))
        if i%100==0:print(json.dumps(dict(stage='local-'+kind,completed=i+1,total=len(records))),flush=True)
    return output


def run(config_path,checkpoint,output,name,provisional=False):
    c=json.loads(Path(config_path).read_text());root=result_folders(resolve_path(output));technical=root/'technical'
    checkpoint=resolve_path(checkpoint);saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
    if not provisional and saved['epoch']<c['training']['epochs']:raise ValueError('Final evaluation requires the full declared training budget')
    torch.set_num_threads(1);torch.set_float32_matmul_precision('highest')
    model=DistanceEncoder(saved['encoder_config']).cuda();model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
    previous=resolve_path(c['context_followup']['distance_run']);old=json.loads((previous/'technical/identity.json').read_text())
    settings=dict(old['config'],output=str(root),checkpoint=str(checkpoint),encoder_name=name,batch_size=256,microbatch=256,
        wandb=dict(c['wandb'],group='cd-mace128-local-20260926'))
    binding=dict(config=c,checkpoint_training_config=saved['config'],checkpoint_sha256=sha(checkpoint),name=name,epoch=saved['epoch'],provisional=provisional,
        old_distance_identity=digest(old),producer_sha256=sha(__file__))
    identity=digest(binding);target=technical/'identity.json'
    if target.exists() and json.loads(target.read_text())!=binding:raise ValueError('Local evaluation identity changed')
    write_json(target,binding)
    original=load_corpus(dict(population_cache=str(resolve_path(c['local_evaluation']['population_cache'])),fixed_dataset=c['fixed_dataset']))
    population_root=resolve_path(c['local_evaluation']['population_cache'])
    for field in ('population.npz','hot-positions.npy','hot-offsets.npy'):
        if sha(population_root/field)!=original.manifest['files'][field]:raise ValueError(f'Changed local observation {field}')
    fixed=resolve_path(c['fixed_dataset']['root']);plan=json.loads((fixed/'plan.json').read_text())
    s=SimpleNamespace(root=root,technical=technical,identity=identity,config=settings,plan=plan,pop=original.pop,
        pointer={'encoder_sha256':sha(checkpoint)},parent=resolve_path(settings['parent_run']),
        geometry=resolve_path(settings['geometry_cache'])/digest(old))
    parent=SimpleNamespace(pop=s.pop,plan=plan,technical=s.parent/'technical',
        identity=digest(json.loads((s.parent/'technical/identity.json').read_text())),config=settings)
    labels=spatial_labels(parent);n=len(s.pop['source']);fit=original.split['train']
    inputs=dict(encoder=dict(name=name,trainable=False,checkpoint_sha256=sha(checkpoint),geometry='nearest80 within8A, cutoff5A, 2blocks',
        exported_dimensions=128,constant_atom_channel=True,history=False,velocities=False,conditions=[],relaxation=False),
        predictor=dict(input='one local 128-vector only',surrounding_embeddings=False,conditions=[]),
        fixed_dataset=c['fixed_dataset'],provisional=provisional,checkpoint_epoch=saved['epoch'])
    write_json(technical/'prediction-context.json',inputs)
    with encoder_cache(s,'local-only',checkpoint,time.time()+6*3600) as cache:
        feature=cache/'fixed.npy'
        if feature.exists():
            if sha(feature)!=json.loads(feature.with_suffix('.json').read_text())['sha256']:raise ValueError('Changed local fixed features')
            z=np.load(feature)
        else:
            parts=[]
            for first in range(0,n,1024):
                ids=np.arange(first,min(first+1024,n));parts.append(encode(model.encoder,positions_for(original,'hot',ids),256,'cuda',compile_first=first==0))
                if first%16384==0:print(json.dumps(dict(stage='local-fixed',rows=first,total=n)),flush=True)
            z=np.concatenate(parts);np.save(feature,z);write_json(feature.with_suffix('.json'),dict(sha256=sha(feature)))
        prepared=json.loads((previous/'technical/prepared.json').read_text())
        if prepared['identity']!=digest(old):raise ValueError('Changed uniform population')
        augmentation=export_other(model.encoder,s,cache,prepared['records'],'uniform')
        paths=json.loads((s.parent/'technical/scan-features.json').read_text())['records']
        records=export_other(model.encoder,s,cache,paths,'scan')
        extra=[];zs=[z[:,None,:]]
        for record in augmentation:
            with np.load(record['features']) as a:
                zs.append(a['z']);extra.append({k:a[k] for k in ('distance','visible_local','visible_context','ptm_local','ptm_context')}|
                    dict(source=np.full(record['rows'],record['source']),role=np.full(record['rows'],record['role'])))
        targets={k:np.concatenate([labels[k],*[a[k] for a in extra]]) for k in ('distance','visible_local','visible_context','ptm_local','ptm_context')}
        pop={k:np.concatenate([s.pop[k],*[a[k] for a in extra]]) for k in ('source','role')}
        w=source_weights(s.pop['source'][fit]);mean=w@z[fit].astype(float);scale=np.sqrt(w@(z[fit]-mean)**2).clip(1e-5)
        corpus=SimpleNamespace(original_n=n,scalers={'z':dict(mean=mean.tolist(),scale=scale.tolist())},
            features={'z':torch.tensor(((np.concatenate(zs)-mean)/scale).astype(np.float32),device='cuda')},
            required_fields=('z',),nominal=torch.zeros((1,3),device='cuda'),distance=torch.tensor(targets['distance'],device='cuda'),pop=pop,
            split={r:np.flatnonzero(pop['role']==r) for r in ('train','selection','calibration','test')},weights={})
        for role in ('train','selection'):
            ids=corpus.split[role];weight=np.zeros(len(ids))
            for mask in (ids<n,ids>=n):weight[mask]=.5*source_weights(pop['source'][ids[mask]])
            corpus.weights[role]=torch.tensor(weight,dtype=torch.float32,device='cuda')
        rows=joint_evaluation(s,corpus,targets,records,saved)
        distance_readout.fit(s,'mace_local',corpus,targets,records)
        metrics=quality(model.encoder,z,original,labels,c)
        write_metric_table(metrics,root/'analyses/representation-v1',family='distance_encoder_local',name='quality')
        write_json(technical/'quality.json',metrics)
        probe_settings=dict(seed=c['seed'],branch='crystallization_supervised',arms=[],tracking_scope='diagnostic',
            baselines=dict(epochs=16,minimum_selection_epoch=12,batch_size=256,learning_rate=.001,
                updates=16*math.ceil(len(fit)/256)),wandb=dict(settings['wandb'],display_name=name+' | local onset probes'))
        study=SimpleNamespace(config=probe_settings,root=root,technical=technical,identity=identity)
        for kind in ('linear','mlp'):
            probe_name=name+'-onset-'+kind
            risks=readout(study,original,z,probe_name,kind,'cuda')
            scores,_=publish_prediction(study,original,probe_name,kind,risks,{'encoder_frozen':True,'external_inputs':[]})
            write_metric_table(scores,root/'analyses'/('onset-'+kind+'-v1'),family='distance_encoder_local',name='scores')
        tracking=SimpleNamespace(config=dict(settings,wandb=dict(settings['wandb'],display_name=name+' | local distance readout')),
            root=root/'mace_local',technical=root/'mace_local/technical',identity=identity)
        update_training_summary(tracking,'mace_local',
            {'encoder/name':name,'encoder/epoch':saved['epoch'],'evaluation/provisional':provisional,
             'evaluation/joint_distance':rows,'evaluation/representation':metrics},evaluation='local')
    write_json(technical/'complete.json',dict(identity=identity,checkpoint_sha256=sha(checkpoint),name=name,provisional=provisional,epoch=saved['epoch']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--name',required=True);p.add_argument('--provisional',action='store_true')
    a=p.parse_args();run(a.config,a.checkpoint,a.output,a.name,a.provisional)
