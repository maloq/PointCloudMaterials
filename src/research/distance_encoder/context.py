"""Evaluate the joint head and refit the two selected context heads after training."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch

from src.data.fixed_cohort.dataset import read_release
from src.data.fixed_cohort.protocol import sha,digest,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.equivariant_context.features import FeatureExtractor,prepare_graphs
from src.research.equivariant_context.geometry_cache import GeometryCache
from src.research.equivariant_context.cache import encoder_cache,RetainedCache
from src.research.spatial_distance import train as readouts
from src.research.spatial_distance.model import LocalDistance
from src.research.spatial_distance.confidence import tables
from src.research.spatial_approach.evaluate import csv_rows
from src.research.supervised_onset.tracking import update_training_summary
from .model import DistanceEncoder


def fixed_features(s,encoder,cache,geometry_root,cutoff):
    extractor=FeatureExtractor(encoder,'cuda',512,compile=True)
    geometry=GeometryCache(geometry_root,cutoff);geometry.domain='hot';receipts={}
    for item in s.plan['sources']:
        sid=item['id'];dest=cache/f'{sid}.npz';receipt=dest.with_suffix('.json')
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if record['identity']!=s.identity or record['sha256']!=sha(dest):raise ValueError('Changed frozen-encoder feature shard')
        else:
            rows=np.flatnonzero(s.pop['source']==sid);parts=[]
            for frame in np.unique(s.pop['frame'][rows]):
                ids=rows[s.pop['frame'][rows]==frame]
                if not (geometry_root/f'{sid}-{frame}.npz').exists():raise FileNotFoundError(f'Missing retained fixed graph {sid}/{frame}')
                a=geometry.frame(item,int(frame),ids,s.pop['atom'][ids],None,pin_memory=True)
                values=extractor(a['arrays'])
                parts.append(dict(rows=ids,query_atom_ids=a['atoms'],actual=a['actual'],**{k:v[a['inverse']] for k,v in values.items()}))
            arrays={k:np.concatenate([p[k] for p in parts]) for k in parts[0]};order=np.argsort(arrays['rows']);arrays={k:v[order] for k,v in arrays.items()}
            np.testing.assert_array_equal(arrays['rows'],rows);np.savez(dest,**arrays)
            record=dict(identity=s.identity,encoder_sha256=s.pointer['encoder_sha256'],rows=len(rows),sha256=sha(dest));write_json(receipt,record)
        receipts[str(sid)]=record
        print(json.dumps(dict(stage='new-encoder-fixed-features',completed=len(receipts),total=len(s.plan['sources']))),flush=True)
    write_json(cache/'manifest.json',dict(identity=s.identity,encoder_sha256=s.pointer['encoder_sha256'],shards=receipts))


def other_features(s,encoder,cache,records,kind):
    extractor=FeatureExtractor(encoder,'cuda',512,compile=True);output=[]
    for record in records:
        source=(s.geometry if kind=='uniform' else s.parent/'technical/sources')/str(record['source'])/record['file']
        key=record['frame'] if kind=='uniform' else record['path_id']
        dest=cache/f'{record["source"]}-{key}.npz';receipt=dest.with_suffix('.json')
        if receipt.exists():
            old=json.loads(receipt.read_text())
            if old['identity']!=s.identity or old['sha256']!=sha(dest) or old['geometry_sha256']!=record['sha256']:raise ValueError('Changed context feature cache')
        else:
            if sha(source)!=record['sha256']:raise ValueError(f'Changed geometry: {source}')
            with np.load(source) as a:
                arrays=prepare_graphs([p[np.linalg.norm(p,axis=-1)<8.] for p in a['positions']],encoder.cutoff,pin_memory=True)
                values=extractor(arrays);values={k:v[a['inverse']] for k,v in values.items()}
                values.update({k:a[k] for k in ('actual','atom','distance','visible_local','visible_context','ptm_local','ptm_context')})
                if kind=='scan':values['travel_A']=a['travel_A']
            np.savez(dest,**values);write_json(receipt,dict(identity=s.identity,sha256=sha(dest),geometry_sha256=record['sha256']))
        output.append(dict(record,features=str(dest)))
    write_json(s.technical/f'{kind}-features.json',dict(identity=s.identity,records=output))
    return output


def joint_evaluation(s,corpus,labels,records,saved):
    root=result_folders(s.root/'joint-local');tech=root/'technical'
    model=LocalDistance('mace_local').cuda();model.layers.load_state_dict(saved['head'])
    stats=corpus.scalers['z'];n=corpus.original_n
    raw_z=corpus.features['z'][:n]*torch.tensor(stats['scale'],device='cuda')+torch.tensor(stats['mean'],device='cuda')
    data=SimpleNamespace(features={'z':raw_z},nominal=corpus.nominal,distance=corpus.distance[:n])
    fixed=readouts.predictions(model,data,np.arange(n),512,64)|s.pop|{k:v[:n] for k,v in labels.items()}
    np.savez_compressed(tech/'predictions.npz',**fixed);paths=[]
    for record in records:
        with np.load(record['features']) as a:
            local=SimpleNamespace(features={'z':torch.tensor(a['z'],device='cuda')},nominal=corpus.nominal,distance=torch.tensor(a['distance'],device='cuda'))
            paths.append(readouts.predictions(model,local,np.arange(len(a['distance'])),512,64)|
                {k:a[k] for k in ('distance','travel_A','visible_local','visible_context','ptm_local','ptm_context')})
    scans={k:np.concatenate([p[k] for p in paths]) for k in paths[0]};scans['offsets']=np.r_[0,np.cumsum([len(p['distance']) for p in paths])]
    np.savez_compressed(tech/'path-predictions.npz',**scans);write_json(tech/'paths.json',records)
    analysis=root/'analyses/distance-v1';snapshot_metric_docs(analysis,'distance_encoder');rows=[]
    for role in ('selection','calibration','test'):
        ids=np.flatnonzero(fixed['role']==role)
        rows.append(dict(population='fixed_at_risk',role=role,**readouts.distance_metrics({k:fixed[k][ids] for k in ('log_likelihood','cdf','mean_A','median_A')},fixed['distance'][ids],fixed['source'][ids],64)))
    ids=np.concatenate([np.arange(scans['offsets'][i],scans['offsets'][i+1]) for i,r in enumerate(records) if r['role']=='test'])
    sources=np.concatenate([np.full(r['rows'],r['source']) for r in records if r['role']=='test'])
    rows.append(dict(population='controlled_scan',role='test',**readouts.distance_metrics({k:scans[k][ids] for k in ('log_likelihood','cdf','mean_A','median_A')},scans['distance'][ids],sources,64)))
    csv_rows(analysis/'tables/distance.csv',rows);write_json(tech/'metrics.json',rows)
    confidence=root/'analyses/confidence-v1';snapshot_metric_docs(confidence,'spatial_confidence')
    alarms,per_path,reliability=tables('mace_local',fixed,scans,records,fixed['cdf'],scans['cdf'])
    for name,values in [('alarms',alarms),('paths',per_path),('confidence-reliability',reliability)]:csv_rows(confidence/'tables'/f'{name}.csv',values)
    write_json(tech/'complete.json',dict(identity=s.identity,encoder_checkpoint_sha256=s.pointer['encoder_sha256'],
        files={name:sha(tech/name) for name in ('predictions.npz','path-predictions.npz','paths.json','metrics.json')}))
    return rows


def run(config):
    c=json.loads(Path(config).read_text());root=result_folders(resolve_path(c['output']));checkpoint=root/'technical/best.pt'
    completed=json.loads((root/'technical/complete.json').read_text())
    if sha(checkpoint)!=completed['best_sha256']:raise ValueError('Encoder completion/checkpoint mismatch')
    saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
    model=DistanceEncoder(saved['encoder_config']).cuda();model.load_state_dict(saved['model']);model.eval().requires_grad_(False)
    old=resolve_path(c['context_followup']['distance_run']);old_identity=json.loads((old/'technical/identity.json').read_text());base=old_identity['config']
    config=dict(base,output=str(root/'context'),checkpoint=str(checkpoint),variants=c['context_followup']['variants'],
        training=dict(base['training'],epochs=c['context_followup']['epochs']),
        wandb=dict(base['wandb'],group=c['wandb']['group']),batch_size=256,microbatch=256)
    binding=dict(config=config,encoder_sha256=sha(checkpoint),original_distance_identity=digest(old_identity),producer_sha256=sha(__file__))
    identity=digest(binding);ctx=result_folders(root/'context')
    dest=ctx/'technical/identity.json'
    if dest.exists() and json.loads(dest.read_text())!=binding:raise ValueError('Context protocol changed')
    write_json(dest,binding)
    fixed,plan=read_release(config['fixed_dataset']['root'])
    with np.load(fixed/'benchmark/population.npz') as a:pop={k:a[k] for k in ('source','role','frame','atom','sample_id','legacy_row')}
    s=SimpleNamespace(config=config,root=ctx,technical=ctx/'technical',identity=identity,checkpoint=checkpoint,
        pointer={'encoder_sha256':sha(checkpoint)},plan=plan,pop=pop,parent=resolve_path(config['parent_run']),
        geometry=resolve_path(config['geometry_cache'])/digest(old_identity))
    prepared=json.loads((old/'technical/prepared.json').read_text())
    if prepared['identity']!=digest(old_identity):raise ValueError('Changed reference augmentation')
    records=json.loads((s.parent/'technical/scan-features.json').read_text())['records'];deadline=time.time()+3*3600
    torch.set_num_threads(1)
    with ExitStack() as leases:
        s.features=leases.enter_context(encoder_cache(s,'fixed',checkpoint,deadline))
        pointer=json.loads(resolve_path(config['fixed_geometry_pointer']).read_text());geometry_root=Path(pointer['path'])
        metadata=json.loads((geometry_root/'entry.json').read_text())['metadata']
        with RetainedCache(geometry_root.parent.parent,1).lease(pointer['key'],deadline=deadline,metadata=metadata,shared=True):
            fixed_features(s,model.encoder,s.features,geometry_root,pointer['cutoff'])
        cache=leases.enter_context(encoder_cache(s,'uniform',checkpoint,deadline));augmented=other_features(s,model.encoder,cache,prepared['records'],'uniform')
        cache=leases.enter_context(encoder_cache(s,'scan',checkpoint,deadline));records=other_features(s,model.encoder,cache,records,'scan')
        corpus,labels=readouts.build_corpus(s,augmented)
        rows=joint_evaluation(s,corpus,labels,records,saved)
        tracking=SimpleNamespace(config=dict(c,wandb=dict(c['wandb'],display_name='Joint MACE | multimaterial distance | early 20–32Å')),
                                 root=root,technical=root/'technical',identity=completed['identity'])
        fields={f"evaluation/{row['population']}/{row['role']}/{key}":value
            for row in rows for key,value in row.items() if key not in ('population','role')}
        update_training_summary(tracking,'joint-mace-distance',fields,evaluation='context')
        for variant in config['variants']:readouts.fit(s,variant,corpus,labels,records)
    write_json(root/'technical/context-state.json',dict(state='complete',identity=identity,models=['joint-local',*config['variants']]))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    run(parser.parse_args().config)
