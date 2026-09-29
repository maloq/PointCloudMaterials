"""Paired distance evaluations for material adaptation; no diagnostic online runs."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.data.fixed_cohort.protocol import digest,sha,write_json
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.project_runtime.paths import resolve_path
from src.research.encoder_context.geometry import graph
from src.research.local_predictability.metrics import source_weights
from src.research.spatial_approach.evaluate import csv_rows
from src.research.spatial_distance.confidence import RADII
from src.research.spatial_distance.train import distance_metrics
from src.research.supervised_onset.tracking import update_training_summary
from .history import HistoryDistanceEncoder
from .history_evaluate import predict
from .material_data import ShootingHistoryDataset


def evaluate_ta(c,interim_checkpoint=None):
    interim=interim_checkpoint is not None
    root=result_folders(resolve_path(c['output'])/('distance-interim-stop-20260927' if interim else 'distance-holdout'));tech=root/'technical'
    paths=dict(parent=resolve_path(c['initial_encoder']),adapted=resolve_path(interim_checkpoint) if interim else resolve_path(c['output'])/'technical/best.pt')
    binding=dict(config=c,checkpoints={k:sha(v) for k,v in paths.items()},
        data_manifest_sha256=sha(resolve_path(c['ta_evaluation']['root'])/'manifest.json'),producer_sha256=sha(__file__),interim_user_stop=interim)
    identity=digest(binding);receipt=tech/'complete.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        if old['identity']!=identity or any(sha(tech/n)!=h for n,h in old['files'].items()):raise ValueError('Changed Ta material evaluation')
        return
    write_json(tech/'identity.json',binding)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    metrics=[];reliability=[];files={};contexts={};checkpoint_epochs={}
    for label,path in paths.items():
        saved=torch.load(path,map_location='cuda',weights_only=False)
        if saved['epoch']<12 and not (interim and label=='adapted'):
            raise ValueError('Final material evaluation requires twelve epochs; explicit interim evaluation is separate')
        checkpoint_epochs[label]=saved['epoch']
        if label=='adapted' and saved['identity']!=digest(json.loads((resolve_path(c['output'])/'technical/identity.json').read_text())):
            raise ValueError('Adapted checkpoint differs from the recorded training identity')
        model=HistoryDistanceEncoder(saved['encoder_config'],c['history']).cuda()
        model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
        for role in ('selection','test'):
            data=ShootingHistoryDataset(c,role,torch.device('cuda'));contexts[role]=data.history_metadata
            size=c['history_evaluation']['batch_size']
            if c['runtime']['compile'] and role=='selection':
                compile_spatial_encoder(model.encoder,graph(data.batch(slice(0,size))[0].reshape(-1,80,3),model.encoder))
            chunks=[]
            for start in range(0,data.n,size):
                x,target,_=data.batch(slice(start,min(start+size,data.n)))
                chunks.append(predict(model,x,target,c))
            values={k:np.concatenate([r[k] for r in chunks]) for k in chunks[0]}
            if any(not np.isfinite(v).all() for v in values.values()):raise FloatingPointError(f'Nonfinite Ta {label}/{role}')
            distance=data.distance.cpu().numpy();sources=data.source_ids;weights=source_weights(sources)
            filename=f'{label}-{role}-predictions.npz'
            np.savez(tech/filename,**values,distance=distance,source=sources,atom=data.atoms,raw_frame=data.raw_frames)
            files[filename]=sha(tech/filename)
            lookup=f'{label}-{role}-sources.json';write_json(tech/lookup,data.source_names);files[lookup]=sha(tech/lookup)
            metrics.append(dict(model=label,role=role,**distance_metrics(values,distance,sources,64)))
            print(json.dumps(dict(stage='interim-evaluation' if interim else 'evaluation',checkpoint_epoch=saved['epoch'],**metrics[-1])),flush=True)
            for k,radius in enumerate(RADII):
                for threshold in (.5,.75,.95):
                    keep=values['cdf'][:,k]>threshold;mass=weights[keep].sum()
                    reliability.append(dict(model=label,role=role,radius_Al_equivalent_A=radius,threshold=threshold,
                        selected_rows=int(keep.sum()),coverage=float(mass),
                        observed_precision=float(np.sum(weights[keep]*(distance[keep]<=radius))/mass) if mass else None,
                        mean_probability=float(np.sum(weights[keep]*values['cdf'][keep,k])/mass) if mass else None))
            del data
        del model;torch.cuda.empty_cache()
    analysis=root/'analyses/material-v1';snapshot_metric_docs(analysis,'distance_encoder_material_interim' if interim else c['metric_family'])
    csv_rows(analysis/'tables/distance.csv',metrics);csv_rows(analysis/'tables/confidence-reliability.csv',reliability)
    write_json(tech/'metrics.json',metrics);files['metrics.json']=sha(tech/'metrics.json')
    write_json(tech/'prediction-context.json',dict(populations=contexts,config=c['material_finetune'],
        encoder='geometry-only shared MACE, constant atom channel; fixed material length normalization',
        predictor='six tracked-atom embeddings and increments; no time, temperature, species, velocities or surrounding patches'))
    tracking_root=resolve_path(c['output']);training_identity=digest(json.loads((tracking_root/'technical/identity.json').read_text()))
    tracking=SimpleNamespace(config=c,root=tracking_root,technical=tracking_root/'technical',identity=training_identity)
    update_training_summary(tracking,'joint-mace-distance',
        {f'{"interim_material_distance" if interim else "material_distance"}/{r["model"]}/{r["role"]}/{k}':v for r in metrics for k,v in r.items() if k not in ('model','role')},evaluation='interim-material-distance' if interim else 'material-distance')
    write_json(receipt,dict(state='complete',identity=identity,files=files,interim_user_stop=interim,checkpoint_epochs=checkpoint_epochs))


def compare_al(c):
    root=resolve_path(c['output']);parent=resolve_path(c['material_finetune']['parent_front'])
    parent_binding=json.loads((parent/'technical/identity.json').read_text())
    child_binding=json.loads((root/'front/technical/identity.json').read_text())
    if parent_binding['checkpoint_sha256']!=c['initial_encoder_sha256'] or parent_binding['geometry_manifest_sha256']!=child_binding['geometry_manifest_sha256']:
        raise ValueError('Al parent/child checkpoint or evaluation population mismatch')
    for filename,fields in [('predictions.npz',('source','atom','frame','sample_id','distance')),
                            ('path-predictions.npz',('atom','distance','offsets'))]:
        with np.load(parent/'technical'/filename) as a,np.load(root/'front/technical'/filename) as b:
            for field in fields:np.testing.assert_array_equal(a[field],b[field])
    analysis=root/'analyses/material-comparison-v1';snapshot_metric_docs(analysis,c['metric_family'])
    for filename in ('distance.csv','alarms.csv','confidence-reliability.csv'):
        rows=[]
        for model,path in [('parent',parent),('adapted',root/'front')]:
            with (path/'analyses/front-v1/tables'/filename).open() as f:
                rows.extend(dict(row,model=model) for row in csv.DictReader(f))
        csv_rows(analysis/'tables'/filename,rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--interim-checkpoint');a=p.parse_args()
    c=json.loads(Path(a.config).read_text())
    if a.interim_checkpoint and c['material_finetune']['material']!='Ta':p.error('Interim checkpoint is supported for Ta distance evaluation only')
    evaluate_ta(c,a.interim_checkpoint) if c['material_finetune']['material']=='Ta' else compare_al(c)
