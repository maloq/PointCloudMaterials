"""Distance readouts on declared external ancestry holdouts at one MD cadence."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.artifacts import result_folders
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.project_runtime.paths import resolve_path
from src.models.encoders.spatial_mace import compile_spatial_encoder
from src.research.encoder_context.geometry import graph
from src.research.local_predictability.metrics import source_weights
from src.research.spatial_approach.evaluate import csv_rows
from src.research.spatial_distance.confidence import RADII
from src.research.spatial_distance.model import log_likelihood, cdf, capped_mean, capped_median
from src.research.spatial_distance.train import distance_metrics
from src.research.supervised_onset.tracking import update_training_summary
from .dense_history import DenseHistoryDataset
from .history import HistoryDistanceEncoder


@torch.no_grad()
def run(config_path):
    c=json.loads(Path(config_path).read_text());training=resolve_path(c['output'])
    root=result_folders(training/'distance-holdout');tech=root/'technical'
    checkpoint=training/'technical/best.pt';saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
    if saved['epoch']<12:raise ValueError('External evaluation requires a completed twelve-epoch fit')
    manifest=resolve_path(c['dense_history']['root'])/'manifest.json'
    binding=dict(config=c,checkpoint_sha256=sha(checkpoint),geometry_manifest_sha256=sha(manifest),producer_sha256=sha(__file__))
    identity=digest(binding);complete=tech/'complete.json'
    if complete.exists():
        old=json.loads(complete.read_text())
        if old['identity']!=identity or any(sha(tech/n)!=h for n,h in old['files'].items()):raise ValueError('Changed external evaluation')
        return
    write_json(tech/'identity.json',binding)
    torch.set_num_threads(1);torch.set_float32_matmul_precision('high')
    model=HistoryDistanceEncoder(saved['encoder_config'],c['history']).cuda()
    model.load_state_dict(saved['model'],strict=True);model.eval().requires_grad_(False)
    metrics=[];reliability=[];size=c['history_evaluation']['batch_size'];files={};contexts={}
    for role in ('selection','test'):
        data=DenseHistoryDataset(c,role,torch.device('cuda'));contexts[role]=data.history_metadata
        if c['runtime']['compile'] and role=='selection':
            compile_spatial_encoder(model.encoder,graph(data.batch(slice(0,size))[0].reshape(-1,80,3),model.encoder))
        chunks=[]
        for start in range(0,data.n,size):
            x,target,_=data.batch(slice(start,min(start+size,data.n)))
            with torch.autocast('cuda',dtype=torch.bfloat16):parts=model(graph(x.reshape(-1,80,3),model.encoder))
            chunks.append(dict(log_likelihood=log_likelihood(parts,target,64).cpu().numpy(),
                cdf=cdf(parts,target.new_tensor(RADII)).cpu().numpy(),mean_A=capped_mean(parts,64).cpu().numpy(),
                median_A=capped_median(parts,64).cpu().numpy()))
        values={k:np.concatenate([r[k] for r in chunks]) for k in chunks[0]}
        if any(not np.isfinite(v).all() for v in values.values()):raise FloatingPointError(f'Nonfinite external predictions: {role}')
        distance=data.distance.cpu().numpy();sources=data.source_ids;weights=source_weights(sources)
        np.savez(tech/f'{role}-predictions.npz',**values,distance=distance,source=sources,material=data.material_ids)
        write_json(tech/f'{role}-lookup.json',dict(sources=data.source_names,materials=data.material_names))
        for name in (f'{role}-predictions.npz',f'{role}-lookup.json'):files[name]=sha(tech/name)
        metrics.append(dict(role=role,**distance_metrics(values,distance,sources,64)))
        for k,radius in enumerate(RADII):
            for threshold in (.5,.75,.95):
                selected=values['cdf'][:,k]>threshold;mass=weights[selected].sum()
                reliability.append(dict(role=role,radius_A=radius,threshold=threshold,selected_rows=int(selected.sum()),
                    coverage=float(mass),mean_probability=float(weights[selected]@values['cdf'][selected,k]/mass) if mass else None,
                    observed_precision=float(weights[selected]@(distance[selected]<=radius)/mass) if mass else None))
        print(json.dumps(dict(stage='external_distance_evaluation',**metrics[-1])),flush=True)
        del data,chunks;torch.cuda.empty_cache()
    analysis=root/'analyses/distance-v1';snapshot_metric_docs(analysis,c['metric_family'])
    csv_rows(analysis/'tables/distance.csv',metrics);csv_rows(analysis/'tables/confidence-reliability.csv',reliability)
    context=json.loads((training/'technical/prediction-context.json').read_text())
    write_json(tech/'prediction-context.json',dict(encoder=context['encoder'],
        predictor=context['predictor'],history_populations=contexts,
        evaluation='external ancestry-disjoint distance holdouts; Ta test is material transfer; no spatial scan paths'))
    training_identity=json.loads((training/'technical/complete.json').read_text())['identity']
    tracking=SimpleNamespace(config=c,root=training,technical=training/'technical',identity=training_identity)
    update_training_summary(tracking,'joint-mace-distance',
        {f'external_distance/{row["role"]}/{k}':v for row in metrics for k,v in row.items() if k!='role'},evaluation='external-distance')
    write_json(complete,dict(state='complete',identity=identity,files=files))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);run(p.parse_args().config)
