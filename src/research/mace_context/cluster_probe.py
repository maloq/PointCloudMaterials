"""Check collective-order information with spatially separated readout splits."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch

from src.analysis.liquid_structure import ORDER_NAMES
from src.experiment_runner.artifacts import result_folders, write_json
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.tracking import tracked_run
from src.project_runtime.paths import load_json
from .cluster_diagnosis import project, projector


def run(config):
    root=result_folders(config['probe_output'])
    torch.set_num_threads(config['cpu_threads'])
    with np.load(Path(config['analysis'])/'technical/analysis_inference_cache.npz') as data:
        features,coords=data['inv_latents'],data['coords']
    manifest=json.loads((Path(config['sample_cache'])/'metadata.json').read_text())
    x,y,positions,frames=[],[],[],[]
    offset=0
    for shard in manifest['shards']:
        frame=Path(shard['file']).stem
        p=np.load(Path(config['output'])/f'technical/physical-{frame}.npz')
        selected=p['sampled_rows']; keep=p['ptm'][selected]==0; selected=selected[keep]
        x.append(features[offset+selected]);y.append(p['observables'][keep]);positions.append(coords[offset+selected,0])
        frames.extend([frame]*len(selected));offset+=shard['count']
    x,y,positions,frames=np.concatenate(x),np.concatenate(y).astype(np.float64),np.concatenate(positions),np.asarray(frames)
    train=positions<85;val=(positions>125)&(positions<145);test=positions>185
    y_mean,y_scale=y[train].mean(0),y[train].std(0)
    if np.any(y_scale<=0):raise ValueError('Physical target has no training variation')
    target=(y-y_mean)/y_scale
    representations=dict(dual=x,inner=x[:,:256],center=x[:,256:],projector=project(projector(config),x[:,:256]))
    rows=[]
    for name,z in representations.items():
        z=z.astype(np.float64)
        scaler=StandardScaler().fit(z[train]);z=scaler.transform(z)
        best=np.full(y.shape[1],np.inf);prediction=np.full_like(y,np.nan);chosen=np.zeros(y.shape[1])
        for alpha in config['probe_alphas']:
            model=Ridge(alpha=alpha,solver='svd').fit(z[train],target[train])
            pred=model.predict(z);error=np.mean(np.square(pred[val]-target[val]),axis=0)
            improved=error<best;best[improved]=error[improved];chosen[improved]=alpha
            prediction[:,improved]=pred[:,improved]*y_scale[improved]+y_mean[improved]
        for frame in np.unique(frames):
            ids=test&(frames==frame)
            for col,observable in enumerate(ORDER_NAMES):
                mse=np.mean(np.square(prediction[ids,col]-y[ids,col]));variance=np.var(y[ids,col])
                rows.append(dict(model=name,frame=frame,observable=observable,alpha=chosen[col],
                    test_r2=1-mse/variance, test_samples=int(ids.sum())))
        print('Physical probe',name,'alphas',chosen,flush=True)
    snapshot_metric_docs(root,'analysis')
    pd.DataFrame(rows).to_csv(root/'tables/probe.csv',index=False)
    write_json(root/'technical/protocol.json',dict(train_samples=int(train.sum()),validation_samples=int(val.sum()),test_samples=int(test.sum()),
        train='x < 85 A',validation='125 < x < 145 A',test='x > 185 A',
        representation='Frozen selected checkpoint; ridge-only fits, feature and target scales learned on training slab.',
        numerical_protocol='Float64 standardized features and targets; SVD ridge solver.',
        selection='Each observable chooses ridge alpha on validation slab only. Test R2 uses its own frame/slab mean as variance baseline.',
        limitation='Spatial separation, with gaps exceeding twice the 17 A support. Same six frames and trajectory; no independent-source or kinetic generalization claim. PTM Other only.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();config=load_json(args.config)
    root=result_folders(config['probe_output'])
    with tracked_run(root/'technical/execution',kind='research',configs=[Path(args.config)],command=[sys.executable,*sys.argv],
                     question='Is collective bond order already decodable from the frozen inner representation?'):
        run(config)


if __name__=='__main__':main()
