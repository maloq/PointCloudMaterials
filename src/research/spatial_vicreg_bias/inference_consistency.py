"""Diagnostic replay of frozen inference and the original K-means label kernel."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.cluster._kmeans import _labels_inertia_threadpool_limit
from sklearn.metrics import pairwise_distances_argmin
from src.research.spatial_vicreg_bias.train import PairEncoder


def run(config,output):
    c=json.loads(Path(config).read_text());corr=json.loads(Path(c['correspondence_config']).read_text())
    parent=json.loads(Path(corr['parent']).read_text());base=Path(parent['cache'])/'assay'
    rows=np.flatnonzero(np.load(base/'uniform.npy')&(np.load(base/'role.npy')=='test'))
    parents=np.load(base/'parents.npy',mmap_mode='r');root=Path(parent['output'])/'S0-seed17'
    saved=torch.load(root/'checkpoints/epoch-04.pt',map_location='cpu',weights_only=False)
    torch.set_num_threads(2);model=PairEncoder(OmegaConf.create(saved['recipe'])).cpu()
    model.load_state_dict(saved['model'],strict=True);model.eval();model.requires_grad_(False)
    values=[]
    with torch.inference_mode():
        for first in range(0,len(rows),256):
            x=torch.from_numpy(np.asarray(parents[rows[first:first+256],:80])/parent['geometry']['length_scale_A'])
            z,y=model(x);values.append(torch.stack([z,y],1).numpy())
    values=np.concatenate(values);report={}
    for j,rep in enumerate(('encoder','projector')):
        with np.load(root/f'analyses/epoch-04/data/{rep}-k7-assignments.npz') as a:
            own=a['cluster'][rows];centers=a['centers']
        x=np.ascontiguousarray(values[:,j]);generic=pairwise_distances_argmin(x,centers)
        native=_labels_inertia_threadpool_limit(x,np.ones(len(x),dtype=x.dtype),centers,n_threads=1,return_inertia=False)
        distances=np.sum((x[:,None,:].astype(float)-centers[None,:,:].astype(float))**2,axis=2)
        exact=np.argmin(distances,axis=1);changed=(generic!=own)|(native!=own)|(exact!=own)
        report[rep]=dict(rows=len(x),generic_disagreements=int(np.sum(generic!=own)),
                         producer_kernel_disagreements=int(np.sum(native!=own)),float64_disagreements=int(np.sum(exact!=own)),
                         original_rows=rows[changed].tolist(),original_labels=own[changed].tolist(),
                         generic_labels=generic[changed].tolist(),producer_kernel_labels=native[changed].tolist(),
                         float64_labels=exact[changed].tolist(),squared_distances=distances[changed].tolist(),
                         mean_square_embedding=float(np.mean(x*x)),mean_coordinate_variance=float(np.mean(np.var(x,axis=0))))
    Path(output).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.config,a.output)
