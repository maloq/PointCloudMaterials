"""Training-fitted clustering spaces, with many-to-one held-out agreement."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

from src.experiment_runner.artifacts import file_hash as sha, write_json
from src.experiment_runner.execution import ExecutionBundle, SlurmQueue
from src.experiment_runner.metric_docs import write_metric_rows, check_metric_docs
from src.project_runtime.paths import resolve_path

FAMILY = 'response_geometry_clusters'


def references(c):
    assay = resolve_path(c['assay'])
    manifest = json.loads((assay.parent/'manifest.json').read_text())
    if manifest['fixed_identity'] != c['fixed_identity']:
        raise ValueError('Changed fixed cohort')
    arrays = {}
    for name in ['role','uniform','source','frame','atom','parents']:
        path = assay/(name+'.npy')
        if sha(path) != manifest['files']['assay/'+name+'.npy']:
            raise ValueError('Changed assay array: '+name)
        arrays[name] = np.load(path, mmap_mode='r')
    fit = np.flatnonzero((arrays['role']=='train') & arrays['uniform'])
    held = np.flatnonzero((arrays['role']=='test') & arrays['uniform'])
    if (len(fit),len(held)) != (74880,24960):
        raise ValueError('All64 uniform comparison population changed')
    path = resolve_path(c['viewer'])
    if sha(path) != c['viewer_sha256']:
        raise ValueError('Changed descriptor reference')
    d = json.loads(path.read_text())['data']
    for name in ['source','frame','atom']:
        if not np.array_equal(arrays[name][held],d[name]):
            raise ValueError('Changed comparison order: '+name)
    return arrays,fit,held,np.array(d['fields']['Joint descriptor clusters'])


def score(labels, truth):
    t = np.zeros((7,7),np.int64);np.add.at(t,(labels,truth),1)
    i,j = linear_sum_assignment(t,maximize=True)
    p = t.sum(1)/t.sum();nonzero=p>0
    return dict(many_to_one=float(t.max(1).sum()/t.sum()),
        one_to_one=float(t[i,j].sum()/t.sum()),
        adjusted_rand=float(adjusted_rand_score(truth,labels)),
        adjusted_mutual_information=float(adjusted_mutual_info_score(truth,labels)),
        occupied_clusters=int(nonzero.sum()),largest_cluster_fraction=float(p.max()),
        effective_clusters=float(np.exp(-(p[nonzero]*np.log(p[nonzero])).sum())))


def evaluate_model(c, model, name, checkpoint_sha, arrays, fit, held):
    import torch
    out = resolve_path(c['output'])/'analyses'/name
    receipt = out/'technical/complete.json'
    if receipt.exists():
        r=json.loads(receipt.read_text())
        if r['checkpoint_sha256'] != checkpoint_sha:raise ValueError('Changed clustering checkpoint')
        return
    (out/'technical').mkdir(parents=True,exist_ok=True)
    model.eval().requires_grad_(False)
    rows=np.r_[fit,held];z=np.empty((len(rows),128),np.float32);pred=np.empty((len(rows),256),np.float32)
    with torch.inference_mode():
        for start in range(0,len(rows),c['batch_size']):
            take=rows[start:start+c['batch_size']]
            q=torch.from_numpy(np.array(arrays['parents'][take,:80],np.float32,copy=True)).cuda()
            encoded=model.encode(q);z[start:start+len(take)]=encoded.cpu().numpy()
            pred[start:start+len(take)]=model.head(encoded).cpu().numpy()
            if start % (c['batch_size']*128)==0:print(name,'encoded',start,'/',len(rows),flush=True)
    if not np.isfinite(z).all() or not np.isfinite(pred).all():raise ValueError('Nonfinite frozen representation')
    d=json.loads(resolve_path(c['viewer']).read_text())['data'];truth=np.array(d['fields']['Joint descriptor clusters'])
    mean=z[:len(fit)].mean(0,dtype=np.float64)
    covariance=np.cov(z[:len(fit)].astype(np.float64),rowvar=False,bias=True)
    e,v=np.linalg.eigh(covariance);alpha=c['whitening_shrinkage']
    regularized=(1-alpha)*e+alpha*e.mean()
    whitening=(v/np.sqrt(regularized))@v.T
    np.savez_compressed(out/'technical/normalization.npz',mean=mean,whitening=whitening,eigenvalues=e,
                        fit_rows=fit,held_rows=held)
    metrics=[];assignments={}
    for space in ['native','whitened','predicted_future']:
        x=z if space=='native' else ((z-mean)@whitening).astype(np.float32) if space=='whitened' else pred
        previous=None
        for seed in c['cluster_seeds']:
            km=MiniBatchKMeans(n_clusters=7,batch_size=4096,n_init=3,max_iter=200,
                reassignment_ratio=0,random_state=seed).fit(x[:len(fit)])
            labels=km.predict(x[len(fit):]).astype(np.uint8)
            row=dict(model=name,space=space,seed=seed,rows=len(held),**score(labels,truth),
                stability_ari_to_first_seed=1. if previous is None else float(adjusted_rand_score(previous,labels)))
            if previous is None:previous=labels
            metrics.append(row);assignments[space+'_'+str(seed)]=labels
            np.save(out/'technical'/f'centers-{space}-{seed}.npy',km.cluster_centers_)
            print(row,flush=True)
    np.savez_compressed(out/'technical/assignments.npz',**assignments,descriptor=truth,
        source=arrays['source'][held],frame=arrays['frame'][held],atom=arrays['atom'][held])
    write_metric_rows(metrics,out,family=FAMILY,name='clustering')
    write_json(receipt,dict(state='complete',checkpoint_sha256=checkpoint_sha,
        reference_sha256=c['viewer_sha256'],training_rows=len(fit),test_rows=len(held),
        encoder_fitting=False,normalization_fit='train only',cluster_selection='none: all fixed variants reported',
        feature_cache='in memory only; assignments and centers retained',metrics=metrics))


def run(c):
    import torch
    from src.research.local_response.model import Predictor
    torch.set_num_threads(2);threadpool_limits(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for filename,expected in c['model_dependencies'].items():
        if sha(Path(filename)) != expected:raise ValueError('Changed native model producer: '+filename)
    arrays,fit,held,_=references(c)
    for item in c['models']:
        checkpoint=resolve_path(item['checkpoint'])
        if sha(checkpoint)!=item['sha256']:raise ValueError('Changed checkpoint')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=Predictor(c['model_config']).cuda();model.load_state_dict(saved['model'],strict=True)
        evaluate_model(c,model,item['name'],item['sha256'],arrays,fit,held)
        del model;torch.cuda.empty_cache()
    publish(c)


def publish(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=resolve_path(c['output']);rows=[]
    for path in sorted((out/'analyses').glob('*/technical/complete.json')):
        rows.extend(json.loads(path.read_text())['metrics'])
    if not rows:raise ValueError('No completed clustering analyses')
    dest=out/'analyses/comparison-v1'
    write_metric_rows(rows,dest,family=FAMILY,name='clustering')
    names=sorted(set(r['model'] for r in rows));spaces=['native','whitened','predicted_future']
    fig,ax=plt.subplots(figsize=(12,max(4,len(names)*.45)),layout='constrained')
    for i,space in enumerate(spaces):
        groups=[[r['many_to_one']*100 for r in rows if r['model']==n and r['space']==space]for n in names]
        ax.barh(np.arange(len(names))+(i-1)*.25,[np.mean(g)for g in groups],height=.24,
                xerr=[np.std(g)for g in groups],label=space)
    ax.set(yticks=np.arange(len(names)),yticklabels=names,xlabel='Many-to-one agreement (%)',
           title='Unrelaxed held-out Al · fixed K=7 · training-fitted clusters');ax.legend()
    (dest/'plots').mkdir(parents=True,exist_ok=True);fig.savefig(dest/'plots/clustering.png',dpi=170);plt.close(fig)
    write_json(out/'technical/complete.json',dict(state='complete',models=len(names),rows=len(rows)))


def launch(c):
    check_metric_docs(family=FAMILY)
    out=resolve_path(c['output']);tech=out/'technical';tech.mkdir(parents=True,exist_ok=True)
    path=tech/'launch.json'
    if path.exists():raise FileExistsError(path)
    repo=Path(__file__).resolve().parents[3]
    bundle=ExecutionBundle.freeze(repo,tech/'code',c,directories=('src','docs/metrics'))
    receipt=dict(jobs={},code=str(bundle.root));environment=dict(PCM_PROJECT_ROOT=str(repo),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1')
    queue=SlurmQueue(tech,bundle,'src.research.response_geometry.clusters',environment,path,receipt,'RESP-CLUSTER')
    with queue.submission():
        queue.submit('run',['--gpus=1','--cpus-per-task=4','--mem=32G','--time=04:00:00','--exclude=node52'],partition=c['gpu_partition'])
    print(receipt,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['run','launch','publish']);p.add_argument('--config',required=True)
    a=p.parse_args();c=json.loads(Path(a.config).read_text());globals()[a.stage](c)
