"""Frozen-encoder topology and forecast probes after MACE training."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from src.data_utils.pretrained_mace import Quadruplets
from src.data_utils.temporal_campaign import write_json
from src.training_methods.pretrained_mace import Learner
from src.training_methods.mace_performance import encode_views as encode
from src.data_utils.pretrained_mace_gpu import GPUQuadruplets


def ridge_score(x,y,v,w,alpha):
    model=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='cholesky'))
    model.fit(x,y);prediction=model.predict(v)
    return dict(mse=float(np.square(prediction-w).mean()),r2=float(r2_score(w,prediction,multioutput='variance_weighted'))),prediction


@torch.no_grad()
def features(model,gpu,indices,cfg):
    zs=[];targets=[];conditions=[];materials=[]
    for start in range(0,len(indices),cfg['batch_size']):
        x,t,c,m=gpu.get(indices[start:start+cfg['batch_size']])
        zs.append(encode(model,x[:,:4],m,cfg['microbatch_size']).cpu().numpy())
        targets.append(t[:,:4].cpu().numpy());conditions.append(c.cpu().numpy());materials.append(m.cpu().numpy())
    return [np.concatenate(v) for v in (zs,targets,conditions,materials)]


def probe(plan,item):
    cfg=json.loads(Path(item['config']).read_text());out=Path(cfg['output'])
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    data=Quadruplets(cfg)
    model=Learner(cfg).cuda().eval()
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model'],strict=True)
    with np.load(out/'scaling.npz') as f:scaling=dict(f)
    rng=np.random.default_rng(plan['probe_seed'])
    pool=data.all_indices('train');train=pool[rng.choice(len(pool),plan['probe_train_anchors'],replace=False)]
    val=data.all_indices('val');gpu=GPUQuadruplets(data,scaling)
    z,y,c,m=features(model,gpu,train,cfg);v,w,d,n=features(model,gpu,val,cfg)
    alpha=plan['probe_ridge_alpha']
    # New probes are trained identically after freezing every encoder. Disabled
    # in-training heads are never used as a representation-quality metric.
    tda,tda_prediction=ridge_score(z.reshape(-1,256),y.reshape(-1,cfg['tda_components']),v.reshape(-1,256),w.reshape(-1,cfg['tda_components']),alpha)
    x=np.concatenate((z[:,0],c),1);u=np.concatenate((v[:,0],d),1)
    future_tda,future_tda_prediction=ridge_score(x,y[:,3],u,w[:,3],alpha)
    future_latent,prediction=ridge_score(x,z[:,3]-z[:,0],u,v[:,3]-v[:,0],alpha)
    for metric,predicted,target,labels in ((tda,tda_prediction,w.reshape(-1,cfg['tda_components']),n.repeat(4)),(future_tda,future_tda_prediction,w[:,3],n)):
        metric['r2_by_material']={name:float(r2_score(target[labels==i],predicted[labels==i],multioutput='variance_weighted')) for i,name in enumerate(('Al','Mg','Ta'))}
        metric['r2_mean_material']=float(np.mean(list(metric['r2_by_material'].values())))
    base=float(np.square(v[:,3]-v[:,0]).mean())
    future_latent['gain_vs_persistence']=1-future_latent['mse']/base
    eligible=data.temporal_mask(val);ranks={};normalized_temporal={};rng=np.random.default_rng(plan['probe_seed'])
    for i,name in enumerate(('Al','Mg','Ta')):
        a=v[n==i,0];eig=np.linalg.eigvalsh(np.cov(a.T)).clip(0);p=eig/eig.sum();ranks[name]=float(np.exp(-np.sum(p*np.log(np.maximum(p,1e-20)))))
        selected=(n==i)&eligible;a=v[selected,0];t=v[selected,2]
        random=t[rng.permutation(len(t))]
        normalized_temporal[name]=float(np.square(a-t).mean()/np.square(a-random).mean())
    result=dict(name=item['name'],checkpoint_epoch=saved['epoch'],train_anchors=len(train),validation_anchors=len(val),tda_probe=tda,future_tda_probe=future_tda,future_latent_probe=future_latent,effective_rank=ranks,temporal_mse_over_shuffled=normalized_temporal,protocol=f'Fixed train-only ridge probes on frozen encoders, identical seed/data/scalers and alpha. No trained auxiliary head is compared. In-domain validation; one seed; configured budget {cfg["epochs"]} epochs, no convergence claim.')
    write_json(out/'probe_metrics.json',result)


def collect(plan):
    rows=[]
    for item in plan['runs']:
        cfg=json.loads(Path(item['config']).read_text());out=Path(cfg['output'])
        row=dict(model=item['name'],state='pending')
        if (out/'probe_metrics.json').exists():
            p=json.loads((out/'probe_metrics.json').read_text());s=json.loads((out/'training_summary.json').read_text())
            row.update(state='trained_and_probed',steps=s['steps'],tda_probe_r2=p['tda_probe']['r2_mean_material'],future_tda_probe_r2=p['future_tda_probe']['r2_mean_material'],latent_forecast_gain=p['future_latent_probe']['gain_vs_persistence'],**{f'rank_{k}':v for k,v in p['effective_rank'].items()},**{f'temporal_ratio_{k}':v for k,v in p['temporal_mse_over_shuffled'].items()})
        spatial=out/'static_spatial_comparison.csv'
        if spatial.exists():
            with spatial.open() as f:values=[r for r in csv.DictReader(f) if r['model']=='pretrained_MACE_finetuned']
            row.update(state='complete',spatial_neighbor_random_ratio=float(np.mean([float(r['neighbor_mse_over_random']) for r in values])),spatial_cluster_agreement=float(np.mean([float(r['adjusted_neighbor_cluster_agreement']) for r in values])))
        rows.append(row)
    out=Path(plan['output']);write_json(out/'comparison.json',rows)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'comparison.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    def cell(row,key):return f'{row[key]:.4f}' if key in row else 'pending'
    lines=['# MACE training and frozen probes','',
        plan['protocol_description'],'',
        '| Model | State | TDA probe R² | Future TDA probe R² | Latent forecast gain | Spatial neighbor/random |','|---|---|---:|---:|---:|---:|']
    for row in rows:lines.append('| '+ ' | '.join([row['model'],row['state']]+[cell(row,k) for k in ('tda_probe_r2','future_tda_probe_r2','latent_forecast_gain','spatial_neighbor_random_ratio')])+' |')
    lines+=['','Ridge probes are fitted after freezing each encoder, including models trained without TDA or forecasting. Targets are the same train-fitted whitened TDA components. Forecast gain is relative to an unchanged embedding; inspect per-material rank and temporal ratios in the CSV for collapse. Lower spatial neighbor/random ratios mean greater spatial coherence, not verified phases. Static Al contains ancestors of training continuations and is not an independent test.','',
        'TDA R² entries average within-material R² over Al/Mg/Ta, so differences between element means cannot alone produce a good score. Detailed runs and static plots are under `runs/`. Full machine-readable results: [comparison.csv](comparison.csv). Queue status: [status.json](status.json).','']
    (out/'RESULTS.md').write_text('\n'.join(lines))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--run');group.add_argument('--collect',action='store_true');args=parser.parse_args()
    plan=json.loads(Path(args.plan).read_text())
    if args.run:probe(plan,next(r for r in plan['runs'] if r['name']==args.run))
    collect(plan)


if __name__=='__main__':main()
