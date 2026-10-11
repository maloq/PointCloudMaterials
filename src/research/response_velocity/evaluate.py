"""All64 static transfer with explicit sampled-velocity and marginal views."""
import numpy as np
import torch
from ase import units
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from src.research.response_geometry.clusters import references,score
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from .common import FAMILY,root,read,sha,write_json,deadline
from .model import Predictor
import time


def clustering(c,arm,seed):
    name=arm['name']+'-'+str(seed);fitfolder=root(c)/'analyses'/name
    out=root(c)/'cluster-evaluation'/name;receipt=out/'technical/complete.json'
    checkpoint=fitfolder/'technical/best.pt'
    if receipt.exists():
        if read(receipt)['checkpoint_sha256']!=sha(checkpoint):raise ValueError('Clustering checkpoint changed')
        return
    if time.time()>=deadline(c,reserve_seconds=1800):raise TimeoutError('Reserve for complete all64 inference')
    arrays,fit,held,truth=references(c['clustering']);rows=np.r_[fit,held]
    native=c.get('initial_velocity')=='saved_md'
    if native:
        from .native import static_velocities
        native_velocity=static_velocities(c,arrays,fit,held)
    threadpool_limits(2);torch.set_num_threads(2)
    model=Predictor(c,arm['velocities']).cuda()
    model.load_state_dict(torch.load(checkpoint,map_location='cuda',weights_only=False)['model'])
    model.eval().requires_grad_(False)
    z=np.empty((len(rows),2,128),np.float32)
    rng=np.random.default_rng(c['audit_seed']);sigma=np.sqrt(units.kB*450/26.9815385)*units.fs
    with torch.inference_mode():
        for start in range(0,len(rows),c['clustering']['batch_size']):
            take=rows[start:start+c['clustering']['batch_size']]
            q=torch.from_numpy(np.array(arrays['parents'][take,:80],np.float32,copy=True)).cuda()
            velocities=(np.asarray(native_velocity[start:start+len(take)])[:,None] if native else
                rng.normal(size=(len(take),c['velocity_fields'],80,3))*sigma)
            embeds=[]
            for field in range(c['velocity_fields']if arm['velocities']else 1):
                x=torch.cat((q,torch.as_tensor(velocities[:,field],device='cuda',dtype=torch.float32)),-1)
                embeds.append(model.encode(x))
            z[start:start+len(take),0]=embeds[0].cpu().numpy()
            z[start:start+len(take),1]=torch.stack(embeds).mean(0).cpu().numpy()
            if start%(c['clustering']['batch_size']*128)==0:print(name,'all64',start,'/',len(rows),flush=True)
    if not np.isfinite(z).all():raise FloatingPointError('Nonfinite all64 embeddings')
    (out/'technical').mkdir(parents=True,exist_ok=True);metrics=[];assignments={}
    spaces=(['native_velocity'] if native else ['sampled_velocity','velocity_average'])if arm['velocities']else ['geometry']
    for k,space in enumerate(spaces):
        previous=None
        for cluster_seed in c['clustering']['cluster_seeds']:
            km=MiniBatchKMeans(n_clusters=7,batch_size=4096,n_init=3,max_iter=200,
                reassignment_ratio=0,random_state=cluster_seed).fit(z[:len(fit),k])
            labels=km.predict(z[len(fit):,k]).astype(np.uint8)
            row=dict(model=name,space=space,seed=cluster_seed,rows=len(held),**score(labels,truth),
                stability_ari_to_first_seed=1. if previous is None else float(adjusted_rand_score(previous,labels)))
            if previous is None:previous=labels
            metrics.append(row);assignments[space+'_'+str(cluster_seed)]=labels
            np.save(out/'technical'/f'centers-{space}-{cluster_seed}.npy',km.cluster_centers_)
    np.savez_compressed(out/'technical/assignments.npz',**assignments,descriptor=truth,
        source=arrays['source'][held],frame=arrays['frame'][held],atom=arrays['atom'][held])
    write_metric_rows(metrics,out,family=FAMILY,name='clustering')
    update_recorded_summary(fitfolder/'technical/wandb.json',
        {f'clustering/{s}/many_to_one':float(np.mean([r['many_to_one']for r in metrics if r['space']==s]))for s in spaces},
        evaluation='all64-native-velocity-v1' if native else 'all64-sampled-velocity-v1',expected=dict(entity='teshbek',project='PointCloudMaterials'))
    write_json(receipt,dict(state='complete',checkpoint_sha256=sha(checkpoint),training_rows=len(fit),test_rows=len(held),
        velocity_origin='Actual simultaneous same-atom saved MD velocities' if native else 'Independent Maxwell450K draws, NOT measured native velocities; same draws across models',
        independent_samples='24960 observations, no inflation for velocity views',metrics=metrics))
