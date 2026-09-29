"""Frozen, source-separated cluster/physical readouts and spatial-field controls."""
import argparse
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
import torch

from src.experiment_runner.metric_docs import write_metric_table
from src.research.equivariant_context.cache import RetainedCache
from src.research.structural_state.common import write_json, sha, digest
from src.research.supervised_onset.tracking import update_training_summary
from .data import load
from .train import initialization, settings


def arrays(c):
    root=Path(c['cache'])/'assay'
    a={p.stem:np.load(p,mmap_mode='r') for p in root.glob('*.npy')}
    a['descriptors']=json.loads((root/'descriptors.json').read_text())
    return a


def weights(source):
    _,inverse,counts=np.unique(source,return_inverse=True,return_counts=True)
    return len(source)/(len(counts)*counts[inverse])


def ridge(x,y,fit,source,alpha):
    """Fixed penalized Gaussian-mean readout; training ancestors only."""
    w=weights(source[fit]);xf=np.asarray(x[fit],float);yf=np.asarray(y[fit],float)
    mean=np.average(xf,axis=0,weights=w);scale=np.sqrt(np.average((xf-mean)**2,axis=0,weights=w))
    scale[scale<1e-8]=1
    design=np.column_stack([np.ones(len(xf)),(xf-mean)/scale])
    penalty=np.eye(design.shape[1])*alpha;penalty[0,0]=0
    coefficient=np.linalg.solve(design.T@(design*w[:,None])+penalty,design.T@(yf*w[:,None]))
    return dict(mean=mean,scale=scale,coefficient=coefficient)


def predict(model,x):
    return model['coefficient'][0]+((x-model['mean'])/model['scale'])@model['coefficient'][1:]


def nuisance(a):
    d=np.minimum(a['distance'],60)/20
    phi=a['support_fraction'];f=a['solid_fraction']
    columns=[phi,phi**2,f,f**2,d,d**2,~a['crystal_present']]
    columns += [np.maximum(d-v,0) for v in (.2,.4,.8,1.6)]
    columns += [a['ptm']==v for v in (1,2,3)]
    # Density is a current local geometry control, never a predictor covariate.
    columns += [a['targets'][:,a['descriptors'].index('geometry/density12')],
                a['targets'][:,a['descriptors'].index('geometry/density80')]]
    return np.column_stack(columns).astype(np.float64)


def cohorts(a):
    return dict(all=np.ones(len(a['source']),bool), noncrystal_center=~np.isin(a['ptm'],[1,2,3]),
        clear_input=~a['visible_A'], mixed_noncrystal_center=(~np.isin(a['ptm'],[1,2,3]))&a['visible_A'],
        clear_pair_union=~a['visible_union'], clear_augmentation_union=~a['visible_augmentation_union'],
        at_most_one_detected=a['support_fraction']<=1/80+1e-8,
        at_most_three_detected=a['support_fraction']<=3/80+1e-8,
        crystal_absent_cell=~a['crystal_present'])


def tracks(a):
    result=dict(uniform=np.asarray(a['uniform']).copy())
    for name in ('all64','legacy16'):
        mask=np.zeros(len(a['source']),bool);mask[a[name+'_order']]=True;result[name]=mask
    return result


def mean_predict(y,labels,fit,source,k,shrink):
    w=weights(source[fit]);global_mean=np.average(y[fit],axis=0,weights=w)
    values=np.broadcast_to(global_mean,(k,y.shape[1])).copy();count=[]
    for j in range(k):
        m=labels[fit]==j;n=float(w[m].sum());count.append(n)
        if n:values[j]=(np.sum(y[fit][m]*w[m,None],axis=0)+shrink*global_mean)/(n+shrink)
    return values,np.asarray(count)


def shuffled_labels(labels,a):
    """Preserve source and coarse physical nuisances; break residual atom correspondence."""
    distance_bin=np.digitize(a['distance'],[3,6,12,24,48])
    fraction_bin=np.digitize(a['support_fraction'],[0,.05,.25,.5,.75,1],right=True)
    keys=np.column_stack([a['source'],a['visible_A'],distance_bin,fraction_bin])
    _,group=np.unique(keys,axis=0,return_inverse=True)
    order=np.argsort(group,kind='stable');cuts=np.r_[0,1+np.flatnonzero(np.diff(group[order])),len(order)]
    rng=np.random.default_rng(20260929);out=labels.copy();movable=0
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        rows=order[lo:hi];out[rows]=labels[rng.permutation(rows)]
        if len(rows)>1:movable+=len(rows)
    return out,movable/len(labels)


def uncertainty(values,seed=20260929,repetitions=1000):
    values=np.asarray(values)
    if len(values)<3:return None
    rng=np.random.default_rng(seed)
    result=values[rng.integers(len(values),size=(repetitions,len(values)))].mean(1)
    return np.quantile(result,[.025,.975]).tolist()


def family_scores(errors,reference,names):
    output={}
    for family in ('geometry','bond_order','cna','tda'):
        # The conditional control observes density; don't count its identity readout as information gain.
        columns=np.asarray([i for i,n in enumerate(names) if n.startswith(family+'/') and '/density' not in n])
        if not len(columns):continue
        source_error=errors[:,columns].mean(1);source_base=reference[:,columns].mean(1)
        denom=source_base.mean()
        output[family]=dict(active_features=len(columns),train_variance_normalized_mse=float(source_error.mean()),
            train_mean_skill=float(1-source_error.mean()/denom) if denom>1e-12 else None,
            normalized_mse_uncertainty=dict(ci95=uncertainty(source_error)))
    return output


def score_readouts(z,a,labels,k,root,tag):
    cs=cohorts(a);ts=tracks(a);role=a['role'];source=a['source'];uniform=a['uniform']
    base=nuisance(a)
    train=(role=='train')&uniform
    mu=np.mean(a['targets'][train],axis=0);sd=np.std(a['targets'][train],axis=0)
    active=sd>1e-8*np.maximum(1,np.abs(mu))
    names=[n for n,v in zip(a['descriptors'],active) if v]
    y=(a['targets'][:,active].astype(float)-mu[active])/sd[active]
    permuted,movable=shuffled_labels(labels,a)
    output=dict(active_features=int(active.sum()),permutation_movable_fraction=movable,cohorts={})
    meta=dict(descriptor_names=names,target_mean=mu[active].tolist(),target_scale=sd[active].tolist())
    write_json(root/'technical'/f'{tag}-readout.json',meta)
    artifacts={};saved_predictions={}
    readout_cohorts=['all','noncrystal_center','clear_input']
    if k==7:readout_cohorts+=['at_most_one_detected','at_most_three_detected']
    for cohort in readout_cohorts:
        fit=train&cs[cohort]
        if fit.sum()<128 or len(np.unique(source[fit]))<3:
            output['cohorts'][cohort]=dict(training_rows=int(fit.sum()),state='insufficient_training_support');continue
        onehot=np.eye(k)[labels];permuted_hot=np.eye(k)[permuted]
        fit_mean=np.average(y[fit],axis=0,weights=weights(source[fit]))
        means,counts=mean_predict(y,labels,fit,source,k,20)
        perm_means,_=mean_predict(y,permuted,fit,source,k,20)
        models={
            'phase_density':(ridge(base,y,fit,source,10.),base),
            'phase_density_cluster':(ridge(np.column_stack([base,onehot]),y,fit,source,10.),np.column_stack([base,onehot])),
            'phase_density_permuted_cluster':(ridge(np.column_stack([base,permuted_hot]),y,fit,source,10.),np.column_stack([base,permuted_hot]))}
        # One continuous readout per K=7 analysis; other K values reuse the same scientific contrast.
        if k==7:models['continuous_embedding']=(ridge(z,y,fit,source,10.),z)
        for model,(packet,_) in models.items():
            for key,val in packet.items():artifacts[f'{cohort}/{model}/{key}']=val
        artifacts[f'{cohort}/cluster_means']=means;artifacts[f'{cohort}/cluster_counts']=counts
        result=dict(training_rows=int(fit.sum()),training_sources=len(np.unique(source[fit])),tracks={})
        for track,tm in ts.items():
            record={}
            for split in ('selection','calibration','test'):
                mask=tm&cs[cohort]&(role==split)
                ids=np.flatnonzero(mask);sources=np.unique(source[ids])
                rr=dict(rows=len(ids),sources=len(sources),models={})
                if len(ids)<30 or len(sources)<3:
                    rr['state']='insufficient_evaluation_support';record[split]=rr;continue
                pred={'cluster_means':means[labels[ids]],'permuted_cluster_means':perm_means[permuted[ids]]}
                pred.update({name:predict(packet,x[ids]) for name,(packet,x) in models.items()})
                base_error=np.stack([np.mean((y[ids][source[ids]==s]-fit_mean)**2,axis=0) for s in sources])
                errors={}
                for name,prediction in pred.items():
                    error=(y[ids]-prediction)**2
                    se=np.stack([error[source[ids]==s].mean(0) for s in sources]);errors[name]=se
                    rr['models'][name]=family_scores(se,base_error,names)
                    saved_predictions[f'{cohort}/{track}/{split}/{name}/source_mse']=se.astype(np.float32)
                saved_predictions[f'{cohort}/{track}/{split}/source_ids']=sources
                saved_predictions[f'{cohort}/{track}/{split}/mean_source_mse']=base_error.astype(np.float32)
                delta=errors['phase_density']-errors['phase_density_cluster']
                if track=='uniform' and split=='test':
                    mean_error=base_error.mean(0);cluster_error=errors['cluster_means'].mean(0)
                    rr['feature_readouts']={n:dict(normalized_mse=float(cluster_error[j]),
                        train_mean_skill=float(1-cluster_error[j]/mean_error[j]) if mean_error[j]>1e-12 else None,
                        conditional_gain=float(delta[:,j].mean())) for j,n in enumerate(names)}
                rr['conditional_cluster_gain']={}
                for family in ('geometry','bond_order','cna','tda'):
                    columns=[i for i,n in enumerate(names) if n.startswith(family+'/') and '/density' not in n]
                    if not columns:continue
                    gain=delta[:,columns].mean(1)
                    rr['conditional_cluster_gain'][family]=dict(delta_normalized_mse=float(gain.mean()),
                        ci95=uncertainty(gain))
                record[split]=rr
            result['tracks'][track]=record
        output['cohorts'][cohort]=result
    np.savez_compressed(root/'data'/f'{tag}-readout-models.npz',**artifacts)
    np.savez_compressed(root/'data'/f'{tag}-source-errors.npz',**saved_predictions)
    return output


def spatial_metrics(z,b,a,labels,neighbor_labels):
    output={};train=(a['role']=='train')&a['uniform']
    scale=float(np.mean(np.var(z[train],axis=0)))
    if scale<=1e-14:raise ValueError('Representation is collapsed; spatial distances cannot be normalized')
    delta=np.mean((z-b)**2,axis=1)/scale
    for cohort,cm in cohorts(a).items():
        mask=cm&a['uniform']&(a['role']=='test');x=z[mask].astype(float)
        record=dict(rows=int(mask.sum()),sources=len(np.unique(a['source'][mask])))
        if len(x)<30:output[cohort]=record;continue
        cov=np.cov(x,rowvar=False);eig=np.maximum(np.atleast_1d(np.linalg.eigvalsh(np.atleast_2d(cov))),0)
        p=eig/max(eig.sum(),1e-15)
        record.update(effective_rank=float(np.exp(-np.sum(p[p>0]*np.log(p[p>0])))),
            variance_relative_to_training=float(np.mean(np.var(x,axis=0))/scale),
            neighbor_squared_distance=float(np.mean(delta[mask])),
            ptm_ami=float(adjusted_mutual_info_score(a['ptm'][mask],labels[mask])))
        ids=np.flatnonzero(mask);random_neighbor=neighbor_labels[ids].copy()
        _,group=np.unique(np.column_stack([a['source'][ids],a['frame'][ids]]),axis=0,return_inverse=True)
        rng=np.random.default_rng(20260929)
        for g in np.unique(group):
            where=np.flatnonzero(group==g);random_neighbor[where]=random_neighbor[rng.permutation(where)]
        record['neighbor_cluster_agreement']=float(np.mean(labels[ids]==neighbor_labels[ids]))
        record['same_source_frame_permuted_neighbor_agreement']=float(np.mean(labels[ids]==random_neighbor))
        crossing=mask&(np.abs(a['support_fraction']-a['support_fraction_B'])>=.25)
        record['support_fraction_crossing_rows']=int(crossing.sum())
        record['support_fraction_crossing_distance']=float(delta[crossing].mean()) if crossing.any() else None
        output[cohort]=record
    return output


def field_profiles(z,a,labels,root,tag):
    train=(a['role']=='train')&a['uniform'];test=(a['role']=='test')&a['uniform']
    # Select native coordinates on train ancestors. Coordinate indices do not align across seeds.
    finite=np.isfinite(a['progress']);fit=train&finite;held=test&finite
    cor=np.asarray([spearmanr(z[fit,j],a['progress'][fit]).statistic if np.ptp(z[fit,j]) else 0 for j in range(z.shape[1])])
    chosen=np.argsort(-np.abs(cor))[:min(6,z.shape[1])]
    crystal=train&(a['support_fraction']>=.9);liquid=train&(a['support_fraction']<=.05)
    output=dict(selected_coordinates=chosen.tolist(),coordinate_selection='training sources only',
        fit_rows=int(fit.sum()),heldout_rows=int(held.sum()))
    if min(crystal.sum(),liquid.sum())<30:return dict(output,state='insufficient_bulk_for_axis')
    axis=z[liquid].mean(0)-z[crystal].mean(0);norm=float(axis@axis)
    if norm<1e-14:return dict(output,state='degenerate_phase_axis')
    score=(z-z[crystal].mean(0))@axis/norm
    xx=[];yy=[];counts=[]
    edges=np.linspace(-24,32,29)
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=held&(a['progress']>=lo)&(a['progress']<hi)
        if m.sum()<30:continue
        xx.append((lo+hi)/2);yy.append(float(np.median(score[m])));counts.append(int(m.sum()))
    crossing={}
    for level in (.1,.9):
        crossing[str(level)]=[float(x0+(level-y0)*(x1-x0)/(y1-y0))
            for x0,x1,y0,y1 in zip(xx[:-1],xx[1:],yy[:-1],yy[1:])
            if x1-x0<=2.01 and y0<level<=y1]
    width=None
    if len(crossing['0.1'])==len(crossing['0.9'])==1 and crossing['0.9'][0]>crossing['0.1'][0]:
        width=crossing['0.9'][0]-crossing['0.1'][0]
    output.update(transition_width_10_90_A=width,upcrossings=crossing,
        phase_profile=dict(distance=xx,median=yy,count=counts),
        phase_profile_reversals=sum(y1<y0-.1 for y0,y1 in zip(yy[:-1],yy[1:])))
    phase=ridge(nuisance(a),score[:,None],train,a['source'],10.)
    residual=score-predict(phase,nuisance(a))[:,0]
    np.savez_compressed(root/'data'/f'{tag}-phase-axis.npz',axis=axis,score=score.astype(np.float32),
        nuisance_residual=residual.astype(np.float32),selected_coordinates=chosen,
        native_coordinates=z[:,chosen].astype(np.float32))
    for key,values in [('phase_axis',score)]+[(f'coordinate_{j}',z[:,j]) for j in chosen]:
        output[key]=dict(train_spearman=float(spearmanr(values[fit],a['progress'][fit]).statistic),
            test_spearman=float(spearmanr(values[held],a['progress'][held]).statistic))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    rng=np.random.default_rng(20260929);ids=np.flatnonzero(held);ids=rng.choice(ids,min(len(ids),8000),replace=False)
    axes[0].scatter(a['progress'][ids],score[ids],c=labels[ids],s=2,cmap='tab10',alpha=.35,rasterized=True)
    axes[0].set(xlabel='Distance contrast: crystal core → liquid (Å)',ylabel='Training-defined phase coordinate')
    edges=np.linspace(-24,32,29);table={}
    for j in chosen:
        mean=float(z[train,j].mean());std=float(z[train,j].std());values=(z[:,j]-mean)/max(std,1e-12)
        xx=[];yy=[];counts=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            m=held&(a['progress']>=lo)&(a['progress']<hi)
            if m.sum()<30:continue
            xx.append((lo+hi)/2);yy.append(float(np.median(values[m])));counts.append(int(m.sum()))
        axes[1].plot(xx,yy,marker='.',label=f'z[{j}]');table[str(j)]=dict(x=xx,median=yy,count=counts)
    axes[1].set(xlabel='Distance contrast (Å)',ylabel='Native coordinate / training SD');axes[1].legend(fontsize=7)
    fig.suptitle(tag+' · held-out sources');fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(root/'plots'/f'{tag}-profiles.{ext}',dpi=180)
    plt.close(fig);write_json(root/'data'/f'{tag}-binned-profiles.json',table)
    return output


def analyze(z,b,a,c,root,tag):
    root=Path(root)
    for part in ('data','plots','tables','technical'):(root/part).mkdir(parents=True,exist_ok=True)
    train=(a['role']=='train')&a['uniform']
    result={}
    for k in c['assay']['ks']:
        km=MiniBatchKMeans(n_clusters=k,batch_size=4096,n_init=3,max_iter=200,
            random_state=20260929,reassignment_ratio=0).fit(z[train])
        labels=km.predict(z)
        name=f'{tag}-k{k}'
        np.savez_compressed(root/'data'/f'{name}-assignments.npz',cluster=labels.astype(np.uint8),
            centers=km.cluster_centers_,source=a['source'],frame=a['frame'],atom=a['atom'],
            all64_order=a['all64_order'],legacy16_order=a['legacy16_order'])
        block=dict(readouts=score_readouts(z,a,labels,k,root,name))
        if b is not None:block['spatial']=spatial_metrics(z,b,a,labels,km.predict(b))
        if k==7:block['profiles']=field_profiles(z,a,labels,root,name)
        result[f'k{k}']=block
        print(f'Finished {name}',flush=True)
    write_json(root/'technical'/f'{tag}-metrics.json',result)
    write_metric_table(result,root,family='spatial_vicreg_bias',name=tag)
    return result


@torch.inference_mode()
def encode(model,a,c,path):
    n=len(a['source']);out=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(n,2,2,128))
    for first in range(0,n,256):
        p=np.asarray(a['parents'][first:first+256]);idx=np.asarray(a['view_indices'][first:first+256])
        choice=np.asarray(a['pair_choice'][first:first+256]);near=idx[np.arange(len(p)),choice]
        b=np.take_along_axis(p,near[:,:,None],axis=1);b=b-b[:,:1]
        x=torch.as_tensor(np.concatenate([p[:,:80],b])/c['geometry']['length_scale_A'],device='cuda')
        z,y=model(x);values=torch.stack([z,y],1).cpu().numpy().reshape(2,len(p),2,128).transpose(1,0,2,3)
        if not np.isfinite(values).all():raise FloatingPointError(f'Nonfinite inference at {first}')
        out[first:first+len(p)]=values
        if first==0:
            zz,yy=model(x);repeat=torch.stack([zz,yy],1).cpu().numpy().reshape(2,len(p),2,128).transpose(1,0,2,3)
            if not np.array_equal(values,repeat):raise ValueError('Matched-batch inference changed')
    out.flush();del out


def publish(c,study,name,result,epoch):
    if epoch not in c['assay']['primary_epochs']:return
    fields={f'evaluation/epoch{epoch}/{rep}/clear_tda_cluster_skill':
        result[rep]['k7']['readouts']['cohorts']['clear_input']['tracks']['uniform']['test']['models']['cluster_means']['tda']['train_mean_skill']
        for rep in ('encoder','projector')}
    update_training_summary(study,name,fields,evaluation=f'spatial-epoch{epoch}')


def run(config,index,deadline=None):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    c,seed,alpha,name,recipe,study=settings(config,index);a=arrays(c)
    cache=RetainedCache(c['feature_cache'],limit=6)
    for epoch in c['assay']['epochs']:
        if deadline is not None and time.time()>deadline-600:return False
        root=study.root/'analyses'/f'epoch-{epoch:02d}'
        if (root/'technical/complete.json').exists():
            publish(c,study,name,json.loads((root/'technical/complete.json').read_text())['metrics'],epoch)
            continue
        checkpoint=study.root/f'checkpoints/epoch-{epoch:02d}.pt';checkpoint_hash=sha(checkpoint)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['identity']!=study.identity or saved['offset']!=0:raise ValueError('Incomplete or changed checkpoint')
        model=initialization(recipe,seed);model.load_state_dict(saved['model'],strict=True);model.eval()
        key=digest(dict(checkpoint=checkpoint_hash,data=saved['data_identity'],producer=sha(Path(__file__))))
        with cache.lease(key,deadline=time.time()+23*3600,metadata=dict(checkpoint=str(checkpoint))) as folder:
            features=folder/'features.npy'
            marker=folder/'features-complete.json'
            if marker.exists():
                if sha(features)!=json.loads(marker.read_text())['sha256']:raise ValueError('Changed feature cache')
            else:
                encode(model,a,c,features);write_json(marker,dict(sha256=sha(features)))
            del model;torch.cuda.empty_cache();values=np.load(features,mmap_mode='r')
            result={}
            for i,rep in enumerate(('encoder','projector')):
                result[rep]=analyze(values[:,0,i],values[:,1,i],a,c,root,rep)
            write_json(root/'technical/complete.json',dict(checkpoint_sha256=checkpoint_hash,
                producer_sha256=sha(Path(__file__)),epoch=epoch,data_identity=saved['data_identity'],metrics=result))
            del values
        # Attach final primary diagnostics to the existing scientific run. Never start an evaluation run.
        publish(c,study,name,result,epoch)
    return True


def nulls(config,deadline=None):
    c=load(config);a=arrays(c)
    for i,step in enumerate((0,1,2,4)):
        if deadline is not None and time.time()>deadline-600:return False
        root=Path(c['output'])/'nulls'/'analyses'/f'diffusion-{step}'
        if (root/'technical/complete.json').exists():continue
        z=np.asarray(a['null_fields'][:,i:i+1])
        result=analyze(z,None,a,c,root,f'diffusion-{step}')
        exposure={name:dict(rows=int(mask.sum()),crystal_reached_fraction=float(a['null_exposure'][mask,i].mean()))
            for name,mask in cohorts(a).items() if mask.any()}
        write_json(root/'technical/complete.json',dict(state='complete',privileged_reference_control=True,
            encoder_equivalent_support=False,steps=step,exposure=exposure,metrics=result))
    for name,column in [('local-q6','bond_order/l6_q'),('averaged-q6','bond_order/l6_qbar')]:
        if deadline is not None and time.time()>deadline-600:return False
        root=Path(c['output'])/'nulls'/'analyses'/name
        if (root/'technical/complete.json').exists():continue
        j=a['descriptors'].index(column)
        result=analyze(np.asarray(a['targets'][:,j:j+1]),None,a,c,root,name)
        write_json(root/'technical/complete.json',dict(state='complete',input='same 80-atom geometry',
            target_overlap=[column],interpretation='Self-readout is tautological; compare other feature families',metrics=result))
    return True


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['encoder','nulls'])
    p.add_argument('--config',required=True);p.add_argument('--index',type=int,default=0);args=p.parse_args()
    if args.stage=='nulls':nulls(args.config)
    else:run(args.config,args.index)
