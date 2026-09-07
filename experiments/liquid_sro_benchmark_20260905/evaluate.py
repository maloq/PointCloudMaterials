"""Source-held-out prediction of liquid structure and dynamics from frozen encoders."""
import argparse
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.distance import cdist
from experiments.smooth_temporal_encoder_20260905.prepare import write_json

FIXED=('SOAP','TDA','TDA_16','DensityPCA','GeoFrame_pretrained')
LEARNED=('MACE','SchNet','DensityMLP','MACE_untrained')
CONTROLS=('ShuffledSOAP',)


def standardize(x,train):
    x=np.asarray(x,dtype=np.float64)
    mean=x[train].mean(0);std=x[train].std(0)
    # Exact constants (including single-material indicators) have no information.
    std=np.maximum(std,max(float(np.median(std))*.001,1e-10))
    return (x-mean)/std,mean,std


def target_pca(x,train,dim):
    # Future TDA input [centers,horizons,144]; fitted on training centers only.
    train_values=x[train].reshape(-1,x.shape[-1]).astype(np.float64)
    mean=train_values.mean(0)
    centered=train_values-mean
    values,vectors=np.linalg.eigh(centered.T@centered/(len(centered)-1))
    basis=vectors[:,-dim:].copy()
    return (x-mean)@basis,dict(mean=mean,basis=basis,explained_fraction=float(values[-dim:].sum()/values.sum()))


def ridge(x,y,split,alphas,clip=False):
    train,val,test=(split==s for s in ('train','val','test'))
    z,mean,std=standardize(x,train)
    center=y[train].mean(0)
    xtx=z[train].T@z[train];xty=z[train].T@(y[train]-center)
    best=float('inf');selected=None
    for alpha in alphas:
        weight=np.linalg.solve(xtx+float(alpha)*np.eye(xtx.shape[0]),xty)
        prediction=z[val]@weight+center
        if clip:prediction=np.clip(prediction,0.,1.)
        error=float(np.square(prediction-y[val]).mean())
        if error<best:best=error;selected=(alpha,weight)
    alpha,weight=selected
    prediction=z[test]@weight+center
    if clip:prediction=np.clip(prediction,0.,1.)
    return prediction,dict(alpha=float(alpha),validation_mse=best),dict(input_mean=mean,input_std=std,weight=weight,intercept=center)


def bootstrap_skill(error,baseline,source,draws):
    groups=np.unique(source)
    a=np.array([error[source==g].sum() for g in groups])
    b=np.array([baseline[source==g].sum() for g in groups])
    samples=100*(1-a[draws].sum(1)/b[draws].sum(1))
    return dict(skill_pct=float(100*(1-error.mean()/baseline.mean())),
                source_ci95_pct=np.quantile(samples,[.025,.975]).tolist(),mse=float(error.mean()))


def save_csv(path,rows):
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def neighbor_candidates(coarse,temp,split,count):
    train,test=np.flatnonzero(split=='train'),np.flatnonzero(split=='test')
    z,_,_=standardize(coarse,split=='train')
    result=np.empty((len(test),count),dtype=np.int64)
    for temperature in np.unique(temp):
        tr=train[temp[train]==temperature];local=np.flatnonzero(temp[test]==temperature)
        distances=cdist(z[test[local]],z[tr],metric='sqeuclidean')
        result[local]=tr[np.argsort(distances,axis=1)[:,:count]]
    return result


def neighbor_error(features,targets,candidates,split,k):
    test=np.flatnonzero(split=='test')
    if features is None:
        chosen=candidates[:,:k]
    else:
        z,_,_=standardize(features,split=='train')
        distance=np.square(z[candidates]-z[test,None]).mean(-1)
        chosen=np.take_along_axis(candidates,np.argsort(distance,axis=1)[:,:k],axis=1)
    return np.mean([np.square(y[chosen]-y[test,None]).mean(axis=(1,2,3)) for y in targets.values()],axis=0)


def effective_rank(x):
    x=x-x.mean(0)
    eig=np.linalg.eigvalsh(x.T@x)
    eig=np.maximum(eig,0);p=eig/eig.sum();p=p[p>0]
    return float(np.exp(-(p*np.log(p)).sum()))


def auxiliary(out,names,meta,alphas):
    order=np.load(out/'order.npy');tda=np.load(out/'tda.npy')
    rows=[]
    for material,label in enumerate(('Al','Mg','Ta')):
        keep=(meta['kind']!='shooting')&(meta['material']==material)
        split=meta['split'][keep];train=split=='train';test=split=='test'
        # Structural fidelity includes all centers; no PTM class labels are used.
        targets={}
        for family,raw in [('order',order[keep,:6]),('tda',tda[keep])]:
            if family=='tda':raw,_=target_pca(raw[:,None],train,16);raw=raw[:,0]
            targets[family]=standardize(raw,train)[0]
        for name in names:
            x=np.load(out/'embeddings'/f'{name}.npy')[keep]
            row=dict(model=name,material=label,n_train=int(train.sum()),n_test=int(test.sum()),effective_rank=effective_rank(x[test]))
            for family,y in targets.items():
                if name in ('TDA','TDA_16') and family=='tda':row['tda_r2']='self-target: excluded';continue
                prediction,_,_=ridge(x,y,split,alphas)
                error=np.square(prediction-y[test]).mean()
                base=np.square(y[test]-y[train].mean(0)).mean()
                row[family+'_r2']=float(1-error/base)
            rows.append(row)
    save_csv(out/'evaluation'/'auxiliary_fidelity.csv',rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=('baselines','all'),default='all')
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg['output']
    protocol=json.loads((Path(__file__).parent/'evaluation_protocol.json').read_text())
    directory=out/('baseline_evaluation' if args.stage=='baselines' else 'evaluation');directory.mkdir(exist_ok=True)
    (directory/'readouts').mkdir(exist_ok=True)
    names=list(FIXED)+list(CONTROLS)
    if args.stage=='all':
        assert json.loads((out/'training_status.json').read_text())['state']=='complete','Finish all nine training runs before final evaluation'
        assert json.loads((out/'selection_status.json').read_text())['state']=='complete','Run the explicit whole-validation checkpoint selection first'
        names += [f'{model}_seed{seed}' for model in LEARNED for seed in cfg['training']['seeds']]
    meta=dict(np.load(out/'metadata.npz'));future=dict(np.load(out/'dynamic_targets.npz'))
    n=len(future['future_order']);keep=np.load(out/'connections.npy')[:n,1]<7
    ids=np.flatnonzero(keep);split=meta['split'][:n][keep];source=meta['source'][:n][keep]
    train,val,test=(split==s for s in ('train','val','test'))
    sets=[set(source[split==s]) for s in ('train','val','test')]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]),'Shooting source leakage'
    # Verify physical snapshot grouping, including ordinary/static copies.
    supplements=json.loads((out/'supplement_manifest.json').read_text())
    source_splits={}
    for item in supplements:
        if item['material']=='Ta':continue  # Documented single-trajectory time/ID split.
        source_splits.setdefault(item['source_group'],set()).add(item['split'])
    assert all(len(v)==1 for v in source_splits.values()),'Supplemental physical-source leakage'
    coarse=np.load(out/'order.npy')[:n][keep].astype(np.float64)
    temp=meta['temperature'][:n][keep]
    nuisance=np.column_stack((coarse,temp[:,None]==np.array([400,450,500])[None]))
    topology,pca=target_pca(future['future_tda'][keep].mean(1),train,cfg['evaluation']['future_tda_pca_dim'])
    np.savez(directory/'future_topology_pca.npz',**pca)
    raw_targets=dict(topology=topology,order=future['future_order'][keep].mean(1)[:,:,protocol['future_order_columns']],mobility=future['mobility'][keep].mean(1))
    targets={};reliability={}
    for family,raw in raw_targets.items():
        shape=raw.shape
        scaled,mean,std=standardize(raw.reshape(len(raw),-1),train)
        targets[family]=scaled.reshape(shape)
        np.savez(directory/f'{family}_scaling.npz',mean=mean,std=std)
        key={'topology':'future_tda','order':'future_order','mobility':'mobility'}[family]
        shots=future[key][keep]
        if family=='topology':shots=(shots-pca['mean'])@pca['basis']
        if family=='order':shots=shots[:,:,:,protocol['future_order_columns']]
        a=shots[:,::2].mean(1);b=shots[:,1::2].mean(1)
        a=((a.reshape(len(a),-1)-mean)/std)[test];b=((b.reshape(len(b),-1)-mean)/std)[test]
        reliability[family]=dict(half_ensemble_correlation=float(np.corrcoef(a.ravel(),b.ravel())[0,1]),
            estimated_full_mean_noise_mse=float(np.square(a-b).mean()/4),
            definition='Odd/even four-shot split: full eight-shot mean noise estimated by squared half-mean difference / 4')
    events=future['acquisition'][keep,:,-1,:].mean(1).astype(np.float64)
    candidates=neighbor_candidates(coarse,temp,split,protocol['neighbor_test']['candidate_count'])
    np.savez(directory/'split_and_targets.npz',rows=ids,split=split,source=source,temperature=temp,**targets,events=events,candidates=candidates)
    rng=np.random.default_rng(20260908)
    draws=rng.integers(0,len(np.unique(source[test])),size=(cfg['evaluation']['bootstrap_replicates'],len(np.unique(source[test]))))
    all_errors={};all_rows=[];selection={};event_rows=[]
    for name in ['CoarseBOO_temperature']+names:
        features=None if name=='CoarseBOO_temperature' else np.load(out/'embeddings'/f'{name}.npy')[:n][keep].astype(np.float64)
        x=nuisance if features is None else np.column_stack((nuisance,features))
        errors={};predictions={};selection[name]={}
        for family,y in targets.items():
            prediction=[]
            for h,horizon in enumerate(cfg['horizons_ps']):
                pred,details,readout=ridge(x,y[:,h],split,cfg['evaluation']['ridge_alpha'])
                prediction.append(pred);selection[name][f'{family}_{horizon}ps']=details
                np.savez(directory/'readouts'/f'{name}_{family}_{horizon}ps.npz',**readout)
            pred=np.stack(prediction,1);predictions[family]=pred
            errors[family]=np.square(pred-y[test]).mean(axis=(1,2))
        errors['neighbors']=neighbor_error(features,targets,candidates,split,protocol['neighbor_test']['neighbor_count'])
        for threshold_index,threshold in enumerate((.65,.70,.75)):
            pred,details,readout=ridge(x,events[:,threshold_index,None],split,cfg['evaluation']['ridge_alpha'],clip=True)
            pred=pred[:,0];truth=events[test,threshold_index]
            # Expected per-shot squared probability error = squared error of mean + shot variance.
            brier=(pred-truth)**2+truth*(1-truth)
            key=f'event_{threshold:.2f}';errors[key]=brier;predictions[key]=pred
            selection[name][key]=details
            np.savez(directory/'readouts'/f'{name}_{key}.npz',**readout)
            event_rows.append(dict(model=name,threshold=threshold,event_prevalence=float(truth.mean()),brier=float(brier.mean())))
        np.savez(directory/f'{name}_test_predictions.npz',**predictions)
        np.savez(directory/f'{name}_test_errors.npz',**errors)
        all_errors[name]=errors
        row=dict(model=name,dimensions=0 if features is None else features.shape[1],effective_rank=0 if features is None else effective_rank(features[test]))
        for family in errors:
            row[family+'_skill_pct']=bootstrap_skill(errors[family],all_errors['CoarseBOO_temperature'][family],source[test],draws)['skill_pct']
        all_rows.append(row);print(json.dumps(row),flush=True)
    aggregated=[];uncertainty={}
    for name in ['CoarseBOO_temperature']+list(FIXED)+list(CONTROLS)+(list(LEARNED) if args.stage=='all' else []):
        members=[f'{name}_seed{s}' for s in cfg['training']['seeds']] if name in LEARNED else [name]
        row=dict(model=name,seeds=len(members));uncertainty[name]={}
        for family in all_errors[members[0]]:
            averaged=np.mean([all_errors[m][family] for m in members],axis=0)
            metrics=bootstrap_skill(averaged,all_errors['CoarseBOO_temperature'][family],source[test],draws)
            scores=np.array([100*(1-all_errors[m][family].mean()/all_errors['CoarseBOO_temperature'][family].mean()) for m in members])
            metrics['seed_sd_pct']=float(scores.std(ddof=1)) if len(members)>1 else 0.
            uncertainty[name][family]=metrics
            row[family+'_skill_pct']=metrics['skill_pct'];row[family+'_seed_sd_pct']=metrics['seed_sd_pct']
        aggregated.append(row)
    save_csv(directory/'individual_results.csv',all_rows);save_csv(directory/'comparison.csv',aggregated)
    save_csv(directory/'event_sensitivity.csv',event_rows)
    write_json(directory/'uncertainty.json',uncertainty);write_json(directory/'readout_selection.json',selection)
    write_json(directory/'target_reliability.json',reliability);write_json(directory/'protocol.json',protocol)
    if args.stage=='all':auxiliary(out,[n for n in names if n not in CONTROLS],meta,cfg['evaluation']['ridge_alpha'])
    write_json(directory/'status.json',dict(state='complete',stage=args.stage,finished_at=datetime.now(timezone.utc).isoformat(),
        n_train=int(train.sum()),n_val=int(val.sum()),n_test=int(test.sum()),
        sources={s:np.unique(source[split==s]).tolist() for s in ('train','val','test')},
        pca_explained_fraction=pca['explained_fraction'],initial_noncoherent_test_event_prevalence=events[test].mean(0).tolist()))


if __name__=='__main__':main()
