"""Common covariance-conditioned linear probes: audit raw-channel/PCA basis bias."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
from experiments.liquid_sro_benchmark_20260905.evaluate import FIXED,LEARNED,CONTROLS,standardize,bootstrap_skill,save_csv
from experiments.smooth_temporal_encoder_20260905.prepare import write_json


def fit(x,y,split,alphas):
    train,val,test=(split==s for s in ('train','val','test'))
    xm=x[train].mean(0);ym=y[train].mean(0);x=x-xm
    gram=x[train].T@x[train];rhs=x[train].T@(y[train]-ym)
    best=(float('inf'),None,None)
    for alpha in alphas:
        weight=np.linalg.solve(gram+alpha*np.eye(gram.shape[0]),rhs)
        error=float(np.square(x[val]@weight+ym-y[val]).mean())
        if error<best[0]:best=(error,weight,alpha)
    error,weight,alpha=best
    return x[test]@weight+ym,dict(validation_mse=error,alpha=alpha),dict(input_mean=xm,weight=weight,intercept=ym)


def main():
    cfg=json.loads((Path(__file__).parent/'config.json').read_text());out=ROOT/cfg['output']
    directory=out/'conditioned_probe';directory.mkdir(exist_ok=True)
    saved=dict(np.load(out/'evaluation/split_and_targets.npz'));ids=saved['rows'];split=saved['split']
    train,test=split=='train',split=='test';source=saved['source'][test]
    coarse=np.load(out/'order.npy')[ids];temp=saved['temperature']
    nuisance=standardize(np.column_stack((coarse,temp[:,None]==np.array([400,450,500])[None])),train)[0]
    names=list(FIXED)+list(CONTROLS)+[f'{m}_seed{s}' for m in LEARNED for s in cfg['training']['seeds']]
    rng=np.random.default_rng(20260908);groups=np.unique(source)
    draws=rng.integers(0,len(groups),size=(1000,len(groups)))
    errors={};details={};rows=[]
    floors=(.01,.0001,.000001)
    for name in names:
        raw=np.load(out/'embeddings'/f'{name}.npy')[ids]
        z,mean,std=standardize(raw,train)
        eigen,basis=np.linalg.eigh(z[train].T@z[train]/train.sum())
        variants={floor:z@(basis/np.sqrt(np.maximum(eigen,eigen[-1]*floor))[None]) for floor in floors}
        np.savez(directory/f'{name}_input_transform.npz',mean=mean,std=std,eigenvalues=eigen,basis=basis)
        errors[name]={};details[name]={}
        row=dict(model=name)
        for family in ('topology','order','mobility'):
            target=saved[family];predictions=[]
            for h in range(3):
                candidates=[]
                for floor,features in variants.items():
                    prediction,choice,readout=fit(np.column_stack((nuisance,features)),target[:,h],split,cfg['evaluation']['ridge_alpha'])
                    candidates.append((choice['validation_mse'],prediction,dict(floor=floor,**choice),readout))
                _,prediction,choice,readout=min(candidates,key=lambda item:item[0])
                predictions.append(prediction);details[name][f'{family}_{h}']=choice
                np.savez(directory/f'{name}_{family}_{h}_readout.npz',**readout,floor=choice['floor'])
            prediction=np.stack(predictions,1)
            np.save(directory/f'{name}_{family}_test_predictions.npy',prediction)
            error=np.square(prediction-target[test]).mean(axis=(1,2));errors[name][family]=error
            baseline=np.load(out/'evaluation/CoarseBOO_temperature_test_errors.npz')[family]
            row[family+'_skill_pct']=float(100*(1-error.mean()/baseline.mean()))
        rows.append(row);np.savez(directory/f'{name}_test_errors.npz',**errors[name]);print(json.dumps(row),flush=True)
    aggregated=[];uncertainty={}
    for name in list(FIXED)+list(CONTROLS)+list(LEARNED):
        members=[f'{name}_seed{s}' for s in cfg['training']['seeds']] if name in LEARNED else [name]
        row=dict(model=name,seeds=len(members));uncertainty[name]={}
        for family in ('topology','order','mobility'):
            baseline=np.load(out/'evaluation/CoarseBOO_temperature_test_errors.npz')[family]
            average=np.mean([errors[n][family] for n in members],axis=0)
            metric=bootstrap_skill(average,baseline,source,draws)
            values=[100*(1-errors[n][family].mean()/baseline.mean()) for n in members]
            metric['seed_sd_pct']=float(np.std(values,ddof=1)) if len(members)>1 else 0.
            uncertainty[name][family]=metric
            row[family+'_skill_pct']=metric['skill_pct'];row[family+'_seed_sd_pct']=metric['seed_sd_pct']
        aggregated.append(row)
    save_csv(directory/'individual_results.csv',rows);save_csv(directory/'comparison.csv',aggregated)
    write_json(directory/'uncertainty.json',uncertainty);write_json(directory/'selection.json',details)
    write_json(directory/'protocol.json',dict(state='complete',purpose='Exploratory readout-conditioning audit after the original comparison, which compared PCA-derived descriptors with correlated raw learned channels. Original comparison remains intact.',
        transformation='Training-only channel standardization followed by PCA and inverse sqrt covariance; eigenvalue floor is lambda_max times relative floor. Do not re-standardize the resulting PC scores.',
        floors=floors,selection='Each family/horizon selects floor and ridge alpha using validation targets only. Identical choices for all representations. Encoder checkpoints and test set unchanged.',score='Same test MSE skill relative to the original coarse-order + temperature baseline'))


if __name__=='__main__':main()
