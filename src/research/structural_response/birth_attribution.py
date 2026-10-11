"""Frozen-model interventions and physical explanations on the unchanged birth cohort."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.special import logit

from src.data.fixed_cohort.protocol import digest, sha, write_json
from src.experiment_runner.metric_docs import write_metric_rows
from .birth_audit import FAMILY, read, setup, measured, groups


def readout(folder, kind, values):
    if kind == 'linear':
        saved = joblib.load(folder/'model.joblib')
        return saved['model'].predict_proba((values-saved['mean'])/saved['scale'])[:,1]
    from catboost import CatBoostClassifier
    model = CatBoostClassifier();model.load_model(str(folder/'model.cbm'))
    return model.predict_proba(values, thread_count=2)[:,1]


def fit_folder(parent, name, space, lead, kind):
    return parent/'analyses'/name/'analyses/birth-prediction/analyses/fits-v1'/name/space/f'lead-{lead:g}ps'/kind/'technical'


def pair_permutation(rows, seed):
    rng=np.random.default_rng(seed);permutation=np.arange(len(rows['id']))
    for pair in np.unique(rows['pair']):
        ids=np.flatnonzero(rows['pair']==pair);permutation[ids]=rng.permutation(ids)
    return permutation


def interventions(c):
    import torch
    from .fast import FastMotion
    parent,recipe,cache,raw,rows,out=setup(c)
    frozen=parent/'technical/code'
    repo=Path(__file__).resolve().parents[3]
    dependencies=['src/research/structural_response/fast.py','src/research/structural_response/model.py',
                  'src/research/continuous_motion/model.py','src/models/encoders/spatial_mace.py']
    for filename in dependencies:
        if sha(repo/filename)!=sha(frozen/filename):raise ValueError('Changed frozen inference dependency: '+filename)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    lead=c['attribution_lead_ps'];j=recipe['birth_evaluation']['leads_ps'].index(lead)
    state=np.asarray(np.load(cache/'states.npy',mmap_mode='r')[:,j]).copy()
    train=rows['role']=='train';held=~train
    radius=np.linalg.norm(state[:,1:13,:3],axis=-1).mean(1)
    reference=float(np.median(radius[train]));factor=reference/radius
    perturb=np.random.default_rng(c['seed']+512).normal(0,.01,state[:,:,:3].shape).astype(np.float32)
    permutation=pair_permutation(rows,c['seed']+171)
    (out/'technical/neural').mkdir(parents=True,exist_ok=True)
    def encode(model, x):
        with torch.no_grad():
            return torch.cat([model.encode(torch.from_numpy(a.copy()).cuda(),create_graph=False).cpu()
                              for a in np.array_split(x,np.arange(32,len(x),32))])
    def law(model, z):
        values=[]
        with torch.no_grad():
            for block in z.split(32):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    mu,lv=model.prediction(model.path(block.cuda(),[lead],particles=8,seed=recipe['seed']+111))
                mu,lv=mu.float(),lv.float();mean=mu.mean(1)[:,0]
                variance=(lv.exp()+mu.square()).mean(1)[:,0]-mean.square()
                values.append(torch.cat((mean,variance.clamp_min(1e-8).log()),-1).cpu().numpy())
        return np.concatenate(values)
    metric_rows=[];replays=[];invariants=[]
    for method in recipe['methods']:
        for arm in recipe['response_arms']:
            name=method+'-'+arm['name'];dest=out/'technical/neural'/name
            dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():
                saved=read(dest/'complete.json');metric_rows.extend(saved['metrics']);replays.extend(saved['replays']);invariants.extend(saved['invariants']);continue
            checkpoint=parent/'analyses'/name/'technical/best.pt'
            done=read(parent/'analyses'/name/'technical/complete.json')
            if sha(checkpoint)!=done['checkpoint_sha256']:raise ValueError('Selected checkpoint changed')
            saved=torch.load(checkpoint,map_location='cuda',weights_only=False)
            model=FastMotion(recipe).cuda();model.set_arm(arm);model.load_state_dict(saved['model']);model.eval()
            z=encode(model,state);features={'structural':z[:,:128].numpy(),'structural_motion':z.numpy()}
            if 'future_law' in c['spaces']:features['future_law']=law(model,z)
            features={k:v for k,v in features.items() if k in c['spaces']}
            local_metrics=[];local_replays=[];local_invariants=[];predictions={}
            rng=np.random.default_rng(c['seed']);rotation,_=np.linalg.qr(rng.normal(size=(3,3)));rotation=rotation.astype(np.float32)
            sample=state[:64].copy();sample[:,:,:3]=sample[:,:,:3]@rotation;sample[:,:,3:]=sample[:,:,3:]@rotation
            rotated=encode(model,sample)
            for space,sl in [('structural',slice(0,128)),('motion',slice(128,None))]:
                error=float(torch.linalg.vector_norm(rotated[:,sl]-z[:64,sl])/torch.linalg.vector_norm(z[:64,sl]).clamp_min(1e-8))
                local_invariants.append(dict(model=name,operation='rotate_Q_and_V',space=space,relative_code_error=error))
            for space,x in features.items():
                for kind in ('linear','catboost'):
                    folder=fit_folder(parent,name,space,lead,kind);receipt=read(folder/'complete.json')
                    if sha(folder/'predictions.npz')!=receipt['predictions_sha256']:raise ValueError('Saved predictions changed')
                    with np.load(folder/'predictions.npz') as a:
                        if not np.array_equal(a['id'],rows['id']):raise ValueError('Prediction row mismatch')
                        original=a['probability'].copy()
                    probability=readout(folder,kind,x);error=np.abs(probability-original)
                    replay=dict(model=name,space=space,readout=kind,max_error=float(error.max()),mean_error=float(error.mean()),q99_error=float(np.quantile(error,.99)))
                    local_replays.append(replay)
                    tolerance=c['replay_tolerance'][space]
                    if error.mean()>tolerance['mean'] or np.quantile(error,.99)>tolerance['q99']:
                        raise ValueError('Frozen prediction replay differs: '+str(replay))
                    predictions[space+'__original__'+kind]=original
                    local_metrics.extend(measured(rows,original,model=name,variant='original',space=space,readout=kind,lead_ps=lead))
            for variant in ('zero_motion','reverse_motion','shuffle_motion_within_pair','shuffle_structure_within_pair','normalize_first_shell','noise_001A'):
                if variant=='normalize_first_shell':
                    changed=state.copy();changed[:,:,:3]*=factor[:,None,None];zz=encode(model,changed)
                elif variant=='noise_001A':
                    changed=state.copy();changed[:,:,:3]+=perturb;zz=encode(model,changed)
                else:
                    zz=z.clone()
                    if variant=='zero_motion':zz[:,128:]=0
                    elif variant=='reverse_motion':zz[:,128:]*=-1
                    elif variant=='shuffle_motion_within_pair':zz[:,128:]=z[permutation,128:]
                    else:zz[:,:128]=z[permutation,:128]
                inputs={'structural':zz[:,:128].numpy(),'structural_motion':zz.numpy()}
                if 'future_law' in c['spaces']:inputs['future_law']=law(model,zz)
                inputs={k:v for k,v in inputs.items() if k in c['spaces']}
                for space,x in inputs.items():
                    for kind in ('linear','catboost'):
                        probability=readout(fit_folder(parent,name,space,lead,kind),kind,x)
                        predictions[space+'__'+variant+'__'+kind]=probability
                        local_metrics.extend(measured(rows,probability,model=name,variant=variant,space=space,readout=kind,lead_ps=lead))
                print('intervention',name,variant,flush=True)
            # A matched permutation null uses frozen probabilities, not fitted labels.
            from src.research.birth_prediction.fit import scores
            for kind in ('linear','catboost'):
                probability=predictions['structural_motion__original__'+kind]
                null=[]
                for repeat in range(25):
                    perm=pair_permutation(rows,c['seed']+900+repeat)
                    null.append(scores(rows['label'][held],probability[perm][held],rows['weight'][held])['ap'])
                local_invariants.append(dict(model=name,operation='matched_probability_permutation',space=kind,
                    mean_ap=float(np.mean(null)),min_ap=float(np.min(null)),max_ap=float(np.max(null)),draws=25))
            np.savez_compressed(dest/'predictions.npz',**predictions)
            write_json(dest/'complete.json',dict(checkpoint_sha256=sha(checkpoint),metrics=local_metrics,
                replays=local_replays,invariants=local_invariants,normalization_reference_A=reference))
            metric_rows.extend(local_metrics);replays.extend(local_replays);invariants.extend(local_invariants)
            write_metric_rows(metric_rows,out,family=FAMILY,name='frozen-interventions')
            write_json(out/'technical/neural-replays.json',replays);write_json(out/'technical/neural-invariants.json',invariants)
            del model;torch.cuda.empty_cache()
    write_json(out/'technical/interventions-complete.json',dict(models=8,lead_ps=lead,rows=len(metric_rows)))


def explain(c):
    from catboost import CatBoostRegressor, Pool
    parent,recipe,_,_,rows,out=setup(c)
    lead=c['attribution_lead_ps'];j=recipe['birth_evaluation']['leads_ps'].index(lead)
    with np.load(out/'technical/physical.npz') as a:x=a['values'][:,j]
    columns=read(out/'technical/columns.json');family=groups(columns)
    plan=read(parent/'analyses/masked_teacher-derivative_joint/analyses/birth-prediction/technical/readout-plan.json')
    masks={k:np.isin(rows['ancestor'],v) for k,v in plan['partition'].items()}
    weights=rows['weight'];held=masks['test'];tr=masks['inner_fit'];va=masks['inner_selection'];final=masks['final_train']
    fidelity=[];importance=[]
    for method in recipe['methods']:
        for arm in recipe['response_arms']:
            name=method+'-'+arm['name'];folder=fit_folder(parent,name,'structural_motion',lead,'catboost')
            with np.load(folder/'predictions.npz') as a:
                if not np.array_equal(a['id'],rows['id']):raise ValueError('Surrogate target rows changed')
                probability=a['probability'].copy()
            target=logit(np.clip(probability,1e-5,1-1e-5))
            dest=out/'technical/surrogates'/name;dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():
                old=read(dest/'complete.json');fidelity.append(old['fidelity']);importance.extend(old['importance']);continue
            def pool(mask):return Pool(x[mask],target[mask],weight=weights[mask]/weights[mask].mean(),feature_names=columns)
            params=dict(iterations=600,depth=4,learning_rate=.04,l2_leaf_reg=10,loss_function='RMSE',
                        random_seed=c['seed'],thread_count=2,allow_writing_files=False)
            reg=CatBoostRegressor(**params)
            reg.fit(pool(tr),eval_set=pool(va),early_stopping_rounds=75,use_best_model=True,verbose=False)
            selected=reg.get_best_iteration()+1
            reg=CatBoostRegressor(**dict(params,iterations=selected));reg.fit(pool(final),verbose=False)
            predicted=reg.predict(x[held]);w=weights[held]/weights[held].sum();truth=target[held]
            variance=float(w@(truth-float(w@truth))**2);mse=float(w@(predicted-truth)**2)
            record=dict(model=name,lead_ps=lead,iterations=selected,logit_rmse=mse**.5,logit_r2=1-mse/variance,
                        probability_correlation=float(np.corrcoef(1/(1+np.exp(-predicted)),probability[held])[0,1]))
            shap=reg.get_feature_importance(Pool(x[held],feature_names=columns),type='ShapValues',thread_count=2)[:,:-1]
            absolute=w@np.abs(shap);local=[]
            for kind,ids in family.items():
                if kind in ('rich','radial','q6','qbar6','order_small'):continue
                local.append(dict(model=name,group=kind,feature='all',mean_absolute_logit_shap=float(absolute[ids].sum())))
            for i in np.argsort(-absolute)[:20]:
                local.append(dict(model=name,group=columns[i].split('/')[0],feature=columns[i],mean_absolute_logit_shap=float(absolute[i])))
            reg.save_model(str(dest/'surrogate.cbm'));np.savez_compressed(dest/'heldout.npz',target=truth,predicted=predicted)
            write_json(dest/'complete.json',dict(fidelity=record,importance=local,selection='inner-training-ancestor RMSE'))
            fidelity.append(record);importance.extend(local)
            write_metric_rows(fidelity,out,family=FAMILY,name='surrogate-fidelity')
            write_metric_rows(importance,out,family=FAMILY,name='physical-attribution')
            print('surrogate',record,flush=True)
    write_json(out/'technical/explanations-complete.json',dict(models=len(fidelity)))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=('interventions','explain'))
    p.add_argument('--config',required=True);a=p.parse_args();c=read(a.config)
    (interventions if a.stage=='interventions' else explain)(c)


if __name__=='__main__':main()
