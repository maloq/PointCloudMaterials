"""Full-timeline descriptor hazard and physical-ridge diagnostics on fixed sources."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from src.data.predictive_memory.prepare import file_hash,write_json
from src.data.predictive_memory.targets import BLOCKS
from src.project_runtime.paths import load_json,resolve_path
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.research.forecast_crystallization.local_metrics import first_sustained_onset,risk_windows
from .metrics import hazard_loss,cumulative_risk,source_weights,weighted_scores,threshold_at_fpr,stratified_bootstrap


def history_features(packet,anchors,duration,repeat=False):
    if duration==0:return packet[:,anchors]
    steps=round(duration/.75);offsets=np.arange(-steps,1)
    observed=packet[:,anchors[:,None]+offsets]
    if repeat:observed=np.broadcast_to(packet[:,anchors,None,:],observed.shape)
    times=offsets*.75;centered=times-times.mean()
    slope=np.einsum('catd,t->cad',observed,centered)/np.sum(centered**2)
    return np.concatenate((observed[:,:,-1],observed[:,:,0],observed.mean(2),observed.std(2),
        slope,observed.min(2),observed.max(2)),axis=-1).astype(np.float32)


def build_arrays(release,cache,observation,*,native=False,at_risk=True):
    anchors=np.array(release['native_anchors' if native else 'dense_anchors'])
    lags=np.array(release['horizon_frames']);parts={k:[] for k in ['x','y','source','split','temperature','center','anchor','future','present']}
    for source in release['sources']:
        data=np.load(cache/source['shard']);packet=data['packet'];nc=len(packet)
        crystal=np.isin(data['labels'],[1,2,3]);onset=first_sustained_onset(crystal,3)
        risk=risk_windows(crystal,onset,anchors,3)
        delay=onset[:,None]-anchors
        event=np.where((delay>0)&(delay<=lags[-1]),np.searchsorted(lags,delay),len(lags))
        shape=(nc,len(anchors));condition=np.zeros((*shape,7),np.float32)
        condition[:,:,[400,450,500,510,520].index(int(source['temperature_K']))]=1
        condition[:,:,5]=anchors[None]*.75/600;condition[:,:,6]=condition[:,:,5]**2
        if observation=='condition':features=condition
        else:
            duration={'packet_H0':0,'packet_H3':3,'packet_H12':12,'packet_H48':48,'packet_repeat12':12,
                      'packet_plus_center_H0':0,'packet_plus_shell25_H0':0}[observation]
            components=[history_features(packet,anchors,duration,observation=='packet_repeat12'),condition]
            if observation=='packet_plus_center_H0':components.append(data['order'][:,anchors])
            if observation=='packet_plus_shell25_H0':components.append(data['shell'][:,anchors])
            features=np.concatenate(components,-1)
        keep=risk if at_risk else np.ones(shape,bool)
        arrays=dict(x=features,y=event,source=np.full(shape,source['id']),
            split=np.full(shape,source.get('validation_role',source['split'])),
            temperature=np.full(shape,source['temperature_K']),center=np.broadcast_to(data['atom_ids'][:,None],shape),
            anchor=np.broadcast_to(anchors,shape),future=packet[:,anchors[:,None]+lags],present=packet[:,anchors])
        for key,value in arrays.items():parts[key].append(value[keep])
    return {key:np.concatenate(values) for key,values in parts.items()}


def check_deadline(config,reserve_seconds=120):
    if time.time()+reserve_seconds>=datetime.fromisoformat(config['training_deadline_utc']).timestamp():
        raise TimeoutError('Training cutoff reached; complete artifacts already preserved')


def hazard_fit(arrays,model_kind,plan,device,config,root):
    seed=config['seed'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    masks={split:np.flatnonzero(arrays['split']==split) for split in ['train','selection','calibration','test']}
    if any(len(i)==0 for i in masks.values()):raise ValueError(f'Empty at-risk split: { {k:len(v) for k,v in masks.items()} }')
    x=arrays['x'];train=masks['train'];center=x[train].mean(0);scale=np.maximum(x[train].std(0),1e-4)
    features=torch.from_numpy((x-center)/scale).to(device);y=torch.from_numpy(arrays['y']).long().to(device)
    train_index=torch.tensor(train,device=device);val_index=torch.tensor(masks['selection'],device=device)
    train_weights=torch.tensor(source_weights(arrays['source'][train])*len(train),device=device,dtype=torch.float32)
    val_weights=torch.tensor(source_weights(arrays['source'][masks['selection']]),device=device,dtype=torch.float32)
    k=len(plan['sampling']['horizons_ps']);best=float('inf');selected=None;best_state=None;trace=[]
    candidates=plan['descriptors']['linear_hazard']['regularization_grid'] if model_kind=='linear' else [None]
    for penalty in candidates:
        check_deadline(config)
        torch.manual_seed(seed)
        model=(nn.Linear(features.shape[1],k) if model_kind=='linear' else
            nn.Sequential(nn.Linear(features.shape[1],256),nn.SiLU(),nn.Linear(256,128),nn.SiLU(),nn.Linear(128,k))).to(device)
        # Initialize hazards at train frequencies to avoid a large artificial early risk.
        last=model if model_kind=='linear' else model[-1]
        with torch.no_grad():
            freq=[]
            yt=arrays['y'][train];w=source_weights(arrays['source'][train])
            for b in range(k):freq.append(np.clip(w[yt==b].sum()/max(w[yt>=b].sum(),1e-12),1e-4,1-1e-4))
            last.bias.copy_(torch.tensor(np.log(np.array(freq)/(1-np.array(freq))),device=device,dtype=torch.float32))
            if model_kind=='linear':last.weight.zero_()
        if model_kind=='linear':
            optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=150,line_search_fn='strong_wolfe',tolerance_grad=1e-6)
            def closure():
                check_deadline(config);optimizer.zero_grad();loss=features.new_zeros(())
                for indices in train_index.split(8192):
                    # Index positions within train_index are contiguous in the source dataset.
                    local=torch.searchsorted(train_index,indices)
                    current=(hazard_loss(model(features[indices]),y[indices])*train_weights[local]).sum()/len(train)
                    current.backward();loss=loss+current.detach()
                regularizer=model.weight.square().sum()/(2*penalty*len(train))
                regularizer.backward()
                return loss+regularizer.detach()
            optimizer.step(closure)
            candidates_states=[(150,model.state_dict())]
        else:
            settings=plan['descriptors']['mlp_hazard'];optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
            local_best=float('inf');patience=0;candidates_states=[]
            for step in range(1,settings['maximum_updates']+1):
                local=torch.randint(len(train),(settings['batch_size'],),device=device);index=train_index[local]
                optimizer.zero_grad();loss=(hazard_loss(model(features[index]),y[index])*train_weights[local]).mean()
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                if step%settings['validation_every']==0:
                    check_deadline(config)
                    with torch.no_grad():v=float((hazard_loss(model(features[val_index]),y[val_index])*val_weights).sum())
                    trace.append(dict(step=step,selection_nll=v))
                    if v<local_best:
                        local_best=v;patience=0;candidates_states=[(step,{k:v.detach().cpu().clone() for k,v in model.state_dict().items()})]
                    else:patience+=1
                    if patience>=settings['early_stop_validation_patience']:break
        step,state=candidates_states[-1];model.load_state_dict(state)
        with torch.no_grad():value=float((hazard_loss(model(features[val_index]),y[val_index])*val_weights).sum())
        if value<best:
            best=value;selected=dict(penalty=penalty,step=step,selection_nll=value)
            best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    logits=[]
    with torch.no_grad():
        for block in features.split(8192):logits.append(model(block).cpu())
    logits=torch.cat(logits);prob=cumulative_risk(logits).numpy()
    root.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model=best_state,model_kind=model_kind,center=center,scale=scale,selection=selected,seed=seed),root/'model.pt')
    np.savez(root/'predictions.npz',probability=prob,logits=logits.numpy(),event_bin=arrays['y'],
        **{k:arrays[k] for k in ['source','split','temperature','center','anchor']})
    write_json(root/'training.json',dict(selection=selected,trace=trace,seed=seed))
    return prob,logits.numpy(),masks


def score_hazard(arrays,probability,logits,masks,plan):
    rows=[];source_rows=[];horizons=plan['sampling']['horizons_ps']
    for h,horizon in enumerate(horizons):
        actual=arrays['y']<=h;c=masks['calibration'];test=masks['test']
        threshold=threshold_at_fpr(actual[c],probability[c,h],arrays['source'][c],plan['assay']['maximum_window_false_positive_rate'])
        for split in ['selection','calibration','test']:
            index=masks[split];score=weighted_scores(actual[index],probability[index,h],arrays['source'][index],threshold)
            rows.append(dict(split=split,horizon_ps=horizon,**score))
        for source in np.unique(arrays['source'][test]):
            index=test[arrays['source'][test]==source]
            score=weighted_scores(actual[index],probability[index,h],arrays['source'][index],threshold)
            source_rows.append(dict(source_id=int(source),temperature_K=float(arrays['temperature'][index[0]]),horizon_ps=horizon,**score))
    for row in rows:
        if row['split']=='test':
            sources=[r for r in source_rows if r['horizon_ps']==row['horizon_ps']]
            values=np.array([[r['log_loss'],r['brier']] for r in sources])
            bounds=stratified_bootstrap(values,[r['temperature_K'] for r in sources])
            row.update(log_loss_ci95=bounds[:,0].tolist(),brier_ci95=bounds[:,1].tolist())
    nll=hazard_loss(torch.from_numpy(logits),torch.from_numpy(arrays['y']).long()).numpy()
    joint={split:float(source_weights(arrays['source'][index])@nll[index]) for split,index in masks.items()}
    return dict(population=rows,per_source=source_rows,joint_event_nll=joint)


def ridge_fit(arrays,release,plan,device,root):
    train=np.flatnonzero(arrays['split']=='train');val=np.flatnonzero(arrays['split']=='selection');test=np.flatnonzero(arrays['split']=='test')
    mean=np.array(release['normalizer']['mean']);scale=np.array(release['normalizer']['scale'])
    x=arrays['x'].astype(np.float64);xc=x[train].mean(0);xs=np.maximum(x[train].std(0),1e-4);x=(x-xc)/xs
    targets=(arrays['future'].astype(np.float64)-mean)/scale;y=targets.reshape(len(x),-1);ym=y[train].mean(0)
    xt=torch.tensor(x,device=device);yt=torch.tensor(y-ym,device=device);ti=torch.tensor(train,device=device)
    matrix=xt[ti].T@xt[ti];rhs=xt[ti].T@yt[ti]
    best=float('inf');selected=None;coefficient=None
    weights=torch.tensor(source_weights(arrays['source'][val]),device=device)
    for alpha in plan['descriptors']['physical_ridge_alpha']:
        beta=torch.linalg.solve(matrix+alpha*torch.eye(matrix.shape[0],device=device,dtype=torch.float64),rhs)
        loss=(xt[val]@beta-yt[val]).square().mean(1)
        value=float(weights@loss)
        if value<best:best=value;selected=alpha;coefficient=beta
    prediction=(xt@coefficient).cpu().numpy()+ym
    prediction=prediction.reshape(targets.shape);error=(prediction-targets)**2
    persistence=((arrays['present'][:,None]-mean)/scale-targets)**2
    rows=[];per_source=[]
    for horizon,lag in enumerate(plan['sampling']['horizons_ps']):
        for method,errors in [('ridge',error),('persistence',persistence)]:
            for block,(lo,hi) in dict(all=(0,128),**BLOCKS).items():
                per=errors[:,horizon,lo:hi].mean(1);w=source_weights(arrays['source'][test])
                rows.append(dict(method=method,horizon_ps=lag,block=block,mse=float(w@per[test])))
                if block=='all':
                    for sid in np.unique(arrays['source'][test]):
                        idx=test[arrays['source'][test]==sid]
                        per_source.append(dict(method=method,horizon_ps=lag,source_id=int(sid),mse=float(per[idx].mean())))
    root.mkdir(parents=True,exist_ok=True)
    np.savez(root/'model.npz',coefficient=coefficient.cpu().numpy(),feature_center=xc,feature_scale=xs,target_center=ym,alpha=selected)
    np.savez(root/'test_predictions.npz',prediction=prediction[test].astype(np.float32),target=targets[test].astype(np.float32),
        **{k:arrays[k][test] for k in ['source','center','anchor']})
    return dict(selected_alpha=selected,selection_mse=best,test=rows,per_source=per_source)


def export_report(output,results):
    import csv
    snapshot_metric_docs(output,'local_predictability')
    rows=[]
    for name,result in results.items():
        for row in result.get('population',result.get('test',[])):
            rows.append(dict(model=name,**row))
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (output/'tables/descriptor_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    write_json(output/'technical/descriptor_results.json',results)
    lines=['# Local predictability: incremental descriptor results','',
        'Single training seed 20260919. Source intervals condition on this seed.','',
        '| Model | Horizon (ps) | Test log loss | Test Brier | Test AP |','| --- | --- | --- | --- | --- |']
    for name,result in results.items():
        for row in result.get('population',[]):
            if row['split']=='test' and row['horizon_ps'] in (9,48):
                ap='undefined' if row['average_precision'] is None else f"{row['average_precision']:.4f}"
                lines.append(f"| {name} | {row['horizon_ps']} | {row['log_loss']:.4f} | {row['brier']:.4f} | {ap} |")
    lines.extend(['','Native models and dense alarm/timing assays are separate stages; unfinished stages are not results.'])
    (output/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def run(config):
    torch.set_num_threads(4);device=torch.device(config['device'])
    plan=json.loads(resolve_path(config['plan']).read_text());cache=resolve_path(config['cache'])
    output=resolve_path(config['output']);release=json.loads((cache/'release.json').read_text())
    if release['state']!='complete':raise RuntimeError('Need complete source release')
    for source in release['sources']:
        if file_hash(cache/source['shard'])!=source['shard_sha256']:raise RuntimeError(f"Changed shard {source['id']}")
    results={};root=output/'technical/descriptors';root.mkdir(parents=True,exist_ok=True)
    for observation in plan['descriptors']['observations']:
        check_deadline(config);arrays=build_arrays(release,cache,observation)
        for model_kind in ['linear','mlp']:
            name=f'{model_kind}-{observation}';destination=root/name
            print('Starting',name,'rows',len(arrays['x']),flush=True)
            if (destination/'result.json').exists():result=json.loads((destination/'result.json').read_text())
            else:
                probability,logits,masks=hazard_fit(arrays,model_kind,plan,device,config,destination)
                result=score_hazard(arrays,probability,logits,masks,plan)
                write_json(destination/'result.json',result)
            results[name]=result;export_report(output,results)
            write_json(output/'technical/baseline_status.json',dict(state='running',completed=list(results),current=name))
        del arrays;torch.cuda.empty_cache()
    for observation in plan['descriptors']['physical_ridge_inputs']:
        check_deadline(config);name=f'ridge-{observation}';destination=root/name
        print('Starting',name,flush=True)
        if (destination/'result.json').exists():result=json.loads((destination/'result.json').read_text())
        else:
            arrays=build_arrays(release,cache,observation,native=True,at_risk=False)
            result=ridge_fit(arrays,release,plan,device,destination);write_json(destination/'result.json',result)
            del arrays;torch.cuda.empty_cache()
        results[name]=result;export_report(output,results)
    write_json(output/'technical/baseline_status.json',dict(state='complete',completed=list(results)))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path)
    args=parser.parse_args();run(load_json(args.config))


if __name__=='__main__':main()
