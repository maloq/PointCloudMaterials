"""Score measured stability controls and training-only TDA readouts."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch

from src.data_utils.topology_targets import BLOCKS, fit_targets
from src.experiment_runner.metric_docs import snapshot_metric_docs
from src.experiment_runner.registry import sha256, write_json
from src.project_runtime.paths import load_json


def balanced_errors(prediction, target, scales):
    return np.stack([np.mean((prediction[...,block]-target[...,block])**2,axis=-1)/scales[d]**2
        for d,block in enumerate(BLOCKS)],-1).mean(-1)


def centered(values, groups):
    result=np.empty_like(values)
    for group in np.unique(groups):
        mask=groups==group
        result[mask]=values[mask]-values[mask].mean(0)
    return result


def source_gain(reference, candidate, sources, rng, draws):
    a=np.array([reference[sources==s].mean() for s in np.unique(sources)])
    b=np.array([candidate[sources==s].mean() for s in np.unique(sources)])
    draw=rng.integers(len(a),size=(draws,len(a)))
    gain=1-b[draw].mean(1)/a[draw].mean(1)
    return dict(skill=float(1-b.mean()/a.mean()),low=float(np.quantile(gain,.025)),
        high=float(np.quantile(gain,.975)),sources=len(a))


def correlation(x,y):
    x,y=np.ravel(x),np.ravel(y)
    valid=np.isfinite(x)&np.isfinite(y)
    x,y=x[valid],y[valid]
    if len(x)<2 or np.std(x)==0 or np.std(y)==0:
        return None
    return float(spearmanr(np.ravel(x),np.ravel(y)).statistic)


def cosine(x,y):
    denominator=np.sqrt(np.sum(x*x)*np.sum(y*y))
    return None if denominator==0 else float(np.sum(x*y)/denominator)


def write_table(root,name,rows):
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with (root/'tables'/f'{name}.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)


def readouts(data,cfg,root):
    rng=np.random.default_rng(cfg['seed']); split=data['split']
    train=split=='train'; val=split=='val'; test=split=='test'
    tables=[]; per_source=[]; comparisons=[]; fits={}; prediction_arrays={}; details={}
    for target_name in ('hot','relaxed'):
        target=data[target_name].astype(np.float64)
        scaling=fit_targets(target[train],32,.05)
        scales=scaling['block_scale']; pixel=scaling['pixel_scale']; mean=target[train].mean(0)
        normalized=(target-mean)/pixel
        baseline=balanced_errors(np.broadcast_to(mean,target.shape),target,scales)
        predictions={'training_mean':np.broadcast_to(mean,target.shape)}
        # A temperature mean can be learned without observing a local patch.
        temperature=np.empty_like(target)
        for temp in np.unique(data['temperature']):
            temperature[data['temperature']==temp]=target[train & (data['temperature']==temp)].mean(0)
        predictions['temperature_mean']=temperature
        selections={}
        for feature_name,features in [('mace',data['z']),('geometry',data['geometry'])]:
            scaler=StandardScaler().fit(features[train].astype(np.float64)); x=scaler.transform(features.astype(np.float64))
            trials=[]
            for alpha in cfg['ridge_alphas']:
                model=Ridge(alpha=alpha).fit(x[train],normalized[train])
                errors=balanced_errors(model.predict(x[val])*pixel+mean,target[val],scales)
                trials.append((float(np.mean([errors[data['source'][val]==s].mean()
                    for s in np.unique(data['source'][val])])),alpha,model))
            _,alpha,model=min(trials,key=lambda item:item[0]);selections[feature_name]=alpha
            predictions[feature_name]=model.predict(x)*pixel+mean
            if feature_name=='mace':
                fits[target_name]=(scaler,model,mean,pixel,scales)
                # Preserve context means while destroying the atom-to-target correspondence.
                shuffled=normalized.copy()
                for context in np.unique(data['context'][train]):
                    rows=np.flatnonzero(train & (data['context']==context))
                    shuffled[rows]=normalized[rng.permutation(rows)]
                null=Ridge(alpha=alpha).fit(x[train],shuffled[train])
                predictions['within_context_shuffle']=null.predict(x)*pixel+mean
                global_null=Ridge(alpha=alpha).fit(x[train],normalized[rng.permutation(np.flatnonzero(train))])
                predictions['global_shuffle']=global_null.predict(x)*pixel+mean
                for count in (16,32):
                    selected=np.concatenate([np.flatnonzero(train & (data['context']==c))[:count]
                        for c in np.unique(data['context'][train])])
                    small=Ridge(alpha=alpha).fit(x[selected],normalized[selected])
                    predictions[f'mace_{count}_per_context']=small.predict(x)*pixel+mean
        for name,prediction in predictions.items():
            errors=balanced_errors(prediction,target,scales)
            for part,mask in [('train',train),('val',val),('test',test)]:
                ycenter=centered(target[mask],data['context'][mask])
                pcenter=centered(prediction[mask],data['context'][mask])
                centerr=balanced_errors(pcenter,ycenter,scales)
                centvar=balanced_errors(np.zeros_like(ycenter),ycenter,scales)
                gain=source_gain(baseline[mask],errors[mask],data['source'][mask],rng,cfg['bootstrap_draws'])
                row=dict(target=target_name,model=name,split=part,n=int(mask.sum()),
                    balanced_mse=float(errors[mask].mean()),**gain,
                    local_centered_skill=float(1-centerr.mean()/centvar.mean()),
                    local_uncentered_skill=float(1-errors[mask].mean()/centvar.mean()),
                    alpha=selections.get(name))
                for d,b in enumerate(BLOCKS):
                    row[f'H{d}_local_centered_r2']=float(1-np.sum((pcenter[:,b]-ycenter[:,b])**2)/np.sum(ycenter[:,b]**2))
                tables.append(row)
            for s in np.unique(data['source'][test]):
                mask=test & (data['source']==s)
                per_source.append(dict(target=target_name,model=name,source=int(s),
                    temperature_K=float(data['temperature'][mask][0]),balanced_mse=float(errors[mask].mean()),
                    skill=float(1-errors[mask].mean()/baseline[mask].mean())))
            prediction_arrays[f'{target_name}_{name}']=prediction.astype(np.float32)
        prediction_arrays[f'{target_name}_block_scale']=scales
        mace_errors=balanced_errors(predictions['mace'][test],target[test],scales)
        for reference in ('geometry','within_context_shuffle','global_shuffle','temperature_mean'):
            errors=balanced_errors(predictions[reference][test],target[test],scales)
            comparisons.append(dict(target=target_name,reference=reference,
                **source_gain(errors,mace_errors,data['source'][test],rng,cfg['bootstrap_draws'])))
        details[target_name]=dict(selected_alphas=selections)
    np.savez_compressed(root/'technical/readouts.npz',**prediction_arrays)
    write_table(root,'tda-generalization',tables);write_table(root,'tda-by-source',per_source)
    write_table(root,'tda-comparisons',comparisons)
    return fits,tables,details


def predict(z,fit):
    scaler,model,mean,pixel,_=fit
    shape=z.shape[:-1]
    return (model.predict(scaler.transform(z.reshape(-1,256).astype(np.float64)))*pixel+mean).reshape(*shape,144)


def diagnostics(cfg,root,forecast_checkpoint):
    tech=root/'technical'
    data={name:dict(np.load(tech/f'{name}.npz')) for name in
        ('probes','temporal','controls','boundaries','siblings','global_precision')}
    fits,probe_table,details=readouts(data['probes'],cfg,root)
    payload=torch.load(forecast_checkpoint,map_location='cpu',weights_only=False)
    manifest_path=Path(cfg['forecast_cache'])/'manifest.json'
    forecast_manifest=json.loads(manifest_path.read_text())
    if payload['cache_manifest_sha256']!=sha256(manifest_path):
        raise ValueError('Forecast normalizer belongs to a different cache manifest')
    if forecast_manifest['protocol']['checkpoint_sha256']!=sha256(Path(cfg['checkpoint'])):
        raise ValueError('The tested encoder differs from the forecast cache producer')
    mean=payload['mean'].numpy();scale=payload['scale'].numpy()
    if mean.shape!=(256,) or np.any(scale<=0): raise ValueError('Unexpected forecast normalizer')
    np.savez(tech/'forecast_scaling.npz',mean=mean,scale=scale)
    temporal=data['temporal'];z=temporal['z'].astype(np.float64)
    change=np.diff(z,axis=1)/scale
    step_mse=float(np.mean(change**2))
    hot_scale=fits['hot'][-1]
    hot_step=float(balanced_errors(temporal['hot'][:,1:],temporal['hot'][:,:-1],hot_scale).mean())
    controls=data['controls'];rows=[]
    for name,value in controls.items():
        if name in ('baseline','clouds'):continue
        dz=(value-controls['baseline'])/scale
        rows.append(dict(control=name,n=len(dz),standardized_mse=float(np.mean(dz**2)),
            mse_over_0p75ps=float(np.mean(dz**2)/step_mse),
            rms_over_0p75ps=float(np.sqrt(np.mean(dz**2)/step_mse)),
            max_abs_channel=float(np.abs(value-controls['baseline']).max()),
            p95_sample_standardized_rms=float(np.quantile(np.sqrt(np.mean(dz**2,axis=1)),.95))))
    write_table(root,'identical-input-controls',rows)
    b=data['boundaries'];dz=(b['next_z']-b['z'])/scale
    motion=(b['fixed_next_z']-b['z'])/scale
    membership=(b['next_z']-b['fixed_next_z'])/scale
    increment=np.mean(dz**2,axis=1)
    boundary=dict(n=len(increment),geometric_rms_A=float(np.sqrt(np.mean(b['displacement_A']**2))),
        retention_mean=float(b['retained'].mean()),
        spearman_geometry_embedding=correlation(b['displacement_A'],np.sqrt(increment)),
        spearman_replacements_embedding=correlation(1-b['retained'],np.sqrt(increment)),
        motion_energy_over_total=float(np.mean(motion**2)/np.mean(dz**2)),
        membership_energy_over_total=float(np.mean(membership**2)/np.mean(dz**2)),
        cross_energy_over_total=float(2*np.mean(motion*membership)/np.mean(dz**2)))
    membership_rows=[]
    for j,count in enumerate((1,2,4)):
        v=(b['swapped_z'][:,j]-b['z'])/scale
        membership_rows.append(dict(replaced_atoms=count,n=len(v),
            embedding_mse_over_0p75ps=float(np.mean(v**2)/step_mse),
            hot_tda_mse_over_0p75ps=float(balanced_errors(b['swapped_hot'][:,j],b['hot'],hot_scale).mean()/hot_step)))
    write_table(root,'controlled-membership',membership_rows)
    lag_rows=[]; physical=[]
    predicted_hot=predict(z,fits['hot']);predicted_relaxed=predict(z,fits['relaxed'])
    np.savez_compressed(tech/'temporal_readouts.npz',hot=predicted_hot,relaxed=predicted_relaxed)
    for lag in cfg['temporal_lags_steps']:
        delta=(z[:,lag:]-z[:,:-lag])/scale
        fixed_delta=(temporal['fixed_z'][:,lag:]-temporal['fixed_z'][:,:-lag])/scale
        start=(z[:,:-lag]-mean)/scale;end=(z[:,lag:]-mean)/scale
        zc=(z-z.mean(1,keepdims=True))/scale
        actual=temporal['hot'][:,lag:]-temporal['hot'][:,:-lag]
        predicted=predicted_hot[:,lag:]-predicted_hot[:,:-lag]
        energy=balanced_errors(np.zeros_like(actual),actual,hot_scale)
        err=balanced_errors(predicted,actual,hot_scale)
        lag_rows.append(dict(lag_ps=lag*.75,pairs=int(delta.shape[0]*delta.shape[1]),
            embedding_standardized_mse=float(np.mean(delta**2)),
            fixed_membership_mse=float(np.mean(fixed_delta**2)),
            global_centered_correlation=cosine(start,end),
            track_centered_correlation=cosine(zc[:,:-lag],zc[:,lag:]),
            observed_tda_increment_skill=float(1-err.mean()/energy.mean()),
            observed_tda_increment_cosine=cosine(predicted/fits['hot'][3],actual/fits['hot'][3]),
            relaxed_readout_delta_mse=float(balanced_errors(predicted_relaxed[:,lag:],predicted_relaxed[:,:-lag],fits['relaxed'][-1]).mean()),
            ptm_label_flip_fraction=float(np.mean(temporal['labels'][:,lag:]!=temporal['labels'][:,:-lag]))))
        latent=np.sqrt(np.mean(delta**2,axis=-1))
        for j,name in enumerate(('PTM_RMSD','PTM_cutoff_margin','q4','q6','density_r12','mean_r12')):
            d=temporal['observables'][:,lag:,j]-temporal['observables'][:,:-lag,j]
            physical.append(dict(lag_ps=lag*.75,observable=name,defined_pairs=int(np.isfinite(d).sum()),
                rms_change=float(np.sqrt(np.nanmean(d*d))),
                spearman_abs_change_vs_embedding=correlation(np.abs(d),latent)))
        physical.append(dict(lag_ps=lag*.75,observable='observed_TDA',rms_change=float(np.sqrt(energy.mean())),
            spearman_abs_change_vs_embedding=correlation(np.sqrt(energy),latent)))
    write_table(root,'temporal-lags',lag_rows);write_table(root,'physical-changes',physical)
    quant=data['global_precision'];precision_rows=[]
    for name in ('quantized_z','fixed_quantized_z'):
        v=(quant[name]-quant['z'])/scale
        precision_rows.append(dict(control=name,n=len(v),mse=float(np.mean(v*v)),
            mse_over_0p75ps=float(np.mean(v*v)/step_mse)))
    qe=balanced_errors(quant['quantized_hot'],quant['hot'],hot_scale)
    precision_rows.append(dict(control='observed_TDA_global_float16',n=len(qe),mse=float(qe.mean()),
        mse_over_0p75ps=float(qe.mean()/hot_step)))
    precision=dict(matched_atom_offset_rms_A=float(np.sqrt(np.mean(quant['displacement_A']**2))),
        ptm_label_flip_fraction=float(np.mean(quant['labels']!=quant['quantized_labels'])),
        ptm_reference_no_template=int(np.isnan(quant['obs'][:,0]).sum()),
        ptm_quantized_no_template=int(np.isnan(quant['quantized_obs'][:,0]).sum()),
        ptm_rmsd_error=float(np.sqrt(np.nanmean((quant['obs'][:,0]-quant['quantized_obs'][:,0])**2))))
    write_table(root,'storage-precision',precision_rows)
    sib=data['siblings'];sibling_rows=[]
    for j,time_ps in enumerate(sib['time_ps']):
        for condition in ('same_momenta','different_momenta'):
            pairs=[(a,b) for a in range(len(sib['z'])) for b in range(a+1,len(sib['z']))
                if (sib['momentum'][a]==sib['momentum'][b])==(condition=='same_momenta')]
            a=np.array([p[0] for p in pairs]);b=np.array([p[1] for p in pairs])
            diff=(sib['z'][a,:,j]-sib['z'][b,:,j])/scale
            tda=balanced_errors(sib['hot'][a,:,j],sib['hot'][b,:,j],hot_scale)
            sibling_rows.append(dict(lag_ps=float(time_ps),condition=condition,pairs=len(pairs),parent_lineages=1,
                embedding_mse=float(np.mean(diff**2)),embedding_mse_over_0p75ps=float(np.mean(diff**2)/step_mse),
                hot_tda_mse=float(np.mean(tda)),ptm_disagreement=float(np.mean(sib['labels'][a,:,j]!=sib['labels'][b,:,j]))))
    write_table(root,'sibling-divergence',sibling_rows)
    details.update(boundaries=boundary,precision=precision,reference_step_mse=step_mse,reference_hot_tda_step_mse=hot_step,
        forecast_checkpoint=str(forecast_checkpoint),forecast_checkpoint_sha256=sha256(Path(forecast_checkpoint)),
        extraction_sha256=sha256(tech/'extraction.json'),n_test_sources=6,observable_names=['PTM_RMSD','PTM_cutoff_margin','q4','q6','density_r12','mean_r12'])
    write_json(tech/'summary.json',details)
    snapshot_metric_docs(root,'mace_encoder_diagnostics')
    figures(root,probe_table,rows,data['boundaries'],increment,membership_rows,lag_rows,sibling_rows)
    print(json.dumps(details,indent=2),flush=True)


def figures(root,probes,controls,b,increment,members,lags,siblings):
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    ax=axes[0,0]
    models=['training_mean','temperature_mean','geometry','within_context_shuffle','mace']
    for i,target in enumerate(('hot','relaxed')):
        scores=[next(r for r in probes if r['target']==target and r['split']=='test' and r['model']==m) for m in models]
        ax.bar(np.arange(len(models))+(i-.5)*.36,[r['skill'] for r in scores],.36,label=target)
    ax.set_xticks(range(len(models)),['Mean','Temp.','Geometry','Shuffled','MACE'],rotation=30)
    ax.set_ylabel('Held-out TDA skill vs train mean');ax.legend();ax.axhline(0,color='k',lw=.7)
    ax=axes[0,1]
    selected=[r for r in controls if r['control'] in ('repeat1','rotation','permutation','radial_fp32','local_float16','embedding_float16')]
    ax.barh([r['control'] for r in selected],[max(r['mse_over_0p75ps'],1e-16) for r in selected])
    ax.set_xscale('log');ax.set_xlabel('Error MSE / 0.75 ps change MSE');ax.axvline(1,color='k',lw=.7)
    ax=axes[0,2]
    scatter=ax.scatter(b['displacement_A'],np.sqrt(increment),c=1-b['retained'],s=20,cmap='viridis')
    fig.colorbar(scatter,ax=ax,label='Fraction of neighbors replaced')
    ax.set_xlabel('Atom-matched motion (Å RMS)');ax.set_ylabel('Embedding change (standardized RMS)')
    ax=axes[1,0]
    ax.bar([str(r['replaced_atoms']) for r in members],[r['embedding_mse_over_0p75ps'] for r in members])
    ax.set_xlabel('Boundary atoms replaced at fixed geometry');ax.set_ylabel('Embedding MSE / 0.75 ps change MSE')
    ax=axes[1,1]
    ax.plot([r['lag_ps'] for r in lags],[r['global_centered_correlation'] for r in lags],'o-',label='Level correlation')
    ax.plot([r['lag_ps'] for r in lags],[r['observed_tda_increment_skill'] for r in lags],'o-',label='TDA increment skill')
    ax.set_xlabel('Lag (ps)');ax.legend();ax.axhline(0,color='k',lw=.7)
    ax=axes[1,2]
    for name in ('same_momenta','different_momenta'):
        records=[r for r in siblings if r['condition']==name]
        ax.plot([r['lag_ps'] for r in records],[r['embedding_mse_over_0p75ps'] for r in records],'o-',label=name.replace('_',' '))
    ax.set_xlabel('Time after branching (ps)');ax.set_ylabel('Sibling MSE / 0.75 ps change MSE');ax.legend()
    fig.suptitle('Forecast MACE encoder: held-out topology, stability and physical evolution')
    fig.savefig(root/'plots/diagnostics.png',dpi=170);fig.savefig(root/'plots/diagnostics.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True)
    args=parser.parse_args();settings=load_json(args.config);cfg=load_json(settings['extraction_config'])
    diagnostics(cfg,Path(cfg['output']),settings['forecast_checkpoint'])


if __name__=='__main__':main()
