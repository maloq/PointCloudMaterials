"""Proper scores, paired source uncertainty, and within-snapshot physical profiles."""
import json
from pathlib import Path
import numpy as np
import torch
from src.data.fixed_cohort.protocol import sha,write_json
from src.project_runtime.paths import resolve_path
from src.experiment_runner.metric_docs import write_metric_rows
from src.research.distance_encoder.model import loss_terms
from src.research.spatial_distance.model import cdf,capped_mean
from .data import population,masks,features,ROLES

def tables(root,name,rows):
    return write_metric_rows(rows, root, family='liquid_predictability', name=name)

def score_values(values,distance,w):
    w=w/w.sum();result=dict(distance_nll=float(w@values['nll']),distance_rmse_A=float(np.sqrt(w@(values['mean_A']-np.minimum(distance,64))**2)))
    for j,r in enumerate((20,32,48)):
        y=distance<=r;prob=values['cdf'][:,j]
        result[f'brier{r}A']=float(w@(prob-y)**2);result[f'prevalence{r}A']=float(w@y);result[f'mean_probability{r}A']=float(w@prob)
    return result

def subsets(meta,ids,clearance,c):
    yield 'all',np.ones(len(ids),bool)
    yield 'original_fixed_all64',(meta['kind'][ids]==0)&(meta['interface_parent_index'][ids]>=0)
    for low,high in zip(c['evaluation']['clearance_bins_A'][:-1],c['evaluation']['clearance_bins_A'][1:]):
        yield f'clearance_{low}_{high}A',(clearance[ids]>=low)&(clearance[ids]<high)
    d=meta['crystal_distance'][ids]
    for low,high in zip(c['evaluation']['distance_bins_A'][:-1],c['evaluation']['distance_bins_A'][1:]):
        yield f'distance_{low}_{high}A',(d>=low)&(d<high)
    yield 'beyond_32A_observation_envelope',d>32

@torch.no_grad()
def export(model,data,c,study):
    root=study.root/'analyses/predictability-v1';tech=root/'technical';tech.mkdir(parents=True,exist_ok=True)
    _,_,clearance=features(c,data.meta);ids=np.concatenate([data.split[r] for r in ROLES]);model.eval()
    result={k:[] for k in ('nll','mean_A','cdf','target_cdf')}
    for start in range(0,len(ids),c['batch_size']):
        b=data.batch(ids[start:start+c['batch_size']])
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=data.device.type=='cuda'):out=model(b)
        _,nll,_=loss_terms(out['parts'],b['distance'],c['loss'])
        values=dict(nll=nll,mean_A=capped_mean(out['parts'],64),cdf=cdf(out['parts'],b['distance'].new_tensor([20,32,48])),
            target_cdf=cdf(out['parts'],b['distance'][:,None,None]).squeeze(-1))
        for k,v in values.items():result[k].append(v.float().cpu().numpy())
    values={k:np.concatenate(v) for k,v in result.items()};np.savez(tech/'predictions.npz',ids=ids,**values)
    scores=[];reliability=[];summary={};begin=0
    for role in ROLES:
        n=len(data.split[role]);role_ids=ids[begin:begin+n];pred={k:v[begin:begin+n] for k,v in values.items()};w=data.weights[role]
        for name,keep in subsets(data.meta,role_ids,clearance,c):
            if not keep.any():continue
            metrics=score_values({k:v[keep] for k,v in pred.items()},data.meta['crystal_distance'][role_ids[keep]],w[keep])
            scores.append(dict(role=role,subset=name,rows=int(keep.sum()),sources=len(np.unique(data.meta['source'][role_ids[keep]])),**metrics))
            if name=='all':summary.update({role+'/'+k:v for k,v in metrics.items()})
        for j,r in enumerate((20,32,48)):
            p=pred['cdf'][:,j];target=data.meta['crystal_distance'][role_ids]<=r
            for k in range(10):
                take=(p>=k/10)&((p<(k+1)/10) if k<9 else (p<=1));mass=float(w[take].sum())
                reliability.append(dict(role=role,radius_A=r,bin_low=k/10,bin_high=(k+1)/10,rows=int(take.sum()),mass=mass,
                    mean_probability=float(w[take]@p[take]/mass) if mass else None,frequency=float(w[take]@target[take]/mass) if mass else None))
        begin+=n
    tables(root,'scores',scores);tables(root,'reliability',reliability)
    write_json(tech/'complete.json',dict(identity=study.identity,checkpoint_sha256=sha(study.technical/'best.pt'),
        predictions_sha256=sha(tech/'predictions.npz'),rows=len(ids),selection='validation distance NLL',test_used_for_selection=False))
    return {'evaluation/'+k:v for k,v in summary.items()}

def profiles(c):
    _,_,_,meta,base=population(c);_,physical,clearance=features(c,meta)
    arm=next(a for a in c['arms'] if a['name']=='joint_mace');split,weights,_=masks(meta,base,arm,c)
    train=split['train'];mean=weights['train']@physical[train];scale=np.sqrt(weights['train']@(physical[train]-mean)**2).clip(1e-8)
    names=[f'radial_{v:.3f}A' for v in np.linspace(.5,7.5,24)]+['smooth_count_5A','smooth_count_8A']+[f'bond_power_l{l}_{r}A' for l in (2,4,6) for r in (5,8)]
    root=resolve_path(c['output'])/'analyses/physical-profiles-v1';rows=[];association=[]
    rng=np.random.default_rng(c['seed']);draws=c['evaluation']['bootstrap_draws']
    for role in ROLES:
        ids=split[role];w=weights[role];x=(physical[ids]-mean)/scale;d=np.minimum(meta['crystal_distance'][ids],64).astype(float)
        groups=np.c_[meta['source'][ids],meta['frame'][ids]];_,inverse=np.unique(groups,axis=0,return_inverse=True)
        mass=np.bincount(inverse,weights=w);xd=x.copy();dd=d.copy()
        for j in range(32):xd[:,j]-=(np.bincount(inverse,weights=w*x[:,j])/mass)[inverse]
        dd-=(np.bincount(inverse,weights=w*d)/mass)[inverse]
        sources,si=np.unique(meta['source'][ids],return_inverse=True);boot=rng.integers(0,len(sources),(draws,len(sources)))
        for j,name in enumerate(names):
            cross=np.bincount(si,weights=w*xd[:,j]*dd);vx=np.bincount(si,weights=w*xd[:,j]**2);vd=np.bincount(si,weights=w*dd**2)
            denom=np.sqrt(vx.sum()*vd.sum());bs=cross[boot].sum(1)/np.sqrt(vx[boot].sum(1)*vd[boot].sum(1))
            association.append(dict(role=role,feature=name,within_snapshot_correlation=float(cross.sum()/denom) if denom else None,
                ci95_low=float(np.quantile(bs,.025)),ci95_high=float(np.quantile(bs,.975)),
                familywise_low=float(np.quantile(bs,.025/32)),familywise_high=float(np.quantile(bs,1-.025/32)),sources=len(sources)))
            for low,high in zip(c['evaluation']['distance_bins_A'][:-1],c['evaluation']['distance_bins_A'][1:]):
                take=(meta['crystal_distance'][ids]>=low)&(meta['crystal_distance'][ids]<high);mw=w[take].sum()
                if mw:rows.append(dict(role=role,feature=name,low_A=low,high_A=high,rows=int(take.sum()),mass=float(mw),
                    standardized_mean=float(w[take]@x[take,j]/mw),within_snapshot_centered_mean=float(w[take]@xd[take,j]/mw)))
    tables(root,'distance-profiles',rows);tables(root,'within-snapshot-associations',association)
    write_json(root/'technical/complete.json',dict(feature_transform='training-only standardization; frame centering is analysis-only',
        model_inputs='no frame, time, temperature, visibility, clearance or distance metadata',features=names))
    render_profiles(root,rows,names)

def render_profiles(root,rows,names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for ax,name in zip(axes.flat,[names[i] for i in (24,25,28,29,30,31)]):
        for role in ('train','selection','test'):
            chosen=[r for r in rows if r['feature']==name and r['role']==role and r['high_A']<=64]
            ax.plot([(r['low_A']+r['high_A'])/2 for r in chosen],[r['within_snapshot_centered_mean'] for r in chosen],'o-',label=role)
        ax.set(title=name,xlabel='Distance to crystal (Å)',ylabel='Within-snapshot deviation / training SD');ax.axhline(0,color='.7',ls='--')
    axes[0,0].legend();(root/'plots').mkdir(exist_ok=True);fig.savefig(root/'plots/physical-profiles.png',dpi=160);plt.close(fig)

def compare(c):
    _,_,_,meta,base=population(c);_,_,clearance=features(c,meta);root=resolve_path(c['output'])/'analyses/comparison-v1'
    rows=[];seed=c['seed'];draws=c['evaluation']['bootstrap_draws'];count=len(c['arms'])-2
    def load(name):
        folder=resolve_path(c['output'])/name/'analyses/predictability-v1/technical';receipt=json.loads((folder/'complete.json').read_text())
        if sha(folder/'predictions.npz')!=receipt['predictions_sha256']:raise ValueError('Changed predictions')
        with np.load(folder/'predictions.npz') as a:return {k:a[k] for k in a.files}
    for arm in c['arms']:
        if arm['model']=='prior':continue
        value=load(arm['name']);reference=load('visible_prior' if arm['population']=='visible_control' else 'prior')
        refmap=np.full(len(meta['atom']),-1,int);refmap[reference['ids']]=np.arange(len(reference['ids']))
        split,weights,_=masks(meta,base,arm,c)
        for role in ('selection','calibration','test'):
            ids=split[role];lookup=np.full(len(meta['atom']),-1,int);lookup[value['ids']]=np.arange(len(value['ids']))
            ix=lookup[ids];ri=refmap[ids]
            if (ix<0).any() or (ri<0).any():raise ValueError('Unmatched evaluation rows')
            for subset,take in subsets(meta,ids,clearance,c):
                if not take.any():continue
                selected=ids[take];w=weights[role][take];w=w/w.sum();sources,inv=np.unique(meta['source'][selected],return_inverse=True)
                if len(sources)<2:continue
                truth=np.minimum(meta['crystal_distance'][selected],64)
                gain=reference['nll'][ri[take]]-value['nll'][ix[take]]
                mse=(value['mean_A'][ix[take]]-truth)**2;refmse=(reference['mean_A'][ri[take]]-truth)**2
                sums=np.stack([np.bincount(inv,weights=w*z) for z in (np.ones(len(w)),gain,mse,refmse)],1)
                rng=np.random.default_rng(seed);boot=rng.integers(0,len(sources),(draws,len(sources)));s=sums[boot].sum(1)
                bg=s[:,1]/s[:,0];br=1-np.sqrt(s[:,2]/s[:,3]);upper=float(np.quantile(br,1-.05/count))
                rows.append(dict(model=arm['name'],population=arm['population'],source_fraction=arm['source_fraction'],role=role,subset=subset,
                    rows=len(selected),sources=len(sources),nll_gain=float(w@gain),nll_gain_ci95_low=float(np.quantile(bg,.025)),nll_gain_ci95_high=float(np.quantile(bg,.975)),
                    nll_gain_familywise_low=float(np.quantile(bg,.025/count)),nll_gain_familywise_high=float(np.quantile(bg,1-.025/count)),
                    rmse_reduction_fraction=float(1-np.sqrt((w@mse)/(w@refmse))),rmse_reduction_ci95_low=float(np.quantile(br,.025)),
                    rmse_reduction_ci95_high=float(np.quantile(br,.975)),rmse_reduction_familywise_upper=upper,
                    excludes_2percent_benefit=upper<c['evaluation']['meaningful_rmse_reduction_fraction']))
    tables(root,'paired-comparisons',rows)
    write_json(root/'technical/complete.json',dict(bootstrap='paired independent source resampling, preserve conditional masses',
        draws=draws,training_seeds=1,training_seed_uncertainty_measured=False,
        practical_threshold=c['evaluation']['meaningful_rmse_reduction_fraction'],scope='tested fitted models only, not an information-theoretic upper bound',
        multiplicity='Bonferroni across non-prior models within each declared population/subset/role; subgroup results exploratory'))
    lines=['# Liquid structure predictability','', 'Primary held-out comparisons: positive NLL gain favors observed geometry.','',
        '| Model | NLL gain | 95% source interval | RMSE reduction |','|---|---:|---|---:|']
    for r in rows:
        if r['role']=='test' and r['subset']=='all':lines.append(f"| {r['model']} | {r['nll_gain']:.4f} | [{r['nll_gain_ci95_low']:.4f}, {r['nll_gain_ci95_high']:.4f}] | {100*r['rmse_reduction_fraction']:.2f}% |")
    lines.extend(['','One training seed. Source uncertainty does not include optimization-seed uncertainty.',
        'The visible control has a separate population. Negative results bound tested predictors, not all information.',
        'The fixed cohort has informed previous development; independent confirmation remains desirable.'])
    (root/'README.md').write_text('\n'.join(lines)+'\n')
