"""Fixed physical future scores and genuinely withheld sub-cadence outcomes."""
import math
import time
from collections import defaultdict
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.research.response_geometry.features import observed
from .data import FAMILY, root, cache, read, write_json
from .model import mixture_nll


def summarize(values, ancestors, rng_seed):
    """Equal ancestor means and a descriptive ancestor bootstrap interval."""
    groups=defaultdict(list)
    for value,ancestor in zip(values,ancestors,strict=True):
        groups[ancestor].append(float(value))
    x=np.array([np.mean(v) for v in groups.values()])
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Empty or nonfinite ancestor-level metric')
    rng=np.random.default_rng(rng_seed)
    draws=x[rng.integers(len(x),size=(1000,len(x)))].mean(1)
    return dict(mean=float(x.mean()),lower=float(np.quantile(draws,.025)),upper=float(np.quantile(draws,.975)),ancestors=len(x))


def predict(model,state,horizons,seed,*,step_ps=None):
    with torch.no_grad():
        z=model.encode(state,create_graph=False)
        # An explicitly labeled linear interpolation reference is used for the
        # fixed-step baseline; interpolated values are NEVER training labels.
        if model.arm=='discrete' and any(abs(h/.1-round(h/.1))>1e-7 for h in horizons):
            end=model.path(z,[.1],particles=8,seed=seed)
            initial=z[:,None,None,:].expand(-1,8,1,-1)
            a,la=model.prediction(initial);b,lb=model.prediction(end)
            fraction=state.new_tensor(horizons)[None,None,:,None]/.1
            return a+(b-a)*fraction,la+(lb-la)*fraction
        return model.prediction(model.path(z,horizons,particles=8,seed=seed,step_ps=step_ps))


def evaluate(c,plan,model,data,bank,folder,deadline):
    center=data['mean'].float().cuda();scale=data['scale'].float().cuda()
    indices=np.array([i for i,r in enumerate(data['rows']) if r['role']=='test'])
    rows=[data['rows'][i] for i in indices];micro=c['training']['microbatch']
    fitting=torch.tensor([r['role']=='train' for r in data['rows']])
    fit_y=(data['target'][fitting].float().cuda()-center)/scale
    marginal_mu=fit_y.mean(0);marginal_lv=fit_y.var(0,unbiased=False).clamp_min(1e-5).log()
    errors=[];likelihood=[];means=[];variances=[];persistence=[];marginal=[];coverage=[]
    for chunk in np.array_split(indices,max(1,math.ceil(len(indices)/micro))):
        if time.time()>=deadline:
            raise TimeoutError('Evaluation deadline; checkpoint retained')
        mu,lv=predict(model,data['state'][chunk].cuda(),c['horizons_ps'],c['seed']+111)
        y=(data['target'][chunk].float().cuda()-center)/scale
        likelihood.append(mixture_nll(mu,lv,y).cpu())
        errors.append((mu.mean(1)-y).square().mean(-1).cpu())
        present=(observed(data['state'][chunk,:,:3].double().cuda())-center)/scale
        persistence.append((present[:,None]-y).square().mean(-1).cpu())
        marginal.append(mixture_nll(marginal_mu[None,None].expand(len(chunk),1,-1,-1),
                         marginal_lv[None,None].expand(len(chunk),1,-1,-1),y).cpu())
        pit=(.5*(1+torch.erf((y[:,None]-mu)/torch.sqrt(2*lv.exp())))).mean(1)
        coverage.append(((pit>=.05)&(pit<=.95)).float().mean(-1).cpu())
        means.append(mu.mean(1).cpu());variances.append((lv.exp()+mu.square()).mean(1).sub(mu.mean(1).square()).cpu())
    nll=torch.cat(likelihood).numpy();mse=torch.cat(errors).numpy();metrics=[]
    pmse=torch.cat(persistence).numpy();mnll=torch.cat(marginal).numpy();cov=torch.cat(coverage).numpy()
    for track,keep in [('all64',np.ones(len(rows),bool)),('legacy16',np.array([r['legacy'] for r in rows]))]:
        ancestors=[r['ancestor'] for r,k in zip(rows,keep) if k]
        for j,h in enumerate(c['horizons_ps']):
            n=summarize(nll[keep,j],ancestors,c['seed']);e=summarize(mse[keep,j],ancestors,c['seed'])
            metrics.append(dict(track=track,horizon_ps=h,rows=int(keep.sum()),ancestors=n['ancestors'],
                                nll=n['mean'],nll_lower=n['lower'],nll_upper=n['upper'],mean_mse=e['mean'],
                                persistence_mse=summarize(pmse[keep,j],ancestors,c['seed'])['mean'],
                                marginal_nll=summarize(mnll[keep,j],ancestors,c['seed'])['mean'],
                                coverage90=summarize(cov[keep,j],ancestors,c['seed'])['mean']))
    np.savez_compressed(folder/'technical/future-predictions.npz',row=indices,mean=torch.cat(means).numpy(),
                        variance=torch.cat(variances).numpy(),nll=nll,mean_mse=mse,
                        persistence_mse=pmse,marginal_nll=mnll,coverage90=cov,
                        target=((data['target'][indices]-data['mean'])/data['scale']).numpy())
    dense_rows=[];horizons=[x/1000 for x in c['teacher']['save_fs']]
    # Fresh MEAM continuation law, held-out source ancestors. Each source's
    # trajectories include actual10/20/50fs observations unseen in fitting.
    for k,r in enumerate(bank):
        if r['role']!='test':
            continue
        if time.time()>=deadline:
            raise TimeoutError('Dense evaluation deadline; checkpoint retained')
        state=r['state'][None].float().cuda()
        mu,lv=predict(model,state,horizons,c['seed']+112)
        y=observed(r['future'][...,:3].reshape(-1,80,3).cuda()).reshape(c['teacher']['shots'],len(horizons),24)
        y=(y-center)/scale
        error=(mu.mean(1)[0]-y.mean(0)).square().mean(-1)
        noise=y.var(0,unbiased=True).mean(-1)/len(y)
        # Actual same-initial-V ballistic continuation is an interpretable
        # motion control; full physics continues to provide the reference.
        ballistic=[]
        physical_state=r['state'][None].cuda()
        for h in horizons:
            q=physical_state[...,:3]+h*physical_state[...,3:]
            ballistic.append((observed(q)-center)/scale)
        ballistic=torch.cat(ballistic)
        be=(ballistic-y.mean(0)).square().mean(-1)-noise
        if model.arm!='discrete':
            with torch.no_grad():
                z=model.encode(state,create_graph=False)
                coarse,_=model.prediction(model.path(z,horizons,particles=1,seed=0,deterministic=True))
                refined,_=model.prediction(model.path(z,horizons,particles=1,seed=0,deterministic=True,
                                                    step_ps=c['model']['step_ps']/2))
                refinement=(coarse-refined).square().mean(-1)[0,0]
        else:
            refinement=None
        for j,h in enumerate(horizons):
            dense_rows.append(dict(ancestor=r['ancestor'],job=r['job'],horizon_ps=h,
                mean_mse=float(error[j]-noise[j]),ballistic_mean_mse=float(be[j]),
                target_mean_noise=float(noise[j]),drift_step_refinement_mse=None if refinement is None else float(refinement[j])))
    for h in horizons:
        selected=[r for r in dense_rows if r['horizon_ps']==h];anc=[r['ancestor'] for r in selected]
        error=summarize([r['mean_mse'] for r in selected],anc,c['seed'])
        ballistic=summarize([r['ballistic_mean_mse'] for r in selected],anc,c['seed'])
        metrics.append(dict(track='dense_meam_test',horizon_ps=h,rows=len(selected),ancestors=error['ancestors'],
                            mean_mse=error['mean'],mean_mse_lower=error['lower'],mean_mse_upper=error['upper'],
                            ballistic_mean_mse=ballistic['mean']))
    # Actual native0.01ps NPT observations are a separate diagnostic. Their
    # original train ancestry is retained; this is not an independent test.
    native=[];dense_times=[.01,.02,.05,.09]
    for job in plan['jobs']:
        item=next(s for s in plan['sources'] if s['key']==job['source_key'])
        if abs(item['source']['native_cadence_ps']-.01)>1e-8:
            continue
        packet=torch.load(cache(c)/'observations'/job['key']/'data.pt',weights_only=False)
        for start in range(0,64,micro):
            if time.time()>=deadline:
                raise TimeoutError('Native dense evaluation deadline; checkpoint retained')
            mu,_=predict(model,packet['state'][start:start+micro].cuda(),dense_times,c['seed']+113)
            target=(packet['dense_target'][start:start+micro].float().cuda()-center)/scale
            err=(mu.mean(1)-target).square().mean(-1).cpu().numpy()
            for sample in err:
                for j,h in enumerate(dense_times):
                    native.append(dict(ancestor=item['source']['ancestry_group'],role=item['role'],
                                       job=job['key'],horizon_ps=h,mean_mse=float(sample[j])))
    for h in dense_times:
        selected=[r for r in native if r['horizon_ps']==h]
        if selected:
            e=summarize([r['mean_mse'] for r in selected],[r['ancestor'] for r in selected],c['seed'])
            metrics.append(dict(track='native001_withheld_times_training_ancestry',horizon_ps=h,
                                rows=len(selected),ancestors=e['ancestors'],mean_mse=e['mean']))
    columns=['track','horizon_ps','rows','ancestors','nll','nll_lower','nll_upper','mean_mse',
             'mean_mse_lower','mean_mse_upper','ballistic_mean_mse','persistence_mse','marginal_nll','coverage90']
    write_metric_rows(metrics,folder,family=FAMILY,name='predictive',columns=columns)
    write_metric_rows(dense_rows,folder,family=FAMILY,name='dense-meam')
    if native:
        write_metric_rows(native,folder,family=FAMILY,name='native001')
    write_json(folder/'technical/evaluation.json',dict(
        discrete_intermediate_reference='linear interpolation of predicted0/100fs moments',
        future_law='saved native source NPT trajectories; single realized future per observation',
        dense_law='new fixed-box BAOAB shots; sensitivity noise held fixed; not reconstruction of historical noise',
        native001='actual withheld0.01ps source frames; existing original train ancestry',
        refinement='deterministic drift-only 0.05 versus0.025ps; not a stochastic convergence claim',
        selection_uses_test=False,mean_error_noise_correction='subtract shot sample variance / shot count; may be negative'))
    return metrics
