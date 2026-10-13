"""Local, frozen diagnostic readouts: present-response residuals and memory."""
from collections import defaultdict
import math
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.research.ni_latent_sim.history_diagnostics import design, solve, predict
from .data import root, atomic_save, write_json, digest
from .memory import load, GROUPS, GROUP_NAMES, FAMILY
from .model import feature_weights
from .evaluate import aggregate


def response_rows(prediction, target, records, **context):
    """Ancestor-balanced ratios and paired numerator/denominator bootstrap."""
    ancestors = np.array([r['ancestor'] for r in records]); unique = sorted(set(ancestors))
    rng = np.random.default_rng(20261017); draws = rng.integers(len(unique),size=(1000,len(unique)))
    rows = []
    for (a,b),group in zip(GROUPS,GROUP_NAMES):
        error = (prediction[...,a:b]-target[...,a:b]).square().mean(-1).cpu().numpy()
        power = target[...,a:b].square().mean(-1).cpu().numpy()
        ee = np.stack([error[ancestors==s].mean(0) for s in unique])
        pp = np.stack([power[ancestors==s].mean(0) for s in unique])
        mse,den = ee.mean(0),pp.mean(0); ratio = mse/den.clip(1e-5)
        boot = ee[draws].mean(1)/pp[draws].mean(1).clip(1e-5)
        lo,hi = np.quantile(boot,[.025,.975],axis=0)
        for t in range(target.shape[1]):
            rows.append(dict(context,group=group,horizon_fs=5*(t+1),mse=float(mse[t]),
                target_power=float(den[t]),relative_mse=float(ratio[t]),zero_relative_mse=float(den[t]/max(den[t],1e-5)),
                lo=float(lo[t]),hi=float(hi[t]),rows=len(records),ancestors=len(unique)))
    return rows


def subsets(c, bank):
    train = np.flatnonzero([b['role']=='train' for b in bank]); ancestors = sorted({bank[i]['ancestor'] for i in train})
    result = {}; all_ids = tuple(train.tolist())
    for seed in c['assays']['source_order_seeds']:
        order = np.random.default_rng(seed).permutation(ancestors)
        for count in c['assays']['source_counts']:
            chosen = tuple(i for i in train if bank[i]['ancestor'] in set(order[:count]))
            result.setdefault(chosen,dict(curve='ancestors',order_seed=seed,count=count))
    generator = np.random.default_rng(c['seed'])
    source_rows = {a:generator.permutation([i for i in train if bank[i]['ancestor']==a]) for a in ancestors}
    for count in c['assays']['within_source_counts']:
        chosen = tuple(sorted(i for a in ancestors for i in source_rows[a][:count]))
        if chosen != all_ids: result.setdefault(chosen,dict(curve='anchors_per_ancestor',order_seed=c['seed'],count=count))
    return [(np.array(ids),meta) for ids,meta in result.items()]


def fit_response(h, initial, target, train, selection, kind, c):
    # H and R0 are normalized only with this readout's fitting rows. R0 is never
    # centered; kernel multiplication preserves exact query linearity.
    mean=h[train].mean(0); scale=h[train].std(0).clamp_min(.01); z=(h-mean)/scale
    rms=initial[train].square().mean(0).sqrt().clamp_min(1e-5); u=initial/rms
    norm=torch.empty(target.shape[1:],dtype=target.dtype)
    for a,b in GROUPS: norm[:,a:b]=target[train,:,a:b].square().mean((0,2)).clamp_min(1e-5).sqrt()[:,None]
    skip = initial[:,None].expand_as(target) if kind!='transport' else torch.zeros_like(target)
    y=((target-skip)/norm).flatten(1)
    distance=torch.cdist(z,z[train]).square()/z.shape[1]
    bandwidth=distance[train].flatten().median().clamp_min(1e-8)
    product=(u@u[train].T)/u.shape[1]
    lengths=c['assays']['state_lengths'] if kind=='residual_state' else [None]
    best=None; candidates=[]
    for length in lengths:
        matrix=product if length is None else product*(1+torch.exp(-distance/(2*bandwidth*length**2)))/2
        for penalty in c['assays']['penalties']:
            coefficients=torch.linalg.solve(matrix[train]+len(train)*penalty*torch.eye(len(train),dtype=h.dtype),y[train])
            prediction=(matrix[selection]@coefficients).reshape(len(selection),*target.shape[1:])*norm+skip[selection]
            error=(prediction-target[selection])/norm
            nll=torch.stack([.5*(error[...,a:b].square()+2*norm[None,:,a:b].log()+math.log(2*math.pi)).mean() for a,b in GROUPS]).mean()
            candidates.append(dict(length=length,penalty=penalty,selection_nll=float(nll)))
            if best is None or float(nll)<best['selection_nll']:
                forecast=(matrix@coefficients).reshape_as(target)*norm+skip
                best=dict(prediction=forecast,selection_nll=float(nll),length=length,penalty=penalty,
                          coefficients=coefficients,input_mean=mean,input_scale=scale,query_rms=rms,target_scale=norm)
    return best,candidates


def response(c):
    out=root(c)/'analyses/response-readouts'; marker=out/'technical/complete.json'
    if marker.exists(): return
    _,_,_,bank,physical=load(c)
    h=physical['history'].flatten(1).double()
    roles=np.array([b['role'] for b in bank]); selection=np.flatnonzero(roles=='selection'); held=np.flatnonzero(roles=='test')
    rows=[]; candidate_rows=[]; packets=[]; panels=subsets(c,bank)
    for query,direction in enumerate(['q','v']):
        target=physical['response'][...,query].double(); initial=physical['initial_response'][...,query].double()
        for label,forecast in [('zero',torch.zeros_like(target)),('persistence',initial[:,None].expand_as(target))]:
            rows.extend(response_rows(forecast[held],target[held],[bank[i] for i in held],
                curve='baseline',subset='all',fit_anchors=0,fit_ancestors=0,readout=label,direction=direction,selection_nll=None))
        for train,meta in panels:
            subset=digest(train.tolist())[:12]
            for kind in ['transport','residual_linear','residual_state']:
                path=out/'technical'/f'{subset}-{direction}-{kind}.pt'
                if path.exists(): saved=torch.load(path,map_location='cpu',weights_only=False)
                else:
                    fitted,candidates=fit_response(h,initial,target,train,selection,kind,c)
                    saved=dict(fit=fitted,candidates=candidates,train=train,selection=selection,test=held,subset=meta)
                    atomic_save(path,saved)
                fitted=saved['fit']; context=dict(curve=meta['curve'],subset=subset,fit_anchors=len(train),
                    fit_ancestors=len({bank[i]['ancestor'] for i in train}),readout=kind,direction=direction,
                    selection_nll=fitted['selection_nll'])
                rows.extend(response_rows(fitted['prediction'][held],target[held],[bank[i] for i in held],**context))
                candidate_rows.extend(dict(context,**{k:v for k,v in a.items() if k!='selection_nll'},candidate_nll=a['selection_nll']) for a in saved['candidates'])
                packets.append(dict(context,metadata=meta))
                write_json(out/'technical/progress.json',dict(fits=len(packets),total=6*len(panels),last=context))
                print('response assay',context,flush=True)
    write_metric_rows(rows,out,family=FAMILY,name='response')
    write_metric_rows(candidate_rows,out,family=FAMILY,name='selection')
    write_json(marker,dict(state='complete',selected_fits=len(packets),subsets=len(panels),anchors=len(bank),
        present_response='analytic smooth91 from current Q/V and direction; supplied identity skip is exact by construction',
        interpretation='predictability given H and R0; not an information bound for H alone'))


def history(c):
    out=root(c)/'analyses/history-readouts'; marker=out/'technical/complete.json'
    if marker.exists(): return
    _,data,x,_,_=load(c); records=data['rows']; roles=np.array([r['role'] for r in records])
    train,selection,held=[np.flatnonzero(roles==role) for role in ['train','selection','test']]
    offsets=c['assays']['history_offsets']; target=x[:,[13+i for i in offsets]].flatten(1)
    weights=feature_weights(data['scale']); rows=[]
    for length,mode in [(4,'ordered'),(14,'ordered'),(14,'unordered_past')]:
        features=design(x[:,:14],length,mode); path=out/'technical'/f'h{length}-{mode}.pt'
        if path.exists(): fitted=torch.load(path,map_location='cpu',weights_only=False)
        else:
            fitted=solve(features,target,train,selection,c['assays']['penalties']); atomic_save(path,fitted)
        prediction=predict(fitted,features[held]).reshape(len(held),len(offsets),244)
        actual=target[held].reshape_as(prediction); variance=fitted['variance'].reshape(len(offsets),244)
        error=(prediction-actual).square()
        scores=dict(mse=(error*weights).sum(-1),nll=(.5*(error/variance+variance.log()+math.log(2*math.pi))*weights).sum(-1))
        for index,offset in enumerate(offsets):
            for metric,values in scores.items():
                mean,lo,hi=aggregate(values[:,index].numpy(),[records[i]['ancestor'] for i in held])
                rows.append(dict(history=length,mode=mode,horizon_ps=offset/10,metric=metric,value=mean,lo=lo,hi=hi,
                    selection_nll=fitted['selection_nll'],penalty=fitted['penalty'],rows=len(held),ancestors=40))
        atomic_save(out/'technical'/f'h{length}-{mode}-predictions.pt',dict(prediction=prediction.float(),indices=held,offsets=offsets))
        print('history assay',length,mode,fitted['selection_nll'],flush=True)
    write_metric_rows(rows,out,family=FAMILY,name='history')
    write_json(marker,dict(state='complete',fits=3,origin_shift_ps=1.,maximum_future_ps=5.,all_rows=len(records)))
