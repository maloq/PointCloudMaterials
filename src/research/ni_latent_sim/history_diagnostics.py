"""Frozen Gaussian ridge readouts on exact, common history/future windows."""
import numpy as np
import torch
from .data import root, load, atomic_save, write_json
from .path_scores import weighted_summary


def design(x, length, mode):
    history = x[:, -length:]
    if mode == 'ordered' or length == 1:
        return history.flatten(1)
    past = history[:, :-1]
    return torch.cat((history[:, -1], past.mean(1), past.std(1, unbiased=False)), -1)


def solve(x, y, train, selection, penalties):
    mean = x[train].mean(0); scale = x[train].std(0).clamp_min(.01)
    xx = ((x-mean)/scale).double(); yy = y.double()
    center = yy[train].mean(0)
    a = xx[train]; b = yy[train]-center
    covariance = a.T@a/len(train)
    eig, vec = torch.linalg.eigh(covariance)
    rhs = vec.T@(a.T@b/len(train))
    candidates = []
    for penalty in penalties:
        weight = vec@(rhs/(eig[:,None].clamp_min(0)+penalty))
        variance = (a@weight-b).square().mean(0).clamp_min(1e-4)
        prediction = xx[selection]@weight+center
        score = .5*((yy[selection]-prediction).square()/variance+variance.log()+np.log(2*np.pi)).mean()
        candidates.append((float(score), penalty, weight, variance))
    score, penalty, weight, variance = min(candidates, key=lambda v:v[0])
    return dict(mean=mean, scale=scale, center=center, weight=weight, variance=variance,
                penalty=penalty, selection_nll=score)


def predict(fit, x):
    return ((x-fit['mean'])/fit['scale']).double()@fit['weight']+fit['center']


def run(c):
    from .diagnostics import selected, model_for, export
    p = selected(c); data = load(p['source_config'], 'data')
    rows = data['rows']; roles = np.array([r['role'] for r in rows])
    train, selection, test = [np.flatnonzero(roles==role) for role in ['train','selection','test']]
    # Old t0+1ps is the new common origin. Fourteen native observations cover
    # [new t0-1.3ps,new t0], and all rows have the same four future horizons.
    x = (data['x']-data['mean'])/data['scale']; history = x[:,:14]
    offsets = [1,5,10,30]; target = x[:,[13+h for h in offsets],256:].flatten(1)
    summary = []
    representations = [('raw280', history)]
    for spec in p['checkpoints']:
        model = model_for(c,spec,device='cpu')
        with torch.no_grad():
            z = torch.cat([model.encode(block) for block in history.split(256)])
        representations.append((spec['arm']['name']+'-code16', z))
    for name, values in representations:
        for length in [1,4,8,14]:
            for mode in (['ordered'] if length==1 else ['ordered','unordered_past']):
                folder = root(c)/'analyses'/f'history-{name}-h{length}-{mode}'
                path = folder/'technical/readout.pt'
                if path.exists():
                    fit = torch.load(path,map_location='cpu',weights_only=False)
                else:
                    fit = solve(design(values,length,mode),target,train,selection,c['history_penalties'])
                    atomic_save(path,fit)
                interventions = [('natural',values)]
                if length>1 and mode=='ordered':
                    changed = values.clone()
                    changed[:,-length:-1] = values[:,-length:-1].flip(1)
                    interventions.append(('reversed_past_fixed_model',changed))
                for intervention, features in interventions:
                    predicted = predict(fit,design(features[test],length,mode)).reshape(len(test),4,24)
                    actual = target[test].double().reshape(len(test),4,24)
                    variance = fit['variance'].reshape(4,24)
                    metrics = dict(mse=(predicted-actual).square().mean(-1),
                        nll=.5*((predicted-actual).square()/variance+variance.log()+np.log(2*np.pi)).mean(-1))
                    for h, offset in enumerate(offsets):
                        for metric, scores in metrics.items():
                            for record in weighted_summary(scores[:,h].numpy(),[rows[i] for i in test],c['seed']):
                                summary.append(dict(representation=name, history=length, mode=mode,
                                    intervention=intervention,horizon_ps=offset/10,metric=metric,
                                    penalty=fit['penalty'],selection_nll=fit['selection_nll'],**record))
                    atomic_save(folder/'technical'/f'{intervention}-predictions.pt',
                                dict(indices=test,prediction=predicted.float(),variance=variance.float(),
                                     origin_shift_ps=1.,future_offsets_ps=[h/10 for h in offsets]))
                write_json(root(c)/'technical/history-progress.json',dict(representation=name,history=length,mode=mode))
                export(c,'history-comparison',summary)
                print('history readout',name,length,mode,'selection NLL',fit['selection_nll'],flush=True)
    write_json(root(c)/'analyses/history-comparison/technical/interpretation.json',dict(
        selection='training ancestors for fit; original selection ancestors for ridge strength; heldout untouched',
        scope='Gaussian linear readout sufficiency diagnostic, not a replacement stochastic simulator',
        capacity='ordered input width grows with history; unordered control retains present plus past mean/std',
        controls='reverse only past slots at fixed fitted weights, keeping the present unchanged',
        time='shift common forecast origin by +1ps; no interpolation or extra timestamp inputs'))
