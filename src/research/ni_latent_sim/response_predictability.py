"""Frozen response readouts: linear in perturbation, optionally nonlinear in state.

These local diagnostic fits reuse saved simulator JVP targets. They neither
fine-tune an encoder nor integrate a trajectory, and do not create W&B runs.
"""
from pathlib import Path
import numpy as np
import torch
from .data import root, read, sha, atomic_save, write_json
from .dense_response import bank
from .path_scores import weighted_summary


def features(history, directions, train):
    # One training-only feature scale acts on both observations and their
    # directional changes. Directional changes have no centering/intercept.
    mean = history[train].mean((0,1))
    scale = history[train].std((0,1)).clamp_min(.001)
    h = ((history-mean)/scale).flatten(1).double()
    d = (directions/scale).double()
    rms = d[train].square().sum(-1).mean().sqrt()
    if not torch.isfinite(rms) or rms <= 0:
        raise ValueError('No finite nonzero perturbation survives this representation')
    d = d/rms
    distance = torch.cdist(h[train],h[train]).square()/h.shape[1]
    triangle = torch.triu_indices(len(train),len(train),offset=1)
    bandwidth = distance[triangle[0],triangle[1]].median()
    if not torch.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError('State kernel has no positive training distance scale')
    return dict(h=h,d=d,mean=mean,scale=scale,direction_rms=rms,bandwidth=bandwidth)


def kernel(a, b, bandwidth, length):
    product = a['d']@b['d'].T
    if length is None:
        return product
    distances = torch.cdist(a['h'],b['h']).square()/a['h'].shape[1]
    return product*(1+torch.exp(-distances/(2*bandwidth*length**2)))/2


def select(inputs, indices):
    return {k:inputs[k][indices] for k in ['h','d']}


def fit(inputs, target, train, selection, kind, settings):
    # Fixed training-only Gaussian reference variance. Selection is a proper
    # mean-prediction likelihood on future responses, excluding the t0 control.
    target_scale = target[train].square().mean((0,2)).sqrt().clamp_min(1e-8)
    y = (target/target_scale[None,:,None]).flatten(1).double()
    candidates = []; best = None
    lengths = [None] if kind == 'linear_direction' else settings['state_lengths']
    training = select(inputs,train)
    for length in lengths:
        gram = kernel(training,training,inputs['bandwidth'],length)
        evaluation = kernel(select(inputs,selection),training,inputs['bandwidth'],length)
        for penalty in settings['penalties']:
            weights = torch.linalg.solve(gram+len(train)*penalty*torch.eye(len(train),dtype=gram.dtype),y[train])
            prediction = (evaluation@weights).reshape(len(selection),*target.shape[1:])
            residual = prediction-y[selection].reshape_as(prediction)
            nll = .5*(residual[:,1:].square()+2*target_scale[None,1:,None].log()+np.log(2*np.pi)).mean()
            candidate = dict(kind=kind,length=length,penalty=penalty,selection_nll=float(nll))
            candidates.append(candidate)
            if best is None or float(nll) < best['selection_nll']:
                best = dict(candidate,weights=weights,target_scale=target_scale,
                            training=training,bandwidth=inputs['bandwidth'],
                            input_mean=inputs['mean'],input_scale=inputs['scale'],direction_rms=inputs['direction_rms'])
    return best,candidates


def predict(fit, inputs):
    matrix = kernel(inputs,fit['training'],fit['bandwidth'],fit['length'])
    y = (matrix@fit['weights']).reshape(len(inputs['h']),len(fit['target_scale']),128)
    return y*fit['target_scale'][None,:,None]


def score_rows(c, prediction, target, records, target_scale, **context):
    times = [0]+list(range(5,101,5)); rows = []
    mse = (prediction-target).square().mean(-1)
    baseline = target.square().mean(-1)
    nll = .5*(mse/target_scale.square()+2*target_scale.log()+np.log(2*np.pi))
    for index,time_fs in enumerate(times):
        for metric,values in [('mse',mse),('zero_mse',baseline),('fixed_variance_nll',nll)]:
            for summary in weighted_summary(values[:,index].numpy(),records,c['seed']):
                rows.append(dict(context,horizon_fs=time_fs,metric=metric,**summary))
    return rows


def run(c):
    from .diagnostics import selected, model_for, export
    p = selected(c); source = read(Path(c['source_run'])/'technical/plan.json')
    teachers = bank(p['source_config'],source)
    normalizer = root(c)/'technical/normalizer.pt'
    if sha(normalizer) != p['normalizer_sha256']:
        raise ValueError('Changed probe observation normalizer')
    normal = torch.load(normalizer,map_location='cpu',weights_only=False)
    mean,scale = normal['mean'],normal['scale']
    history = (torch.stack([r['history'] for r in teachers])-mean)/scale
    pairs = (torch.stack([r['pairs'] for r in teachers])-mean)/scale
    epsilon = p['perturbation']['epsilon']['response_epsilon']
    target = torch.stack([torch.cat((r['initial_response'][None],r['response']),0) for r in teachers])/scale[:128,None]
    records = [dict(id=r['job'],ancestor=r['ancestor'],role=r['role'],regime='physical_anchor') for r in teachers]
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate physical anchor in response probe')
    roles = np.array([r['role'] for r in records])
    train,selection,test = [np.flatnonzero(roles==role) for role in ['train','selection','test']]
    heldout = [records[i] for i in test]
    representations = [('raw280',history,(pairs[:,:,1]-pairs[:,:,0])/(2*epsilon))]
    with torch.no_grad():
        for spec in p['probe_checkpoints']:
            model = model_for(c,spec,device='cpu')
            h = model.encode(history)
            encoded_pairs = model.encode(pairs)
            directions = (encoded_pairs[:,:,1]-encoded_pairs[:,:,0])/(2*epsilon)
            representations.append((spec['arm']['name']+'-code16',h,directions))
    write_json(root(c)/'analyses/response-predictability/technical/data.json',dict(
        source_identity=source['identity'],rows=records,
        counts={role:int(np.sum(roles==role)) for role in ['train','selection','test']},
        ancestors={role:len({r['ancestor'] for r in records if r['role']==role}) for role in ['train','selection','test']},
        directions='separate Q/V queries; no additional independent source counts',
        target='saved two-shot average geometry JVPs, not synthetic encoder targets',
        query='last observation slot secant; measured older slots unchanged',
        time='twenty output slots at5..100fs; t0 is a separate present-response diagnostic; no time input'))
    # Baselines are common across representation comparisons; exact initial
    # geometry JVP is explicitly an oracle present-response control.
    baselines = []
    for direction,label in enumerate(['Q','V']):
        yy = target[...,direction].double()
        target_scale = yy[train].square().mean((0,2)).sqrt().clamp_min(1e-8)
        for name,forecast in [('zero',torch.zeros_like(yy)),
                              ('present_geometry_jvp',yy[:,:1].expand_as(yy)),
                              ('present_geometry_secant',representations[0][2][:,direction,None,:128].expand_as(yy))]:
            baselines.extend(score_rows(c,forecast[test],yy[test],heldout,target_scale,
                                        representation=name,readout='baseline',direction=label))
    export(c,'response-predictability-baselines',baselines)
    for name,h,directions in representations:
        all_rows = []; candidate_rows = []
        for direction,label in enumerate(['Q','V']):
            inputs = features(h,directions[:,direction],train)
            yy = target[...,direction].double()
            for kind in ['linear_direction','state_conditioned']:
                folder = root(c)/'analyses'/('response-predictability-'+name)
                fit_path = folder/'technical'/f'{label}-{kind}.pt'
                receipt = fit_path.with_suffix('.json')
                if receipt.exists():
                    if sha(fit_path) != read(receipt)['sha256']:
                        raise ValueError('Changed saved response readout')
                    saved = torch.load(fit_path,map_location='cpu',weights_only=False)
                    fitted,candidates = saved['fit'],saved['candidates']
                else:
                    fitted,candidates = fit(inputs,yy,train,selection,kind,c['probe'])
                    atomic_save(fit_path,dict(fit=fitted,candidates=candidates))
                    write_json(receipt,dict(sha256=sha(fit_path),plan=p['identity']))
                prediction = predict(fitted,select(inputs,test))
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError('Nonfinite response readout: '+name+' '+label)
                atomic_save(folder/'technical'/f'{label}-{kind}-predictions.pt',dict(
                    rows=heldout,prediction=prediction,target=yy[test],target_scale=fitted['target_scale']))
                all_rows.extend(score_rows(c,prediction,yy[test],heldout,fitted['target_scale'],
                                           representation=name,readout=kind,direction=label))
                candidate_rows.extend(dict(representation=name,direction=label,**r) for r in candidates)
                write_json(root(c)/'technical/predictability-progress.json',dict(
                    representation=name,direction=label,readout=kind,selection_nll=fitted['selection_nll']))
                print('response readout',name,label,kind,fitted['selection_nll'],flush=True)
        export(c,'response-predictability-'+name,all_rows)
        export(c,'response-predictability-selection-'+name,candidate_rows)


def check():
    """A complete small fit plus exact directional-linearity/PSD checks."""
    g = torch.Generator().manual_seed(20261011)
    history = torch.randn(120,4,8,generator=g,dtype=torch.float64)
    directions = torch.randn(120,8,generator=g,dtype=torch.float64)
    coefficients = torch.randn(8,21*128,generator=g,dtype=torch.float64)
    target = (directions@coefficients).reshape(120,21,128)
    train,selection,test = np.arange(80),np.arange(80,100),np.arange(100,120)
    inputs = features(history,directions,train)
    fitted,_ = fit(inputs,target,train,selection,'linear_direction',dict(penalties=[1e-8],state_lengths=[1.]))
    prediction = predict(fitted,select(inputs,test))
    relative = float((prediction-target[test]).square().mean()/target[test].square().mean())
    a = select(inputs,test); b = select(inputs,train)
    neg = dict(h=a['h'],d=-a['d'])
    gram = kernel(b,b,inputs['bandwidth'],1.)
    odd_error = float((kernel(a,b,inputs['bandwidth'],1.)+kernel(neg,b,inputs['bandwidth'],1.)).abs().max())
    eigen_min = float(torch.linalg.eigvalsh(gram).min())
    if relative > 1e-10 or odd_error > 1e-12 or eigen_min < -1e-10:
        raise ValueError(dict(relative_mse=relative,odd_error=odd_error,minimum_eigenvalue=eigen_min))
    return dict(known_linear_response_relative_mse=relative,oddness_error=odd_error,
                conditional_kernel_minimum_eigenvalue=eigen_min)
