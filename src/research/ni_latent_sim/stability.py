"""Common-noise finite-time Jacobian control of the learned stochastic map.

This is a local regularizer, not a proof of global stability or a physical
Lyapunov exponent. Simulator-response matching remains a separate factor.
"""
import csv
import math
import numpy as np
import torch
from .data import read, root, cache, sha, digest, write_json, FAMILY
from .artifacts import write_rows as write_metric_rows, table_path, export_root


def unit(v):
    return v / v.flatten(1).norm(dim=1).clamp_min(1e-12).view(-1,1,1)


def advance(model, history, noises):
    for noise in noises:
        history = model.step(history, noise)
    return history


def gain(model, history, noises, direction, iterations, create_graph):
    """Power iteration on J^T J; differentiate the final JVP, not the search."""
    fn = lambda h: advance(model, h, noises)
    direction = unit(direction).detach()
    for _ in range(iterations):
        _, jv = torch.autograd.functional.jvp(fn, history, direction)
        _, jt = torch.autograd.functional.vjp(fn, history, v=unit(jv))
        direction = unit(jt).detach()
    _, jv = torch.autograd.functional.jvp(fn, history, direction,
                                        create_graph=create_graph)
    return jv.flatten(1).norm(dim=1)


def penalty(model, inputs, settings, seed):
    generator = torch.Generator(device=inputs.device).manual_seed(seed)
    batch = settings['batch_size']
    if batch > len(inputs) or batch % 2:raise ValueError('Jacobian batch must be even and fit the training batch')
    with torch.no_grad():
        h = model.encode(inputs[:batch,-model.memory:]).detach()
        # Include states visited by autonomous dynamics, where the earlier
        # extrapolation failed, without introducing held-out observations.
        start = settings['free_state_steps'][seed % len(settings['free_state_steps'])]
        noises = torch.randn(start,batch//2,model.d,device=h.device,generator=generator)
        h[batch//2:] = advance(model,h[batch//2:],noises)
    noises = torch.randn(settings['map_steps'],batch,model.d,device=h.device,generator=generator)
    direction = torch.randn(h.shape,device=h.device,generator=generator)
    gains = gain(model,h,noises,direction,settings['power_iterations'],True)
    loss = torch.relu(torch.log(gains.clamp_min(1e-12))-math.log(settings['gain_cap'])).square().mean()
    return loss,dict(jacobian_penalty=float(loss.detach()),
                     jacobian_gain_mean=float(gains.detach().mean()),
                     jacobian_gain_max=float(gains.detach().max()))


def prepare(c):
    previous = root(dict(output=c['reuse_data']['output']))/'technical/plan.json'
    if sha(previous)!=c['reuse_data']['plan_sha256']:raise ValueError('Changed reused latent data plan')
    old=read(previous)
    receipt=cache(c)/'data.json'
    if sha(receipt)!=c['reuse_data']['receipt_sha256']:raise ValueError('Changed reused sealed data receipt')
    sealed=read(receipt)
    if sealed['identity']!=old['identity']:raise ValueError('Reused data and source plan differ')
    for name,expected in sealed['files'].items():
        if sha(cache(c)/(name+'.pt'))!=expected:raise ValueError('Changed reused data: '+name)
    plan=dict(config=c,parent_identity=old['parent_identity'],
              reused_plan_identity=old['identity'],jobs=old['jobs'],
              partition=old['partition'],sources=old['sources'])
    plan['identity']=digest(plan)
    path=root(c)/'technical/plan.json'
    if path.exists() and read(path)!=plan:raise ValueError('Changed stability experiment; use a new revision')
    write_json(path,plan)
    context=read(previous.parent/'prediction-context.json')
    context.update(predictor='four 16-dimensional codes at exact 0.1ps offsets; identical frozen 280-coordinate inputs; no explicit conditions',
                   stability='train-only Jacobians of learned 1ps common-noise map at real and freely evolved latent states; not new MD teachers')
    write_json(root(c)/'technical/prediction-context.json',context)
    return plan


def report(c):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=export_root(root(c)/'analyses/stability-comparison-v1');roll=[];birth=[];short=[];regimes=[]
    tables=[(roll,'free_rollout'),(birth,'nucleus_prediction'),(short,'heldout_physical_response')]
    if 'enrichment' in c:tables.append((regimes,'free_rollout_by_regime'))
    for arm in c['arms']:
        for target,name in tables:
            path=table_path(root(c)/'analyses'/arm['name'],name)
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    target.append(dict(row,train_steps=arm['train_steps'],jacobian=arm['jacobian'],response=arm['response'],seed=arm['seed']))
    for values,name in tables:
        write_metric_rows(values,out,family=FAMILY,name=name,columns=list(values[0]))
    plots=out/'plots';plots.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,steps in enumerate([10,60]):
        for col,metric in enumerate(['mse','particle_failure_rate']):
            ax=axes[row,col]
            for response in [False,True]:
                for jacobian in [False,True]:
                    selected=[r for r in roll if r['space']=='structure' and r['metric']==metric and r['train_steps']==steps and r['response']==response and r['jacobian']==jacobian]
                    times=sorted({float(r['horizon_ps']) for r in selected})
                    scores={(r['seed'],float(r['horizon_ps'])):float(r['value']) for r in selected}
                    values=np.array([[scores[s,t] for t in times] for s in c['seeds']])
                    label=('Physical response' if response else 'Values')+(' + Jacobian' if jacobian else '')
                    ax.plot(times,values.mean(0),'o-',label=label)
                    ax.fill_between(times,values.min(0),values.max(0),alpha=.12)
            ax.set_title(f'Training to {steps*.1:g} ps');ax.set_xlabel('Autonomous rollout (ps)')
            ax.set_ylabel('Standardized structural MSE' if metric=='mse' else 'Fraction of failing particles')
            if metric=='mse':ax.set_yscale('log')
            ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Two seeds; shading is seed range, not uncertainty interval')
    fig.tight_layout();fig.savefig(plots/'stability.png',dpi=170);fig.savefig(plots/'stability.pdf');plt.close(fig)
    write_json(out/'technical/complete.json',dict(state='complete',models=len(c['arms']),source='frozen score tables'))
