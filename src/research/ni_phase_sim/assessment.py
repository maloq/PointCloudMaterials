"""Independent free rollouts, tangent transport and paired ancestor diagnostics."""
from collections import defaultdict
import time
import numpy as np
import torch
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import update_recorded_summary
from src.research.measurement_sim.features import SmoothMeasurements,structural
from .data import root,read,load,sha,atomic_save,write_json,FAMILY
from .model import PhaseProcess,ballistic


def measurements(state):
    scale=state.new_tensor([5.,5.,5.,10.,10.,10.])
    with torch.no_grad(): return SmoothMeasurements().to(state.device)(state*scale)


def measure_errors(prediction,target,predicted_response,true_response,norm):
    """Per-anchor errors; physical units and train-scaled direction magnitudes."""
    n,t=prediction.shape[:2]; scale=prediction.new_tensor([5.,10.])
    error=(prediction-target).reshape(n,t,80,2,3)
    metrics={}
    for name,stop in [('all79',80),('inner16',17)]:
        squared=error[:,:,1:stop].square().mean((2,4))*scale.square()
        metrics[name+'/q_mse_A2']=squared[...,0]
        metrics[name+'/v_mse_A2_ps2']=squared[...,1]
        de=(predicted_response-true_response).reshape(n,t,80,2,3,2)
        mse=de[:,:,1:stop].square().mean((2,4))
        power=norm['response_power'][None]
        for k,query in enumerate(['Q','V']):
            for a,output in enumerate(['q','v']):
                metrics[f'{name}/response_{query}_{output}']=mse[...,a,k]/power[...,a,k]
    q=prediction[...,:3]*5
    distance=(q[:,:,:,None]-q[:,:,None,:]).square().sum(-1)
    distance=distance+torch.eye(80,device=q.device)[None,None]*1e6
    metrics['minimum_pair_distance_A']=distance.amin((-1,-2)).sqrt()
    metrics['overlap_fraction']= (distance.amin((-1,-2))<1.).float()
    return metrics


def aggregate(metrics,records,model):
    rows=[]; ancestors=sorted({r['ancestor'] for r in records})
    for key,value in metrics.items():
        for time_index in [0,3,9,19]:
            per_source=[float(value[[i for i,r in enumerate(records) if r['ancestor']==a],time_index].mean()) for a in ancestors]
            rows.append(dict(model=model,metric=key,horizon_fs=(time_index+1)*5,value=float(np.mean(per_source)),
                             states=len(records),ancestors=len(ancestors)))
    return rows


def figures(folder,prediction,state,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    destination=folder.parent/'plots/trajectories'; destination.mkdir(parents=True,exist_ok=True)
    times=np.arange(21)*5
    for i in range(min(4,len(state))):
        figure,axes=plt.subplots(2,2,figsize=(11,7),sharex=True)
        truth=state[i].cpu().numpy(); pred=torch.cat((state[i,:1],prediction[i]),0).cpu().numpy()
        base=torch.cat((state[i,:1],ballistic(state[i:i+1,0],20)[0]),0).cpu().numpy()
        for column,atom in enumerate([1,8]):
            for coordinate,unit,scale,offset in [('x','Å',5,0),('v_x','Å/ps',10,3)]:
                ax=axes[int(offset>0),column]
                ax.plot(times,truth[:,atom,offset]*scale,label='full-cell NVE',color='black')
                ax.plot(times,pred[:,atom,offset]*scale,label='neural rollout')
                ax.plot(times,base[:,atom,offset]*scale,label='ballistic',linestyle='--')
                ax.set_ylabel(f'{coordinate} ({unit})'); ax.set_title(f'Fixed atom ID {records[i]["atom_ids"][atom]}')
                ax.grid(alpha=.2)
        axes[0,0].legend(); axes[1,0].set_xlabel('Elapsed time (fs)'); axes[1,1].set_xlabel('Elapsed time (fs)')
        figure.suptitle('Predetermined example '+records[i]['job']['key']+'; center-relative coordinates')
        figure.tight_layout(); figure.savefig(destination/f'example-{i}.png',dpi=140); plt.close(figure)


def run(c,arm,deadline):
    folder=root(c)/'analyses'/arm['name']/'technical'
    if (folder/'evaluation-complete.json').exists(): return True
    saved=torch.load(folder/'final.pt',map_location='cuda',weights_only=False)
    model=PhaseProcess(c,arm).cuda().eval(); model.load_state_dict(saved['model'])
    plan,records,state,tangent=load(c,'test'); norm={k:v.cuda() for k,v in saved['normalization'].items()}
    predictions=[]; derivatives=[]; latent_metrics=[]; size=c['evaluation']['batch_size']
    # Diagnostic evaluation of both directions; no fit or checkpoint selection.
    for offset in range(0,len(state),size):
        if time.time()>deadline-120: return False
        x=state[offset:offset+size,0]; responses=[]
        with torch.no_grad():
            prediction=model(x,20); predictions.append(prediction.cpu())
            for k in range(2):
                direction=tangent[offset:offset+size,0,...,k]
                _,d=torch.func.jvp(lambda value:model(value,20),(x,),(direction,))
                responses.append(d.cpu())
            derivatives.append(torch.stack(responses,-1))
            if model.latent:
                h=state[offset:offset+size,:-1].flatten(0,1)
                future=state[offset:offset+size,1:].flatten(0,1)
                before=tangent[offset:offset+size,:-1,...,0].flatten(0,1)
                after=tangent[offset:offset+size,1:,...,0].flatten(0,1)
                _,left=torch.func.jvp(lambda value:model.step(model.encode(value)),(h,),(before,))
                _,right=torch.func.jvp(model.encode,(future,),(after,))
                dims=tuple(range(1,left.ndim))
                latent_metrics.extend([dict(job=records[offset+i//20]['job']['key'],horizon_fs=(i%20+1)*5,
                    response_mse=float(a),target_power=float(b)) for i,(a,b) in enumerate(zip(
                        (left-right).square().mean(dims),right.square().mean(dims)))])
    prediction=torch.cat(predictions).cuda(); response=torch.cat(derivatives).cuda()
    actual=state[:,1:]; true_response=tangent[:,1:]
    baseline=ballistic(state[:,0],20)
    times=torch.arange(1,21,device='cuda')*.01
    ballistic_response=tangent[:,None,0].expand(-1,20,-1,-1,-1).clone()
    ballistic_response[...,:3,:]+=times[None,:,None,None,None]*tangent[:,None,0,:,3:,:]
    rows=[]; all_metrics={}
    for name,value,derivative in [('neural',prediction,response),('ballistic',baseline,ballistic_response),
        ('persistence',state[:,:1].expand(-1,20,-1,-1),tangent[:,:1].expand(-1,20,-1,-1,-1))]:
        metrics=measure_errors(value,actual,derivative,true_response,norm)
        all_metrics[name]={k:v.cpu() for k,v in metrics.items()}
        rows.extend(aggregate(metrics,records,name))
    # Explicit zero-response comparator uses the identical normalizers.
    zero=measure_errors(actual,actual,torch.zeros_like(response),true_response,norm)
    zero={k:v for k,v in zero.items() if '/response_' in k}
    all_metrics['zero_response']={k:v.cpu() for k,v in zero.items()}
    rows.extend(aggregate(zero,records,'zero_response'))
    # Geometric measurement predictions use the decoded atomic state directly.
    _,train_records,train_state,_=load(c,'train')
    measured_train=measurements(train_state[:,0]); mean=measured_train.mean(0); scale=measured_train.std(0).clamp_min(1e-5)
    measurement_rows=[]
    for index in [0,3,9,19]:
        truth=measurements(actual[:,index])
        for name,values in [('neural',prediction),('ballistic',baseline)]:
            estimated=measurements(values[:,index])
            error=((estimated-truth)/scale).square()
            for group,lo,hi in [('density',0,74),('velocity_shell',74,83),('bond_rates',83,87),('flow',87,91)]:
                per=error[:,lo:hi].mean(-1)
                ancestors=sorted({r['ancestor'] for r in records})
                values_by_source=[float(per[[i for i,r in enumerate(records) if r['ancestor']==a]].mean()) for a in ancestors]
                measurement_rows.append(dict(model=name,horizon_fs=(index+1)*5,group=group,
                    normalized_mse=float(np.mean(values_by_source)),states=len(records),ancestors=len(ancestors)))
    # Historical BO/TDA at100fs: original fixed columns, decoded atomic geometry.
    names=plan['columns'][:153]; factor=c['length_factor']
    reference=structural((train_state[:,0,:,:3]*5).cpu().numpy(),names,factor)
    truth=structural((actual[:,-1,:,:3]*5).cpu().numpy(),names,factor)
    structural_scale=reference.std(0); active=structural_scale>1e-8
    for name,values in [('neural',prediction),('ballistic',baseline)]:
        estimated=structural((values[:,-1,:,:3]*5).cpu().numpy(),names,factor)
        for group,lo,hi in [('bond_order',0,40),('tda',40,153)]:
            columns=np.arange(lo,hi)[active[lo:hi]]
            per=(((estimated[:,columns]-truth[:,columns])/structural_scale[columns])**2).mean(-1)
            ancestors=sorted({r['ancestor'] for r in records})
            values_by_source=[per[[i for i,r in enumerate(records) if r['ancestor']==a]].mean() for a in ancestors]
            measurement_rows.append(dict(model=name,horizon_fs=100,group=group,
                normalized_mse=float(np.mean(values_by_source)),states=len(records),ancestors=len(ancestors)))
    # Numerical derivative audit, kept separate from scientific performance.
    x=state[:4,0]; direction=tangent[:4,0,...,0]
    with torch.no_grad():
        _,jvp=torch.func.jvp(lambda value:model(value,20),(x,),(direction,))
        secant=(model(x+.1*direction,20)-model(x-.1*direction,20))/.2
        numerical=float((jvp-secant).norm()/jvp.norm().clamp_min(1e-8))
        encoded=model.encode(state[:,0]).flatten(1)
        latent_std=float(encoded.std(0).mean())
    atomic_save(folder/'predictions.pt',dict(prediction=prediction.cpu(),response=response.cpu(),
        state=state.cpu(),tangent=tangent.cpu(),metrics=all_metrics,
        rows=[dict(job=r['job']['key'],source=r['source'],ancestor=r['ancestor'],atom_ids=r['atom_ids']) for r in records]))
    write_metric_rows(rows,folder.parent,family=FAMILY,name='prediction')
    write_metric_rows(measurement_rows,folder.parent,family=FAMILY,name='measurements')
    if latent_metrics: write_metric_rows(latent_metrics,folder.parent,family=FAMILY,name='latent_transport')
    figures(folder,prediction,state,records)
    summary={f'evaluation/{r["metric"]}/{r["horizon_fs"]}fs':r['value'] for r in rows if r['model']=='neural'}
    summary.update({'evaluation/jvp_secant_error':numerical,'evaluation/latent_coordinate_std':latent_std})
    write_json(folder/'evaluation-metrics.json',summary)
    # The scientific evaluation is durable before the optional API update.
    write_json(folder/'evaluation-complete.json',dict(checkpoint_sha256=sha(folder/'final.pt'),states=len(records),
        ancestors=len({r['ancestor'] for r in records}),jvp_secant_error=numerical))
    update_recorded_summary(folder/'wandb.json',summary,evaluation='phase-space',expected=dict(identity=saved['identity']))
    return True


def compare(c):
    """Within-seed model differences, bootstrapping independent source ancestors."""
    complete={}
    for arm in c['arms']:
        folder=root(c)/'analyses'/arm['name']/'technical'
        if (folder/'evaluation-complete.json').exists():
            complete[arm['name']]=torch.load(folder/'predictions.pt',map_location='cpu',weights_only=False)
    rows=[]; rng=np.random.default_rng(c['seed'])
    for seed in sorted({a['seed'] for a in c['arms']}):
        for a,b in [('latent','direct'),('latent-physical','latent'),('latent-transport','latent-physical')]:
            left=complete.get(f'{a}-s{seed}'); right=complete.get(f'{b}-s{seed}')
            if left is None or right is None: continue
            if [r['job'] for r in left['rows']] != [r['job'] for r in right['rows']]:
                raise ValueError('Unpaired phase-space comparison')
            sources=sorted({r['ancestor'] for r in left['rows']})
            for key in left['metrics']['neural']:
                delta=left['metrics']['neural'][key]-right['metrics']['neural'][key]
                per=np.stack([delta[[i for i,r in enumerate(left['rows']) if r['ancestor']==s]].mean(0).numpy() for s in sources])
                sample=rng.integers(len(sources),size=(c['evaluation']['bootstrap'],len(sources)))
                boot=per[sample].mean(1)
                for index in [0,3,9,19]:
                    lower,upper=np.quantile(boot[:,index],[.025,.975])
                    rows.append(dict(left=a,right=b,seed=seed,metric=key,horizon_fs=(index+1)*5,
                        difference=float(per[:,index].mean()),lower95=float(lower),upper95=float(upper),ancestors=len(sources)))
    if rows: write_metric_rows(rows,root(c)/'analyses/comparison',family=FAMILY,name='paired')
    write_json(root(c)/'technical/summary.json',dict(evaluated=list(complete),planned=[a['name'] for a in c['arms']],updated=time.time()))
