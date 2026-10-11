"""Matched Q/QV fits with predictive selectors and explicit information controls."""
import math
import time
import numpy as np
import torch
from src.experiment_runner.checkpoints import TrainingState
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from src.research.response_geometry.train import distribution_distances
from .common import FAMILY,root,read,save,sha,digest,write_json,load,deadline
from .model import Predictor,initialize,derivatives


def inputs(records,device='cuda'):
    x=torch.stack([torch.cat((r['position'],r['velocity']),-1)for r in records]).float().to(device)
    basis=torch.stack([torch.cat((r['basis'][0],r['basis'][1]),1)for r in records]).float().to(device)
    return x,basis


def fit(c,arm,seed):
    torch.set_num_threads(2);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    name=arm['name']+'-'+str(seed);out=root(c)/'analyses'/name;tech=out/'technical';tech.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(run=read(root(c)/'technical/binding.json')['identity'],arm=arm,seed=seed))
    if (tech/'complete.json').exists():
        if read(tech/'complete.json')['identity']!=identity:raise ValueError('Changed fit identity')
        return
    data=load(c);records=data['records'];cfg=c['training'];micro=cfg['microbatch'];batch=cfg['batch_size']
    ids={role:np.array([i for i,r in enumerate(records)if r['role']==role])for role in ['train','selection','test']}
    train,selection,test=[ids[r]for r in ['train','selection','test']]
    counts=c.get('parent_counts',[90,15,30])
    if [len(train),len(selection),len(test)]!=[n*c['velocity_fields'] for n in counts]:raise ValueError('Velocity cohort changed')
    x,basis=inputs(records);center=data['center'].float().cuda();scale=data['scale'].float().cuda()
    hscale=data['response_scale'].float().cuda().repeat_interleave(2)
    targets=torch.stack([r['values'].mean(0)for r in records]).float().cuda()
    target=(targets-center)/scale;response=torch.zeros(len(records),len(center),4,device='cuda')
    for i in np.r_[train,test]:response[i]=records[i]['responses'].mean(0).float().cuda()/scale[:,None]
    values=torch.stack([records[i]['values']for i in train]).float().cuda()
    if arm['velocities'] or c.get('initial_velocity')=='saved_md':
        # With one measured V per geometry, both arms target the same conditional
        # future bank. A velocity-marginal variance is not estimable from one V.
        distance,dscale=distribution_distances(values)
        lookup=torch.arange(len(train),device='cuda')
    else:
        # Independent velocity-field means estimate geometry-marginal future distance.
        bank=values.mean(1).reshape(counts[0],c['velocity_fields'],-1)
        distance,dscale=distribution_distances(bank)
        lookup=torch.arange(counts[0],device='cuda').repeat_interleave(c['velocity_fields'])
    model=Predictor(c,arm['velocities']).cuda()
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=cfg['encoder_lr']),
        dict(params=model.head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed);history=[];best=math.inf;selected=0;start=1;elapsed=0.
    if (tech/'last.pt').exists():
        st=TrainingState.read(tech/'last.pt',identity=identity,device='cuda')
        st.restore(model,optimizer,rng,restore_torch_rng=True);p=st.payload
        history=p['history'];best=p['best'];selected=p['selected'];start=p['update']+1;elapsed=p['elapsed'];zscale=p['zscale']
    else:
        initialize(model,x[train],micro)
        with torch.no_grad():z=torch.cat([model.encode(b)for b in x[train].split(micro)])
        zscale=float((2*z.var(0,unbiased=False).mean()).clamp_min(1e-8))
    @torch.no_grad()
    def predict(ix):return torch.cat([model(b)for b in x[ix].split(micro)])
    def select():return float(.5*((predict(selection)-target[selection]).square().mean()+math.log(2*math.pi)))
    context=dict(identity=identity,arm=arm,seed=seed,training=cfg,batch_size=batch,microbatch=micro,
        selector='validation conditional future-feature Gaussian NLL, equal horizon weights, every5 updates',
        inputs=read(root(c)/'technical/prediction-context.json'),parameters=sum(p.numel()for p in model.parameters()),
        distance_scale=float(dscale),embedding_distance_scale=zscale)
    write_json(tech/'identity.json',context);stop_at=deadline(c,reserve_seconds=300)
    with online_training(c['wandb'],run_id='resp-velocity-'+identity[:14],name=name,config=context,folder=tech,
        receipt_path=tech/'wandb.json',job_type='encoder',group=c['protocol'],receipt_fields=dict(identity=identity))as run:
        if start==1:
            model.eval();best=select();save(tech/'best.pt',dict(identity=identity,model=model.state_dict(),update=0,selection_nll=best))
        for update in range(start,cfg['updates']+1):
            if time.time()>=stop_at:raise TimeoutError('Fit time budget; resume checkpoint retained')
            begin=time.monotonic();order=rng.choice(train,batch,replace=cfg.get('sample_with_replacement',False));optimizer.zero_grad(set_to_none=True)
            model.train();totals=np.zeros(3)
            for offset in range(0,batch,micro):
                ix=order[offset:offset+micro];z=model.encode(x[ix]);prediction=model.head(z)
                lv=.5*(prediction-target[ix]).square().mean();lr=lv.new_zeros(())
                if arm['response']:
                    directions=4 if arm['velocities']else 2
                    h=derivatives(model,x[ix],basis[ix,...,:directions],create_graph=True)
                    lr=.5*((h-response[ix,:,:directions])/hscale[:directions]).square().mean()
                local=lookup[ix];keep=local[:,None]!=local[None,:]
                dz=(z[:,None]-z[None,:]).square().mean(-1)/zscale
                # Duplicate geometries are excluded from the Q-only distance objective.
                lm=.5*(dz[keep]-distance[local[:,None],local[None,:]][keep]).square().mean()
                loss=(lv+float(arm['response'])*lr+c['metric_weight']*lm)*len(ix)/batch
                if not torch.isfinite(loss):raise FloatingPointError(f'{name} update{update} nonfinite objective')
                loss.backward();totals+=np.array([float(lv.detach()),float(lr.detach()),float(lm.detach())])*len(ix)/batch
            gradient=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True));optimizer.step();model.eval()
            row=dict(update=update,value_nll=float(totals[0]+.5*math.log(2*math.pi)),distance_half_mse=float(totals[2]),gradient_norm=gradient)
            if arm['response']:row['response_half_mse']=float(totals[1])
            if update%5==0:
                value=select();row['selection_nll']=value
                if value<best:
                    best=value;selected=update;save(tech/'best.pt',dict(identity=identity,model=model.state_dict(),update=update,selection_nll=best))
            torch.cuda.synchronize();elapsed+=time.monotonic()-begin;history.append(row)
            TrainingState.capture(model,optimizer,rng,identity=identity,capture_torch_rng=True,history=history,best=best,
                selected=selected,update=update,elapsed=elapsed,zscale=zscale).save(tech/'last.pt')
            run.log(row,step=update)
            if update==1 or update%25==0:print(name,row,flush=True)
        saved=torch.load(tech/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model']);model.eval()
        prediction=predict(test);h=torch.cat([derivatives(model,x[ix],basis[ix]).detach()for ix in np.array_split(test,max(1,math.ceil(len(test)/micro)))])
        shots=torch.stack([records[i]['values']for i in test]).float().cuda()/scale
        hs=torch.stack([records[i]['responses']for i in test]).float().cuda()/scale[None,None,:,None]
        free=torch.stack([records[i]['force_free']for i in test]).float().cuda()/scale
        value_noise=shots.var(1,unbiased=True)/shots.shape[1];response_noise=hs.var(1,unbiased=True)/hs.shape[1]
        # Shuffling preserves local speed vectors but destroys their atom assignment.
        shuffled=x[test].clone();perm=np.random.default_rng(c['audit_seed']).permutation(79)+1
        shuffled[:,1:,3:]=shuffled[:,perm,3:]
        with torch.no_grad():permuted=torch.cat([model(b)for b in shuffled.split(micro)])
        rows=[]
        for k,horizon in enumerate(c['horizons_fs']):
            s=slice(k*128,(k+1)*128);err=(prediction[:,s]-target[test,s]).square()
            free_delta=free[:,:,s]-shots[:,:,s]
            r=dict(horizon_fs=horizon,selected_update=selected,selection_nll=best,
                test_nll=float(.5*(err.mean()+math.log(2*math.pi))),
                value_corrected_mse=float((err-value_noise[:,s]).mean()),
                position_response_corrected_mse=float(((h[:,s,:2]-response[test,s,:2]).square()-response_noise[:,s,:2]).mean()),
                force_free_corrected_mse=float((free_delta.mean(1).square()-free_delta.var(1,unbiased=True)/free_delta.shape[1]).mean()),
                shuffled_velocity_corrected_mse=float(((permuted[:,s]-target[test,s]).square()-value_noise[:,s]).mean()))
            if arm['velocities']:
                r['velocity_response_corrected_mse']=float(((h[:,s,2:]-response[test,s,2:]).square()-response_noise[:,s,2:]).mean())
            rows.append(r)
        np.savez_compressed(tech/'predictions.npz',prediction=prediction.cpu().numpy(),response=h.cpu().numpy(),
            shuffled_prediction=permuted.cpu().numpy(),target=target[test].cpu().numpy(),
            source=np.array([records[i]['source']for i in test]),velocity_field=np.array([records[i]['field']for i in test]))
        summary={f'test/{r["horizon_fs"]}fs/{k}':v for r in rows for k,v in r.items()if k not in ['horizon_fs','selected_update','selection_nll']}
        run.summary.update(dict(summary,training_seconds=elapsed,selected_update=selected))
    columns=['update','value_nll','distance_half_mse','gradient_norm']+(['response_half_mse']if arm['response']else [])+['selection_nll']
    write_metric_rows(history,out,family=FAMILY,name='learning',columns=columns)
    write_metric_rows(rows,out,family=FAMILY,name='predictive')
    write_json(tech/'complete.json',dict(state='complete',identity=identity,checkpoint_sha256=sha(tech/'best.pt'),metrics=rows))
