"""Matched predictive objectives and simulator-distribution distance supervision."""
import math
import time
import numpy as np
import torch
from src.experiment_runner.checkpoints import TrainingState
from src.experiment_runner.execution import allocation_deadline
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import online_training,update_recorded_summary
from src.research.local_response.model import Predictor,initialize,responses
from .common import FAMILY,root,read,sha,digest,save,write_json,legacy,load


def distribution_distances(values):
    """Unbiased squared mean-feature distance between independent shot banks.

    Off-diagonal pair estimates may be negative; retain them, without a positive
    clipping bias. Repeated parent IDs are excluded from the distance loss.
    """
    mean=values.mean(1);b=values.shape[1]
    noise=values.var(1,unbiased=True).sum(-1)/b
    d=(mean[:,None]-mean[None,:]).square().sum(-1)-noise[:,None]-noise[None,:]
    keep=~torch.eye(len(mean),dtype=torch.bool,device=values.device)
    scale=d[keep].mean()
    if not torch.isfinite(scale) or scale<=0:raise ValueError('Future-distribution signal below estimable noise floor')
    return d/scale,scale


def fit(c,arm,seed):
    torch.set_num_threads(2);torch.set_default_dtype(torch.float32);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    name=arm['name']+'-'+str(seed);out=root(c)/'analyses'/name;tech=out/'technical';tech.mkdir(parents=True,exist_ok=True)
    identity=digest(dict(run=read(root(c)/'technical/binding.json')['identity'],arm=arm,seed=seed))
    if (tech/'complete.json').exists():
        if read(tech/'complete.json')['identity']!=identity:raise ValueError('Changed fit identity')
        return
    data=load(c);records=data['parents'];cfg=c['training'];micro=cfg['microbatch'];batch=cfg['batch_size']
    train=np.array([i for i,r in enumerate(records)if r['role']=='train' and
        (arm['parents']==720 or r['original_index'] is not None)])
    selection=np.array([i for i,r in enumerate(records)if r['role']=='selection'])
    test=np.array([i for i,r in enumerate(records)if r['role']=='test'])
    if (len(train),len(selection),len(test))!=(arm['parents'],15,30):raise ValueError('Changed fit cohort')
    block={'legacy_delta':0,'rich_delta':1,'rich_joint':2}[arm['target']];s=slice(block*256,(block+1)*256)
    scale=data['scale'][s].cuda();center=data['center'][s].cuda();hscale=data['response_scale'][block].cuda()
    q=torch.stack([r['q']for r in records]).cuda();v=torch.stack([r['basis']for r in records]).cuda()
    target=torch.stack([r['values'][:,s].mean(0)for r in records]).cuda();target=(target-center)/scale
    ht=torch.zeros(len(records),256,2,device='cuda')
    for i in np.r_[train,test]:ht[i]=records[i]['responses'][:,s].mean(0).cuda()/scale[:,None]
    distances=None;distance_scale=None
    if arm['metric_weight']:
        values=torch.stack([r['values'][:8,s]for r in records]).cuda()
        distances,distance_scale=distribution_distances(values[train])
        lookup=torch.full((len(records),),-1,dtype=torch.long,device='cuda');lookup[train]=torch.arange(len(train),device='cuda')
    model=Predictor(legacy(c)).cuda();optimizer=torch.optim.AdamW([
        dict(params=model.encoder.parameters(),lr=cfg['encoder_lr']),dict(params=model.head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed);history=[];best=math.inf;selected=0;start=1;elapsed=0.
    if (tech/'last.pt').exists():
        state=TrainingState.read(tech/'last.pt',identity=identity,device='cuda');state.restore(model,optimizer,rng,restore_torch_rng=True)
        st=state.payload;history=st['history'];best=st['best'];selected=st['selected'];start=st['update']+1;elapsed=st['elapsed'];zscale=st['zscale']
    else:
        initialize(model,q[train],micro)
        with torch.no_grad():z=torch.cat([model.encode(x)for x in q[train].split(micro)])
        zscale=float((2*z.var(0,unbiased=False).mean()).clamp_min(1e-8))
    @torch.no_grad()
    def predict(ids):return torch.cat([model(x)for x in q[ids].split(micro)])
    def select():return float(.5*((predict(selection)-target[selection]).square().mean()+math.log(2*math.pi)))
    context=dict(identity=identity,arm=arm,seed=seed,training=cfg,actual_batch=batch,microbatch=micro,
        replacement=len(train)<batch,selector='selection future-feature Gaussian NLL every5 updates plus initialization',
        input_record=read(root(c)/'technical/prediction-context.json'),z_distance_scale=zscale,
        future_distance_scale=float(distance_scale)if distance_scale is not None else None)
    write_json(tech/'identity.json',context);deadline=allocation_deadline(reserve_seconds=300)
    run_id='resp-geometry-'+identity[:14]
    with online_training(c['wandb'],run_id=run_id,name=name,config=context,folder=tech,receipt_path=tech/'wandb.json',
        job_type='encoder',group=c['protocol'],receipt_fields=dict(identity=identity))as run:
        if start==1:
            model.eval();best=select();save(tech/'best.pt',dict(identity=identity,model=model.state_dict(),update=0,selection_nll=best))
        for update in range(start,cfg['updates']+1):
            if time.time()>=deadline:raise TimeoutError('Fit resume checkpoint retained before allocation expiry')
            began=time.monotonic();order=rng.choice(train,batch,replace=len(train)<batch)
            optimizer.zero_grad(set_to_none=True);model.train();losses=np.zeros(3)
            for begin in range(0,batch,micro):
                ix=order[begin:begin+micro];encoded=model.encode(q[ix]);p=model.head(encoded)
                lv=.5*(p-target[ix]).square().mean();lr=p.new_zeros(());lm=p.new_zeros(())
                if arm['response_weight']:
                    h=responses(model,q[ix],v[ix],create_graph=True)
                    lr=.5*((h-ht[ix])/hscale).square().mean()
                if arm['metric_weight']:
                    pred=(encoded[:,None]-encoded[None,:]).square().mean(-1)/zscale
                    local=lookup[ix];truth=distances[local[:,None],local[None,:]]
                    mask=local[:,None]!=local[None,:]
                    lm=.5*(pred[mask]-truth[mask]).square().mean()
                loss=(lv+arm['response_weight']*lr+arm['metric_weight']*lm)*len(ix)/batch
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite objective {name} update{update}')
                loss.backward();losses+=np.array([float(lv.detach()),float(lr.detach()),float(lm.detach())])*len(ix)/batch
            grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True));optimizer.step();model.eval()
            row=dict(update=update,value_nll=float(losses[0]+.5*math.log(2*math.pi)),gradient_norm=grad)
            if arm['response_weight']:row['response_half_mse']=float(losses[1])
            if arm['metric_weight']:row['distance_half_mse']=float(losses[2])
            if update%5==0:
                value=select();row['selection_nll']=value
                if value<best:
                    best=value;selected=update;save(tech/'best.pt',dict(identity=identity,model=model.state_dict(),update=update,selection_nll=best))
            torch.cuda.synchronize();elapsed+=time.monotonic()-began;history.append(row)
            TrainingState.capture(model,optimizer,rng,identity=identity,capture_torch_rng=True,history=history,best=best,
                selected=selected,update=update,elapsed=elapsed,zscale=zscale).save(tech/'last.pt')
            run.log(row,step=update)
            if update==1 or update%25==0:print(name,row,flush=True)
        saved=torch.load(tech/'best.pt',map_location='cuda',weights_only=False);model.load_state_dict(saved['model']);model.eval()
        prediction=predict(test);derivative=torch.cat([responses(model,q[ix],v[ix]).detach()for ix in np.array_split(test,4)])
        normalized_shots=torch.stack([records[i]['values'][:,s]for i in test]).cuda()/scale
        normalized_h=torch.stack([records[i]['responses'][:,s]for i in test]).cuda()/scale[None,None,:,None]
        row=dict(selected_update=selected,selection_nll=best,test_nll=float(.5*((prediction-target[test]).square().mean()+math.log(2*math.pi))),
            test_value_corrected_mse=float((prediction-target[test]).square().mean()-normalized_shots.var(1,unbiased=True).mean()/32),
            test_response_corrected_mse=float((derivative-ht[test]).square().mean()-normalized_h.var(1,unbiased=True).mean()/8))
        np.savez_compressed(tech/'predictions.npz',prediction=prediction.cpu().numpy(),response=derivative.cpu().numpy(),
            parent=test,source=np.array([records[i]['source']for i in test]))
        run.summary.update(dict(row,training_seconds=elapsed))
    # Optional fields are blank at unscheduled updates, not fabricated zeroes.
    columns=['update','value_nll','gradient_norm']+(['response_half_mse']if arm['response_weight']else [])+(['distance_half_mse']if arm['metric_weight']else [])+['selection_nll']
    write_metric_rows(history,out,family=FAMILY,name='learning',columns=columns)
    write_metric_rows([row],out,family=FAMILY,name='predictive')
    write_json(tech/'complete.json',dict(state='complete',identity=identity,checkpoint_sha256=sha(tech/'best.pt'),metrics=row))


def clustering(c,arm,seed):
    from .clusters import references,evaluate_model
    if time.time()>=allocation_deadline(reserve_seconds=600):raise TimeoutError('Reserve for structural inference')
    name=arm['name']+'-'+str(seed);out=root(c)/'analyses'/name;path=out/'technical/best.pt'
    if not (out/'technical/complete.json').exists():raise ValueError('Scientific fit incomplete')
    config=dict(c['clustering'],output=str(root(c)/'cluster-evaluation'))
    arrays,fit_rows,held,_=references(config)
    model=Predictor(legacy(c)).cuda();saved=torch.load(path,map_location='cuda',weights_only=False);model.load_state_dict(saved['model'])
    evaluate_model(config,model,name,sha(path),arrays,fit_rows,held)
    record=read(root(c)/'cluster-evaluation/analyses'/name/'technical/complete.json')
    fields={f'clustering/{space}/many_to_one':float(np.mean([r['many_to_one']for r in record['metrics']if r['space']==space]))
        for space in ['native','whitened','predicted_future']}
    update_recorded_summary(out/'technical/wandb.json',fields,evaluation='unrelaxed-clustering-v1',expected=dict(entity='teshbek',project='PointCloudMaterials'))
