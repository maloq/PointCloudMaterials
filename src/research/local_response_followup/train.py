"""Matched fixed-update fits, weight controls and delayed response supervision."""
import math
import time
import numpy as np
import torch

from src.experiment_runner.checkpoints import TrainingState
from src.experiment_runner.execution import allocation_deadline
from src.experiment_runner.metric_docs import write_metric_rows
from src.experiment_runner.wandb_tracking import online_training
from src.research.local_response.model import Predictor, initialize, responses
from .common import FAMILY, root, read, sha, digest, save, write_json, load, source_config


def location(c, arm, seed):
    return root(c)/'analyses'/f'{arm}-seed-{seed}'


def response_weight(arm, update):
    if update <= arm['warmup_updates']:return 0.
    fraction = min(1.,(update-arm['warmup_updates'])/arm['ramp_updates']) if arm['ramp_updates'] else 1.
    return arm['response_weight']*fraction


@torch.no_grad()
def predict(model,q,micro):
    return torch.cat([model(x) for x in q.split(micro)])


def fit(c, arm, seed):
    torch.set_num_threads(2);torch.set_default_dtype(torch.float32);torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32=False
    binding=read(root(c)/'technical/binding.json')
    identity=digest(dict(binding=binding['identity'],arm=arm,seed=seed))
    out=location(c,arm['name'],seed);tech=out/'technical';tech.mkdir(parents=True,exist_ok=True)
    if (tech/'complete.json').exists():
        rec=read(tech/'complete.json')
        if rec['identity']!=identity or any(sha(tech/name)!=value for name,value in rec['outputs'].items()):
            raise ValueError('Completed fixed-budget fit changed')
        return
    data=load(c);records=data['parents'];cfg=c['training'];micro=cfg['microbatch']
    ids={role:np.array([i for i,r in enumerate(records) if r['role']==role]) for role in ('train','selection','test')}
    if tuple(map(len,ids.values()))!=(90,15,30) or len(ids['train'])>cfg['batch_size']:
        raise ValueError('Original source/sample contract changed')
    q=torch.stack([r['q'] for r in records]).cuda();basis=torch.stack([r['basis'] for r in records]).cuda()
    target=torch.stack([(r['values'][:8 if r['role']=='train' else 32].mean(0)-data['center'])/data['scale'] for r in records]).cuda()
    htarget=torch.zeros(len(records),256,2,device='cuda')
    for i in ids['train']:htarget[i]=(records[i]['responses'].mean(0)/data['scale'][:,None]).cuda()
    response_scale=data['response_scale'].cuda()
    model=Predictor(source_config(c)).cuda()
    optimizer=torch.optim.AdamW([dict(params=model.encoder.parameters(),lr=cfg['encoder_lr']),
        dict(params=model.head.parameters(),lr=cfg['head_lr'])],weight_decay=cfg['weight_decay'])
    rng=np.random.default_rng(seed);history=[];best=math.inf;best_epoch=0;start=1;elapsed=0.
    if (tech/'last.pt').exists():
        state=TrainingState.read(tech/'last.pt',identity=identity,device='cuda')
        state.restore(model,optimizer,rng,restore_torch_rng=True)
        p=state.payload;history=p['history'];best=p['best'];best_epoch=p['best_epoch'];start=p['epoch']+1;elapsed=p['elapsed']
    else:initialize(model,q[ids['train']],micro)
    context=dict(identity=identity,config=c,arm=arm,seed=seed,
        input_contract=read(root(c)/'technical/prediction-context.json'),
        selector='Original32-shot validation feature Gaussian NLL; fixed400 updates; no patience stopping',
        actual_optimizer_batch=90,microbatch=micro,label_bank_sha256=binding['data_sha256'])
    write_json(tech/'identity.json',context)
    deadline=allocation_deadline(reserve_seconds=180)
    with online_training(c['wandb'],run_id='local-follow-'+identity[:14],name=f'Local80 follow-up | {arm["name"]} | {seed}',
        config=context,folder=tech,receipt_path=tech/'wandb.json',job_type='encoder',group=c['protocol'],
        receipt_fields=dict(identity=identity)) as run:
        def score():
            model.eval()
            return float(.5*((predict(model,q[ids['selection']],micro)-target[ids['selection']]).square().mean()+math.log(2*math.pi)))
        if start==1:
            best=score();save(tech/'best.pt',dict(identity=identity,model=model.state_dict(),epoch=0,selection_nll=best))
        for epoch in range(start,cfg['updates']+1):
            if time.time()>=deadline:raise TimeoutError('Allocation ending; fixed-budget optimizer state saved')
            weight=response_weight(arm,epoch)
            torch.cuda.synchronize();began=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
            order=rng.permutation(ids['train']);value_loss=0.;response_loss=0.
            for begin in range(0,len(order),micro):
                ix=order[begin:begin+micro];prediction=model(q[ix]);lv=.5*(prediction-target[ix]).square().mean()
                ld=prediction.new_zeros(())
                if weight>0:
                    derivative=responses(model,q[ix],basis[ix],create_graph=True)
                    ld=.5*((derivative-htarget[ix])/response_scale).square().mean()
                loss=(lv+weight*ld)*len(ix)/len(order)
                if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite {arm["name"]} loss at update{epoch}')
                loss.backward();value_loss+=float(lv.detach())*len(ix)/len(order)
                response_loss+=float(ld.detach())*len(ix)/len(order)
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True))
            optimizer.step();selection=score();torch.cuda.synchronize();elapsed+=time.monotonic()-began
            if selection<best:
                best=selection;best_epoch=epoch
                save(tech/'best.pt',dict(identity=identity,model=model.state_dict(),epoch=epoch,selection_nll=best))
            row=dict(epoch=epoch,response_weight=weight,train_value_nll=value_loss+.5*math.log(2*math.pi),
                train_response_half_mse=response_loss if weight>0 else None,selection_nll=selection,
                best_epoch=best_epoch,gradient_norm=norm,training_seconds=elapsed)
            history.append(row)
            TrainingState.capture(model,optimizer,rng,identity=identity,capture_torch_rng=True,
                history=history,best=best,best_epoch=best_epoch,epoch=epoch,elapsed=elapsed).save(tech/'last.pt')
            write_json(tech/'progress.json',dict(state='running',arm=arm['name'],seed=seed,**row))
            metrics={k:v for k,v in row.items() if k not in ('training_seconds','response_weight') and v is not None}
            if arm['ramp_updates']:metrics['response_weight']=weight
            run.log(metrics,step=epoch)
            if epoch==1 or epoch%25==0:print(dict(arm=arm['name'],seed=seed,**row),flush=True)
        outputs={}
        for endpoint in ['final','selected']:
            if endpoint=='selected':
                model.load_state_dict(torch.load(tech/'best.pt',map_location='cuda',weights_only=False)['model'])
            model.eval();prediction=predict(model,q,micro).cpu().numpy()
            derivative=torch.cat([responses(model,q[i:i+micro],basis[i:i+micro]).detach().cpu() for i in range(0,len(q),micro)]).numpy()
            name=f'predictions-{endpoint}.npz'
            np.savez_compressed(tech/name,prediction=prediction,response=derivative,
                parent=np.array([r['index'] for r in records]),source=np.array([r['source'] for r in records]))
            outputs[name]=sha(tech/name)
        outputs.update({name:sha(tech/name) for name in ['best.pt','last.pt']})
        write_metric_rows(history,out,family=FAMILY,name='learning')
        run.summary.update(dict(selected_epoch=best_epoch,selection_nll=best,training_seconds=elapsed,
                                optimizer_updates=len(history)))
    if len(history)!=cfg['updates']:raise ValueError('Fixed optimization budget was not completed')
    write_json(tech/'complete.json',dict(state='complete',identity=identity,arm=arm['name'],seed=seed,
        selected_epoch=best_epoch,selection_nll=best,training_seconds=elapsed,optimizer_updates=len(history),outputs=outputs))
    write_json(tech/'progress.json',read(tech/'complete.json'))
