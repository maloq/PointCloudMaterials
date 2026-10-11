"""Matched label-free likelihood training; finite physical response controls."""
import time
import numpy as np
import torch
from src.experiment_runner.wandb_tracking import online_training
from .artifacts import write_rows as write_metric_rows
from .data import load,root,read,digest,sha,write_json,atomic_save,FAMILY
from .model import make_model,nll


def short_loss(model,bank,ids,mean,scale,arm,seed):
    history=torch.stack([bank[i]['history'] for i in ids]).cuda()
    history=(history-mean)/scale
    target=(torch.stack([bank[i]['target'] for i in ids]).cuda()-mean[:128])/scale[:128]
    def predict(h):
        g=torch.Generator(device='cuda').manual_seed(seed)
        mu,_=model.rollout(h,1,4,g)
        return mu[:,:,0,:128].mean(1)
    value=.5*(predict(history)-target).square().mean();response=value.new_zeros(())
    if arm['response']:
        pairs=torch.stack([bank[i]['pairs'] for i in ids]).cuda()
        teacher=torch.stack([bank[i]['response'] for i in ids]).cuda()/scale[:128,None]
        for k in range(2):
            hminus=history.clone();hplus=history.clone()
            hminus[:,-1]=(pairs[:,k,0]-mean)/scale;hplus[:,-1]=(pairs[:,k,1]-mean)/scale
            predicted=predict(hplus)-predict(hminus)  # denominator 2*epsilon=1
            response+=.25*(predicted-teacher[...,k]).square().mean()/teacher[...,k].square().mean().clamp_min(1e-5)
    return value,response


def run(c,p,arm,deadline):
    folder=root(c)/'analyses'/arm['name']/'technical';folder.mkdir(parents=True,exist_ok=True)
    identity=digest([p['identity'],arm])
    if (folder/'complete.json').exists():return True
    torch.manual_seed(arm['seed']);model=make_model(arm).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=c['training']['lr'],weight_decay=1e-5)
    data=load(c,'data');mean=data['mean'].cuda();scale=data['scale'].cuda();x=(data['x'].cuda()-mean)/scale
    if 'dense_response' in c:
        from .dense_response import bank as dense_bank
        bank=dense_bank(c,p)
    else:
        bank=load(c,'short')
    ids=np.flatnonzero([r['role']=='train' for r in data['rows']]);selection=np.flatnonzero([r['role']=='selection' for r in data['rows']])
    # Fixed validation subset, independent of outcomes and all held-out sources.
    selection=np.random.default_rng(c['seed']).choice(selection,min(256,len(selection)),replace=False)
    shortids=np.flatnonzero([r['role']=='train' for r in bank]);rng=np.random.default_rng(arm['seed'])
    total=c['training']['updates'];start=0;best=float('inf');selected=0;history=[]
    steps=arm.get('train_steps',10)
    offsets={10:[1,5,10],60:[1,5,10,30,60]}[steps]
    forecast_slots=[v-1 for v in offsets]
    if (folder/'last.pt').exists():
        old=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if old['identity']!=identity:raise ValueError('Changed model resume')
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer']);start=old['update'];best=old['best'];selected=old['selected'];history=old['history'];rng.bit_generator.state=old['rng']
    def save(update):
        atomic_save(folder/'last.pt',dict(identity=identity,model=model.state_dict(),optimizer=optimizer.state_dict(),update=update,best=best,selected=selected,history=history,rng=rng.bit_generator.state,arm=arm))
    with online_training(c['wandb'],run_id='nils-'+identity[:16],name=arm['name'],config=dict(c,arm=arm),folder=folder,receipt_path=folder/'wandb.json',job_type='encoder',group=c['protocol'],receipt_fields=dict(identity=identity)) as tracking:
        for update in range(start+1,total+1):
            if time.time()>deadline-600:save(update-1);return False
            begin=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
            frac=min(update/200,1)*.5*(1+np.cos(np.pi*max(update-200,0)/max(total-200,1)))
            for group in optimizer.param_groups:group['lr']=c['training']['lr']*max(frac,.01)
            rows=torch.as_tensor(rng.choice(ids,c['training']['batch_size']),device='cuda')
            if c['training'].get('fixed_anchor',False):t=torch.full((len(rows),),3,device='cuda')
            else:t=torch.as_tensor(rng.integers(3,64-steps,len(rows)),device='cuda')
            h=x[rows[:,None],t[:,None]+torch.arange(-3,1,device='cuda')]
            target=x[rows[:,None],t[:,None]+torch.tensor(offsets,device='cuda')]
            g=torch.Generator(device='cuda').manual_seed(int(rng.integers(2**31)))
            with torch.autocast('cuda',dtype=torch.bfloat16):
                mu,lv=model.rollout(h,steps,4,g)
                now,_=model.emit(model.encode(h[:,-1]))
            likelihood=nll(mu[:,:,forecast_slots].float(),lv[:,:,forecast_slots].float(),target).mean()
            reconstruction=.5*(now.float()-h[:,-1]).square().mean()
            (likelihood+.2*reconstruction).backward()
            short_rows=rng.choice(shortids,16);short_seed=int(rng.integers(2**31))
            if 'dense_response' in c:
                from .dense_response import objective
                v,d=objective(model,[bank[i] for i in short_rows],mean,scale,arm,short_seed)
            else:v,d=short_loss(model,bank,short_rows,mean,scale,arm,short_seed)
            (.1*v+.1*d).backward()
            regularizer={}
            if arm.get('jacobian',False):
                from .stability import penalty
                loss,regularizer=penalty(model,h.detach(),c['stability'],arm['seed']+update*1009)
                (c['stability']['weight']*loss).backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10,error_if_nonfinite=True);optimizer.step();torch.cuda.synchronize()
            row=dict(update=update,nll=float(likelihood.detach()),reconstruction_half_mse=float(reconstruction.detach()),teacher_half_mse=float(v.detach()),relative_response=float(d.detach()),gradient_norm=float(norm))
            row.update(regularizer)
            if update%200==0 or update==total:
                model.eval()
                with torch.no_grad():
                    g=torch.Generator(device='cuda').manual_seed(c['seed']+17)
                    mu,lv=model.rollout(x[selection,:4],steps,8,g)
                    val=float(nll(mu[:,:,forecast_slots],lv[:,:,forecast_slots],x[selection][:,[3+v for v in offsets]]).mean())
                row['selection_nll']=val
                if val<best:
                    best=val;selected=update;atomic_save(folder/'best.pt',dict(model=model.state_dict(),arm=arm,identity=identity,update=update))
            history.append(row);tracking.log(row,step=update)
            if update==1 or update%100==0 or update==total:save(update)
            write_json(folder/'progress.json',dict(state='training',**row,total=total,seconds_per_update=time.monotonic()-begin))
            if update==1 or update%200==0:print(arm['name'],row,flush=True)
        tracking.summary.update(dict(selected_update=selected,selection_nll=best))
    write_metric_rows(history,folder.parent,family=FAMILY,name='learning',columns=list(dict.fromkeys(k for row in history for k in row)))
    write_json(folder/'complete.json',dict(identity=identity,updates=total,selected_update=selected,selection_nll=best,checkpoint_sha256=sha(folder/'best.pt')))
    return True
