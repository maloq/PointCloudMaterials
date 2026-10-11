"""Matched value-only and tangent-supervised 1600-update eSEN continuations."""
import time
import numpy as np
import torch
from src.research.continuous_motion.train import Sampler
from src.research.structural_response.fast import long_backward,short_objective
from src.research.structural_response.train import predict_batches,common_dynamics
from src.experiment_runner.wandb_tracking import online_training
from src.experiment_runner.metric_docs import write_metric_rows
from .data import load,read,root,digest,sha,write_json,atomic_save,FAMILY
from .model import Motion

def run(c,p,arm,deadline,micro):
    name=arm['name'];folder=root(c)/'analyses'/name/'technical';folder.mkdir(parents=True,exist_ok=True)
    identity=digest([p['identity'],name])
    if (folder/'complete.json').exists():
        done=read(folder/'complete.json')
        if done['identity']!=identity or sha(folder/'best.pt')!=done['checkpoint_sha256']:raise ValueError('Changed fit receipt')
        return True
    torch.manual_seed(c['seed']);model=Motion(c).cuda();model.set_arm(arm);common_dynamics(c,model,folder)
    data=load(c,'data');targets=load(c,'targets');zall=targets['codes'] if arm['frozen'] else None
    cfg=c['response_training'];rng=np.random.default_rng(c['seed'])
    opt=torch.optim.AdamW([dict(params=[v for n,v in model.named_parameters() if not n.startswith('geometry.')],lr=cfg['learning_rate']),dict(params=model.geometry.parameters(),lr=cfg['learning_rate']*cfg['encoder_lr_scale'])],weight_decay=cfg['weight_decay'])
    sampler=Sampler(data['rows'],'train');short_sampler=Sampler(targets['short'],'train')
    selection=np.unique(Sampler(data['rows'],'selection').draw(np.random.default_rng(c['seed']+81),256))
    start=0;history=[];best=float('inf');selected=0
    if (folder/'last.pt').exists():
        old=torch.load(folder/'last.pt',map_location='cuda',weights_only=False)
        if old['identity']!=identity:raise ValueError('Changed response resume')
        model.load_state_dict(old['model']);opt.load_state_dict(old['optimizer']);rng.bit_generator.state=old['rng']
        start=old['update'];history=old['history'];best=old['best'];selected=old['selected']
    def payload(update):return dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng.bit_generator.state,update=update,history=history,best=best,selected=selected,config=c,arm=arm)
    with online_training(c['wandb'],run_id='niresp-'+identity[:16],name='ni-esen-'+name,config=dict(c,arm=arm,identity=identity),folder=folder,receipt_path=folder/'wandb.json',job_type='encoder' if not arm['frozen'] else 'predictor',group=c['protocol'],receipt_fields=dict(identity=identity)) as tracking:
        for update in range(start+1,cfg['updates']+1):
            if time.time()>deadline-600:
                atomic_save(folder/'last.pt',payload(update-1));write_json(folder/'progress.json',dict(state='checkpointed',update=update-1,total=cfg['updates']));return False
            begin=time.monotonic();model.train();opt.zero_grad(set_to_none=True)
            ids=torch.from_numpy(sampler.draw(rng,cfg['batch_size']))
            nll,latent=long_backward(model,data,targets['future'],zall,ids,c,micro,int(rng.integers(2**31)))
            value=tangent=0.
            for j in short_sampler.draw(rng,cfg['short_batch']):
                v,d=short_objective(model,targets['short'][j],c,arm,int(rng.integers(2**31)))
                loss=(cfg['short_weight']*v+cfg['derivative_weight']*d)/cfg['short_batch']
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite response loss')
                loss.backward();value+=float(v.detach())/cfg['short_batch'];tangent+=float(d.detach())/cfg['short_batch']
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize()
            seconds=time.monotonic()-begin
            row=dict(update=update,future_nll=nll,long_structural_half_mse=latent,short_half_mse=value,relative_tangent=tangent,gradient_norm=float(norm))
            if update%cfg['selection_every']==0 or update==cfg['updates']:
                val=predict_batches(model,data,targets['future'],zall,selection,c,micro);score=float(val['nll'].mean());row['selection_nll']=score
                if score<best:
                    best=score;selected=update;atomic_save(folder/'best.pt',dict(identity=identity,model=model.state_dict(),update=update,config=c,arm=arm))
            history.append(row);tracking.log(row,step=update)
            if update%cfg['save_every']==0 or update==cfg['updates']:atomic_save(folder/'last.pt',payload(update))
            write_json(folder/'progress.json',dict(state='training',update=update,total=cfg['updates'],update_seconds=seconds,estimated_remaining_compute_seconds=seconds*(cfg['updates']-update),**{k:v for k,v in row.items() if k!='update'}))
            if update==1 or update%25==0:print(name,update,seconds,row,flush=True)
        tracking.summary.update(dict(selected_update=selected,selection_nll=best))
    columns=list(dict.fromkeys(k for r in history for k in r))
    write_metric_rows(history,folder.parent,family=FAMILY,name='learning',columns=columns)
    write_json(folder/'complete.json',dict(identity=identity,checkpoint_sha256=sha(folder/'best.pt'),update=cfg['updates'],selected_update=selected,selection_nll=best))
    return True
