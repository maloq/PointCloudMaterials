"""Fixed-budget joint training, online tracking and exact checkpoint resumption."""
import math,os,time
from contextlib import nullcontext
import numpy as np
import torch
from src.experiment_runner.wandb_tracking import online_training
from src.experiment_runner.metric_docs import write_metric_rows
from .data import load,teacher_bank,root,read,sha,digest,atomic_save,write_json,FAMILY
from .model import Model


def fitted(c,arm):
    folder=root(c)/'analyses'/arm['name']/'technical';receipt=read(folder/'complete.json')
    if sha(folder/'final.pt')!=receipt['checkpoint_sha256']:raise ValueError('Changed completed checkpoint')
    saved=torch.load(folder/'final.pt',map_location='cuda',weights_only=False)
    if saved['identity']!=digest([read(root(c)/'technical/plan.json')['identity'],arm]):raise ValueError('Model identity mismatch')
    model=Model(c,arm).cuda();model.load_state_dict(saved['ema']);return model.eval(),saved


def fit(c,arm,deadline,debug=False,max_updates=None):
    folder=root(c)/'analyses'/arm['name']/'technical';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return True
    p,bank=load(c,arm);train=bank['train_rows'];settings=c['training']
    x=bank['x'][train,4:].cuda();h=bank['history'][train].cuda();weights=bank['weights'].cuda()
    t=None
    if arm['teacher']!='none':
        packet=teacher_bank(c,bank);mask=packet['train']
        t={k:packet[k][mask].cuda() for k in ('value','response')};t['power']=packet['power'].cuda()
    identity=digest([p['identity'],arm]);torch.manual_seed(arm['seed'])
    model=Model(c,arm).cuda()
    with torch.no_grad():
        model.location.copy_(bank['x'][train].mean((0,1)))
        model.scale.copy_(bank['x'][train].std((0,1),unbiased=False).clamp_min(1e-5))
    del bank
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['lr'],weight_decay=settings['weight_decay'])
    ema={k:v.detach().clone() for k,v in model.state_dict().items()};start=0;records=[]
    if (folder/'last.pt').exists():
        previous=torch.load(folder/'last.pt',map_location='cuda',weights_only=False)
        if previous['identity']!=identity:raise ValueError('Changed resume identity')
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer'])
        ema=previous['ema'];start=previous['update'];records=previous['history']
        torch.set_rng_state(previous['torch_rng'].cpu());torch.cuda.set_rng_state(previous['cuda_rng'].cpu())
    write_json(folder/'execution.json',dict(identity=identity,allocation=os.environ.get('SLURM_JOB_ID'),
        gpu=torch.cuda.get_device_name(),parameters=sum(v.numel() for v in model.parameters()),
        train_rows=len(train),train_ancestors=p['ancestors']['train'],history_frames=arm['history_frames'],
        regional=arm['regional'],bottleneck=arm['bottleneck'],teacher=arm['teacher'],
        target='60 future local244 states; no explicit temperature/age/time covariates',
        precision='BF16 flow; FP32 derivative head and reductions; physical teachers retain float64 states/JVPs',
        started=time.time(),resumed_update=start,debug=debug))
    def save(update,final=False):
        atomic_save(folder/('final.pt' if final else 'last.pt'),dict(identity=identity,arm=arm,
            model=model.state_dict(),ema=ema,optimizer=optimizer.state_dict(),update=update,
            history=records,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state()))
    context=nullcontext(None) if debug else online_training(c['wandb'],run_id='niorder-'+identity[:16],
        name=arm['name'],config=dict(c,arm=arm),folder=folder,receipt_path=folder/'wandb.json',
        job_type='predictor',group=c['protocol'],receipt_fields=dict(identity=identity))
    total=settings['updates'];warmup=settings['warmup'];begin=time.monotonic()
    with context as tracking:
        model.train()
        for update in range(start+1,total+1):
            if time.time()>deadline-180:save(update-1);return False
            ids=torch.randint(len(x),(settings['batch_size'],),device='cuda')
            warm=min(update/warmup,1.);cosine=.01+.99*.5*(1+math.cos(math.pi*max(0,update-warmup)/max(1,total-warmup)))
            for group in optimizer.param_groups:group['lr']=settings['lr']*warm*cosine
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.objective(h[ids],x[ids],weights)
            if t is not None and update%settings['teacher_every']==0:
                selected=torch.randint(len(t['value']),(settings['teacher_batch'],),device='cuda')
                auxiliary,details=model.teacher_objective(t,selected,arm['teacher']=='derivative')
                loss=loss+auxiliary;parts.update(details)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective at '+str(update))
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.,error_if_nonfinite=True);optimizer.step()
            with torch.no_grad():
                parameters=dict(model.named_parameters())
                for key,value in model.state_dict().items():
                    if key in parameters:ema[key].lerp_(value.detach(),1-c['flow']['ema_decay'])
                    else:ema[key].copy_(value)
            record=dict(update=update,loss=float(loss.detach()),gradient_norm=float(norm),lr=optimizer.param_groups[0]['lr'],
                        **{k:float(v.detach()) for k,v in parts.items()})
            records.append(record)
            if tracking:tracking.log({k:v for k,v in record.items() if k!='update'},step=update)
            if update==1 or update%settings['save_every']==0 or update==total:
                save(update);write_json(folder/'progress.json',dict(record,total=total,updated=time.time(),
                    seconds_per_update=(time.monotonic()-begin)/(update-start)))
                print(arm['name'],update,'/',total,record,flush=True)
            if max_updates and update>=max_updates:save(update);return False
        save(total,True)
        if not debug:
            columns=sorted({k for r in records for k in r})
            write_metric_rows([{k:r.get(k) for k in columns} for r in records],folder.parent,family=FAMILY,name='training')
        write_json(folder/'complete.json',dict(state='complete',identity=identity,checkpoint_sha256=sha(folder/'final.pt'),updates=total))
    return True
