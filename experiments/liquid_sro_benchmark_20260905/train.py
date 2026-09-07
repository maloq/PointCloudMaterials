"""Train matched self-supervised atomic encoders without structural labels."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch import nn
from src.models.encoders.atomic_graph import ReferenceMACEEncoder,SchNetEncoder,DensityMLPEncoder
from experiments.smooth_temporal_encoder_20260905.prepare import write_json
from experiments.smooth_temporal_encoder_20260905.run import make_loss


def construct(name,out):
    if name in ('MACE','MACE_untrained'):
        return ReferenceMACEEncoder(accelerated=True).cuda()
    if name=='SchNet':
        return SchNetEncoder().cuda()
    if name=='DensityMLP':
        model=DensityMLPEncoder().cuda()
        scaling=torch.load(out/'density_scaling.pt',weights_only=True)
        model.mean.copy_(scaling['mean']);model.std.copy_(scaling['std'])
        return model
    raise ValueError(f'Unknown benchmark encoder: {name}')


def jitter(x,settings,generator):
    noise=(torch.randn(x.shape,device=x.device,generator=generator)*settings['jitter_std_A']).clamp(-settings['jitter_clip_A'],settings['jitter_clip_A'])
    noise[:,0]=0.
    return x+noise


@torch.no_grad()
def encode(model,clouds,material,edges,counts):
    values=[]
    for start in range(0,len(clouds),512):
        z=model(clouds[start:start+512].clone(),material[start:start+512],edges[start:start+512],counts[start:start+512])
        if not torch.isfinite(z).all():
            raise FloatingPointError(f'Nonfinite saved representation at rows {start}')
        values.append(z.cpu().numpy())
    return np.concatenate(values)


def run(cfg,out,model_names):
    assert json.loads((out/'features_status.json').read_text())['state']=='complete'
    clouds=torch.tensor(np.load(out/'clouds.npy'),device='cuda')
    metadata=dict(np.load(out/'metadata.npz'))
    material=torch.tensor(metadata['material'],device='cuda')
    edges=torch.tensor(np.load(out/'edges.npy'),device='cuda')
    counts=torch.tensor(np.load(out/'edge_counts.npy'),device='cuda')
    train=torch.tensor(np.flatnonzero(metadata['split']=='train'),device='cuda')
    val=torch.tensor(np.flatnonzero(metadata['split']=='val'),device='cuda')
    settings=cfg['training'];batch_size=settings['batch_size']
    val_rng=torch.Generator(device='cuda').manual_seed(777)
    va=jitter(clouds[val],settings,val_rng);vb=jitter(clouds[val],settings,val_rng)
    loss_fn=make_loss(settings['projector_dim'])
    summary_path=out/'training_summary.json'
    records=json.loads(summary_path.read_text()) if summary_path.exists() else []
    assert all(r['name'] not in model_names for r in records),'Archive earlier results for the explicitly requested models before retraining'
    for name in model_names:
        for seed in settings['seeds']:
            directory=out/'models'/f'{name}_seed{seed}'
            directory.mkdir(parents=True,exist_ok=False)
            torch.manual_seed(seed)
            model=construct(name,out)
            torch.manual_seed(seed+1)
            projector=nn.Sequential(nn.Linear(128,512),nn.BatchNorm1d(512),nn.ReLU(),
                nn.Linear(512,512),nn.BatchNorm1d(512),nn.ReLU(),nn.Linear(512,settings['projector_dim'])).cuda()
            parameters=[p for module in (model,projector) for p in module.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(parameters,lr=settings['learning_rate'],weight_decay=settings['weight_decay'],fused=True)
            order_rng=torch.Generator(device='cuda').manual_seed(seed+2000)
            noise_rng=torch.Generator(device='cuda').manual_seed(seed+4000)
            best,best_epoch=float('inf'),-1
            started=time.monotonic()
            for epoch in range(settings['epochs']):
                model.train();projector.train();epoch_start=time.monotonic()
                rate=settings['learning_rate']*min(1.,(epoch+1)/5)*.5*(1+np.cos(np.pi*epoch/settings['epochs']))
                for group in optimizer.param_groups:group['lr']=rate
                total=0.
                for indices in train[torch.randperm(len(train),device='cuda',generator=order_rng)].split(batch_size):
                    x=clouds[indices]
                    a,b=jitter(x,settings,noise_rng),jitter(x,settings,noise_rng)
                    duplicated=indices.repeat(2)
                    z=model(torch.cat((a,b)),material[duplicated],edges[duplicated],counts[duplicated])
                    projected=projector(z)
                    loss,parts=loss_fn._loss(projected[:len(x)],projected[len(x):])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite training loss: {name}/{seed}/{epoch}')
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    grad=torch.nn.utils.clip_grad_norm_(parameters,5.,error_if_nonfinite=True)
                    optimizer.step();total+=float(loss.detach())*len(x)
                model.eval();projector.eval();validation=0.
                with torch.no_grad():
                    for local in torch.arange(len(val),device='cuda').split(batch_size):
                        ids=val[local].repeat(2)
                        zz=model(torch.cat((va[local],vb[local])).clone(),material[ids],edges[ids],counts[ids])
                        pp=projector(zz)
                        value,_=loss_fn._loss(pp[:len(local)],pp[len(local):])
                        if not torch.isfinite(value):
                            raise FloatingPointError(f'Nonfinite validation loss: {name}/{seed}/{epoch}')
                        validation+=float(value)*len(local)
                validation/=len(val)
                record=dict(epoch=epoch,train_loss=total/len(train),validation_loss=validation,
                    learning_rate=rate,gradient_norm=float(grad),seconds=time.monotonic()-epoch_start,
                    encoder_batch_rms=float(z.detach().square().mean().sqrt()),
                    losses={k:float(v.detach()) for k,v in parts.items()})
                with (directory/'epochs.jsonl').open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')
                payload=dict(state_dict=model.state_dict(),projector=projector.state_dict(),optimizer=optimizer.state_dict(),config=cfg,name=name,seed=seed,**record)
                torch.save(payload,directory/'last.pt')
                if validation<best:
                    best,best_epoch=validation,epoch;torch.save(payload,directory/'best.pt')
                write_json(directory/'status.json',dict(state='running',best_epoch=best_epoch,**record))
                if epoch%5==0 or epoch==settings['epochs']-1:
                    print(name,seed,epoch+1,'train',record['train_loss'],'val',validation,'seconds',record['seconds'],flush=True)
            torch.save(payload,directory/'final.pt')
            model.load_state_dict(torch.load(directory/'best.pt',map_location='cuda',weights_only=False)['state_dict'])
            np.save(out/'embeddings'/f'{name}_seed{seed}.npy',encode(model,clouds,material,edges,counts))
            summary=dict(name=name,seed=seed,best_epoch=best_epoch,validation_loss=best,seconds=time.monotonic()-started,
                encoder_parameters=sum(p.numel() for p in model.parameters()),projector_parameters=sum(p.numel() for p in projector.parameters()),
                trained_parameters=sum(p.numel() for p in parameters),checkpoint=str(directory/'best.pt'))
            records.append(summary);write_json(out/'training_summary.json',records)
            write_json(directory/'status.json',dict(state='complete',**summary))
            print('COMPLETE',name,seed,summary,flush=True)
            del optimizer,model,projector,parameters,payload
            torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--models',nargs='+',choices=('MACE','SchNet','DensityMLP'),default=('MACE','SchNet','DensityMLP'))
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=ROOT/cfg['output']
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    status=dict(state='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat())
    write_json(out/'training_status.json',status)
    try:
        run(cfg,out,args.models);status.update(state='complete',finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state='failed',error=repr(error),traceback=traceback.format_exc());raise
    finally:write_json(out/'training_status.json',status)


if __name__=='__main__':main()
