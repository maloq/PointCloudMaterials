"""Matched continuation controls for objective-induced loss of structural access.

Each condition starts from the same original best spatial checkpoint for its
seed, with a fresh AdamW optimizer. This is a controlled adaptation experiment,
not a resumption of the pilot's optimizer or a full MACE benchmark.
"""
import argparse
from datetime import datetime, timezone
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
from experiments.smooth_temporal_encoder_20260905.run import construct, load_split, make_loss, paired_loss, encode_moments
from experiments.smooth_temporal_encoder_20260905.evaluate import structure_fit, spectrum
from experiments.smooth_temporal_encoder_20260905.prepare import write_json
from experiments.mace_diagnosis_20260905.readouts import metrics, predict


@torch.no_grad()
def assess(model,q,metadata,qs,ys):
    z={s:encode_moments(model,x[:,0]) for s,x in q.items()}
    zs=encode_moments(model,qs)
    selection,probe,p=structure_fit(z['train'],metadata['train']['labels'],z['val'],metadata['val']['labels'],zs,ys)
    return dict(selection=selection,static=metrics(ys,p),md_test=metrics(metadata['test']['labels'],predict(probe,z['test'])),
                static_spectrum=spectrum(zs)),probe


def run(cfg,out):
    pilot=ROOT/cfg['pilot']
    pcfg=json.loads((pilot/'config.json').read_text())
    manifest=json.loads((pilot/'data/manifest.json').read_text())
    settings=cfg['objective_audit']
    scaling={k:v.cuda() for k,v in torch.load(pilot/'scaling.pt',weights_only=True).items()}
    q,metadata={},{}
    for split in ('train','val','test'):
        x,metadata[split]=load_split(pilot,manifest,split)
        q[split]=torch.tensor(x,device='cuda')
        del x
    qs=torch.tensor(np.load(out/'static_original.moments.npy'),device='cuda')
    ys=np.load(out/'static_sample_metadata.npz')['labels']
    objective=make_loss(128)
    results=[]
    for seed in settings['seeds']:
        start_payload=torch.load(pilot/f'mace_product_seed{seed}/best.pt',map_location='cuda',weights_only=False)
        baseline=construct(pcfg,'mace_product',scaling).eval()
        baseline.load_state_dict(start_payload['state_dict'])
        baseline_metrics,_=assess(baseline,q,metadata,qs,ys)
        write_json(out/f'original_seed{seed}.json',baseline_metrics)
        del baseline
        for condition in settings['conditions']:
            directory=out/'objective_runs'/f'{condition}_seed{seed}'
            directory.mkdir(parents=True,exist_ok=False)
            torch.manual_seed(seed)
            model=construct(pcfg,'mace_product',scaling)
            model.load_state_dict(start_payload['state_dict'])
            projector=nn.Sequential(nn.Linear(128,256),nn.SiLU(),nn.Linear(256,256),nn.SiLU(),nn.Linear(256,128)).cuda() if condition=='separate_projector' else nn.Identity()
            optimizer=torch.optim.AdamW(list(model.parameters())+list(projector.parameters()),lr=settings['learning_rate'],weight_decay=.0001,fused=True)
            generator=torch.Generator(device='cuda').manual_seed(seed+1000)
            validation_order=torch.randperm(len(q['val']),device='cuda',generator=torch.Generator(device='cuda').manual_seed(100))
            weight=0. if condition=='no_spatial' else .25
            best,best_epoch,started=float('inf'),-1,time.monotonic()
            for epoch in range(settings['epochs']):
                model.train();projector.train()
                rate=settings['learning_rate']*.5*(1+np.cos(np.pi*epoch/settings['epochs']))
                for group in optimizer.param_groups:
                    group['lr']=rate
                total=0.
                order=torch.randperm(len(q['train']),device='cuda',generator=generator)
                for ids in order.split(settings['batch_size']):
                    z=projector(model.forward_moments(q['train'][ids].flatten(0,1))).reshape(len(ids),3,128)
                    value=paired_loss(objective,z,weight)
                    optimizer.zero_grad(set_to_none=True)
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(list(model.parameters())+list(projector.parameters()),10.,error_if_nonfinite=True)
                    optimizer.step()
                    total+=float(value.detach())*len(ids)
                model.eval();projector.eval()
                validation=0.
                with torch.no_grad():
                    for ids in validation_order.split(settings['batch_size']):
                        z=projector(model.forward_moments(q['val'][ids].flatten(0,1))).reshape(len(ids),3,128)
                        validation+=float(paired_loss(objective,z,weight))*len(ids)
                validation/=len(q['val'])
                record=dict(epoch=epoch,train_loss=total/len(q['train']),validation_loss=validation)
                with (directory/'epochs.jsonl').open('a') as handle:
                    handle.write(json.dumps(record,allow_nan=False)+'\n')
                payload=dict(state_dict=model.state_dict(),projector=projector.state_dict(),optimizer=optimizer.state_dict(),config=cfg,seed=seed,condition=condition,**record)
                torch.save(payload,directory/'last.pt')
                if validation<best:
                    best,best_epoch=validation,epoch
                    torch.save(payload,directory/'best.pt')
                if epoch%10==0 or epoch==settings['epochs']-1:
                    print(condition,seed,epoch+1,'val',validation,flush=True)
            torch.save(payload,directory/'final.pt')
            model.load_state_dict(torch.load(directory/'best.pt',map_location='cuda',weights_only=False)['state_dict'])
            result,probe=assess(model,q,metadata,qs,ys)
            torch.save(probe,directory/'probe.pt')
            result.update(seed=seed,condition=condition,best_epoch=best_epoch,seconds=time.monotonic()-started)
            write_json(directory/'metrics.json',result)
            results.append(result)
            write_json(out/'objectives.json',results)
            print('Result',condition,seed,result['static']['macro_f1'],flush=True)
            del optimizer,model,projector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    cfg=json.loads(args.config.read_text())
    out=ROOT/cfg['output']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    status=dict(state='running',pid=os.getpid(),started_at=datetime.now(timezone.utc).isoformat())
    write_json(out/'objectives_status.json',status)
    try:
        run(cfg,out)
        status.update(state='complete',finished_at=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        status.update(state='failed',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        write_json(out/'objectives_status.json',status)


if __name__=='__main__':
    main()
